import asyncio
import os
from typing import Any, AsyncIterator, cast

from dotenv import load_dotenv
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy

from prompts import GEORGIAN_PROMPT, INITIAL_GREETING
from speech_gate import PhantomSpeechFilter, SpeechOnlyAudioGate, build_user_params, build_vad_processor

load_dotenv()

# Phone audio from Twilio is 8 kHz
PHONE_SAMPLE_RATE = 8000

# Fast Gemini model chain ordered by lowest latency -> automatic retry on transient 503 spikes
_GEMINI_FALLBACK_CHAIN = (
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
)


class FastGoogleLLMService(GoogleLLMService):
    """Low-latency Gemini LLM service with parallel tail-latency hedging."""

    _HEDGE_DELAY_SECS = 2.6

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _open_model_stream(self, context: LLMContext, model_name: str):
        """Opens a streaming request for a specific model and awaits its first content chunk."""
        adapter = self.get_llm_adapter()
        params = await adapter.get_llm_invocation_params(
            context,
            system_instruction=self._settings.system_instruction,
            ensure_last_message_is_user=not any(
                model_name.startswith(p) for p in self._PREFILL_SUPPORTED_PATTERNS
            ),
        )
        messages = params["messages"]
        system_instruction = params["system_instruction"]
        tools = params["tools"] if params["tools"] else (self._tools or [])
        tool_config = self._tool_config if self._tool_config else None

        generation_params = self._build_generation_params(
            system_instruction=system_instruction,
            tools=tools,
            tool_config=tool_config,
        )
        generation_config = GenerateContentConfig(**generation_params)
        assert self._client is not None

        raw_stream = await asyncio.wait_for(
            self._client.aio.models.generate_content_stream(
                model=model_name,
                contents=cast(Any, messages),
                config=generation_config,
            ),
            timeout=5.0,
        )
        aiter = raw_stream.__aiter__()
        try:
            first_chunk = await asyncio.wait_for(
                aiter.__anext__(), timeout=5.0
            )
            return model_name, first_chunk, aiter, raw_stream
        except Exception:
            aclose = getattr(raw_stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
            raise

    async def _cancel_losing_tasks(self, tasks: list[asyncio.Task]):
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                res = await t
                if res and len(res) == 4:
                    aclose = getattr(res[3], "aclose", None)
                    if aclose is not None:
                        await aclose()
            except (asyncio.CancelledError, Exception):
                pass

    async def _stream_response(self, context: LLMContext) -> AsyncIterator[GenerateContentResponse]:
        orig_model = self._settings.model or "gemini-3-flash-preview"
        candidates = [orig_model] + list(_GEMINI_FALLBACK_CHAIN)
        self._log_llm_response(context)

        active_tasks: list[asyncio.Task] = []
        next_idx = 0
        last_err: Exception | None = None
        winner = None

        try:
            while next_idx < len(candidates) or active_tasks:
                if next_idx < len(candidates):
                    model_to_launch = candidates[next_idx]
                    next_idx += 1
                    if next_idx > 1:
                        logger.debug(f"Hedging LLM request with '{model_to_launch}'")
                    active_tasks.append(
                        asyncio.create_task(self._open_model_stream(context, model_to_launch))
                    )

                wait_timeout = self._HEDGE_DELAY_SECS if next_idx < len(candidates) else None
                done, _ = await asyncio.wait(
                    active_tasks,
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for d in done:
                    active_tasks.remove(d)
                    try:
                        winner = d.result()
                        break
                    except StopAsyncIteration:
                        return
                    except Exception as e:
                        last_err = e
                        logger.debug(f"Gemini candidate error ({type(e).__name__}): {e}")

                if winner is not None:
                    break

            if winner is None:
                if last_err:
                    raise last_err
                return

            if active_tasks:
                asyncio.create_task(self._cancel_losing_tasks(list(active_tasks)))
                active_tasks.clear()

            win_model, first_chunk, aiter, raw_stream = winner
            logger.debug(f"Gemini winner model: {win_model}")
            try:
                yield first_chunk
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            aiter.__anext__(), timeout=self._stream_idle_timeout_secs
                        )
                    except StopAsyncIteration:
                        break
                    yield chunk
            finally:
                aclose = getattr(raw_stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
        finally:
            if active_tasks:
                asyncio.create_task(self._cancel_losing_tasks(list(active_tasks)))


def build_llm():
    """Pick the AI brain. LLM_PROVIDER=gemini (free tier, default) or anthropic (paid)."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "anthropic":
        # Claude Haiku: fast and cheap. Prompt caching cuts cost on the long system prompt.
        return AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMService.Settings(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
                system_instruction=GEORGIAN_PROMPT,
                temperature=0.3,
                max_tokens=160,
                enable_prompt_caching=True,
            ),
        )

    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip() or "gemini-3-flash-preview"
        settings = dict(
            model=model,
            system_instruction=GEORGIAN_PROMPT,
            temperature=0.3,
            max_tokens=160,
            thinking=GoogleLLMService.ThinkingConfig(thinking_level="minimal"),
        )
        return FastGoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            settings=GoogleLLMService.Settings(**settings),
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini' or 'anthropic'.")


def build_tts(sample_rate: int = 16000, voice: str = None, rate: str = None, pitch: str = None):
    """
    Build the Georgian Text-to-Speech service.
    
    TTS_PROVIDER:
      - 'edge' (Default / Recommended): Native Georgian Neural Voice (0% foreign accent).
        Voices: 'ka-GE-EkaNeural' (Female, Georgian native), 'ka-GE-GiorgiNeural' (Male, Georgian native).
        Recorded and trained in Georgia with authentic Georgian phonology. 100% free and quota-free.
      - 'elevenlabs': ElevenLabs Text-to-Dialogue using multilingual models. Note: ElevenLabs
        voices are English/foreign speakers and will have a noticeable foreign accent.
    """
    provider = os.getenv("TTS_PROVIDER", "edge").strip().lower()

    if provider == "edge":
        from edge_tts_service import EdgeTTSService

        selected_voice = voice or os.getenv("EDGE_TTS_VOICE", "ka-GE-EkaNeural").strip()
        selected_rate = rate or os.getenv("EDGE_TTS_RATE", "+0%").strip()
        selected_pitch = pitch or os.getenv("EDGE_TTS_PITCH", "+0Hz").strip()
        logger.info(f"Using Native Georgian Edge TTS: voice={selected_voice}, rate={selected_rate}, sample_rate={sample_rate}")
        return EdgeTTSService(voice=selected_voice, rate=selected_rate, pitch=selected_pitch, sample_rate=sample_rate)

    if provider == "elevenlabs":
        from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService

        voice_id = voice or os.getenv("ELEVENLABS_VOICE_ID", "pFZP5JQG7iQjIQuC4Bku").strip()
        model = os.getenv("ELEVENLABS_MODEL", "eleven_v4_turbo").strip()
        stability = float(os.getenv("ELEVENLABS_STABILITY", "0.45"))
        logger.warning(
            f"Using ElevenLabs Dialogue TTS ({voice_id}). Note: ElevenLabs voices have an English accent in Georgian. "
            f"For 0% accent native Georgian, set TTS_PROVIDER=edge."
        )

        return ElevenLabsDialogueTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            settings=ElevenLabsDialogueTTSService.Settings(
                voice=voice_id,
                model=model,
                language="ka",
                stability=stability,
            ),
        )

    raise ValueError(f"Unknown TTS_PROVIDER '{provider}'. Use 'edge' or 'elevenlabs'.")


async def run_bot(websocket, stream_sid: str, call_sid: str):
    logger.info(f"Starting bot for call {call_sid} (stream {stream_sid})")

    # Converts between Twilio's media-stream format and Pipecat audio frames.
    # With Twilio credentials it can also hang up the call when the bot ends.
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID") or None,
        auth_token=os.getenv("TWILIO_AUTH_TOKEN") or None,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # 1. Listen: Deepgram real-time Georgian speech-to-text
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language="ka",
            smart_format=True,
        ),
    )

    # 2. Think: Gemini (free tier, for testing) or Claude (paid)
    llm = build_llm()

    # 3. Speak: Native Georgian Voice (Edge TTS or ElevenLabs Dialogue)
    tts = build_tts(sample_rate=PHONE_SAMPLE_RATE)

    # Conversation memory + fast turn-taking (~0.38s total silence detection).
    # VAD runs before STT and PhantomSpeechFilter drops transcripts Deepgram
    # invents from line noise/silence, so the bot never answers "nobody".
    vad = build_vad_processor()
    phantom_filter = PhantomSpeechFilter()
    context = LLMContext()
    mute_strategies = (
        [AlwaysUserMuteStrategy()]
        if os.getenv("MUTE_USER_WHILE_BOT_SPEAKS", "false").strip().lower() == "true"
        else []
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_params(
            user_speech_timeout=0.15,
            mute_strategies=mute_strategies,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            SpeechOnlyAudioGate(),
            stt,
            phantom_filter,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=PHONE_SAMPLE_RATE,
            audio_out_sample_rate=PHONE_SAMPLE_RATE,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Call connected: {call_sid}. Sending greeting.")
        await task.queue_frames([TTSSpeakFrame(INITIAL_GREETING)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Call ended: {call_sid}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
