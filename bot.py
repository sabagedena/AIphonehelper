import asyncio
import os
import re
import time
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

# Default model: on the free tier (measured 2026-10-28) gemini-flash-lite-latest gave its first token
# in ~1s (0.7-2.5s), while gemini-3.1-flash-lite-preview took 2-18s and gemini-3-flash-preview
# allows only 20 requests/day.
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"

# Hedge / fallback chain, fastest first. Its latency spikes are per-request jitter, so the hedge
# races a second request to the same fast model; slower models only replace failed requests.
_GEMINI_FALLBACK_CHAIN = (
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-lite-latest",
)

# Models that answered 429 (quota exhausted) are skipped until Google's retryDelay passes, re-probed
# at least hourly. Free-tier daily quotas are tiny (gemini-3-flash: 20 requests/day), and trying a
# dead model costs every turn an extra round trip.
_model_cooldown_until: dict[str, float] = {}
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s")
_DEFAULT_QUOTA_COOLDOWN_SECS = 60.0
_MAX_QUOTA_COOLDOWN_SECS = 3600.0


class FastGoogleLLMService(GoogleLLMService):
    """Low-latency Gemini LLM service with parallel tail-latency hedging."""

    # A warm gemini-flash-lite-latest request usually returns its first token in 0.6-1.0s; when it
    # hasn't by now, a second request mostly answers within 0.1-0.4s
    _HEDGE_DELAY_SECS = 1.0

    # The Gemini client's aiohttp connector drops idle connections after 15s, and on a call the gap
    # between replies is usually longer, so each turn paid a fresh TLS handshake (~250ms+).
    # A model-metadata GET keeps the connection open; it uses no generation quota.
    _KEEPWARM_IDLE_SECS = 10.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_request_at = 0.0
        self._keepwarm_task: asyncio.Task | None = None

    async def start(self, frame):
        await super().start(frame)
        if self._keepwarm_task is None:
            self._keepwarm_task = self.create_task(self._keepwarm_loop(), "gemini_keepwarm")

    async def stop(self, frame):
        await self._stop_keepwarm()
        await super().stop(frame)

    async def cancel(self, frame):
        await self._stop_keepwarm()
        await super().cancel(frame)

    async def _stop_keepwarm(self):
        if self._keepwarm_task:
            await self.cancel_task(self._keepwarm_task)
            self._keepwarm_task = None

    async def _keepwarm_loop(self):
        while True:
            if self._client and time.monotonic() - self._last_request_at >= self._KEEPWARM_IDLE_SECS:
                self._last_request_at = time.monotonic()
                try:
                    await self._client.aio.models.get(model=self._settings.model or DEFAULT_GEMINI_MODEL)
                except Exception as e:
                    logger.debug(f"Gemini keep-warm note: {e}")
            await asyncio.sleep(1.0)

    async def _open_model_stream(self, context: LLMContext, model_name: str):
        """Opens a streaming request for a specific model and awaits its first content chunk."""
        self._last_request_at = time.monotonic()
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

    def _note_quota_error(self, model_name: str, err: Exception):
        """Benches a model whose quota is exhausted, so later turns don't waste a round trip on it."""
        if getattr(err, "code", None) != 429:
            return
        m = _RETRY_DELAY_RE.search(str(err))
        delay = min(float(m.group(1)), _MAX_QUOTA_COOLDOWN_SECS) if m else _DEFAULT_QUOTA_COOLDOWN_SECS
        _model_cooldown_until[model_name] = time.monotonic() + delay
        logger.warning(f"Gemini '{model_name}' quota exhausted; skipping it for {delay:.0f}s")

    async def _stream_response(self, context: LLMContext) -> AsyncIterator[GenerateContentResponse]:
        orig_model = self._settings.model or DEFAULT_GEMINI_MODEL
        candidates = [orig_model] + list(_GEMINI_FALLBACK_CHAIN)
        now = time.monotonic()
        available = [m for m in candidates if _model_cooldown_until.get(m, 0.0) <= now]
        if available:
            candidates = available
        self._log_llm_response(context)

        active_tasks: dict[asyncio.Task, str] = {}
        next_idx = 0
        hedged = False
        last_err: Exception | None = None
        winner = None

        try:
            while next_idx < len(candidates) or active_tasks:
                # One request at a time, plus a single hedge once the first is slow; further
                # candidates only replace failed requests, so a slow API can't multiply quota use.
                if next_idx < len(candidates) and (not active_tasks or not hedged):
                    if active_tasks:
                        hedged = True
                    model_to_launch = candidates[next_idx]
                    next_idx += 1
                    if next_idx > 1:
                        logger.debug(f"Hedging LLM request with '{model_to_launch}'")
                    task = asyncio.create_task(self._open_model_stream(context, model_to_launch))
                    active_tasks[task] = model_to_launch

                can_hedge = next_idx < len(candidates) and not hedged
                done, _ = await asyncio.wait(
                    active_tasks,
                    timeout=self._HEDGE_DELAY_SECS if can_hedge else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for d in done:
                    model_name = active_tasks.pop(d)
                    try:
                        winner = d.result()
                        break
                    except StopAsyncIteration:
                        return
                    except Exception as e:
                        last_err = e
                        self._note_quota_error(model_name, e)
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
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
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
