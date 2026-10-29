import asyncio
import functools
import os
import random
import re
import ssl
import time
from typing import Any, AsyncIterator, cast

import certifi
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentConfig,
    GenerateContentResponse,
    HttpOptions,
    Part,
)
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from prompts import (
    INITIAL_GREETING,
    LLM_FAILURE_REPLY,
    LLM_STILL_WAITING_FILLER,
    LLM_WAIT_FILLERS,
    build_system_prompt,
)
from speech_gate import (
    BotReplyTracker,
    FlushReportingDeepgramSTTService,
    PhantomSpeechFilter,
    SpeechOnlyAudioGate,
    build_user_params,
    build_vad_processor,
)

load_dotenv()

# Phone audio from Twilio is 8 kHz
PHONE_SAMPLE_RATE = 8000

# Default model: on the free tier (measured 2026-10-28) gemini-flash-lite-latest gave its first token
# in ~1s (0.7-2.5s), while gemini-3.1-flash-lite-preview took 2-18s and gemini-3-flash-preview
# allows only 20 requests/day. On 2026-10-29 it slowed to 4-10 s; FastGoogleLLMService hedges to
# gemini-3.6-flash (~1.1 s that day, but a free-tier per-minute limit of under ten requests).
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"

# Edge TTS pacing when EDGE_TTS_RATE is unset; the cached phone greeting uses the same value
DEFAULT_EDGE_TTS_RATE = "+12%"

# Fallback chain for failed requests (and the 3 s hedge), fastest first
_GEMINI_FALLBACK_CHAIN = (
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
)

# Models that answered 429 (quota exhausted) are skipped until Google's retryDelay passes, re-probed
# at least hourly. Free-tier daily quotas are tiny (gemini-3-flash: 20 requests/day), and trying a
# dead model costs every turn an extra round trip.
_model_cooldown_until: dict[str, float] = {}
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s")
_DEFAULT_QUOTA_COOLDOWN_SECS = 60.0
_MAX_QUOTA_COOLDOWN_SECS = 3600.0
# Gemini streams a short reply in well under a second between chunks; a gap this long is a stall
_MID_STREAM_IDLE_SECS = 4.0


# The prompt allows at most four sentences even for a detailed answer; flash-lite sometimes ignores it
# and talks for 15+ seconds, which older callers can't follow (and which ran into max_tokens mid-word)
_MAX_REPLY_SENTENCES = 4
_SENTENCE_END = re.compile(r"[.!?…](?:[»“”\"')]*)(?=\s|$)")


class _SentenceGate:
    """Passes a streamed reply on in whole sentences only, holding back the unfinished tail.

    pipecat's TTS speaks whole sentences anyway, so holding the tail costs no latency. At the end
    the tail is spoken, unless Gemini stopped at max_tokens (then it is a cut-off fragment like
    „…ამის“) or the reply already has ``max_sentences`` sentences.
    """

    def __init__(self, max_sentences: int):
        self._max = max_sentences
        self._sentences = 0
        self._pending = ""
        self._truncated = False
        self.full = False

    def feed(self, chunk: GenerateContentResponse) -> bool:
        """Rewrites the chunk's text to the sentences it completes. False if nothing to pass on."""
        for candidate in chunk.candidates or []:
            if candidate.finish_reason == FinishReason.MAX_TOKENS:
                self._truncated = True
        parts = [
            p
            for c in chunk.candidates or []
            for p in ((c.content.parts if c.content else None) or [])
            if p.text and not p.thought
        ]
        if not parts:
            return not self.full and _chunk_has_content(chunk)
        text = self._pending + "".join(p.text for p in parts)
        out = []
        start = 0
        for m in _SENTENCE_END.finditer(text):
            out.append(text[start : m.end()])
            start = m.end()
            self._sentences += 1
            if self._sentences >= self._max:
                self.full = True
                break
        self._pending = "" if self.full else text[start:]
        parts[0].text = "".join(out)
        for p in parts[1:]:
            p.text = ""
        return bool(parts[0].text)

    def finish(self) -> GenerateContentResponse | None:
        """The held-back tail as a last chunk, or None when it must not be spoken."""
        tail, self._pending = self._pending.strip(), ""
        if not tail or self.full:
            return None
        if self._truncated:
            logger.warning(f"Gemini hit max_tokens; dropping the cut-off end of the reply: {tail!r}")
            return None
        return GenerateContentResponse(candidates=[Candidate(content=Content(role="model", parts=[Part(text=" " + tail)]))])


def _merge_consecutive_user_turns(messages: list) -> list:
    """Joins back-to-back user turns into one utterance.

    When the caller keeps talking over the bot, each cancelled reply leaves its user turn behind,
    and Gemini gets several user turns in a row. Flash-lite then answered the first of them: on a
    live call „…მეკობრეები?“ followed by „არა … ქართული ფილმი მსგავსი“ got a pirate film again
    (2 of 3 times); joined into one turn it answered the correction 8 of 8 times.
    """
    merged: list = []
    for m in messages:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and getattr(m, "role", None) == "user"
            and getattr(prev, "role", None) == "user"
            and all(p.text is not None for p in (prev.parts or []) + (m.parts or []))
        ):
            text = " ".join(p.text.strip() for p in (prev.parts or []) + (m.parts or []) if p.text.strip())
            merged[-1] = Content(role="user", parts=[Part(text=text)])
        else:
            merged.append(m)
    return merged


@functools.cache
def _ssl_context() -> ssl.SSLContext:
    """One TLS context for every Gemini client (creating one costs ~0.6 s)."""
    return ssl.create_default_context(cafile=certifi.where())


def _chunk_has_content(chunk: GenerateContentResponse) -> bool:
    """True if the chunk carries spoken text or a function call (not just usage or thoughts)."""
    for candidate in chunk.candidates or []:
        for part in (candidate.content.parts if candidate.content else None) or []:
            if (part.text and not part.thought) or part.function_call:
                return True
    return False


async def _read_until_content(aiter) -> list[GenerateContentResponse]:
    """Reads stream chunks up to and including the first one with content."""
    chunks = []
    while True:
        try:
            chunk = await aiter.__anext__()
        except StopAsyncIteration:
            raise RuntimeError("Gemini returned an empty response") from None
        chunks.append(chunk)
        if _chunk_has_content(chunk):
            return chunks


class FastGoogleLLMService(GoogleLLMService):
    """Low-latency Gemini LLM service with parallel tail-latency hedging."""

    # A warm gemini-flash-lite-latest request usually returns its first token in 0.6-1.0s. When it
    # hasn't by 1 s, the hedge goes to another model family: on overloaded days (2026-10-29:
    # flash-lite took 4-10 s and returned 503s, while gemini-3.6-flash answered in ~1.1 s) a second
    # request to the same model is just as slow. A further same-model request follows at 3 s.
    _HEDGE_DELAYS_SECS = (1.0, 3.0)
    _DIVERSE_HEDGE_MODEL = "gemini-3.6-flash"
    # Past these with nothing to say, the bot says a short „one moment“ (then „a little longer“)
    # so the line isn't silent; past _GIVE_UP_SECS it apologizes and asks the caller to repeat
    _FILLER_DELAYS_SECS = (2.5, 7.0)
    _GIVE_UP_SECS = 15.0
    # On an overloaded day opening a stream took over 5 s; cancelling it then and starting over
    # (the old 5 s limit) only made the caller wait longer, so slow requests keep running
    _OPEN_TIMEOUT_SECS = 12.0

    # The Gemini client's aiohttp connector drops idle connections after 15s, and on a call the gap
    # between replies is usually longer, so each turn paid a fresh TLS handshake (~250ms+).
    # A model-metadata GET keeps the connection open; it uses no generation quota.
    _KEEPWARM_IDLE_SECS = 10.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_request_at = 0.0
        self._keepwarm_task: asyncio.Task | None = None

    def create_client(self):
        # genai.Client builds three SSL contexts from the CA bundle (~0.6 s each here), which made
        # every phone call's pipeline take 2-3 s longer to start; one shared context is enough
        self._client = genai.Client(
            api_key=self._api_key,
            http_options=HttpOptions(client_args={"verify": _ssl_context()}, async_client_args={"verify": _ssl_context(), "ssl": _ssl_context()}),
        )

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
        messages = _merge_consecutive_user_turns(params["messages"])
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
            timeout=self._OPEN_TIMEOUT_SECS,
        )
        aiter = raw_stream.__aiter__()
        try:
            # Only a chunk with something to say makes this request the winner. Gemini can open with
            # usage-only or thought-only chunks and then stall or end; declaring those the winner
            # stopped the hedge and left a live caller in silence.
            leading_chunks = await asyncio.wait_for(_read_until_content(aiter), timeout=self._OPEN_TIMEOUT_SECS)
            return model_name, leading_chunks, aiter, raw_stream
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
        hedges = 0
        last_err: Exception | None = None
        winner = None
        started_at = time.monotonic()
        fillers_said = 0

        def launch(model_name: str):
            if active_tasks:
                logger.debug(f"Hedging LLM request with '{model_name}'")
            task = asyncio.create_task(self._open_model_stream(context, model_name))
            active_tasks[task] = model_name

        try:
            while next_idx < len(candidates) or active_tasks:
                elapsed = time.monotonic() - started_at
                # One request at a time, plus a hedge each time a deadline in _HEDGE_DELAYS_SECS
                # passes with no answer; a failed request is replaced by the next candidate at once.
                if next_idx < len(candidates) and not active_tasks:
                    launch(candidates[next_idx])
                    next_idx += 1
                elif hedges < len(self._HEDGE_DELAYS_SECS) and elapsed >= self._HEDGE_DELAYS_SECS[hedges]:
                    # The first hedge goes to another model family: when the first model's servers
                    # are overloaded, a second request to them is just as slow
                    other = self._DIVERSE_HEDGE_MODEL if hedges == 0 else None
                    if other and other not in active_tasks.values() and _model_cooldown_until.get(other, 0.0) <= time.monotonic():
                        launch(other)
                    elif next_idx < len(candidates):
                        launch(candidates[next_idx])
                        next_idx += 1
                    hedges += 1

                if fillers_said < len(self._FILLER_DELAYS_SECS) and elapsed >= self._FILLER_DELAYS_SECS[fillers_said]:
                    # Nothing to say yet: a person would say „one moment“ instead of going silent.
                    # Spoken as its own utterance: as reply text, pipecat's sentence aggregator held
                    # it until the answer's first word arrived, so the caller heard silence anyway.
                    filler = random.choice(LLM_WAIT_FILLERS) if fillers_said == 0 else LLM_STILL_WAITING_FILLER
                    fillers_said += 1
                    logger.debug(f"Gemini slow ({elapsed:.1f}s); saying filler {filler!r}")
                    await self.push_frame(TTSSpeakFrame(filler, append_to_context=False))

                if elapsed >= self._GIVE_UP_SECS:
                    logger.error(f"No Gemini answer in {self._GIVE_UP_SECS:.0f}s; asking the caller to repeat")
                    break

                deadlines = [self._GIVE_UP_SECS]
                if hedges < len(self._HEDGE_DELAYS_SECS):
                    deadlines.append(self._HEDGE_DELAYS_SECS[hedges])
                if fillers_said < len(self._FILLER_DELAYS_SECS):
                    deadlines.append(self._FILLER_DELAYS_SECS[fillers_said])
                timeout = max(0.0, min(deadlines) - (time.monotonic() - started_at))
                if not active_tasks:
                    continue
                done, _ = await asyncio.wait(active_tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

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
                # Every model failed. Say so instead of leaving the caller in silence, which on a
                # call sounds like the line went dead.
                logger.error(f"All Gemini candidates failed; asking the caller to repeat. Last error: {last_err}")
                yield GenerateContentResponse(
                    candidates=[Candidate(content=Content(role="model", parts=[Part(text=LLM_FAILURE_REPLY)]))]
                )
                return

            if active_tasks:
                asyncio.create_task(self._cancel_losing_tasks(list(active_tasks)))
                active_tasks.clear()

            win_model, leading_chunks, aiter, raw_stream = winner
            logger.debug(f"Gemini winner model: {win_model}")
            reply = _SentenceGate(_MAX_REPLY_SENTENCES)
            try:
                for chunk in leading_chunks:
                    if reply.feed(chunk):
                        yield chunk
                while not reply.full:
                    try:
                        chunk = await asyncio.wait_for(aiter.__anext__(), timeout=_MID_STREAM_IDLE_SECS)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        # Speak what already arrived rather than keep the caller waiting
                        logger.warning(f"Gemini stream stalled for {_MID_STREAM_IDLE_SECS:.0f}s; ending reply early")
                        break
                    if reply.feed(chunk):
                        yield chunk
                if reply.full:
                    logger.debug(f"Reply reached {_MAX_REPLY_SENTENCES} sentences; dropping the rest")
                tail = reply.finish()
                if tail:
                    yield tail
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


def build_stt():
    """Deepgram Nova-3 Georgian speech-to-text, reporting its Finalize answers to the turn strategy."""
    return FlushReportingDeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=FlushReportingDeepgramSTTService.Settings(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language="ka",
            smart_format=True,
        ),
    )


def build_llm():
    """Pick the AI brain. LLM_PROVIDER=gemini (free tier, default) or anthropic (paid). Build one per call:
    the system prompt carries the time the call started."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "anthropic":
        # Claude Haiku: fast and cheap. Prompt caching cuts cost on the long system prompt.
        return AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMService.Settings(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
                system_instruction=build_system_prompt(),
                temperature=0.3,
                max_tokens=300,
                enable_prompt_caching=True,
            ),
        )

    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        settings = dict(
            model=model,
            system_instruction=build_system_prompt(),
            temperature=0.3,
            max_tokens=300,
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
        selected_rate = rate or os.getenv("EDGE_TTS_RATE", DEFAULT_EDGE_TTS_RATE).strip()
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
    stt = build_stt()

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
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_params(user_speech_timeout=0.15),
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
            BotReplyTracker(),
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
