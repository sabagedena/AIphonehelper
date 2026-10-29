import asyncio
import json
import os
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger
from dotenv import load_dotenv

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

from bot import DEFAULT_EDGE_TTS_RATE, build_llm, build_stt, build_tts
from prompts import INITIAL_GREETING
from speech_gate import BotReplyTracker, PhantomSpeechFilter, SpeechOnlyAudioGate, build_user_params, build_vad_processor

load_dotenv()

app = FastAPI(title="Georgian AI Phone Helper")

BROWSER_SAMPLE_RATE = 16000


class BrowserAudioInput(FrameProcessor):
    """Pushes audio from browser websocket into Pipecat pipeline."""

    def __init__(self):
        super().__init__()
        self._started = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._started = True
        await self.push_frame(frame, direction)

    async def push_pcm(self, pcm_bytes: bytes):
        # Mic audio that arrives before the pipeline has started would only be dropped with errors
        if not self._started:
            return
        frame = InputAudioRawFrame(
            audio=pcm_bytes,
            sample_rate=BROWSER_SAMPLE_RATE,
            num_channels=1,
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)


class BrowserAudioOutput(BaseOutputTransport):
    """Sends the bot's audio to the browser at playback speed.

    An output transport, not a plain processor: only an output transport reports
    BotStartedSpeakingFrame / BotStoppedSpeakingFrame, which the turn strategy and the echo guard
    need to know the bot is talking. Sending a whole reply in one burst also made pipecat think the
    bot had finished while the browser was still playing it.
    """

    # How far ahead of playback audio may be sent, so the browser never runs dry
    _LEAD_SECS = 0.12

    def __init__(self, websocket: WebSocket, params: TransportParams, **kwargs):
        super().__init__(params=params, **kwargs)
        self._websocket = websocket
        self._play_until = 0.0

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame, direction):
        if isinstance(frame, InterruptionFrame):
            self._play_until = 0.0
            await self._send_json({"type": "clear_audio"})
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        try:
            await self._websocket.send_bytes(frame.audio)
        except Exception:
            return False
        now = time.monotonic()
        self._play_until = max(self._play_until, now) + len(frame.audio) / (2 * frame.sample_rate * frame.num_channels)
        await asyncio.sleep(max(0.0, self._play_until - now - self._LEAD_SECS))
        return True

    async def _send_json(self, data: dict):
        try:
            await self._websocket.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass


class BrowserTranscriptSender(FrameProcessor):
    """Shows the call's transcript in the browser: the caller's words (place before the user
    aggregator, which consumes them) or the bot's replies (place right after the LLM)."""

    def __init__(self, websocket: WebSocket, role: str):
        super().__init__()
        self._websocket = websocket
        self._role = role
        self._bot_text: list[str] = []

    async def _send(self, text: str):
        text = text.strip()
        if not text:
            return
        try:
            await self._websocket.send_text(json.dumps({"type": f"transcript_{self._role}", "text": text}, ensure_ascii=False))
        except Exception:
            pass

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if self._role == "user" and isinstance(frame, TranscriptionFrame):
            await self._send(frame.text)
        elif self._role == "bot":
            if isinstance(frame, LLMFullResponseStartFrame):
                self._bot_text = []
            elif isinstance(frame, LLMTextFrame):
                self._bot_text.append(frame.text)
            elif isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame)) and self._bot_text:
                await self._send("".join(self._bot_text))
                self._bot_text = []
        await self.push_frame(frame, direction)


@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    """Returns the interactive in-browser phone test interface."""
    with open("web_ui.html", "r", encoding="utf-8") as f:
        return f.read()


@app.websocket("/browser-call")
async def browser_call_ws(websocket: WebSocket):
    """Bidirectional WebSocket for in-browser real-time voice call testing."""
    await websocket.accept()
    logger.info("Browser call connected!")

    audio_in = BrowserAudioInput()
    audio_out = BrowserAudioOutput(
        websocket,
        TransportParams(audio_out_enabled=True, audio_out_sample_rate=BROWSER_SAMPLE_RATE),
    )

    # 1. Listen (Deepgram Georgian)
    stt = build_stt()

    # 2. Think (Gemini / Claude)
    llm = build_llm()

    # 3. Speak: Native Georgian Voice (0% foreign accent, recorded in Georgia)
    selected_voice = websocket.query_params.get("voice") or os.getenv("EDGE_TTS_VOICE", "ka-GE-EkaNeural")
    selected_rate = websocket.query_params.get("rate") or os.getenv("EDGE_TTS_RATE", DEFAULT_EDGE_TTS_RATE)
    logger.info(f"Setting up TTS: voice={selected_voice}, rate={selected_rate}")
    tts = build_tts(
        sample_rate=BROWSER_SAMPLE_RATE,
        voice=selected_voice,
        rate=selected_rate,
    )

    # VAD + Voice Isolator before STT + phantom filter: ignore background noise & silence hallucinations
    vad = build_vad_processor()
    phantom_filter = PhantomSpeechFilter()
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_params(user_speech_timeout=0.15),
    )

    pipeline = Pipeline(
        [
            audio_in,
            vad,
            SpeechOnlyAudioGate(),
            stt,
            phantom_filter,
            BrowserTranscriptSender(websocket, "user"),
            context_aggregator.user(),
            llm,
            BrowserTranscriptSender(websocket, "bot"),
            tts,
            BotReplyTracker(),
            audio_out,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=BROWSER_SAMPLE_RATE,
            audio_out_sample_rate=BROWSER_SAMPLE_RATE,
            enable_metrics=True,
        ),
    )

    runner = PipelineRunner(handle_sigint=False)

    async def read_browser_mic():
        try:
            while True:
                msg = await websocket.receive()
                if "bytes" in msg and msg["bytes"]:
                    await audio_in.push_pcm(msg["bytes"])
                elif "text" in msg:
                    data = json.loads(msg["text"])
                    if data.get("type") == "hangup":
                        break
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as e:
            logger.debug(f"Mic loop ended: {e}")
        finally:
            try:
                await task.cancel()
            except Exception:
                pass

    async def run_pipeline():
        try:
            await runner.run(task)
        except Exception as e:
            logger.debug(f"Pipeline finished: {e}")

    # Send initial greeting
    async def greet():
        await asyncio.sleep(0.35)
        logger.info("Sending initial greeting to browser caller...")
        try:
            await task.queue_frames([TTSSpeakFrame(INITIAL_GREETING)])
        except Exception:
            pass

    asyncio.create_task(greet())
    mic_task = asyncio.create_task(read_browser_mic())
    pipe_task = asyncio.create_task(run_pipeline())

    try:
        await asyncio.wait(
            [mic_task, pipe_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for t in [mic_task, pipe_task]:
            if not t.done():
                t.cancel()
        logger.info("Browser call ended cleanly.")


if __name__ == "__main__":
    import uvicorn
    # Pay the one-time Gemini client setup (TLS context, imports) now rather than in the first call
    build_llm()
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("server_web:app", host=host, port=port, reload=False)
