import asyncio
import json
import os

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService

from bot import build_llm, build_tts
from prompts import GEORGIAN_PROMPT, INITIAL_GREETING
from speech_gate import PhantomSpeechFilter, SpeechOnlyAudioGate, build_user_params, build_vad_processor

load_dotenv()

app = FastAPI(title="Georgian AI Phone Helper")

BROWSER_SAMPLE_RATE = 16000


class BrowserAudioInput(FrameProcessor):
    """Pushes audio from browser websocket into Pipecat pipeline."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def push_pcm(self, pcm_bytes: bytes):
        frame = InputAudioRawFrame(
            audio=pcm_bytes,
            sample_rate=BROWSER_SAMPLE_RATE,
            num_channels=1,
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)


class BrowserAudioOutput(FrameProcessor):
    """Sends audio and transcripts back through browser websocket."""

    def __init__(self, websocket: WebSocket):
        super().__init__()
        self._websocket = websocket

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame):
            try:
                # Send raw binary audio buffer to browser
                await self._websocket.send_bytes(frame.audio)
            except Exception:
                pass
        elif isinstance(frame, InterruptionFrame):
            try:
                await self._websocket.send_text(json.dumps({"type": "clear_audio"}))
            except Exception:
                pass
        elif isinstance(frame, TranscriptionFrame):
            # User spoke something (Deepgram transcribed it)
            if frame.text and frame.text.strip():
                try:
                    await self._websocket.send_text(json.dumps({
                        "type": "transcript_user",
                        "text": frame.text.strip()
                    }))
                except Exception:
                    pass
        elif isinstance(frame, TextFrame) and not getattr(frame, "will_be_spoken", False):
            # Bot spoke something
            if frame.text and frame.text.strip():
                try:
                    await self._websocket.send_text(json.dumps({
                        "type": "transcript_bot",
                        "text": frame.text.strip()
                    }))
                except Exception:
                    pass

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
    audio_out = BrowserAudioOutput(websocket)

    # 1. Listen (Deepgram Georgian)
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language="ka",
            smart_format=True,
        ),
    )

    # 2. Think (Gemini / Claude)
    llm = build_llm()

    # 3. Speak: Native Georgian Voice (0% foreign accent, recorded in Georgia)
    selected_voice = websocket.query_params.get("voice") or os.getenv("EDGE_TTS_VOICE", "ka-GE-EkaNeural")
    selected_rate = websocket.query_params.get("rate") or os.getenv("EDGE_TTS_RATE", "+0%")
    logger.info(f"Setting up TTS: voice={selected_voice}, rate={selected_rate}")
    tts = build_tts(
        sample_rate=BROWSER_SAMPLE_RATE,
        voice=selected_voice,
        rate=selected_rate,
    )

    # VAD before STT + phantom filter: never answer transcripts Deepgram invents from silence
    vad = build_vad_processor(stop_secs=0.2)
    phantom_filter = PhantomSpeechFilter()
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_params(user_speech_timeout=0.18),
    )

    pipeline = Pipeline(
        [
            audio_in,
            vad,
            SpeechOnlyAudioGate(),
            stt,
            phantom_filter,
            context_aggregator.user(),
            llm,
            tts,
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
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("server_web:app", host=host, port=port, reload=False)
