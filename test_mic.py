import asyncio
import math
import os
import struct
import sys
import pyaudio

from dotenv import load_dotenv
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "WARNING"))

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from bot import build_llm, build_stt, build_tts
from prompts import GEORGIAN_PROMPT, INITIAL_GREETING
from speech_gate import BotReplyTracker, PhantomSpeechFilter, SpeechOnlyAudioGate, build_user_params, build_vad_processor

load_dotenv()

MIC_SAMPLE_RATE = 16000


def _probe_level(p: pyaudio.PyAudio, idx: int, secs: float = 0.6) -> float | None:
    """Open the device at 16 kHz and return its RMS level, or None if it can't be opened."""
    try:
        s = p.open(format=pyaudio.paInt16, channels=1, rate=MIC_SAMPLE_RATE, input=True,
                   input_device_index=idx, frames_per_buffer=320)
    except Exception:
        return None
    try:
        data = b"".join(s.read(320, exception_on_overflow=False) for _ in range(int(secs * 50)))
    finally:
        s.stop_stream()
        s.close()
    data = data[3200:]
    n = len(data) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"{n}h", data)
    return math.sqrt(sum(x * x for x in shorts) / n)


def find_best_input_device():
    """Pick a working microphone.

    Order: MIC_DEVICE_INDEX from .env -> headset mic (WASAPI first, it runs natively
    at 16 kHz with no resampling) -> laptop Microphone Array -> system default.
    Devices that can't open at 16 kHz or deliver pure digital silence (muted /
    disabled in Windows privacy settings) are skipped.
    """
    p = pyaudio.PyAudio()
    try:
        forced = os.getenv("MIC_DEVICE_INDEX", "").strip()
        if forced:
            idx = int(forced)
            return idx, p.get_device_info_by_index(idx)["name"], _probe_level(p, idx)

        def host(d):
            return p.get_host_api_info_by_index(d["hostApi"])["name"]

        # Prefer MME on Windows for rock-solid 16 kHz software resampling across all drivers
        host_rank = {"MME": 0, "Windows DirectSound": 1, "Windows WASAPI": 2}
        devices = []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            name = d.get("name", "")
            if d.get("maxInputChannels", 0) <= 0 or host(d) not in host_rank:
                continue
            if "Stereo Mix" in name or "Hands-Free" in name or "Sound Mapper" in name:
                continue
            if "Headset" in name or "MAJOR" in name:
                kind = 0
            elif "Microphone" in name:
                kind = 1
            else:
                kind = 2
            devices.append((kind, host_rank[host(d)], i, name))

        for _, _, idx, name in sorted(devices):
            level = _probe_level(p, idx)
            if level is None:
                continue  # can't open at 16 kHz
            if level < 3:
                print(f"  ! Skipping '{name}' - it records pure silence (muted or disabled).")
                continue
            return idx, name, level
        return None, None, None
    finally:
        p.terminate()


class UserDisplay(FrameProcessor):
    """Shows what the *user* said (only transcripts that passed the phantom filter)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            print("\n🤖 [დამხმარე საუბრობს...]")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            print("👂 [გისმენთ... შეგიძლიათ ისაუბროთ]")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            print("🎤 [თქვენ საუბრობთ...]")
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                print(f"\n🗣️  თქვენ: {text}")

        await self.push_frame(frame, direction)


class BotDisplay(FrameProcessor):
    """Shows the bot's actual reply text (placed right after the LLM)."""

    def __init__(self):
        super().__init__()
        self._buf = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buf = []
        elif isinstance(frame, LLMTextFrame):
            self._buf.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._buf).strip()
            if text:
                print(f"🤖  დამხმარე: {text}")
            self._buf = []
        elif isinstance(frame, InterruptionFrame) and self._buf:
            print("✋ [შეწყვეტილია]")
            self._buf = []

        await self.push_frame(frame, direction)


async def main():
    mic_idx, mic_name, level = find_best_input_device()

    print("\n" + "=" * 65)
    print("  GEORGIAN AI VOICE HELPER - LIVE CHAT")
    print("=" * 65)
    print(f"  - Microphone: [{mic_idx}] {mic_name or 'Default'}")
    if level is not None:
        print(f"  - Mic noise:  {level:.0f} RMS (silence)")
    print("  - Listening:  Deepgram Nova-3 (Georgian) + Silero voice gate")
    print("  - Brain:      Google Gemini (Georgian Rules)")
    print("  - Voice:      Native Georgian Voice (Eka)")
    print("  - Tip:        set MIC_DEVICE_INDEX in .env to force a mic")
    print("=" * 65 + "\n")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=mic_idx,
        )
    )

    # 0. Voice isolation & detection BEFORE STT, so only the foreground user's voice is heard
    vad = build_vad_processor()

    # 1. Listen (Deepgram Georgian)
    stt = build_stt()

    # 1b. Drop transcripts Deepgram invents from silence/noise
    phantom_filter = PhantomSpeechFilter()

    # 2. Think (Gemini / Claude)
    llm = build_llm()

    # 3. Speak (Native Georgian Voice)
    tts = build_tts(sample_rate=MIC_SAMPLE_RATE)

    # Context: turns start only on real voice; bot starts answering ~0.38s after you stop
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
            UserDisplay(),
            context_aggregator.user(),
            llm,
            BotDisplay(),
            tts,
            BotReplyTracker(),
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=MIC_SAMPLE_RATE,
            audio_out_sample_rate=MIC_SAMPLE_RATE,
            enable_metrics=False,
        ),
    )

    runner = PipelineRunner(handle_sigint=True)

    # Play initial greeting
    async def greet():
        await asyncio.sleep(0.6)
        print("\n🤖 დამხმარე: " + INITIAL_GREETING)
        await task.queue_frames([TTSSpeakFrame(INITIAL_GREETING)])

    asyncio.create_task(greet())
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nზარი დასრულდა. (Call ended)")
