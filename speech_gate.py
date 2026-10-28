"""Turn-taking helpers that stop the bot from replying to things nobody said.

Why this exists:
    Deepgram's Georgian model "hallucinates" text from pure silence or mic hiss
    (e.g. it heard "located on the right bank of the river" while nobody spoke).
    Pipecat's default turn-start strategy also starts a user turn from ANY
    transcript, so those phantom words were sent to the LLM and the bot answered.

The fix has three layers:
    1. Silero VAD runs *before* STT (``VADProcessor``) and ``SpeechOnlyAudioGate``
       only forwards mic audio to Deepgram while a real human voice is heard.
    2. Transcripts are only accepted while VAD hears a voice, or shortly after
       it stops (STT finals arrive a bit late).
    3. Transcripts with low Deepgram confidence are dropped. Real Georgian
       speech scores ~0.9-1.0, silence hallucinations score ~0.05-0.15.

User turns are started by VAD only (never by a transcript alone) and ended by a
short, predictable silence timeout instead of the Smart Turn model, which does
not support Georgian and made replies feel slow.
"""

import os
import time

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# Deepgram confidence below this is treated as a hallucination.
MIN_STT_CONFIDENCE = float(os.getenv("MIN_STT_CONFIDENCE", "0.45"))
# How long after VAD says "stopped" we still accept late STT results.
TRANSCRIPT_GRACE_SECS = 2.0


def _deepgram_confidence(frame: Frame) -> float | None:
    """Read the confidence score from a Deepgram result, if present."""
    result = getattr(frame, "result", None)
    try:
        return float(result.channel.alternatives[0].confidence)
    except Exception:
        return None


class SpeechOnlyAudioGate(FrameProcessor):
    """Only lets microphone audio reach the STT while VAD hears a human voice.

    Deepgram can't hallucinate from audio it never receives. A short rolling
    pre-roll buffer is flushed when speech starts so the first syllable is not
    clipped (VAD needs ~0.2s to confirm speech). Place between VADProcessor and STT.
    """

    def __init__(self, preroll_secs: float = 0.8):
        super().__init__()
        self._preroll_secs = preroll_secs
        self._buffer: list[InputAudioRawFrame] = []
        self._buffered_secs = 0.0
        self._open = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            if self._open:
                await self.push_frame(frame, direction)
            else:
                self._buffer.append(frame)
                self._buffered_secs += len(frame.audio) / (2 * frame.num_channels * frame.sample_rate)
                while self._buffer and self._buffered_secs > self._preroll_secs:
                    old = self._buffer.pop(0)
                    self._buffered_secs -= len(old.audio) / (2 * old.num_channels * old.sample_rate)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            self._open = True
            for buffered in self._buffer:
                await self.push_frame(buffered, direction)
            self._buffer.clear()
            self._buffered_secs = 0.0
        elif isinstance(frame, VADUserStoppedSpeakingFrame) and direction == FrameDirection.DOWNSTREAM:
            self._open = False

        await self.push_frame(frame, direction)


class PhantomSpeechFilter(FrameProcessor):
    """Drops STT transcripts that were not backed by real, detected speech.

    Must be placed after the STT service and after a ``VADProcessor``.
    """

    def __init__(self, min_confidence: float = MIN_STT_CONFIDENCE, grace_secs: float = TRANSCRIPT_GRACE_SECS):
        super().__init__()
        self._min_confidence = min_confidence
        self._grace_secs = grace_secs
        self._vad_speaking = False
        self._last_vad_stop = 0.0

    def _speech_recent(self) -> bool:
        return self._vad_speaking or (time.monotonic() - self._last_vad_stop) < self._grace_secs

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_speaking = False
            self._last_vad_stop = time.monotonic()
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            text = frame.text.strip()
            if not text:
                return
            if not self._speech_recent():
                logger.debug(f"Dropped phantom transcript (no voice detected): {text!r}")
                return
            conf = _deepgram_confidence(frame)
            if conf is not None and conf < self._min_confidence:
                logger.debug(f"Dropped low-confidence transcript ({conf:.2f}): {text!r}")
                return
            if isinstance(frame, TranscriptionFrame):
                from_finalize = bool(getattr(getattr(frame, "result", None), "from_finalize", False))
                if from_finalize or not self._vad_speaking:
                    frame.finalized = True

        await self.push_frame(frame, direction)


def build_vad_processor(stop_secs: float = 0.2) -> VADProcessor:
    """Silero VAD tuned to ignore headset hiss but catch normal speech quickly."""
    return VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.55,
                start_secs=0.15,
                stop_secs=stop_secs,
                min_volume=0.1,
            )
        )
    )


def build_user_params(user_speech_timeout: float = 0.18, mute_strategies=None) -> LLMUserAggregatorParams:
    """Aggregator params: VAD-only turn start, fast silence-timeout turn end.

    Total pause before the bot starts thinking ~= VAD stop_secs (0.2s) + user_speech_timeout (0.18s) = ~0.38s.
    VAD runs in the separate ``VADProcessor``, so no vad_analyzer here.
    """
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=user_speech_timeout)],
        ),
        user_mute_strategies=mute_strategies or [],
    )
