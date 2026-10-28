"""Real-time foreground voice isolation, spectral noise suppression, and speech gating.

Why this exists:
    1. When a user speaks on a phone call or microphone with background noise
       (room noise, street traffic, TV, other people talking in the distance, fans,
       or 50/60 Hz hum), raw background audio can:
       - Keep VAD stuck in SPEAKING so the AI never responds or responds late,
       - Bleed background words/noise into Deepgram STT while the user is talking,
       - Falsely interrupt the AI while it is speaking.
    2. Deepgram's Georgian model can also hallucinate words from low-level hiss/noise
       if non-voice audio reaches STT.

The multi-layer solution implemented here:
    1. ``RealtimeVoiceIsolator`` + ``VoiceIsolatedVADProcessor``:
       - 50%-overlapped STFT bandpass filter (110 Hz – 3450/4500 Hz) removes rumble,
         50/60 Hz mains hum, plosive thumps, and high-frequency hiss.
       - Adaptive per-bin Wiener spectral subtraction suppresses stationary background
         noise while preserving natural Georgian consonants.
       - Dynamic Foreground Speaker Gate tracks both the ambient noise floor and the
         foreground speaker's close-talk voice level, plus vocal formant concentration
         (250–3000 Hz) and pitch-band harmonic crest (130–1000 Hz). Distant background
         voices/TV and ambient noise are replaced with pure digital silence (0), while
         a 3-hop (48 ms) lookahead and 256 ms hangover preserve every foreground word
         from its first consonant to its final release.
    2. ``SpeechOnlyAudioGate``:
       - Only forwards voice-isolated audio to Deepgram while VAD confirms human speech,
         with a 0.35s pre-roll and 0.24s clean trailing silence before STT finalize.
    3. ``PhantomSpeechFilter``:
       - Drops any low-confidence or noise-induced transcripts.
"""

import os
import time

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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

# Deepgram confidence below this is treated as a noise hallucination.
MIN_STT_CONFIDENCE = float(os.getenv("MIN_STT_CONFIDENCE", "0.65"))
# Minimum foreground voice-band RMS (int16 scale) required to open the voice isolator gate.
MIN_VOICE_RMS = float(os.getenv("MIN_VOICE_RMS", "600.0"))
# How long after VAD says "stopped" we still accept late STT finals.
TRANSCRIPT_GRACE_SECS = 2.0


def _deepgram_confidence(frame: Frame) -> float | None:
    """Read the confidence score from a Deepgram result, if present."""
    result = getattr(frame, "result", None)
    try:
        return float(result.channel.alternatives[0].confidence)
    except Exception:
        return None


class RealtimeVoiceIsolator:
    """Real-time STFT Spectral Noise Suppressor & Foreground Speaker Isolator.

    Operates on 16-bit signed mono PCM at 8 kHz (telephone) or 16 kHz (mic/browser).
    Suppresses stationary background noise via smoothed Wiener filtering and gates out
    non-foreground sounds (distant background talkers, TV, street noise, room rumble)
    so only the close-talk user's voice passes through.
    """

    def __init__(self, sample_rate: int = 8000, min_voice_rms: float = MIN_VOICE_RMS):
        self.sample_rate = sample_rate
        self.min_voice_rms = min_voice_rms
        self.bot_speaking = False
        self.bot_stopped_at = 0.0
        self._init_dsp(sample_rate)

    def _init_dsp(self, sr: int):
        self.sample_rate = sr
        # 32ms window, 16ms hop (50% overlap) -> 31.25 Hz bin resolution at 8kHz & 16kHz
        self.n_fft = 256 if sr <= 8000 else 512
        self.hop = self.n_fft // 2
        self.n_bins = self.n_fft // 2 + 1

        # Periodic sqrt-Hann window for exact reconstruction with 50% overlap-add
        t = np.arange(self.n_fft, dtype=np.float32)
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / self.n_fft))
        self.window = np.sqrt(hann).astype(np.float32)

        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sr).astype(np.float32)
        # Bandpass mask: reject <110 Hz rumble/hum/plosives and high-frequency hiss
        bp = np.ones(self.n_bins, dtype=np.float32)
        bp[freqs < 110.0] = 0.0
        bp[(freqs >= 110.0) & (freqs < 160.0)] = 0.4
        hi_cut = 3450.0 if sr <= 8000 else 4500.0
        hi_stop = 3750.0 if sr <= 8000 else 5500.0
        bp[(freqs > hi_cut) & (freqs <= hi_stop)] = 0.25
        bp[freqs > hi_stop] = 0.02
        self.bandpass_mask = bp

        # Core human vocal formant bins (250 Hz - 3000 Hz) & pitch harmonic band (130 Hz - 1000 Hz)
        self.formant_bins = (freqs >= 250.0) & (freqs <= 3000.0)
        self.pitch_bins = (freqs >= 130.0) & (freqs <= 1000.0)
        self.voice_bins = (freqs >= 140.0) & (freqs <= 3400.0)

        self.noise_mag = np.full(self.n_bins, 150.0, dtype=np.float32)
        self.noise_initialized = False
        self.prev_gain = np.ones(self.n_bins, dtype=np.float32)

        self.in_fifo = np.zeros(0, dtype=np.float32)
        self.ola_buf = np.zeros(self.n_fft, dtype=np.float32)
        self.out_fifo = np.zeros(0, dtype=np.int16)

        # 3-hop (48ms) lookahead buffer so word-initial consonants ('გ', 'კ', 'მ', 'ს') are never clipped
        self.lookahead_hops = 3
        self.hop_queue: list[np.ndarray] = []

        # Adaptive level trackers (in time-domain int16 RMS units)
        self.noise_floor_rms = 200.0
        self.foreground_voice_rms = 1800.0
        self.calibration_hops = 0
        self.active_streak = 0
        self.hangover_hops = 0
        self.max_hangover_hops = 16  # 16 * 16ms = 256ms smooth hangover across intra-sentence micro-pauses
        self.gate_env = 0.0
        self.is_foreground_active = False

    def set_bot_speaking(self, speaking: bool):
        if self.bot_speaking and not speaking:
            self.bot_stopped_at = time.monotonic()
        self.bot_speaking = speaking

    def process_pcm16(self, pcm_bytes: bytes, sample_rate: int | None = None) -> bytes:
        if sample_rate and sample_rate != self.sample_rate:
            self._init_dsp(sample_rate)

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        n_in = len(samples)
        if n_in == 0:
            return pcm_bytes

        self.in_fifo = np.concatenate((self.in_fifo, samples))
        in_bot_echo_guard = self.bot_speaking or ((time.monotonic() - self.bot_stopped_at) < 0.22)

        while len(self.in_fifo) >= self.n_fft:
            frame = self.in_fifo[: self.n_fft]
            self.in_fifo = self.in_fifo[self.hop :]

            win_frame = frame * self.window
            spec = np.fft.rfft(win_frame)
            mag = np.abs(spec).astype(np.float32)
            power = mag * mag

            total_power = float(np.sum(power[1:])) + 1e-6
            formant_power = float(np.sum(power[self.formant_bins]))
            formant_ratio = formant_power / total_power

            # Spectral crest factor in pitch band (130-1000 Hz): close-talk voiced speech has strong harmonic peaks
            pitch_power = power[self.pitch_bins]
            pitch_crest = float(np.max(pitch_power) / (np.mean(pitch_power) + 1e-6))

            # Exact Parseval voice-band RMS in int16 scale (compensating for sqrt-Hann 0.5 window power)
            voice_rms = float(np.sqrt(np.sum(power[self.voice_bins]) * 4.0 / (self.n_fft * self.n_fft)))

            if not self.noise_initialized:
                self.noise_mag = np.maximum(mag, 50.0)
                self.noise_floor_rms = float(np.clip(voice_rms, 120.0, 550.0))
                self.noise_initialized = True
            elif self.calibration_hops < 15 and voice_rms < 1100.0 and not in_bot_echo_guard:
                self.calibration_hops += 1
                self.noise_mag = np.maximum(self.noise_mag * 0.85 + mag * 0.15, mag * 0.7)
                self.noise_floor_rms = float(
                    np.clip(
                        max(self.noise_floor_rms * 0.9, 0.75 * self.noise_floor_rms + 0.25 * voice_rms),
                        100.0,
                        600.0,
                    )
                )

            # Dynamic foreground thresholds:
            # Close-talk user voice must stand out above ambient noise floor AND be near foreground speaker level
            base_min_rms = self.min_voice_rms * (2.0 if in_bot_echo_guard else 1.0)
            onset_thresh = max(
                base_min_rms,
                self.noise_floor_rms * (3.5 if in_bot_echo_guard else 2.8),
                self.foreground_voice_rms * (0.42 if in_bot_echo_guard else 0.30),
            )
            sustain_thresh = max(
                self.min_voice_rms * 0.55,
                self.noise_floor_rms * 1.8,
                self.foreground_voice_rms * 0.16,
            )

            thresh = sustain_thresh if self.is_foreground_active else onset_thresh
            req_ratio = 0.32 if self.is_foreground_active else 0.45
            req_crest = 2.2 if self.is_foreground_active else 3.2
            req_streak = 4 if in_bot_echo_guard else 2

            hop_is_voice = (voice_rms >= thresh) and (formant_ratio >= req_ratio) and (pitch_crest >= req_crest)
            strong_onset = (
                not in_bot_echo_guard
                and (voice_rms >= onset_thresh * 1.25)
                and (formant_ratio >= 0.50)
                and (pitch_crest >= 4.0)
            )

            if hop_is_voice:
                self.active_streak += 1
                if self.active_streak >= req_streak or strong_onset or self.is_foreground_active:
                    self.is_foreground_active = True
                    self.hangover_hops = self.max_hangover_hops
                    # Adapt foreground speaker level toward clear speech peaks
                    if voice_rms > self.foreground_voice_rms * 0.55:
                        self.foreground_voice_rms = float(
                            np.clip(0.92 * self.foreground_voice_rms + 0.08 * voice_rms, 900.0, 8000.0)
                        )
            else:
                self.active_streak = 0
                if self.hangover_hops > 0:
                    self.hangover_hops -= 1
                else:
                    self.is_foreground_active = False

                # Update background noise profile when foreground user is not speaking (and bot isn't speaking)
                if not self.is_foreground_active and not in_bot_echo_guard:
                    self.noise_mag = 0.90 * self.noise_mag + 0.10 * mag
                    self.noise_floor_rms = float(
                        np.clip(0.92 * self.noise_floor_rms + 0.08 * voice_rms, 40.0, 600.0)
                    )
                elif not in_bot_echo_guard:
                    # Track downward minima even during speech pauses
                    self.noise_mag = np.minimum(self.noise_mag * 1.002, 0.80 * self.noise_mag + 0.20 * mag)

            # Wiener / spectral subtraction gain per frequency bin:
            # Keep full natural formant fidelity (floor_gain=0.65) during active foreground speech
            # so Deepgram Nova-3 hears 100% undistorted Georgian consonants, while completely
            # muting non-foreground periods via gate_env=0.0.
            snr_post = (mag * mag) / (self.noise_mag * self.noise_mag + 1e-6)
            over_sub = 0.45 if self.is_foreground_active else 2.0
            floor_gain = 0.65 if self.is_foreground_active else 0.02
            gain_raw = np.clip((snr_post - over_sub) / (snr_post + 1e-6), floor_gain, 1.0).astype(np.float32)
            gain_smooth = np.convolve(gain_raw, np.array([0.25, 0.50, 0.25], dtype=np.float32), mode="same")
            gain = 0.70 * gain_smooth + 0.30 * self.prev_gain
            self.prev_gain = gain

            clean_un_gated = (
                np.fft.irfft(spec * (gain * self.bandpass_mask), n=self.n_fft).astype(np.float32) * self.window
            )
            self.hop_queue.append(clean_un_gated)
            if len(self.hop_queue) <= self.lookahead_hops:
                continue

            delayed_frame = self.hop_queue.pop(0)

            # Because delayed_frame is 3 hops (48ms) behind the detector, when is_foreground_active
            # turns True on the vowel, delayed_frame is right at the start of the initial consonant!
            if self.is_foreground_active:
                if self.hangover_hops > 4:
                    self.gate_env = 1.0
                else:
                    self.gate_env = self.hangover_hops / 5.0
            else:
                self.gate_env = 0.0

            enhanced_frame = delayed_frame * self.gate_env

            self.ola_buf += enhanced_frame
            out_hop = np.clip(self.ola_buf[: self.hop], -32768.0, 32767.0).astype(np.int16)
            self.ola_buf[: self.hop] = self.ola_buf[self.hop :]
            self.ola_buf[self.hop :] = 0.0

            self.out_fifo = np.concatenate((self.out_fifo, out_hop))

        if len(self.out_fifo) >= n_in:
            out = self.out_fifo[:n_in]
            self.out_fifo = self.out_fifo[n_in:]
        else:
            pad = np.zeros(n_in - len(self.out_fifo), dtype=np.int16)
            out = np.concatenate((self.out_fifo, pad))
            self.out_fifo = np.zeros(0, dtype=np.int16)

        return out.tobytes()


class VoiceIsolatedVADProcessor(VADProcessor):
    """VADProcessor with built-in real-time STFT spectral noise suppression and foreground voice isolation.

    Cleans every incoming ``InputAudioRawFrame`` before both Silero VAD analysis and
    downstream forwarding to ``SpeechOnlyAudioGate`` and ``DeepgramSTTService``.
    """

    def __init__(self, *, min_voice_rms: float = MIN_VOICE_RMS, **kwargs):
        super().__init__(**kwargs)
        self._isolator = RealtimeVoiceIsolator(sample_rate=8000, min_voice_rms=min_voice_rms)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._isolator.set_bot_speaking(True)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._isolator.set_bot_speaking(False)
        elif isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            cleaned_pcm = self._isolator.process_pcm16(frame.audio, frame.sample_rate)
            frame = InputAudioRawFrame(
                audio=cleaned_pcm,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )

        await super().process_frame(frame, direction)


class SpeechOnlyAudioGate(FrameProcessor):
    """Only lets voice-isolated microphone audio reach the STT while VAD hears the user's voice.

    Deepgram can't hallucinate from audio it never receives. A short rolling
    pre-roll buffer (0.35s) is flushed when speech starts so the first syllable
    is never clipped, and 0.24s of clean digital silence is flushed when speech
    stops so Deepgram immediately finalizes the last word.
    """

    def __init__(self, preroll_secs: float = 0.35, postroll_silence_secs: float = 0.24):
        super().__init__()
        self._preroll_secs = preroll_secs
        self._postroll_silence_secs = postroll_silence_secs
        self._buffer: list[InputAudioRawFrame] = []
        self._buffered_secs = 0.0
        self._open = False
        self._last_sample_rate = 8000
        self._last_num_channels = 1
        self._last_frame_bytes = 320

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._last_sample_rate = frame.sample_rate
            self._last_num_channels = frame.num_channels
            self._last_frame_bytes = len(frame.audio) or 320
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
            if self._open:
                self._open = False
                # Push clean digital silence frames before VADUserStoppedSpeakingFrame reaches Deepgram
                # so Deepgram's acoustic decoder cleanly flushes the final word before Finalize.
                frame_dur = self._last_frame_bytes / (2 * self._last_num_channels * self._last_sample_rate)
                num_silence_frames = max(1, round(self._postroll_silence_secs / max(0.01, frame_dur)))
                silence_bytes = b"\x00" * self._last_frame_bytes
                for _ in range(num_silence_frames):
                    await self.push_frame(
                        InputAudioRawFrame(
                            audio=silence_bytes,
                            sample_rate=self._last_sample_rate,
                            num_channels=self._last_num_channels,
                        ),
                        direction,
                    )

        await self.push_frame(frame, direction)


class PhantomSpeechFilter(FrameProcessor):
    """Drops STT transcripts that were not backed by real foreground user speech.

    Must be placed after the STT service and after ``VoiceIsolatedVADProcessor``.
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
                logger.debug(f"Dropped phantom transcript (no foreground voice detected): {text!r}")
                return
            conf = _deepgram_confidence(frame)
            # Require higher confidence for tiny 1-2 character noise fragments
            req_conf = max(self._min_confidence, 0.78) if len(text) <= 2 else self._min_confidence
            if conf is not None and conf < req_conf:
                logger.debug(f"Dropped low-confidence transcript ({conf:.2f} < {req_conf:.2f}): {text!r}")
                return
            if isinstance(frame, TranscriptionFrame):
                from_finalize = bool(getattr(getattr(frame, "result", None), "from_finalize", False))
                if from_finalize or not self._vad_speaking:
                    frame.finalized = True

        await self.push_frame(frame, direction)


def build_vad_processor(stop_secs: float = 0.20, min_voice_rms: float = MIN_VOICE_RMS) -> VoiceIsolatedVADProcessor:
    """Builds a VoiceIsolatedVADProcessor that strips background noise and focuses only on the user's voice.

    ``RealtimeVoiceIsolator`` provides a 256 ms foreground-voice hangover across intra-sentence
    micro-pauses, and Silero VAD adds ``stop_secs=0.20`` (matching Pipecat's STT p99 latency
    expectation) for a ~0.45s total intra-sentence bridge with zero mid-sentence chopping.
    """
    return VoiceIsolatedVADProcessor(
        min_voice_rms=min_voice_rms,
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.70,
                start_secs=0.16,
                stop_secs=stop_secs,
                min_volume=0.62,
            )
        ),
    )


def build_user_params(user_speech_timeout: float = 0.20, mute_strategies=None) -> LLMUserAggregatorParams:
    """Aggregator params: VAD-only turn start, fast silence-timeout turn end."""
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=user_speech_timeout)],
        ),
        user_mute_strategies=mute_strategies or [],
    )
