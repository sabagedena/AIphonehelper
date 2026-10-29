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
       - Dynamic Foreground Speaker Gate tracks the ambient noise floor, the loudest recent
         background (TV, distant talkers), and the foreground speaker's close-talk voice
         level, plus speech-band concentration (120–3400 Hz) and pitch-band harmonic crest
         (130–1000 Hz). Distant background
         voices/TV and ambient noise are replaced with pure digital silence (0), while
         a 3-hop (48 ms) lookahead and 256 ms hangover preserve every foreground word
         from its first consonant to its final release.
    2. ``SpeechOnlyAudioGate``:
       - Only forwards voice-isolated audio to Deepgram while VAD confirms human speech,
         with a 0.35s pre-roll and 0.24s clean trailing silence before STT finalize.
    3. ``PhantomSpeechFilter``:
       - Drops any low-confidence or noise-induced transcripts.
    4. ``NoiseSafeUserTurnStartStrategy`` + ``BotReplyTracker``:
       - While the bot is thinking or speaking (including gaps between its sentences),
         only recognized words (not VAD alone) may interrupt it, so a cough or a door slam
         never cuts the bot off.
    5. ``CompleteUtteranceUserTurnStopStrategy`` + ``FlushReportingDeepgramSTTService``:
       - Ends the caller's turn only after Deepgram has delivered the last words, and waits
         longer when the words so far sound unfinished, so a pause mid-sentence doesn't
         send half a question to the LLM.
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
from deepgram.listen.v1.types import ListenV1Results
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ControlFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
from pipecat.turns.user_start import BaseUserTurnStartStrategy
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# Deepgram confidence below this is treated as a noise hallucination. On live calls real Georgian
# words scored 0.57-0.64 and were lost, while noise fragments scored 0.17-0.39.
MIN_STT_CONFIDENCE = float(os.getenv("MIN_STT_CONFIDENCE", "0.55"))
# Minimum foreground voice-band RMS (int16 scale) required to open the voice isolator gate.
MIN_VOICE_RMS = float(os.getenv("MIN_VOICE_RMS", "600.0"))
# Starting estimate of the caller's close-talk voice level (int16 voice-band RMS), and caps on how far
# a loud caller can raise the onset / sustain bar above MIN_VOICE_RMS
_DEFAULT_FOREGROUND_RMS = 1800.0
_MAX_FOREGROUND_ONSET_RMS = 750.0
_MAX_FOREGROUND_SUSTAIN_RMS = 400.0
# The background tracker rises gradually (~0.16 s time constant): a TV plays for many seconds, while a
# caller's soft first words last one or two and would otherwise be learned as background (seen on a
# live call: soft speech raised it from 285 to 1214 in 2 s and was never heard)
_BACKGROUND_ATTACK = float(os.getenv("GATE_BACKGROUND_ATTACK", "0.1"))
_BACKGROUND_MAX_SHARE = float(os.getenv("GATE_BACKGROUND_MAX_SHARE", "0.6"))
# A one-word transcript may interrupt the speaking bot only above this Deepgram confidence.
BARGE_IN_MIN_CONFIDENCE = float(os.getenv("BARGE_IN_MIN_CONFIDENCE", "0.80"))
# How long after VAD says "stopped" we still accept late STT finals.
TRANSCRIPT_GRACE_SECS = 2.0


def _plain_words(text: str) -> list[str]:
    """Lower-cased words without punctuation, for comparing transcripts."""
    return re.findall(r"[^\W\d_]+", text.lower())


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

        # Speech band (120 Hz - 3400 Hz, fundamental through formants) & pitch harmonic band (130 Hz - 1000 Hz).
        # The band must include the voice fundamental: vowels like 'ო'/'უ' carry most of their energy
        # below 250 Hz, and a 250 Hz floor scored them as 5% speech and zeroed whole words.
        self.formant_bins = (freqs >= 120.0) & (freqs <= 3400.0)
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
        # Recent loudest background (TV, other talkers): fast attack, ~3.7s half-life decay
        self.background_peak_rms = 0.0
        # Voice-band RMS of the last 3 hops (48 ms). The background tracker follows their median, and
        # only on harmonic (voice-like) hops: it exists to out-shout TV and distant talkers, and a door
        # knock or clap (broadband, a few hops) must not raise the bar for the caller's next words.
        self.recent_voice_rms: list[float] = []
        self.foreground_voice_rms = _DEFAULT_FOREGROUND_RMS
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
            self.recent_voice_rms.append(voice_rms)
            if len(self.recent_voice_rms) > 3:
                self.recent_voice_rms.pop(0)

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
                # Distant TV or talkers have speech-like spectra, so a new onset must also clearly
                # beat the loudest recent background, not just its average level
                self.background_peak_rms * 2.0,
                # Capped: a caller who spoke loudly once must still be heard when they answer softly
                min(self.foreground_voice_rms * (0.42 if in_bot_echo_guard else 0.30), _MAX_FOREGROUND_ONSET_RMS),
            )
            sustain_thresh = max(
                self.min_voice_rms * 0.55,
                self.noise_floor_rms * 1.8,
                # A knock or clap can open the gate over a playing TV; don't let the TV hold it open
                self.background_peak_rms * 0.8,
                min(self.foreground_voice_rms * 0.16, _MAX_FOREGROUND_SUSTAIN_RMS),
            )

            thresh = sustain_thresh if self.is_foreground_active else onset_thresh
            req_ratio = 0.32 if self.is_foreground_active else 0.45
            req_crest = 2.2 if self.is_foreground_active else 3.2
            # Speech sustains for 3+ hops (48 ms); a knock or clap is loud for 1-2 and then decays.
            # A one-hop shortcut here let every knock open the gate over a playing TV for ~300 ms,
            # long enough for Silero to hear the TV as the caller. The 3-hop lookahead still
            # delivers the word's first consonant.
            req_streak = 4 if in_bot_echo_guard else 3

            hop_is_voice = (voice_rms >= thresh) and (formant_ratio >= req_ratio) and (pitch_crest >= req_crest)

            if hop_is_voice:
                self.active_streak += 1
                if self.active_streak >= req_streak or self.is_foreground_active:
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
                    # The level above only ever rose toward loud peaks, so one loud sentence made every
                    # quieter answer look like distant background. Ease back to the default (~6 s).
                    if self.foreground_voice_rms > _DEFAULT_FOREGROUND_RMS:
                        self.foreground_voice_rms = max(_DEFAULT_FOREGROUND_RMS, self.foreground_voice_rms * 0.998)

                # Update background noise profile when foreground user is not speaking (and bot isn't speaking).
                # Falls fast but rises at most ~1.5% per hop, so a word the detector hasn't confirmed yet
                # isn't learned as "noise" (which would raise the onset threshold and mute the speaker),
                # while a steady TV or street noise is still learned within about a second.
                if not self.is_foreground_active and not in_bot_echo_guard:
                    sustained_rms = sorted(self.recent_voice_rms)[len(self.recent_voice_rms) // 2]
                    if sustained_rms > self.background_peak_rms and pitch_crest >= 6.0:
                        self.background_peak_rms += _BACKGROUND_ATTACK * (sustained_rms - self.background_peak_rms)
                    else:
                        self.background_peak_rms *= 0.997
                    # Never above a fraction of the caller's own level: a soft-spoken caller whose first
                    # words haven't opened the gate yet must not become the "background" that keeps it shut
                    self.background_peak_rms = min(
                        self.background_peak_rms, _BACKGROUND_MAX_SHARE * self.foreground_voice_rms
                    )
                    self.noise_mag = np.where(
                        mag < self.noise_mag,
                        0.85 * self.noise_mag + 0.15 * mag,
                        np.minimum(self.noise_mag * 1.015, 0.90 * self.noise_mag + 0.10 * mag),
                    ).astype(np.float32)
                    if voice_rms < self.noise_floor_rms:
                        target = 0.85 * self.noise_floor_rms + 0.15 * voice_rms
                    else:
                        target = min(self.noise_floor_rms * 1.015, 0.92 * self.noise_floor_rms + 0.08 * voice_rms)
                    self.noise_floor_rms = float(np.clip(target, 40.0, 1200.0))
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
        # Words of the last final passed on, and where in Deepgram's audio stream it ended (seconds)
        self._last_final_words: list[str] = []
        self._last_final_end = 0.0

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
            # Require higher confidence for tiny 1-2 character noise fragments, but not for the short
            # answers callers actually give („კი“, „ჰო“): dropping one leaves the bot silent
            tiny_fragment = len(text) <= 2 and not all(w in _COMPLETE_SHORT_REPLIES for w in _plain_words(text))
            req_conf = max(self._min_confidence, 0.78) if tiny_fragment else self._min_confidence
            if conf is not None and conf < req_conf:
                logger.debug(f"Dropped low-confidence transcript ({conf:.2f} < {req_conf:.2f}): {text!r}")
                return
            if isinstance(frame, TranscriptionFrame):
                if self._repeats_previous_final(frame):
                    logger.debug(f"Dropped repeated tail word from Deepgram's flush: {text!r}")
                    return
                result = getattr(frame, "result", None)
                self._last_final_words = _plain_words(text)
                self._last_final_end = float(getattr(result, "start", 0.0) or 0.0) + float(
                    getattr(result, "duration", 0.0) or 0.0
                )

        await self.push_frame(frame, direction)

    def _repeats_previous_final(self, frame: TranscriptionFrame) -> bool:
        """Deepgram sometimes re-transcribes the last ~0.2 s of audio as the word the previous final
        already ended on („…თვეა წელიწადში?“ then „წელიწადში.“), in its answer to a Finalize or in a
        regular final (a live call got „სანაპიროზე.“ three times)."""
        result = getattr(frame, "result", None)
        if not self._last_final_words:
            return False
        start = float(getattr(result, "start", 0.0) or 0.0)
        duration = float(getattr(result, "duration", 0.0) or 0.0)
        if duration > 0.5 or abs(start - self._last_final_end) > 0.1:
            return False
        words = _plain_words(frame.text)
        return bool(words) and self._last_final_words[-len(words):] == words


@dataclass
class STTFlushedFrame(ControlFrame):
    """Deepgram has answered a Finalize: every word spoken before the last VAD stop is transcribed.

    A control frame, not a system frame, so it reaches the turn strategy after the transcript it follows.
    Uninterruptible: the caller's barge-in clears queued frames, and losing this one made the turn
    wait the full flush timeout.
    """

    interruptible: bool = field(default=False, init=False)


class FlushReportingDeepgramSTTService(DeepgramSTTService):
    """Deepgram STT that reports when its answer to a Finalize request (sent on each VAD stop) arrives.

    Deepgram sends the last words of an utterance in that answer, 0.1-0.4 s after VAD stops. A turn
    that ended before it lost them: on live calls „ამ ბავშვებისთვის შეიძლება ამ ფილმის ყურება?“
    reached the LLM as „ბავშვებისთვის“. The answer can also be empty, which pipecat doesn't
    forward, so this frame is pushed either way.
    """

    async def _on_message(self, message):
        await super()._on_message(message)
        if isinstance(message, ListenV1Results) and message.is_final and getattr(message, "from_finalize", False):
            await self.push_frame(STTFlushedFrame())


# Words a Georgian sentence doesn't end on: conjunctions, demonstratives and possessives waiting for
# their noun („…ფილმი მინდა რომ“, „კომპანიას ჰქვია ეგ“)
_CONTINUATION_WORDS = frozenset(
    "და რომ თუ მაგრამ ან ანუ რადგან იმიტომ როცა როდესაც როგორც სანამ ხოლო თან ოღონდ მერე მერედა "
    "ჯერ აი ეს ის ეგ ამ იმ მაგ ჩემი ჩემს შენი თქვენი რაღაც რაღაცა ერთი ძალიან უფრო ყველაზე "
    "მე შენ ჩვენ თქვენ მინდა მაინტერესებს მითხარი მითხარით მომიყევი მომიყევით".split()
)
# One-word turns that are a complete reply on their own
_COMPLETE_SHORT_REPLIES = frozenset(
    "კი ჰო ხო დიახ ჰოდა არა არ კარგი კარგით კაი კარგად მადლობა გმადლობთ გაიხარე ნახვამდის "
    "გასაგებია გავიგე ზუსტად სწორია ალო გისმენთ მესმის ვერა ვერ".split()
)
# How long past the base timeout to keep listening, by how finished the caller's words sound
_EXTRA_WAIT_UNPUNCTUATED_SECS = 0.25
# Deepgram often ends a finished Georgian question with a comma („…რომელ ხაზზეა,“), so a comma
# alone gets a shorter wait than a conjunction does
_EXTRA_WAIT_COMMA_SECS = 0.6
_EXTRA_WAIT_ONE_WORD_SECS = 0.9
_EXTRA_WAIT_DANGLING_SECS = 1.2


def utterance_end_wait(text: str) -> float:
    """Extra silence (seconds) to wait before ending a turn whose transcript is ``text``.

    Older callers pause mid-sentence, and Deepgram puts final punctuation only on words that sound
    finished, so a turn that ends without it, on a lone word, or on a comma or conjunction is usually
    a pause, not the end. On live calls „მითხარი, … როგორ მოვუარო, … ჩემი ყვავილი დაჭკნა“ reached
    the LLM as three questions.
    """
    text = text.strip()
    if not text:
        return 0.0
    trailing_off = text.endswith(("...", "…"))
    if text[-1] in ".?!" and not trailing_off:
        return 0.0
    words = _plain_words(text)
    if not words:
        return 0.0
    if words[-1] in _CONTINUATION_WORDS:
        return _EXTRA_WAIT_DANGLING_SECS
    if trailing_off or text[-1] in ",;:-–—":
        return _EXTRA_WAIT_COMMA_SECS
    if len(words) == 1:
        return 0.0 if words[0] in _COMPLETE_SHORT_REPLIES else _EXTRA_WAIT_ONE_WORD_SECS
    if all(w in _COMPLETE_SHORT_REPLIES for w in words):
        return 0.0
    return _EXTRA_WAIT_UNPUNCTUATED_SECS


class CompleteUtteranceUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Ends the user turn once Deepgram has flushed the last words and the sentence sounds finished.

    After VAD stops: waits for Deepgram's answer to the Finalize (at most ``stt_flush_timeout``), then
    for ``user_speech_timeout`` plus ``utterance_end_wait(text)``, all counted from the VAD stop. Any
    VAD start in between cancels the wait and the turn continues.
    """

    def __init__(self, *, user_speech_timeout: float = 0.15, stt_flush_timeout: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self._user_speech_timeout = user_speech_timeout
        self._stt_flush_timeout = stt_flush_timeout
        self._text = ""
        self._turn_active = False
        self._vad_speaking = False
        # monotonic time of the last VAD stop (each one sends Deepgram a Finalize), and of its last answer
        self._vad_stopped_at: float | None = None
        self._flushed_at = 0.0
        self._timer: asyncio.Task | None = None

    async def handle_user_turn_started(self):
        await super().handle_user_turn_started()
        self._text = ""
        self._turn_active = True
        await self._cancel_timer()

    async def handle_user_turn_stopped(self):
        await super().handle_user_turn_stopped()
        self._text = ""
        self._turn_active = False
        self._vad_stopped_at = None
        await self._cancel_timer()

    async def cleanup(self):
        await self._cancel_timer()
        await super().cleanup()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._vad_speaking = True
            await self._cancel_timer()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_speaking = False
            self._vad_stopped_at = time.monotonic()
            await self._schedule()
        elif isinstance(frame, STTFlushedFrame):
            self._flushed_at = time.monotonic()
            await self._schedule()
        elif isinstance(frame, TranscriptionFrame) and self._turn_active:
            self._text = f"{self._text} {frame.text}".strip()
            if not self._vad_speaking and self._vad_stopped_at is None:
                # Words with no VAD stop this turn (the turn began on them): time the silence from now
                self._vad_stopped_at = time.monotonic()
                self._flushed_at = self._vad_stopped_at
            await self._schedule()
        return ProcessFrameResult.CONTINUE

    async def _cancel_timer(self):
        if self._timer:
            await self.task_manager.cancel_task(self._timer)
            self._timer = None

    async def _schedule(self):
        await self._cancel_timer()
        if not self._turn_active or self._vad_speaking or not self._text or self._vad_stopped_at is None:
            return
        flushed = self._flushed_at >= self._vad_stopped_at
        if flushed:
            deadline = self._vad_stopped_at + self._user_speech_timeout + utterance_end_wait(self._text)
        else:
            deadline = self._vad_stopped_at + self._stt_flush_timeout
        self._timer = self.task_manager.create_task(self._wait(deadline, flushed), f"{self}::end_of_turn")

    async def _wait(self, deadline: float, flushed: bool):
        try:
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        except asyncio.CancelledError:
            return
        self._timer = None
        if not flushed:
            logger.debug(f"{self}: no Deepgram flush within {self._stt_flush_timeout:.1f}s; ending turn without it")
            self._flushed_at = time.monotonic()
            await self._schedule()
            return
        await self.trigger_user_turn_stopped()


@dataclass
class BotReplyFrame(SystemFrame):
    """Upstream note from ``BotReplyTracker``: the bot's reply is being synthesized (``active``) or not."""

    active: bool = False


class BotReplyTracker(FrameProcessor):
    """Tells the turn strategy, upstream, while a reply is still being synthesized.

    The output transport reports "bot stopped speaking" whenever 0.35 s pass without audio, which
    happens between two sentences of one reply when the next one's audio is late. To the turn
    strategy the bot then looked idle, and a cough or TV noise in that gap started a user turn and
    cut the rest of the reply off. Place it right after the TTS service.
    """

    def __init__(self):
        super().__init__()
        self._active = False
        # An LLM response is open (its end frame not yet through), and TTS is synthesizing. The
        # „ერთი წამით“ filler is a TTS utterance of its own inside an open response, so between it
        # and the answer only the first is true.
        self._llm_open = False
        self._tts_open = False

    async def _update(self):
        active = self._llm_open or self._tts_open
        if active != self._active:
            self._active = active
            await self.push_frame(BotReplyFrame(active=active), FrameDirection.UPSTREAM)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self._llm_open = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._llm_open = False
        elif isinstance(frame, TTSStartedFrame):
            self._tts_open = True
        elif isinstance(frame, TTSStoppedFrame):
            # After TTSStoppedFrame all audio is queued, so BotStoppedSpeakingFrame marks the real end
            self._tts_open = False
        elif isinstance(frame, InterruptionFrame):
            self._llm_open = self._tts_open = False
        else:
            return
        await self._update()


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


class NoiseSafeUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Starts the user turn on VAD while the bot is idle, but only on real words while it is busy.

    A VAD start fires on coughs, TV, door slams and barks too. While the bot is idle that's
    harmless (a turn with no transcript gets no answer). But while the bot is thinking or
    speaking, a turn start interrupts it, and when no words follow, pipecat asks the caller to
    repeat themselves. So while the bot is busy, the caller barges in only once Deepgram has
    heard words that passed ``PhantomSpeechFilter``: two interim words, or a final transcript
    (a single-word final needs higher confidence, e.g. „კი“ / „არა“).

    While the reply is still being prepared and nothing is playing yet, no echo or speaker noise can
    be heard, so any recognized word counts: a caller who paused mid-sentence and goes on („მეტრო …
    ახმეტელის თეატრი რომელ ხაზზეა?“) cancels the half-question's reply, and the LLM answers both parts.
    """

    # A gap between two sentences of one reply never lasts this long (pipecat's TTS stop timeout is 3 s)
    _REPLY_GAP_MAX_SECS = 3.0

    def __init__(self, *, reply_wait_secs: float = 6.0, min_interim_words: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._reply_wait_secs = reply_wait_secs
        self._min_interim_words = min_interim_words
        self._bot_speaking = False
        self._bot_stopped_at = 0.0
        # Set by BotReplyTracker while the reply's audio is still being synthesized
        self._reply_active = False
        self._turn_active = False
        # Set when the user's turn ends and the LLM starts thinking; cleared once the bot speaks.
        # Expires on its own because an empty turn never gets a reply.
        self._awaiting_reply_since: float | None = None
        # Deepgram sometimes sends a late extra final for words it already transcribed. Audio only
        # reaches it after a VAD start, so a transcript with no VAD start since the turn ended is
        # such a tail, not the caller speaking again, and must not cancel the reply being prepared.
        self._voice_since_turn_end = True
        # The caller's current speech began before the reply's audio did, so it can't be echo of the
        # bot: one recognized word is enough to let them finish
        self._caller_spoke_first = False

    def _bot_audible(self) -> bool:
        """The bot is playing a reply, or is between two of its sentences."""
        if self._bot_speaking:
            return True
        return self._reply_active and (time.monotonic() - self._bot_stopped_at) < self._REPLY_GAP_MAX_SECS

    def _bot_busy(self) -> bool:
        if self._bot_audible():
            return True
        since = self._awaiting_reply_since
        return since is not None and (time.monotonic() - since) < self._reply_wait_secs

    async def handle_user_turn_started(self):
        await super().handle_user_turn_started()
        self._turn_active = True
        self._awaiting_reply_since = None
        self._caller_spoke_first = False

    async def handle_user_turn_stopped(self):
        await super().handle_user_turn_stopped()
        self._turn_active = False
        self._awaiting_reply_since = time.monotonic()
        self._voice_since_turn_end = False
        self._caller_spoke_first = False

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._awaiting_reply_since = None
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = time.monotonic()
            # Still mid-reply (the filler played, the answer hasn't): keep waiting for it as after
            # the caller's turn, so noise can't start a turn that cancels it
            self._awaiting_reply_since = time.monotonic() if self._reply_active else None
        elif isinstance(frame, BotReplyFrame):
            self._reply_active = frame.active
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._voice_since_turn_end = True
            self._caller_spoke_first = not self._bot_audible()
            if not self._bot_busy():
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if self._is_real_speech(frame):
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
            if isinstance(frame, TranscriptionFrame) and not self._turn_active:
                # Don't let a rejected noise word leak into the caller's next turn
                await self.trigger_reset_aggregation()
        return ProcessFrameResult.CONTINUE

    def _is_real_speech(self, frame: TranscriptionFrame | InterimTranscriptionFrame) -> bool:
        words = len(frame.text.split())
        if words == 0:
            return False
        if not self._bot_busy():
            return True
        if not self._voice_since_turn_end:
            return False
        if not self._bot_audible() or self._caller_spoke_first:
            return True
        if isinstance(frame, InterimTranscriptionFrame):
            return words >= self._min_interim_words
        if words >= 2:
            return True
        conf = _deepgram_confidence(frame)
        return conf is None or conf >= BARGE_IN_MIN_CONFIDENCE


def build_user_params(user_speech_timeout: float = 0.15, mute_strategies=None) -> LLMUserAggregatorParams:
    """Aggregator params: noise-safe turn start (VAD when idle, words to barge in), and a turn end that
    waits for Deepgram's last words and for the sentence to sound finished.

    MUTE_USER_WHILE_BOT_SPEAKS=true ignores the caller while the bot talks (no barge-in), on every
    entry point."""
    if mute_strategies is None and os.getenv("MUTE_USER_WHILE_BOT_SPEAKS", "false").strip().lower() == "true":
        mute_strategies = [AlwaysUserMuteStrategy()]
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[NoiseSafeUserTurnStartStrategy()],
            stop=[CompleteUtteranceUserTurnStopStrategy(user_speech_timeout=user_speech_timeout)],
        ),
        user_mute_strategies=mute_strategies or [],
    )
