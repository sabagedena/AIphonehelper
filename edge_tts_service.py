import asyncio
from contextlib import aclosing
import io
import re
import time
from typing import AsyncGenerator, Optional

import aiohttp
import edge_tts
from edge_tts.communicate import (
    _SSL_CTX,
    TTSConfig,
    connect_id,
    date_to_string,
    escape,
    get_headers_and_data,
    mkssml,
    remove_incompatible_characters,
    split_text_by_byte_length,
    ssml_headers_plus_data,
)
from edge_tts.constants import SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL
from edge_tts.drm import DRM
from loguru import logger
import numpy as np
import soundfile as sf
import soxr

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

# Georgian numeral dictionary (0 - 9999)
_GEORGIAN_UNITS = {
    0: "ნული", 1: "ერთი", 2: "ორი", 3: "სამი", 4: "ოთხი", 5: "ხუთი",
    6: "ექვსი", 7: "შვიდი", 8: "რვა", 9: "ცხრა", 10: "ათი",
    11: "თერთმეტი", 12: "თორმეტი", 13: "ცამეტი", 14: "თოთხმეტი", 15: "თხუთმეტი",
    16: "თექვსმეტი", 17: "ჩვიდმეტი", 18: "თვრამეტი", 19: "ცხრამეტი"
}

_GEORGIAN_VIGESIMAL = {
    20: ("ოცი", "ოცდა"),
    40: ("ორმოცი", "ორმოცდა"),
    60: ("სამოცი", "სამოცდა"),
    80: ("ოთხმოცი", "ოთხმოცდა"),
}

_GEORGIAN_SCALES = (
    (1_000_000_000, "მილიარდი"),
    (1_000_000, "მილიონი"),
    (1000, "ათასი"),
)

def number_to_georgian(n: int) -> str:
    """Converts integers (0 - 999,999,999,999) into natural Georgian words."""
    if n >= 1_000_000_000_000:
        return str(n)
    for scale, name in _GEORGIAN_SCALES:
        if n >= scale:
            count, rem = divmod(n, scale)
            head = name if count == 1 else number_to_georgian(count) + " " + name
            if rem == 0:
                return head
            # The scale word drops its final 'ი' before a remainder: 'ორი ათას ოცდაექვსი'
            return head[:-1] + " " + number_to_georgian(rem)
    if n in _GEORGIAN_UNITS:
        return _GEORGIAN_UNITS[n]
    if n < 100:
        base = (n // 20) * 20
        rem = n % 20
        if rem == 0:
            return _GEORGIAN_VIGESIMAL[base][0]
        return _GEORGIAN_VIGESIMAL[base][1] + _GEORGIAN_UNITS[rem]
    if n < 1000:
        h = n // 100
        rem = n % 100
        h_str = "ასი" if h == 1 else (_GEORGIAN_UNITS[h][:-1] if _GEORGIAN_UNITS[h].endswith("ი") else _GEORGIAN_UNITS[h]) + "ასი"
        if rem == 0:
            return h_str
        prefix = "ას " if h == 1 else (_GEORGIAN_UNITS[h][:-1] if _GEORGIAN_UNITS[h].endswith("ი") else _GEORGIAN_UNITS[h]) + "ას "
        return prefix + number_to_georgian(rem)

# Matched case-insensitively, so 'Sms', 'Pin', 'WHATSAPP' are caught too
_LATIN_TO_GEORGIAN_PHONETICS = {
    r"\bPIN\b": "პინ",
    r"\bSMS\b": "ესემეს",
    r"\bWi-?Fi\b": "ვაიფაი",
    r"\bWhatsApp\b": "ვაცაპი",
    r"\bViber\b": "ვაიბერი",
    r"\bFacebook\b": "ფეისბუქი",
    r"\bGoogle\b": "გუგლი",
    r"\bYouTube\b": "იუთუბი",
    r"\bSIM\b": "სიმ",
    r"₾": " ლარი",
    r"\$": " დოლარი",
    r"%": " პროცენტი",
}

def normalize_georgian_text(text: str) -> str:
    """
    Normalizes text for natural, native Georgian neural TTS:
    - Replaces English/Latin tech terms with natural Georgian phonetics.
    - Converts digits with Georgian suffixes (e.g., '112-ზე' -> 'ას თორმეტზე').
    - Converts numbers (up to hundreds of billions) to full Georgian words.
    - Expands currency and percentage symbols.
    """
    # 1. Expand Latin tech keywords to Georgian phonetic equivalents
    for pattern, replacement in _LATIN_TO_GEORGIAN_PHONETICS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Convert hyphenated number-suffix combinations (e.g. 112-ზე, 5-ში, 25-ს)
    def replace_num_suffix(m):
        num = int(m.group(1))
        suffix = m.group(2)
        word = number_to_georgian(num)
        if word.endswith("ი") and suffix in ["ზე", "ში", "ს", "ით", "ად", "დან", "მდე"]:
            word = word[:-1]
        return word + suffix

    text = re.sub(r"\b(\d+)-([ა-ჰ]+)\b", replace_num_suffix, text)

    # 3. Convert standalone numbers
    def replace_num(m):
        num = int(m.group(1))
        return number_to_georgian(num)

    text = re.sub(r"\b(\d+)\b", replace_num, text)
    return text


class EdgeTTSService(TTSService):
    """
    Native Georgian Neural TTS using Microsoft Edge Speech with persistent
    warm WebSocket reuse and low-latency incremental MP3-to-PCM streaming.
    Voices:
      - 'ka-GE-EkaNeural' (Female, native Georgian)
      - 'ka-GE-GiorgiNeural' (Male, native Georgian)
    Completely native Georgian accent, 100% free, zero foreign accent.
    """

    # Refresh warm WebSocket proactively before Edge TTS idle timeout (~30s)
    _WS_MAX_IDLE_SECS = 22.0
    # MP3 bytes to accumulate before first incremental decode (10 frames = ~240ms @ 48kbps)
    _FIRST_DECODE_BYTES = 1440
    # MP3 bytes step for subsequent incremental decodes (20 frames = ~480ms @ 48kbps)
    _STEP_DECODE_BYTES = 2880
    # Tail safety margin in source samples (1 MP3 frame = 576 samples @ 24kHz)
    _MP3_TAIL_GUARD_SAMPLES = 576

    def __init__(
        self,
        voice: str = "ka-GE-EkaNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        sample_rate: int = 16000,
        **kwargs,
    ):
        settings = TTSSettings(
            voice=voice,
            model="edge-tts-neural",
            language="ka",
        )
        kwargs.setdefault("push_start_frame", True)
        kwargs.setdefault("push_stop_frames", True)
        super().__init__(sample_rate=sample_rate, settings=settings, **kwargs)
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._target_sample_rate = sample_rate
        self._tts_config = TTSConfig(self._voice, self._rate, "+0%", self._pitch, "SentenceBoundary")

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_last_used: float = 0.0
        self._ws_lock = asyncio.Lock()
        self._keepwarm_task: Optional[asyncio.Task] = None
        self._running = False

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._running = True
        if self._keepwarm_task is None or self._keepwarm_task.done():
            self._keepwarm_task = asyncio.create_task(self._keepwarm_loop())

    async def stop(self, frame: EndFrame):
        self._running = False
        await self._close_ws_and_session()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._running = False
        await self._close_ws_and_session()
        await super().cancel(frame)

    async def cleanup(self):
        self._running = False
        await self._close_ws_and_session()
        await super().cleanup()

    async def _close_ws_and_session(self):
        if self._keepwarm_task and not self._keepwarm_task.done():
            self._keepwarm_task.cancel()
            try:
                await self._keepwarm_task
            except (asyncio.CancelledError, Exception):
                pass
        self._keepwarm_task = None

        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
        self._session = None

    async def _keepwarm_loop(self):
        """Pre-connects and keeps an Edge TTS WebSocket warm so synthesis starts in ~0ms."""
        while self._running:
            try:
                if not self._ws_lock.locked():
                    idle = time.monotonic() - self._ws_last_used
                    if self._ws is None or self._ws.closed or idle > self._WS_MAX_IDLE_SECS:
                        async with self._ws_lock:
                            idle = time.monotonic() - self._ws_last_used
                            if self._ws is None or self._ws.closed or idle > self._WS_MAX_IDLE_SECS:
                                await self._connect_warm_ws(force_new=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"EdgeTTS keepwarm note: {e}")
            await asyncio.sleep(3.0)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=8, sock_read=25)
            self._session = aiohttp.ClientSession(trust_env=True, timeout=timeout)
        return self._session

    async def _connect_warm_ws(self, force_new: bool = False) -> aiohttp.ClientWebSocketResponse:
        if not force_new and self._ws is not None and not self._ws.closed:
            idle = time.monotonic() - self._ws_last_used
            if idle <= self._WS_MAX_IDLE_SECS:
                return self._ws

        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

        session = await self._ensure_session()
        url = (
            f"{WSS_URL}&ConnectionId={connect_id()}"
            f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
            f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}"
        )
        try:
            ws = await session.ws_connect(
                url,
                compress=15,
                headers=DRM.headers_with_muid(WSS_HEADERS),
                ssl=_SSL_CTX,
                heartbeat=15.0,
            )
        except aiohttp.ClientResponseError as e:
            if e.status == 403:
                DRM.handle_client_response_error(e)
                url = (
                    f"{WSS_URL}&ConnectionId={connect_id()}"
                    f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
                    f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}"
                )
                ws = await session.ws_connect(
                    url,
                    compress=15,
                    headers=DRM.headers_with_muid(WSS_HEADERS),
                    ssl=_SSL_CTX,
                    heartbeat=15.0,
                )
            else:
                raise

        await ws.send_str(
            f"X-Timestamp:{date_to_string()}\r\n"
            "Content-Type:application/json; charset=utf-8\r\n"
            "Path:speech.config\r\n\r\n"
            '{"context":{"synthesis":{"audio":{"metadataoptions":{'
            '"sentenceBoundaryEnabled":"true","wordBoundaryEnabled":"false"'
            "},"
            '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"'
            "}}}}\r\n"
        )
        self._ws = ws
        self._ws_last_used = time.monotonic()
        return ws

    async def _stream_mp3_chunks(self, escaped_segment: bytes) -> AsyncGenerator[bytes, None]:
        """Streams raw MP3 bytes for a single SSML segment over the warm WebSocket."""
        async with self._ws_lock:
            for attempt in range(2):
                ws = await self._connect_warm_ws(force_new=(attempt > 0))
                ssml = mkssml(self._tts_config, escaped_segment)
                req = ssml_headers_plus_data(connect_id(), date_to_string(), ssml)
                audio_received = False
                turn_ended = False
                try:
                    await ws.send_str(req)
                    async for received in ws:
                        self._ws_last_used = time.monotonic()
                        if received.type == aiohttp.WSMsgType.BINARY:
                            if len(received.data) < 2:
                                continue
                            header_length = int.from_bytes(received.data[:2], "big")
                            if header_length > len(received.data):
                                continue
                            parameters, data = get_headers_and_data(received.data, header_length)
                            if parameters.get(b"Path") == b"audio" and data:
                                audio_received = True
                                yield data
                        elif received.type == aiohttp.WSMsgType.TEXT:
                            encoded = received.data.encode("utf-8")
                            sep = encoded.find(b"\r\n\r\n")
                            if sep != -1:
                                parameters, _ = get_headers_and_data(encoded, sep)
                                if parameters.get(b"Path") == b"turn.end":
                                    turn_ended = True
                                    break
                        elif received.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            self._ws = None
                            break
                    if audio_received:
                        return
                except Exception as e:
                    self._ws = None
                    if attempt == 1 or audio_received:
                        raise e
                    logger.debug(f"EdgeTTS warm socket retry after: {e}")
                finally:
                    # Abandoned mid-turn (barge-in cancellation or generator closed): the socket still
                    # holds unread audio for this request and would replay it on the next reply.
                    if not turn_ended and self._ws is ws:
                        self._ws = None
                        self._discard_ws(ws)

    @staticmethod
    def _discard_ws(ws: aiohttp.ClientWebSocketResponse):
        async def _close():
            try:
                await ws.close()
            except Exception:
                pass

        try:
            asyncio.get_running_loop().create_task(_close())
        except RuntimeError:
            pass

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if not text or not text.strip():
            return

        # Normalize text to 100% pure Georgian phonetics before synthesis
        clean_text = normalize_georgian_text(text.strip())
        segments = list(
            split_text_by_byte_length(
                escape(remove_incompatible_characters(clean_text)),
                4096,
            )
        )
        if not segments:
            return

        try:
            await self.start_ttfb_metrics()
            t0 = time.monotonic()
            ttfb_stopped = False
            chunk_size = int(self._target_sample_rate * 2 * 0.02)  # 20ms PCM16 mono

            for seg in segments:
                mp3_buffer = bytearray()
                emitted_source_samples = 0
                leading_silence_trimmed = False
                next_decode_threshold = self._FIRST_DECODE_BYTES
                resampler: Optional[soxr.ResampleStream] = None
                pcm_Carry = bytearray()

                # aclosing() releases the socket lock as soon as this reply is interrupted
                async with aclosing(self._stream_mp3_chunks(seg)) as mp3_stream:
                    async for mp3_chunk in mp3_stream:
                        mp3_buffer.extend(mp3_chunk)
                        if len(mp3_buffer) >= next_decode_threshold:
                            next_decode_threshold = len(mp3_buffer) + self._STEP_DECODE_BYTES
                            try:
                                audio_data, source_sr = sf.read(io.BytesIO(mp3_buffer), dtype="int16")
                            except Exception:
                                continue

                            safe_end = len(audio_data) - self._MP3_TAIL_GUARD_SAMPLES
                            if not leading_silence_trimmed and safe_end > 0:
                                active_idx = np.flatnonzero(np.abs(audio_data[:safe_end]) > 80)
                                if len(active_idx) > 0:
                                    lead_guard = int(source_sr * 0.015)  # keep 15ms soft lead-in
                                    emitted_source_samples = max(0, int(active_idx[0]) - lead_guard)
                                    leading_silence_trimmed = True
                                else:
                                    # Still pure leading silence; wait for next chunk
                                    emitted_source_samples = safe_end
                                    continue

                            if safe_end > emitted_source_samples:
                                new_samples = audio_data[emitted_source_samples:safe_end]
                                emitted_source_samples = safe_end
                                if source_sr != self._target_sample_rate:
                                    if resampler is None:
                                        resampler = soxr.ResampleStream(
                                            source_sr, self._target_sample_rate, 1, dtype="int16"
                                        )
                                    resampled = resampler.resample_chunk(new_samples, last=False)
                                else:
                                    resampled = new_samples

                                pcm_Carry.extend(resampled.tobytes())
                                while len(pcm_Carry) >= chunk_size:
                                    out_chunk = bytes(pcm_Carry[:chunk_size])
                                    del pcm_Carry[:chunk_size]
                                    if not ttfb_stopped:
                                        await self.stop_ttfb_metrics()
                                        ttfb_stopped = True
                                        logger.debug(
                                            f"EdgeTTS first audio in {time.monotonic() - t0:.2f}s for: {text[:35]}..."
                                        )
                                    yield TTSAudioRawFrame(
                                        audio=out_chunk,
                                        sample_rate=self._target_sample_rate,
                                        num_channels=1,
                                        context_id=context_id,
                                    )

                if not mp3_buffer:
                    continue

                # Decode remaining samples at end of segment and trim excess trailing silence
                audio_data, source_sr = sf.read(io.BytesIO(mp3_buffer), dtype="int16")
                if not leading_silence_trimmed and len(audio_data) > 0:
                    active_idx = np.flatnonzero(np.abs(audio_data) > 80)
                    if len(active_idx) > 0:
                        lead_guard = int(source_sr * 0.015)
                        emitted_source_samples = max(0, int(active_idx[0]) - lead_guard)
                    leading_silence_trimmed = True

                active_tail = np.flatnonzero(np.abs(audio_data) > 80)
                if len(active_tail) > 0:
                    tail_guard = int(source_sr * 0.08)  # keep 80ms natural tail breath
                    trimmed_end = min(len(audio_data), int(active_tail[-1]) + tail_guard)
                else:
                    trimmed_end = len(audio_data)

                if trimmed_end > emitted_source_samples:
                    new_samples = audio_data[emitted_source_samples:trimmed_end]
                    if source_sr != self._target_sample_rate:
                        if resampler is None:
                            resampler = soxr.ResampleStream(
                                source_sr, self._target_sample_rate, 1, dtype="int16"
                            )
                        resampled = resampler.resample_chunk(new_samples, last=True)
                    else:
                        resampled = new_samples
                    pcm_Carry.extend(resampled.tobytes())
                elif resampler is not None:
                    resampled = resampler.resample_chunk(
                        audio_data[0:0], last=True
                    )
                    pcm_Carry.extend(resampled.tobytes())

                if not ttfb_stopped and pcm_Carry:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True
                    logger.debug(
                        f"EdgeTTS first audio in {time.monotonic() - t0:.2f}s for: {text[:35]}..."
                    )

                for i in range(0, len(pcm_Carry), chunk_size):
                    out_chunk = bytes(pcm_Carry[i : i + chunk_size])
                    yield TTSAudioRawFrame(
                        audio=out_chunk,
                        sample_rate=self._target_sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )

            logger.debug(f"EdgeTTS finished segment in {time.monotonic() - t0:.2f}s for: {text[:35]}...")

        except Exception as e:
            logger.error(f"EdgeTTS error: {e}")
            yield ErrorFrame(error=f"EdgeTTS error: {e}")
