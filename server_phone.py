import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import audioop
import ctypes
import hashlib
import math
import os
import random
import re
import socket
import struct
import sys
import threading
import time
import uuid
from typing import Callable, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
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
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from bot import build_llm, build_tts
from prompts import INITIAL_GREETING
from speech_gate import (
    PhantomSpeechFilter,
    SpeechOnlyAudioGate,
    build_user_params,
    build_vad_processor,
)

load_dotenv()

PHONE_SAMPLE_RATE = 8000
FRAME_SAMPLES = 160  # 20ms @ 8000Hz
FRAME_BYTES_PCM16 = FRAME_SAMPLES * 2  # 320 bytes (16-bit signed mono)
SILENCE_PCM16 = b"\x00" * FRAME_BYTES_PCM16

_GREETING_PCM16_CACHE: Optional[bytes] = None


def warm_greeting_cache() -> Optional[bytes]:
    """Pre-synthesizes INITIAL_GREETING at 8kHz PCM16 so incoming phone calls hear the greeting in ~0.3s."""
    global _GREETING_PCM16_CACHE
    if _GREETING_PCM16_CACHE is not None:
        return _GREETING_PCM16_CACHE
    try:
        import io
        import edge_tts
        import numpy as np
        import soundfile as sf
        import soxr
        from edge_tts_service import normalize_georgian_text

        voice = os.getenv("EDGE_TTS_VOICE", "ka-GE-EkaNeural").strip()
        rate = os.getenv("EDGE_TTS_RATE", "+12%").strip()
        pitch = os.getenv("EDGE_TTS_PITCH", "+0Hz").strip()
        clean_text = normalize_georgian_text(INITIAL_GREETING.strip())

        async def _synth() -> bytes:
            comm = edge_tts.Communicate(clean_text, voice=voice, rate=rate, pitch=pitch)
            mp3_buf = bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3_buf.extend(chunk["data"])
            audio_np, sr = sf.read(io.BytesIO(mp3_buf), dtype="int16")
            active = np.flatnonzero(np.abs(audio_np) > 80)
            if len(active) > 0:
                start_idx = max(0, int(active[0]) - int(sr * 0.015))
                end_idx = min(len(audio_np), int(active[-1]) + int(sr * 0.08))
                audio_np = audio_np[start_idx:end_idx]
            if sr != PHONE_SAMPLE_RATE:
                audio_np = soxr.resample(audio_np, sr, PHONE_SAMPLE_RATE, quality="HQ").astype(np.int16)
            return audio_np.tobytes()

        _GREETING_PCM16_CACHE = asyncio.run(_synth())
    except Exception as e:
        logger.debug(f"Could not pre-cache greeting: {e}")
    return _GREETING_PCM16_CACHE

# Enable UTF-8 console output on Windows so Georgian script & emojis never raise UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Enable 1ms kernel timer resolution on Windows for jitter-free 20ms RTP audio
if sys.platform == "win32":
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def get_public_ip() -> Optional[str]:
    """Fetches public IP for display/diagnostics."""
    for service in ["https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com"]:
        try:
            r = requests.get(service, timeout=3)
            if r.status_code == 200 and r.text.strip():
                return r.text.strip()
        except Exception:
            continue
    return None


def get_local_ip(target_host: str = "89.150.1.11", target_port: int = 5060) -> str:
    """Finds the local interface IP used to reach the SIP server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((target_host, target_port))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


def sip_get_header(msg: str, name: str, compact: str = "") -> str:
    """Case-insensitive SIP single header lookup."""
    targets = [name.lower() + ":"]
    if compact:
        targets.append(compact.lower() + ":")
    for line in msg.split("\r\n"):
        if not line:
            break
        low = line.lower()
        for t in targets:
            if low.startswith(t):
                return line.split(":", 1)[1].strip()
    return ""


def sip_get_all_headers(msg: str, name: str, compact: str = "") -> list[str]:
    """Returns full header lines (normalized to standard header name) in order."""
    targets = [name.lower() + ":"]
    if compact:
        targets.append(compact.lower() + ":")
    out = []
    for line in msg.split("\r\n"):
        if not line:
            break
        low = line.lower()
        for t in targets:
            if low.startswith(t):
                val = line.split(":", 1)[1].strip()
                out.append(f"{name}: {val}")
                break
    return out


class RTPSession:
    """Bidirectional 20ms G.711 (PCMU/PCMA) RTP audio engine using 16-bit signed PCM."""

    def __init__(
        self,
        local_ip: str,
        local_port: int,
        remote_addr: Tuple[str, int],
        codec_pt: int = 0,
        barge_in_rms_threshold: int = 1400,
    ):
        self.local_ip = local_ip
        self.local_port = local_port
        self.remote_addr = remote_addr
        self.codec_pt = codec_pt  # 0 = PCMU (mu-law), 8 = PCMA (A-law)
        self.barge_in_rms_threshold = barge_in_rms_threshold

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
        except Exception:
            pass
        self.sock.bind((self.local_ip, self.local_port))
        self.sock.setblocking(False)

        self.running = False
        self._out_lock = threading.Lock()
        self._out_pcm = bytearray()
        self._in_pcm = bytearray()

        self.seq = random.randint(100, 10000)
        self.timestamp = random.randint(1000, 100000)
        self.ssrc = random.randint(0x10000000, 0x7FFFFFFF)

        self.last_rx_time = time.monotonic()
        self.last_tx_speech_time = 0.0
        self.rx_packets = 0
        self.tx_packets = 0
        self._audio_callback: Optional[Callable[[bytes], None]] = None
        self._thread: Optional[threading.Thread] = None

    def set_audio_callback(self, cb: Callable[[bytes], None]):
        self._audio_callback = cb

    def start(self):
        if self.running:
            return
        self.running = True
        self.last_rx_time = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name=f"RTP-{self.local_port}", daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass

    def enqueue_audio(self, pcm16_bytes: bytes):
        if not pcm16_bytes:
            return
        with self._out_lock:
            self._out_pcm.extend(pcm16_bytes)

    def clear_output(self):
        with self._out_lock:
            self._out_pcm.clear()

    def buffered_output_secs(self) -> float:
        with self._out_lock:
            return len(self._out_pcm) / (PHONE_SAMPLE_RATE * 2)

    def _encode_g711(self, pcm16: bytes, pt: int) -> bytes:
        if pt == 8:
            return audioop.lin2alaw(pcm16, 2)
        return audioop.lin2ulaw(pcm16, 2)

    def _decode_g711(self, payload: bytes, pt: int) -> Optional[bytes]:
        if pt == 0:
            return audioop.ulaw2lin(payload, 2)
        if pt == 8:
            return audioop.alaw2lin(payload, 2)
        return None

    def _loop(self):
        next_tick = time.perf_counter()
        marker_first = True

        while self.running:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(max(0.001, next_tick - now))
                continue

            next_tick += 0.020
            if now - next_tick > 0.100:
                # Resync clock if thread was stalled
                next_tick = now + 0.020

            # 1. Outbound RTP packet (20ms = 320 bytes PCM16 -> 160 bytes G.711)
            is_speech_chunk = False
            with self._out_lock:
                if len(self._out_pcm) >= FRAME_BYTES_PCM16:
                    out_pcm = bytes(self._out_pcm[:FRAME_BYTES_PCM16])
                    del self._out_pcm[:FRAME_BYTES_PCM16]
                    is_speech_chunk = True
                elif len(self._out_pcm) > 0:
                    out_pcm = bytes(self._out_pcm) + b"\x00" * (FRAME_BYTES_PCM16 - len(self._out_pcm))
                    self._out_pcm.clear()
                    is_speech_chunk = True
                else:
                    out_pcm = SILENCE_PCM16

            if is_speech_chunk:
                self.last_tx_speech_time = time.monotonic()

            try:
                payload = self._encode_g711(out_pcm, self.codec_pt)
                pt_byte = (self.codec_pt & 0x7F) | (0x80 if marker_first else 0x00)
                marker_first = False
                header = struct.pack(
                    "!BBHII",
                    0x80,
                    pt_byte,
                    self.seq & 0xFFFF,
                    self.timestamp & 0xFFFFFFFF,
                    self.ssrc,
                )
                self.sock.sendto(header + payload, self.remote_addr)
                self.seq = (self.seq + 1) & 0xFFFF
                self.timestamp = (self.timestamp + FRAME_SAMPLES) & 0xFFFFFFFF
                self.tx_packets += 1
            except OSError:
                if not self.running:
                    break

            # 2. Inbound RTP packets from phone line
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(2048)
                except BlockingIOError:
                    break
                except OSError:
                    break

                if len(data) < 12:
                    continue
                version = (data[0] >> 6) & 0x03
                if version != 2:
                    continue

                cc = data[0] & 0x0F
                ext = (data[0] >> 4) & 0x01
                pt = data[1] & 0x7F
                hdr_len = 12 + cc * 4
                if ext and len(data) >= hdr_len + 4:
                    ext_len = struct.unpack("!H", data[hdr_len + 2 : hdr_len + 4])[0] * 4
                    hdr_len += 4 + ext_len

                if len(data) <= hdr_len:
                    continue

                rtp_payload = data[hdr_len:]
                self.last_rx_time = time.monotonic()
                self.rx_packets += 1

                decoded = self._decode_g711(rtp_payload, pt)
                if decoded:
                    self._in_pcm.extend(decoded)
                    # Cap input jitter buffer at 100ms (5 frames) to prevent latency buildup
                    max_buf = FRAME_BYTES_PCM16 * 5
                    if len(self._in_pcm) > max_buf:
                        del self._in_pcm[: len(self._in_pcm) - max_buf]

            # 3. Emit exact 20ms frame to Pipecat pipeline
            if len(self._in_pcm) >= FRAME_BYTES_PCM16:
                in_chunk = bytes(self._in_pcm[:FRAME_BYTES_PCM16])
                del self._in_pcm[:FRAME_BYTES_PCM16]
            else:
                in_chunk = SILENCE_PCM16

            # Echo guard while AI is actively speaking (+ 180ms hybrid echo decay tail):
            # Filters out quiet PSTN/mobile hybrid echo so the bot never interrupts itself,
            # while still letting loud human barge-in speech pass through.
            if (time.monotonic() - self.last_tx_speech_time) < 0.18 and in_chunk != SILENCE_PCM16:
                rms = audioop.rms(in_chunk, 2)
                if rms < self.barge_in_rms_threshold:
                    in_chunk = SILENCE_PCM16

            if self._audio_callback is not None:
                try:
                    self._audio_callback(in_chunk)
                except Exception:
                    pass


class PhoneCall:
    """Represents an active SIP telephone call and its RTP session."""

    def __init__(
        self,
        sip_server: "MyPhoneSIPServer",
        call_id: str,
        caller_number: str,
        local_tag: str,
        rtp: RTPSession,
        invite_msg: str,
        ok_response: str,
    ):
        self.sip_server = sip_server
        self.call_id = call_id
        self.caller_number = caller_number
        self.local_tag = local_tag
        self.rtp = rtp
        self.invite_msg = invite_msg
        self.ok_response = ok_response
        self.acked = False
        self.active = True
        self.started_at = time.monotonic()

    def hangup(self):
        if not self.active:
            return
        self.active = False
        self.rtp.stop()
        self.sip_server.send_bye(self)


class VoIPAudioInput(BaseInputTransport):
    """Pipecat input transport fed by the 20ms RTP receive clock."""

    def __init__(self, call: PhoneCall, params: TransportParams, **kwargs):
        super().__init__(params=params, **kwargs)
        self._call = call

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        loop = self.get_event_loop()

        def _on_rtp_pcm(pcm16: bytes):
            if not self._call.active:
                return
            audio_frame = InputAudioRawFrame(
                audio=pcm16,
                sample_rate=PHONE_SAMPLE_RATE,
                num_channels=1,
            )
            asyncio.run_coroutine_threadsafe(self.push_audio_frame(audio_frame), loop)

        self._call.rtp.set_audio_callback(_on_rtp_pcm)

    async def stop(self, frame: EndFrame):
        self._call.rtp.set_audio_callback(None)
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._call.rtp.set_audio_callback(None)
        await super().cancel(frame)


class VoIPAudioOutput(BaseOutputTransport):
    """Pipecat output transport feeding 16-bit PCM into the 20ms RTP transmitter."""

    def __init__(self, call: PhoneCall, params: TransportParams, **kwargs):
        super().__init__(params=params, **kwargs)
        self._call = call

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            self._call.rtp.clear_output()
        await super().process_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if not self._call.active:
            return False
        self._call.rtp.enqueue_audio(frame.audio)
        # Pace against actual RTP playout buffer (~60ms cushion) so BotStarted/StoppedSpeaking
        # events match real telephone playout and barge-in clears immediately.
        while self._call.active and self._call.rtp.buffered_output_secs() > 0.06:
            await asyncio.sleep(0.01)
        return True


class VoIPTransport(BaseTransport):
    def __init__(self, call: PhoneCall, params: TransportParams):
        super().__init__()
        self._input = VoIPAudioInput(call, params)
        self._output = VoIPAudioOutput(call, params)

    def input(self) -> BaseInputTransport:
        return self._input

    def output(self) -> BaseOutputTransport:
        return self._output


class UserDisplay(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            print("\n🤖 [დამხმარე საუბრობს...]")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            print("👂 [გისმენთ... შეგიძლიათ ისაუბროთ]")
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            print("🎤 [აბონენტი საუბრობს...]")
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                print(f"\n🗣️  აბონენტი: {text}")
        await self.push_frame(frame, direction)


class BotDisplay(FrameProcessor):
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


async def handle_call_session(call: PhoneCall):
    print("\n" + "=" * 65)
    print("📞 შემომავალი ზარი მიღებულია! (Incoming Call Answered)")
    print(f"   დამრეკი (Caller): {call.caller_number}")
    print(f"   RTP პორტი:        {call.rtp.local_ip}:{call.rtp.local_port} <-> {call.rtp.remote_addr[0]}:{call.rtp.remote_addr[1]}")
    print(f"   კოდეკი (Codec):   {'PCMU (G.711 mu-law)' if call.rtp.codec_pt == 0 else 'PCMA (G.711 A-law)'}")
    print("=" * 65)

    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=PHONE_SAMPLE_RATE,
        audio_out_sample_rate=PHONE_SAMPLE_RATE,
    )
    transport = VoIPTransport(call, transport_params)

    vad = build_vad_processor(stop_secs=0.2)
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
            language="ka",
            smart_format=True,
        ),
    )
    phantom_filter = PhantomSpeechFilter()
    llm = build_llm()
    tts = build_tts(sample_rate=PHONE_SAMPLE_RATE)

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_params(user_speech_timeout=0.35),
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
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=PHONE_SAMPLE_RATE,
            audio_out_sample_rate=PHONE_SAMPLE_RATE,
            enable_metrics=False,
        ),
    )

    runner = PipelineRunner(handle_sigint=False)

    async def greet():
        await asyncio.sleep(0.3)
        if not call.active:
            return
        print("\n🤖 დამხმარე: " + INITIAL_GREETING)
        if _GREETING_PCM16_CACHE:
            print("\n🤖 [დამხმარე საუბრობს...]")
            call.rtp.enqueue_audio(_GREETING_PCM16_CACHE)
            while call.active and call.rtp.buffered_output_secs() > 0.05:
                await asyncio.sleep(0.05)
            if call.active:
                print("👂 [გისმენთ... შეგიძლიათ ისაუბროთ]")
        else:
            await task.queue_frames([TTSSpeakFrame(INITIAL_GREETING)])

    greet_task = asyncio.create_task(greet())

    async def monitor_call():
        while call.active:
            await asyncio.sleep(0.25)
            # Timeout if RTP stream stops completely for >35s
            if time.monotonic() - call.rtp.last_rx_time > 35.0:
                logger.info("RTP inactivity timeout reached; ending call.")
                call.active = False
                break
        print("\n📴 ზარი დასრულდა. (Call hung up)")
        call.rtp.stop()
        await task.cancel()

    monitor_task = asyncio.create_task(monitor_call())

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error during phone call session: {e}")
    finally:
        call.active = False
        call.rtp.stop()
        for t in (greet_task, monitor_task):
            if not t.done():
                t.cancel()


class MyPhoneSIPServer:
    """RFC 3261 SIP User Agent tailored for MyPhone.ge (Asterisk PBX) with NAT & OPTIONS support."""

    def __init__(
        self,
        server: str,
        port: int,
        username: str,
        password: str,
        local_ip: str,
        sip_port: int = 5060,
        rtp_port_range: Tuple[int, int] = (10000, 20000),
        on_call: Optional[Callable[[PhoneCall], None]] = None,
    ):
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.local_ip = local_ip
        self.sip_port = sip_port
        self.rtp_port_low, self.rtp_port_high = rtp_port_range
        self.on_call = on_call

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((self.local_ip, self.sip_port))
        except OSError:
            # If 5060 is held by another local app, pick a free port >= 5064
            for candidate in range(5064, 5100, 2):
                try:
                    self.sock.bind((self.local_ip, candidate))
                    self.sip_port = candidate
                    break
                except OSError:
                    continue
        self.sock.settimeout(0.5)

        self.running = False
        self.registered = threading.Event()
        self.reg_call_id = f"{uuid.uuid4().hex[:24]}@{self.local_ip}"
        self.reg_tag = uuid.uuid4().hex[:8]
        self.reg_cseq = 0
        self.reg_expires = 60
        self.last_reg_sent = 0.0
        self.last_reg_ok = 0.0
        self.last_keepalive = 0.0

        self.calls: Dict[str, PhoneCall] = {}
        self._recv_thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.send_register()
        self._recv_thread = threading.Thread(target=self._loop, name="SIP-MainLoop", daemon=True)
        self._recv_thread.start()

    def stop(self):
        if not self.running:
            return
        for c in list(self.calls.values()):
            try:
                c.hangup()
            except Exception:
                pass
        try:
            self.send_register(expires=0)
            time.sleep(0.15)
        except Exception:
            pass
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass

    def _allocate_rtp_port(self) -> int:
        used = {c.rtp.local_port for c in self.calls.values() if c.active}
        for _ in range(100):
            port = random.randrange(self.rtp_port_low, self.rtp_port_high, 2)
            if port not in used:
                return port
        return 12000

    def send_register(self, expires: Optional[int] = None, auth_header: str = ""):
        exp = self.reg_expires if expires is None else expires
        self.reg_cseq += 1
        branch = f"z9hG4bK{uuid.uuid4().hex[:24]}"
        lines = [
            f"REGISTER sip:{self.server} SIP/2.0",
            f"Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch={branch};rport",
            f'From: "{self.username}" <sip:{self.username}@{self.server}>;tag={self.reg_tag}',
            f'To: "{self.username}" <sip:{self.username}@{self.server}>',
            f"Call-ID: {self.reg_call_id}",
            f"CSeq: {self.reg_cseq} REGISTER",
            f"Contact: <sip:{self.username}@{self.local_ip}:{self.sip_port};transport=UDP>",
            f"Expires: {exp}",
            "Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO, PUBLISH, MESSAGE",
            "Max-Forwards: 70",
            "User-Agent: GeorgianVoiceAI/2.0",
        ]
        if auth_header:
            lines.append(auth_header)
        lines.append("Content-Length: 0")
        lines.append("\r\n")
        msg = "\r\n".join(lines)
        self.sock.sendto(msg.encode("utf-8"), (self.server, self.port))
        self.last_reg_sent = time.monotonic()

    def _handle_register_401(self, msg: str):
        www_auth = sip_get_header(msg, "WWW-Authenticate") or sip_get_header(msg, "Proxy-Authenticate")
        if not www_auth:
            return
        realm_m = re.search(r'realm="([^"]+)"', www_auth)
        nonce_m = re.search(r'nonce="([^"]+)"', www_auth)
        if not (realm_m and nonce_m):
            return
        realm = realm_m.group(1)
        nonce = nonce_m.group(1)
        uri = f"sip:{self.server};transport=UDP"
        ha1 = hashlib.md5(f"{self.username}:{realm}:{self.password}".encode("utf-8")).hexdigest()
        ha2 = hashlib.md5(f"REGISTER:{uri}".encode("utf-8")).hexdigest()
        digest = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode("utf-8")).hexdigest()

        hdr_name = "Authorization" if "WWW-Authenticate" in msg else "Proxy-Authorization"
        auth_line = (
            f'{hdr_name}: Digest username="{self.username}",realm="{realm}",'
            f'nonce="{nonce}",uri="{uri}",response="{digest}",algorithm=MD5'
        )
        self.send_register(auth_header=auth_line)

    def _reply_simple_ok(self, req: str):
        vias = "\r\n".join(sip_get_all_headers(req, "Via", "v"))
        from_h = sip_get_header(req, "From", "f")
        to_h = sip_get_header(req, "To", "t")
        if ";tag=" not in to_h:
            to_h = f"{to_h};tag={uuid.uuid4().hex[:8]}"
        call_id = sip_get_header(req, "Call-ID", "i")
        cseq = sip_get_header(req, "CSeq")
        resp = (
            "SIP/2.0 200 OK\r\n"
            f"{vias}\r\n"
            f"From: {from_h}\r\n"
            f"To: {to_h}\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: {cseq}\r\n"
            f"Contact: <sip:{self.username}@{self.local_ip}:{self.sip_port};transport=UDP>\r\n"
            "Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO, PUBLISH, MESSAGE\r\n"
            "Accept: application/sdp\r\n"
            "User-Agent: GeorgianVoiceAI/2.0\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        self.sock.sendto(resp.encode("utf-8"), (self.server, self.port))

    def _parse_Caller(self, req: str) -> str:
        rpid = sip_get_header(req, "Remote-Party-ID")
        if rpid:
            m = re.search(r'"([^"]+)"|\bsip:([^@>;]+)', rpid)
            if m:
                return m.group(1) or m.group(2)
        from_h = sip_get_header(req, "From", "f")
        m = re.search(r"sip:([^@>;]+)", from_h)
        return m.group(1) if m else from_h

    def _parse_sdp(self, req: str) -> Tuple[str, int, int]:
        """Extracts (remote_rtp_ip, remote_rtp_port, preferred_codec_pt) from INVITE SDP."""
        remote_ip = self.server
        remote_port = 10000
        codec_pt = 0  # Default PCMU

        parts = req.split("\r\n\r\n", 1)
        if len(parts) < 2:
            return remote_ip, remote_port, codec_pt

        sdp = parts[1]
        for line in sdp.split("\r\n"):
            if line.startswith("c=IN IP4 "):
                remote_ip = line.split("c=IN IP4 ", 1)[1].split("/")[0].strip()
            elif line.startswith("m=audio "):
                tokens = line.split()
                if len(tokens) >= 4:
                    remote_port = int(tokens[1].split("/")[0])
                    offered_pts = [int(x) for x in tokens[3:] if x.isdigit()]
                    for pt in offered_pts:
                        if pt in (0, 8):
                            codec_pt = pt
                            break
        return remote_ip, remote_port, codec_pt

    def _build_invite_ok(self, req: str, local_tag: str, local_rtp_port: int, codec_pt: int) -> str:
        vias = "\r\n".join(sip_get_all_headers(req, "Via", "v"))
        record_routes = "\r\n".join(sip_get_all_headers(req, "Record-Route"))
        from_h = sip_get_header(req, "From", "f")
        to_h = sip_get_header(req, "To", "t")
        if ";tag=" not in to_h:
            to_h = f"{to_h};tag={local_tag}"
        call_id = sip_get_header(req, "Call-ID", "i")
        cseq = sip_get_header(req, "CSeq")

        codec_name = "PCMA" if codec_pt == 8 else "PCMU"
        sess_id = random.randint(100000, 999999)
        sdp = (
            "v=0\r\n"
            f"o={self.username} {sess_id} {sess_id} IN IP4 {self.local_ip}\r\n"
            "s=GeorgianVoiceAI\r\n"
            f"c=IN IP4 {self.local_ip}\r\n"
            "t=0 0\r\n"
            f"m=audio {local_rtp_port} RTP/AVP {codec_pt} 101\r\n"
            f"a=rtpmap:{codec_pt} {codec_name}/8000\r\n"
            "a=rtpmap:101 telephone-event/8000\r\n"
            "a=fmtp:101 0-16\r\n"
            "a=ptime:20\r\n"
            "a=sendrecv\r\n"
        )

        headers = [
            "SIP/2.0 200 OK",
            vias,
        ]
        if record_routes:
            headers.append(record_routes)
        headers.extend(
            [
                f"From: {from_h}",
                f"To: {to_h}",
                f"Call-ID: {call_id}",
                f"CSeq: {cseq}",
                f"Contact: <sip:{self.username}@{self.local_ip}:{self.sip_port};transport=UDP>",
                "Allow: INVITE, ACK, CANCEL, OPTIONS, BYE, REFER, SUBSCRIBE, NOTIFY, INFO, PUBLISH, MESSAGE",
                "User-Agent: GeorgianVoiceAI/2.0",
                "Content-Type: application/sdp",
                f"Content-Length: {len(sdp)}",
                "",
                sdp,
            ]
        )
        return "\r\n".join(headers)

    def _handle_invite(self, req: str):
        call_id = sip_get_header(req, "Call-ID", "i")
        if not call_id:
            return

        remote_ip, remote_port, codec_pt = self._parse_sdp(req)

        # If this Call-ID already exists (retransmitted INVITE or session-timer re-INVITE):
        if call_id in self.calls and self.calls[call_id].active:
            call = self.calls[call_id]
            call.rtp.remote_addr = (remote_ip, remote_port)
            call.ok_response = self._build_invite_ok(req, call.local_tag, call.rtp.local_port, call.rtp.codec_pt)
            self.sock.sendto(call.ok_response.encode("utf-8"), (self.server, self.port))
            return

        vias = "\r\n".join(sip_get_all_headers(req, "Via", "v"))
        from_h = sip_get_header(req, "From", "f")
        to_h = sip_get_header(req, "To", "t")
        cseq = sip_get_header(req, "CSeq")
        local_tag = uuid.uuid4().hex[:8]

        # 1. Send 100 Trying & 180 Ringing immediately
        trying = (
            "SIP/2.0 100 Trying\r\n"
            f"{vias}\r\n"
            f"From: {from_h}\r\n"
            f"To: {to_h}\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: {cseq}\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        ringing = (
            "SIP/2.0 180 Ringing\r\n"
            f"{vias}\r\n"
            f"From: {from_h}\r\n"
            f"To: {to_h};tag={local_tag}\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: {cseq}\r\n"
            f"Contact: <sip:{self.username}@{self.local_ip}:{self.sip_port};transport=UDP>\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        self.sock.sendto(trying.encode("utf-8"), (self.server, self.port))
        self.sock.sendto(ringing.encode("utf-8"), (self.server, self.port))

        # 2. Start RTP session & send 200 OK with SDP
        local_rtp_port = self._allocate_rtp_port()
        rtp = RTPSession(
            local_ip=self.local_ip,
            local_port=local_rtp_port,
            remote_addr=(remote_ip, remote_port),
            codec_pt=codec_pt,
        )
        rtp.start()

        ok_resp = self._build_invite_ok(req, local_tag, local_rtp_port, codec_pt)
        self.sock.sendto(ok_resp.encode("utf-8"), (self.server, self.port))

        caller_num = self._parse_Caller(req)
        call = PhoneCall(
            sip_server=self,
            call_id=call_id,
            caller_number=caller_num,
            local_tag=local_tag,
            rtp=rtp,
            invite_msg=req,
            ok_response=ok_resp,
        )
        self.calls[call_id] = call

        if self.on_call:
            self.on_call(call)

    def send_bye(self, call: PhoneCall):
        contact = sip_get_header(call.invite_msg, "Contact", "m")
        m = re.search(r"<(sip:[^>]+)>|(sip:[^\s;>]+)", contact)
        target_uri = (m.group(1) or m.group(2)) if m else f"sip:{self.server}"
        from_h = sip_get_header(call.invite_msg, "From", "f")
        to_h = sip_get_header(call.invite_msg, "To", "t")
        if ";tag=" not in to_h:
            to_h = f"{to_h};tag={call.local_tag}"
        branch = f"z9hG4bK{uuid.uuid4().hex[:24]}"
        bye_msg = (
            f"BYE {target_uri} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.local_ip}:{self.sip_port};branch={branch};rport\r\n"
            f"From: {to_h}\r\n"
            f"To: {from_h}\r\n"
            f"Call-ID: {call.call_id}\r\n"
            "CSeq: 200 BYE\r\n"
            f"Contact: <sip:{self.username}@{self.local_ip}:{self.sip_port};transport=UDP>\r\n"
            "Max-Forwards: 70\r\n"
            "User-Agent: GeorgianVoiceAI/2.0\r\n"
            "Content-Length: 0\r\n\r\n"
        )
        try:
            self.sock.sendto(bye_msg.encode("utf-8"), (self.server, self.port))
        except Exception:
            pass

    def _loop(self):
        while self.running:
            now = time.monotonic()

            # 1. Retry initial REGISTER every 2s until registered; once registered, refresh every 25s
            reg_interval = 25.0 if self.registered.is_set() else 2.0
            if now - self.last_reg_sent >= reg_interval:
                try:
                    self.send_register()
                except Exception as e:
                    logger.debug(f"SIP re-register error: {e}")

            # 2. Retransmit 200 OK for any un-ACKed answered calls (every 0.5s up to 4s)
            for call in list(self.calls.values()):
                if call.active and not call.acked and (now - call.started_at) < 4.0:
                    try:
                        self.sock.sendto(call.ok_response.encode("utf-8"), (self.server, self.port))
                    except Exception:
                        pass

            # 4. Receive incoming SIP packet
            try:
                data, addr = self.sock.recvfrom(8192)
            except socket.timeout:
                continue
            except OSError:
                if not self.running:
                    break
                continue

            if not data or data.strip() == b"":
                continue

            msg = data.decode("utf-8", errors="replace")
            first_line = msg.split("\r\n", 1)[0].strip()

            if first_line.startswith("SIP/2.0 "):
                # SIP Response
                cseq = sip_get_header(msg, "CSeq")
                if "REGISTER" in cseq:
                    if "401 " in first_line or "407 " in first_line:
                        self._handle_register_401(msg)
                    elif "200 " in first_line:
                        exp_hdr = sip_get_header(msg, "Expires")
                        if exp_hdr.isdigit():
                            self.reg_expires = int(exp_hdr)
                        self.last_reg_ok = time.monotonic()
                        self.registered.set()
            else:
                # SIP Request from server
                method = first_line.split(" ", 1)[0].upper()
                if method in ("OPTIONS", "NOTIFY", "INFO", "MESSAGE", "UPDATE"):
                    self._reply_simple_ok(msg)
                elif method == "INVITE":
                    self._handle_invite(msg)
                elif method == "ACK":
                    call_id = sip_get_header(msg, "Call-ID", "i")
                    if call_id in self.calls:
                        self.calls[call_id].acked = True
                elif method in ("BYE", "CANCEL"):
                    self._reply_simple_ok(msg)
                    call_id = sip_get_header(msg, "Call-ID", "i")
                    if call_id in self.calls:
                        call = self.calls[call_id]
                        call.active = False
                        call.rtp.stop()


def on_incoming_call(call: PhoneCall):
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(handle_call_session(call))
        finally:
            loop.close()

    t = threading.Thread(target=_run, name=f"CallSession-{call.call_id[:8]}", daemon=True)
    t.start()


def main():
    server = os.getenv("MYPHONE_SIP_SERVER", "89.150.1.11").strip()
    port = int(os.getenv("MYPHONE_SIP_PORT", "5060"))
    user = os.getenv("MYPHONE_SIP_USERNAME", "1055986e0").strip()
    pwd = os.getenv("MYPHONE_SIP_PASSWORD", "63gkxnhy").strip()
    phone_num = os.getenv("MYPHONE_PHONE_NUMBER", "+995322195820").strip()

    local_ip = get_local_ip(server, port)
    public_ip = get_public_ip() or local_ip

    print("\n" + "=" * 65)
    print("  GEORGIAN AI VOICE HELPER - REAL TELEPHONE SERVER")
    print("=" * 65)
    print(f"  - Provider:   MyPhone.ge (SIP: {server}:{port})")
    print(f"  - Phone line: {phone_num} (or 032 2195820 / 2195820)")
    print(f"  - Account:    {user}")
    print(f"  - Local IP:   {local_ip}")
    print(f"  - Public IP:  {public_ip}")
    print("  - Listening:  Deepgram Nova-3 (Georgian)")
    print(f"  - Brain:      Google Gemini ({os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite')})")
    print(f"  - Voice:      Microsoft Edge Neural ({os.getenv('EDGE_TTS_VOICE', 'ka-GE-EkaNeural')})")
    print("=" * 65)
    print("  🚀 სტატუსი: რეგისტრაცია და ხმის მომზადება... (Registering & warming voice...)")
    warm_greeting_cache()

    sip_server = MyPhoneSIPServer(
        server=server,
        port=port,
        username=user,
        password=pwd,
        local_ip=local_ip,
        sip_port=5060,
        on_call=on_incoming_call,
    )
    sip_server.start()

    if sip_server.registered.wait(timeout=8.0):
        print(f"  ✅ წარმატებით დარეგისტრირდა! (REGISTERED on UDP port {sip_server.sip_port})")
        print("  ✅ Asterisk OPTIONS keepalive & Symmetric RTP NAT active!")
        print(f"\n👉 ახლა აიღეთ მობილური და დარეკეთ ნომერზე: {phone_num} (ან 032 2195820)")
        print("   ხელოვნური ინტელექტი გიპასუხებთ ქართულად!\n")
    else:
        print("  ⚠️  რეგისტრაციის მოლოდინი გრძელდება ფონურ რეჟიმში...")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nსერვერი გაითიშა...")
    finally:
        sip_server.stop()


if __name__ == "__main__":
    main()
