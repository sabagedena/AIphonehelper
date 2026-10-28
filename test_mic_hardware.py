"""Interactive mic visualizer that auto-detects or tests any microphone."""
import sys
import pyaudio
import math
import struct
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

p = pyaudio.PyAudio()

print("=" * 65)
print("  MICROPHONE LIVE TEST & LEVEL METER")
print("=" * 65)

# List available input devices
valid_devs = []
for i in range(p.get_device_count()):
    d = p.get_device_info_by_index(i)
    name = d.get("name", "")
    if d.get("maxInputChannels", 0) > 0 and "Stereo Mix" not in name:
        valid_devs.append((i, name, int(d.get("defaultSampleRate", 16000))))

print("Available input devices:")
for idx, name, srate in valid_devs:
    print(f"  [{idx}] {name} ({srate}Hz)")
print("-" * 65)

if len(sys.argv) > 1:
    target_idx = int(sys.argv[1])
else:
    # Pick a non-Stereo-Mix device (prefer default if valid, else first valid)
    target_idx = None
    try:
        def_info = p.get_default_input_device_info()
        def_name = def_info.get("name", "")
        if "Stereo Mix" not in def_name and def_info.get("maxInputChannels", 0) > 0:
            target_idx = def_info["index"]
    except Exception:
        pass

    if target_idx is None:
        target_idx = valid_devs[0][0] if valid_devs else 0

dev_info = p.get_device_info_by_index(target_idx)
dev_name = dev_info["name"]
rate = int(dev_info.get("defaultSampleRate", 16000))

# Prefer 16000 if supported, or native rate
rate_to_use = 16000
try:
    s = p.open(format=FORMAT, channels=1, rate=16000, input=True, input_device_index=target_idx)
    s.close()
except Exception:
    rate_to_use = rate

print(f"Selected Device: [{target_idx}] {dev_name} ({rate_to_use}Hz)")
print("Speak into your microphone! Press Ctrl+C to stop.\n")

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=rate_to_use,
    input=True,
    input_device_index=target_idx,
    frames_per_buffer=CHUNK,
)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        count = len(data) // 2
        shorts = struct.unpack(f"{count}h", data)
        sum_sq = sum(s * s for s in shorts)
        rms = math.sqrt(sum_sq / max(1, count)) / 32768.0
        
        # Linear RMS for display
        raw_rms = int(rms * 32768)
        # Decibel calculation: -60 dB (quiet whisper) to 0 dB (clipping)
        db = 20 * math.log10(rms) if rms > 1e-5 else -90.0
        # Map -50 dB ... -10 dB to 0% ... 100%
        pct = int(max(0, min(100, (db + 50) * (100 / 40))))
        bars = int((pct / 100) * 30)
        bar_str = "=" * bars + "-" * (30 - bars)
        print(f"\rLevel: [{bar_str}] {pct:3d}%  (Raw: {raw_rms:5d} | {db:5.1f} dBFS)", end="", flush=True)
        time.sleep(0.04)
except KeyboardInterrupt:
    print("\n\nTest stopped.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

