# CLAUDE.md — Georgian AI Voice Helper & Phone Assistant

This file provides comprehensive context, architectural design, execution workflows, and coding conventions for Claude (and other AI agents) working on this repository.

---

## 1. Project Overview & Mission

This project is a **real-time, sub-second latency conversational Voice AI assistant in Georgian (`ka`)**, specifically tailored for elderly speakers in Georgia. It assists with everyday tasks (phone usage, scam protection, bank/utility notifications, healthcare/emergency redirection).

### Key Highlights
- **100% Native Accent**: Uses Microsoft Edge Neural Georgian TTS (`ka-GE-EkaNeural` / `ka-GE-GiorgiNeural`) with zero foreign accent and zero API costs.
- **Real-Time Georgian STT**: Deepgram Nova-3 (`ka`) streaming speech-to-text with custom anti-hallucination gating.
- **Intelligent LLM Engine**: Google Gemini 3.1 Flash Lite / 3 Flash with automated hedge racing across model candidates, or Anthropic Claude (Haiku 4.5).
- **Direct SIP/RTP Georgian Landline**: Native Python SIP 2.0 & RTP stack (`server_phone.py`) registered directly with Georgian telecom **MyPhone.ge** (`89.150.1.11:5060`) answering the Georgian phone number **`+995 32 219 58 20` (`032 219 58 20`)**.
- **Alternative Ingress**: Local microphone mode (PyAudio), browser web UI (FastAPI + WebSockets), and Twilio media stream webhook (`server.py`).

---

## 2. Architecture & Pipeline Dataflow

```
[Audio Input] (RTP / Mic / WebRTC / Twilio)
     │
     ▼
[RealtimeVoiceIsolator & STFT Bandpass] (speech_gate.py)
  • 110Hz - 3450Hz (Phone) / 4500Hz (Mic) STFT bandpass filter
  • Adaptive per-bin Wiener spectral noise reduction
  • Dynamic voice-band RMS & formant concentration gate
     │
     ▼
[Silero VAD] (pipecat.audio.vad.silero)
  • Dynamic speech start / stop detection (0.38s - 0.42s turn turnaround)
     │
     ▼
[SpeechOnlyAudioGate] (speech_gate.py)
  • Blocks non-speech background audio from reaching STT
  • Pre-roll (0.35s) & post-roll trailing silence (0.24s)
     │
     ▼
[Deepgram Nova-3 STT] (`ka` language, smart format)
     │
     ▼
[PhantomSpeechFilter] (speech_gate.py)
  • Drops hallucinated transcripts below MIN_STT_CONFIDENCE (default: 0.55)
     │
     ▼
[LLM Context & Aggregator]
     │
     ▼
[LLM Engine] (bot.py)
  • FastGoogleLLMService (models ranked by their measured time to first text on recent turns, the
    fastest asked first, hedges to the next ranked models at 1.0s and 2.0s, a failed request replaced
    at once, replies passed on in whole sentences and capped at four, a cut-off
    fragment dropped when max_tokens is hit, back-to-back user turns joined into one before sending,
    spoken fillers at 2.5s („ერთი წამით“, „აბა, ვნახოთ“, …) and 7s (sent to TTS as their own utterances,
    rotated per call so none repeats back to back), a spoken
    apology if every model fails or nothing arrives in 15s, a keep-warm
    connection; models that return 429 quota errors are skipped until their retry delay passes)
  • OR AnthropicLLMService (Claude Haiku 4.5 with prompt caching)
  • System prompt: build_system_prompt() (prompts.py) = GEORGIAN_PROMPT + the call's Tbilisi date/time
     │
     ▼
[Georgian Edge Neural TTS] (edge_tts_service.py)
  • Text normalization: Latin loanwords -> Georgian phonetic Mkedruli
  • Number normalization: Digits (up to hundreds of billions) -> Georgian words (number_to_georgian);
    phone numbers and *codes read in dictation groups, times, decimals, ordinals, case endings
  • Persistent warm WebSocket connection reuse
  • Resampling via soxr & streaming chunk playback
     │
     ▼
[Audio Output] (RTP 8kHz / Speaker 16kHz / Web 16kHz)
```

---

## 3. Directory & File Structure

| File | Purpose |
| :--- | :--- |
| `server_phone.py` | **Direct SIP/RTP Telephone Server**. Pure-Python asynchronous SIP 2.0 user agent with MD5 digest auth, SDP negotiation, PCMU/PCMA/PCM16 decoding, RTP jitter buffer, and pre-cached greeting audio (`_GREETING_PCM16_CACHE`). |
| `bot.py` | **Core Voice Pipeline & LLM/TTS Factories**. Defines `FastGoogleLLMService` (parallel tail-latency hedging), `build_llm()`, `build_tts()`, and Twilio pipeline runner `run_bot()`. |
| `speech_gate.py` | **Multi-stage Audio Filtering & Anti-Hallucination**. Implements `RealtimeVoiceIsolator` (STFT + Wiener), `SpeechOnlyAudioGate`, and `PhantomSpeechFilter`. |
| `edge_tts_service.py` | **Custom Native Georgian TTS Service**. High-performance Pipecat TTS integration using Microsoft Edge Neural. Includes `number_to_georgian()` (vigesimal numeral system) and `normalize_georgian_text()`. |
| `prompts.py` | **System Prompts & Greetings**. Contains `GEORGIAN_PROMPT` (elderly assistant persona, scam alerts, emergency 112 guidance, phonetic Georgian rules) and `INITIAL_GREETING`. |
| `test_mic.py` | **Local Microphone Voice Assistant**. Runs the assistant through connected PC microphones/headsets via PyAudio. |
| `test_mic_hardware.py` | **Hardware Level Visualizer**. Console dBFS volume meter to verify mic levels and hardware devices. |
| `list_mics.py` | **Audio Device Enumerator**. Lists all host audio input devices with indices and sample rates. |
| `server_web.py` | **Web UI Server**. FastAPI server with WebSocket audio bridge for browser testing at `http://localhost:8000`. |
| `web_ui.html` | **Browser Interface**. Audio visualizer, call controls, speech rate adjustment, and live transcripts. |
| `server.py` | **Twilio Webhook Server**. FastAPI server handling Twilio bi-directional audio streams. |
| `Dockerfile` | **Production Container**. Python 3.12 container with PortAudio, libsndfile, ffmpeg. |
| `requirements.txt` | **Python Dependencies**. Pipecat, Deepgram, Google GenAI, Anthropic, Edge-TTS, PyAudio, Soxr, etc. |
| `.env` / `.env.example`| **Configuration Secrets**. API keys and service endpoints. |

---

## 4. Environment & Configuration (`.env`)

```ini
# --- Speech-to-Text (Deepgram) ---
DEEPGRAM_API_KEY=your_deepgram_api_key
DEEPGRAM_MODEL=nova-3

# --- LLM Brain (Gemini or Anthropic) ---
LLM_PROVIDER=gemini                    # 'gemini' or 'anthropic'
GOOGLE_API_KEY=your_google_key
GEMINI_MODEL=gemini-3.1-flash-lite     # tried first until the per-turn speed ranking knows better; gemini-3-flash-preview free tier is 20 req/day

# Anthropic (Optional)
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your_anthropic_key
# ANTHROPIC_MODEL=claude-haiku-4-5

# --- Text-to-Speech (Microsoft Edge Neural) ---
TTS_PROVIDER=edge                      # 'edge' (native Georgian) or 'elevenlabs'
EDGE_TTS_VOICE=ka-GE-EkaNeural         # Female: ka-GE-EkaNeural | Male: ka-GE-GiorgiNeural
EDGE_TTS_RATE=+12%                     # Pacing: +10% to +15% recommended for natural speech
EDGE_TTS_PITCH=+0Hz

# --- Georgian SIP Phone Line (MyPhone.ge) ---
MYPHONE_SIP_SERVER=89.150.1.11
MYPHONE_SIP_PORT=5060
MYPHONE_SIP_USERNAME=your_sip_username
MYPHONE_SIP_PASSWORD=your_sip_password
MYPHONE_PHONE_NUMBER=+995322195820     # Direct incoming number: 032 219 58 20

# --- Audio & Gate Tuning ---
MIN_STT_CONFIDENCE=0.55                # Deepgram confidence floor to discard phantom noise
BARGE_IN_MIN_CONFIDENCE=0.80           # Confidence a one-word transcript needs to interrupt the bot
MIN_VOICE_RMS=600.0                    # Foreground voice threshold for spectral isolator
MUTE_USER_WHILE_BOT_SPEAKS=false       # Barge-in control
RECORD_CALLS=false                     # true: save each caller's audio to recordings/*.wav for gate tuning
SAVE_TRANSCRIPTS=true                  # each call as text (caller number, both sides) in transcripts/*.txt
```

---

## 5. Development Workflows & Running the System

### Windows Virtual Environment
The active virtual environment is located at `.\.venv\Scripts\python.exe`. Always invoke Python using this path or activate the environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 1. Georgian SIP Phone Server (MyPhone.ge)
Starts registration with MyPhone SIP PBX and answers calls to `032 219 58 20`:
```powershell
.\.venv\Scripts\python.exe server_phone.py
```
*Note: Ensure UDP port 5060 and dynamic RTP ports (10000-20000) are not blocked by the local firewall.*

### 2. Local Microphone Mode
Talk directly through your headset or laptop microphone:
```powershell
.\.venv\Scripts\python.exe test_mic.py
```

### 3. Check / Calibrate Microphone Levels
```powershell
# Enumerate audio devices
.\.venv\Scripts\python.exe list_mics.py

# Live dBFS VU meter (pass device index optionally, e.g. 1)
.\.venv\Scripts\python.exe test_mic_hardware.py
```

### 4. Browser Web Simulation
```powershell
.\.venv\Scripts\python.exe server_web.py
# Open http://localhost:8000 in your browser
```

### 5. Twilio Webhook Server
```powershell
.\.venv\Scripts\python.exe server.py
# Expose port with: ngrok http 8000
```

---

## 6. Critical Engineering Patterns & Rules

### A. Georgian Text & Speech Normalization Rules
Edge TTS requires pure Georgian characters (Mkedruli) to avoid unnatural robotic sounds or pronunciation failure:
1. **Never pass Latin characters to Georgian TTS**:
   - Technical terms and loanwords must be written in Georgian phonetically:
     `PIN` → `პინ-კოდი`, `SMS` → `ესემესი`, `Wi-Fi` → `ვაიფაი`, `WhatsApp` → `ვაცაპი`, `Viber` → `ვაიბერი`, `Facebook` → `ფეისბუქი`, `Google` → `გუგლი`.
2. **Never pass digits or symbols to TTS**:
   - Numbers must be converted to full Georgian words using `number_to_georgian()` in `edge_tts_service.py`:
     `112` → `ას თორმეტი`, `25 ₾` → `ოცდახუთი ლარი`, `10 წუთი` → `ათი წუთი`.
   - The vigesimal structure (base-20) in Georgian must be respected (`ოცდა...`, `ორმოცდა...`, `სამოცდა...`, `ოთხმოცდა...`).
3. **Always run text through `normalize_georgian_text()`** before speech synthesis.

### B. Elderly-Centric Persona & Safety Guardrails (`prompts.py`)
- **Tone**: Warm, patient, respectful, addressing users with „თქვენ“; „ბატონო“ / „ქალბატონო“ only when the caller's gender is clear (never „ბატონო ან ქალბატონო“).
- **Brevity**: Maximum 2-3 short sentences per turn. Elderly listeners lose track with long instructions.
- **Anti-Scam Protection**: Never ask for card numbers, PIN codes, or SMS codes. Immediately warn: *„ეს თაღლითობაა! ბანკი არასდროს გკითხავთ კოდს ტელეფონით. დაუყოვნებლივ გათიშეთ ზარი.“*
- **Emergency / Health**: Never diagnose medical conditions. If severe distress is detected: *„გთხოვთ ახლავე გათიშოთ ზარი და დარეკოთ ას თორმეტზე!“*.

### C. Low-Latency Pipeline Optimizations
- **Greeting Pre-Caching**: `warm_greeting_cache()` in `server_phone.py` pre-synthesizes the initial greeting into 8kHz PCM16 bytes so callers hear the greeting within ~0.3s of answering.
- **Persistent TTS WebSocket**: `EdgeTTSService` maintains an active warm WebSocket connection with Microsoft Edge speech endpoints to avoid TCP/TLS handshake latency on every utterance.
- **Fast Turn Turnaround**: Turn-end timeout is tuned to ~0.38s - 0.42s for snappy real-time conversation.
- **Windows High-Resolution Timers**: `server_phone.py` executes `timeBeginPeriod(1)` via `ctypes.windll.winmm` on Windows to guarantee 1ms timer precision for jitter-free 20ms RTP packet transmission.
- **UTF-8 Console Output**: Both `server_phone.py` and `test_mic.py` reconfigure stdout/stderr with `reconfigure(encoding="utf-8")` to prevent Windows UnicodeEncodeError crashes when printing Georgian characters.

### D. Anti-Hallucination & Speech Gating (`speech_gate.py`)
- Background hiss, telephone line noise, or distant TV chatter can cause Deepgram Nova-3 to output hallucinated Georgian phrases when silence is fed.
- `SpeechOnlyAudioGate` buffers 0.35s before voice starts and cuts off audio transmission when VAD stops.
- Transcripts with confidence score < `MIN_STT_CONFIDENCE` (0.55) are discarded by `PhantomSpeechFilter`.
- `NoiseSafeUserTurnStartStrategy` starts the user turn on VAD only while the bot is idle. While the bot is thinking or speaking, only recognized words may interrupt it (two interim words, or a final transcript; a one-word final needs `BARGE_IN_MIN_CONFIDENCE`, default 0.80). Without this, any noise that passes VAD cuts the bot off and pipecat asks the caller to repeat.
- While the reply is still being prepared (nothing playing yet), any recognized word counts, so a caller who paused mid-sentence and goes on cancels the half-question's reply and the LLM gets both parts.
- `BotReplyTracker` (right after the TTS service in every pipeline) keeps the bot "busy" between two sentences of one reply; pipecat reports the bot stopped whenever 0.35 s pass without audio, and noise in that gap used to cut replies off.
- `CompleteUtteranceUserTurnStopStrategy` ends the turn only after Deepgram has answered the Finalize sent on VAD stop (`FlushReportingDeepgramSTTService` reports it, 1.5 s cap). Those answers carry the caller's last words; ending the turn earlier lost them. It then waits `utterance_end_wait()` longer when the words sound unfinished (no final punctuation, a lone word, a trailing comma or conjunction).
- Build Deepgram with `build_stt()` (bot.py), never `DeepgramSTTService` directly.
- Audio that bypasses the pipeline (the cached phone greeting) must be wrapped in `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame` so the turn strategy and echo guard know the bot is speaking.
- If modifying audio logic, never send raw continuous non-voice audio directly to Deepgram.

---

## 7. Working on this Codebase (Guidelines for Claude)

1. **Verify Windows Compatibility**: Paths must handle Windows backslashes / forward slashes properly. External processes should use `.\.venv\Scripts\python.exe`.
2. **Preserve Native Georgian Phonetics**: Any new prompts, bot responses, or test utterances must follow Georgian phonetic conventions and number conversions.
3. **Keep Pipeline Asynchronous**: All network I/O, audio streaming, and SIP/RTP handlers are asynchronous (`asyncio`). Avoid synchronous blocking calls in frame processors or RTP loops.
4. **SIP/RTP Diagnostics**: When troubleshooting phone call audio, check RTP sequence numbers, timestamp increments (160 samples per 20ms frame at 8kHz), and socket buffer sizes.
