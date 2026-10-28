# Georgian AI Voice Helper & Phone Assistant

A real-time Georgian voice AI assistant designed for natural conversation, with zero foreign accent, instant turn-taking, and phone/local mic integration.

---

## 🌟 Key Features
- **👂 Deepgram Nova-3 (`ka`)**: Real-time Georgian Speech-to-Text with anti-hallucination speech gating (`speech_gate.py`).
- **🧠 Google Gemini 3.1 Flash Lite** (with automatic fast hedging across Gemini 3.5 Flash Lite / 3.6 Flash, or Claude): Low-latency Georgian conversational intelligence and safe guidance.
- **🗣️ Microsoft Edge Neural TTS (`ka-GE-EkaNeural` / `ka-GE-GiorgiNeural`)**: 100% authentic native Georgian accent, 0% foreign accent, persistent warm WebSocket connection reuse, and incremental streaming audio decoding.
- **⚡ Real-Time Conversational Flow**: ~0.38s turn-end detection, pre-warmed LLM & TTS connections, instant barge-in support, and +12% natural voice pacing.
- **🎙️ Headset & Laptop Mic Support**: Auto-detects Bluetooth headsets (e.g. Marshall Major IV) and laptop microphone arrays with dBFS level metering.
- **📞 Twilio Phone Ready**: Fully configured telephone webhook server for live cellular calls.

---

## 🚀 How to Run

### 1. Speak via Your Microphone (Local Mode)
To talk directly to the assistant through your headphones or laptop mic:
```powershell
.\.venv\Scripts\python.exe test_mic.py
```
*(Tip: Test your mic levels anytime with `.\.venv\Scripts\python.exe test_mic_hardware.py`)*

### 2. Test in Web Browser
To simulate a phone call inside your browser:
```powershell
.\.venv\Scripts\python.exe server_web.py
```
Open **`http://localhost:8000`** in your browser to test different voices and speeds.

### 3. Real Telephone Calls (Twilio Phone Line)
Start the telephone webhook server:
```powershell
.\.venv\Scripts\python.exe server.py
```
Expose it to the internet with `ngrok http 8000` and link your Twilio number's incoming webhook to `https://<ngrok-url>/incoming-call`.

---

## ⚙️ Configuration (`.env`)

| Variable | Description | Default |
| :--- | :--- | :--- |
| `DEEPGRAM_API_KEY` | Deepgram speech-to-text API key | (Configured) |
| `GOOGLE_API_KEY` | Google AI Studio Gemini API key | (Configured) |
| `GEMINI_MODEL` | Gemini LLM model | `gemini-3.1-flash-lite` |
| `TTS_PROVIDER` | Speech synthesis provider (`edge` or `elevenlabs`) | `edge` |
| `EDGE_TTS_VOICE` | Voice: `ka-GE-EkaNeural` (Female) / `ka-GE-GiorgiNeural` (Male) | `ka-GE-EkaNeural` |
| `EDGE_TTS_RATE` | Speech rate multiplier | `+12%` |
| `MIC_DEVICE_INDEX` | Force specific mic index from `list_mics.py` (empty = auto) | `` |
| `MIN_STT_CONFIDENCE`| Drops phantom words generated from background noise | `0.45` |
