# Georgian AI Voice Helper & Phone Assistant

A real-time Georgian voice AI assistant designed for natural conversation, with zero foreign accent, instant turn-taking, and phone/local mic integration.

---

## 🌟 Key Features
- **👂 Deepgram Nova-3 (`ka`)**: Real-time Georgian Speech-to-Text with anti-hallucination speech gating (`speech_gate.py`).
- **🧠 Google Gemini 3.1 Flash Lite** (with automatic fast hedging across Gemini 3.5 Flash Lite / 3.6 Flash, or Claude): Low-latency Georgian conversational intelligence and safe guidance.
- **🗣️ Microsoft Edge Neural TTS (`ka-GE-EkaNeural` / `ka-GE-GiorgiNeural`)**: 100% authentic native Georgian accent, 0% foreign accent, persistent warm WebSocket connection reuse, and incremental streaming audio decoding.
- **⚡ Real-Time Conversational Flow**: ~0.38s turn-end detection, pre-warmed LLM & TTS connections, instant barge-in support, and +12% natural voice pacing.
- **🎙️ Headset & Laptop Mic Support**: Auto-detects Bluetooth headsets (e.g. Marshall Major IV) and laptop microphone arrays with dBFS level metering.
- **📞 Georgian SIP Phone Line (MyPhone.ge) & Twilio Ready**: Direct SIP/RTP telephone server (`server_phone.py`) registered to Georgian landline number **`+995 32 219 58 20` (`032 219 58 20`)**, plus Twilio webhook server (`server.py`).

---

## 🚀 How to Run

### 1. Call from Your Phone (Georgian Number via MyPhone.ge SIP)
Start the Georgian SIP/RTP phone server:
```powershell
.\.venv\Scripts\python.exe server_phone.py
```
Then dial **`032 219 58 20`** (or **`+995 32 219 58 20`**) from your mobile or landline phone to talk directly with the Georgian AI assistant.

### 2. Speak via Your Microphone (Local Mode)
To talk directly to the assistant through your headphones or laptop mic:
```powershell
.\.venv\Scripts\python.exe test_mic.py
```
*(Tip: Test your mic levels anytime with `.\.venv\Scripts\python.exe test_mic_hardware.py`)*

### 3. Test in Web Browser
To simulate a phone call inside your browser:
```powershell
.\.venv\Scripts\python.exe server_web.py
```
Open **`http://localhost:8000`** in your browser to test different voices and speeds.

### 4. Real Telephone Calls (Twilio Webhook)
Start the Twilio telephone webhook server:
```powershell
.\.venv\Scripts\python.exe server.py
```
Expose it to the internet with `ngrok http 8000` and link your Twilio number's incoming webhook to `https://<ngrok-url>/incoming-call`.

---

## ☁️ Deploy 24/7 (No Laptop Needed)

Run the phone server on a small Linux VPS so the number answers even when your laptop is off.

**Server:** Ubuntu 24.04, 2 vCPU / 2–4 GB RAM, in Europe close to Georgia (e.g. Hetzner Germany/Finland, DigitalOcean or Vultr Frankfurt). PaaS hosts like Render, Railway or Heroku **will not work** — they only forward HTTP, and SIP/RTP needs raw UDP ports.

```bash
# 1. Install Docker on the server
curl -fsSL https://get.docker.com | sh

# 2. Get the code onto the server (git clone or scp), then copy .env separately
scp .env root@<server-ip>:/opt/georgian-ai/.env

# 3. Open the SIP and RTP audio ports (also allow them in the provider's cloud firewall)
ufw allow 22/tcp && ufw allow 5060/udp && ufw allow 10000:20000/udp && ufw enable

# 4. Build and start; it restarts automatically after crashes and reboots
cd /opt/georgian-ai && docker compose up -d --build

# Logs / restart after pulling new code
docker compose logs -f
git pull && docker compose up -d --build
```

**Before going live:**
- **Stop `server_phone.py` on your laptop.** Only one server should be registered to the MyPhone account at a time.
- If registration fails from the VPS (403 or timeout), ask MyPhone.ge support to allow the server's IP.
- Use a paid Gemini API key — an always-on line will hit free-tier daily limits.
- With `RECORD_CALLS=true`, recordings are saved in `/opt/georgian-ai/recordings/` on the server.

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
| `MIN_STT_CONFIDENCE`| Drops phantom words generated from background noise | `0.55` |
