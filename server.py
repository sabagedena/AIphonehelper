import json
import os
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from loguru import logger
from dotenv import load_dotenv

from bot import run_bot

load_dotenv()

app = FastAPI(title="Georgian AI Phone Helper")

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Georgian AI Phone Helper for Older People",
        "language": "ka",
    }

@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio Webhook: Called when a phone call arrives.
    Returns TwiML instructing Twilio to open a bi-directional audio WebSocket stream.
    """
    host = request.headers.get("host")
    logger.info(f"Incoming call received. Routing stream to wss://{host}/ws")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/ws" />
    </Connect>
</Response>"""
    return HTMLResponse(content=twiml, media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint handling raw 8kHz PCM/mu-law audio packets from Twilio.
    """
    await websocket.accept()
    logger.info("Twilio WebSocket connection established.")

    try:
        # Twilio sends metadata messages first
        stream_sid = None
        call_sid = None

        while not (stream_sid and call_sid):
            msg_text = await websocket.receive_text()
            data = json.loads(msg_text)
            event = data.get("event")

            if event == "connected":
                logger.info("Twilio media stream connected.")
            elif event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["callSid"]
                logger.info(f"Twilio call started. StreamSid: {stream_sid}, CallSid: {call_sid}")

        # Hand over the live socket to the Pipecat voice pipeline
        await run_bot(websocket, stream_sid, call_sid)

    except Exception as e:
        logger.error(f"Error in phone stream session: {e}")
    finally:
        logger.info("Phone stream closed.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, reload=True)
