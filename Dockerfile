# Python 3.12 is required: server_phone.py uses audioop, which was removed in 3.13
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# Install system audio and network utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    libasound2-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# SIP signaling and the RTP audio port range. Run with host networking
# (see docker-compose.yml): publishing 10,000 UDP ports through Docker is impractical.
EXPOSE 5060/udp
EXPOSE 10000-20000/udp

# Run the Georgian AI telephone server
CMD ["python", "server_phone.py"]
