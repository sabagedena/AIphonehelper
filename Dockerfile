FROM python:3.12-slim

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

# Expose SIP port and HTTP health check port
EXPOSE 5060/udp
EXPOSE 8000/tcp

# Run the Georgian AI telephone server
CMD ["python", "server_phone.py"]
