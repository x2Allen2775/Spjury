# Use stable Python version compatible with MediaPipe
FROM python:3.11

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV + MediaPipe + FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create necessary folders (safety)
RUN mkdir -p uploads outputs

# Expose Flask/Gunicorn port
EXPOSE 8000

# Use Gunicorn for production
CMD ["gunicorn", "--workers", "2", "--threads", "2", "--timeout", "600", "--bind", "0.0.0.0:8000", "app:app"]
