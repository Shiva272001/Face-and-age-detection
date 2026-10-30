FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies required by OpenCV and PyTorch/YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code, models, and static assets
COPY . .

# Expose port
EXPOSE 8000

# Start server honoring the environment PORT (default 8000)
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
