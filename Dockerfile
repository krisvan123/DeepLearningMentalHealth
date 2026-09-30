FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML pipeline, model weights, responses, and backend API
COPY backend/ ./backend/
COPY config/ ./config/
COPY model/ ./model/
COPY models/ ./models/
COPY preprocessing/ ./preprocessing/
COPY respon/ ./respon/
COPY response/ ./response/
COPY utils/ ./utils/

# Default port for Hugging Face Spaces is 7860
ENV PORT=7860
EXPOSE 7860

# Launch FastAPI server with Uvicorn
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860"]
