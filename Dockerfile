# Python 3.10 slim base
FROM python:3.10-slim

# Install system dependencies (FFmpeg, curl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face default non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    CLOUD_MODE=1

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user:user requirements.txt $HOME/app/
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all project files
COPY --chown=user:user . $HOME/app/

# Ensure working and static dirs exist (handles root or subfolder uploads)
RUN mkdir -p $HOME/app/temp_work $HOME/app/output_clips $HOME/app/static && \
    (cp -f index.html manifest.json sw.js icon-*.png static/ 2>/dev/null || true)

# Expose HF Spaces port
EXPOSE 7860

# Run FastAPI app
CMD ["python", "app.py"]
