# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive 

# Set the working directory in the container
WORKDIR /app

# Install FFmpeg and other dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Create necessary directories
RUN mkdir -p config output cache assets/music

# Create default config files if they don't exist
RUN if [ ! -f config/api_keys.json ]; then \
    echo '{"openai": "", "elevenlabs": "", "pexels": "", "pixabay": "", "unsplash": ""}' > config/api_keys.json; \
    fi

RUN if [ ! -f config/default_settings.json ]; then \
    echo '{"default_llm_provider": "openai", "default_llm_model": "gpt-4", "script": {"tone": "informative", "include_timestamps": true}, "voiceover": {"provider": "elevenlabs"}, "visual_assets": {"providers": ["pexels", "pixabay", "unsplash"]}, "output_dir": "output"}' > config/default_settings.json; \
    fi

# Expose the port the web UI runs on
EXPOSE 5000

# Command to run the web UI
CMD ["python", "ui/web/app.py"]

# Alternatively, use this to run the CLI
# ENTRYPOINT ["python", "main.py"]
