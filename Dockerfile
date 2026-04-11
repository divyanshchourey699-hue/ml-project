# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory and ensure it has correct permissions
RUN mkdir -p output && chmod 777 output

# Expose the port (7860 is standard for Hugging Face)
EXPOSE 7860

# Run uvicorn on container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
