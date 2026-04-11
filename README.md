# 🧠 NeuralLens: Multi-Modal AI Dashboard

NeuralLens is a powerful, multi-modal AI dashboard that integrates Image Classification, Video Object Detection (YOLOv8), Audio Language Identification, and Recommendation Engines into a single, sleek interface.

## 🚀 Deployment Guide (Hugging Face Spaces)

To deploy this project to Hugging Face Spaces, follow these steps:

### 1. Create a New Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click on **New** -> **Space**.
3. Give your space a name (e.g., `neural-lens`).
4. Select **Docker** as the SDK.
5. Choose **Blank** (or "Container") as the template.
6. Click **Create Space**.

### 2. Upload Files
Upload all the project files from your local directory to the Hugging Face repository. You can do this via:
*   **Web Interface**: Drag and drop all files (including the `models/` folder).
*   **Git**: Clone the space repository and push your local files.

### 3. Automatic Build
Once you upload the `Dockerfile`, Hugging Face will automatically start building the container. It will:
*   Install system dependencies (`ffmpeg`, `libGL`).
*   Install Python packages from `requirements.txt`.
*   Load the ML models and start the FastAPI server.

## 🛠 Features
*   **Coffee Bean Quality**: Ensemble classification using SVM, kNN, RF, and CNN.
*   **Traffic Detection**: Real-time video inference using YOLOv8.
*   **Language ID**: CNN-based identification for 10 Indian languages.
*   **FIFA Recommender**: Recommendation system for football players.
*   **E-Commerce Tiering**: Price tiering classification.

## 📦 Dependencies
*   FastAPI & Uvicorn (Backend)
*   TensorFlow & Ultralytics (AI/ML)
*   OpenCV & FFmpeg (Media Processing)
*   Librosa & Soundfile (Audio Processing)
