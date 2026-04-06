import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import joblib
import librosa
import tensorflow as tf
from skimage.feature import hog
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import io
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.staticfiles import StaticFiles
from glob import glob
from fastapi.responses import FileResponse
import time
import subprocess
import imageio_ffmpeg as ffmpeg


app = FastAPI()

# ✅ create folder if not exists
if not os.path.exists("output"):
    os.makedirs("output")
app.mount("/output", StaticFiles(directory="output"), name="output")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODELS ----------------
svm = joblib.load("models/svm_model.pkl")
knn = joblib.load("models/knn_model.pkl")
rf  = joblib.load("models/rf_model.pkl")
cnn = tf.keras.models.load_model("models/cnn_model.h5")
img_le = joblib.load("models/image_label_encoder.pkl")

audio_model = tf.keras.models.load_model("models/language_model.h5")
audio_le = joblib.load("models/audio_label_encoder.pkl")

log_model = joblib.load("models/logistic_model.pkl")
gnb_model = joblib.load("models/gnb_model.pkl")

players = pickle.load(open("models/players.pkl", "rb"))

yolo = YOLO("models/Trained_modelv8.pt")

# ---------------- COFFEE ----------------
@app.post("/predict/coffee")
async def predict_coffee(file: UploadFile = File(...)):
    try:
        # ---------------- READ IMAGE ----------------
        contents = await file.read()

        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_arr = np.array(pil_img)

        # Resize (IMPORTANT: same as training)
        img_resized = cv2.resize(img_arr, (128, 128))

        # ---------------- HOG FEATURES (FIXED) ----------------
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),      # 🔥 MUST BE (8,8)
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

        feat_hog = features.reshape(1, -1)

        # ✅ DEBUG (optional – remove later)
        print("HOG shape:", feat_hog.shape)
        print("Expected:", svm.n_features_in_)

        # ---------------- ML MODELS ----------------
        svm_pred = svm.predict(feat_hog)[0]
        knn_pred = knn.predict(feat_hog)[0]
        rf_pred  = rf.predict(feat_hog)[0]

        # ---------------- CNN MODEL ----------------
        img_cnn = img_resized / 255.0
        img_cnn = np.expand_dims(img_cnn, axis=0)   # (1,128,128,3)

        cnn_pred = np.argmax(cnn.predict(img_cnn), axis=1)[0]

        # ---------------- ENSEMBLE ----------------
        preds = [svm_pred, knn_pred, rf_pred, cnn_pred]

        final = Counter(preds).most_common(1)[0][0]
        confidence = preds.count(final) / 4

        label = img_le.inverse_transform([final])[0]

        # ---------------- RESPONSE ----------------
        return {
            "class": label,
            "confidence": float(confidence)
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# ---------------- AUDIO ----------------
from fastapi import UploadFile, File
import numpy as np
import librosa
import soundfile as sf
import tempfile

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Save temp file (works for mp3/mpeg/wav)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temp:
            temp.write(contents)
            temp_path = temp.name

        # Load audio (librosa supports mp3/mpeg too)
        y, sr = librosa.load(temp_path, sr=22050, duration=5)
        y = librosa.util.normalize(y)

        # Convert to Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel)

        # Fix shape (same as training)
        if mel_db.shape[1] < 216:
            mel_db = np.pad(mel_db, ((0, 0), (0, 216 - mel_db.shape[1])))

        mel_db = mel_db[:, :216]
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

        feat = mel_db.reshape(1, 128, 216, 1)

        pred = audio_model.predict(feat)
        idx = np.argmax(pred)

        label = audio_le.inverse_transform([idx])[0]

        return {
            "language": label,
            "confidence": float(np.max(pred))
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------- E-COM ----------------
@app.get("/predict/ecom")
def predict_ecom(price: float):
    X = np.array([[price]])

    p1 = log_model.predict(X)[0]
    p2 = gnb_model.predict(X)[0]

    final = round((p1 + p2) / 2)

    return {
        "class": "Premium" if final == 1 else "Standard",
        "score": float(price / 1000)
    }

# ---------------- FIFA ----------------
@app.get("/predict/fifa")
def recommend(player_name: str):
    df = players

    # Normalize input
    player_name = player_name.lower().strip()

    # Normalize dataset names
    df["name_lower"] = df["short_name"].str.lower()

    # Find closest match
    matches = df[df["name_lower"].str.contains(player_name)]

    if len(matches) == 0:
        return {"error": "Player not found"}

    idx = matches.index[0]

    features = [
        "overall", "potential", "age_fifa", "height_cm",
        "Per 90 Minutes_Gls", "Per 90 Minutes_Ast",
        "Per 90 Minutes_Tackles_Tkl", "Per 90 Minutes_Tkl+Int",
        "Challenges_Tkl%"
    ]

    df = df.dropna(subset=features)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    sim = cosine_similarity([X[idx]], X)[0]
    top = sim.argsort()[-6:][::-1][1:]

    result = []
    for i in top:
        result.append({
            "name": df.iloc[i]["short_name"],
            "score": float(sim[i])
        })

    return {"players": result}

@app.get("/player/details")
def get_player(player_name: str):
    df = players

    player = df[df["short_name"] == player_name]

    if len(player) == 0:
        return {"error": "Player not found"}

    row = player.iloc[0]

    return {
        "name": row["short_name"],
        "overall": int(row["overall"]),
        "potential": int(row["potential"])
    }

@app.get("/players/list")
def get_players():
    names = players["short_name"].dropna().unique().tolist()
    return {"players": names[:200]}  # limit for speed


# ---------------- TRAFFIC ----------------
@app.post("/predict/traffic")
async def predict_traffic(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # 📁 Save input video
        input_path = "input.mp4"
        with open(input_path, "wb") as f:
            f.write(contents)

        raw_path = "output/output.avi"   # 🔥 raw stable file
        final_path = "output/output.mp4" # 🔥 final playable file

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return {"error": "Failed to open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None or fps != fps:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width == 0 or height == 0:
            return {"error": "Invalid video dimensions"}

        print(f"FPS: {fps}, Width: {width}, Height: {height}")

        # 🔥 USE AVI (STABLE)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

        total = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = yolo(frame)

            for r in results:
                boxes = r.boxes
                total += len(boxes)

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{conf:.2f}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2)

            out.write(frame)

        # 🔥 RELEASE
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Frames processed: {frame_count}")

        # ❗ check raw video
        if not os.path.exists(raw_path):
            return {"error": "Raw video not created"}

        if os.path.getsize(raw_path) < 1000:
            return {"error": "Raw video corrupted"}

        # 🔥 CONVERT TO MP4 (FINAL FIX)
        import imageio_ffmpeg as ffmpeg
        import subprocess

        ffmpeg_path = ffmpeg.get_ffmpeg_exe()

        subprocess.run([
            ffmpeg_path,
            "-y",
            "-i", raw_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            final_path
        ])

        # ❗ check final video
        if not os.path.exists(final_path):
            return {"error": "Final video not created"}

        time.sleep(1)

        return {
            "video_url": "http://127.0.0.1:8000/video",
            "detections": total
        }

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/video")
def get_video():
    return FileResponse(
        "output/output.mp4",
        media_type="video/mp4",
        headers={"Cache-Control": "no-store"}
    )