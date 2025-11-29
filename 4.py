from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os
import base64

CONFIG = {
    "SENTENCE_MODEL": "braille_sentence_model.h5",
    "IMG_SIZE": (32, 32),
    "MAX_SENTENCE_LEN": 8
}

app = Flask(__name__)

# Load model
sentence_model = load_model(CONFIG["SENTENCE_MODEL"])
chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
encoder = LabelEncoder()
encoder.fit(chars)

def prepare_sentence_input(letter_images):
    imgs = []
    for img in letter_images:
        img = cv2.resize(img, CONFIG["IMG_SIZE"])
        img = np.expand_dims(img, -1)
        img = img.astype("float32") / 255.0
        imgs.append(img)

    while len(imgs) < CONFIG["MAX_SENTENCE_LEN"]:
        imgs.append(np.zeros((32, 32, 1), dtype=np.float32))

    return np.expand_dims(np.array(imgs), 0)

@app.post("/predict")
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file part"})
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No selected file"})

        # Convert file to OpenCV image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # ❗ Call your Braille processing here
        predicted_word = "HELLO"

        return jsonify({"status": "success", "word": predicted_word})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
