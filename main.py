# braille_reader_final_sapi.py (Windows SAPI TTS का उपयोग)
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import sys
import os
import threading 
import win32com.client as wincl # 🟢 pyttsx3 की जगह SAPI TTS

# ---------------- CONFIGURATION ----------------
CONFIG = {
    # सुनिश्चित करें कि braille_sentence_model.h5 मौजूद है
    "SENTENCE_MODEL": "braille_sentence_model.h5", 
    "IMG_SIZE": (32, 32),
    "CLUSTER_DISTANCE": 80, 
    "MIN_DOT_AREA": 60,      
    "MAX_DOT_AREA": 400,     
    "ADAPTIVE_C": 10,        
    "MIN_CIRCULARITY": 0.60, 
    "MAX_SENTENCE_LEN": 8
}

# ---------------- LOAD MODEL ----------------
try:
    if not os.path.exists(CONFIG["SENTENCE_MODEL"]):
        raise FileNotFoundError(f"Model file not found: {CONFIG['SENTENCE_MODEL']}")

    sentence_model = load_model(CONFIG["SENTENCE_MODEL"])
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ") 
    encoder = LabelEncoder()
    encoder.fit(chars)
    print("[INFO] Sentence model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Model loading or setup failed. Error: {e}")
    sys.exit(1)

# ---------------- TTS (Text-to-Speech) - FINAL STABLE FIX ----------------
# 🟢 SAPI TTS इंजन शुरू करें
speaker = wincl.Dispatch("SAPI.SpVoice")
speaker.Rate = 1 # बोलने की दर सेट करें (तेज/धीमा)

def speak_sapi(processed_text):
    """SAPI इंजन में बोलने के लिए एक साधारण फ़ंक्शन।"""
    try:
        # SAPI.SpVoice.Speak() थ्रेडिंग में pyttsx3.runAndWait() से अधिक स्थिर है
        speaker.Speak(processed_text, 0)
    except Exception as e:
        print(f"[SAPI SPEAK FAILED] Error: {e}")

def speak(text):
    """दिए गए टेक्स्ट को एक अलग थ्रेड में SAPI TTS का उपयोग करके बोलता है।"""
    text_to_speak = text.strip()
    
    if not text_to_speak:
        return

    # अक्षरों के बीच खाली जगहें दें ताकि SAPI इसे स्पष्ट रूप से बोले
    processed_text = " ".join(list(text_to_speak))
    
    try:
        print(f"[TTS SAPI] Speaking: {text_to_speak}") 
        
        # 🟢 TTS को एक नए थ्रेड में लॉन्च करें
        # कोई Lock आवश्यक नहीं है, क्योंकि SAPI इंजन इंटरनल डेडलॉक पैदा नहीं करता
        tts_thread = threading.Thread(target=speak_sapi, args=(processed_text,))
        tts_thread.start()
        
    except Exception as e:
        print(f"[TTS SAPI ERROR] Failed to speak: {e}")

# ---------------- PREDICT SENTENCE ----------------
def prepare_sentence_input(letter_images):
    """इमेज को 1-चैनल इनपुट (N, 8, 32, 32, 1) के लिए तैयार करता है।"""
    imgs = []
    for img in letter_images:
        img = cv2.resize(img, CONFIG["IMG_SIZE"])
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.astype("float32") / 255.0
        imgs.append(img)
    while len(imgs) < CONFIG["MAX_SENTENCE_LEN"]:
        imgs.append(np.zeros((CONFIG["IMG_SIZE"][0], CONFIG["IMG_SIZE"][1], 1), dtype=np.float32))
    return np.expand_dims(np.array(imgs), 0)

def predict_sentence(letter_images):
    """तैयार की गई लेटर इमेज से पूरा वाक्य अनुमान लगाता है।"""
    if not letter_images:
        return ""
    X = prepare_sentence_input(letter_images)
    preds = sentence_model.predict(X, verbose=0)[0] 
    chars_out = [encoder.inverse_transform([np.argmax(p)])[0] for p in preds]
    return "".join(chars_out).strip()

# ---------------- DOT DETECTION ----------------
def detect_dots_improved(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5) 
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, CONFIG["ADAPTIVE_C"])
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        is_correct_area = area > CONFIG["MIN_DOT_AREA"] and area < CONFIG["MAX_DOT_AREA"]
        is_circular = False
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity >= CONFIG["MIN_CIRCULARITY"]: 
                is_circular = True
        if is_correct_area and is_circular:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(roi, center, int(radius), (0, 255, 0), 2)
                dots.append(center)
    return dots

# ---------------- CLUSTER DOTS ----------------
def cluster_dots(dots):
    if not dots: return []
    dots = np.array(dots)
    dots = dots[dots[:, 0].argsort()]
    cells = []
    current = [dots[0]]
    for i in range(1, len(dots)):
        if abs(dots[i][0] - dots[i-1][0]) < CONFIG["CLUSTER_DISTANCE"]:
            current.append(dots[i])
        else:
            cells.append(current)
            current = [dots[i]]
    cells.append(current) 
    return cells

# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)
last_sentence = "" 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 

while True:
    ret, frame = cap.read()
    if not ret: break
        
    frame = cv2.flip(frame, 1) 
    h, w = frame.shape[:2]

    # ROI (Region of Interest)
    roi_w, roi_h = 800, 350
    x1, y1 = w // 2 - roi_w // 2, h // 2 - roi_h // 2
    x2, y2 = x1 + roi_w, y1 + roi_h
    roi = frame[y1:y2, x1:x2]

    dots = detect_dots_improved(roi)
    cell_groups = cluster_dots(dots)

    letter_images = []
    for group in cell_groups:
        if not group: continue
        
        xs = [p[0] for p in group]
        ys = [p[1] for p in group]
        
        x1c, y1c = max(min(xs) - 20, 0), max(min(ys) - 20, 0)
        x2c, y2c = min(max(xs) + 20, roi.shape[1]), min(max(ys) + 20, roi.shape[0])
        
        cell = roi[y1c:y2c, x1c:x2c]
        if cell.size == 0: continue
            
        gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        enhanced_cell = clahe.apply(gray_cell) 
        _, thresh_cell = cv2.threshold(enhanced_cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        letter_images.append(thresh_cell) 
    
    sentence = ""
    if letter_images:
        sentence = predict_sentence(letter_images)

    # स्क्रीन पर आउटपुट प्रिंट करें
    cv2.putText(frame, f"WORD: {sentence}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

    # ------------------ TTS LOGIC (SAPI FLOW) ------------------
    current_sentence = sentence.strip() 
    
    # TTS तभी ट्रिगर करें जब नया डिटेक्ट किया गया वाक्य पिछले बोले गए वाक्य से अलग हो
    if current_sentence and current_sentence != last_sentence: 
        speak(current_sentence)
        last_sentence = current_sentence
        
    # अगर कैमरा खाली है, तो last_sentence को रीसेट करें (अगले अक्षर को बोलने के लिए)
    if not current_sentence:
        last_sentence = "" 

    # -------------------------------------------------------------------

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.imshow("ROI (Detected Dots - Green Circles)", roi)
    cv2.imshow("FRAME", frame)
    
    if letter_images:
        # केवल पहला डिटेक्ट किया गया सेल दिखाएँ
        cv2.imshow("Model Input (Thresh Cell)", letter_images[0])

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
