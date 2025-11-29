# synthetic_sentence_dataset.py
import os
import cv2
import random
import csv
import numpy as np
import sys

# ---------------- CONFIG ----------------
LETTER_DIR = "dataset" # A-Z folders with images
OUTPUT_DIR = "synthetic_sentences"
CSV_FILE = os.path.join(OUTPUT_DIR, "labels.csv")
NUM_SENTENCES = 2500 # अच्छी ट्रेनिंग के लिए संख्या बढ़ाई गई
MIN_LEN = 3
MAX_LEN = 8
IMG_SIZE = (32,32)
LETTER_MARGIN = 5 # सेल्स के बीच पिक्सल मार्जिन

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Letter folder mapping (A से Z)
letter_paths = {chr(65+i): os.path.join(LETTER_DIR, chr(65+i)) for i in range(26)}

def get_random_letter_img(letter):
    """एक रैंडम इमेज को लोड करता है, उसे रिसाइज़ करता है, और 1-चैनल बाइनरी थ्रेशोल्डिंग लागू करता है।"""
    folder = letter_paths[letter]
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not files:
        raise ValueError(f"No images found in the dataset folder for letter: {letter} ({folder})")
        
    fname = random.choice(files)
    img = cv2.imread(os.path.join(folder, fname))
    
    # 1. इमेज को ग्रेस्केल में बदलें
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE)

    # 2. Otsu's Thresholding (लाइव रीडर से मैच करने के लिए)
    # THRESH_BINARY_INV: डॉट्स काले (0), बैकग्राउंड सफेद (255)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) 
    
    return img # 1-चैनल बाइनरी इमेज वापस करता है

# --- मुख्य जनरेशन लूप ---
with open(CSV_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_name", "sentence_text"])
    
    for i in range(1, NUM_SENTENCES+1):
        sentence_len = random.randint(MIN_LEN, MAX_LEN)
        letters = [random.choice(list(letter_paths.keys())) for _ in range(sentence_len)]
        sentence_text = "".join(letters)
        
        try:
            imgs = [get_random_letter_img(l) for l in letters]
        except ValueError as e:
            print(f"[ERROR] Skipping generation: {e}")
            sys.exit(1)

        # मार्जिन को 1-चैनल (ग्रेस्केल) के रूप में परिभाषित करें
        margin = np.zeros((IMG_SIZE[1], LETTER_MARGIN), dtype=np.uint8) 
        
        sentence_img = imgs[0]
        for img in imgs[1:]:
            sentence_img = cv2.hconcat([sentence_img, margin, img])
        
        img_name = f"img_{i:04d}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), sentence_img)
        writer.writerow([img_name, sentence_text])
        
        if i % 100 == 0:
            print(f"[INFO] {i}/{NUM_SENTENCES} sentences generated")

print(f"[DONE] Synthetic sentences + labels.csv created in {OUTPUT_DIR}")