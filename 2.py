# train_braille_sentence_model.py
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# train_braille_sentence_model.py (या 2.py) के शीर्ष पर


# ... (बाकी कोड) ...
# ---------------- CONFIG ----------------
DATA_DIR = "synthetic_sentences"
CSV_FILE = os.path.join(DATA_DIR, "labels.csv")
IMG_SIZE = (32,32)
MAX_SENTENCE_LEN = 8 # अधिकतम 8 अक्षर प्रति वाक्य
BATCH_SIZE = 32
EPOCHS = 40 # सटीकता के लिए Epochs बढ़ाया गया
MODEL_SAVE = "braille_sentence_model.h5"

# ---------------- Load Dataset ----------------
df = pd.read_csv(CSV_FILE)
sentences = df['sentence_text'].values
image_files = df['image_name'].values

X = []
y = []

for img_file, sent in zip(image_files, sentences):
    # 🟢 1. इमेज को ग्रेस्केल (1-चैनल) में लोड करें
    img = cv2.imread(os.path.join(DATA_DIR, img_file), cv2.IMREAD_GRAYSCALE) 
    
    letter_width = img.shape[1] // len(sent)
    letters_imgs = []
    
    for i in range(len(sent)):
        letter_img = img[:, i*letter_width:(i+1)*letter_width]
        letter_img = cv2.resize(letter_img, IMG_SIZE)
        
        # Normalization
        letter_img = letter_img.astype("float32") / 255.0
        
        # 🟢 2. 1-चैनल इनपुट शेप के लिए विस्तार करें (32, 32) -> (32, 32, 1)
        letter_img = np.expand_dims(letter_img, axis=-1) 
        letters_imgs.append(letter_img)
    
    # Pad to MAX_SENTENCE_LEN
    while len(letters_imgs) < MAX_SENTENCE_LEN:
        # 🟢 3. 1-चैनल शून्य पैडिंग
        letters_imgs.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32)) 
    
    X.append(letters_imgs)
    y.append(sent.ljust(MAX_SENTENCE_LEN)) # लेबल को भी पैड करें

X = np.array(X, dtype=np.float32) 
print("[INFO] X shape (N, TimeSteps, H, W, Channels):", X.shape)

# Encode characters (26 अक्षर + 1 स्पेस)
chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ") 
encoder = LabelEncoder()
encoder.fit(chars)

y_encoded = np.zeros((len(y), MAX_SENTENCE_LEN, len(chars)), dtype=np.float32)
for i, sent in enumerate(y):
    for j, c in enumerate(sent):
        if c in encoder.classes_:
            y_encoded[i,j] = to_categorical(encoder.transform([c])[0], num_classes=len(chars))

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ---------------- Build Model (CNN-LSTM Sequence) ----------------
# 🟢 4. इनपुट शेप को 1 चैनल पर सेट करें: (MAX_SENTENCE_LEN, 32, 32, 1)
input_layer = Input(shape=(MAX_SENTENCE_LEN, IMG_SIZE[0], IMG_SIZE[1], 1)) 

# 1. CNN Features Extraction (TimeDistributed के साथ)
x = TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'))(input_layer)
x = TimeDistributed(MaxPooling2D(2,2))(x)
x = TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same'))(x)
x = TimeDistributed(MaxPooling2D(2,2))(x)
x = TimeDistributed(Flatten())(x)

# 2. LSTM Sequence Modeling
x = LSTM(128, return_sequences=True)(x) # return_sequences=True आवश्यक है
x = Dropout(0.5)(x)

# 3. Output (TimeDistributed Dense)
x = TimeDistributed(Dense(len(chars), activation='softmax'))(x)

model = Model(inputs=input_layer, outputs=x)
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- Train ----------------
print(f"\n[INFO] Starting training for {EPOCHS} epochs...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

model.save(MODEL_SAVE)
print(f"[DONE] Model saved as {MODEL_SAVE}")