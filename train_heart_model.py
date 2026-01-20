import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# ==========================================
# STEP 1: LOAD THE DATA
# ==========================================
# We use the processed file because it has all 70,000 unique patients.
df = pd.read_csv('cardiac_failure_processed.csv')

# Drop non-medical columns
# 'id': Random number
# 'Unnamed: 0': Row number
# 'cardio': The answer (Target)
X = df.drop(['cardio', 'id', 'Unnamed: 0'], axis=1)
y = df['cardio']

print(f"Loaded {len(df)} patients.")
print("Features:", X.columns.tolist())

# ==========================================
# STEP 2: PREPROCESSING (The "Equalizer")
# ==========================================
# Problem: 'Age' is 0.6, but 'Height' is 170. 
# The model will think Height is 300x more important than Age.
# Solution: StandardScaler forces all columns to be roughly -1 to 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# !!! IMPORTANT FOR YOUR ANDROID APP !!!
# You need these numbers to process the User's input on the phone.
print("\n=== COPY THESE VALUES TO YOUR ANDROID APP ===")
print("// Feature Order: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active")
print("float[] MEANS = {", ", ".join([f"{x:.4f}f" for x in scaler.mean_]), "};")
print("float[] SCALES = {", ", ".join([f"{x:.4f}f" for x in scaler.scale_]), "};")
print("=============================================\n")

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 3: BUILD THE MODEL
# ==========================================
model = tf.keras.Sequential([
    # Input: 11 Features (Age, Gender, Height, Weight, BP_High, BP_Low, Chol, Gluc, Smoke, Alco, Active)
    tf.keras.layers.Input(shape=(11,)),
    
    # Layer 1: 32 Neurons (Judges)
    tf.keras.layers.Dense(32, activation='relu'),
    
    # Layer 2: 16 Neurons
    tf.keras.layers.Dense(16, activation='relu'),
    
    # Output: Risk Score (0.0 to 1.0)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# STEP 4: TRAIN
# ==========================================
print("Training on 70,000 Samsung Health profiles...")
model.fit(X_train, y_train, epochs=80, batch_size=64, verbose=1)

# ==========================================
# STEP 5: SAVE FOR ANDROID
# ==========================================
# Check accuracy
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Get detailed metrics
y_pred_proba = model.predict(X_test, verbose=0)

# Adjust threshold to reduce false negatives
# Default is 0.5, but we decrease it to catch more sick people even if it means more false alarms
THRESHOLD = 0.4  # Predict sick if model is 35%+ confident
y_pred = (y_pred_proba > THRESHOLD).astype(int).flatten()
# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\n=== DETAILED METRICS (Threshold: {THRESHOLD}) ===")
print(f"True Positives (Correctly identified sick): {tp}")
print(f"True Negatives (Correctly identified healthy): {tn}")
print(f"False Positives (Healthy labeled as sick): {fp}")
print(f"False Negatives (Sick labeled as healthy): {fn}")

print(f"\n=== PERFORMANCE METRICS ===")
print(f"Precision: {precision_score(y_test, y_pred):.4f} (of predicted sick, how many actually are?)")
print(f"Recall: {recall_score(y_test, y_pred):.4f} (of all sick people, how many did we catch?)")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

print(f"\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Sick']))
print("========================\n")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('samsung_lifestyle_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success! 'samsung_lifestyle_model.tflite' saved.")