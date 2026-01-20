import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. LOAD THE BIG DATASET (70,000 Records)
df = pd.read_csv('cardiac_failure_processed.csv')

# 2. SELECT FEATURES (Inputs)
# Note: 'cardio' is the target (0=Healthy, 1=Sick)
# We drop 'id' and 'Unnamed: 0' because they are just index numbers, not medical data.
X = df.drop(['cardio', 'id', 'Unnamed: 0'], axis=1)
y = df['cardio']

print("Training on 70,000 patients with features:", X.columns.tolist())

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. STANDARDIZE (Crucial for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- COPY TO ANDROID ---
print("\n=== UPDATED ANDROID VALUES ===")
print("Use these for: ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']")
print("MEANS:", scaler.mean_)
print("SCALES:", scaler.scale_)
print("==============================\n")

# 5. BUILD MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(11,)), # We now have 11 Inputs
    tf.keras.layers.Dense(32, activation='relu'), # More neurons because data is bigger
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. TRAIN
print("Training on massive dataset...")
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=1)

# 7. SAVE
model.save('samsung_lifestyle_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('samsung_lifestyle_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Saved 'samsung_lifestyle_model.tflite' (The Big One)")