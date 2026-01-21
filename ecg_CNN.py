import wfdb
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Your exact path verified by diagnostic
DATA_PATH = r"C:/Users/harry/Downloads/GitHub/H4H-Hackathon/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0"

RECORDS = ['100', '106', '109', '118', '119', '124', '200', '202', '210', '213', '214', '219']
WINDOW_SIZE = 187  

# ==========================================
# 2. THE EXTRACTOR (VERBOSE MODE)
# ==========================================
def load_and_train():
    X_list = []
    y_list = []
    
    print(f"[*] Reading files from: {DATA_PATH}")
    
    for record_name in RECORDS:
        file_path = os.path.join(DATA_PATH, record_name)
        print(f"\n--- Processing Record {record_name} ---")
        
        try:
            # Load Data
            record = wfdb.rdrecord(file_path)
            annotation = wfdb.rdann(file_path, 'atr')
            
            # Extract Signal (MLII Lead)
            signal = record.p_signal[:, 0]
            
            # Extract Annotations
            indices = annotation.sample
            symbols = annotation.symbol
            
            print(f"   -> Signal Length: {len(signal)}")
            print(f"   -> Annotations: {len(symbols)}")
            
            # Loop through beats
            count_extracted = 0
            for i, idx in enumerate(indices):
                sym = symbols[i]
                
                # CLEAN THE SYMBOL (Remove spaces)
                if isinstance(sym, str):
                    sym = sym.strip()
                
                # LABELING: Separate Normal vs Abnormal
                label = -1
                if sym in ['N', '/']:  # Normal or Paced (treated as normal)
                    label = 0
                elif sym in ['V', 'E', 'A', 'a', 'J', 'S', 'F']:  # Various arrhythmias
                    label = 1
                
                if label != -1:
                    # SLICE THE WINDOW
                    start = idx - (WINDOW_SIZE // 2)
                    end = idx + (WINDOW_SIZE // 2) + 1  # +1 to ensure WINDOW_SIZE samples
                    
                    # Handle boundaries with padding
                    if start < 0 or end > len(signal):
                        # Pad edges as needed
                        pad_left = max(0, -start)
                        pad_right = max(0, end - len(signal))
                        
                        # Extract what we can from signal
                        actual_start = max(0, start)
                        actual_end = min(end, len(signal))
                        
                        beat_clip = signal[actual_start:actual_end]
                        beat_clip = np.pad(beat_clip, (pad_left, pad_right), mode='constant', constant_values=0)
                    else:
                        beat_clip = signal[start:end]
                    
                    if len(beat_clip) == WINDOW_SIZE:
                        X_list.append(beat_clip)
                        y_list.append(label)
                        count_extracted += 1

        except Exception as e:
            print(f"   [ERROR] Error on {record_name}: {e}")

    # Final Check
    if len(X_list) == 0:
        print("\n[CRITICAL] No beats were extracted. Check the logs above.")
        print(f"[DEBUG] X_list length: {len(X_list)}, y_list length: {len(y_list)}")
        return

    # Convert to Numpy
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Reshape for CNN (Samples, 187, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"\n[SUCCESS] TOTAL DATASET: {len(X)} samples")
    print(f"   Normal (0): {np.sum(y == 0)}")
    print(f"   PVC/Sick (1): {np.sum(y == 1)}")

    # ==========================================
    # 3. TRAIN THE NPU MODEL
    # ==========================================
    print("\n[TRAINING] Starting Training...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
        # Layer 1: Scanner
        tf.keras.layers.Conv1D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        # Layer 2: Deep Scanner
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        # Classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[RESULT] Final Accuracy: {acc*100:.2f}%")
    
    # Save
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('ecg_cnn_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("[SUCCESS] Model saved as 'ecg_cnn_model.tflite'")

if __name__ == "__main__":
    load_and_train()