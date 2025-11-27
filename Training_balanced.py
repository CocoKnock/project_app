import os
import numpy as np
import librosa
import joblib
import random
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
MODEL_FILE = "svm_audio_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"

# Feature Settings
N_MFCC = 20
F_MIN = 50 

def extract_features(file_path):
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path)
        
        # 2. Normalize Volume (Remove Loudness Bias)
        y = librosa.util.normalize(y)
        
        # 3. Trim Silence
        y, _ = librosa.effects.trim(y, top_db=30)

        # --- FEATURE EXTRACTION ---
        
        # A. Basic MFCCs (Texture)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, fmin=F_MIN)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # B. Delta MFCCs (Change over time / Decay)
        # This captures the "ringing" difference between Malakanin vs Malakatad
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta_mfcc.T, axis=0)

        # C. Spectral Centroid (Brightness/Pitch)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)

        # D. Spectral Flatness (Tonality)
        # 1.0 = White noise, 0.0 = Pure sine wave
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)

        # Combine: 20 MFCCs + 20 Deltas + 1 Centroid + 1 Flatness = 42 Features
        return np.hstack([mfcc_mean, delta_mean, cent_mean, flatness_mean])
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 1. LOADING DATASET ---")
    features = []
    labels = []

    if not os.path.exists(DATASET_PATH):
        print(f"Error: '{DATASET_PATH}' not found.")
        exit()

    for label_name in os.listdir(DATASET_PATH):
        label_dir = os.path.join(DATASET_PATH, label_name)
        if os.path.isdir(label_dir):
            print(f"Processing class: {label_name}...")
            count = 0
            for filename in os.listdir(label_dir):
                if filename.lower().endswith(('.wav', '.mp3')):
                    path = os.path.join(label_dir, filename)
                    data = extract_features(path)
                    if data is not None:
                        features.append(data)
                        labels.append(label_name)
                        count += 1
            print(f"   -> Loaded {count} files.")

    if not features:
        print("No files loaded. Exiting.")
        exit()

    X = np.array(features)
    y = np.array(labels)
    
    print(f"\nTotal Dataset: {len(X)} samples")

    # --- 2. PREPROCESSING ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Stratified Split (80% Train, 20% Test) to ensure we test on all classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 3. TRAINING ---
    print("\n--- 3. TRAINING MODEL ---")
    # Using 'linear' kernel often works better with high-dimensional data like Deltas
    # class_weight='balanced' is ESSENTIAL for your biased dataset
    svm = SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')
    
    svm.fit(X_train_scaled, y_train)
    print("Training Complete.")

    # --- 4. EVALUATION ---
    print("\n--- 4. TEST RESULTS ---")
    y_pred = svm.predict(X_test_scaled)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # --- 5. SAVE ---
    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)
    print(f"\n[DONE] Model saved to {MODEL_FILE}")