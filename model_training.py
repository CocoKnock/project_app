import os
import numpy as np
import librosa
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
MODEL_FILE = "svm_audio_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"

# Feature Extraction Settings
N_MFCC = 40
F_MIN = 50  # Ignores frequencies below 50Hz (Hum/DC Offset)

# Maturity Weights for Score Calculation
MATURITY_WEIGHTS = {
    'malauhog': 0,     # Young
    'malakanin': 50,   # Semi-Mature
    'malakatad': 100   # Mature
}

def extract_features(file_path):
    try:
        # Load audio (Standard load to avoid resampy errors)
        audio, sr = librosa.load(file_path)
        
        # Extract MFCCs with low-cut filter
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, fmin=F_MIN)
        
        # Average across time
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 1. LOADING FULL DATASET ---")
    features = []
    labels = []

    if not os.path.exists(DATASET_PATH):
        print(f"Error: '{DATASET_PATH}' folder not found.")
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
    
    print(f"\nTotal Training Samples: {len(X)}")

    # --- 2. PREPROCESSING ---
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale Features (Fit on EVERYTHING)
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. TRAINING (FULL DATA) ---
    print("\n--- 3. TRAINING SVM MODEL (100% Data) ---")
    
    # class_weight='balanced' fixes the bias if one folder has more files than others
    svm = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')
    
    svm.fit(X_scaled, y_encoded)
    print("Training Complete.")

    # --- 4. SANITY CHECK (Test on Training Data) ---
    print("\n--- 4. SANITY CHECK (Testing on Training Data) ---")
    # We test on the same data just to verify the model learned the logic
    # ideally this should be near 100% accuracy now
    probs_all = svm.predict_proba(X_scaled)
    class_names = le.classes_

    # Show first 3 samples as example
    for i in range(min(3, len(X))):
        print(f"\nSample {i+1} (Actual: {le.inverse_transform([y_encoded[i]])[0]})")
        probs = probs_all[i]
        current_score = 0
        
        for class_name, prob in zip(class_names, probs):
            print(f"   {class_name.capitalize()}: {prob*100:.2f}%")
            weight = MATURITY_WEIGHTS.get(class_name.lower(), 0)
            current_score += (prob * weight)
            
        print(f"   => MATURITY SCORE: {current_score:.2f}")

    # --- 5. SAVE ARTIFACTS ---
    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)
    print(f"\n[DONE] Model saved to {MODEL_FILE}")