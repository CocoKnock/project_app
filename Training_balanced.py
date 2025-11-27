import os
import numpy as np
import librosa
import joblib
import random
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
MODEL_FILE = "svm_audio_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"

N_MFCC = 40
F_MIN = 50  # Cut low hum
CLIPPING_THRESHOLD = 0.05  # If > 5% of the audio is at max volume, it's distorted

def is_bad_audio(y):
    """
    Returns True if audio is silent or heavily distorted (crackling/clipping).
    """
    # 1. Check for Silence
    if np.max(np.abs(y)) < 0.005:
        return True, "Silent"
    
    # 2. Check for Heavy Clipping (Crackling)
    # Check how many samples are close to 1.0 or -1.0
    clipped_samples = np.sum(np.abs(y) > 0.99)
    clip_ratio = clipped_samples / len(y)
    
    if clip_ratio > CLIPPING_THRESHOLD:
        return True, f"Distorted ({clip_ratio*100:.1f}% clipped)"
        
    return False, "OK"

def extract_features(file_path):
    try:
        # Load audio
        audio, sr = librosa.load(file_path)
        
        # --- QUALITY CHECK ---
        bad, reason = is_bad_audio(audio)
        if bad:
            print(f"   [SKIP] {os.path.basename(file_path)} -> {reason}")
            return None

        # --- NORMALIZE VOLUME ---
        # This ensures loudness doesn't bias the model
        audio = librosa.util.normalize(audio)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, fmin=F_MIN)
        return np.mean(mfccs.T, axis=0)
        
    except Exception as e:
        print(f"   [ERROR] {os.path.basename(file_path)}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 1. SCANNING & BALANCING DATASET ---")
    
    # 1. Collect all valid file paths by class
    files_by_class = {}
    
    for label in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.wav', '.mp3'))]
            files_by_class[label] = files
            print(f"   Found {len(files)} files for '{label}'")

    if not files_by_class:
        print("No data found!")
        exit()

    # 2. Find the minimum count to balance
    # Example: If Malauhog=30, Malakanin=50, Malakatad=100 -> Min is 30.
    min_count = min(len(f) for f in files_by_class.values())
    print(f"\n   [BALANCING] Limiting all classes to {min_count} samples each.")
    
    features = []
    labels = []

    # 3. Process exactly 'min_count' files from each folder
    for label, file_list in files_by_class.items():
        print(f"\nProcessing {label}...")
        
        # Shuffle to get a random selection
        random.shuffle(file_list)
        
        count = 0
        for file_path in file_list:
            if count >= min_count: 
                break # Stop once we hit the limit
            
            data = extract_features(file_path)
            
            if data is not None:
                features.append(data)
                labels.append(label)
                count += 1
                
        print(f"   -> Successfully extracted {count} samples.")

    X = np.array(features)
    y = np.array(labels)

    # --- 2. PREPROCESSING ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("\n--- 2. SCALING ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. TRAINING (100% Data, Perfectly Balanced) ---
    print("\n--- 3. TRAINING SVM ---")
    # Even though we manually balanced, we keep class_weight='balanced' as safety
    svm = SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced')
    svm.fit(X_scaled, y_encoded)
    print("Training Complete.")

    # --- 4. SELF-TEST ---
    print("\n--- 4. INTERNAL VERIFICATION ---")
    y_pred = svm.predict(X_scaled)
    acc = accuracy_score(y_encoded, y_pred)
    
    print(f"Model Consistency Score: {acc*100:.2f}%")
    print("(If this is low, your audio files might be too noisy/indistinguishable)")
    
    print("\nConfusion Matrix (Should be balanced):")
    print(confusion_matrix(y_encoded, y_pred))

    # --- 5. SAVE ---
    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)
    print(f"\n[DONE] Saved to {MODEL_FILE}")