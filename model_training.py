import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
MODEL_FILE = "svm_audio_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"

# Feature Extraction Settings
N_MFCC = 40
F_MIN = 50  # <--- CRITICAL: Ignores frequencies below 50Hz (Hum/DC Offset)

# Maturity Weights for Score Calculation
MATURITY_WEIGHTS = {
    'malauhog': 0,     # Young
    'malakanin': 50,   # Semi-Mature
    'malakatad': 100   # Mature
}

def extract_features(file_path):
    try:
        # Load audio (Removed 'res_type' to fix resampy error)
        audio, sr = librosa.load(file_path)
        
        # Extract MFCCs with low-cut filter (fmin)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, fmin=F_MIN)
        
        # Average across time
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 1. LOADING DATASET ---")
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

    # --- 2. PREPROCESSING ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split 70% Train, 30% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 3. TRAINING ---
    print("\n--- 3. TRAINING MODEL ---")
    # probability=True is REQUIRED for the percentage outputs
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    svm.fit(X_train_scaled, y_train)
    print("Training Complete.")

    # --- 4. ACCURACY CHECK ---
    print("\n--- 4. EVALUATION ---")
    y_pred = svm.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Global Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # --- 5. DEMONSTRATION OF OUTPUTS (Probabilities + Score) ---
    print("\n--- 5. SAMPLE OUTPUTS (What the App will see) ---")
    
    # Get probabilities for the test set
    probs_all = svm.predict_proba(X_test_scaled)
    class_names = le.classes_

    # Show first 5 test samples
    for i in range(min(5, len(X_test))):
        print(f"\nSample {i+1} (Actual: {le.inverse_transform([y_test[i]])[0]})")
        
        current_score = 0
        probs = probs_all[i]
        
        # Calculate Probabilities & Maturity Score
        for class_name, prob in zip(class_names, probs):
            # 1. Output Probability
            print(f"   {class_name.capitalize()}: {prob*100:.2f}%")
            
            # 2. Add to Score
            weight = MATURITY_WEIGHTS.get(class_name.lower(), 0)
            current_score += (prob * weight)
            
        print(f"   => MATURITY SCORE: {current_score:.2f} / 100")

    # --- 6. SAVE ARTIFACTS ---
    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)
    print(f"\n[DONE] Model saved to {MODEL_FILE}")