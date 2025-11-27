import os
import numpy as np
import scipy.io.wavfile as wav
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
MODEL_FILE = "svm_audio_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "encoder.pkl"

def extract_fft_features(file_path):
    try:
        # 1. Read Audio (Raw)
        sample_rate, data = wav.read(file_path)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data[:, 0]
            
        # Convert to float and Normalize (-1 to 1)
        data = data.astype(np.float32)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        # 2. Compute FFT
        n = len(data)
        fft_spectrum = np.abs(np.fft.rfft(data))
        frequencies = np.fft.rfftfreq(n, d=1/sample_rate)

        # 3. FILTER: Keep only 50Hz - 3000Hz 
        # (Removes DC Offset/Hum AND the High-Pitch Solenoid Click)
        mask = (frequencies >= 50) & (frequencies <= 3000)
        valid_freqs = frequencies[mask]
        valid_mags = fft_spectrum[mask]

        if len(valid_mags) == 0: return None

        # --- FEATURE 1: DOMINANT FREQUENCY ---
        # The loudest frequency in the coconut's range
        peak_idx = np.argmax(valid_mags)
        dominant_freq = valid_freqs[peak_idx]

        # --- FEATURES 2, 3, 4: ENERGY BANDS ---
        # We calculate "How much Bass vs. Treble" the sound has.
        total_energy = np.sum(valid_mags) + 1e-6

        # Low Band (Malauhog/Young): 50 - 800 Hz
        mask_low = (valid_freqs >= 50) & (valid_freqs < 800)
        energy_low = np.sum(valid_mags[mask_low]) / total_energy

        # Mid Band (Malakanin/Semi): 800 - 1800 Hz
        mask_mid = (valid_freqs >= 800) & (valid_freqs < 1800)
        energy_mid = np.sum(valid_mags[mask_mid]) / total_energy

        # High Band (Malakatad/Mature): 1800 - 3000 Hz
        mask_high = (valid_freqs >= 1800) & (valid_freqs <= 3000)
        energy_high = np.sum(valid_mags[mask_high]) / total_energy

        # Return vector: [Freq, %Low, %Mid, %High]
        return np.array([dominant_freq, energy_low, energy_mid, energy_high])

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 1. LOADING FULL DATASET (FFT PHYSICS) ---")
    features = []
    labels = []

    if not os.path.exists(DATASET_PATH):
        print("Error: Dataset folder not found.")
        exit()

    for label_name in os.listdir(DATASET_PATH):
        label_dir = os.path.join(DATASET_PATH, label_name)
        if os.path.isdir(label_dir):
            print(f"Processing class: {label_name}...")
            count = 0
            for filename in os.listdir(label_dir):
                if filename.lower().endswith(('.wav', '.mp3')):
                    path = os.path.join(label_dir, filename)
                    data = extract_fft_features(path)
                    if data is not None:
                        features.append(data)
                        labels.append(label_name)
                        count += 1
            print(f"   -> Loaded {count} files.")

    if not features:
        print("No data found.")
        exit()

    X = np.array(features)
    y = np.array(labels)
    
    print(f"\nTotal Training Data: {len(X)} samples")

    # --- 2. PREPROCESSING ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale Features (Crucial: Freq is ~1000, Energy is ~0.5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 3. TRAINING (100% DATA) ---
    print("\n--- 3. TRAINING MODEL (NO SPLIT) ---")
    
    # 'linear' kernel creates simple, clear boundaries based on the physics
    # class_weight='balanced' forces the model to respect smaller classes
    svm = SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')
    
    svm.fit(X_scaled, y_encoded)
    print("Training Complete.")

    # --- 4. VERIFICATION ---
    print("\n--- 4. SELF-CHECK (Accuracy on Training Data) ---")
    y_pred = svm.predict(X_scaled)
    acc = accuracy_score(y_encoded, y_pred)
    print(f"Consistency Score: {acc*100:.2f}%")
    
    print("\nConfusion Matrix (Should be balanced):")
    print(confusion_matrix(y_encoded, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_encoded, y_pred, target_names=le.classes_))

    # --- 5. SAVE ---
    joblib.dump(svm, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)
    print(f"\n[DONE] Model saved to {MODEL_FILE}")