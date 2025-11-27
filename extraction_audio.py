import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm  # pip install tqdm (optional, for progress bar)

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
OUTPUT_CSV = "coconut_audio_data.csv"

# Settings
N_MFCC = 20
F_MIN = 50

def extract_all_features(file_path, label):
    try:
        # Load Audio
        y, sr = librosa.load(file_path)
        
        # 1. PHYSICAL FEATURES (Time Domain)
        # RMS (Loudness)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Zero Crossing Rate (Sharpness/Percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 2. SPECTRAL FEATURES (Frequency Domain)
        # Spectral Centroid (Brightness)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)
        
        # Spectral Rolloff (Shape)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        # Spectral Bandwidth (Range)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bw_mean = np.mean(bw)
        bw_std = np.std(bw)

        # 3. MFCCs (Texture)
        # We perform the same normalization/trimming as training to be consistent
        y_norm = librosa.util.normalize(y)
        y_trim, _ = librosa.effects.trim(y_norm, top_db=30)
        mfccs = librosa.feature.mfcc(y=y_trim, sr=sr, n_mfcc=N_MFCC, fmin=F_MIN)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # --- CONSTRUCT DICTIONARY ---
        data = {
            "filename": os.path.basename(file_path),
            "label": label,
            "duration": librosa.get_duration(y=y, sr=sr),
            # Time Domain
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "zcr_mean": zcr_mean,
            "zcr_std": zcr_std,
            # Freq Domain
            "centroid_mean": cent_mean,
            "centroid_std": cent_std,
            "rolloff_mean": rolloff_mean,
            "rolloff_std": rolloff_std,
            "bandwidth_mean": bw_mean,
            "bandwidth_std": bw_std
        }
        
        # Add MFCCs as separate columns
        for i in range(N_MFCC):
            data[f"mfcc_{i+1}"] = mfccs_mean[i]
            
        return data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- MAIN LOOP ---
if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print("Dataset folder not found!")
        exit()

    all_data = []
    
    print("Scanning dataset...")
    # Walk through folders
    for label in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, label)
        
        if os.path.isdir(folder_path):
            print(f"Processing Class: {label}")
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3'))]
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                
                # Extract
                features = extract_all_features(file_path, label)
                
                if features:
                    all_data.append(features)
                    
            print(f"  -> Finished {label}")

    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSUCCESS! Data saved to '{OUTPUT_CSV}'.")
        print("Please upload this file to the chat for analysis.")
    else:
        print("No data extracted.")