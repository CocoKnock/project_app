import customtkinter as ctk
import pyaudio
import numpy as np
import librosa
import threading
import joblib
import wave
import os
import time
import random # Used for dummy prediction if model is missing

# --- CROSS-PLATFORM GPIO HANDLING ---
try:
    import RPi.GPIO as GPIO
    PLATFORM = "PI"
except (ImportError, RuntimeError):
    PLATFORM = "PC"
    print("⚠️ RPi.GPIO not found. Using MOCK GPIO for Laptop testing.")
    
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        HIGH = 1
        LOW = 0
        
        @staticmethod
        def setmode(mode):
            print(f"[HARDWARE MOCK] Mode set to: {mode}")

        @staticmethod
        def setup(pin, mode):
            print(f"[HARDWARE MOCK] Pin {pin} setup")

        @staticmethod
        def output(pin, state):
            state_str = "ON (High)" if state else "OFF (Low)"
            print(f"[HARDWARE MOCK] Solenoid Pin {pin} -> {state_str}")

        @staticmethod
        def cleanup():
            print("[HARDWARE MOCK] GPIO Cleanup")

    GPIO = MockGPIO
# ------------------------------------

# --- CONFIGURATION ---
SOLENOID_PIN = 17       
SAMPLE_RATE = 44100     
CHUNK_SIZE = 1024
RECORD_SECONDS = 3      
AUDIO_FILENAME = "temp_recording.wav"

# ML CONFIG
MODEL_PATH = "svm_model.pkl"   
SCALER_PATH = "scaler.pkl"     
CLASSES = ["Malauhog", "Malakatad", "Malakanin"]

class CoconutTesterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Coconut Maturity Tester (I2S)")
        self.geometry("600x500")
        ctk.set_appearance_mode("Dark")
        
        # Load ML Model
        self.model = self.load_model(MODEL_PATH)
        self.scaler = self.load_model(SCALER_PATH)

        # GUI Layout
        self.setup_ui()

        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SOLENOID_PIN, GPIO.OUT)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)

    def setup_ui(self):
        # Header
        self.label_title = ctk.CTkLabel(self, text="Acoustic Coconut Classifier", font=("Roboto", 24, "bold"))
        self.label_title.pack(pady=20)

        # Status Display
        self.status_label = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status_label.pack(pady=5)

        # Action Button
        self.btn_scan = ctk.CTkButton(self, text="TAP & SCAN", command=self.start_sequence_thread, height=50, font=("Roboto", 16))
        self.btn_scan.pack(pady=20)

        # Results Frame
        self.res_frame = ctk.CTkFrame(self)
        self.res_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Dominant Frequency Display
        self.lbl_freq = ctk.CTkLabel(self.res_frame, text="Dom Freq: --- Hz", font=("Roboto", 16))
        self.lbl_freq.pack(pady=10)

        # Classification Display
        self.lbl_result = ctk.CTkLabel(self.res_frame, text="Result: ---", font=("Roboto", 28, "bold"), text_color="#3B8ED0")
        self.lbl_result.pack(pady=10)

        # Confidence Text
        self.lbl_conf = ctk.CTkLabel(self.res_frame, text="Confidence:\n---", font=("Roboto", 14))
        self.lbl_conf.pack(pady=10)

    def load_model(self, path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def start_sequence_thread(self):
        """Runs the hardware and processing in a separate thread to keep UI responsive."""
        self.btn_scan.configure(state="disabled")
        self.status_label.configure(text="Tapping & Recording...", text_color="yellow")
        
        # Start thread
        thread = threading.Thread(target=self.run_process)
        thread.start()

    def run_process(self):
        """Main logic flow: Tap -> Record -> Process -> Predict"""
        try:
            # 1. Start Audio Stream (Non-blocking usually, but here we record synchronous to taps)
            frames = []
            p = pyaudio.PyAudio()

            # Find I2S Input Device (User needs to ensure I2S is default or specify index)
            # For this code, we assume default input is the I2S mic
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=CHUNK_SIZE)

            print("Recording started...")
            
            # 2. Activate Solenoid 3 times while recording
            start_time = time.time()
            tap_count = 0
            
            while (time.time() - start_time) < RECORD_SECONDS:
                # Read Audio
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)

                # Handle Tapping Logic (Time-based non-blocking)
                elapsed = time.time() - start_time
                
                # Tap logic: Tap at 0.5s, 1.5s, 2.5s (approx)
                if tap_count < 3:
                    if (elapsed > 0.5 and tap_count == 0) or \
                       (elapsed > 1.5 and tap_count == 1) or \
                       (elapsed > 2.5 and tap_count == 2):
                        self.trigger_solenoid()
                        tap_count += 1

            # Stop Recording
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save WAV for Librosa processing
            wf = wave.open(AUDIO_FILENAME, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # 3. Feature Extraction
            self.update_status("Processing Signal...")
            dom_freq, mfccs_mean = self.extract_features(AUDIO_FILENAME)

            # 4. ML Prediction
            if self.model:
                # Prepare feature vector: [Dom_Freq, MFCC_1, ..., MFCC_13]
                # Note: Scaling is crucial for SVM.
                features = np.hstack(([dom_freq], mfccs_mean))
                features = features.reshape(1, -1)
                
                if self.scaler:
                    features = self.scaler.transform(features)

                prediction_index = self.model.predict(features)[0]
                probabilities = self.model.predict_proba(features)[0]

                predicted_class = CLASSES[prediction_index]
                
                # Format output
                conf_text = f"Malauhog: {probabilities[0]*100:.1f}%\n" \
                            f"Malakanin: {probabilities[1]*100:.1f}%\n" \
                            f"Malakatad: {probabilities[2]*100:.1f}%"

                # Update UI
                self.update_results(dom_freq, predicted_class, conf_text)
            else:
                self.update_status("Error: Model not loaded")

        except Exception as e:
            print(f"Error: {e}")
            self.update_status(f"Error: {e}")
        
        finally:
            self.btn_scan.configure(state="normal")

    def trigger_solenoid(self):
        """Actuates the solenoid briefly"""
        GPIO.output(SOLENOID_PIN, GPIO.HIGH)
        time.sleep(0.05) # Strike duration
        GPIO.output(SOLENOID_PIN, GPIO.LOW)
        print("Tap!")

    def extract_features(self, audio_file):
        # Load audio with Librosa
        y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

        # 1. Dominant Frequency using FFT
        # Get magnitude of Fourier transform
        fft_spectrum = np.fft.rfft(y)
        fft_frequencies = np.fft.rfftfreq(len(y), 1 / sr)
        
        # Find index of max magnitude
        magnitude = np.abs(fft_spectrum)
        peak_freq_index = np.argmax(magnitude)
        dom_freq = fft_frequencies[peak_freq_index]

        # 2. MFCC extraction (13 coefficients is standard)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # SVM requires 1D array, so we take the mean of MFCCs over time
        mfccs_mean = np.mean(mfccs.T, axis=0)

        return dom_freq, mfccs_mean

    def update_status(self, text):
        self.status_label.configure(text=text, text_color="white")

    def update_results(self, freq, result, conf_details):
        self.lbl_freq.configure(text=f"Dom Freq: {freq:.2f} Hz")
        self.lbl_result.configure(text=result)
        
        # Color coding based on result
        if result == "Malauhog": color = "#FF5555" # Redish
        elif result == "Malakanin": color = "#FFD700" # Gold
        else: color = "#2CC985" # Green
        
        self.lbl_result.configure(text_color=color)
        self.lbl_conf.configure(text=conf_details)
        self.status_label.configure(text="Scan Complete", text_color="gray")

    def on_closing(self):
        GPIO.cleanup()
        self.destroy()

if __name__ == "__main__":
    app = CoconutTesterApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()