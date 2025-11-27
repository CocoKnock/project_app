from pathlib import Path
import customtkinter as ctk
from tkinter import Canvas, messagebox
import cv2
import joblib
import numpy as np
import pandas as pd
import sys
import os
import time
import platform
import threading
from joblib import load
from PIL import Image, ImageTk
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

# --- AUDIO IMPORTS ---
import pyaudio
import wave
import librosa

# ---------------------------------------------------------
# GLOBALS & STATE
# ---------------------------------------------------------
video_capture = None
picam = None
camera_after = None

selected_file = None       # Image file path
selected_audio_file = None # Audio file path

# Global dictionary to hold results from both steps
processing_results = {
    "image_probs": None, 
    "image_score": 0.0,   
    "image_class": None,  
    "audio_probs": None,  
    "audio_score": 0.0,
    "audio_class": None
}

USE_PICAMERA2 = False

# --- CONFIGURATION ---
SOLENOID_PIN = 17
SAMPLE_RATE = 48000
CHUNK_SIZE = 4096
RECORD_SECONDS = 3

# UPDATED FOR NEW TRAINING (Delta Features)
N_MFCC = 20   # Changed from 40 to 20
F_MIN = 50    # Low-cut filter

# --- HARDWARE SETUP (MOCK & REAL) ---
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SOLENOID_PIN, GPIO.OUT)
    PLATFORM = "PI"
except (ImportError, RuntimeError):
    PLATFORM = "PC"
    class MockGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setup(pin, mode): pass
        @staticmethod
        def output(pin, state):
            if state: print(f"[MOCK] Solenoid STRIKE!")
        @staticmethod
        def cleanup(): pass
    GPIO = MockGPIO


# ---------------------------------------------------------
# FUZZY LOGIC ENGINE (MATLAB REPLICATION)
# ---------------------------------------------------------
class CoconutFuzzySystem:
    def __init__(self):
        # x-axis range for defuzzification (0 to 100)
        self.x = np.arange(0, 101, 1)

    # Membership Functions
    def trapmf(self, x, a, b, c, d):
        """Trapezoidal Membership Function"""
        y = np.zeros_like(x, dtype=float)
        mask1 = (x > a) & (x < b)
        y[mask1] = (x[mask1] - a) / (b - a)
        mask2 = (x >= b) & (x <= c)
        y[mask2] = 1.0
        mask3 = (x > c) & (x < d)
        y[mask3] = (d - x[mask3]) / (d - c)
        return y

    def trimf(self, x, a, b, c):
        """Triangular Membership Function"""
        y = np.zeros_like(x, dtype=float)
        mask1 = (x > a) & (x < b)
        y[mask1] = (x[mask1] - a) / (b - a)
        mask2 = (x >= b) & (x < c)
        y[mask2] = (c - x[mask2]) / (c - b)
        return y

    def compute(self, img_score, aud_score):
        # 1. FUZZIFY INPUTS
        xi = np.array([img_score])
        xa = np.array([aud_score])

        # Image MFs
        img_hog = self.trapmf(xi, -50, 0, 30, 50)[0]
        img_kan = self.trimf(xi, 30, 50, 70)[0]
        img_tad = self.trapmf(xi, 50, 70, 100, 150)[0]

        # Audio MFs
        aud_hog = self.trapmf(xa, -50, 0, 30, 50)[0]
        aud_kan = self.trimf(xa, 30, 50, 70)[0]
        aud_tad = self.trapmf(xa, 50, 70, 100, 150)[0]

        # 2. EVALUATE RULES
        out_mf_hog = self.trapmf(self.x, -50, 0, 30, 50)
        out_mf_kan = self.trimf(self.x, 30, 50, 70)
        out_mf_tad = self.trapmf(self.x, 50, 70, 100, 150)

        # Rule evaluation (Min operator + Weights)
        r1 = min(img_hog, aud_hog) * 1.0
        r2 = min(img_kan, aud_hog) * 0.5
        r3 = min(img_tad, aud_hog) * 0.5
        r4 = min(img_hog, aud_kan) * 0.3
        r5 = min(img_kan, aud_kan) * 1.0
        r6 = min(img_tad, aud_kan) * 0.3
        r7 = min(img_hog, aud_tad) * 0.5
        r8 = min(img_kan, aud_tad) * 0.5
        r9 = min(img_tad, aud_tad) * 1.0

        # 3. AGGREGATION (Max Operator)
        active_hog = np.fmin(max(r1, r2, r3), out_mf_hog)
        active_kan = np.fmin(max(r4, r5, r6), out_mf_kan)
        active_tad = np.fmin(max(r7, r8, r9), out_mf_tad)

        aggregated = np.fmax(active_hog, np.fmax(active_kan, active_tad))

        # 4. DEFUZZIFICATION (Centroid)
        numerator = np.sum(self.x * aggregated)
        denominator = np.sum(aggregated)

        if denominator == 0:
            return 0.0 
        
        centroid = numerator / denominator
        return centroid


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
try:
    cam_scaler = joblib.load("camera_scaler.pkl")
    cam_model = load("camera_model.pkl")
    print("[INFO] Camera models loaded.")
except Exception as e:
    print(f"[WARN] Camera Model/Scaler not loaded: {e}")
    cam_scaler = None
    cam_model = None

try:
    # Ensure these are the NEW files from train_model_delta.py
    audio_model = joblib.load("svm_audio_model.pkl")
    audio_scaler = joblib.load("scaler.pkl") 
    audio_encoder = joblib.load("encoder.pkl")
    print("[INFO] Audio models loaded.")
except Exception as e:
    print(f"[WARN] Audio Models not loaded: {e}")
    audio_model = None
    audio_scaler = None
    audio_encoder = None

try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except Exception as e:
    print("PiCamera2 not available:", e)
    USE_PICAMERA2 = False


# ---------------------------------------------------------
# PATHS & THEME
# ---------------------------------------------------------
if getattr(sys, 'frozen', False):
    base_dir = Path(sys.executable).parent
else:
    base_dir = Path(__file__).parent

folder_path1 = base_dir / "Data Collection Captures"
folder_path2 = base_dir / "Data Detection Captures" 
os.makedirs(folder_path1, exist_ok=True)
os.makedirs(folder_path2, exist_ok=True)

ctk.set_appearance_mode("Light")
BG = "#E8E5DA"
BTN = "#2E8B57"
BTN_HOVER = "#58D68D"
TEXT = "#1B5E20"
CARD = "#F7F6F3"

FONT_TITLE = ("Arial", 46, "bold")
FONT_SUB = ("Arial", 36, "bold")
FONT_NORMAL = ("Arial", 18)


# ---------------------------------------------------------
# MAIN WINDOW SETUP
# ---------------------------------------------------------
root = ctk.CTk()
root.title("COCONUTZ - Coconut Type Classifier")

_machine = platform.machine().lower()
if ("arm" in _machine) or ("raspberry" in _machine):
    root.attributes("-fullscreen", True)
else:
    root.geometry("800x480")
    root.resizable(False, False)

root.configure(fg_color=BG)
main_container = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
main_container.pack(fill="both", expand=True)
main_container.grid_rowconfigure(0, weight=1)
main_container.grid_columnconfigure(0, weight=1)

current_page_frame = None


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def make_button(master, text, command, width=None, height=None, font=FONT_NORMAL, fg_color=BTN):
    btn = ctk.CTkButton(
        master, text=text, command=command,
        fg_color=fg_color, hover_color=BTN_HOVER,
        text_color="white", corner_radius=10, font=font
    )
    if width: btn.configure(width=width)
    if height: btn.configure(height=height)
    return btn

def make_label(master, text, font=FONT_NORMAL, anchor="center"):
    return ctk.CTkLabel(master, text=text, font=font, text_color=TEXT, anchor=anchor)

def stop_camera():
    global video_capture, picam, camera_after
    if camera_after: root.after_cancel(camera_after)
    camera_after = None
    if video_capture and hasattr(video_capture, "release"): video_capture.release()
    video_capture = None
    if picam:
        try: picam.stop(); picam.close()
        except: pass
        picam = None

def calculate_maturity_score(probs_dict):
    if not probs_dict: return 0.0
    p_hog = probs_dict.get('malauhog', 0.0)
    p_kan = probs_dict.get('malakanin', 0.0)
    p_tad = probs_dict.get('malakatad', 0.0)
    score = (p_hog * 0) + (p_kan * 50) + (p_tad * 100)
    return score

# ---------------------------------------------------------
# FEATURE EXTRACTION (IMAGE & AUDIO)
# ---------------------------------------------------------
def compute_moments(pixel_array):
    if len(pixel_array) == 0: return 0.0, 0.0, 0.0, 0.0
    if len(np.unique(pixel_array)) == 1: return float(pixel_array[0]), 0.0, 0.0, 0.0
    m1 = np.mean(pixel_array)
    m2 = np.std(pixel_array)
    m3 = skew(pixel_array, bias=False); m3 = 0.0 if np.isnan(m3) else m3
    m4 = kurtosis(pixel_array, bias=False); m4 = 0.0 if np.isnan(m4) else m4
    return m1, m2, m3, m4

def camera_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return [0.0] * 108
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([1, 1, 1]), np.array([255, 255, 255])
    mask = cv2.inRange(img, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return [0.0] * 108
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi_hsv = hsv[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]
    step_h, step_w = h // 3, w // 3
    feature_vector = []
    for row in range(3):
        for col in range(3):
            y_s, y_e = row*step_h, (row+1)*step_h if row<2 else h
            x_s, x_e = col*step_w, (col+1)*step_w if col<2 else w
            tile_hsv = roi_hsv[y_s:y_e, x_s:x_e]
            tile_mask = roi_mask[y_s:y_e, x_s:x_e]
            for i in range(3):
                feature_vector.extend(compute_moments(tile_hsv[:,:,i][tile_mask>0]))
    return feature_vector

# --- UPDATED AUDIO FEATURE EXTRACTION (DELTA + FLATNESS) ---
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        
        # 1. Normalize Volume (Crucial for Bias)
        y = librosa.util.normalize(y)
        
        # 2. Trim Silence
        y, _ = librosa.effects.trim(y, top_db=30)

        # 3. Basic MFCC (20)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, fmin=F_MIN)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # 4. Delta MFCC (Decay/Ringing)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta_mfcc.T, axis=0)

        # 5. Spectral Centroid
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # 6. Spectral Flatness
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # Combine into one vector (42 features total)
        return np.hstack([mfcc_mean, delta_mean, cent, flatness])
        
    except Exception as e:
        print(f"Audio Feat Error: {e}")
        return None

# --- IMAGE PREPROCESSING TASK ---
def camera_prepro(image_path, threaded=False, on_complete=None):
    def process_task():
        global processing_results
        try:
            img = cv2.imread(image_path)
            if img is None: raise ValueError("Unreadable image")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([5, 30, 30]), np.array([90, 255, 255]))
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = np.uint8(labels == largest_label) * 255
            else: mask = np.zeros_like(mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            processed_file_local = os.path.splitext(image_path)[0] + "_processed.jpg"
            cv2.imwrite(processed_file_local, cv2.resize(masked_img, (224, 224)))

            features = camera_features(processed_file_local)
            
            if cam_model and cam_scaler:
                x_scaled = cam_scaler.transform([features])
                probs = cam_model.predict_proba(x_scaled)[0]
                class_map = {0: "malauhog", 1: "malakanin", 2: "malakatad"}
                img_probs = {class_map[i]: float(probs[i]) for i in range(len(probs))}
                
                # STORE IN GLOBAL RESULTS
                processing_results["image_probs"] = img_probs
                processing_results["image_score"] = calculate_maturity_score(img_probs)
                processing_results["image_class"] = max(img_probs, key=img_probs.get)
            else:
                processing_results["image_probs"] = None
            
            return processed_file_local, True
        except Exception as e:
            print(f"Img Process Error: {e}")
            return None, False

    if threaded:
        def thread_wrapper():
            res_path, success = process_task()
            if on_complete: root.after(0, lambda: on_complete(res_path, success))
        threading.Thread(target=thread_wrapper, daemon=True).start()
    else:
        return process_task()


# ---------------------------------------------------------
# PAGE MANAGER
# ---------------------------------------------------------
pages = {}
def switch_page(page_name):
    global current_page_frame
    stop_camera()
    if current_page_frame: current_page_frame.destroy()
    if page_name in pages:
        current_page_frame = pages[page_name]()
        current_page_frame.grid(row=0, column=0, sticky="nsew")
    else: print(f"[WARN] Unknown page: {page_name}")


# ---- PAGE 1: MAIN MENU ----
def load_main_page():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=2)
    frame.grid_rowconfigure(1, weight=6)   
    frame.grid_columnconfigure(0, weight=1)
    make_label(frame, "COCONUTZ\n Coconut Maturity Classifier", font=FONT_TITLE).grid(row=0, column=0, pady=(60, 30), sticky="n")
    btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
    btn_frame.grid(row=1, column=0, sticky="n")
    make_button(btn_frame, "Data Detection", lambda: switch_page("data_detection1")).grid(row=0, column=0, pady=15, ipadx=30, ipady=10)
    make_button(btn_frame, "Exit", root.destroy).grid(row=2, column=0, pady=40, ipadx=30)
    return frame
pages["main"] = load_main_page      

# ---- PAGE 2: IMAGE CAPTURE ----
def load_data_detection_page_1():
    global video_capture, picam, camera_after, selected_file
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(1, weight=4)
    frame.grid_columnconfigure(0, weight=1)
    make_label(frame, "Image Capture", font=FONT_SUB).grid(row=0, column=0, pady=6)
    
    cam_card = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=12)
    cam_card.grid(row=1, column=0, padx=12, pady=8, sticky="nsew")
    cam_canvas = Canvas(cam_card, bg="#5A56C8", highlightthickness=0)
    cam_canvas.pack(fill="both", expand=True, padx=8, pady=8)

    selected_file = None
    stop_camera()

    def update_frame(img_arr):
        if img_arr is None: return
        w, h = cam_canvas.winfo_width(), cam_canvas.winfo_height()
        if w < 10: w, h = 560, 280
        img = Image.fromarray(cv2.resize(img_arr, (w, h)))
        imgtk = ImageTk.PhotoImage(image=img)
        cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
        cam_canvas.image = imgtk

    if USE_PICAMERA2:
        picam = Picamera2()
        picam.configure(picam.create_preview_configuration(main={"size": (640, 480)}))
        picam.start()
        def loop_pi():
            global camera_after
            try:
                frame_arr = cv2.rotate(picam.capture_array(), cv2.ROTATE_180)
                update_frame(frame_arr)
                camera_after = root.after(30, loop_pi)
            except: stop_camera()
        loop_pi()
        def capture():
            global selected_file
            try:
                frame_arr = cv2.rotate(picam.capture_array(), cv2.ROTATE_180)
                path = folder_path2 / f"detect_img_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR))
                selected_file = str(path)
                stop_camera()
                switch_page("data_detection2")
            except Exception as e: messagebox.showerror("Error", str(e))
    else:
        video_capture = cv2.VideoCapture(0)
        def loop_cv():
            global camera_after
            try:
                ret, frame = video_capture.read()
                if ret: update_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                camera_after = root.after(30, loop_cv)
            except: stop_camera()
        loop_cv()
        def capture():
            global selected_file
            try:
                ret, frame = video_capture.read()
                if ret:
                    path = folder_path2 / f"detect_img_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    cv2.imwrite(str(path), frame)
                    selected_file = str(path)
                    stop_camera()
                    switch_page("data_detection2")
            except Exception as e: messagebox.showerror("Error", str(e))

    controls = ctk.CTkFrame(frame, fg_color=BG)
    controls.grid(row=2, column=0, pady=10)
    make_button(controls, "Capture", capture).pack(side="left", padx=10)
    make_button(controls, "Back", lambda: [stop_camera(), switch_page("main")]).pack(side="left", padx=10)
    return frame
pages["data_detection1"] = load_data_detection_page_1

# ---- PAGE 3: IMAGE PREVIEW ----
def load_data_detection_page_2():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    make_label(frame, "Captured Image", font=FONT_SUB).grid(row=0, column=0, pady=6)
    if selected_file:
        img = Image.open(selected_file)
        img.thumbnail((500, 300))
        imgtk = ImageTk.PhotoImage(img)
        lbl = ctk.CTkLabel(frame, image=imgtk, text="")
        lbl.image = imgtk
        lbl.grid(row=1, column=0)
    
    btns = ctk.CTkFrame(frame, fg_color=BG)
    btns.grid(row=2, column=0, pady=20)
    make_button(btns, "Retake", lambda: switch_page("data_detection1")).pack(side="left", padx=10)
    make_button(btns, "Next: Process", lambda: switch_page("data_detection3")).pack(side="left", padx=10)
    return frame
pages["data_detection2"] = load_data_detection_page_2

# ---- PAGE 4: IMAGE PROCESSING ----
def load_data_detection_page_3():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.pack(fill="both", expand=True)
    make_label(frame, "Processing Image...", font=FONT_SUB).place(relx=0.5, rely=0.5, anchor="center")

    def done(path, success):
        if success: 
            # Redirect to Audio Page (detection4) instead of summary
            switch_page("data_detection4")
        else: 
            messagebox.showerror("Error", "Image Processing failed")
            switch_page("data_detection1")

    if selected_file:
        camera_prepro(selected_file, threaded=True, on_complete=done)
    return frame
pages["data_detection3"] = load_data_detection_page_3

# ---- PAGE 4.5: AUDIO RECORDING & ANALYSIS (With Clean Audio Logic) ----
def load_data_detection_audio_page():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    make_label(frame, "Audio Detection", font=FONT_TITLE).grid(row=0, pady=20)
    status_lbl = make_label(frame, "Position Solenoid & Press Start", font=FONT_NORMAL)
    status_lbl.grid(row=1, pady=10)

    def trigger_solenoid():
        GPIO.output(SOLENOID_PIN, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)

    def process_audio_result(audio_path):
        global processing_results
        
        # 1. Extract (Using new Delta/Flatness logic)
        features = extract_audio_features(audio_path)
        
        # 2. Predict
        if features is not None and audio_model and audio_scaler:
            try:
                # Scale
                x = np.array([features])
                x_scaled = audio_scaler.transform(x)
                
                # Probability
                probs = audio_model.predict_proba(x_scaled)[0]
                classes = audio_encoder.classes_
                
                # Map results
                aud_probs = {cls_name: float(prob) for cls_name, prob in zip(classes, probs)}
                
                # Save Global Results
                processing_results["audio_probs"] = aud_probs
                processing_results["audio_score"] = calculate_maturity_score(aud_probs)
                processing_results["audio_class"] = max(aud_probs, key=aud_probs.get)
                
                print(f"[AUDIO] Score: {processing_results['audio_score']:.2f}")
                switch_page("data_detection5") # Go to Summary
            except Exception as e:
                print(f"Prediction Error: {e}")
                messagebox.showerror("Error", f"Prediction Error: {e}")
                switch_page("main")
        else:
            messagebox.showerror("Error", "Audio Analysis Failed (Check Models)")
            switch_page("main")

    def start_recording():
        btn_rec.configure(state="disabled", text="RECORDING...", fg_color="#555")
        
        def run_thread():
            try:
                ts = time.strftime('%Y%m%d-%H%M%S')
                temp_audio_path = folder_path2 / f"detect_audio_{ts}.wav"
                
                p = pyaudio.PyAudio()
                
                # --- I2S SEARCH ---
                idx = 1
                for i in range(p.get_device_count()):
                    name = p.get_device_info_by_index(i).get('name', '').lower()
                    if any(k in name for k in ["i2s", "snd_rpi", "simple"]):
                        idx = i; break
                
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, input_device_index=idx, frames_per_buffer=CHUNK_SIZE)
                
                raw_frames = []
                start_time = time.time()
                tap_count = 0
                
                # 5dB Gain
                GAIN_DB = 5.0
                gain_factor = 10 ** (GAIN_DB / 20.0)
                
                status_lbl.configure(text="Recording (5dB Boost)...", text_color="#E67E22")
                
                # RECORD RAW FIRST (To avoid DC Offset chopping)
                while (time.time() - start_time) < RECORD_SECONDS:
                    raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    raw_frames.append(raw)
                    
                    elapsed = time.time() - start_time
                    if tap_count < 3:
                        if (elapsed > 0.5 and tap_count==0) or (elapsed > 1.5 and tap_count==1) or (elapsed > 2.5 and tap_count==2):
                            trigger_solenoid(); tap_count += 1

                stream.stop_stream(); stream.close(); p.terminate()
                
                # PROCESS RAW DATA (One chunk)
                full_bytes = b''.join(raw_frames)
                data_np = np.frombuffer(full_bytes, dtype=np.int16).astype(np.float32)
                
                # 1. Remove DC Offset (Global)
                data_np = data_np - np.mean(data_np)
                
                # 2. Apply Gain
                data_np = data_np * gain_factor
                
                # 3. Clip
                data_np = np.clip(data_np, -32768, 32767)
                
                # Save
                wf = wave.open(str(temp_audio_path), 'wb')
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes(data_np.astype(np.int16).tobytes()); wf.close()
                
                status_lbl.configure(text="Analyzing...", text_color="blue")
                root.after(100, lambda: process_audio_result(str(temp_audio_path)))
                
            except Exception as e:
                print(e)
                status_lbl.configure(text="Error", text_color="red")
                btn_rec.configure(state="normal", text="START")

        threading.Thread(target=run_thread, daemon=True).start()

    btn_rec = make_button(frame, "START RECORDING", start_recording, width=250, height=60, fg_color="#D32F2F")
    btn_rec.grid(row=2, pady=30)
    return frame
pages["data_detection4"] = load_data_detection_audio_page


# ---- PAGE 5: SUMMARY & RESULTS ----
def load_data_detection_page_5():
    global selected_file, processing_results

    frame = ctk.CTkFrame(main_container, fg_color="#E8E5DA")
    frame.grid(row=0, column=0, sticky="nsew")
    main_container.grid_rowconfigure(0, weight=1)
    main_container.grid_columnconfigure(0, weight=1)

    # ---- Responsive Layout ----
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1, uniform="col")  # Left: Image
    frame.grid_columnconfigure(1, weight=1, uniform="col")  # Right: Info

    # ==========================
    # FUZZY LOGIC COMPUTATION
    # ==========================
    fuzzy = CoconutFuzzySystem()
    img_s = processing_results['image_score']
    aud_s = processing_results['audio_score']
    
    # Get crisp output (Centroid)
    final_score = fuzzy.compute(img_s, aud_s)
    
    # Map back to text label based on centroid location
    if final_score < 35: final_decision = "MALAUHOG"
    elif final_score < 65: final_decision = "MALAKANIN"
    else: final_decision = "MALAKATAD"

    # ==========================
    # LEFT SIDE: IMAGE PREVIEW
    # ==========================
    image_frame = ctk.CTkFrame(frame, fg_color="#E8E5DA")
    image_frame.grid(row=0, column=0, sticky="nsew", padx=(30, 15), pady=30)
    image_frame.grid_rowconfigure(0, weight=1)
    image_frame.grid_columnconfigure(0, weight=1)

    if selected_file and os.path.exists(selected_file):
        try:
            canvas_card = ctk.CTkFrame(image_frame, corner_radius=15, fg_color="#FFFFFF")
            canvas_card.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            canvas_card.grid_rowconfigure(0, weight=1)
            canvas_card.grid_columnconfigure(0, weight=1)

            tk_canvas = Canvas(canvas_card, bg="#FFFFFF", highlightthickness=0)
            tk_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

            def update_summary_preview():
                try:
                    img = Image.open(selected_file)
                    w = tk_canvas.winfo_width() or 560
                    h = tk_canvas.winfo_height() or 280
                    img = img.resize((w, h), Image.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    tk_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                    tk_canvas.image = imgtk
                except Exception:
                    pass
                tk_canvas.after(200, update_summary_preview)

            update_summary_preview()

        except Exception as e:
            make_label(image_frame, "No image available.", font=FONT_NORMAL).grid(
                row=0, column=0, sticky="nsew", pady=8
            )
    else:
        make_label(image_frame, "No image available.", font=FONT_NORMAL).grid(
            row=0, column=0, sticky="nsew", pady=8
        )

    # ==========================
    # RIGHT SIDE: SUMMARY INFO
    # ==========================
    info_frame = ctk.CTkFrame(frame, fg_color="#E8E5DA")
    info_frame.grid(row=0, column=1, sticky="nsew", padx=(15, 30), pady=30)
    for i in range(8):
        info_frame.grid_rowconfigure(i, weight=1)
    info_frame.grid_columnconfigure(0, weight=1)

    make_label(info_frame, "Classification Summary", font=("Aerial", 24, "bold")).grid(
        row=0, column=0, sticky="n", pady=(10, 15)
    )

    make_label(info_frame, f"FINAL: {final_decision} ({final_score:.1f})", font=("Aerial", 20, "bold")).grid(
        row=1, column=0, sticky="n", pady=(5, 10)
    )

    # HEADERS
    header_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
    header_frame.grid(row=2, column=0, pady=(5, 5))
    header_frame.grid_columnconfigure((0, 1), weight=1, uniform="headers")

    make_label(header_frame, "Camera", font=FONT_NORMAL).grid(row=0, column=0, sticky="ew")
    make_label(header_frame, "Audio", font=FONT_NORMAL).grid(row=0, column=1, sticky="ew")

    # VALUES (Probabilities)
    classes = ["malauhog", "malakanin", "malakatad"]
    
    img_probs = processing_results.get("image_probs", {})
    aud_probs = processing_results.get("audio_probs", {})

    for i, cls in enumerate(classes):
        row_index = 3 + i
        # Safe get with default 0.0
        cam_val = img_probs.get(cls, 0.0) if img_probs else 0.0
        aud_val = aud_probs.get(cls, 0.0) if aud_probs else 0.0
        
        val_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        val_frame.grid(row=row_index, column=0, sticky="ew", pady=2)
        val_frame.grid_columnconfigure((0, 1), weight=1, uniform="vals")

        make_label(val_frame, f"{cls.capitalize()}: {cam_val*100:.1f}%", font=FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=20)
        make_label(val_frame, f"{cls.capitalize()}: {aud_val*100:.1f}%", font=FONT_NORMAL).grid(row=0, column=1, sticky="w", padx=20)

    # MATURITY SCORES ROW
    score_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
    score_frame.grid(row=6, column=0, sticky="ew", pady=10)
    score_frame.grid_columnconfigure((0, 1), weight=1, uniform="vals")
    
    make_label(score_frame, f"Score: {processing_results['image_score']:.1f}", font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=20)
    make_label(score_frame, f"Score: {processing_results['audio_score']:.1f}", font=("Arial", 16, "bold")).grid(row=0, column=1, sticky="w", padx=20)

    # ==========================
    # BUTTONS
    # ==========================
    make_button(info_frame, "Back to Main", lambda: switch_page("main")).grid(
        row=7, column=0, pady=(20, 10), sticky="se", padx=20
    )

    return frame

pages["data_detection5"] = load_data_detection_page_5

# ----------------------------
# App start
# ----------------------------
switch_page("main")
root.bind("<Key>", lambda e: root.attributes("-fullscreen", False) if e.keysym == "Escape" else None)
root.mainloop()