from pathlib import Path
import customtkinter as ctk
from tkinter import Canvas, messagebox
import cv2
import numpy as np
import pandas as pd
import sys
import os
import time
import platform
import threading
import datetime
import shutil
from joblib import load
from PIL import Image, ImageTk

# --- AUDIO IMPORTS ---
import pyaudio
import wave

# --- HARDWARE SETUP (MOCK & REAL) ---
try:
    import RPi.GPIO as GPIO
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

# --- CONFIGURATION ---
SOLENOID_PIN = 17
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
RECORD_SECONDS = 3

# Globals
video_capture = None
picam = None
camera_after = None

selected_file = None       # Will remain None for Data Collection (Audio Only)
selected_audio_file = None 
selected_label = None

processed_file = None
hsv_class = None

USE_PICAMERA2 = False

# --- MODEL LOADING ---
try:
    model = load("camera_model.pkl")
except Exception as e:
    print(f"[WARN] Model not loaded: {e}")
    model = None

try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except Exception as e:
    print("PiCamera2 not available:", e)
    USE_PICAMERA2 = False


# Paths
if getattr(sys, 'frozen', False):
    base_dir = Path(sys.executable).parent
else:
    base_dir = Path(__file__).parent

csv_path = base_dir / "coconut_features.csv"

DATA_DIR = base_dir / "dataset"
folder_path1 = base_dir / "Data Collection Captures"
folder_path2 = base_dir / "Data Detection Captures"

os.makedirs(folder_path1, exist_ok=True)
os.makedirs(folder_path2, exist_ok=True)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# Theme Helpers
ctk.set_appearance_mode("Light")
BG = "#E8E5DA"
BTN = "#2E8B57"
BTN_HOVER = "#58D68D"
TEXT = "#1B5E20"
CARD = "#F7F6F3"

FONT_TITLE = ("Arial", 46, "bold")
FONT_SUB = ("Arial", 36, "bold")
FONT_NORMAL = ("Arial", 18)


# Window setup
root = ctk.CTk()
root.title("COCONUTZ - Coconut Type Classifier")

_machine = platform.machine().lower()
if ("arm" in _machine) or ("raspberry" in _machine):
    root.attributes("-fullscreen", True)
else:
    root.geometry("800x480")
    root.resizable(False, False)

root.configure(fg_color=BG)

# GPIO Init
GPIO.setmode(GPIO.BCM)
GPIO.setup(SOLENOID_PIN, GPIO.OUT)

main_container = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
main_container.pack(fill="both", expand=True)
main_container.grid_rowconfigure(0, weight=1)
main_container.grid_columnconfigure(0, weight=1)

current_page_frame = None

#  UI Helpers
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
    lbl = ctk.CTkLabel(master, text=text, font=font, text_color=TEXT, anchor=anchor)
    return lbl

def stop_camera():
    global video_capture, picam, camera_after
    if camera_after: root.after_cancel(camera_after)
    camera_after = None

    if video_capture and hasattr(video_capture, "release"):
        video_capture.release()
    video_capture = None

    if picam:
        try:
            picam.stop()
            picam.close()
        except: pass
        picam = None

# --- CAMERA PROCESSING ---
def camera_features(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return [np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])]

def camera_prepro(image_path, threaded=False, on_complete=None):
    def process_task():
        global hsv_class, processed_file
        processed_file = None
        hsv_class = None
        try:
            img = cv2.imread(image_path)
            if img is None: raise ValueError("Unreadable image")

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([5, 30, 30]), np.array([90, 255, 255]))
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            resized_img = cv2.resize(masked_img, (224, 224))
            
            processed_file = os.path.splitext(image_path)[0] + "_processed.jpg"
            cv2.imwrite(processed_file, resized_img)
            
            features = camera_features(processed_file)
            
            if model:
                x = np.array([features])
                probs = model.predict_proba(x)[0]
                hsv_class = {model.classes_[i]: float(probs[i]) for i in range(len(model.classes_))}
            else:
                hsv_class = None

        except Exception as e:
            print(f"Cam Process Error: {e}")
            processed_file, hsv_class = None, None

        if threaded and on_complete:
            root.after(0, lambda: on_complete(processed_file, hsv_class))
        elif not threaded:
            return processed_file, hsv_class

    if threaded:
        threading.Thread(target=process_task, daemon=True).start()
    else:
        return process_task()

def save_features_to_csv(features, filepath, label):
    df = pd.DataFrame([[label] + features], columns=["Label", "H_mean", "S_mean", "V_mean"])
    df.to_csv(filepath, mode="a", header=not Path(filepath).exists(), index=False)


# --- PAGE MANAGER ---
pages = {}

def switch_page(page_name):
    global current_page_frame
    stop_camera()
    if current_page_frame:
        current_page_frame.destroy()
    if page_name in pages:
        current_page_frame = pages[page_name]()
        current_page_frame.grid(row=0, column=0, sticky="nsew")
    else:
        print(f"[WARN] Unknown page: {page_name}")


# =========================================================
#                       PAGES
# =========================================================

# ---- Main Page ----
def load_main_page():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure((0,1), weight=1) 
    frame.grid_columnconfigure(0, weight=1)

    title = make_label(frame, "COCONUTZ\n Coconut Maturity Classifier", font=FONT_TITLE)
    title.grid(row=0, column=0, pady=(60, 30), sticky="s")

    btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
    btn_frame.grid(row=1, column=0, sticky="n")

    make_button(btn_frame, "Data Collection", lambda: switch_page("data_collection2"), width=300).pack(pady=10)
    make_button(btn_frame, "Data Detection", lambda: switch_page("data_detection1"), width=300).pack(pady=10)
    make_button(btn_frame, "Exit", root.destroy, width=300, fg_color="#C62828").pack(pady=20)

    return frame
pages["main"] = load_main_page

# ---- DATA COLLECTION 1 (Camera) ----
# NOTE: This is skipped in the flow as per request, but kept in code if needed later.
def load_data_collection_page_1():
    # ... (Code omitted for brevity as it is skipped, but logic exists in previous version)
    # Since we are skipping this, we just redirect back or show error if accessed manually
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    make_label(frame, "Camera Skipped").pack()
    make_button(frame, "Back", lambda: switch_page("main")).pack()
    return frame

# pages["data_collection1"] = load_data_collection_page_1 # COMMENTED OUT TO DISABLE ROUTING

# ---- DATA COLLECTION 2 (Label Selection) ----
def load_data_collection_page_2():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.pack_propagate(False)
    
    make_label(frame, "Step 1: Select Classification", font=FONT_SUB).pack(pady=40)

    def set_label_next(label):
        global selected_label, selected_file
        selected_label = label
        selected_file = None # Reset image file to None since we are skipping camera
        
        # --- MODIFIED: SKIPS CAMERA (data_collection1) AND GOES TO AUDIO ---
        switch_page("data_collection_audio") 

    classes = ["malauhog", "malakanin", "malakatad"]
    for c in classes:
        make_button(frame, c.capitalize(), lambda l=c: set_label_next(l), width=250, height=50).pack(pady=10)

    make_button(frame, "Back", lambda: switch_page("main"), fg_color="#555").pack(pady=30)
    return frame
pages["data_collection2"] = load_data_collection_page_2

# ---- DATA COLLECTION (AUDIO) ----
def load_data_collection_audio_page():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    make_label(frame, f"Audio Collection ({selected_label.capitalize()})", font=FONT_TITLE).grid(row=0, pady=20)
    
    status_lbl = make_label(frame, "Ready to Record (3 Seconds)", font=FONT_NORMAL, anchor="center")
    status_lbl.grid(row=1, pady=10)

    def trigger_solenoid():
        GPIO.output(SOLENOID_PIN, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)

    def start_recording():
        btn_rec.configure(state="disabled", text="RECORDING...", fg_color="#555")
        
        def run_thread():
            global selected_audio_file
            try:
                ts = time.strftime('%Y%m%d-%H%M%S')
                temp_audio_name = f"tmp_audio_{ts}.wav"
                temp_audio_path = folder_path1 / temp_audio_name
                
                p = pyaudio.PyAudio()
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
                
                frames = []
                start_time = time.time()
                tap_count = 0
                
                status_lbl.configure(text="Recording & Tapping...", text_color="#E67E22")
                
                while (time.time() - start_time) < RECORD_SECONDS:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                    
                    elapsed = time.time() - start_time
                    if tap_count < 3:
                        if (elapsed > 0.5 and tap_count == 0) or \
                           (elapsed > 1.5 and tap_count == 1) or \
                           (elapsed > 2.5 and tap_count == 2):
                            trigger_solenoid()
                            tap_count += 1
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                wf = wave.open(str(temp_audio_path), 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                selected_audio_file = str(temp_audio_path)
                status_lbl.configure(text="Recording Complete", text_color="green")
                
                root.after(500, lambda: switch_page("data_collection3"))
                
            except Exception as e:
                print(e)
                status_lbl.configure(text=f"Error: {e}", text_color="red")
                btn_rec.configure(state="normal", text="START RECORDING", fg_color=BTN)

        threading.Thread(target=run_thread, daemon=True).start()

    btn_rec = make_button(frame, "START RECORDING", start_recording, width=250, height=60, fg_color="#D32F2F")
    btn_rec.grid(row=2, pady=30)
    
    return frame
pages["data_collection_audio"] = load_data_collection_audio_page

# ---- DATA COLLECTION 3 (Confirm & Save) ----
def load_data_collection_page_3():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.pack_propagate(False)
    
    make_label(frame, f"Save Audio for {selected_label}?", font=FONT_TITLE).pack(pady=(40, 20))
    # make_label(frame, "Both Image and Audio will be saved.", font=FONT_NORMAL).pack(pady=10)
    
    def confirm_save():
        global selected_file, selected_audio_file
        try:
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            
            # Create class folder if not exists
            class_folder = DATA_DIR / selected_label
            if not os.path.exists(class_folder): os.makedirs(class_folder)

            # --- MODIFIED: ONLY SAVE IMAGE IF IT EXISTS (It won't in Audio Only mode) ---
            if selected_file and os.path.exists(selected_file):
                processed_path, _ = camera_prepro(selected_file, threaded=False)
                features = camera_features(processed_path)
                save_features_to_csv(features, csv_path, selected_label)
                
                final_img_name = f"{selected_label}_{timestamp}.jpg"
                dst_img_path = class_folder / final_img_name
                shutil.move(processed_path, dst_img_path)
                if os.path.exists(selected_file): os.remove(selected_file)
                print(f"[INFO] Image saved.")

            # 2. Save Audio
            if selected_audio_file and os.path.exists(selected_audio_file):
                final_audio_name = f"{selected_label}_{timestamp}.wav"
                dst_audio_path = class_folder / final_audio_name
                shutil.move(selected_audio_file, dst_audio_path)
                print(f"[INFO] Audio saved: {final_audio_name}")
            
            switch_page("data_collection4")
            
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")
            print(e)

    btn_row = ctk.CTkFrame(frame, fg_color="transparent")
    btn_row.pack(pady=30)
    
    make_button(btn_row, "SAVE AUDIO", confirm_save, width=150).pack(side="left", padx=20)
    make_button(btn_row, "DISCARD", lambda: switch_page("data_collection4"), width=150, fg_color="#C62828").pack(side="left", padx=20)

    return frame
pages["data_collection3"] = load_data_collection_page_3

# ---- DATA COLLECTION 4 (Loop) ----
def load_data_collection_page_4():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    make_label(frame, "Data Saved Successfully!", font=FONT_TITLE).pack(pady=(60, 20))
    make_label(frame, "Collect more data?", font=FONT_SUB).pack(pady=20)
    
    make_button(frame, "YES (New Audio)", lambda: switch_page("data_collection2"), width=250).pack(pady=10)
    make_button(frame, "Main Menu", lambda: switch_page("main"), width=250, fg_color="#555").pack(pady=10)
    return frame
pages["data_collection4"] = load_data_collection_page_4

# ---- DATA DETECTION 1 ----
# Used for actual testing/detection, still requires camera.
def load_data_detection_page_1():
    global video_capture, picam, camera_after, selected_file
    
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    make_label(frame, "Detection (Camera Capture)", font=FONT_SUB).grid(row=0, pady=10)
    
    cam_card = ctk.CTkFrame(frame, fg_color=CARD)
    cam_card.grid(row=1, padx=20, pady=10, sticky="nsew")
    
    cam_canvas = Canvas(cam_card, bg="#E8E5DA", highlightthickness=0)
    cam_canvas.pack(fill="both", expand=True)
    
    selected_file = None
    stop_camera()

    def update_loop():
        global camera_after
        frame_rgb = None
        if USE_PICAMERA2 and picam:
             try:
                 arr = picam.capture_array()
                 frame_rgb = cv2.rotate(arr, cv2.ROTATE_180)
             except: pass
        elif video_capture:
             ret, arr = video_capture.read()
             if ret: frame_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        
        if frame_rgb is not None:
             w, h = cam_canvas.winfo_width(), cam_canvas.winfo_height()
             if w > 10 and h > 10:
                 img = Image.fromarray(cv2.resize(frame_rgb, (w, h)))
                 imgtk = ImageTk.PhotoImage(image=img)
                 cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                 cam_canvas.image = imgtk
        camera_after = root.after(30, update_loop)

    if USE_PICAMERA2:
        picam = Picamera2()
        picam.configure(picam.create_preview_configuration(main={"size": (640, 480)}))
        picam.start()
    else:
        video_capture = cv2.VideoCapture(0)
    
    update_loop()

    def capture_next():
        global selected_file
        ts = time.strftime('%Y%m%d-%H%M%S')
        fname = f"tmp_detect_{ts}.jpg"
        fpath = folder_path1 / fname
        frame_bgr = None
        if USE_PICAMERA2:
            arr = picam.capture_array()
            frame_bgr = cv2.cvtColor(cv2.rotate(arr, cv2.ROTATE_180), cv2.COLOR_RGB2BGR)
        elif video_capture:
            ret, arr = video_capture.read()
            if ret: frame_bgr = arr
        if frame_bgr is not None:
            cv2.imwrite(str(fpath), frame_bgr)
            selected_file = str(fpath)
            stop_camera()
            switch_page("data_detection2")

    ctrls = ctk.CTkFrame(frame, fg_color="transparent")
    ctrls.grid(row=2, pady=20)
    make_button(ctrls, "CAPTURE", capture_next, width=200).pack(side="left", padx=10)
    make_button(ctrls, "BACK", lambda: [stop_camera(), switch_page("main")]).pack(side="left", padx=10)
    
    return frame

pages["data_detection1"] = load_data_detection_page_1

# ---- Data Detection Page 2 (Show captured image) ----
def load_data_detection_page_2():
    global selected_file
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    
    make_label(frame, "Review Capture", font=FONT_SUB).grid(row=0, pady=10)
    
    try:
        img = Image.open(selected_file)
        img.thumbnail((500, 300))
        pk = ImageTk.PhotoImage(img)
        lbl = ctk.CTkLabel(frame, image=pk, text="")
        lbl.image = pk
        lbl.grid(row=1)
    except:
        make_label(frame, "Error loading image").grid(row=1)

    btn_row = ctk.CTkFrame(frame, fg_color="transparent")
    btn_row.grid(row=2, pady=20)
    make_button(btn_row, "RETAKE", lambda: switch_page("data_detection1")).pack(side="left", padx=10)
    make_button(btn_row, "ANALYZE", lambda: (camera_prepro(selected_file), switch_page("data_detection3"))).pack(side="left", padx=10)
    
    return frame
pages["data_detection2"] = load_data_detection_page_2

# ---- Data Detection Page 3 (Processing) ----
def load_data_detection_page_3():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    make_label(frame, "Processing Image...", font=FONT_TITLE).pack(expand=True)

    def on_detection_done(processed_path, result):
        global hsv_class
        hsv_class = result
        if result is not None:
            switch_page("data_detection5")
        else:
            messagebox.showerror("Failed", "Classification failed")
            switch_page("data_detection1")

    if selected_file:
        camera_prepro(selected_file, threaded=True, on_complete=on_detection_done)
    
    return frame
pages["data_detection3"] = load_data_detection_page_3

# ---- Data Detection Page 5 (Summary) ----
def load_data_detection_page_5():
    global selected_file, hsv_class
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_rowconfigure(0, weight=1)

    # Left: Image
    img_frame = ctk.CTkFrame(frame, fg_color="transparent")
    img_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    try:
        img = Image.open(selected_file)
        img.thumbnail((350, 350))
        pk = ImageTk.PhotoImage(img)
        lbl = ctk.CTkLabel(img_frame, image=pk, text="")
        lbl.image = pk
        lbl.pack(expand=True)
    except: pass

    # Right: Results
    res_frame = ctk.CTkFrame(frame, fg_color="transparent")
    res_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
    
    make_label(res_frame, "Classification Results", font=FONT_SUB).pack(pady=10)
    
    final_class = max(hsv_class, key=hsv_class.get) if hsv_class else "Unknown"
    make_label(res_frame, final_class.upper(), font=("Arial", 40, "bold"), text_color="#D35400").pack(pady=20)
    
    if hsv_class:
        for k, v in hsv_class.items():
            make_label(res_frame, f"{k}: {v*100:.1f}%", font=FONT_NORMAL).pack(anchor="w", padx=40)

    make_button(res_frame, "MAIN MENU", lambda: switch_page("main"), width=200).pack(side="bottom", pady=20)
    
    return frame
pages["data_detection5"] = load_data_detection_page_5


# --- APP START ---
switch_page("main")

def on_key(event):
    if event.keysym == "Escape":
        root.attributes("-fullscreen", False)
root.bind("<Key>", on_key)

def on_close():
    stop_camera()
    GPIO.cleanup()
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()