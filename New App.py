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

# ---------------------------------------------------------
# GLOBALS & STATE
# ---------------------------------------------------------
video_capture = None
picam = None
camera_after = None

selected_file = None

# We use this dictionary to store full results for the Fuzzy Logic
processing_results = {
    "image_probs": None,  # {'malauhog': 0.1, ...}
    "image_score": 0.0,   # 0-100
    "image_class": None,  # "malakanin"
    "audio_probs": None,  # (Placeholder for now)
    "audio_score": 0.0,
    "audio_class": None
}

USE_PICAMERA2 = False

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
try:
    scaler = joblib.load("camera_scaler.pkl")
    model = load("camera_model.pkl")
except Exception as e:
    print(f"[WARN] Model/Scaler not loaded: {e}")
    scaler = None
    model = None

try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except Exception as e:
    print("PiCamera2 not available:", e)
    USE_PICAMERA2 = False


# ---------------------------------------------------------
# PATHS & THEME
# ---------------------------------------------------------
if getattr(sys, 'frozen', False):  # Running as EXE
    base_dir = Path(sys.executable).parent
else:
    base_dir = Path(__file__).parent

csv_path = base_dir / "coconut_features.csv"
folder_path1 = base_dir / "Data Collection Captures"
folder_path2 = base_dir / "Data Detection Captures"
os.makedirs(folder_path1, exist_ok=True)
os.makedirs(folder_path2, exist_ok=True)

ctk.set_appearance_mode("Light")
try:
    ctk.set_default_color_theme("theme.json")
except Exception:
    pass

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

# Force landscape 800x480 or fullscreen
_machine = platform.machine().lower()
if ("arm" in _machine) or ("raspberry" in _machine):
    root.attributes("-fullscreen", True)
else:
    root.geometry("800x480")
    root.resizable(False, False)

root.configure(fg_color=BG)
root.update_idletasks()

main_container = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
main_container.pack(fill="both", expand=True)
main_container.grid_rowconfigure(0, weight=1)
main_container.grid_columnconfigure(0, weight=1)

current_page_frame = None


# ---------------------------------------------------------
# HELPER FUNCTIONS (UI & LOGIC)
# ---------------------------------------------------------
def make_button(master, text, command, width=None, height=None, font=FONT_NORMAL):
    btn = ctk.CTkButton(
        master, text=text, command=command,
        fg_color=BTN, hover_color=BTN_HOVER,
        text_color="white", corner_radius=10, font=font
    )
    if width or height:
        btn.configure(width=width, height=height)
    return btn

def make_label(master, text, font=FONT_NORMAL, anchor="center"):
    return ctk.CTkLabel(master, text=text, font=font, text_color=TEXT, anchor=anchor)

def stop_camera():
    global video_capture, picam, camera_after
    try:
        if camera_after is not None:
            root.after_cancel(camera_after)
    except Exception:
        pass
    camera_after = None

    try:
        if video_capture is not None:
            if hasattr(video_capture, "isOpened") and video_capture.isOpened():
                video_capture.release()
            video_capture = None
    except Exception:
        video_capture = None

    try:
        if picam is not None:
            try:
                picam.stop()
                picam.close()
            except Exception:
                pass
            picam = None
    except Exception:
        picam = None

# --- NEW: MATURITY SCORE CALCULATION ---
def calculate_maturity_score(probs_dict):
    """
    Input: {'malauhog': 0.1, 'malakanin': 0.8, 'malakatad': 0.1}
    Formula: (P_hog * 0) + (P_kan * 50) + (P_tad * 100)
    """
    if not probs_dict:
        return 0.0
    
    p_hog = probs_dict.get('malauhog', 0.0)
    p_kan = probs_dict.get('malakanin', 0.0)
    p_tad = probs_dict.get('malakatad', 0.0)
    
    score = (p_hog * 0) + (p_kan * 50) + (p_tad * 100)
    return score

# --- NEW: FUZZY / WEIGHTED LOGIC ---
def apply_weighted_logic(img_score, img_class, aud_score, aud_class):
    """
    Applies the specific weighting rules:
    1. Agreement -> Weight 1.0 for both.
    2. Audio=Malakanin (Conflict) -> Audio Weight 0.3, Image Weight 1.0.
    3. Other Conflicts -> Audio Weight 0.5, Image Weight 0.5.
    """
    w_aud = 0.5 # Default
    w_img = 0.5 # Default
    
    # Rule 1: Agreement
    if img_class == aud_class:
        w_aud = 1.0
        w_img = 1.0
    
    # Rule 2: Audio is Malakanin (Unsure) but Image disagrees
    elif aud_class == "malakanin" and img_class != "malakanin":
        w_aud = 0.3
        w_img = 0.3
        
    # Rule 3: Other conflicts (Default 0.5 vs 0.5 is applied)
    else:
        w_aud = 0.5
        w_img = 0.5
        
    # Weighted Average Calculation
    final_score = ((aud_score * w_aud) + (img_score * w_img)) / (w_aud + w_img)
    
    # Determine Final Class Label
    if final_score < 33.33:
        final_label = "malauhog"
    elif final_score < 66.66:
        final_label = "malakanin"
    else:
        final_label = "malakatad"
        
    return final_score, final_label


# ---------------------------------------------------------
# FEATURE EXTRACTION & PROCESSING
# ---------------------------------------------------------
def compute_moments(pixel_array):
    if len(pixel_array) == 0:
        return 0.0, 0.0, 0.0, 0.0
    if len(np.unique(pixel_array)) == 1:
        return float(pixel_array[0]), 0.0, 0.0, 0.0
    m1 = np.mean(pixel_array)
    m2 = np.std(pixel_array)
    m3 = skew(pixel_array, bias=False)
    m4 = kurtosis(pixel_array, bias=False)
    m3 = 0.0 if np.isnan(m3) else m3
    m4 = 0.0 if np.isnan(m4) else m4
    return m1, m2, m3, m4

def camera_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return [0.0] * 108

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([1, 1, 1]) 
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower, upper)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return [0.0] * 108

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi_hsv = hsv[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]
    
    step_h = h // 3
    step_w = w // 3
    feature_vector = []

    for row in range(3):
        for col in range(3):
            y_start = row * step_h
            y_end = (row + 1) * step_h if row < 2 else h
            x_start = col * step_w
            x_end = (col + 1) * step_w if col < 2 else w
            
            tile_hsv = roi_hsv[y_start:y_end, x_start:x_end]
            tile_mask = roi_mask[y_start:y_end, x_start:x_end]
            
            h_c = tile_hsv[:, :, 0]
            s_c = tile_hsv[:, :, 1]
            v_c = tile_hsv[:, :, 2]
            
            valid_h = h_c[tile_mask > 0]
            valid_s = s_c[tile_mask > 0]
            valid_v = v_c[tile_mask > 0]
            
            feature_vector.extend(compute_moments(valid_h)) 
            feature_vector.extend(compute_moments(valid_s)) 
            feature_vector.extend(compute_moments(valid_v)) 

    return feature_vector

def camera_prepro(image_path, threaded=False, on_complete=None):
    def process_task():
        global processing_results
        
        # Reset results
        processed_file_local = None
        img_probs = None

        try:
            img = cv2.imread(image_path)
            if img is None: raise ValueError("Unreadable image")

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([5, 30, 30])
            upper = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = np.uint8(labels == largest_label) * 255
            else:
                mask = np.zeros_like(mask)

            mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            mask = mask_filled
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            masked_img = cv2.bitwise_and(img, img, mask=mask)
            resized_img = cv2.resize(masked_img, (224, 224))
            processed_file_local = os.path.splitext(image_path)[0] + "_processed.jpg"
            cv2.imwrite(processed_file_local, resized_img)

            # Features & Prediction
            features = camera_features(processed_file_local)

            if model is not None and scaler is not None:
                x = np.array([features])
                x_scaled = scaler.transform(x)
                probs = model.predict_proba(x_scaled)[0]
                
                # Map probabilities
                class_map = {0: "malauhog", 1: "malakanin", 2: "malakatad"}
                img_probs = {class_map[i]: float(probs[i]) for i in range(len(probs))}
                
                # --- UPDATE GLOBAL RESULTS ---
                processing_results["image_probs"] = img_probs
                # Calculate Image Score immediately (0-100)
                processing_results["image_score"] = calculate_maturity_score(img_probs)
                # Determine Image Class
                processing_results["image_class"] = max(img_probs, key=img_probs.get)
                
            else:
                print("[WARN] Model missing.")
                processing_results["image_probs"] = None

        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            processed_file_local = None

        if threaded and callable(on_complete):
            root.after(0, lambda: on_complete(processed_file_local, processing_results["image_probs"]))
        elif not threaded:
            return processed_file_local, processing_results["image_probs"]

    if threaded:
        threading.Thread(target=process_task, daemon=True).start()
        return None, None
    else:
        return process_task()


# ---------------------------------------------------------
# PAGE LOGIC
# ---------------------------------------------------------
pages = {}

def clear_current_page():
    global current_page_frame
    stop_camera()
    if current_page_frame is not None:
        current_page_frame.destroy()
        current_page_frame = None

def switch_page(page_name):
    global current_page_frame
    clear_current_page()
    if page_name in pages:
        current_page_frame = pages[page_name]()
        current_page_frame.grid(row=0, column=0, sticky="nsew")
    else:
        print(f"[WARN] Unknown page: {page_name}")


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


# ---- PAGE 2: CAPTURE ----
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
        if w < 10: w=560
        if h < 10: h=280
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
                path = folder_path1 / f"tmp_capture_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
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
                    path = folder_path1 / f"tmp_capture_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
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


# ---- PAGE 3: PREVIEW ----
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
    make_button(btns, "Process", lambda: switch_page("data_detection3")).pack(side="left", padx=10)
    return frame
pages["data_detection2"] = load_data_detection_page_2


# ---- PAGE 4: PROCESSING ----
def load_data_detection_page_3():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.pack(fill="both", expand=True)
    make_label(frame, "Processing...", font=FONT_SUB).place(relx=0.5, rely=0.5, anchor="center")

    def done(path, res):
        if res: switch_page("data_detection5")
        else: 
            messagebox.showerror("Error", "Processing failed")
            switch_page("data_detection1")

    if selected_file:
        camera_prepro(selected_file, threaded=True, on_complete=done)
    return frame
pages["data_detection3"] = load_data_detection_page_3


# ---- PAGE 5: SUMMARY & FUZZY LOGIC ----
def load_data_detection_page_5():
    global selected_file, processing_results

    frame = ctk.CTkFrame(main_container, fg_color="#E8E5DA")
    frame.grid(row=0, column=0, sticky="nsew")
    frame.grid_columnconfigure((0, 1), weight=1)

    # 1. SETUP AUDIO (Dummy Data for now)
    # We assume Audio is Malakanin (Unsure) to test the weighting logic
    dummy_audio_probs = {'malauhog': 0.1, 'malakanin': 0.8, 'malakatad': 0.1} 
    processing_results["audio_probs"] = dummy_audio_probs
    processing_results["audio_score"] = calculate_maturity_score(dummy_audio_probs)
    processing_results["audio_class"] = "malakanin" # Simulating conflict scenario

    # 2. RUN WEIGHTED LOGIC
    final_score, final_label = apply_weighted_logic(
        processing_results["image_score"], 
        processing_results["image_class"], 
        processing_results["audio_score"], 
        processing_results["audio_class"]
    )

    # UI: Left Image
    if selected_file:
        try:
            img = Image.open(selected_file)
            img.thumbnail((350, 250))
            imgtk = ImageTk.PhotoImage(img)
            lbl = ctk.CTkLabel(frame, image=imgtk, text="")
            lbl.image = imgtk
            lbl.grid(row=0, column=0, rowspan=4, padx=20)
        except: pass

    # UI: Right Info
    info = ctk.CTkFrame(frame, fg_color=BG)
    info.grid(row=0, column=1, sticky="nw", pady=20)

    make_label(info, "Classification Result", font=("Arial", 28, "bold")).pack(pady=10)
    make_label(info, f"FINAL DECISION: {final_label.upper()}", font=("Arial", 22, "bold"), text_color="red").pack(pady=5)
    make_label(info, f"Combined Score: {final_score:.2f} / 100", font=FONT_NORMAL).pack(pady=5)

    # Details Table
    details = ctk.CTkFrame(info, fg_color=CARD)
    details.pack(fill="x", pady=15, padx=10)
    
    # Header
    ctk.CTkLabel(details, text="Source", font=("Arial",14,"bold")).grid(row=0, col=0, padx=5)
    ctk.CTkLabel(details, text="Class", font=("Arial",14,"bold")).grid(row=0, col=1, padx=5)
    ctk.CTkLabel(details, text="Score", font=("Arial",14,"bold")).grid(row=0, col=2, padx=5)

    # Image Row
    ctk.CTkLabel(details, text="Camera").grid(row=1, col=0, pady=2)
    ctk.CTkLabel(details, text=processing_results["image_class"]).grid(row=1, col=1)
    ctk.CTkLabel(details, text=f"{processing_results['image_score']:.1f}").grid(row=1, col=2)

    # Audio Row
    ctk.CTkLabel(details, text="Audio").grid(row=2, col=0, pady=2)
    ctk.CTkLabel(details, text=processing_results["audio_class"]).grid(row=2, col=1)
    ctk.CTkLabel(details, text=f"{processing_results['audio_score']:.1f}").grid(row=2, col=2)

    make_button(info, "Main Menu", lambda: switch_page("main")).pack(side="bottom", pady=20)

    return frame
pages["data_detection5"] = load_data_detection_page_5

# ---------------------------------------------------------
# START APP
# ---------------------------------------------------------
switch_page("main")
root.bind("<Key>", lambda e: root.attributes("-fullscreen", False) if e.keysym == "Escape" else None)
root.mainloop()