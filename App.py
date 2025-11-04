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
from joblib import load
from PIL import Image, ImageTk

# Globals
video_capture = None
picam = None
camera_after = None

selected_file = None
selected_label = None

processed_file = None
hsv_class = None

USE_PICAMERA2 = False

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
if getattr(sys, 'frozen', False):  # Running as EXE
    base_dir = Path(sys.executable).parent
else:
    base_dir = Path(__file__).parent

csv_path = base_dir / "coconut_features.csv"
folder_path1 = base_dir / "Data Collection Captures"
folder_path2 = base_dir / "Data Detection Captures"
os.makedirs(folder_path1, exist_ok=True)
os.makedirs(folder_path2, exist_ok=True)


# Theme Helpers
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

FONT_TITLE = ("Aerial", 46, "bold")
FONT_SUB = ("Aerial", 36, "bold")
FONT_NORMAL = ("Aerial", 18)


# Window setup
root = ctk.CTk()
root.title("COCONUTZ - Coconut Type Classifier")

# Force the same 5-inch LCD size (landscape 800x480) on PC, fullscreen on Pi
_machine = platform.machine().lower()
if ("arm" in _machine) or ("raspberry" in _machine):
    root.attributes("-fullscreen", True)
else:
    root.geometry("800x480")
    root.resizable(False, False)

root.configure(fg_color=BG)
root.update_idletasks()

# Main container where pages live (use grid inside)
main_container = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
main_container.pack(fill="both", expand=True)
# make main_container act like a grid parent
main_container.grid_rowconfigure(0, weight=1)
main_container.grid_columnconfigure(0, weight=1)

current_page_frame = None

#  UI Helpers
def make_button(master, text, command, width=None, height=None, font=FONT_NORMAL):
    btn = ctk.CTkButton(
        master,
        text=text,
        command=command,
        fg_color=BTN,
        hover_color=BTN_HOVER,
        text_color="white",
        corner_radius=10,
        font=font
    )
    if width or height:
        btn.configure(width=width, height=height)
    return btn

def make_label(master, text, font=FONT_NORMAL, anchor="center"):
    lbl = ctk.CTkLabel(master, text=text, font=font, text_color=TEXT, anchor=anchor)
    return lbl

def make_textbox(master, text, font=FONT_NORMAL):
    tb = ctk.CTkTextbox(master, width=10, height=2, corner_radius=8)
    tb.insert("0.0", text)
    tb.configure(state="disabled")
    return tb


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
            except Exception:
                pass
            picam = None
    except Exception:
        picam = None


# Camera preprocessing and Features
def camera_features(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    return [h_mean, s_mean, v_mean]


def camera_prepro(image_path, threaded=False, on_complete=None):
    """
    Preprocesses an image and classifies it.
    - If threaded=True, runs asynchronously and calls on_complete(processed_file, hsv_class)
    - If threaded=False, runs synchronously and returns (processed_file, hsv_class)
    """

    def process_task():
        global hsv_class
        processed_file = None
        hsv_class = None

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Unreadable image")

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([5, 30, 30])
            upper = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            masked_img = cv2.bitwise_and(img, img, mask=mask)
            resized_img = cv2.resize(masked_img, (224, 224))

            processed_file = os.path.splitext(image_path)[0] + "_processed.jpg"
            cv2.imwrite(processed_file, resized_img)

            features = camera_features(processed_file)

            if model is not None:
                try:
                    x = np.array([features])
                    probs = model.predict_proba(x)[0]
                    class_labels = model.classes_
                    hsv_class = {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}
                except Exception as e:
                    print("[ERROR] Model prediction failed:", e)
                    hsv_class = None
            else:
                print("[INFO] No model — only preprocessing done.")
                hsv_class = None

        except Exception as e:
            print(f"[ERROR] Preprocessing/classification failed: {e}")
            processed_file, hsv_class = None, None

        # Handle completion
        if threaded and callable(on_complete):
            root.after(0, lambda: on_complete(processed_file, hsv_class))
        elif not threaded:
            return processed_file, hsv_class

    # Threaded or synchronous execution
    if threaded:
        threading.Thread(target=process_task, daemon=True).start()
        return None, None
    else:
        return process_task()



def save_features_to_csv(features, filepath, label):
    df = pd.DataFrame(
        [[label] + features],
        columns=["Label", "H_mean", "S_mean", "V_mean"]
    )
    df.to_csv(filepath, mode="a", header=not Path(filepath).exists(), index=False)


# Page manager
pages = {}

def clear_current_page():
    global current_page_frame
    try:
        stop_camera()
    except Exception:
        pass
    if current_page_frame is not None:
        current_page_frame.destroy()
        current_page_frame = None

def switch_page(page_name):
    global current_page_frame
    clear_current_page()
    if page_name in pages:
        current_page_frame = pages[page_name]()
        # place the page into the main_container grid (row 0 col 0)
        current_page_frame.grid(row=0, column=0, sticky="nsew")
    else:
        print(f"[WARN] Unknown page: {page_name}")

# ------------------------
# UI Page implementations
# ------------------------

# ---- Main Page ----
def load_main_page():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid(row=0, column=0, sticky="nsew")

    frame.grid_rowconfigure(0, weight=2)   
    frame.grid_rowconfigure(1, weight=6)   
    frame.grid_columnconfigure(0, weight=1)

    title_label = make_label(frame, "COCONUTZ\n Coconut Maturity Classifier", font=FONT_TITLE)
    title_label.grid(row=0, column=0, pady=(60, 30), sticky="n")

    button_frame = ctk.CTkFrame(frame, fg_color="transparent")
    button_frame.grid(row=1, column=0, sticky="n")
    button_frame.grid_columnconfigure(0, weight=1)

    btn_pad = 15

    btn_data_collection = make_button(button_frame, "Data Collection", lambda: switch_page("data_collection1")
)
    btn_data_collection.grid(row=0, column=0, pady=btn_pad, padx=8, ipadx=30, ipady=10, sticky="ew")

    btn_data_detection = make_button(button_frame, "Data Detection", lambda: switch_page("data_detection1")
)
    btn_data_detection.grid(row=1, column=0, pady=btn_pad, padx=8, ipadx=30, ipady=10, sticky="ew")

    btn_exit = make_button(button_frame, "Exit", root.destroy)
    btn_exit.grid(row=2, column=0, pady=(btn_pad, 40), padx=8, sticky="ew")

    button_frame.grid_propagate(False)

    return frame


pages["main"] = load_main_page

# ---- Data Collection Page 1 (Camera) ----
def load_data_collection_page_1():
    global video_capture, picam, camera_after, selected_file

    frame = ctk.CTkFrame(main_container, fg_color=BG)
    # header, camera area (large), controls
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=4)  # give camera larger space
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)

    header = make_label(frame, "Data Collection (Camera)", font=FONT_SUB)
    header.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=6)

    # Camera card (full width)
    cam_card = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=12)
    cam_card.grid(row=1, column=0, columnspan=2, padx=12, pady=8, sticky="nsew")
    cam_card.grid_rowconfigure(0, weight=1)
    cam_card.grid_columnconfigure(0, weight=1)

    # Use a tkinter Canvas for the video preview inside CTkFrame
    cam_canvas = Canvas(cam_card, bg="#E8E5DA", highlightthickness=0)
    cam_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    selected_file = None
    stop_camera()

    # --- camera setup
    if USE_PICAMERA2:
        print("Using PiCamera2 (Raspberry Pi Camera)")
        picam = PiCamera2()
        picam.set_controls({"Rotation": 180})
        preview_config = picam.create_preview_configuration(main={"size": (640, 480)})
        picam.configure(preview_config)
        picam.start()

        def update_frame_picam():
            global camera_after
            try:
                frame_arr = picam.capture_array()
                frame_rgb = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2RGB)
                w = cam_canvas.winfo_width() or 560
                h = cam_canvas.winfo_height() or 280
                frame_resized = cv2.resize(frame_rgb, (w, h))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                cam_canvas.image = imgtk
            except Exception:
                stop_camera()
                return
            camera_after = root.after(10, update_frame_picam)

        def capture_and_next():
            global selected_file
            try:
                frame_arr = picam.capture_array()
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                tmp_name = f"tmp_capture_{timestamp}.jpg"
                tmp_path = folder_path1 / tmp_name
                cv2.imwrite(str(tmp_path), cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR))
                selected_file = str(tmp_path)
                stop_camera()
                switch_page("data_collection2")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to Capture: {e}")

        update_frame_picam()

    else:
        print("Using OpenCV VideoCapture (Webcam)")
        video_capture = cv2.VideoCapture(0)

        def update_frame_cam():
            global camera_after
            try:
                if video_capture is None:
                    return
                ret, frame_arr = video_capture.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2RGB)
                    w = cam_canvas.winfo_width() or 560
                    h = cam_canvas.winfo_height() or 280
                    # protect against zero-size
                    if w <= 0: w = 560
                    if h <= 0: h = 280
                    frame_resized = cv2.resize(frame_rgb, (w, h))
                    img = Image.fromarray(frame_resized)
                    imgtk = ImageTk.PhotoImage(image=img)
                    cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                    cam_canvas.image = imgtk
            except Exception:
                stop_camera()
                return
            camera_after = root.after(30, update_frame_cam)

        def capture_and_next():
            global selected_file
            try:
                if video_capture is None:
                    messagebox.showerror("Camera Error", "Camera not initialized.")
                    return
                ret, frame_arr = video_capture.read()
                if ret:
                    timestamp = time.strftime('%Y%m%d-%H%M%S')
                    tmp_name = f"tmp_capture_{timestamp}.jpg"
                    tmp_path = folder_path1 / tmp_name
                    cv2.imwrite(str(tmp_path), frame_arr)
                    selected_file = str(tmp_path)
                    stop_camera()
                    switch_page("data_collection2")
                else:
                    messagebox.showerror("Capture Error", "Failed to read frame from camera.")
            except Exception as e:
                messagebox.showerror("Capture Error", f"Failed to Capture: {e}")

        update_frame_cam()

    # Controls (centered)
    controls = ctk.CTkFrame(frame, fg_color=BG, corner_radius=0)
    controls.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(6,12), padx=8)
    controls.grid_columnconfigure(0, weight=1)
    controls.grid_columnconfigure(1, weight=1)

    capture_btn = make_button(controls, "Feature Extraction", capture_and_next)
    capture_btn.grid(row=0, column=0, padx=6, ipadx=20, ipady=8, sticky="e")

    back_btn = make_button(controls, "Main Menu", lambda: switch_page("main"))
    back_btn.grid(row=0, column=1, padx=6, ipadx=8, ipady=8, sticky="w")

    return frame

pages["data_collection1"] = load_data_collection_page_1

# ---- Data Collection Page 2 (Label selection) ----
def load_data_collection_page_2():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_rowconfigure(3, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    header = make_label(frame, "What is the classification of the\nCoconut?", font=FONT_SUB)
    header.grid(row=0, column=0, sticky="nsew", pady=6)

    def set_label_and_next(label):
        global selected_label
        selected_label = label
        switch_page("data_collection3")

    # Buttons column (centered)
    btn_frame = ctk.CTkFrame(frame, fg_color=BG)
    btn_frame.grid(row=1, column=0, pady=6, padx=8, sticky="nsew")
    btn_frame.grid_columnconfigure(0, weight=1)

    malauhog_btn = make_button(btn_frame, "Malauhog", lambda: set_label_and_next("malauhog"))
    malauhog_btn.grid(row=0, column=0, padx=8, pady=15, ipadx=20, ipady=8, sticky="n")

    malakanin_btn = make_button(btn_frame, "Malakanin", lambda: set_label_and_next("malakanin"))
    malakanin_btn.grid(row=1, column=0, padx=8, pady=15, ipadx=20, ipady=8, sticky="n")

    malakatad_btn = make_button(btn_frame, "Malakatad", lambda: set_label_and_next("malakatad"))
    malakatad_btn.grid(row=2, column=0, padx=8, pady=15, ipadx=20, ipady=8, sticky="n")

    back_btn = make_button(frame, "Back", lambda: switch_page("data_collection1"))
    back_btn.grid(row=3, column=0, pady=(8, 12), ipadx=8, ipady=8, sticky="n")

    return frame

pages["data_collection2"] = load_data_collection_page_2

# ---- Data Collection Page 3 (Confirm) ----
def load_data_collection_page_3():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=2)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_columnconfigure(2, weight=1)

    header = make_label(frame, "Are you sure this is\nthe right classification?", font=FONT_TITLE)
    header.grid(row=0, column=1, sticky="nsew", pady=8)

    def confirm_yes():
        global selected_file, selected_label

        if not selected_file or not selected_label:
            messagebox.showwarning("Missing Info", "No image or label selected.")
            return

        try:
            # 1️⃣ Run preprocessing (synchronously)
            processed_path, _ = camera_prepro(selected_file, threaded=False)
            if not processed_path or not Path(processed_path).exists():
             raise ValueError("Processing failed — no output image created.")

            # 2️⃣ Extract features from processed image and save
            features = camera_features(processed_path)
            if features is not None:
                save_features_to_csv(features, csv_path, selected_label)

            # 3️⃣ Rename/move the processed image (final only)
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            dst_name = f"{selected_label}_{timestamp}{Path(processed_path).suffix}"
            dst_path = folder_path1 / dst_name
            Path(processed_path).replace(dst_path)

            # 4️⃣ Delete the original raw image
            orig = Path(selected_file)
            if orig.exists() and orig != dst_path:
                orig.unlink(missing_ok=True)

            # 5️⃣ Cleanup any leftover *_processed.jpg files
            for f in folder_path1.glob("*_processed.jpg"):
                if f.name != dst_path.name:
                    f.unlink(missing_ok=True)

            selected_file = str(dst_path)
            switch_page("data_collection4")

        except Exception as e:
            print(f"[ERROR] Collection confirm failed: {e}")
            messagebox.showerror("Error", f"Something went wrong: {e}")

    yes = make_button(frame, "Yes", confirm_yes)
    yes.grid(row=1, column=1, padx=10, ipadx=25, ipady=12, sticky="w")

    no = make_button(frame, "No", lambda: switch_page("data_collection2"))
    no.grid(row=1, column=1, padx=10, ipadx=25, ipady=12, sticky="e")

    return frame


pages["data_collection3"] = load_data_collection_page_3

# ---- Data Collection Page 4 (More?) ----
def load_data_collection_page_4():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=2)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_columnconfigure(2, weight=1)

    header = make_label(frame, "Would you like to\ncollect more data?", font=FONT_TITLE)
    header.grid(row=0, column=1, sticky="nsew", pady=8)

    yes = make_button(frame, "Yes", lambda: switch_page("data_collection1"))
    yes.grid(row=1, column=1, pady=6, ipadx=30, ipady=12, sticky="w")

    mm_btn = make_button(frame, "Main Menu", lambda: switch_page("main"))
    mm_btn.grid(row=1, column=1, pady=(6,12), ipadx=20, ipady=10, sticky="e")

    return frame

pages["data_collection4"] = load_data_collection_page_4

# ---- Data Detection Page 1 (Capture) ----
def load_data_detection_page_1():
    global video_capture, picam, camera_after, selected_file

    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=4)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)

    header = make_label(frame, "Image Capture", font=FONT_SUB)
    header.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=6)

    cam_card = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=12)
    cam_card.grid(row=1, column=0, columnspan=2, padx=12, pady=8, sticky="nsew")
    cam_card.grid_rowconfigure(0, weight=1)
    cam_card.grid_columnconfigure(0, weight=1)

    cam_canvas = Canvas(cam_card, bg="#5A56C8", highlightthickness=0)
    cam_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    selected_file = None
    stop_camera()

    if USE_PICAMERA2:
        print("Using PiCamera2 (Raspberry Pi Camera)")
        picam = PiCamera2()
        picam.set_controls({"Rotation": 180})
        preview_config = picam.create_preview_configuration(main={"size": (640, 480)})
        picam.configure(preview_config)
        picam.start()

        def update_frame_picam():
            global camera_after
            try:
                frame_arr = picam.capture_array()
                frame_rgb = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2RGB)
                w = cam_canvas.winfo_width() or 560
                h = cam_canvas.winfo_height() or 280
                frame_resized = cv2.resize(frame_rgb, (w, h))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                cam_canvas.image = imgtk
            except Exception:
                stop_camera()
                return
            camera_after = root.after(10, update_frame_picam)

        def capture_and_next():
            global selected_file
            try:
                frame_arr = picam.capture_array()
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                tmp_name = f"tmp_capture_{timestamp}.jpg"
                tmp_path = folder_path1 / tmp_name
                cv2.imwrite(str(tmp_path), cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR))
                selected_file = str(tmp_path)
                stop_camera()
                switch_page("data_detection2")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to Capture: {e}")

        update_frame_picam()

    else:
        print("Using OpenCV VideoCapture (Webcam)")
        video_capture = cv2.VideoCapture(0)

        def update_frame_cam():
            global camera_after
            try:
                if video_capture is None:
                    return
                ret, frame_arr = video_capture.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_arr, cv2.COLOR_BGR2RGB)
                    w = cam_canvas.winfo_width() or 560
                    h = cam_canvas.winfo_height() or 280
                    if w <= 0: w = 560
                    if h <= 0: h = 280
                    frame_resized = cv2.resize(frame_rgb, (w, h))
                    img = Image.fromarray(frame_resized)
                    imgtk = ImageTk.PhotoImage(image=img)
                    cam_canvas.create_image(0, 0, image=imgtk, anchor="nw")
                    cam_canvas.image = imgtk
            except Exception:
                stop_camera()
                return
            camera_after = root.after(30, update_frame_cam)

        def capture_and_next():
            global selected_file
            try:
                if video_capture is None:
                    messagebox.showerror("Camera Error", "Camera not initialized.")
                    return
                ret, frame_arr = video_capture.read()
                if ret:
                    timestamp = time.strftime('%Y%m%d-%H%M%S')
                    tmp_name = f"tmp_capture_{timestamp}.jpg"
                    tmp_path = folder_path1 / tmp_name
                    cv2.imwrite(str(tmp_path), frame_arr)
                    selected_file = str(tmp_path)
                    stop_camera()
                    switch_page("data_detection2")
                else:
                    messagebox.showerror("Capture Error", "Failed to read frame from camera.")
            except Exception as e:
                messagebox.showerror("Capture Error", f"Failed to Capture: {e}")

        update_frame_cam()

    controls = ctk.CTkFrame(frame, fg_color=BG)
    controls.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(6,12), padx=8)
    controls.grid_columnconfigure(0, weight=1)
    controls.grid_columnconfigure(1, weight=1)

    capture_btn = make_button(controls, "Feature Extraction", capture_and_next)
    capture_btn.grid(row=0, column=0, padx=6, ipadx=20, ipady=8, sticky="e")

    back_btn = make_button(controls, "Main Menu", lambda: switch_page("main"))
    back_btn.grid(row=0, column=1, padx=6, ipadx=8, ipady=8, sticky="w")

    return frame

pages["data_detection1"] = load_data_detection_page_1

# ---- Data Detection Page 2 (Show captured image) ----
def load_data_detection_page_2():
    global selected_file

    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=4)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    header = make_label(frame, "Captured Image", font=FONT_SUB)
    header.grid(row=0, column=0, sticky="nsew", pady=6)

    if not selected_file or not os.path.exists(selected_file):
        make_label(frame, "No image captured!", font=FONT_NORMAL).grid(row=1, column=0, pady=10)
    else:
        try:
            canvas_card = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=12)
            canvas_card.grid(row=1, column=0, padx=12, pady=8, sticky="nsew")
            canvas_card.grid_rowconfigure(0, weight=1)
            canvas_card.grid_columnconfigure(0, weight=1)

            tk_canvas = Canvas(canvas_card, bg="#5A56C8", highlightthickness=0)
            tk_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

            def update_preview():
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
                tk_canvas.after(200, update_preview)

            update_preview()

        except Exception as e:
            make_label(frame, f"Error loading image: {e}", font=FONT_NORMAL).grid(row=1, column=0, pady=10)

    btn_row = ctk.CTkFrame(frame, fg_color=BG)
    btn_row.grid(row=2, column=0, pady=(6,12), sticky="nsew")
    btn_row.grid_columnconfigure((0,1), weight=1)

    make_button(btn_row, "Try Again", lambda: switch_page("data_detection1")).grid(row=0, column=0, padx=8, ipadx=16, ipady=6, sticky="e")
    make_button(btn_row, "Proceed", lambda: (camera_prepro(selected_file), switch_page("data_detection3"))).grid(row=0, column=1, padx=8, ipadx=16, ipady=6, sticky="w")

    return frame

pages["data_detection2"] = load_data_detection_page_2

# ---- Data Detection Page 3 (Processing) ----
def load_data_detection_page_3():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    status = make_label(frame, "Processing Image...", font=FONT_SUB)
    status.grid(row=0, column=0, sticky="nsew", pady=8)

    def on_detection_done(processed_path, result):
        global hsv_class
        hsv_class = result
        if result is not None:
            switch_page("data_detection4")
        else:
            messagebox.showerror("Detection Failed", "Could not process or classify the image.")
            switch_page("data_detection1")  # or wherever you want to return

    # Start threaded preprocessing and classification
    if selected_file:
        camera_prepro(selected_file, threaded=True, on_complete=on_detection_done)
    else:
        messagebox.showwarning("No Image", "Please select an image before detection.")
        switch_page("data_detection1")

    return frame

pages["data_detection3"] = load_data_detection_page_3


pages["data_detection3"] = load_data_detection_page_3

# ---- Data Detection Page 4 (Audio capture placeholder) ----
def load_data_detection_page_4():
    frame = ctk.CTkFrame(main_container, fg_color=BG)
    frame.grid_rowconfigure(0, weight=2)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_columnconfigure(2, weight=1)

    header = make_label(frame, "Audio Data Capture", font=FONT_TITLE)
    header.grid(row=0, column=1, sticky="nsew", pady=8)

    next_btn = make_button(frame, "Activate Solenoid", lambda: switch_page("data_detection5"))
    next_btn.grid(row=1, column=1, pady=6, ipadx=30, ipady=12, sticky="w")

    back_btn = make_button(frame, "Back", lambda: switch_page("main"))
    back_btn.grid(row=1, column=1, pady=(6,12), ipadx=30, ipady=10, sticky="e")

    return frame

pages["data_detection4"] = load_data_detection_page_4

# ---- Data Detection Page 5 (Summary) ----
def load_data_detection_page_5():
    global selected_file, hsv_class

    frame = ctk.CTkFrame(main_container, fg_color="#E8E5DA")
    frame.grid(row=0, column=0, sticky="nsew")
    main_container.grid_rowconfigure(0, weight=1)
    main_container.grid_columnconfigure(0, weight=1)

    # ---- Responsive Layout ----
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1, uniform="col")  # Left: Image
    frame.grid_columnconfigure(1, weight=1, uniform="col")  # Right: Info

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

    final_class = max(hsv_class, key=hsv_class.get) if hsv_class else "N/A"
    make_label(info_frame, f"Final Class: {final_class}", font=("Aerial", 20, "bold")).grid(
        row=1, column=0, sticky="n", pady=(5, 10)
    )

    # Camera/Audio section headers
    header_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
    header_frame.grid(row=2, column=0, pady=(5, 5))
    header_frame.grid_columnconfigure((0, 1), weight=1, uniform="headers")

    make_label(header_frame, "Camera Classification", font=FONT_NORMAL).grid(
        row=0, column=0, sticky="ew"
    )
    make_label(header_frame, "Audio Classification", font=FONT_NORMAL).grid(
        row=0, column=1, sticky="ew"
    )

    # Values for camera/audio
    classes = ["malauhog", "malakanin", "malakatad"]
    for i, cls in enumerate(classes):
        row_index = 3 + i
        cam_val = hsv_class.get(cls, 0.0) if hsv_class else 0.0
        aud_val = 0.0
        val_frame = ctk.CTkFrame(info_frame, fg_color="transparent")
        val_frame.grid(row=row_index, column=0, sticky="ew", pady=2)
        val_frame.grid_columnconfigure((0, 1), weight=1, uniform="vals")

        make_label(val_frame, f"{cls}: {cam_val:.2f}", font=FONT_NORMAL).grid(
            row=0, column=0, sticky="w", padx=10
        )
        make_label(val_frame, f"{cls}: {aud_val:.2f}", font=FONT_NORMAL).grid(
            row=0, column=1, sticky="w", padx=10
        )

    # ==========================
    # SAVE IMAGE + BACK BUTTON
    # ==========================
    data_det_dir = Path("Data Detection Captures")
    data_det_dir.mkdir(exist_ok=True)
    try:
        save_path = data_det_dir / f"{final_class}_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        import shutil
        shutil.copy(selected_file, save_path)
        print(f"[INFO] Final classified image saved: {save_path}")

        tmp_dir = Path(selected_file).parent
        for tmp_file in tmp_dir.glob("tmp_capture_*.jpg"):
            try:
                tmp_file.unlink()
                print(f"[CLEANUP] Deleted temp file: {tmp_file}")
            except Exception as e:
                print(f"[WARN] Could not delete {tmp_file}: {e}")

        if Path(selected_file) != save_path and Path(selected_file).exists():
            try:
                Path(selected_file).unlink()
                print(f"[CLEANUP] Deleted original capture: {selected_file}")
            except Exception as e:
                print(f"[WARN] Could not delete original capture: {e}")

        selected_file = str(save_path)

    except Exception as e:
        print(f"[WARN] Failed to save classified image: {e}")

    # ---- Bottom-aligned button ----
    make_button(info_frame, "Back to Main", lambda: switch_page("main")).grid(
        row=7, column=0, pady=(20, 10), sticky="se", padx=20
    )

    return frame

pages["data_detection5"] = load_data_detection_page_5

# ----------------------------
# App start
# ----------------------------
switch_page("main")

# Add a keyboard shortcut to exit fullscreen on Pi (Esc)
def on_key(event):
    if event.keysym == "Escape":
        try:
            # toggle fullscreen off
            root.attributes("-fullscreen", False)
        except Exception:
            pass

root.bind("<Key>", on_key)

root.mainloop()
