import customtkinter as ctk
import pyaudio
import wave
import os
import time
import threading
import datetime

# --- HARDWARE SETUP (MOCK & REAL) ---
try:
    import RPi.GPIO as GPIO
    PLATFORM = "PI"
except (ImportError, RuntimeError):
    PLATFORM = "PC"
    # Mock GPIO class for Laptop testing
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
RECORD_SECONDS = 3  # Must match your training logic
DATA_DIR = "dataset"
CLASSES = ["Malauhog", "Malakanin", "Malakatad"]

class DataCollectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Coconut Data Collector")
        self.geometry("500x450")
        ctk.set_appearance_mode("Dark")

        # Setup Folders
        self.setup_directories()

        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SOLENOID_PIN, GPIO.OUT)

        # Variables
        self.selected_class = ctk.StringVar(value=CLASSES[0])
        self.is_recording = False

        self.setup_ui()

    def setup_directories(self):
        """Creates the dataset folder structure automatically"""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        for cls in CLASSES:
            path = os.path.join(DATA_DIR, cls)
            if not os.path.exists(path):
                os.makedirs(path)

    def setup_ui(self):
        # Title
        title = ctk.CTkLabel(self, text="Data Collection Mode", font=("Roboto", 22, "bold"))
        title.pack(pady=20)

        # Class Selection
        lbl_select = ctk.CTkLabel(self, text="Select True Category:", font=("Roboto", 14))
        lbl_select.pack(pady=(10, 5))

        # Segmented Button for quick switching
        self.seg_button = ctk.CTkSegmentedButton(self, values=CLASSES, 
                                                 variable=self.selected_class,
                                                 command=self.update_counter_label)
        self.seg_button.pack(pady=10)
        self.seg_button.set(CLASSES[0]) # Default

        # Record Button
        self.btn_record = ctk.CTkButton(self, text="START RECORDING", 
                                        command=self.start_recording_thread,
                                        width=200, height=60,
                                        fg_color="#D32F2F", hover_color="#B71C1C",
                                        font=("Roboto", 18, "bold"))
        self.btn_record.pack(pady=30)

        # Info Labels
        self.lbl_status = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.lbl_status.pack(pady=10)

        self.lbl_count = ctk.CTkLabel(self, text="Samples collected: 0")
        self.lbl_count.pack(pady=5)
        
        # Initialize counter
        self.update_counter_label(CLASSES[0])

    def update_counter_label(self, value):
        """Updates the label showing how many files are in the current folder"""
        folder_path = os.path.join(DATA_DIR, value)
        count = len([name for name in os.listdir(folder_path) if name.endswith(".wav")])
        self.lbl_count.configure(text=f"Current {value} samples: {count}")

    def start_recording_thread(self):
        if self.is_recording: return
        
        self.is_recording = True
        self.btn_record.configure(state="disabled", text="RECORDING...", fg_color="#555555")
        
        thread = threading.Thread(target=self.record_process)
        thread.start()

    def record_process(self):
        try:
            current_label = self.selected_class.get()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{current_label}_{timestamp}.wav"
            filepath = os.path.join(DATA_DIR, current_label, filename)

            # Audio Stream Setup
            p = pyaudio.PyAudio()
            # Note: Ensure Input Device Index is correct on Raspi if needed
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=CHUNK_SIZE)

            self.update_status(f"Recording {current_label}...")
            
            frames = []
            start_time = time.time()
            tap_count = 0

            # --- RECORDING & TAPPING LOOP ---
            while (time.time() - start_time) < RECORD_SECONDS:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)

                elapsed = time.time() - start_time
                
                # Solenoid Logic (0.5s, 1.5s, 2.5s)
                if tap_count < 3:
                    if (elapsed > 0.5 and tap_count == 0) or \
                       (elapsed > 1.5 and tap_count == 1) or \
                       (elapsed > 2.5 and tap_count == 2):
                        self.trigger_solenoid()
                        tap_count += 1

            # Cleanup Audio
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save File
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            self.update_status(f"Saved: {filename}")
            
            # Update UI Counter
            self.after(100, lambda: self.update_counter_label(current_label))

        except Exception as e:
            self.update_status(f"Error: {e}")
            print(e)
        
        finally:
            self.is_recording = False
            self.btn_record.configure(state="normal", text="START RECORDING", fg_color="#D32F2F")

    def trigger_solenoid(self):
        GPIO.output(SOLENOID_PIN, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)

    def update_status(self, text):
        self.lbl_status.configure(text=text)

    def on_closing(self):
        GPIO.cleanup()
        self.destroy()

if __name__ == "__main__":
    app = DataCollectorApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()