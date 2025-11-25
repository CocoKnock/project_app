# Coconut Classification Device Based on Image HSV and Acoustic Feedback Feature Extraction and Analysis

## Data Collection Flow (Updated Nov 25, 2025)

This project includes a small GUI for collecting labeled coconut data (image + audio). The data collection flow was updated so that classification selection happens before camera capture. The current sequence is:

1. Step 1 — Classification: Select the class/label (e.g., Malauhog, Malakanin, Malakatad).
2. Step 2 — Camera Capture: Capture an image of the fruit.
3. Step 3 — Audio Recording: Record the tap/knock audio (the app triggers the solenoid taps during recording if hardware is connected).
4. Step 4 — Confirm & Save: Review and save the image and audio together into the dataset folder.

Navigation notes:
- From the main menu, press "Data Collection" to begin (this now opens the classification selection first).
- After saving, choose "YES (New Capture)" to begin another collection — this will return to Step 1 (classification).

Requirements for full data collection:
- Camera (webcam or Raspberry Pi camera)
- Microphone or audio input device
- Optional: GPIO solenoid wiring if you want automatic tapping during recording

If you run the GUI locally, start the application with one of the project entry scripts (for example `python test.py` or `python main.py`) depending on which script you prefer to run.

## Change Log
- 2025-11-25: Reordered data collection flow so classification is step 1 and camera capture is step 2. UI labels and navigation updated in `test.py`.
This repository contains the deployment package for the Raspberry Pi program developed for our research project.
