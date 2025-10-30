import cv2 # <-- New required import
# import os # Already imported in your code
# ... other imports ...

# --- Image Capture Function ---
def capture_image(filepath: str) -> bool:
    """
    Opens the default camera, captures a single frame, and saves it.

    Args:
        filepath: The full path (including directory and filename) to save the image.

    Returns:
        True if the image was successfully saved, False otherwise.
    """
    # 1. Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return False

    # 2. Capture a frame
    # read() returns a tuple: (success_flag, image_frame)
    ret, frame = cap.read()

    # 3. Release the camera resource immediately
    cap.release()

    if ret:
        # 4. Save the captured frame to the specified filepath
        # cv2.imwrite supports formats like .jpg, .png, etc.
        cv2.imwrite(filepath, frame)
        print(f"🖼️ Image captured and saved to: {filepath}")
        return True
    else:
        print("❌ Error: Failed to capture frame from camera.")
        return False