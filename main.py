import os

from utils.audio.Record import record_audio, play_audio, save_audio
from utils.LoadConfig import CONFIG
from utils.FileManagement import get_next_filename

from utils.image.Capture import capture_image

# --- Audio File Path parameters ---
audio_save_path = CONFIG['AudioFilePaths']['AudioSavePath']
filename_prefix_audio = CONFIG['AudioFilePaths']['FileNamePrefix']
format_audio = CONFIG['AudioFilePaths']['AudioFormat']

# --- Image File Path parameters ---
image_save_path = CONFIG['ImageFilePaths']['ImageSavePath']
filename_prefix_image = CONFIG['ImageFilePaths']['FileNamePrefix']
format_image = CONFIG['ImageFilePaths']['ImageFormat']

audio_recording = record_audio()
play_audio(audio_recording)

audio_filepath = get_next_filename(audio_save_path, filename_prefix_audio, format_audio)
image_filepath = get_next_filename(image_save_path, filename_prefix_image, format_image)

save_audio(audio_recording, audio_filepath)
capture_image(image_filepath)

print (audio_filepath)
print (image_filepath)