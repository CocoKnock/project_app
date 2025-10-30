import sounddevice as sd
import scipy.io.wavfile as wav
import os
from utils.LoadConfig import CONFIG

# --- Audio config parameters ---
global rate 
rate = CONFIG['AudioSettings']['SampleRate']
channels = CONFIG['AudioSettings']['Channels']
duration = CONFIG['AudioSettings']['Duration']

def record_audio():
    sd.default.samplerate = rate
    sd.default.channels = channels

    audio_recording = sd.rec(int(duration * rate))
    print("Recording Audio...")
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    return audio_recording

def play_audio(audio_data):
    sd.play(audio_data, rate)
    print("Playing Audio...")
    sd.wait()  # Wait until the file is done playing
    print("Playback complete.")

def save_audio(audio_data, filename):
    wav.write(filename, rate, audio_data)
    print(f"Audio saved as {filename}")