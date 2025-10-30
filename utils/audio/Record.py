import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa
import librosa.display
import sounddevice as s
from utils.LoadConfig import CONFIG

def record_audio():
    rate = CONFIG['AudioSettings']['SampleRate']
    channels = CONFIG['AudioSettings']['Channels']
    format = CONFIG['AudioSettings']['Format']

    return rate, channels, format