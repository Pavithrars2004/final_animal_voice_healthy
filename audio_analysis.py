import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def process_audio(filepath):
    # Load audio file
    y, sr = librosa.load(filepath, sr=None)

    # Generate Spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    # Save the graph
    graph_path = f"static/uploads/{os.path.basename(filepath).split('.')[0]}_graph.png"
    plt.savefig(graph_path)
    plt.close()

    # Generate random temperature
    temperature = round(random.uniform(36.0, 40.0), 1)

    # Determine health status
    health_status = "Healthy" if temperature < 38.5 else "Unhealthy"

    return health_status, temperature, graph_path
