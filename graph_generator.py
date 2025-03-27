import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_spectrogram(audio_path, output_path):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)

        # Compute the Short-Time Fourier Transform (STFT)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        # Create the spectrogram plot
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogram of Animal Sound")

        # Save the spectrogram image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        print(f"Spectrogram saved: {output_path}")

    except Exception as e:
        print(f"Error generating spectrogram: {e}")

