import librosa
import numpy as np
import os

# Function to extract MFCC from audio file
def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)  # Taking the mean across time
    return mfcc

# Prepare the dataset for training (this includes both features and labels)
def prepare_data(dataset_path):
    data = []
    labels = []

    for label in ['healthy', 'unhealthy']:
        path = os.path.join(dataset_path, label)
        for file in os.listdir(path):
            if file.endswith(".wav"):
                file_path = os.path.join(path, file)
                mfcc_features = extract_mfcc(file_path)
                data.append(mfcc_features)
                labels.append(label)

    return np.array(data), np.array(labels)
