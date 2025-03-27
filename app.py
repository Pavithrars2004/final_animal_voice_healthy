from flask import Flask, render_template, request
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid Tkinter issues

from werkzeug.utils import secure_filename
import random
from audio_analysis import process_audio

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        file = request.files["audio"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the audio file
        health_status, temperature, graph_path = process_audio(filepath)

        return render_template("result.html", filename=filename, health=health_status, temp=temperature, graph=graph_path)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
