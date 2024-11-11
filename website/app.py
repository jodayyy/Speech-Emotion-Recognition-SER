from flask import Flask, request, render_template, jsonify, Response
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from keras.models import load_model
from keras import backend as K
import joblib
import noisereduce as nr
from pydub import AudioSegment

app = Flask(__name__)

# Ensure uploads directory exists
os.makedirs('uploads', exist_ok=True)

# Load your trained model and other utilities
model = load_model('SERv3.h5')
scaler = joblib.load('scalerv2.joblib')
encoder = joblib.load('encoderv2.joblib')

AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
AudioSegment.ffprobe   = "C:\\ffmpeg\\bin\\ffprobe.exe"

def get_last_uploaded_file_path():
    global last_uploaded_file
    if last_uploaded_file is not None:
        return os.path.join('uploads', last_uploaded_file)
    else:
        return None

# Function to extract features
def extract_features(data, sample_rate):
    # Trim silence from the beginning and end
    data, _ = librosa.effects.trim(data)
    
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    # Normalization
    result = (result - np.mean(result)) / np.std(result)
    
    return result

# Function to predict emotion, same as in your original code
def predict_emotion(file_path):
    # Convert to WAV if not already
    audio = AudioSegment.from_file(file_path)
    audio.export(file_path, format="wav")

    # Load audio file
    data, sample_rate = librosa.load(file_path, sr=48000)

    # Noise reduction
    reduced_noise_data = nr.reduce_noise(y=data, sr=sample_rate)

    # Extract features from the audio with noise reduction
    features = extract_features(reduced_noise_data, sample_rate)
    features_scaled = scaler.transform([features])
    features_scaled = np.expand_dims(features_scaled, axis=2)
    prediction = model.predict(features_scaled)
    predicted_emotion = encoder.inverse_transform(prediction)
    confidence = np.max(prediction) * 100  # Convert confidence to percentage
    confidence = f"{confidence:.2f}%"  # Format as a string with two decimal places
    return predicted_emotion[0][0], confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        global last_uploaded_file

        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            last_uploaded_file = file.filename  # Update the last uploaded file
            # No longer directly returning the result
            return 'File successfully uploaded'
    else:
        return render_template('upload.html')
    
@app.route('/real_time_ser')
def real_time_ser():
    return render_template('rtser.html')

@app.route('/predict_emotion', methods=['GET'])
def predict():
    last_uploaded_file_path = get_last_uploaded_file_path()  
    emotion, confidence = predict_emotion(last_uploaded_file_path)
    return jsonify({"emotion": emotion, "confidence": str(confidence)})

@app.route('/predict_recorded_emotion', methods=['POST'])
def predict_recorded_emotion():
    if 'audio_data' not in request.files:
        return Response("No audio file", status=400)
    
    audio_file = request.files['audio_data']
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join('uploads', filename)
    audio_file.save(file_path)

    # Use existing emotion prediction function
    emotion, confidence = predict_emotion(file_path)
    return jsonify({"emotion": emotion, "confidence": str(confidence)})

@app.route('/process_real_time_audio', methods=['POST'])
def process_real_time_audio():
    if 'real_time_audio' not in request.files:
        return Response("No audio file", status=400)

    real_time_audio = request.files['real_time_audio']
    filename = secure_filename(real_time_audio.filename)
    file_path = os.path.join('uploads', filename)
    real_time_audio.save(file_path)

    # Process and predict emotion
    emotion, confidence = predict_emotion(file_path)
    return jsonify({"emotion": emotion, "confidence": str(confidence)})

if __name__ == '__main__':
    app.run(debug=True)