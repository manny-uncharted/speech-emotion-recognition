from keras.models import load_model
import numpy as np
import librosa
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
import pydub
import pathlib
import subprocess
from audio_recorder_streamlit import audio_recorder
import numpy as np
import soundfile
import io


st.header("Speech Emotion Recognition")
st.subheader("Upload an audio file to predict the emotion")
st.caption("The application would infer the one label from the following list: 'ps', 'fear', 'happy', 'sad', 'angry', 'disgust', 'neutral'")
st.caption("The audio file should be in .wav format and should be of duration 3 seconds")

# load the model
model_cnn = load_model('speech_recog_003.h5')
model_lstm = load_model('speech_recog_001.h5')

# Define class names
class_names = [
    'ps',
    'fear',
    'happy',
    'sad',
    'angry',
    'disgust',
    'neutral'
]


#Functions
@st.cache_data
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

@st.cache_data
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=5)

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

@st.cache_data
def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    sr = sample_rate
    
    # without augmentation
    res1 = extract_features(data, sr)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sr)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sr)
    res3 = extract_features(data_stretch_pitch, sr)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

option_audio = st.selectbox('Do you want to speak to the microphone or input your own audio in wav format', ('Speak', 'Upload'))
st.write('You selected:', option_audio)
if option_audio == 'Speak':
    # Load the audio file
    audio_bytes = audio_recorder(
        text = "Record your audio",
        energy_threshold=(-1.0, 1.0),
        pause_threshold=7.0,
    )
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        s = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_raw(s, sample_width=2, frame_rate=16000, channels=1).export("audio.wav", format="wav")
        option = st.selectbox(
            'What model would you like to use?',
            ('CNN', 'LSTM'))
        st.write('You selected:', option)

        if option == 'CNN':
            model = model_cnn
            # sample_rate = sample_rate
            feature = get_features(audio)
            scaler = StandardScaler()
            feature = scaler.fit_transform(feature)
            with st.spinner("Predicting..."):
                prediction = model.predict(feature)
                class_index = np.argmax(prediction[0])
                predicted_class = class_names[class_index]
                st.success("The predicted emotion is: {}".format(predicted_class))

        elif option == 'LSTM':
            model = model_lstm
            # audio_file.type
            # print(audio_bytes.type)
            with st.spinner("Predicting..."):
                mfccs = extract_mfcc(audio_bytes)
                X = np.array(mfccs)
                X = np.expand_dims(X, -1)
                prediction = model.predict(X)
                class_index = np.argmax(prediction[0])
                predicted_class = class_names[class_index]
                st.success("The predicted emotion is: {}".format(predicted_class))
elif option_audio == 'Upload':
    audio_file = st.file_uploader("Upload Audio in wav format", type=["wav"])
    if audio_file is not None:
        option = st.selectbox(
            'What model would you like to use?',
            ('CNN', 'LSTM'))
        st.write('You selected:', option)

        if option == 'CNN':
            model = model_cnn
            # sample_rate = sample_rate
            feature = get_features(audio_file)
            scaler = StandardScaler()
            feature = scaler.fit_transform(feature)
            with st.spinner("Predicting..."):
                prediction = model.predict(feature)
                class_index = np.argmax(prediction[0])
                predicted_class = class_names[class_index]
                st.success("The predicted emotion is: {}".format(predicted_class))

        elif option == 'LSTM':
            model = model_lstm
            # audio_file.type
            print(audio_file.type)
            with st.spinner("Predicting..."):
                mfccs = extract_mfcc(audio_file)
                X = np.array(mfccs)
                X = np.expand_dims(X, -1)
                prediction = model.predict(X)
                class_index = np.argmax(prediction[0])
                predicted_class = class_names[class_index]
                st.success("The predicted emotion is: {}".format(predicted_class))
