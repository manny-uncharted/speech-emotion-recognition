from keras.models import load_model
import numpy as np
import librosa
import streamlit as st


st.header("Speech Emotion Recognition")
st.subheader("Upload an audio file to predict the emotion")
st.caption("The application would infer the one label from the following list: 'ps', 'fear', 'happy', 'sad', 'angry', 'disgust', 'neutral'")
st.caption("The audio file should be in .wav format and should be of duration 3 seconds")

# load the model
model = load_model('speech_recog_001.h5')

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

# Load the audio file
audio_file = st.file_uploader("Upload Audio", type=["wav"])

if audio_file is not None:
    with st.spinner("Predicting..."):
        mfccs = extract_mfcc(audio_file)
        X = np.array(mfccs)
        X = np.expand_dims(X, -1)
        prediction = model.predict(X)
        class_index = np.argmax(prediction[0])
        predicted_class = class_names[class_index]
        st.success("The predicted emotion is: {}".format(predicted_class))