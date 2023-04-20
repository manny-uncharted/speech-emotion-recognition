from keras.models import load_model
import numpy as np
import librosa

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

# Load the audio file
audio_path = 'OAF_base_fear.wav'

# Extract the MFCCs
def extract_mfcc(filename):
     y, sr = librosa.load(filename, duration=3, offset=0.5)
     mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
     return mfcc
mfccs = extract_mfcc(audio_path)
X = np.array(mfccs)
X = np.expand_dims(X, -1)

# Make a prediction
prediction = model.predict(X)

class_index = np.argmax(prediction[0])
predicted_class = class_names[class_index]

print('Predicted class: {}'.format(predicted_class))