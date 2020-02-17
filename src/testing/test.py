import glob
import random
import numpy as np
import librosa
import soundfile

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn import svm
from sklearn.decomposition import PCA

EMOTIONS = ["sad", "happy", "neutral", "fear", "surprise", "disgust"]

#output = speech.speech_to_text('sound/sound.wav')
#print("Output: " + output)
#keywords_spotted = keyword_spotting.spot_keyword(output)

#print("Number of keywords spotted:" + str(keywords_spotted))

#similar_occurences = repetitive.run(output)
#print("Number of similar occurences: " + str(similar_occurences))

#tes = "But we don’t always randomise enough data and our test data becomes stale and etc. etc."
#result = repetitive.run(tes)
#print("Result: " + str(result))

def read_data(emotion):
    print("Reading data for %s..." % emotion)
    files = glob.glob("data/%s/*" % emotion)
    print("There are " + str(len(files)) + " files for " + emotion)
    return files


def make_sets():
    print("Making sets...")
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    # Make the set for every emotion
    for emotion in EMOTIONS:
        files = read_data(emotion)
        files = extract_features2(files)

        # Randomise and split the data
        random.shuffle(files)
        training = files[:int(len(files) * 0.8)]
        prediction = files[-int(len(files) * 0.2):]



        # Add the training data and labels to arrays
        for item in training:
            training_data.append(item)
            training_labels.append(EMOTIONS.index(emotion))

        # Add the prediction data and labels to arrays
        for item in prediction:
            prediction_data.append(item)
            prediction_labels.append(EMOTIONS.index(emotion))

    print("Sets made. %i training and %i prediction" % (len(training_data), len(prediction_data)))

    print("[+] Number of training samples: ", training_data.shape[0])
    print("[+] Number of testing samples: ", prediction_data.shape[0])
    print("[+] Number of features: ", training_data.shape[1])

    return training_data, training_labels, prediction_data, prediction_labels


def extract_features(data):
    print("Extracting features...")
    for i in range(len(data)):
        (rate,sig) = wav.read(data[i])
        mfcc_feat = mfcc(sig, rate, nfft=1103)
        fbank_feat = logfbank(sig, rate)
        data[i] = perform_pca(fbank_feat)

    print("Features extracted")
    return data

def extract_features2(data):
    for i in range(len(data)):
        result = np.array([])

        with soundfile.SoundFile(data[i]) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
            data[i] = result

    return data

def perform_pca(features):
    pca = PCA(n_components=2)
    pca.fit(features)
    result = pca.singular_values_.reshape(-1, 1)
    print(result)
    return result

# Main
training_data, training_labels, prediction_data, prediction_labels = make_sets()

print(training_data[0])
print(training_labels[0])

print("Training classifier...")
clf = svm.SVC()
clf.fit(training_data, training_labels)
print("Classifier trained")

print("Predicting...")
result = clf.predict(prediction_data[0])
print("Predicted")
print(result)


