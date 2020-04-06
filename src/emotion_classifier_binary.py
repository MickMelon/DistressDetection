import soundfile
import numpy as np
import librosa
import glob
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from enum import Enum
from collections import namedtuple
from distress_score import DistressScore

# A named tuple to allow for a structured return type
EmotionClassifierBinaryResult = namedtuple("EmotionClassifierBinaryResult",
                                     "distress_score distress_proba no_distress_proba")


# The name of the dataset
class DatasetName(Enum):
    English = 0,
    Ravdess = 1


# The name of the classifier
class ClassifierName(Enum):
    MLP = 0,
    SVM = 1,
    AdaBoost = 2,
    NN = 3


# The emotion classifier implementation class
class EmotionClassifierBinary:
    # All the available emotions that may be classified
    AVAILABLE_EMOTIONS = {
        "angry",
        "sad",
        "neutral",
        "happy",
        "fearful",
        "disgust",
        "surprised"
    }

    # All the emotions that would be related to distress
    DISTRESS_EMOTIONS = {
        "angry",
        "fearful",
        "sad"
    }

    # Trains a new model with the specified dataset and classifier
    @classmethod
    def from_new(cls, save_model=True, dataset=DatasetName.English, classifier=ClassifierName.MLP):
        # Create a new class instance
        instance = cls()

        # Load the dataset into the class properties
        instance.X_train, instance.X_test, instance.y_train, instance.y_test = instance.__load_data(dataset=dataset)

        # Check which classifier was specified and create the object from it
        if classifier is ClassifierName.SVM:
            instance.model = svm.SVC()
        elif classifier is ClassifierName.AdaBoost:
            instance.model = AdaBoostClassifier(n_estimators=100)
        elif classifier is ClassifierName.NN:
            instance.model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
        else: # Default classifier will be MLP
            if classifier is ClassifierName.MLP:
                model_params = {
                    'alpha': 0.01,
                    'batch_size': 256,
                    'epsilon': 1e-08,
                    'hidden_layer_sizes': (300,),
                    'learning_rate': 'adaptive',
                    'max_iter': 500,
                }
                instance.model = MLPClassifier(**model_params)

        # Train the specified classifier with the specified dataset
        instance.model.fit(instance.X_train, instance.y_train)

        # Save the model to disk if specified
        if save_model:
            if not os.path.isdir("models"):
                os.mkdir("models")
            pickle.dump(instance.model, open("models/mlp_classifier.model", "wb"))

        # Return the completed class instance
        return instance

    # Creates an EmotionClassifier object from an existing model
    @classmethod
    def from_existing(cls, model_file_path):
        # Create a new class instance
        instance = cls()

        # Load the model from disk. Throw an exception if the model was not found
        if os.path.exists(model_file_path):
            instance.model = pickle.load(open(model_file_path, 'rb'))
            return instance
        else:
            raise Exception("Specified model file path could not be found")

    # Loads in the specified dataset
    def __load_data(self, dataset, test_size=0.2):
        if dataset == DatasetName.English:
            return self.__load_data_english(test_size)
        elif dataset == DatasetName.Ravdess:
            return self.__load_data_ravdess(test_size)

    # Loads in the English dataset
    def __load_data_english(self, test_size=0):
        if not os.path.exists("data_old"):
            raise Exception("Path file English dataset not found")

        # Go through each emotion and load all the files for that emotion.
        X, y = [], []
        for emotion in self.AVAILABLE_EMOTIONS:
            for file in glob.glob("data_old/%s/*.wav" % emotion):
                # Extract the features and add the result into the test set.
                features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)
                X.append(features)

                # Append the appropriate label depending on whether the emotion is specified
                # as in distress or not in distress in the DISTRESS_EMOTIONS global array.
                # 1: in-distress
                # 0: not in-distress
                if emotion in self.DISTRESS_EMOTIONS:
                    y.append("1")
                else:
                    y.append("0")

        # Testing stuff, load the RAVDESS in too
        int2emotion = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        for file in glob.glob("data/Actor_*/*.wav"):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]
            if emotion not in self.AVAILABLE_EMOTIONS:
                continue
            features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)
            X.append(features)
            if emotion in self.DISTRESS_EMOTIONS:
                y.append("1")
            else:
                y.append("0")

        # Split the test set into training and prediction
        return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

    # Loads in the RAVDESS dataset
    def __load_data_ravdess(self, test_size=0.2):
        # A dictionary to convert the number key to emotion string, as the RAVDESS files are saved
        # with a number key indicating the particular emotion
        int2emotion = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        X, y = [], []

        # Go through each file, extract features, and add to test set with appropriate label
        for file in glob.glob("data/Actor_*/*.wav"):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]
            if emotion not in self.AVAILABLE_EMOTIONS:
                continue
            features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)
            X.append(features)
            y.append(emotion)

        # Split the test set into training and prediction
        return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

    # Extracts the specified features
    def __extract_features(self, file, **kwargs):
        # Get the specified arguments
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")

        # Extract the specified features for the specified sound file
        with soundfile.SoundFile(file) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if chroma or contrast:
                stft = np.abs(librosa.stft(X))
            result = np.array([])
            # Mel-frequency cepstrum coefficient features
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            # Chroma features
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            # Mel spectogram features
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))
            # Contrast features
            if contrast:
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, contrast))
            # Tonnetz features
            if tonnetz:
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
                result = np.hstack((result, tonnetz))

        # Return the extracted features from the sound
        return result

    def test(self):
        # predict 25% of data to measure how good we are
        y_pred = self.model.predict(self.X_test)
        print(y_pred)

       # print(" ************** y_pred_proba ***************")
       # y_pred_proba = self.model.predict_proba(self.X_test)
      #  print(y_pred_proba)
      #  print("*********************************************")

        # We want to get the 2 highest probs
        # emotion: value

        # calculate the accuracy
        accuracy = accuracy_score(y_true=self.y_test, y_pred=y_pred)

        print("Accuracy: {:.2f}%".format(accuracy * 100))

       # print("For the first sound clip.....")
       # print(y_pred_proba[0][3])

        available_emotions = list(self.AVAILABLE_EMOTIONS)

       # for prod in y_pred_proba:
       #     count = 0
       #     highest_score = 0
        #    highest_emotion = "none"
#
       #     second_highest_score = 0
        #    second_highest_emotion = "none"
#
       #     for emotion in prod:
                # print("Emotion %s predicted %s (original %s)" % (available_emotions[count], round(emotion * 100, 2), emotion))
                # print("type is %s " % (type(emotion)))

      #          current_score = round(emotion * 100, 2)
            #    if current_score > highest_score:
             #       highest_score = current_score
            #        highest_emotion = available_emotions[count]
             #   elif current_score > second_highest_score:
            #        second_highest_score = current_score
            #        second_highest_emotion = available_emotions[count]

          #      count = count + 1

          #  print("***")
         #   print("Predicted %s with a score of %s" % (highest_emotion, str(highest_score)))
          #  print("Second highest was %s with a score of %s" % (second_highest_emotion, str(second_highest_score)))
          #  print("***")

        return accuracy * 100

    def create_test_set(self):
        X, y = [], []
        for emotion in self.AVAILABLE_EMOTIONS:
            for file in glob.glob("data_old/%s/*.wav" % emotion):
                features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)
                X.append(features)

                if emotion in self.DISTRESS_EMOTIONS:
                    y.append("1")
                else:
                    y.append("0")

        return X, y


    # Carry out prediction on a set of features
    def predict_set(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))


    # Predict a single file
    def predict(self, input):
        print("Predicting...")
        # Extract the features from the file
        for file in glob.glob(input):
            features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)

        # Classify the file and return the result
        proba = self.model.predict_proba(features.reshape(1, -1))

        # Get the results for distress and no distress
        distress_proba = round(proba[0][0] * 100, 2)
        no_distress_proba = round(proba[0][1] * 100, 2)

        # Display the scores
        print(f"Distress score is {distress_proba} and no-distress score is {no_distress_proba}")

        # Calculate the distress score depending on the results
        if distress_proba > 90:
            score = DistressScore.HIGH
        elif distress_proba > 80:
            score = DistressScore.MEDIUM
        elif distress_proba > 70:
            score = DistressScore.LOW
        else:
            score = DistressScore.NONE

        # Finally, return the named tuple construct
        return EmotionClassifierBinaryResult(
            distress_score=score,
            distress_proba=distress_proba,
            no_distress_proba=no_distress_proba
        )
