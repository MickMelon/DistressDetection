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

EmotionClassifierResult = namedtuple("EmotionClassifierResult",
                                     "highest_name highest_score second_highest_name second_highest_score")


class DatasetName(Enum):
    English = 0,
    Ravdess = 1


class ClassifierName(Enum):
    MLP = 0,
    SVM = 1,
    AdaBoost = 2,
    NN = 3

class EmotionClassifierMlp():
    # The trained MLP classifier model


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

    DISTRESS_EMOTIONS = {
        "angry",
        "sad",
        "fearful",
        "disgust"
    }

    @classmethod
    def from_new(cls, save_model=False, dataset=DatasetName.English, classifier=ClassifierName.MLP):
        instance = cls()

        instance.X_train, instance.X_test, instance.y_train, instance.y_test = instance.__load_data(dataset=dataset)

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
        instance.model.fit(instance.X_train, instance.y_train)

        if save_model:
            if not os.path.isdir("models"):
                os.mkdir("models")
            pickle.dump(instance.model, open("models/mlp_classifier.model", "wb"))

        return instance

    @classmethod
    def from_existing(cls, model_file_path):
        instance = cls()

        if os.path.exists(model_file_path):
            instance.model = pickle.load(open(model_file_path, 'rb'))
            return instance
        else:
            raise Exception("Specified model file path could not be found")

    # Loads in the specified dataset
    def __load_data(self, dataset, test_size=0.2):
        print("What teh fuck????")
        if dataset == DatasetName.English:
            return self.__load_data_english(test_size)
        elif dataset == DatasetName.Ravdess:
            return self.__load_data_ravdess(test_size)

    # Loads in the English dataset
    def __load_data_english(self, test_size=0):
        print("What teh fuck?wat")
        if not os.path.exists("data_old"):
            raise Exception("Path file English dataset not found")

        X, y = [], []
        for emotion in self.AVAILABLE_EMOTIONS:
            for file in glob.glob("data_old/%s/*.wav" % emotion):
                features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)
                X.append(features)

                if emotion in self.DISTRESS_EMOTIONS:
                    y.append("1")
                else:
                    y.append("0")

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

        return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

    # Loads in the RAVDESS dataset
    def __load_data_ravdess(self, test_size=0.2):
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

        for file in glob.glob("data/Actor_*/*.wav"):
            basename = os.path.basename(file)
            emotion = int2emotion[basename.split("-")[2]]
            if emotion not in self.AVAILABLE_EMOTIONS:
                continue
            features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)
            X.append(features)
            y.append(emotion)

        return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

    # Extracts the specified features
    def __extract_features(self, file, **kwargs):
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")
        with soundfile.SoundFile(file) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if chroma or contrast:
                stft = np.abs(librosa.stft(X))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))
            if contrast:
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, contrast))
            if tonnetz:
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
                result = np.hstack((result, tonnetz))
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


    def predict_binary(self, input):
        print("Predicting...")
        for file in glob.glob(input):
            features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)

        proba = self.model.predict(features.reshape(1, -1))

        print(proba)
        return proba

    # Predicts what emotion the input is
    def predict(self, input):
        print("Predicting...")
        for file in glob.glob(input):
            features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)

        proba = self.model.predict_proba(features.reshape(1, -1))

        available_emotions = list(self.AVAILABLE_EMOTIONS)
        for prod in proba:
            count = 0
            highest_score = 0
            highest_emotion = "none"

            second_highest_score = 0
            second_highest_emotion = "none"

            for emotion in prod:
                current_score = round(emotion * 100, 2)
                if current_score > highest_score:
                    highest_score = current_score
                    highest_emotion = available_emotions[count]
                elif current_score > second_highest_score:
                    second_highest_score = current_score
                    second_highest_emotion = available_emotions[count]

                count = count + 1

            print("***")
            print("Predicted %s with a score of %s" % (highest_emotion, str(highest_score)))
            print("Second highest was %s with a score of %s" % (second_highest_emotion, str(second_highest_score)))
            print("***")

        return EmotionClassifierResult(
            highest_name=highest_emotion,
            highest_score=highest_score,
            second_highest_name=second_highest_emotion,
            second_highest_score=second_highest_score
        )
