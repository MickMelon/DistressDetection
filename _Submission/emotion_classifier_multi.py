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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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


MULTI_LAYER_PERCEPTRON = "MultiLayerPerceptron"
SUPPORT_VECTOR_MACHINE = "SupportVectorMachine"
ADA_BOOST = "AdaBoost"
NEURAL_NETWORK = "NeuralNetwork"
DECISION_TREE = "DecisionTree"
RANDOM_FOREST = "RandomForest"
GAUSSIAN_NB = "GaussianNB"


# The emotion classifier implementation class
class EmotionClassifierMulti:
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
    }

    # Trains a new model with the specified dataset and classifier
    @classmethod
    def from_new(cls, save_model=True, dataset=DatasetName.English, classifier=MULTI_LAYER_PERCEPTRON):
        # Create a new class instance
        instance = cls()

        # Load the dataset into the class properties
        instance.X_train, instance.X_test, instance.y_train, instance.y_test = instance.__load_data(dataset=dataset)

        # Check which classifier was specified and create the object from it

        # Support Vector Machine
        if classifier is SUPPORT_VECTOR_MACHINE:
            instance.model = svm.SVC()
        # AdaBoost
        elif classifier is ADA_BOOST:
            instance.model = AdaBoostClassifier(n_estimators=100)
        # Neural Network
        elif classifier is NEURAL_NETWORK:
            instance.model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
        # Decision Tree
        elif classifier is DECISION_TREE:
            instance.model = DecisionTreeClassifier(max_depth=5)
        # Random Forest
        elif classifier is RANDOM_FOREST:
            instance.model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        # GaussianNB
        elif classifier is GAUSSIAN_NB:
            instance.model = GaussianNB()
        # Multi Layer Perceptron
        elif classifier is MULTI_LAYER_PERCEPTRON:
            model_params = {
                'alpha': 0.01,
                'batch_size': 256,
                'epsilon': 1e-08,
                'hidden_layer_sizes': (300,),
                'learning_rate': 'adaptive',
                'max_iter': 500,
            }
            instance.model = MLPClassifier(**model_params)
        else:
            raise Exception("Invalid classifier specified")

        if not hasattr(instance, 'model'):
            raise Exception(f"No model was created for the classifier: {classifier}")

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

        X, y = [], []
        for emotion in self.AVAILABLE_EMOTIONS:
            for file in glob.glob("data_old/%s/*.wav" % emotion):
                features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)
                X.append(features)
                y.append(emotion)

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
            y.append(emotion)

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
        y_pred = self.model.predict(self.X_test)

        total_correct = 0
        for i in range(len(y_pred)):
            print(f"Comparing {y_pred[i]} with {self.y_test[i]}")

            # now get the binary that y_pred is and compare the binary of y_test
            if y_pred[i] in self.DISTRESS_EMOTIONS and self.y_test[i] in self.DISTRESS_EMOTIONS:
                total_correct += 1
            elif y_pred[i] not in self.DISTRESS_EMOTIONS and self.y_test[i] not in self.DISTRESS_EMOTIONS:
                total_correct += 1

        print(f"Total correct: {total_correct} out of {len(y_pred)}")
        per = total_correct * (100 / len(y_pred))
        print(f"Percentage is {per}")
        return per

    def create_test_set(self):
        X, y = [], []
        for emotion in self.AVAILABLE_EMOTIONS:
            for file in glob.glob("data_old/%s/*.wav" % emotion):
                features = self.__extract_features(file, mfcc=True, chroma=True, mel=True)
                X.append(features)
                y.append(emotion)

        return X, y

    def predict_set(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))


    def predict_set2(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        total_correct = 0
        for i in range(len(y_pred)):
            print(f"Comparing {y_pred[i]} with {y_test[i]}")

            # now get the binary that y_pred is and compare the binary of y_test
            if y_pred[i] in self.DISTRESS_EMOTIONS and y_test[i] in self.DISTRESS_EMOTIONS:
                total_correct += 1
            elif y_pred[i] not in self.DISTRESS_EMOTIONS and y_test[i] not in self.DISTRESS_EMOTIONS:
                total_correct += 1

        print(f"Total correct: {total_correct} out of {len(y_pred)}")
        per = total_correct * (100 / len(y_pred))
        print(f"Percentage is {per}")
        return per

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





