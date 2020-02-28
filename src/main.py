import python_vad
from emotion_classifier import EmotionClassifier


def train_model():
    emc = EmotionClassifier.from_new()


def test_model():
    emc = EmotionClassifier.from_existing('models/mlp_classifier.model')
    X, y = emc.create_test_set()
    emc.predict_set(X, y)


print("Starting...")
python_vad.start()