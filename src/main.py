import python_vad
from emotion_classifier import EmotionClassifier
from emotion_classifier_multi import ClassifierName
from emotion_classifier_multi import EmotionClassifierMulti
from repetitive_speech_detector import RepetitiveSpeechDetector


def train_model():
    emc = EmotionClassifier.from_new()


def test_model():
    emc = EmotionClassifier.from_existing('models/mlp_classifier.model')
    X, y = emc.create_test_set()
    emc.predict_set(X, y)


def test_classifiers():
    svm_average = 0
    mlp_average = 0
    ada_average = 0

    for i in range(5):
        svm = EmotionClassifierMulti.from_new(classifier=ClassifierName.SVM)
        svm_result = svm.test()
        print(f"SVM Result: {str(svm_result)}")
        svm_average += svm_result

    for i in range(5):
        mlp = EmotionClassifierMulti.from_new(classifier=ClassifierName.MLP)
        mlp_result = mlp.test()
        print(f"MLP Result: {str(mlp_result)}")
        mlp_average += mlp_result

    for i in range(5):
        ada = EmotionClassifierMulti.from_new(classifier=ClassifierName.AdaBoost)
        ada_result = ada.test()
        print(f"AdaBoost Result: {str(ada_result)}")
        ada_average += ada_result

    print("DONE")
    print(f"SVM Average: {str(svm_average / 5)}")
    print(f"MLP Average: {str(mlp_average / 5)}")
    print(f"AdaBoost Average: {str(ada_average / 5)}")


def test_rsd():
    rsd = RepetitiveSpeechDetector()
    result = rsd.check("The first three are not sentences because they do not contain a verb")
    print(f'Result {result}')


print("Starting...")
test_rsd()
#test_classifiers()
#python_vad.start()