from emotion_classifier_binary import EmotionClassifierBinary
from emotion_classifier_multi import ClassifierName
from emotion_classifier_multi import EmotionClassifierMulti
from repetitive_speech_detector import RepetitiveSpeechDetector
from keyword_spotter import KeywordSpotter
from collections import Counter
import random
from distress_score import DistressScore


def train_model():
    EmotionClassifierBinary.from_new()

def test_model():
    emc = EmotionClassifierBinary.from_existing('models/mlp_classifier.model')
    X, y = emc.create_test_set()
    emc.predict_set(X, y)

def test_emc():
    emc = EmotionClassifierBinary.from_existing('models/mlp_classifier.model')
    emc.predict('data_old/angry/a04.wav')

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
    result = rsd.check("He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun.")
    print(f'Result {result}')


def test_kws():
    kws = KeywordSpotter()
    kws.set_keywords(["cloud", "scientist", "planet", "star", "telescope", "supernova", "percentage", "put", "world", "observatory", "news"])
    dataset = ["Astronomers have observed a distant planet where it probably rains iron.",
               "It sounds like a science fiction movie, but this is the nature of some of the extreme worlds we're now discovering.",
                "Wasp-76b, as it's known, orbits so close in to its host star, its dayside temperatures exceed 2,400C - hot enough to vaporise metals."
                "The planet's nightside, on the other hand, is 1,000 degrees cooler, allowing those metals to condense and rain out.",
                "It's a bizarre environment, according to Dr David Ehrenreich from the University of Geneva.",
                "'Imagine instead of a drizzle of water droplets, you have iron droplets splashing down,' he told BBC News.",
                "The Swiss researcher and colleagues have just published their findings on this strange place in the journal Nature.",
                "The team describes how it used the new Espresso instrument at the European Southern Observatory's Very Large Telescope in Chile to study the chemistry of Wasp-76b in fine detail",
                "The planet, which is 640 light-years from us, is so close to its star it takes just 43 hours to complete one revolution.",
                "Another of the planet's interesting features is that it always presents the same face to the star - a behaviour scientists call being 'tidally locked'. Earth's Moon does exactly the same thing; we only ever see one side.",
                "This means, of course, the permanent dayside of Wasp-76b is being roasted.",
                "In fact, this hemisphere must be so hot that all clouds are dispersed, and all molecules in the atmosphere are broken apart into individual atoms.",
                "What's more, the extreme temperature difference this produces between the lit and unlit portions of the planet will be driving ferocious winds, up to 18,000km/h says Dr Ehrenreich's team.",
                "Using the Espresso spectrometer, the scientists detected a strong iron vapour signature at the evening frontier, or terminator, where the day on Wasp-76b transitions to night. But when the group observed the morning transition, the iron signal was gone.",
                "'What we surmise is that the iron is condensing on the nightside, which, although still hot at 1,400C, is cold enough that iron can condense as clouds, as rain, possibly as droplets. These could then fall into the deeper layers of the atmosphere which we can't access with our instrument,' Dr Ehrenreich explained.",
                "Wasp-76b is a monster gas planet that's twice the width of our Jupiter. Its unusual name comes from the UK-led Wasp telescope system that detected the world four years ago.",
                "One of the scientists on the discovery team, Prof Don Pollacco from Warwick University, said it was hard to envisage such exotic worlds.",
                "'This thing orbits so close to its star, it's essentially dancing in the outer atmosphere of that star and being subjected to all kinds of physics that, to put it bluntly, we don't really understand,' he told BBC News.",
                "'It will either end up in the star or the radiation field from the star will blow away the planet's atmosphere to leave just a hot, rocky core.'",
                "Dr Ehrenreich is a fan of graphic novels and asked the Swiss illustrator Frederik Peeters to produce an interpretation of Wasp-76b.",
                "'Often with these discoveries, we see detailed 3D compositions where it's difficult for people to tell whether it's a real picture or just a computer-generated image. By putting some fun into it, we're not fooling anyone,' he said."]

    overall_result = dict()
    for test in dataset:
        result = kws.check(test)
        # Merge dictionaries
        overall_result = Counter(overall_result) + Counter(result.matching_keywords)

    for result in overall_result:
        print(f'{result}: {overall_result[result]}')


def test_rsd():
    rsd = RepetitiveSpeechDetector()
    dataset = ["Hello, how are you today?",
               "Hello, how are you today?",
               "How are you today?",
               "How are you doing on this fine day?",
               "The moon orbits the earth",
               "My favourite colour is red",
               "Hello friend. How are you getting to work today?",
               "Today is the day that I'm going to learn how to say hello in German",
               "How do you do?",
               "Doris enjoyed tapping her nails on the table to annoy everyone",
               "The minute she landed she understood the reason this was a fly-over state.",
               "They got there early, and they got really good seats.",
               "Hello Betty, how are the kids doing today?",
               "Someone I know recently combined Maple Syrup & buttered Popcorn thinking it would taste like caramel popcorn.",
               "Hello, how are you?"]

    #random.shuffle(dataset)
    for test in dataset:
        print("***")
        print(f"Checking {test}")
        result = rsd.check(test)
        if result.distress_score is not DistressScore.NONE:
            print(result)
        else:
            print("no")
        print("***")
