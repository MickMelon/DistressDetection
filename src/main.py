import emotion
import keyword_spotting
import repetitive
import decision
import speech

from concrete_classes.emotion_classifier_mlp import EmotionClassifierMlp
from concrete_classes.emotion_classifier_mlp import EmotionClassifierResult
from concrete_classes.emotion_classifier_mlp import DatasetName
from concrete_classes.emotion_classifier_mlp import ClassifierName
from concrete_classes.keyword_spotter_text import KeywordSpotterText
from concrete_classes.repetitive_speech_detector_text import RepetitiveSpeechDetectorText

import python_vad
import threading

print("Starting...")
#speech.speech_to_text('here_7.wav')
python_vad.start()
exit(1)

# When voice detected, trigger function with (filename, time)
# Adds to queue
# Process all items in queue one at a time
# Put through the pipeline, get output from decision system







#emotion_result = emotion.run()

#text = speech.speech_to_text('sound/sound.wav')

#kws_result = keyword_spotting.spot_keyword(text)
#rsd_result = repetitive.run(text)

#decision_result = decision.make_decision(kws_result, emotion_result, rsd_result, False)
#print(decision_result)


# Test all classifiers

# Create classifier train and run 5 times for each classifier, get average result
#total = 0
#for i in range(5):
#    mlp = EmotionClassifierMlp.from_new(True)
#    accuracy = mlp.test()
#    total = accuracy + total

#mlp_average = total / 5
#print("Average for MLP is %s" % str(mlp_average))

#total = 0
#for i in range(5):
#    mlp = EmotionClassifierMlp.from_new(True, classifier=ClassifierName.SVM)
#    accuracy = mlp.test()
#    total = accuracy + total

#svm_average = total / 5
#print("Average for SVM is %s" % str(svm_average))

total = 0
for i in range(5):
    mlp = EmotionClassifierMlp.from_new(True, classifier=ClassifierName.AdaBoost)
    accuracy = mlp.test()
    total = accuracy + total

adaboost_average = total / 5
print("Average for AdaBoost is %s" % str(adaboost_average))

total = 0
for i in range(5):
    mlp = EmotionClassifierMlp.from_new(True, classifier=ClassifierName.NN)
    accuracy = mlp.test()
    total = accuracy + total

nn_average = total / 5
print("Average for NN is %s" % str(nn_average))

# Also compare the features and pre-processing shit (after)

# At end, compare all average results


exit(1)

e = EmotionClassifierMlp.from_new(True, dataset=DatasetName.English)
e.test()

SOUND_FILE = 'test_data/su12.wav'
result = e.predict(SOUND_FILE)
print("Result from EMC:")
print("I'm %s percent sure it is %s" % (result.highest_score, result.highest_name))
print("Otherwise I'd be %s percent sure it is %s" % (result.second_highest_score, result.second_highest_name))
exit(1)


emc = EmotionClassifierMlp.from_new('models/mlp_classifier.model', dataset=DatasetName.Ravdess)
kws = KeywordSpotterText()
rsd = RepetitiveSpeechDetectorText(fill_backlog=True)

SOUND_FILE = 'test_data/u.wav'

text = speech.speech_to_text(SOUND_FILE)
emc_result = emc.predict(SOUND_FILE)
kws_result = kws.check(text)
rsd_result = rsd.check(text)

print("Result from EMC:")
print("I'm %s percent sure it is %s" % (emc_result.highest_score, emc_result.highest_name))
print("Otherwise I'd be %s percent sure it is %s" % (emc_result.second_highest_score, emc_result.second_highest_name))
