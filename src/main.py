import emotion
import keyword_spotting
import repetitive
import decision
import speech

from concrete_classes.emotion_classifier_mlp import EmotionClassifierMlp
from concrete_classes.emotion_classifier_mlp import EmotionClassifierResult
from concrete_classes.emotion_classifier_mlp import DatasetName
from concrete_classes.keyword_spotter_text import KeywordSpotterText
from concrete_classes.repetitive_speech_detector_text import RepetitiveSpeechDetectorText

#emotion_result = emotion.run()

#text = speech.speech_to_text('sound/sound.wav')

#kws_result = keyword_spotting.spot_keyword(text)
#rsd_result = repetitive.run(text)

#decision_result = decision.make_decision(kws_result, emotion_result, rsd_result, False)
#print(decision_result)



emc = EmotionClassifierMlp.from_existing('models/mlp_classifier.model')
kws = KeywordSpotterText()
rsd = RepetitiveSpeechDetectorText(fill_backlog=True)

text = speech.speech_to_text('sound/sound.wav')
emc_result = emc.predict('sound/sound.wav')
kws_result = kws.check(text)
rsd_result = rsd.check(text)

print("Result from EMC:")
print("I'm %s percent sure it is %s" % (emc_result.highest_score, emc_result.highest_name))
print("Otherwise I'd be %s percent sure it is %s" % (emc_result.second_highest_score, emc_result.second_highest_name))
