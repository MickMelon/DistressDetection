import emotion
import keyword_spotting
import repetitive
import decision
import speech

from concrete_classes.emotion_classifier_mlp import EmotionClassifierMlp
from concrete_classes.emotion_classifier_mlp import DatasetName

#emotion_result = emotion.run()

#text = speech.speech_to_text('sound/sound.wav')

#kws_result = keyword_spotting.spot_keyword(text)
#rsd_result = repetitive.run(text)

#decision_result = decision.make_decision(kws_result, emotion_result, rsd_result, False)
#print(decision_result)



mlp = EmotionClassifierMlp.from_existing('models/mlp_classifier.model')
mlp.predict('sound/sound.wav')

mlp = EmotionClassifierMlp.from_new(save_model=True, dataset=DatasetName.English)
mlp.predict('sound/sound.wav')