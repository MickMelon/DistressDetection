import emotion
import keyword_spotting
import repetitive
import decision
import speech

emotion_result = emotion.run()

text = speech.speech_to_text('sound/sound.wav')

kws_result = keyword_spotting.spot_keyword(text)
rsd_result = repetitive.run(text)

decision_result = decision.make_decision(kws_result, emotion_result, rsd_result, False)
print(decision_result)