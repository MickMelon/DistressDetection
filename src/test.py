import speech
import keyword_spotting
import repetitive

#output = speech.speech_to_text('sound/sound.wav')
#keywords_spotted = keyword_spotting.spot_keyword(output)

#print("Number of keywords spotted:" + str(keywords_spotted))

#similar_occurences = repetitive.run(output)
#print("Number of similar occurences: " + str(similar_occurences))

tes = "Hello mate how are you doing today?"
result = repetitive.run(tes)
print("Result: " + str(result))