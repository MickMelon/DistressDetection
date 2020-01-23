import speech_recognition

recogniser = speech_recognition.Recognizer()

soundFile = speech_recognition.AudioFile('sound/sound.wav')
with soundFile as source:
    audio = recogniser.record(source)

recogniser.recognize_google(audio)

