import speech_recognition

recogniser = speech_recognition.Recognizer()


def speech_to_text(audioFileLocation):
    soundFile = speech_recognition.AudioFile(audioFileLocation)
    with soundFile as source:
        audio = recogniser.record(source)

    output = recogniser.recognize_google(audio)
    return output
