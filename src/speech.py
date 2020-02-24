import speech_recognition

recogniser = speech_recognition.Recognizer()


def speech_to_text(audioFileLocation):
    try:
        soundFile = speech_recognition.AudioFile(audioFileLocation)
        with soundFile as source:
            audio = recogniser.record(source)

        output = recogniser.recognize_google(audio)
    except:
        output = ""

    return output
