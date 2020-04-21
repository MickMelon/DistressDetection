import speech_recognition

recogniser = speech_recognition.Recognizer()

# Carries out speech-to-text recognition using the SpeechRecogniser.
# Takes in the input of the audio file location and outputs the text.
def speech_to_text(audioFileLocation):
    try:
        soundFile = speech_recognition.AudioFile(audioFileLocation)
        with soundFile as source:
            audio = recogniser.record(source)

        output = recogniser.recognize_google(audio)
    except:
        output = ""

    return output
