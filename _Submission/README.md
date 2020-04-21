# Speech Distress Detection Program
Michael McMillan - 1800833

Abertay University Honours Project

How Effective is Speech Analysis in Detecting a Dementia Patient in Distress?

## Contents

- `decision_output/` - This folder contains all the XML files outputted by the decision system.
- `models/` - This folder contains the pre-trained classification model, in this case there is a Support Vector Machine model in there.
- `contractions.py` - Contains a dictionary of contractions used in the text pre-processing steps.
- `decision.py` - Contains the decision system.
- `distress_score.py` - Contains an enum for the distress score used by each module.
- `emotion_classifier_binary.py` - Contains the binary emotion classifier.
- `emotion_classifier_multi.py` - Contains the multi-class emotion classifier.
- `keyword_spotter.py` - Contains the Keyword Spotter module.
- `main.py` - Contains the code that starts up the program.
- `python_vad.py` - Contains the Voice Activity Detection code and the logic for executing the distress detection modules in a new thread once an audio event has been detected.
- `repetitive_speech_detector.py` - Contains the Repetitive Speech Detector module.
- `speech.py` - Contains the speech recognition module.
- `testing.py` - Contains functions that were used to test the modules of the program.
## Usage

To use this, ensure that all the relevant packages have been installed by PIP package manager. There's probably a better way, but I would just keep trying to run the program to see which packages it's missing and keep installing them until the program is happy.

To run the program, use `python main.py`

You will require a microphone input to speak into it. Otherwise, you could use a virtual microphone for it to listen to the audio coming directly from your PC. Handy for running during YouTube videos and the like.

This will continuously listen for a person speaking, begin recording, wait for the person to finish speaking, end recording and save WAV file to disk. The distress detection modules are then triggered in another thread, while the program will go back to listening for the next audio event.

The decision from each module along with the overall decision are printed to the console and also saved to an XML file.

You can find a few of the outputted XML files in the `decision_output` folder. A pre-trained Support Vector Machine model can be found in the `models` folder.


## Modules

### Repetitive Speech Detector
This takes in text input from the speech recognition module and checks to see if it is similar to any previously spoken texts. It will output a distress score depending on the number of repeated occurrences. Each new text input is then stored in the database for future runs. The code for this is in the `repetitive_speech_detector.py` file.

To create for testing purposes, use `rsd = RepetitiveSpeechDetector(dataset)` where `dataset` contains an initial list of sentences that need to be checked. To run the detection, use the `rsd.check(text)` function. This will return a `DistressScore` object.

### Keyword Spotter
This takes in text input from the speech recognition module and reads through it to see if any of the specified keywords are present. The code for this is in the `keyword_spotter.py` file.

To create for testing purposes, use `kws = KeywordSpotter()`. Keywords to check can be set using the `kws.set_keywords(keyword_list)` function. To run the spotter, use the `kws.check(text)` function where `text` is the input sentence and not just a single keyword, although it will work with a single keyword too, but that is not the intention. This will return a `DistressScore` object.

### Emotion Classifier

This takes in a WAV audio file location and runs the specified classification module to detect whether the voice is thought to be in-distress or not in-distress. There are two versions: binary and multi-class. The binary model is trained by labelling the training set with `in-distress` or `not-distress` (actually 1 or 0), while the multi-class labels for each of the seven universal emotions. Both versions will output a binary distress value in the end as a `DistressScore` object.

To create a new one for testing purposes:
`emc = EmotionClassifierBinary.from_new(classifier=emotion_classifier_binary.SUPPORT_VECTOR_MACHINE)`
The classifier can be changed by replacing `SUPPORT_VECTOR_MACHINE` with one of the other constants that can be found in the file.

or to use an existing pre-trained model:
`emc = EmotionClassiferBinary.from_existing(model_file_path)`

Then to run prediction on a single audio file, use `emc.predict(file_path)`.


