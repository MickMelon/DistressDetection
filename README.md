# Speech Distress Detection Program

University Honours Project

How Effective is Speech Analysis in Detecting a Dementia Patient in Distress?

## Description

The number of dementia patients is increasing in an ever-growing population causing a rise in NHS costs and care time. Emerging technologies are being used to enhance care to allow a person to live at home for longer. How can these systems be improved by using speech analysis to detect a dementia patient in distressand to what extent would speech analysis be effective?

This project aimedto improve assistive living environments by developing a prototype system that will use speech analysis to identify common audible signs of distress in the speech of a dementia patient.

The prototype system developed incorporated text analysis and machine learning techniques to detect common audible distress symptomsdisplayed by a dementia patient. Three distress detection modules were created: keyword spotting, repetitive speech detection, and emotion classification. The former two worked from text outputted by speech recognition whilst the later used machine learning techniques. Each module was executed after voice activity was detected from a microphone input before giving their own distress scores to a decision system which used module weightings to decide the overall distress score.

Due to ethical testing limitations, the system was tested on various datasets and sources that do not contain genuinely distressful sounds, with the aim of proving the concept that distress can indeed be detected in the speech of a dementia patient. This concept was proven as specific words such as a carer's name or swear words, repetitive phrases and questioning, and negative emotions could all be detected with a reasonably high accuracy. 

The results have shown that this is a very prospecting area that should be further investigated, ideally with real world trials. Additional modules could be added, for example detection of phrases containing greater than one word and repetitive keyword detection.Further investigation on various feature extraction methods and pre-processing steps would be beneficial in improving the emotion classification aspect. The main priority for future work would be to work closely with dementia patients and their carers to gather a more personal insight and to conduct tests on real-world scenarios in which a dementia patient is undergoing genuine distress.

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


