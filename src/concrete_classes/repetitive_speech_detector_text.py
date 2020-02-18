import difflib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from base_classes.base_repetitive_speech_detector import RepetitiveSpeechDetector


# Text based repetitive speech detector
class RepetitiveSpeechDetectorText(RepetitiveSpeechDetector):
    backlog = []

    # Construct with auto filled backlog
    def __init__(self):
        self.__fill_backlog()

    # Construct with specified backlog
    def __init__(self, backlog):
        self.backlog = backlog

    # Fill the backlog with test data
    def __fill_backlog(self):
        fill_with = ["The first three are not sentences because they do not contain a verb",
                     "They need an additional clause so as to form a complete sentence and be understood.",
                     "A compound sentence contains two or more clauses of equal status",
                     "You can’t check that an email is sent out on the customer’s 65th Birthday",
                     "But we don’t always randomise enough data and our test data becomes stale and etc. etc."]

        for i in range(len(fill_with)):
            self.backlog.append(self.__remove_stop_words(fill_with[i]))

        print("Filled backlog with " + str(len(self.backlog)) + " sentences")

    # Remove stop words from a piece of text
    def __remove_stop_words(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_sentence = [w for w in tokens if not w in stop_words]

        final = ''
        for i in range(len(filtered_sentence)):
            final += filtered_sentence[i]
            final += ' '

        return final

    # Check if two sentences are similar
    def __is_similar(self, sentence1, sentence2):
        seq = difflib.SequenceMatcher(None, sentence1.lower(), sentence2.lower())
        d = seq.ratio() * 100
        return d > 80

    # Interface function, checks if the input is a repeat of what has been previously said
    def check(self, input):
        self.fill_backlog()

        processed = self.__remove_stop_words(input)
        print("Processed: " + processed)
        similar_occurences = 0

        for i in range(len(self.backlog)):
            if processed in self.backlog[i]:
                similar_occurences += 1
                print("Found similar text (" + processed + ") and (" + self.backlog[i] + ")")

        self.backlog.append(processed)
        self.backlog.remove(self.backlog[0])

        return similar_occurences