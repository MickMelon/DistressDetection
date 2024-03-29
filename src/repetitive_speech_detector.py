import difflib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import namedtuple
from distress_score import DistressScore

# A named tuple to allow for a structured return type
RepetitiveSpeechDetectorResult = namedtuple("RepetitiveSpeechDetectorResult",
                                     "distress_score processed_input matching_sentences")


# Text based repetitive speech detector
class RepetitiveSpeechDetector:
    backlog = []

    # Fill the backlog with test data
    def __fill_backlog(self):
        fill_with = ["The first three are not sentences because they do not contain a verb",
                     "They need an additional clause so as to form a complete sentence and be understood.",
                     "A compound sentence contains two or more clauses of equal status",
                     "You can’t check that an email is sent out on the customer’s 65th Birthday",
                     "But we don’t always randomise enough data and our test data becomes stale and etc. etc."]

        for i in range(len(fill_with)):
            self.backlog.append(self.__preprocess(fill_with[i]))

    # Remove stop words from a piece of text
    def __remove_stop_words(self, text):
        stop_words = set(stopwords.words('english'))
        text_words = nltk.word_tokenize(text)
        filtered_sentence = [w for w in text_words if not w in stop_words]

        final = ''
        for i in range(len(filtered_sentence)):
            final += filtered_sentence[i]
            final += ' '

        return final

    # Carry out lemmatisation on the input text
    def __lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()
        text_words = nltk.word_tokenize(text)
        for word in text_words:
            if word in "?:!.,;":
                text_words.remove(word)

        lemmatized_words = []
        for word in text_words:
            lemmatized_words.append(lemmatizer.lemmatize(word, pos="v"))

        lemmatized_sentence = " ".join(lemmatized_words)
        return lemmatized_sentence

    # Carry out all the preprocessing steps
    def __preprocess(self, text):
       # text = self.__remove_stop_words(text)
        text = self.__lemmatize(text)
        return text

    # Interface function, checks if the input is a repeat of what has been previously said
    def check(self, text):
        # If the text is empty, immediately return no-distress because there
        # is nothing to process
        if text == "":
            return RepetitiveSpeechDetectorResult(
                distress_score=DistressScore.NONE,
                processed_input=text,
                matching_sentences=[]
            )

        # Preprocess the text and get the processed words into an array
        processed = self.__preprocess(text)
        processed_words = processed.split()

        # Intialise the matching sentences array
        matching_sentences = []

        # Compare input with each item in backlog
        for i in range(len(self.backlog)):
            matching_words = 0
            non_matching_words = 0
            backlog_sentence = self.backlog[i]
            backlog_item_words = backlog_sentence.split()

            # Loop through each word in the backlog item
            for i in range(len(backlog_item_words)):
                backlog_word = backlog_item_words[i]
                # Check if backlog item word matches any input words
                if backlog_word in processed_words:
                    matching_words += 1
                else:
                    non_matching_words += 1

            # If 60% of words match, it is a matching sentence
            per = matching_words * (100 / len(backlog_item_words))
            if per >= 60:
                matching_sentences.append(backlog_sentence)

        # Append the processed text to the backlog for future processing
        self.backlog.append(processed)

        # Calculate the distress score depending on how many matching sentences there are
        qty = len(matching_sentences)
        if qty > 4:
            score = DistressScore.HIGH
        elif qty > 2:
            score = DistressScore.MEDIUM
        elif qty > 0:
            score = DistressScore.LOW
        else:
            score = DistressScore.NONE

        # Return the final distress result
        return RepetitiveSpeechDetectorResult(
            distress_score=score,
            processed_input=processed,
            matching_sentences=matching_sentences
        )
