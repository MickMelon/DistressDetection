import nltk
from nltk.stem import WordNetLemmatizer
from collections import namedtuple
from distress_score import DistressScore

KeywordSpotterResult = namedtuple("KeywordSpotterResult",
                                  "distress_score processed_input matching_keywords")


# Text based keyword spotter
class KeywordSpotter:
    keywords = []

    # Construct a keyword spotter with some test keywords
    def __init__(self):
        self.keywords = ["background", "depth", "large", "stockings", "john", "keyword", "test", "sell"]

    def set_keywords(self, keywords):
        self.keywords = keywords

    def __lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()
        text_words = nltk.word_tokenize(text)
        for word in text_words:
            word = word.lower()

            if word in "?:!.,;":
                text_words.remove(word)

        lemmatized_words = []
        for word in text_words:

            word = lemmatizer.lemmatize(word, pos="v")
            lemmatized_words.append(lemmatizer.lemmatize(word, pos="v"))

        lemmatized_sentence = " ".join(lemmatized_words)
        return lemmatized_sentence

    # Interface to the keyword spotter. Takes in input text and checks if it
    # contains any of the keywords. Returns the number of keywords spotted.
    def check(self, text):
        if text == "":
            return KeywordSpotterResult(
                distress_score=DistressScore.NONE,
                processed_input=text,
                matching_keywords=dict()
            )

        word_dict = dict()
        processed = self.__lemmatize(text)
        split_text = processed.lower().split()
        total_spotted = 0

        # Count how many times each keyword appears in the text
        for keyword in self.keywords:
            word_dict[keyword] = split_text.count(keyword.lower())

        if total_spotted > 5:
            score = DistressScore.HIGH
        elif total_spotted > 3:
            score = DistressScore.MEDIUM
        elif total_spotted > 1:
            score = DistressScore.LOW
        else:
            score = DistressScore.NONE

        return KeywordSpotterResult(
            distress_score=score,
            processed_input=processed,
            matching_keywords=word_dict,
        )
