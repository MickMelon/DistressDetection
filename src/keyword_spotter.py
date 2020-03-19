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
    def check(self, input):
        if input == "":
            return KeywordSpotterResult(
                distress_score=DistressScore.NONE,
                processed_input=input,
                matching_keywords=dict(),
            )

        word_dict = dict()
        processed = self.__lemmatize(input)
        split_text = processed.lower().split()
        total_spotted = 0

        for i in range(len(self.keywords)):
            keyword = self.keywords[i]

            if keyword.lower() in split_text:
                total_spotted += 1
                if keyword in word_dict:
                    word_dict[keyword] += 1
                else:
                    word_dict[keyword] = 1

        if total_spotted < 2:
            score = DistressScore.NONE
        elif total_spotted < 4:
            score = DistressScore.LOW
        elif total_spotted < 6:
            score = DistressScore.MEDIUM
        else:
            score = DistressScore.HIGH

        return KeywordSpotterResult(
            distress_score=score,
            processed_input=processed,
            matching_keywords=word_dict,
        )
