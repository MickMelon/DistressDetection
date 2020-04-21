import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from collections import namedtuple
from distress_score import DistressScore
import re
from contractions import contractions_dict

# A named tuple to allow for a structured return type
KeywordSpotterResult = namedtuple("KeywordSpotterResult",
                                  "distress_score processed_input matching_keywords")


# Text based keyword spotter
class KeywordSpotter:
    keywords = []
    lemmatizer = WordNetLemmatizer()
    stemmer = LancasterStemmer()

    # Construct a keyword spotter with some test keywords
    def __init__(self):
        self.keywords = ["background", "depth", "large", "stockings", "john", "keyword", "test", "sell"]

    # Construct a keyword spotter with specified keywords
    def set_keywords(self, keywords):
        self.keywords = keywords

    # Carry out lemmitisation on the text input
    def __lemmatize(self, text_words):
        lemmatized_words = []
        for word in text_words:
            word = self.lemmatizer.lemmatize(word, pos="v")
            lemmatized_words.append(self.lemmatizer.lemmatize(word, pos="v"))

        return lemmatized_words

    # Expand contractions, this will turn words like "we'll" into "we will"
    def __expand_contractions(self, text):
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contractions_dict.get(match) \
                if contractions_dict.get(match) \
                else contractions_dict.get(match.lower())
            expanded_contraction = expanded_contraction
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    # Remove punctuation from the input text
    def __remove_punctuation(self, text_words):
        for word in text_words:
            if word in "?:!.,;":
                text_words.remove(word)

        return text_words

    # Carry out stem word removal on the input text
    def __stem(self, text_words):
        stemmed_words = []
        for word in text_words:
            word = self.stemmer.stem(word)
            stemmed_words.append(word)

        return stemmed_words

    # Carry out all pre-processing steps on the input text
    def __preprocess(self, text):
        text = text.lower()
        text = self.__expand_contractions(text)
        text_words = nltk.word_tokenize(text)
        text_words = self.__lemmatize(text_words)
        text_words = self.__remove_punctuation(text_words)

        processed_text = ' '.join(text_words)
        return processed_text

    def preprocess(self, text):
        return self.__preprocess(text)

    # Interface to the keyword spotter. Takes in input text and checks if it
    # contains any of the keywords. Returns the number of keywords spotted.
    def check(self, text):
        # If the text is empty, immediately return no-distress because there
        # is nothing to process
        if text == "":
            return KeywordSpotterResult(
                distress_score=DistressScore.NONE,
                processed_input=text,
                matching_keywords=dict()
            )

        # Set up the words list and preprocess the input text ready for
        # keyword spotting to be carried out
        word_dict = dict()
        processed = self.__preprocess(text)
        split_text = processed.lower().split()
        total_spotted = 0

        # Count how many times each keyword appears in the text
        for keyword in self.keywords:
            word_dict[keyword] = split_text.count(keyword.lower())

        # Calculate the distress score
        if total_spotted > 5:
            score = DistressScore.HIGH
        elif total_spotted > 3:
            score = DistressScore.MEDIUM
        elif total_spotted > 1:
            score = DistressScore.LOW
        else:
            score = DistressScore.NONE

        # Return the final distress result
        return KeywordSpotterResult(
            distress_score=score,
            processed_input=processed,
            matching_keywords=word_dict,
        )
