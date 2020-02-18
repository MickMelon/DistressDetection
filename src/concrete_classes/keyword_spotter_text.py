from base_classes.base_keyword_spotter import KeywordSpotter

# Text based keyword spotter
class KeywordSpotterText(KeywordSpotter):
    keywords = []

    # Construct a keyword spotter with some test keywords
    def __init__(self):
        keywords = ["background", "depth", "large", "stockings", "john", "keyword", "test", "sell"]

    # Construct a keyword spotter with the specified keywords
    def __init__(self, keywords):
        self.keywords = keywords

    # Interface to the keyword spotter. Takes in input text and checks if it
    # contains any of the keywords. Returns the number of keywords spotted.
    def check(self, input):
        words_spotted = 0
        split_text = input.split()

        for i in range(len(self.keywords)):
            if self.keywords[i] in split_text:
                print("Spotted word " + self.keywords[i])
                words_spotted += 1

        return words_spotted
