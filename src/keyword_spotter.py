# Text based keyword spotter
class KeywordSpotter:
    keywords = []

    # Construct a keyword spotter with some test keywords
    def __init__(self):
        self.keywords = ["background", "depth", "large", "stockings", "john", "keyword", "test", "sell"]

    def set_keywords(self, keywords):
        self.keywords = keywords

    # Interface to the keyword spotter. Takes in input text and checks if it
    # contains any of the keywords. Returns the number of keywords spotted.
    def check(self, input):
        print(f'Received input {input}')
        words_spotted = 0
        split_text = input.split()

        for i in range(len(self.keywords)):
            if self.keywords[i] in split_text:
                print("Spotted word " + self.keywords[i])
                words_spotted += 1

        print(f'Words spotted: {words_spotted}')
        return words_spotted
