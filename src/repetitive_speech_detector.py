import difflib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Text based repetitive speech detector
class RepetitiveSpeechDetector:
    backlog = []

    # Construct with auto filled backlog
    def __init__(self, fill_backlog=False):
        if fill_backlog:
            self.__fill_backlog()

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
        text_words = nltk.word_tokenize(text)
        filtered_sentence = [w for w in text_words if not w in stop_words]

        final = ''
        for i in range(len(filtered_sentence)):
            final += filtered_sentence[i]
            final += ' '

        return final

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

    def __preprocess(self, text):
        text = self.__remove_stop_words(text)
        text = self.__lemmatize(text)
        return text

    # Interface function, checks if the input is a repeat of what has been previously said
    def check(self, input):
        processed = self.__preprocess(input)
        processed_words = processed.split()

        print(f"Input: {input}")
        print(f"Processed: {processed}")

        matching_sentences = 0

        # Compare input with each item in backlog
        for i in range(len(self.backlog)):
            matching_words = 0
            non_matching_words = 0
            backlog_item_words = self.backlog[i].split()

            # Loop through each word in the backlog item
            for i in range(len(backlog_item_words)):
                backlog_word = backlog_item_words[i]
                # Check if backlog item word matches any input words
                if backlog_word in processed_words:
                    matching_words += 1
                else:
                    non_matching_words += 1

            # If 80% of words match, it is a matching sentence
            per = matching_words * (100/len(backlog_item_words))
            print(f"Percentage is {per}")

            if per > 80:
                matching_sentences += 1

        self.backlog.append(processed)
        self.backlog.remove(self.backlog[0])

        return matching_sentences
