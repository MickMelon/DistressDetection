import difflib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

backlog = []

# Input text is received
# Compared with previous texts to see if similar
# If so, return similarity score
# If not, return 0
# Add input text to array
# Remove oldest text in array


def fill_backlog():
    fill_with = ["The first three are not sentences because they do not contain a verb",
                 "They need an additional clause so as to form a complete sentence and be understood.",
                 "A compound sentence contains two or more clauses of equal status",
                 "You can’t check that an email is sent out on the customer’s 65th Birthday",
                 "But we don’t always randomise enough data and our test data becomes stale and etc. etc."]

    for i in range(len(fill_with)):
        backlog.append(remove_stop_words(fill_with[i]))

    print("Filled backlog with " + str(len(backlog)) + " sentences")


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_sentence = [w for w in tokens if not w in stop_words]

    final = ''
    for i in range(len(filtered_sentence)):
        final += filtered_sentence[i]
        final += ' '

    return final


def is_similar(sentence1, sentence2):
    seq = difflib.SequenceMatcher(None, sentence1.lower(), sentence2.lower())
    d = seq.ratio() * 100
    return d > 80


def run(text):
    fill_backlog()

    processed = remove_stop_words(text)
    print("Processed: " + processed)
    similar_occurences = 0

    for i in range(len(backlog)):
        if processed in backlog[i]:
            similar_occurences += 1
            print("Found similar text (" + processed + ") and (" + backlog[i] + ")")

    backlog.append(processed)
    backlog.remove(backlog[0])
    print("Backlog 0 is now " + backlog[0])

    return similar_occurences