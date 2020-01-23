import difflib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

backlog = ["Hello mate being motherfucking stupid prick today. Wanker",
           "Go fuck yourself prick",
           "Hello mate today",
           "Today you are a prick",
           "You are being a prick today mate hello"]

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
    processed = remove_stop_words(text)
    print("Processed: " + processed)
    similar_occurences = 0

    for i in range(len(backlog)):
        if is_similar(processed, backlog[i]):
            similar_occurences += 1
            print("Found similar text (" + processed + ") and (" + backlog[i] + ")")

    backlog.append(processed)
    return similar_occurences