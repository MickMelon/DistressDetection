keywords = ["background", "depth", "large", "stockings", "john", "keyword", "test", "sell"]


def spot_keyword(text):
    words_spotted = 0
    split_text = text.split()

    for i in range(len(keywords)):
        if keywords[i] in split_text:
            print("Spotted word " + keywords[i])
            words_spotted += 1

    return words_spotted