keywords = ["background", "depth", "large", "stockings", "john", "keyword", "test"]

def spot_keyword(text):
    words_spotted = 0
    for i in range(len(keywords)):
        if keywords[i] in text:
            print("Spotted word " + keywords[i])
            words_spotted += 1

    return words_spotted
