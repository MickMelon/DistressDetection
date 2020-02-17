from enum import Enum


class DistressScore(Enum):
    NONE = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3


def make_decision(number_keywords_spotted, distress_classifier_result, repetitive_speech_detected, shouting_detected):
    # Calculate score
    score = 0

    score = score + (number_keywords_spotted * 10)

    if distress_classifier_result:
        score = score + 50

    if repetitive_speech_detected:
        score = score + 20

    if shouting_detected:
        score = score + 20

    # Decide distress level from score
    if score > 80:
        return DistressScore.HIGH

    if score > 60:
        return DistressScore.MEDIUM

    if score > 40:
        return DistressScore.LOW

    return DistressScore.NONE