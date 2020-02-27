from enum import Enum
import xml.etree.cElementTree as etree
import xml.dom.minidom
import time


# Enum to show the distress score
class DistressScore(Enum):
    NONE = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3


# Makes a decision on the distress level depending on the params given
def make_decision(text, number_keywords_spotted, distress_classifier_result, repetitive_speech_detected):
    # Calculate score
    score = 0

    score = score + (number_keywords_spotted * 10)

    if repetitive_speech_detected:
        score = score + 20

    result = DistressScore.NONE

    # Decide distress level from score
    if score > 80:
        result = DistressScore.HIGH

    if score > 60:
        result = DistressScore.MEDIUM

    if score > 40:
        result = DistressScore.LOW

    save_decision(result, text, number_keywords_spotted, distress_classifier_result, repetitive_speech_detected)

    return result


# Saves the decision to an XML file
def save_decision(result, text, number_keywords_spotted, distress_classifier_result, repetitive_speech_detected):
    root = etree.Element("Result")

    # Time
    etree.SubElement(root, "Time").text = str(time.time())
    etree.SubElement(root, "SpokenText").text = text

    # Emc
    emc = etree.SubElement(root, "EMC")
    etree.SubElement(emc, "FirstEmotion").text = distress_classifier_result.highest_name
    etree.SubElement(emc, "FirstScore").text = str(distress_classifier_result.highest_score)
    etree.SubElement(emc, "SecondEmotion").text = distress_classifier_result.second_highest_name
    etree.SubElement(emc, "SecondScore").text = str(distress_classifier_result.second_highest_score)

    # Kws
    kws = etree.SubElement(root, "KWS")
    etree.SubElement(kws, "NoSpotted").text = str(number_keywords_spotted)

    # Rsd
    rsd = etree.SubElement(root, "RSD")
    etree.SubElement(rsd, "IsRepeated").text = str(repetitive_speech_detected)

    # Decision
    etree.SubElement(root, "Decision").text = str(result)

    raw = etree.tostring(root, encoding='unicode')
    pretty = xml.dom.minidom.parseString(raw).toprettyxml(indent='    ')
    with open('decision_output/filename.xml', 'w') as f:
        f.write(pretty)