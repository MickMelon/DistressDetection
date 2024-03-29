import xml.etree.cElementTree as etree
import xml.dom.minidom
from datetime import datetime
from distress_score import DistressScore

from keyword_spotter import KeywordSpotterResult
from emotion_classifier_binary import EmotionClassifierBinaryResult
from repetitive_speech_detector import RepetitiveSpeechDetectorResult

# The weighting for each module for the final distress score decision
KWS_WEIGHT = 30
EMC_WEIGHT = 40
RSD_WEIGHT = 30


# Makes a decision on the distress level depending on the params given
def make_decision(text,
                  kws_result: KeywordSpotterResult,
                  emc_result: EmotionClassifierBinaryResult,
                  rsd_result: RepetitiveSpeechDetectorResult):

    # Get the score from each module
    kws_score = kws_result.distress_score
    emc_score = emc_result.distress_score
    rsd_score = rsd_result.distress_score

    score_percentage = 0

    # Get KWS final score
    if kws_score is DistressScore.LOW:
        score_percentage += round(KWS_WEIGHT / 3)
    elif kws_score is DistressScore.MEDIUM:
        score_percentage += round((KWS_WEIGHT / 3) * 2)
    elif kws_score is DistressScore.HIGH:
        score_percentage += round(KWS_WEIGHT)

    # Get EMC final score
    if emc_score is DistressScore.LOW:
        score_percentage += round(EMC_WEIGHT / 3)
    elif emc_score is DistressScore.MEDIUM:
        score_percentage += round((EMC_WEIGHT / 3) * 2)
    elif emc_score is DistressScore.HIGH:
        score_percentage += round(EMC_WEIGHT)

    # Get RSD final score
    if rsd_score is DistressScore.LOW:
        score_percentage += round(RSD_WEIGHT / 3)
    elif rsd_score is DistressScore.MEDIUM:
        score_percentage += round((RSD_WEIGHT / 3) * 2)
    elif rsd_score is DistressScore.HIGH:
        score_percentage += round(RSD_WEIGHT)

    # Get overall distress score
    if score_percentage > 75:
        overall_distress_score = DistressScore.HIGH
    elif score_percentage > 55:
        overall_distress_score = DistressScore.MEDIUM
    elif score_percentage > 35:
        overall_distress_score = DistressScore.LOW
    else:
        overall_distress_score = DistressScore.NONE

    # Save the decision to XML
    save_decision(text, overall_distress_score, score_percentage, kws_result, emc_result, rsd_result)
    print_to_console(text, overall_distress_score, score_percentage, kws_result, emc_result, rsd_result)

    # Return the final overall distress score from the calculated combination
    # of all the distress modules
    return overall_distress_score


def print_to_console(text, overall_distress_score, score_percentage, kws_result, emc_result, rsd_result):
    print("== [Decision System] ==")
    print(f"Processed Text: {rsd_result.processed_input}")
    print(f"Overall Score: {overall_distress_score}")
    print(f"Score Percentage: {score_percentage}")
    print(f"KWS Score: {kws_result.distress_score}")
    print(f"EMC Score: {emc_result.distress_score}")
    print(f"RSD Score: {rsd_result.distress_score}")
    print("=======================")


# Save the final decision with all details from each distress module
def save_decision(text, overall_distress_score, score_percentage, kws_result, emc_result, rsd_result):
    current_time = f'{datetime.now():%Y-%m-%d %H-%M-%S}'

    root = etree.Element("Result")

    # Time
    etree.SubElement(root, "OverallScore").text = str(overall_distress_score)
    etree.SubElement(root, "ScorePercentage").text = str(score_percentage)
    etree.SubElement(root, "Time").text = current_time
    etree.SubElement(root, "SpokenText").text = text

    # KWS
    kws = etree.SubElement(root, "KWS")
    etree.SubElement(kws, "DistressScore").text = str(kws_result.distress_score)
    etree.SubElement(kws, "ProcessedInput").text = kws_result.processed_input
    keywords = etree.SubElement(kws, "MatchingKeywords")
    for keyword in kws_result.matching_keywords:
        single_keyword = etree.SubElement(keywords, "SingleKeyword")
        etree.SubElement(single_keyword, "Keyword").text = keyword
        etree.SubElement(single_keyword, "NoOccurrences").text = str(kws_result.matching_keywords[keyword])

    # EMC
    emc = etree.SubElement(root, "EMC")
    etree.SubElement(emc, "DistressScore").text = str(emc_result.distress_score)
    etree.SubElement(emc, "InDistressProba").text = str(emc_result.distress_proba)
    etree.SubElement(emc, "NoDistressProba").text = str(emc_result.no_distress_proba)

    # RSD
    rsd = etree.SubElement(root, "RSD")
    etree.SubElement(rsd, "DistressScore").text = str(rsd_result.distress_score)
    etree.SubElement(rsd, "ProcessedInput").text = rsd_result.processed_input
    matching_sentences = etree.SubElement(rsd, "MatchingSentences")
    for sentence in rsd_result.matching_sentences:
        etree.SubElement(matching_sentences, "Sentence").text = sentence

    # Write to file
    raw = etree.tostring(root, encoding='unicode')
    pretty = xml.dom.minidom.parseString(raw).toprettyxml(indent='    ')
    with open(f'decision_output/{current_time}.xml', 'w') as f:
        f.write(pretty)