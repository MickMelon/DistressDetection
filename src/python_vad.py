"""
#--- Steve Cox --- 1/10/19
# Copyright (c) Stef van der Struijk
# License: GNU Lesser General Public License

# Modified code to play sound from buffer recording
# Added code to wait till sound is finished play so no echo occurs

# Modification of:
# https://github.com/wiseman/py-webrtcvad (MIT Copyright (c) 2016 John Wiseman)
# https://github.com/wangshub/python-vad (MIT Copyright (c) 2017 wangshub)

Requirements:
+ pyaudio - `pip install pyaudio`
+ py-webrtcvad - `pip install webrtcvad`
"""
import threading
import speech
import webrtcvad
import collections
import sys
import pyaudio
from array import array
import wave
import time
import contextlib
from emotion_classifier import EmotionClassifier
from keyword_spotter import KeywordSpotter
from repetitive_speech_detector import RepetitiveSpeechDetector
import decision
from decision import DistressScore
import os

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
PADDING_DURATION_MS = 1500  # 1 sec judgement
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read
CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)

# --- Steve Cox
NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)

NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2
START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

emc = EmotionClassifier.from_existing('models/mlp_classifier.model')
kws = KeywordSpotter()
rsd = RepetitiveSpeechDetector(fill_backlog=True)


def start():
    vad = webrtcvad.Vad(1)

    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     start=False,
                     input_device_index=2,
                     frames_per_buffer=CHUNK_SIZE)

    got_a_sentence = False
    while True:
        ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
        triggered = False
        voiced_frames = []
        ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
        ring_buffer_index = 0

        ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
        ring_buffer_index_end = 0
        buffer_in = ''
        # WangS
        raw_data = array('h')
        index = 0
        start_point = 0
        start_time = time.time()
        print("* recording: ")
        stream.start_stream()

        while not got_a_sentence:
            chunk = stream.read(CHUNK_SIZE)
            # add WangS
            raw_data.extend(array('h', chunk))
            index += CHUNK_SIZE
            time_use = time.time() - start_time

            active = vad.is_speech(chunk, RATE)

           # sys.stdout.write('1' if active else '_')
            ring_buffer_flags[ring_buffer_index] = 1 if active else 0
            ring_buffer_index += 1
            ring_buffer_index %= NUM_WINDOW_CHUNKS

            ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
            ring_buffer_index_end += 1
            ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

            # start point detection
            if not triggered:
                ring_buffer.append(chunk)
                num_voiced = sum(ring_buffer_flags)
                if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                    sys.stdout.write(' Open ')
                    triggered = True
                    start_point = index - CHUNK_SIZE * 20  # start point
                    ring_buffer.clear()
            # end point detection
            else:
                ring_buffer.append(chunk)
                num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)

                if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or time_use > 10:
                    sys.stdout.write(' Close ')
                    triggered = False
                    got_a_sentence = True

            sys.stdout.flush()

        sys.stdout.write('\n')

        stream.stop_stream()
        print("* done recording")
        got_a_sentence = False

        # write to file
        raw_data.reverse()
        for index in range(start_point):
            raw_data.pop()

        raw_data.reverse()
        raw_data = normalize(raw_data)
        wav_data = raw_data[44:len(raw_data)]

        write_wave(f'vad_output/{start_time}.wav', wav_data, RATE)
        voice_detected(f'vad_output/{start_time}.wav')

    stream.close()


def voice_detected(file_name):
    t = threading.Thread(target=analyse_speech, args=[file_name])
    t.start()


def analyse_speech(file_name):
    kws_result = 0
    rsd_result = 0
    text = speech.speech_to_text(file_name)

    if not text == "":
        kws_result = kws.check(text)
        rsd_result = rsd.check(text)
    else:
        print("No speech detected in voice clip. Skipping KWS and RSD...")

    emc_result = emc.predict_binary(file_name)

    decision_result = decision.make_decision(text, kws_result, emc_result, rsd_result)
    print(f"The score is {decision_result}")

    # Only keep the sound file if distressed was detected
    if decision_result is DistressScore.NONE:
        os.remove(file_name)


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 32767  # 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


