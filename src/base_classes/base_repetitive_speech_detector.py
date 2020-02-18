from abc import ABC, abstractmethod

class RepetitiveSpeechDetector(ABC):
    @abstractmethod
    def check(self, input):
        pass
