from abc import ABC, abstractmethod

class KeywordSpotter(ABC):
    @abstractmethod
    def check(self, input):
        pass
