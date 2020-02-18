from abc import ABC, abstractmethod

class EmotionClassifier(ABC):
    @abstractmethod
    def predict(self, input):
        pass
