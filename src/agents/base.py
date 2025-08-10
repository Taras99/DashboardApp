from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def act(self, observation) -> int:
        """Return action given observation. Actions expected to be -1,0,1"""
        raise NotImplementedError

    def reset(self):
        """Optional reset between episodes"""
        pass
