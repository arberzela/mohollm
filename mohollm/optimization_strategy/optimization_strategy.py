from abc import ABC, abstractmethod


class OptimizationStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    def optimize_threaded(self):
        return self.optimize()
