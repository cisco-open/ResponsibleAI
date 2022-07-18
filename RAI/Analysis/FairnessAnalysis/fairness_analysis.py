from RAI.Analysis import Analysis
from RAI.AISystem import AISystem
import os


class FairnessAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None  # calculate result

    def initialize(self):
        if self.result is None:
            self.result = {"main_value": "test"}

    def to_string(self):
        return self.result["main_value"]

    def to_html(self):
        pass
