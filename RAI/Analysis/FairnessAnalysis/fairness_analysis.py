from RAI.Analysis import Analysis
from RAI.AISystem import AISystem
import os


class FairnessAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None  # calculate result
        self.values = ai_system.get_metric_values()[dataset]['group_fairness']
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {'statistical_parity_difference': 0.1 >= self.values['statistical_parity_difference'] >= -0.1,
                  'equal_opportunity_difference': 0.1 >= self.values['equal_opportunity_difference'] >= -0.1,
                  'average_odds_difference': 0.1 >= self.values['average_odds_difference'] >= -0.1,
                  'disparate_impact_ratio': 1.25 >= self.values['disparate_impact_ratio'] >= 0.8}
        return result

    def to_string(self):
        failures = 0
        total = 0
        for val in self.result:
            total += 1
            failures += not self.result[val]
        result = "==== Group Fairness Analysis Results ====\n"
        result += str(total - failures) + " of " + str(total) + " tests passed.\n"
        names = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_odds_difference', 'disparate_impact_ratio']
        conditions = ['between 0.1 and -0.1' for _ in range(4)]
        conditions.append('between 1.25 and 0.8')
        metric_info = self.ai_system.get_metric_info()['group_fairness']
        for i in range(len(names)):
            result += '\n' + metric_info[names[i]]['display_name'] + ' Test:\n'
            result += 'This metric is ' + metric_info[names[i]]['explanation'] + '\n'
            result += "It's value of " + str(self.values[names[i]]) + " is "
            if self.result[names[i]]:
                result += "between " + conditions[i] + " indicating that there is fairness.\n"
            else:
                result += "not between " + conditions[i] + " indicating that there is unfairness.\n"
        return result

    def to_html(self):
        pass
