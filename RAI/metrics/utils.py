
import json
import os


class AvailableMetrics:

    @staticmethod
    def _parse_file(file_path):
        with open(file_path, 'r') as readfile:
            data = json.loads(readfile.read())
            return {
                'name': data['name'],
                'display_name': data['display_name'],
                'metrics': {
                    metric: {'display_name': values['display_name'], 'type': values['type']}
                    for metric, values in data['metrics'].items()
                }
            }

    @classmethod
    def _get_available_metrics(cls) -> dict:
        metrics = []
        current_dir = os.path.dirname(os.path.realpath(__file__))
        for subfolder in [f.path for f in os.scandir(current_dir) if f.is_dir()]:
            for file_path in os.listdir(subfolder):
                if os.path.isfile(os.path.join(subfolder, file_path)):
                    if file_path.endswith('.json'):
                        metrics.append(cls._parse_file(os.path.join(subfolder, file_path)))
        return metrics

    @classmethod
    def get(cls):
        return cls._get_available_metrics()

    @classmethod
    def display(cls):
        metrics = cls._get_available_metrics()
        print('\nAvailable Metrics:\n'
              '----------------------\n\n'
              f'{json.dumps(metrics, indent=4)}\n'
              '----------------------\n\n'
              'Examples: \n'
              '  (group1.metric1 + group2.metric2 * 3)\n'
              '  group_fairness.average_odds_difference * 0.2 + 1 + adversarial_robustness.inaccuracy * 12.5\n'
              '  individual_fairness.generalized_entropy_index * 14 + individual_fairness.coefficient_of_variation * 3')
