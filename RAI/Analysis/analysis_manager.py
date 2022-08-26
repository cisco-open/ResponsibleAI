from RAI.AISystem import AISystem
from .analysis_registry import registry


class AnalysisManager:

    def __init__(self):
        pass

    def _get_available_analysis(self, ai_system: AISystem, dataset: str):
        compatible_groups = {}
        for group in registry:
            if registry[group].is_compatible(ai_system, dataset):
                compatible_groups[group] = registry[group]
        return compatible_groups

    def get_available_analysis(self, ai_system: AISystem, dataset: str):
        return [name for name in self._get_available_analysis(ai_system, dataset)]

    def run_analysis(self, ai_system: AISystem, dataset: str, analysis_names, tag=None, connection=None):
        available_analysis = self._get_available_analysis(ai_system, dataset)
        result = {}
        if isinstance(analysis_names, str):
            analysis_names = [analysis_names]
        for analysis_name in analysis_names:
            if analysis_name in available_analysis:
                analysis_result = available_analysis[analysis_name](ai_system, dataset, tag)
                analysis_result.set_connection(connection)
                analysis_result.initialize()
                result[analysis_name] = analysis_result
        return result

    def run_all(self, ai_system: AISystem, dataset: str, tag: str):
        return self.run_analysis(ai_system, dataset, self.get_available_analysis(ai_system, dataset), tag)
