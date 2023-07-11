# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


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
        """
        :param AISystem: input the ai_system obj
        :param dataset: input the dataset

        :Returns: list.

        Returns the lists of analysis data
        """
        return [name for name in self._get_available_analysis(ai_system, dataset)]

    def run_analysis(self, ai_system: AISystem, dataset: str, analysis_names, tag=None, connection=None):
        """
        :param AISystem: input the ai_system obj
        :param dataset: input the dataset
        :param tag: By default None else given tag Name
        :param analysis_names: analysis_names data set
        :param connection: By default None

        :Returns: Dict.

        Returns the analysis data result analysis
        """
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
        """
        :param AISystem: input the ai_system obj
        :param dataset: input the dataset
        :param tag: By default None else given tag Name

        :Returns: None.

        Returns the analysis data result analysis

        """
        return self.run_analysis(ai_system, dataset, self.get_available_analysis(ai_system, dataset), tag)
