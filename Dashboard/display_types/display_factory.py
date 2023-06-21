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


from .display_registry import registry
from .traceable_element import TraceableElement
__all__ = ['get_display', 'is_compatible']


def get_display(metric_name: str, metric_type: str, dbUtils):
    metric_type = metric_type + "Element"
    additional_features = {}
    if metric_type.lower() in registry:
        requirements = registry[metric_type.lower()].get_requirements()
        if 'features' in requirements:
            additional_features['features'] = dbUtils.get_project_info()["features"]
        return registry[metric_type.lower()](metric_name, **additional_features)
    return None


def is_compatible(metric_type: str, requirements: list):
    metric_type += "Element"
    result = metric_type.lower() in registry
    if result and 'Traceable' in requirements:
        result = issubclass(registry[metric_type.lower()], TraceableElement)
    return result
