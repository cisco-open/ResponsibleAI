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


from abc import ABC, abstractmethod
from .display_registry import register_class


class DisplayElement(ABC):
    def __init__(self, name):
        self.requires_tag_chooser = False
        self._name = name
        self._data = {}
        self._metric_object = {}

    def __init_subclass__(cls, requirements=[], **kwargs):
        super().__init_subclass__(**kwargs)
        cls.requirements = requirements
        register_class(cls.__name__, cls)

    @classmethod
    def get_requirements(cls):
        return cls.requirements

    @abstractmethod
    def append(self, data, tag):
        pass

    @abstractmethod
    def to_string(self):
        pass

    def to_display(self):
        pass
