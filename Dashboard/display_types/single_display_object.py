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


from abc import ABCMeta, abstractmethod
from .display_object import DisplayElement


# Single display elements are display elements that can only be shown one at a time
# For example, matrices and images.
class SingleDisplayElement(DisplayElement, metaclass=ABCMeta):
    def __init__(self, name):
        super().__init__(name)
        self.requires_tag_chooser = True

    def __init_subclass__(cls, class_location=None, reqs=[], **kwargs):
        super().__init_subclass__(**kwargs)
        cls.requirements = reqs

    def get_tags(self):
        return self._data["tag"]

    @abstractmethod
    def display_tag_num(self, tag_num):
        pass
