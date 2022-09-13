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


__all__ = ['all_complexity_classes', 'all_task_types', 'all_data_types',
           'all_output_requirements', 'all_dataset_requirements', 'all_metric_types']

all_complexity_classes = {"constant", "linear", "multi_linear", "polynomial", "exponential"}
all_task_types = {"binary_classification", "classification", "clustering", "regression", "generate"}
all_data_types = {"numeric", "image", "text"}
all_output_requirements = {"predict", "predict_proba", "generate_text", "generate_image"}
all_dataset_requirements = {"X", "y", "sensitive_features"}
all_metric_types = {"numeric", "Dict", "multivalued", "other", "vector", "vector-dict", "Matrix", "boolean", "Boolean"}
