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

__all__ = ['Model']
import RAI.dataset


class Model:
    """
    Model is RAIs abstraction for the ML Model performing inferences.
    When constructed, models are optionally passed the name, the models functions for inferences,
    its name, the model, its optimizer, its loss function, its class and a description.
    Attributes of the model are used to determine which metrics are relevant.
    """

    def __init__(self, output_features=None, predict_fun=None, predict_prob_fun=None,
                 generate_text_fun=None, generate_image_fun=None, name=None, display_name=None,
                 agent=None, loss_function=None, optimizer=None, model_class=None, description=None) -> None:
        assert name is not None, "Please provide a model name"
        self.output_types = {}
        self.predict_fun = predict_fun
        if isinstance(output_features, RAI.dataset.Feature):
            output_features = [output_features]
        elif output_features is not None:
            assert "output_features must be a Feature or array of Features"
        self.output_features = output_features if output_features is not None else []
        if predict_fun is not None:
            self.output_types["predict"] = predict_fun
        self.predict_prob_fun = predict_prob_fun
        if predict_prob_fun is not None:
            self.output_types["predict_proba"] = predict_prob_fun
        self.generate_text_fun = generate_text_fun
        self.generate_image_fun = generate_image_fun
        if generate_text_fun is not None:
            self.output_types["generate_text"] = generate_text_fun
        if generate_image_fun is not None:
            self.output_types["generate_image"] = generate_image_fun
        self.name = name
        self.display_name = name
        if display_name is not None:
            self.display_name = display_name
        self.agent = agent
        self.input_type = None
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_class = model_class
        self.description = description
