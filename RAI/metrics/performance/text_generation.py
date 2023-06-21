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


from nltk.translate.bleu_score import corpus_bleu
from RAI.metrics.metric_group import MetricGroup
import os
from torchmetrics.text.rouge import ROUGEScore
import numpy as np


class TextGeneration(MetricGroup, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system) -> None:
        super().__init__(ai_system)
        self.max_samples = 500

    def update(self, data):
        pass

    def getConfig(self):
        return self.config

    def compute(self, data_dict):
        gt_text = data_dict["data"].y
        gen_text = data_dict["generate_text"]
        self.metrics["rouge_1"].value, self.metrics["rouge_2"].value, \
            self.metrics["rouge_l"].value, self.metrics["rouge_l_sum"].value = _rouge(gt_text, gen_text)
        self.metrics["bleu"].value = _bleu(gt_text, gen_text)


def _rouge(gt_text, gen_text):
    rouge = ROUGEScore()
    result = rouge(gt_text, gen_text)
    result_1 = {"measure": result['rouge1_fmeasure'].item(),
                "precision": result['rouge1_precision'].item(),
                "recall": result['rouge1_recall'].item()}
    result_2 = {"measure": result['rouge2_fmeasure'].item(),
                "precision": result['rouge2_precision'].item(),
                "recall": result['rouge2_recall'].item()}
    result_l = {"measure": result['rougeL_fmeasure'].item(),
                "precision": result['rougeL_precision'].item(),
                "recall": result['rougeL_recall'].item()}
    result_l_sum = {
        "measure": result['rougeLsum_fmeasure'].item(),
        "precision": result['rougeLsum_precision'].item(),
        "recall": result['rougeLsum_recall'].item()
    }
    return result_1, result_2, result_l, result_l_sum


def _bleu(gt_text, gen_text):
    gt_bleu = []
    gen_bleu = []
    bleu_score = []
    for gen in gen_text:
        gen_bleu.append(gen.split())
    for i, gt in enumerate(gt_text):
        gt_bleu.append([gt.split()])
    bleu_score.append(corpus_bleu(gt_bleu, gen_bleu))
    return np.average(bleu_score)
