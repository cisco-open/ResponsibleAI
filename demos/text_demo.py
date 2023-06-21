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

# Description
# This demos show how RAI and its dashboard can be used for evaluating the natural language modeling tasks


# importing modules
import os
import sys
import inspect
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import random
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# importing RAI modules
from RAI.AISystem import AISystem, Model
from RAI.db.service import RaiDB
from RAI.dataset import Dataset, Feature, NumpyData
from RAI.utils import df_to_RAI

# setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
load_dotenv(f'{currentdir}/../.env')

def main():
    random.seed(0)
    np.random.seed(10)

    # Get data
    nltk.download("vader_lexicon")
    dataset = pd.DataFrame(load_dataset("rotten_tomatoes", split="train+test"))
    dataset['label'] = dataset['label'].astype('int64')
    dataset['text'] = dataset['text'].replace(-1, 0)
    new_dataset = dataset

    sentiment_model = SentimentIntensityAnalyzer()

    def score_text(input_text: list[str]) -> list[int]:
        result = []
        for val in input_text:
            result.append(0 if sentiment_model.polarity_scores(val[0])["compound"] <= 0 else 1)
        return result

    # Pass data to RAI
    meta, X, y, _ = df_to_RAI(new_dataset, target_column="label", text_columns='text')
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1)
    dataset = Dataset({"train": NumpyData(xTrain, yTrain, xTrain), "test": NumpyData(xTest, yTest, xTest)})

    # Define model within RAI
    output_feature = Feature(name="Sentiment", dtype="numeric", description="Review Sentiment Rating",
                             categorical=True, values={0: "Negative", 1: "Positive"})
    model = Model(agent=sentiment_model, output_features=output_feature, name="T5 small", predict_fun=score_text,
                  description="SentimentAnalysis", model_class="T5")

    # Create the AI System
    ai = AISystem(name="Sentiment_Analysis_1", task='binary_classification', meta_database=meta, dataset=dataset, model=model)
    ai.initialize()

    preds = []
    for val in xTest:
        score = score_text([val])[0]
        preds.append(score)

    ai.compute({"test": {"predict": preds}}, tag='initial_preds')

    r = RaiDB(ai)
    r.reset_data()
    r.add_measurement()
    r.export_metadata()
    r.export_visualizations("test", "test")
    ai.display_metric_values()


if __name__ == '__main__':
    main()
