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


# This code requires the sentencepiece package
# importing modules
import os
import sys
import inspect
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

from transformers import T5Tokenizer, T5ForConditionalGeneration


# importing RAI modules
from RAI.AISystem import AISystem, Model
from RAI.db.service import RaiDB
from RAI.dataset import Dataset, NumpyData
from RAI.utils import df_to_RAI

# setup path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

load_dotenv(f'{currentdir}/../.env')

# Get Model
t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')


# Function to produce summarized text
def summarize(text):
    while isinstance(text, list) or isinstance(text, np.ndarray):
        text = text[0]
    text = "summarize: " + text
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512)
    summary_ids = t5.generate(input_ids)
    return [tokenizer.decode(summary_ids[0])]


def main():
    random.seed(0)
    np.random.seed(10)

    # Get dataset
    dataset = load_dataset("gigaword", split="test")
    df = pd.DataFrame(dataset)

    # Convert dataset to RAI
    meta, X, y, output = df_to_RAI(df, target_column="summary", text_columns=['document', 'summary'])
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
    x_test = x_test[:10]
    y_test = y_test[:10]

    # Produce Summarizations
    summaries = []
    for i, val in enumerate(x_test):
        summary = summarize(val[0])[0]
        summaries.append(summary)

    # Create RAIs representation of the model
    model = Model(agent=t5, output_features=output, name="t5", generate_text_fun=summarize,
                  description="Text Summarizer", model_class="t5")

    # Create RAIs representation of the data splits
    dataset = Dataset({"train": NumpyData(x_train, y_train), "test": NumpyData(x_test, y_test)})

    # Create a RAI AISystem to calculate metrics and run analysis
    ai = AISystem(name="Text_Summarizer_t5", task='generate', meta_database=meta, dataset=dataset, model=model)
    configuration = {"time_complexity": "polynomial"}
    ai.initialize(user_config=configuration)

    # Compute metrics on the summarization
    ai.compute({"test": {"generate_text": summaries}}, tag='t5_small')

    r = RaiDB(ai)
    r.reset_data()
    r.add_measurement()
    r.export_metadata()
    r.export_visualizations("test", "test")


if __name__ == '__main__':
    main()
