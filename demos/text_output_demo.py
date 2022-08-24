import os
import sys
import inspect
import pandas as pd
from RAI.AISystem import AISystem, Model
from RAI.redis import RaiRedis
from RAI.dataset import Dataset, NumpyData
from RAI.utils import df_to_RAI
from sklearn.model_selection import train_test_split
import random
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


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

    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()
    r.add_measurement()
    r.export_visualizations("test", "test")


if __name__ == '__main__':
    main()
