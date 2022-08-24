import os
import sys
import inspect
from RAI.AISystem import AISystem, Model
from RAI.redis import RaiRedis
from RAI.dataset import Dataset, Feature, NumpyData
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from RAI.utils import df_to_RAI
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import random
import numpy as np
import pandas as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


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
    output_feature = Feature(name="Sentiment", dtype="Numeric", description="Review Sentiment Rating",
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

    r = RaiRedis(ai)
    r.connect()
    r.reset_redis()
    r.add_measurement()
    r.export_metadata()
    r.export_visualizations("test", "test")
    ai.display_metric_values()

if __name__ == '__main__':
    main()
