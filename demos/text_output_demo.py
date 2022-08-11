import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import pandas as pd
from RAI.AISystem import AISystem, Model
from RAI.redis import RaiRedis
from RAI.dataset import Dataset, Data
from RAI.utils import df_to_RAI
from sklearn.model_selection import train_test_split
import random
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration


t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')


# TODO: Refine RAI to model transformation
def summarize(text):
    while isinstance(text, list) or isinstance(text, np.ndarray):
        text = text[0]
    text = "summarize: " + text
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512)
    summary_ids = t5.generate(input_ids)
    return tokenizer.decode(summary_ids[0])


def main():
    use_dashboard = True
    random.seed(0)
    np.random.seed(10)

    dataset = load_dataset("gigaword", split="test")
    df = pd.DataFrame(dataset)

    meta, X, y, output = df_to_RAI(df, target_column="summary", text_columns=['document', 'summary'])
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1)

    # I don't have a GPU!
    xTest = xTest[:10]
    yTest = yTest[:10]

    preds = []
    for i, val in enumerate(xTest):
        summary = summarize(val[0])
        preds.append(summary)

    model = Model(agent=t5, output_features=output, name="t5", generate_text_fun=summarize,
                  description="Text Summarizer", model_class="t5")
    configuration = {"time_complexity": "polynomial"}
    dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})
    ai = AISystem(name="Text_Summarizer_t5", task='generate', meta_database=meta, dataset=dataset, model=model)
    ai.initialize(user_config=configuration)
    ai.compute({"test": {"generate_text": preds}}, tag='initial_preds')
    if use_dashboard:
        r = RaiRedis(ai)
        r.connect()
        r.reset_redis(summarize_data=False)
        r.add_measurement()
        r.add_dataset()
        r.export_visualizations()

    ai.display_metric_values()

    from RAI.Analysis import AnalysisManager
    analysis = AnalysisManager()
    #print("available analysis: ", analysis.get_available_analysis(ai, "test"))
    #result = analysis.run_all(ai, "test", "Test run!")
    # result = analysis.run_analysis(ai, "test", "CleverUntargetedScore", "Testing")
    # for analysis in result:
    #     print("Analysis: " + analysis)
    #     print(result[analysis].to_string())


if __name__ == '__main__':
    main()

