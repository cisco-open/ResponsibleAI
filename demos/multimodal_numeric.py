import pandas as pd
from sklearn.model_selection import train_test_split
from RAI.AISystem import AISystem, Model
from RAI.dataset import Data, Dataset, Feature
from RAI.redis import RaiRedis
from RAI.utils import df_to_RAI, torch_to_RAI
from datasets import load_dataset
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier

use_dashboard = True

def main():
    # Get Dataset
    data_path = "../data/adult/"
    train_data = pd.read_csv(data_path + "train.csv", header=0,
                             skipinitialspace=True, na_values="?")
    test_data = pd.read_csv(data_path + "test.csv", header=0,
                            skipinitialspace=True, na_values="?")
    all_data = pd.concat([train_data, test_data], ignore_index=True)

    # Get Image Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    xTrainImage, _ = torch_to_RAI(trainloader)
    image_df = pd.DataFrame(xTrainImage[:len(all_data)].tolist())
    all_data["image"] = image_df

    # Get text data
    text_df = pd.DataFrame(load_dataset("gigaword", split="train")[:len(all_data)])
    all_data["text"] = text_df["document"]

    # Get features, and X y for image data
    meta, X, y, output = df_to_RAI(all_data, target_column="income-per-year", normalize="Scalar",
                                   text_columns=["text"], image_columns=["image"])

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=1, stratify=y)

    clf = RandomForestClassifier(n_estimators=4, max_depth=6)
    model = Model(agent=clf, output_features=output, name="cisco_income_ai", predict_fun=clf.predict, predict_prob_fun=clf.predict_proba,
                  description="Income Prediction AI", model_class="Random Forest Classifier", )
    configuration = {"fairness": {"priv_group": {"race": {"privileged": 1, "unprivileged": 0}},
                                  "protected_attributes": ["race"], "positive_label": 1},
                     "time_complexity": "polynomial"}

    dataset = Dataset({"train": Data(xTrain, yTrain), "test": Data(xTest, yTest)})
    ai = AISystem(name="multi_modal_classification",  task='binary_classification', meta_database=meta, dataset=dataset, model=model)
    ai.initialize(user_config=configuration)

    clf.fit(xTrain[:, :-2], yTrain)

    print("\n\nTESTING PREDICTING METRICS:")
    test_preds = clf.predict(xTest[:, :-2])
    ai.compute({"test": {"predict": test_preds}}, tag='model1')

    if use_dashboard:
        r = RaiRedis(ai)
        r.connect()
        r.reset_redis()
        r.add_measurement()

    ai.display_metric_values("test")


    from RAI.Analysis import AnalysisManager

    analysis = AnalysisManager()
    print("available analysis: ", analysis.get_available_analysis(ai, "test"))
    # result = analysis.run_analysis(ai, ["test"], ["FairnessAnalysis"])
    result = analysis.run_all(ai, "test", "Test run!")
    for analysis in result:
        print("Analysis: " + analysis)
        print(result[analysis].to_string())


if __name__ == "__main__":
    main()
