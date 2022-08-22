import random
import numpy as np
from RAI.AISystem import AISystem
from RAI.Analysis import Analysis
from RAI.dataset import IteratorData, NumpyData
import os
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go


class DataVisualization(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None
        self.ai_system = ai_system
        self.dataset = dataset
        self.values = ai_system.get_metric_values()[dataset]
        self.tag = tag
        self.total_examples = 5
        self.eps = 0.1
        self.max_progress_tick = self.total_examples
        self.output_feature = self.ai_system.model.output_features[0]
        self.input_features = self.ai_system.meta_database.features
        self.gen_results = {"num_features": self.values["metadata"]["sample_count"]}

    def initialize(self):
        if self.result is None:
            self.result = self._compute()

    def _compute(self):
        result = {}
        result["x"] = self.get_feature_stats(self.input_features)
        result["num_samples"] = self.values["metadata"]["sample_count"]
        return result

    def get_feature_stats(self, features):
        results = []
        for i, feature in enumerate(features):  # TODO: find a cleaner way to get dtype
            feature_dict = {"name": feature.name, "feature": feature}
            if feature.dtype.lower().startswith("int") or feature.dtype.lower().startswith("float") or \
                    feature.dtype.lower() == "numeric":
                if feature.categorical:
                    feature_dict['dtype'] = "categorical"
                    feature_dict['value'] = self.values["frequency_stats"]["relative_freq"][feature.name]
                    feature_dict["dis_type"] = "chart"
                    feature_dict["dis_name"] = "Relative Frequency"
                else:
                    feature_dict["dtype"] = "scalar"
                    feature_dict["value"] = {"median": self.values["summary_stats"]["median"][feature.name],
                                             "q1": self.values["summary_stats"]["quantile_1"][feature.name],
                                             "q3": self.values["summary_stats"]["quantile_3"][feature.name],
                                             "min": self.values["summary_stats"]["min"][feature.name],
                                             "max": self.values["summary_stats"]["max"][feature.name]}
                    feature_dict["dis_type"] = "box_chart"
                    feature_dict["dis_name"] = "Box Chart"
            elif feature.dtype.lower() == "text":
                feature_dict["dtype"] = "text"
                feature_dict["value"] = None
                feature_dict["dis_type"] = None
                feature_dict["dis_name"] = ""
            elif feature.dtype.lower() == "image":
                feature_dict["dtype"] = "image"
                feature_dict["value"] = {"mean": self.values["image_stats"]["mean"],
                                         "std": self.values["image_stats"]["std"]}
                feature_dict["dis_type"] = "chart"
                feature_dict["dis_name"] = "Image stats"
            results.append(feature_dict)
        return results

    def _feature_to_html(self, feature):
        ts = {"text-align": "center", "display": "block"}
        result = [html.H4(feature["name"], style=ts), html.B("Description: " + feature["feature"].description),
                  html.Br(), html.B("Data type: " + feature["feature"].dtype), html.Br()]
        if feature['dis_type'] == "number":
            result.append(html.B(feature["dis_name"] + ": " + str(feature["value"])))
        elif feature['dis_type'] == "chart":
            fig = go.Figure([go.Bar(x=[i], y=[feature["value"][i]], showlegend=False) for i in feature["value"]])
            fig.update_layout(title={'text': feature["dis_name"] + " of " + feature["name"], 'y': 0.9, 'x': 0.5,
                                     'xanchor': 'center', 'yanchor': 'top'})
            fig = html.Div(dcc.Graph(figure=fig), style={"display": "block", "margin": "0 auto 0 auto", "width": "60%"})
            result.append(fig)
        elif feature['dis_type'] == "box_chart":
            fig = go.Figure(go.Box())
            fig.update_traces(q1=[feature["value"]["q1"]], median=[feature["value"]["median"]],
                              q3=[feature["value"]["q3"]], lowerfence=[feature["value"]["min"]],
                              upperfence=[feature["value"]["max"]])
            fig.update_layout(title={'text': feature["dis_name"] + " of " + feature["name"], 'y': 0.9, 'x': 0.5,
                                     'xanchor': 'center', 'yanchor': 'top'})
            fig = html.Div(dcc.Graph(figure=fig), style={"display": "block", "margin": "0 auto 0 auto", "width": "60%"})
            result.append(fig)
        return html.Div(result)


    def to_string(self):
        result = "\n==== Data Interpretation ====\n"
        result += "Please view this analysis in the Dashboard."
        return result

    def to_html(self):
        out_html = []
        ts = {"text-align": "center", "display": "block"}
        out_html.append(html.H1("Data Interpretation", style=ts))
        out_html.append(html.Div(html.B("Number of Examples: " + str(self.result["num_samples"]))))
        if len(self.result['x']) > 0:
            out_html.append(html.Br())
            out_html.append(html.H3("X Data"))
            for feature in self.result['x']:
                out_html.append(self._feature_to_html(feature))
                out_html.append(html.Br())
        out_html = html.Div(out_html)
        return out_html

