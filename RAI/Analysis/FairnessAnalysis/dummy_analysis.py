from RAI.Analysis import Analysis
from RAI.AISystem import AISystem
import os
from dash import html
import time


class DummyAnalysis(Analysis, class_location=os.path.abspath(__file__)):
    def __init__(self, ai_system: AISystem, dataset: str, tag: str = None):
        super().__init__(ai_system, dataset, tag)
        self.result = None  # calculate result
        self.ai_system = ai_system
        self.dataset = dataset
        self.tag = tag
        self.loops = 30
        self.max_progress_tick = self.loops

    def initialize(self):
        self._compute()
        pass

    def _compute(self):
        for i in range(self.loops):
            time.sleep(1)
            self.progress_tick()

    def to_string(self):
        return "TEST"

    def to_html(self):
        '''       
        res = res.reshape(3, 32, 32)
        img = np.array(res.detach())
        img = np.transpose(np.uint8(img * 255), (1, 2, 0))
        print("img: ", img)
        fig_1 = go.Figure(go.Image(z=img))
        fig_1.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig_graph_1 = html.Div(dcc.Graph(figure=fig_1), style={"display": "inline-block", "padding": "0"})
        '''
        result = html.Div([
            html.H4("This Dash HTML is generated at runtime!!"),
            html.Button("Here is a button")], style={})

        return result
