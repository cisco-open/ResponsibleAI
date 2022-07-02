from .display_object import DisplayElement
import plotly.graph_objs as go


# For single valued numeric metrics, like accuracy
class NumericalElement(DisplayElement):
    def __init__(self, name):
        super().__init__(name)
        self.x = 0
        self._data["x"] = []
        self._data["y"] = []
        self._data["tag"] = []
        self._data["text"] = []

    def append(self, metric_data, tag):
        self._data["x"].append(self.x)
        self.x += 1
        self._data["y"].append(metric_data)
        self._data["tag"].append(tag)
        self._data["text"].append("%.2f" % metric_data)

    def to_string(self):
        print(self._data)

    def to_display(self):
        sc_data = {'mode': 'lines+markers+text',
                   'name': f"{self._name}", 'orientation': 'v', 'showlegend': True,
                   'text': self._data["text"], 'x': self._data["x"], 'xaxis': 'x', 'y': self._data['y'], 'yaxis': 'y',
                   'type': 'scatter', 'textposition': 'top center',
                   'hovertemplate': 'metric=' + self._name + '<br>x=%{x}<br>value=%{y}<br>text=%{text}<extra></extra>'}
        fig = go.Figure()
        fig.add_trace(go.Scatter(**sc_data))
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=sc_data["x"], ticktext=self._data["tag"]),
            legend=dict(title_font_family="Times New Roman",
                        font=dict(family="Times New Roman", size=14, color="black"),
                        bgcolor="Azure",
                        bordercolor="Black",
                        borderwidth=1))
        return fig
