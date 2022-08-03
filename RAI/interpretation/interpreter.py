from RAI.interpretation.gradcam import GradCAM

class Interpreter:
    def __init__(self, methods: list[str], model=None, datasets=None):
        """
        methods: list of the interpretation methods (e.g., ["gradcam"]). All names are in lowercase
        model: model to interpret
        imgs: one of 1. list of (train/test, img index), 2. list of categories, 3. PIL Images
        """
        self.methods = methods
        self.model = model
        self.datasets = datasets
        self.interpretation = dict()

    def getModelInterpretation(self):
        if "gradcam" in self.methods:
            self._gradcam()
        return self.interpretation

    def _gradcam(self):
        self.gradcam = GradCAM(self.model, self.datasets)
        self.gradcam.compute()  # imgs to be displayed are saved in the dictionary self.gradcam.viz_results
        self.interpretation["gradcam"] = self.gradcam.viz_results

