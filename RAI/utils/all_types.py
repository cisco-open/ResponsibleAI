__all__ = ['all_complexity_classes', 'all_task_types', 'all_data_types', 'all_data_types_lax',
           'all_output_requirements', 'all_dataset_requirements', 'all_metric_types']

all_complexity_classes = {"constant", "linear", "multi_linear", "polynomial", "exponential"}
all_task_types = {"binary_classification", "classification", "clustering", "regression"}
all_data_types = {"Numeric", "image", "text"}
all_data_types_lax = {"integer", "float", "Numeric", "image", "text"}
all_output_requirements = {"predict", "predict_proba", "generate_text"}
all_dataset_requirements = {"X", "y", "sensitive_features"}
all_metric_types = {"Numeric", "multivalued", "other", "vector", "vector-dict", "Matrix", "boolean"}