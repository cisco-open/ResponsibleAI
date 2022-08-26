from .display_registry import registry
from .traceable_element import TraceableElement
__all__ = ['get_display', 'is_compatible']


def get_display(metric_name: str, metric_type: str, redisUtil):
    metric_type = metric_type + "Element"
    additional_features = {}
    if metric_type.lower() in registry:
        requirements = registry[metric_type.lower()].get_requirements()
        if 'features' in requirements:
            additional_features['features'] = redisUtil.get_project_info()["features"]
        return registry[metric_type.lower()](metric_name, **additional_features)
    return None


def is_compatible(metric_type: str, requirements: list):
    metric_type += "Element"
    result = metric_type.lower() in registry
    if result and 'Traceable' in requirements:
        result = issubclass(registry[metric_type], TraceableElement)
    return result
