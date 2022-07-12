from .display_registry import register_class, registry


class DisplayFactory:

    def get_display(self, metric_name, metric_type, redisUtil):
        metric_type = metric_type + "Element"
        print("type: ", metric_type)
        print("registry: ", registry)
        additional_features = {}
        if metric_type in registry:
            requirements = (registry[metric_type](metric_name)).requirements
            if 'features' in requirements:
                additional_features['features'] = redisUtil.get_project_info()["features"]
            return registry[metric_type](metric_name, **additional_features)
        return None

