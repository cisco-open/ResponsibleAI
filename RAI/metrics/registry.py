registry = {}


def register_class(class_name, class_object):
    if class_name != "":
        registry[class_name] = class_object


