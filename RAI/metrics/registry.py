registry = {}


def register_class(class_name, class_object):
    if class_name != "":
        if class_name in registry:
            raise NameError("Class Name: " + class_name + " already exists. Please enter a unique class name.")
        registry[class_name] = class_object
