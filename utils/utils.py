import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def namespace2dict(namespace):
    config = {}
    for key, value in vars(namespace).items():
        if isinstance(value, argparse.Namespace):
            new_value = namespace2dict(value)
        else:
            new_value = value
        config[key] = new_value
    return config