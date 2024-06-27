import argparse
import os
import yaml
import torch 
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np 

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace