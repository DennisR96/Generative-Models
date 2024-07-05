import argparse, os, yaml
import wandb
import torch

from utils.utils import dict2namespace
from networks import load_network
from datasets import load_dataset
from models import load_model

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config.yaml",
        help="path to the YAML configuration file"
    )
    return parser


if __name__ == "__main__":
    # Parser
    parser = get_parser()
    args = parser.parse_args()
    
    # Load Config YAML as Namespace 
    config_path = "config/mnist.yaml"
    with open(os.path.join(args.config), "r") as f:
        config_yaml = yaml.safe_load(f)
    config = dict2namespace(config_yaml)
    
    # Logging
    wandb.init(
    project="my-awesome-project",
    config=config_yaml)

    # Load Network
    network = load_network(config)
    
    # Load Dataset
    dataset = load_dataset(config)
    
    # Load Model
    model = load_model(config, dataset, network)
    
    # Training
    model.train()
    
    
    
    
    