from .DIV2K import DIV2k
from .FFHQ import FFHQ
from .MNIST import MNIST

def load_dataset(config):
    """
    Load the Dataset (data.Dataset) Class

    Args:
        config (_type_): YAML Configuration

    Returns:
        data.Dataset: Dataset Class
    """
    if config.dataset.name == "MNIST":
        return MNIST(config)
    elif config.dataset.name == "FFHQ":
        return FFHQ(config)
    elif config.dataset.name == "DIV2K":
        return DIV2K(config)
    
    