from .DIV2K import DIV2K
from .FFHQ import FFHQ
from .MNIST import MNIST
from .CELEBA import CELEBA

def load_dataset(config):
    """
    Load the Dataset (data.Dataset) Class

    Args:
        config (_type_): YAML Configuration

    Returns:
        data.Dataset: Dataset Class
    """
    if config.dataset.name == "MNIST":
        print("-- Dataset: MNIST --")
        return MNIST(config)
    elif config.dataset.name == "FFHQ":
        print("-- Dataset: FFHQ --")
        return FFHQ(config)
    elif config.dataset.name == "DIV2K":
        print("-- Dataset: DIV2K --")
        return DIV2K(config)
    elif config.dataset.name == "CELEBA":
        print("-- Dataset: CelebA --")
        return CELEBA(config)
        
    
    