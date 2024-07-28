import torch

def get_device(self):
        """
        Setup Available Device
        """
        if self.config.device:
            return self.config.device
        elif torch.cuda.is_available():
            print("-- CUDA Device detected --")
            device = "cuda"
        elif torch.backends.mps.is_available():
            print("-- MPS Device detected --")
            device = "mps"
        else:
            print("-- CPU Device detected --")
            device = "cpu"            
        return device
