import torch

def psnr(image_1, image_2):
    """
    Metric: Peak Signal-to-Noise Ratio

    Args:
        image_1 (torch.Tensor):     Image Tensor [0, 1]
        image_2 (torch.Tensor):     Image Tensor [0, 1]
    """
    
    # -- 1. Calculate Mean Squared Error -- 
    mse = torch.mean((image_1 - image_2)**2)
    
    # -- 2. Calculate PSNR -- 
    if mse == 0: 
        psnr = float("inf")
    else:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr