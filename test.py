import os 
import torch


path = "results/my-awesome-project_celebA_test_17/models/22_my-awesome-project_celebA_test_17.pth"

ckpt = torch.load(path)

ckpt["epoch"] = 22

torch.save(ckpt, path)

print(ckpt["epoch"])

