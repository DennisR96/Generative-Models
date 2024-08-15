import os
import diffusers
import torch
from PIL import Image

files = [i for i in os.listdir("datasets/FFHQ") if i.endswith(".png")]
filepaths = [os.path.join("datasets/FFHQ", i) for i in files]

pipe = diffusers.AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("mps")

for i in range(len(files)):
    img = Image.open(filepaths[i])
    image = pipe(image=img, prompt="", strength=0.4).images[0]
    image.save(f"ffhq_diff/{files[i]}")


