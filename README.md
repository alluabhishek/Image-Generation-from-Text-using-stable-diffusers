# Image-Generation-from-Text-using-stable-diffusers
Hugging Chat Diffusers have been used to generate an image from text.

Step 1 -
pip install diffusers transformers accelerate torch

Step 2 - 
!pip show torch 

Step 3 - 
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors = True)
pipe = pipe.to("cuda")

Step 4 - 
prompt = """give a promt of your choice"""

Step 5 -
image = pipe(prompt).images[0]

Step 6 - 
print("[PROMPT]: ", prompt)
plt.imshow(image)
plt.axis("off")
