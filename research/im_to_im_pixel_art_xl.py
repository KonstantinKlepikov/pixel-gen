# https://huggingface.co/nerijs/pixel-art-xl
# https://huggingface.co/docs/diffusers/using-diffusers/img2img
import datetime

import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from PIL import Image

model_id = 'stabilityai/stable-diffusion-xl-base-1.0'

pipe = AutoPipelineForImage2Image.from_pretrained(
    model_id, torch_dtype=torch.float16, variant='fp16', use_safetensors=True
)
pipe.enable_model_cpu_offload()

# prepare image
init_image = load_image(Image.open('examples/hero_001.jpg'))

prompt = 'Pixelart man colored by bright colors and with a tape recorder'
negative_prompt = '3d render, realistic'

# pass prompt and image to pipeline
image = pipe(
    prompt,
    image=init_image,
    negative_prompt=negative_prompt,
    strength=0.51,
).images[0]
image.save(
    'results/img_to_img_pixel_art_xl_'
    f'{datetime.datetime.now().strftime("%H:%M:%S_%Y-%m-%d")}.png'
)
