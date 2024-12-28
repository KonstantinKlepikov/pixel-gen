# https://huggingface.co/nerijs/pixel-art-xl
import datetime

import torch
from diffusers import DiffusionPipeline, LCMScheduler

# pipe = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')
# pipe.load_lora_weights('nerijs/pixel-art-xl')
# pipe.to('cuda')

# prompt = 'pixel, an astronaut riding a horse, shine color palette'
# image = pipe(prompt).images[0]
# image.save(
#     'results/pixel_art_xl_'
#     f'{datetime.datetime.now().strftime("%H:%M:%S_%Y-%m-%d")}.png'
# )

# out of memory


model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
lcm_lora_id = 'latent-consistency/lcm-lora-sdxl'

pipe = DiffusionPipeline.from_pretrained(model_id, variant='fp16')
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(lcm_lora_id, adapter_name='lora')
pipe.load_lora_weights(
    'nerijs/pixel-art-xl', weight_name='pixel-art-xl.safetensors', adapter_name='pixel'
)

pipe.set_adapters(['lora', 'pixel'], adapter_weights=[1.0, 1.2])
pipe.to(device='cuda', dtype=torch.float16)

prompt = (
    'christmas tree, presents under it and happy plumber mario, pixelart, isometric'
)
negative_prompt = '3d render, realistic'

num_images = 10

for _ in range(num_images):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=8,
        guidance_scale=1.5,
    ).images[0]

    image.save(
        'results/pixel_art_xl_'
        f'{datetime.datetime.now().strftime("%H:%M:%S_%Y-%m-%d")}.png'
    )
