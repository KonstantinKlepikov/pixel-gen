# https://huggingface.co/nerijs/pixel-art-xl
import datetime
import pathlib
from functools import lru_cache

import click
import torch
from diffusers import DiffusionPipeline, LCMScheduler

MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
LORA_ID = 'latent-consistency/lcm-lora-sdxl'
NEGATIVE_PROMPT = '3d render, realistic'
DEST_PATH = pathlib.Path('results/')


@lru_cache
def get_pipe():
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, variant='fp16')
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(LORA_ID, adapter_name='lora')
    pipe.load_lora_weights(
        'nerijs/pixel-art-xl',
        weight_name='pixel-art-xl.safetensors',
        adapter_name='pixel',
    )

    pipe.set_adapters(['lora', 'pixel'], adapter_weights=[1.0, 1.2])
    return pipe


@click.command()
@click.option('--count', default=1, help='Number of generated images')
@click.option('--steps', default=8, help='Number of inference steps')
@click.option(
    '--prompt',
    default=(
        'christmas tree, presents under it and happy plumber mario, pixelart, isometric'
    ),
    prompt='Your prompt',
)
def generate(count: int, prompt: str, steps: int):
    pipe = get_pipe()
    pipe.to(device='cuda', dtype=torch.float16)

    dt = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    p = DEST_PATH / dt
    p.mkdir()

    with open(p / 'prompt.txt', 'w') as f:
        f.write(prompt)

    for i in range(count):
        image = pipe(
            prompt=(prompt),
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=1.5,
        ).images[0]

        image.save(p / f'{i}.png')


if __name__ == '__main__':
    generate()
