# https://huggingface.co/kohbanye/pixel-art-style
import datetime

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained('kohbanye/pixel-art-style')
pipe.to('cuda')

prompt = 'An astronaut riding a horse, shine color palette, pixelartstyle'
image = pipe(prompt).images[0]
image.save(
    'results/pixel_art_style_'
    f'{datetime.datetime.now().strftime("%H:%M:%S_%Y-%m-%d")}.png'
)
