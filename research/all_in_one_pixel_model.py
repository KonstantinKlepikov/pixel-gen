# https://huggingface.co/PublicPrompts/All-In-One-Pixel-Model
import datetime

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained('PublicPrompts/All-In-One-Pixel-Model')
pipe.to('cuda')

prompt = 'An astronaut riding a horse, shine color palette, pixelartstyle'
image = pipe(prompt).images[0]
image.save(
    'results/all_in_one_pixel_model_'
    f'{datetime.datetime.now().strftime("%H:%M:%S_%Y-%m-%d")}.png'
)
