# https://huggingface.co/artificialguybr/PixelArtRedmond
import datetime

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')
pipe.load_lora_weights('artificialguybr/PixelArtRedmond')
pipe.to('cuda')

prompt = 'Pixel Art, an astronaut riding a horse, shine color palette'
image = pipe(prompt).images[0]
image.save(
    'results/pixel_art_redmont_'
    f'{datetime.datetime.now().strftime("%H:%M:%S_%Y-%m-%d")}.png'
)

# out of memory
# FIXME: use like pixel_art_xl
