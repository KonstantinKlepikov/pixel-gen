import datetime
from enum import Enum, member

import click
import torch

from .constants import DEST_PATH, NEGATIVE_PROMPT
from .pill_pixelating import pixelate
from .pixel_art_xl import get_pipe as art_pipe
from .pixel_party_xl import get_pipe as party_pipe


class DiffusionModel(Enum):
    ART = member(art_pipe)
    PARTY = member(party_pipe)


@click.command()
@click.option('--count', default=1, help='Number of generated images')
@click.option('--steps', default=8, help='Number of inference steps')
@click.option(
    '--cfg',
    type=click.FLOAT,
    default=1.5,
    help='How closely the model follows the prompt. Typically between 5.0 and 12.0',
)
@click.option('-pill', is_flag=True, help='Use PILL pixelating')
@click.option(
    '--model',
    type=click.Choice(DiffusionModel, case_sensitive=False),
    default=DiffusionModel.ART,
    help='Available Model',
)
@click.option(
    '--prompt',
    default=(
        'christmas tree, presents under it and happy plumber mario, pixelart, isometric'
    ),
    prompt='Your prompt',
)
def generate(
    count: int,
    prompt: str,
    steps: int,
    pill: bool,
    cfg: float,
    model: DiffusionModel,
):
    pipe = model.value()
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
            guidance_scale=cfg,
        ).images[0]

        path_to_img = p / f'{i}.png'
        image.save(path_to_img)
        if pill:
            pixelate(path_to_img=path_to_img)


if __name__ == '__main__':
    generate()
