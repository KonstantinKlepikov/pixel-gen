import datetime
from enum import Enum, member

import click
import torch
from loguru import logger

from .background import remove_bg
from .constants import DEST_PATH, NEGATIVE_PROMPT
from .pill_pixelating import pixelate
from .pixel_art_xl import get_pipe as art_pipe
from .pixel_party_xl import get_pipe as party_pipe
from .simplifying import simplificate


class DiffusionModel(Enum):
    ART = member(art_pipe)
    PARTY = member(party_pipe)


@click.command()
@click.option('--count', default=1, help='Number of generated images. Default to 1')
@click.option('--steps', default=8, help='Number of inference steps. Default to 8')
@click.option(
    '--cfg',
    type=click.FLOAT,
    default=1.5,
    help=(
        'How closely the model follows the prompt. Typically between 0.0 and 2.0. '
        'Default to 1.5'
    ),
)
@click.option('-pill', is_flag=True, help='Use PILL pixelating')
@click.option('-rmbg', is_flag=True, help='Remove background')
@click.option(
    '--simple',
    default=1,
    help='Number for PILL simplification pallete. Default to 1 (no changes)',
)
@click.option(
    '--model',
    type=click.Choice(DiffusionModel, case_sensitive=False),
    default=DiffusionModel.ART,
    help='Available Model. Default to ART',
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
    simple: int,
    rmbg: bool,
    cfg: float,
    model: DiffusionModel,
):
    pipe = model.value()
    pipe.to(device='cuda', dtype=torch.float16)

    dt = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    p = DEST_PATH / dt
    p.mkdir()

    logger.add(p / 'gen.log')
    logger.info(f'Prompt: {prompt}')
    logger.info(f'Model: {model.name}')
    logger.info(f'Parameters: {count=}, {steps=}, {pill=}, {simple=}, {cfg=}')

    for i in range(count):
        image = pipe(
            prompt=(prompt),
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=cfg,
        ).images[0]

        path_to_img = p / f'{i}.png'
        image.save(path_to_img)
        logger.info(f'Image {i} on path {path_to_img}')

        if rmbg:
            path_to_img = remove_bg(path_to_img=path_to_img)
            logger.info('Background removed')
        if pill:
            path_to_img = pixelate(path_to_img=path_to_img)
            logger.info('Pixelated')
        if simple > 1:
            path_to_img = simplificate(path_to_img=path_to_img, num=simple)
            logger.info('Simplificated')


if __name__ == '__main__':
    generate()
