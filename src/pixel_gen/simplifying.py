import pathlib

from PIL import Image


def simplificate(path_to_img: pathlib.Path, num: int) -> pathlib.Path:
    current = path_to_img.stem + '_simplificate.png'
    path_to_save = path_to_img.with_name(current)
    img = Image.open(path_to_img)
    rgb_img = img.convert('RGB')
    result = rgb_img.quantize(
        colors=num,
        method=Image.Quantize.MAXCOVERAGE,
    ).convert('RGB')
    result.save(path_to_save)
    return path_to_save
