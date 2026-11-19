import pathlib

from PIL import Image


def pixelate(path_to_img: pathlib.Path) -> pathlib.Path:
    current = path_to_img.stem + '_pixelate.png'
    path_to_save = path_to_img.with_name(current)
    img = Image.open(path_to_img)
    img_small = img.resize((104, 104), resample=Image.Resampling.BILINEAR)
    result = img_small.resize(img.size, Image.Resampling.NEAREST)
    result.save(path_to_save)
    return path_to_save
