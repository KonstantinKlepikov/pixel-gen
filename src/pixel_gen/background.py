import pathlib

from PIL import Image
from rembg import remove


def remove_bg(path_to_img: pathlib.Path) -> pathlib.Path:
    current = path_to_img.stem + '_rmbg.png'
    path_to_save = path_to_img.with_name(current)
    img = Image.open(path_to_img)
    result = remove(img)
    result.save(path_to_save)
    return path_to_save
