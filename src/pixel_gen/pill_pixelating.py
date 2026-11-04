import pathlib

from PIL import Image


def pixelate(path_to_img: pathlib.Path) -> None:
    img = Image.open(path_to_img)

    img_small = img.resize((104, 104), resample=Image.Resampling.BILINEAR)

    result = img_small.resize(img.size, Image.Resampling.NEAREST)

    # Save
    result.save(path_to_img)
