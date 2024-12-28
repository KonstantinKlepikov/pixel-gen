# https://github.com/sedthh/pyxelate
import pyxelate as pxl
from skimage import io

image = io.imread('examples/that_ass.jpg')

downsample_by = 7  # new image will be 1/7th of the original in size
palette = 7  # find 7 colors

new_image = pxl.Pyx(
    factor=downsample_by,
    palette=palette,
    # dither='atkinson',
    alpha=0.6,
).fit_transform(image)

io.imsave('results/that_assl.png', new_image)
