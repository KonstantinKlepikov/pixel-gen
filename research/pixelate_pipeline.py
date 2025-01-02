import pyxelate as pxl
from pixelate import pixelate
from skimage import io

pixelate('examples/that_ass.jpg', 'results/that_ass_pixelate_pipeline.png', 10)

image = io.imread('results/that_ass_pixelate_pipeline.png')

downsample_by = 5  # new image will be 1/xth of the original in size
palette = 10  # find x colors

new_image = pxl.Pyx(
    factor=downsample_by,
    palette=palette,
    # dither='atkinson',
).fit_transform(image)

io.imsave('results/that_ass_pixelate_pipeline.png', new_image)
