# https://stackoverflow.com/questions/47143332/how-to-pixelate-a-square-image-to-256-big-pixels-with-python
from PIL import Image

img = Image.open('examples/hero_001.jpg')

img_small = img.resize((104, 104), resample=Image.Resampling.BILINEAR)

result = img_small.resize(img.size, Image.Resampling.NEAREST)

# Save
result.save('results/hero_001.png')
