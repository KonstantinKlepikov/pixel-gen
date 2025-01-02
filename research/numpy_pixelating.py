# https://stackoverflow.com/questions/47143332/how-to-pixelate-a-square-image-to-256-big-pixels-with-python
# TODO: fixme
import matplotlib.pyplot as plt
import numpy as np


def pixelate_rgb(img: np.ndarray, window: int) -> np.ndarray:
    n, m, _ = img.shape
    n, m = n - n % window, m - m % window
    img1 = np.zeros((n, m, 3))
    for x in range(0, n, window):
        for y in range(0, m, window):
            img1[x : x + window, y : y + window] = img[  # noqa: E203
                x : x + window, y : y + window  # noqa: E20
            ].mean(axis=(0, 1))
    return img1


img = plt.imread('examples/that_ass.jpg')

fig, ax = plt.subplots(1, 4, figsize=(20, 10))

ax[0].imshow(pixelate_rgb(img, 5))
ax[1].imshow(pixelate_rgb(img, 10))
ax[2].imshow(pixelate_rgb(img, 20))
ax[3].imshow(pixelate_rgb(img, 30))

# remove frames
[a.set_axis_off() for a in ax.flatten()]
plt.subplots_adjust(wspace=0.03, hspace=0)

plt.savefig('results/that_ass_numpy.png')
