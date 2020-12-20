import numpy as np
from PIL import Image
from numpy import asarray
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from matplotlib import pyplot as plt

def active_contour_model(filename):
    img = asarray(Image.open(filename))

    s = np.linspace(0, 6 * np.pi, 400)
    r = 120 + 100 * np.sin(s)
    c = 150 + 100 * np.cos(s)
    init = np.array([r, c]).T

    snake = active_contour(gaussian(img, 3,multichannel=False),
                           init, alpha=0.015, beta=10, gamma=0.001,
                           coordinates='rc')

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()


active_contour_model("mona.jpg")
