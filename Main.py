import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, filters, feature, exposure, color, util, img_as_float
import skimage.io as io
from skimage.morphology import closing, square, area_closing, area_opening, diameter_closing
from skimage.color import rgb2gray
from skimage.transform import *
import scipy
from scipy import misc
import cv2 as cv
from skimage.util import crop
from skimage.exposure import match_histograms

reference = data.coffee()
image = data.chelsea()

#goruntu iyileştirme islemleri,filtreler (10 farklı filtre icermeli)



#histogram equalization resimleri gösterme
def equalize_histogram(image, reference):
    matched_image = match_histograms(image, reference, multichannel=True)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
    ax1.imshow(image)
    ax1.set_title('Source')
    ax2.imshow(reference)
    ax2.set_title('Reference')
    ax3.imshow(matched_image)
    ax3.set_title('Matched Image')

    plt.tight_layout()
    plt.show()
    return matched_image


#histogram eşitleme grafiği gösterme
def plot_equalized_histogram(image,reference,matched_image):

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

    for i, img in enumerate((image, reference, matched_image)):
        for c, c_color in enumerate(('red', 'green', 'blue')):
            img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
            axes[c, i].plot(bins, img_hist / img_hist.max())
            img_cdf, bins = exposure.cumulative_distribution(img[..., c])
            axes[c, i].plot(bins, img_cdf)
            axes[c, 0].set_ylabel(c_color)

    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')

    plt.tight_layout()
    plt.show()


#resize,rotation,cropping,swirling, gibi 5 farklı donusum islemi
def rotate_image(image,degree):

    rotated_image=rotate(image,degree)
    return rotated_image



def resize_image(image,amount):
    image_resized = resize(image, (amount[0],amount[1]), anti_aliasing=True)
    io.imshow(image_resized)
    io.show()

def swirl_image(image,strength,radius):
    swirled_image = swirl(image, rotation=0, strength=strength, radius=radius)
    io.imshow(swirled_image)
    io.show()

def crop_image(image,crop_width):
    cropped_image=crop(image,crop_width,copy=True)
    io.imshow(cropped_image)
    io.show()

def rescale_image(image,amount):
    rescaled_image=rescale(image,amount, anti_aliasing=False)
    io.imshow(rescaled_image)
    io.show()

#yogunluk donusumu işlemleri (degerleri kullanıcı verebilmeli)

#morfolojik işlemler (10 farklı morfolojik işlem icermeli)


from skimage.morphology import square,skeletonize_3d,thin,disk, dilation,erosion,skeletonize
from skimage.util import invert

#new_image = dilation(image)
#new_image=erosion(image)
#image=thin(data.horse())
#image=skeletonize_3d(data.horse())

#insta filtresi oluştur.

