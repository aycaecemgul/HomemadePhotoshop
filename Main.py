import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
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



#goruntu iyileştirme islemleri,filtreler
#'Wiener', "Prewitt V", "Prewitt H", "Hessian", 'Median', "Meijering", "Frangi", "Laplacian", "Gaussian", 'Sato'



#resize,rotation,cropping,swirling, gibi 5 farklı donusum islemi
def rotate_image_90(filename):
    image = Image.open(filename)
    image = asarray(image)
    rotated_image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    rotated_image = Image.fromarray(rotated_image)
    rotated_image.save(filename)
    return filename

def rotate_image_180(filename):
    image = Image.open(filename)
    image = asarray(image)
    rotated_image = cv.rotate(image, cv.ROTATE_180)
    rotated_image = Image.fromarray(rotated_image)
    rotated_image.save(filename)
    return filename

def rotate_image_270(filename):
    image = Image.open(filename)
    image = asarray(image)
    rotated_image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
    rotated_image = Image.fromarray(rotated_image)
    rotated_image.save(filename)
    return filename

def h_flip(filename):
    image = Image.open(filename)
    image = asarray(image)
    rotated_image = cv.flip(image,1)
    rotated_image = Image.fromarray(rotated_image)
    rotated_image.save(filename)
    return filename

def v_flip(filename):
    image = Image.open(filename)
    image = asarray(image)
    rotated_image = cv.flip(image,0)
    rotated_image = Image.fromarray(rotated_image)
    rotated_image.save(filename)
    return filename

def resize_image(filename,amount1,amount2):
    image = Image.open(filename)
    image = asarray(image)
    image_resized = cv.resize(image, (amount1,amount2),interpolation=cv.INTER_NEAREST)
    image_resized=Image.fromarray(image_resized)
    image_resized.save(filename)
    return filename

def swirl_image(filename,strength,radius):
    image = Image.open(filename)
    image = asarray(image)
    swirled_image = swirl(image, rotation=5, strength=strength, radius=radius,mode='reflect')
    swirled = Image.fromarray((swirled_image * 255).astype(np.uint8))
    swirled.save(filename)
    return filename

def crop_image(filename,x1,x2,y1,y2):
    image = Image.open(filename)
    image = asarray(image)
    crop_img = image[y1:y2, x1:x2]
    cropped_image=Image.fromarray(crop_img)
    cropped_image.save(filename)
    return filename

def rescale_image(filename,amount):
    image = Image.open(filename)
    image = asarray(image)
    w = int(image.shape[1]*amount)
    h = int(image.shape[0] * amount)
    image_resized = cv.resize(image, (w,h), interpolation=cv.INTER_AREA)
    image_resized=Image.fromarray(image_resized)
    image_resized.save(filename)
    return filename
#histogram equalization resimleri gösterme
def equalize_histogram(filename, filename2):
    image = Image.open(filename)
    image = asarray(image)

    reference = Image.open(filename2)
    reference = asarray(reference)

    matched_image = match_histograms(image, reference, multichannel=True)
    plot=plot_equalized_histogram(image,reference,matched_image)
    image=Image.fromarray(matched_image)
    image.save(filename)
    return filename,plot


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
    return plt


#yogunluk donusumu işlemleri (degerleri kullanıcı verebilmeli)

#morfolojik işlemler (10 farklı morfolojik işlem icermeli)
from skimage.morphology import square,skeletonize_3d,thin,disk, dilation,erosion,skeletonize
from skimage.util import invert

#new_image = dilation(image)
#new_image=erosion(image)
#image=thin(data.horse())
#image=skeletonize_3d(data.horse())

#insta filtresi oluştur.

