import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from numpy import asarray
from skimage import data, io, filters, feature, exposure, color, util, img_as_float, morphology
import skimage.io as io
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.filters import LPIFilter2D, wiener, sobel, prewitt_v
from skimage.morphology import closing, square, area_closing, area_opening, diameter_closing, convex_hull, \
    convex_hull_image, disk, black_tophat, opening, skeletonize_3d
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import *
import scipy
from scipy import misc
import cv2 as cv
from skimage.util import crop, invert
from skimage.exposure import match_histograms, rescale_intensity
from skimage import color, data, restoration
from scipy.signal import convolve2d


#goruntu iyileştirme islemleri,filtreler
# "Sobel, ""Prewitt V", "Prewitt H", "Hessian", 'Median', "Meijering", "Frangi", "Laplacian", "Gaussian", 'Sato'
@adapt_rgb(each_channel)
def sobel_each(image):
    return sobel(image)

def sobel_filter(filename):
    img=asarray(Image.open(filename))
    img=sobel_each(img)
    plt.imsave(filename,img)
    return filename


def prewitt_V(filename):
    img=asarray(Image.open(filename))
    img = rgb2gray(img)
    img=prewitt_v(img)
    plt.imsave(filename,img,cmap="gray")
    return filename

def prewitt_H(filename):
    img = asarray(Image.open(filename))
    img = rgb2gray(img)
    img = filters.prewitt_h(img)
    plt.imsave(filename,img,cmap="gray")
    return filename

#DONE
def hessian_filter(filename):
    img = asarray(Image.open(filename))
    img = rgb2gray(img)
    img = filters.hessian(img,mode="reflect")
    plt.imsave(filename,img,cmap="gray")
    return filename
#DONE
def median_filter(filename):
    img = asarray(Image.open(filename))
    img = rgb2gray(img)
    img = filters.median(img)
    plt.imsave(filename,img,cmap="gray")
    return filename

#DONE
def meijering_filter(filename):
    img = asarray(Image.open(filename))
    img = rgb2gray(img)
    img = filters.meijering(img)
    plt.imsave(filename,img,cmap="gray")
    return filename

#DONE
def frangi_filter(filename):
    img = asarray(Image.open(filename))
    img = rgb2gray(img)
    img = filters.frangi(img)
    plt.imsave(filename,img,cmap="gray")
    return filename

#DONE ?
def laplacian_filter(filename):
    img = asarray(Image.open(filename))
    img = rgb2gray(img)
    img = filters.laplace(img)
    plt.imsave(filename,img,cmap="gray")
    return filename

#DONE
def gaussian_filter(filename):
    img = asarray(Image.open(filename))
    img = rgb2gray(img)
    img = filters.gaussian(img)
    plt.imsave(filename,img,cmap="gray")
    return filename

#DONE
def sato_filter(filename):
    img = asarray(Image.open(filename))
    img = rgb2gray(img)
    img = filters.sato(img)
    plt.imsave(filename,img,cmap="gray")
    return filename

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


#DONE
def rescale_int(filename,val1,val2):
    image = asarray(Image.open(filename))
    image=rescale_intensity(image,in_range=(val1,val2))
    plt.imsave(filename,image)
    return filename

def adjust_ga(filename):
    image = asarray(Image.open(filename))
    image=exposure.adjust_gamma(image)
    plt.imsave(filename, image)
    return filename

def adjust_lo(filename,gain=1):
    image = Image.open(filename)
    image = asarray(image)
    image=exposure.adjust_log(image,gain=gain)
    plt.imsave(filename,image)
    return filename

def adjust_sig(filename,cutoff=0.5,gain=10):
    image = Image.open(filename)
    image = asarray(image)
    image=exposure.adjust_sigmoid(image,cutoff=cutoff,gain=gain)
    plt.imsave(filename,image)
    return filename


#10 morphological operations

@adapt_rgb(each_channel)
def erosion_each(image):
    return morphology.erosion(image)

def erosion_func(filename):
    image = Image.open(filename)
    image = asarray(image)
    erosion=erosion_each(image)
    plt.imsave(filename, erosion)
    return filename

#Dilation enlarges bright regions and shrinks dark regions.
@adapt_rgb(each_channel)
def dilation_each(image):
    return morphology.dilation(image,disk(6))

def dilation_func(filename):
    image = Image.open(filename)
    image = asarray(image)
    dilation=dilation_each(image)
    plt.imsave(filename, dilation)
    return filename

#I THINK THEYRE DONE
def thin_func(filename):
    image = Image.open(filename)
    image = asarray(image)
    image = invert(asarray(image))
    image=rgb2gray(image)
    thin=morphology.thin(image)
    plt.imsave(filename, thin,cmap='gray')
    return filename

#I THINK THEYRE DONE
def skeletonize_func(filename):
    image = Image.open(filename)
    image = invert(asarray(image))
    image=rgb2gray(image)
    skeletonize=morphology.skeletonize(image)
    plt.imsave(filename, skeletonize,cmap='gray')
    return filename

#I THINK THEYRE DONE
def skeletonize3d_func(filename):
    image = Image.open(filename)
    image = invert(asarray(image))
    image=rgb2gray(image)
    skeletonize=morphology.skeletonize_3d(image)
    plt.imsave(filename, skeletonize,cmap='gray')
    return filename

#Opening can remove small bright spots and connect small dark cracks.
#defined as an erosion followed by a dilation
def opening_func(filename):
    image = Image.open(filename)
    image = asarray(image)
    image=rgb2gray(image)
    image=morphology.opening(image,disk(4))
    plt.imsave(filename, image,cmap='gray')
    return filename

# Closing can remove small dark spots and connect small bright cracks.
#defined as a dilation followed by an erosion
def closing_func(filename):
    image = Image.open(filename)
    image = asarray(image)
    image=rgb2gray(image)
    closed=morphology.closing(image,disk(6))
    plt.imsave(filename, closed,cmap='gray')
    return filename

#The convex_hull_image is the set of pixels included in the
# smallest convex polygon that surround all white pixels in the input image
def convex_func(filename):
    image = Image.open(filename)
    image = asarray(image)
    image=rgb2gray(image)
    image=convex_hull_image(image)
    plt.imsave(filename, image,cmap='gray')
    return filename

#defined as the image minus its morphological opening. This operation returns
# the bright spots of the image that are smaller than the structuring element.
def white_top_func(filename):
    image = Image.open(filename)
    image = asarray(image)
    image = rgb2gray(image)
    image=morphology.white_tophat(image,disk(6))
    plt.imsave(filename, image,cmap='gray')
    return filename

#defined as its morphological closing minus the original image. This operation returns
# the dark spots of the image that are smaller than the structuring element.
def black_top_func(filename):
    image = Image.open(filename)
    image = asarray(image)
    image = rgb2gray(image)
    image=black_tophat(image)
    plt.imsave(filename, image,cmap='gray')
    return filename


