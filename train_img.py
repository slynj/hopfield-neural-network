import numpy as np
import skimage.io
import skimage.color
import skimage.transform
import skimage.filters
import matplotlib.pyplot as plt
from hopfield_network import HopfieldNetwork

np.random.seed(523)

def img_preprocessing(path, w=128, h=128):
    """ Preprocesses the image into a black/white image with a default size of
        128 * 128. Returns a flattened 1D vectory where the elements are bipolar (-1/1).

    Args:
        path (str): imge file path
        w (int, optional): Width of the img. Defaults to 128.
        h (int, optional): Height of the img. Defaults to 128.

    Returns:
        numpy.ndarray: 1D array with the length w * h. 
    """
    img = skimage.io.imread(path)
    img_bw = skimage.color.rgb2gray(img)
    img_resize = skimage.transform.resize(img_bw, (w, h), mode='reflect')

    threshold = skimage.filters.threshold_mean(img_resize)
    binary = img_resize > threshold
    bipolar = 2 * (binary * 1) - 1

    flatten = np.reshape(bipolar, (w * h))

    return flatten

