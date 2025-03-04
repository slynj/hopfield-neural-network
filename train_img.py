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
    

    # convert img
    img = skimage.io.imread(path)

    # check if if it has 4 channels (rgba) => set to rgb
    if img.shape[-1] >= 3:
        img = img[:, :, :3]

    img_bw = skimage.color.rgb2gray(img)
    img_resize = skimage.transform.resize(img_bw, (w, h), mode='reflect')

    # set state
    threshold = skimage.filters.threshold_mean(img_resize)
    binary = img_resize > threshold
    bipolar = 2 * (binary * 1) - 1

    # resize to 1D array
    flatten = np.reshape(bipolar, (w * h))

    return flatten


def img_corrupted(input, level):
    """ Generate corrupted input given probability and input array (1D array)

    Args:
        input (array): 1D array
        level (flaot): Probability of each value being inverted

    Returns:
        array: 1D array, corrupted
    """
    corrupted = np.copy(input)
    invert = np.random.binomial(n=1, p=level, size=len(input))

    for i, v in enumerate(input):
        if invert[i]:
            corrupted[i] = -1 * v

    return corrupted


def reshape_sq(arr):
    """ Reshpae 1D data to square matrix.

    Args:
        arr (array): 1D array.

    Returns:
        numpy.ndarray: Square matrix.
    """
    dimension = int(np.sqrt(len(arr)))
    arr = np.reshape(arr, (dimension, dimension))
    return arr


def plot(data, test, predicted, figsize=(5, 6)):
    """ Plots the train, test, predicted data (img).

    Args:
        data (array of arrays): Array of numpy.ndarray (1D) or any other 
                                array like types. Storing train data.
        test (array of arrays): Same thing, storing corrupted data.
        predicted (array of arrays): Same thing, storing predicted data.
        figsize (tuple, optional): Final result figure size. Defaults to (5, 6).
    """
    data = [reshape_sq(d) for d in data]
    test = [reshape_sq(t) for t in test]
    predicted = [reshape_sq(p) for p in predicted]

    fig, grid_fig = plt.subplots(len(data), 3, figsize=figsize)

    # only works when there are 1+ images in data (fix later!)
    for i in range(len(data)):
        if i == 0:
            grid_fig[i, 0].set_title("Train Data")
            grid_fig[i, 1].set_title("Input Data")
            grid_fig[i, 2].set_title("Output Data")

        grid_fig[i, 0].imshow(data[i], cmap='gray')
        grid_fig[i, 0].axis("off")

        grid_fig[i, 1].imshow(test[i], cmap='gray')
        grid_fig[i, 1].axis("off")

        grid_fig[i, 2].imshow(predicted[i], cmap='gray')
        grid_fig[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("result/result.png")
    plt.show()


def main():
    # load imgs
    oikawa = "train_img/oikawa.jpeg"
    gojo = "train_img/gojo.webp"
    sjs = "train_img/sjs.png"

    data = [oikawa, gojo, sjs]

    # preprocessing
    print("Data Processing ...")
    data = [img_preprocessing(d) for d in data] # now a list of numpy arrays

    # network model
    hnet = HopfieldNetwork()
    hnet.train(data)

    # generate test
    test = [img_corrupted(d, 0.4) for d in data]
    
    # get result
    predicted = hnet.predict(test)

    print("Show Prediction Results ...")
    plot(data, test, predicted)


if __name__ == '__main__':
    main()
