import cv2
import numpy as np
import matplotlib.pyplot as plt


def reshapeToEven(img):
    """
    Change image shape so it's height and width become even number

    Params:

    img - image

    Returns: np.array with resized image
    """
    # reshaping an image to pass the closest even width and height numbers
    # mosaicing alghoritm works only with images of size (even, even, 3)
    img = img[:img.shape[0] - img.shape[0] % 2, :img.shape[1] - img.shape[1] % 2, :]

    return img


def BayerDemosaicing(img):
    # DEMOSAICING

    # kernel for red and blue pixel values:
    # [green, red, [green, blue, ...
    # blue, green][red, green]  ...
    redblueNearest = np.array([[1, 1],
                               [1, 1]])

    # kernel for green pixel values:
    greenNearest = np.array([[0.5, 0.5],
                             [0.5, 0.5]])

    kernels = [redblueNearest, greenNearest, redblueNearest]

    # applying filters on every color channel of the mosaiced image
    R, G, B = [cv2.filter2D(img[:, :, n], -1, kernels[n]) for n in range(3)]
    img = np.dstack((R, G, B))

    return img


def XTransDemosaicing(img):
    # DEMOSAICING
    deXTransMask = np.array([[0., 0., 0., 0., 0., 0.],
                            [0., 0.25, 0.5, 0.5, 0.25, 0.],
                            [0., 0.5, 1., 1., 0.5, 0.],
                            [0., 0.5, 1., 1., 0.5, 0.],
                            [0., 0.25, 0.5, 0.5, 0.25, 0.],
                            [0., 0., 0., 0., 0., 0.]])

    deXTransFilter = np.array([deXTransMask * w for w in [1/2, 1/5, 1/2]])

    B, G, R = [cv2.filter2D(img[:, :, n], -1, deXTransFilter[n]) for n in range(3)]
    img = np.dstack((B, G, R))

    return img


if __name__ == '__main__':
    bayer_img = cv2.imread("namib_bayer.jpg")
    xtrans_img = cv2.imread("namib_xtrans.jpg")
    bayer = BayerDemosaicing(bayer_img)
    xtrans = XTransDemosaicing(xtrans_img)
    cv2.imwrite("bayer.jpg", bayer)
    cv2.imwrite("xtrans.jpg", xtrans)



