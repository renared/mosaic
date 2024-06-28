import cv2
from glob import glob
import numpy as np
import pynndescent
from numba import njit, objmode, types
import pickle
import os
from joblib import Memory
import xxhash

cachedir = ".cache"

memory = Memory(cachedir, verbose=1)
    
features_desc = (
    "mean_L", "mean_A", "mean_B", "var",
)

def read_image(fn: str, size: tuple[int, int] | None = None):
    """
    Open an image file with opencv
    ------------------------------
    fn: filepath
    size: tuple (height, width) of target size  

    Returns
    -------
    im: np.ndarray of image in BGR color space  
    """
    im = cv2.imread(fn, cv2.IMREAD_COLOR)
    if size is not None:
        im = cv2.resize(im, size[::-1])
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    return im

@njit
def make_feature_vector(im: np.ndarray):
    """
    Calculate the features of an image
    """
    assert len(im.shape) == 3
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    mean_rgb = np.array([np.mean(im[:, :, k]) for k in range(im.shape[2])])
    var = np.mean(np.abs(im - mean_rgb[None, None]))
    return np.concatenate((mean_rgb, np.array([var])))

@memory.cache
@njit
def extract_feature_image(im:np.ndarray, kernel_size:tuple[int, int], stride:tuple[int, int]):
    """
    Compute the features of an image over a rolling window
    ------------------------------------------------------
    im: input image
    kernel_size: tuple (height, width) of the rolling window
    stride: tuple (dy, dx) of the hop sizes of the rolling window

    Returns
    -------
    feature_im: image of shape ((im.shape[0] - kernel_size[0]) // stride[0], (im.shape[1] - kernel_size[1]) // stride[1], num_features)
    """
    h, w, c = im.shape
    kh, kw = kernel_size
    sh, sw = stride
    ho, wo = int((h - kh) / sh + 1), int((w - kw) / sw + 1) 
    out = np.zeros((ho, wo, len(features_desc)))
    for i in range(ho):
        for j in range(wo):
            out[i, j] = make_feature_vector(im[i * sh : i * sh + kh, j * sw : j * sw + kw])
    return out

# @njit
# def mosaic_from_index_image(idxim:np.ndarray, thumbnails:np.ndarray, size: tuple[int, int] | None = None):
#     """
#     Construct a mosaic from an image (matrix) of indices corresponding to the thumbnails
#     ------------------------------------------------------------------------------------
#     idxim: ndarray of shape (height1, width1)
#     thumbnails: ndarray of shape (N, height0, width0, channels)

#     Returns
#     -------
#     im: mosaic image of shape (height0 * height1, width0 * width1, channels)
#     """
#     hnum, wnum = idxim.shape
#     hb, wb = thumbnails.shape[1:3]
#     ho, wo = hb * hnum, wb * wnum
#     out = np.zeros((ho, wo, thumbnails.shape[3]), dtype=thumbnails.dtype)
#     for i in range(hnum):
#         for j in range(wnum):
#             t = thumbnails[idxim[i, j]]
#             if size is not None: cv2.resize(t, size[::-1])
#             out[i * hb : i * hb + hb, j * wb : j * wb + wb] = t
#     return out

@njit
def mosaic_from_thumbnail_image(thumbim:np.ndarray):
    # thumbim has shape (hnum, wnum, hb, wb, channels)
    hnum, wnum = thumbim.shape[:2]
    hb, wb = thumbim.shape[2:4]
    ho, wo = hb * hnum, wb * wnum
    out = np.zeros((ho, wo, thumbim.shape[-1]), dtype=thumbim.dtype)
    for i in range(hnum):
        for j in range(wnum):
            out[i * hb : i * hb + hb, j * wb : j * wb + wb] = thumbim[i, j]
    return out

@memory.cache
# TODO invalidate if files changed
def make_dataset(filenames, thumbsize=(128, 128)):
    thumbs = np.stack([read_image(fn, size=thumbsize) for fn in filenames])
    features = np.stack([ make_feature_vector(thumb) for thumb in thumbs ])
    nnindex = pynndescent.NNDescent(features, verbose=True, n_jobs=-1, n_neighbors=1)
    return thumbs, features, nnindex

data_filenames = glob("./datasets/caplier/*.jpg")
thumbs, features, nnindex = make_dataset(data_filenames)

# load test image
im0 = read_image(r"D:\home\Pictures\Screenshots\Screenshot 2024-06-10 220119.png")
print("Extracting feature image...")
f0 = extract_feature_image(im0, (16, 16), (8, 8))

indices, distances = nnindex.query(
    f0.reshape(-1, f0.shape[2]),    # reshape the features image into (height * width, num_features)
    k=1)
indices = indices[:, 0] # ne garder que le plus proche voisin (c'est d'ailleurs le seul retourné)

# idxim = indices[:, 0].reshape(*f0.shape[:2])    # reshape the obtained index image into (height, width)

print("Making mosaic...")
# outim = mosaic_from_index_image(idxim, thumbs, size=(16, 16))
rescaled_thumbs = np.empty((indices.shape[0], 16, 16, 3))
for i, idx in enumerate(indices):
    rescaled_thumbs[i] = cv2.resize(thumbs[idx], (16, 16))
thumbim = rescaled_thumbs.reshape(*f0.shape[:2], 16, 16, 3)
outim = mosaic_from_thumbnail_image(thumbim)


print("Saving to out.png")
cv2.imwrite("out.png", outim)