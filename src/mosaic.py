import cv2
from glob import glob
import os
import numpy as np
import pynndescent
from tqdm import tqdm
from numba import njit, objmode, types

num_features = 4

def read_image(fn:str):
    im = cv2.imread(fn, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

@njit
def make_feature_vector(im:np.ndarray):
    assert len(im.shape) == 3
    mean_rgb = np.array([np.mean(im[:, :, k]) for k in range(im.shape[2])])
    var = 0#np.mean(np.abs(im - mean_rgb[None, None]))
    return np.concatenate((mean_rgb, np.array([var])))

@njit
def extract_feature_image(im:np.ndarray, kernel_size:tuple[int, int], stride:tuple[int, int]):
    """"""
    h, w, c = im.shape
    kh, kw = kernel_size
    sh, sw = stride
    ho, wo = int((h - kh) / sh + 1), int((w - kw) / sw + 1) 
    out = np.zeros((ho, wo, num_features))
    for i in range(ho):
        for j in range(wo):
            out[i, j] = make_feature_vector(im[i * sh : i * sh + kh, j * sw : j * sw + kw])
    return out

def mosaic_from_index_image(idxim:np.ndarray, block_size:tuple[int, int], data_filenames:list[str]):
    hnum, wnum = idxim.shape
    hb, wb = block_size
    ho, wo = hb * hnum, wb * wnum
    out = np.zeros((ho, wo, 3), dtype=np.float32)
    for i in range(hnum):
        for j in range(wnum):
            thum = cv2.imread(data_filenames[idxim[i, j]])
            thum = cv2.cvtColor(thum, cv2.COLOR_BGR2RGB)
            thum = cv2.resize(thum, block_size[::-1])
            out[i * hb : i * hb + hb, j * wb : j * wb + wb] = thum
    return out

# construct dataset
data_filenames = glob("./input/*.jpg")
data_features = np.stack([ make_feature_vector(read_image(fn)) for fn in data_filenames ])
print(f"{data_features.shape=}")

# make knn index
index = pynndescent.NNDescent(data_features, verbose=True, n_jobs=-1, n_neighbors=1)

# load test image
im0 = read_image(r"D:\home\Downloads\20240507_155839.jpg")
print("Extracting feature image...")
f0 = extract_feature_image(im0, (16, 16), (16, 16))

# print("Querying...")
indices, distances = index.query(f0.reshape(-1, f0.shape[2]), k=1)
idxim = indices[:, 0].reshape(*f0.shape[:2])
# idxim = np.random.randint(0, len(data_filenames), size=f0.shape[:2])

print("Making mosaic...")
outim = mosaic_from_index_image(idxim, (16, 16), data_filenames)

print("Saving to out.png")
cv2.imwrite("out.png", cv2.cvtColor(outim, cv2.COLOR_RGB2BGR))