import cv2
from glob import glob
import numpy as np
import pynndescent
from numba import njit, objmode, types
import pickle
import os

def hashmoi(bytes) -> str:
    import hashlib
    return hashlib.sha256(bytes, usedforsecurity=False).hexdigest()[:8]

num_features = 4

def read_image(fn:str, size=None):
    im = cv2.imread(fn, cv2.IMREAD_COLOR)
    if size is not None:
        im = cv2.resize(im, size[::-1])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    return im

@njit
def make_feature_vector(im:np.ndarray):
    assert len(im.shape) == 3
    mean_rgb = np.array([np.mean(im[:, :, k]) for k in range(im.shape[2])])
    var = np.mean(np.abs(im - mean_rgb[None, None]))
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

@njit
def mosaic_from_index_image(idxim:np.ndarray, block_size:tuple[int, int], thumbnails:np.ndarray):
    hnum, wnum = idxim.shape
    hb, wb = block_size
    ho, wo = hb * hnum, wb * wnum
    out = np.zeros((ho, wo, 3), dtype=thumbnails.dtype)
    for i in range(hnum):
        for j in range(wnum):
            out[i * hb : i * hb + hb, j * wb : j * wb + wb] = thumbnails[idxim[i, j]]
    return out

# construct or load dataset
data_filenames = glob("./input/*.jpg")
hashstr = hashmoi(str(sorted(data_filenames)).encode())
if os.path.exists(index_fn := "./.index/"+hashstr+".pkl"):
    print(f"Loading {index_fn}...")
    with open(index_fn, "rb") as fp:
        index = pickle.load(fp)
else:
    print(f"Creating {index_fn}...")
    data_features = np.stack([ make_feature_vector(read_image(fn)) for fn in data_filenames ])
    print(f"{data_features.shape=}")
    # make knn index
    index = pynndescent.NNDescent(data_features, verbose=True, n_jobs=-1, n_neighbors=1)
    with open(index_fn, "wb") as fp:
        pickle.dump(index, fp)

# load test image
im0 = read_image(r"D:\backupTODO\DATA4\Dossier personnel\Desktop\scorch_flame02b (c) 01.png")
print("Extracting feature image...")
f0 = extract_feature_image(im0, (16, 16), (8, 8))

# print("Querying...")
indices, distances = index.query(f0.reshape(-1, f0.shape[2]), k=1)
idxim = indices[:, 0].reshape(*f0.shape[:2])
# idxim = np.random.randint(0, len(data_filenames), size=f0.shape[:2])

print("Loading thumbnails")
thumbs = np.stack([ read_image(fn, size=(16, 16)) for fn in data_filenames ])

print("Making mosaic...")
outim = mosaic_from_index_image(idxim, (16, 16), thumbs)

print("Saving to out.png")
cv2.imwrite("out.png", cv2.cvtColor(outim, cv2.COLOR_LAB2BGR))