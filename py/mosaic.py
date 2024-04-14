import os
import numpy as np
from PIL import Image as pilimage
import time
import sys
import scipy.spatial.kdtree as KDTree


def load_patches(path):
    patches = []
    for dirname, _, files in os.walk(path, topdown=False):
        for filename in files:
            patches.append(np.array(pilimage.open(os.path.join(dirname, filename))))
    patches = patches
    # (N, x, y, c)
    return patches


def resize_patches_to_blocksize(patches, blocksize):
    resized_patches = []
    for patch in patches:
        im = pilimage.fromarray(patch)
        resized_patches.append(np.array(im.resize((blocksize, blocksize))))
    return resized_patches
    # (n, 1)


def compute_patches_meancolors(patches):
    return np.mean(patches, axis=(1, 2))
    # (n, 3)


# return index of matching patch
# O(n) !!!!
def color_nearest_patch_index(patches_mean_colors, color):
    distances = np.sum((patches_mean_colors - color) ** 2, axis=1)
    # (n)
    return np.argmin(distances)
    # int


def match_blocks_patches_indexes(blocks_mean_colors, patches_mean_colors):
    rows, cols, _ = blocks_mean_colors.shape
    indexes = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            indexes[r, c] = color_nearest_patch_index(
                patches_mean_colors, blocks_mean_colors[r, c]
            )
    return indexes
    # (rows, cols) ???


def load_inputimage(path):
    return np.array(pilimage.open(path))


def write_image(image, path):
    out = pilimage.fromarray(image)
    out.save(path)


def slice_blocks(image, blocksize):
    height, width, _ = image.shape
    rows = height // blocksize
    cols = width // blocksize

    blocks = image.reshape(rows, blocksize, cols, blocksize, 3)
    # (rows, blocksize, cols, blocksize, 3)
    return blocks.swapaxes(1, 2)
    # (rows, cols, blocksize, blocksize, 3)


def compute_blocks_meancolors(blocks):
    return np.mean(blocks, axis=(2, 3))
    # (rows, cols, 3)


def make_pixel_image(rows, cols, blockcolors, scale=1):
    image = np.zeros((rows * scale, cols * scale, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            image[
                r * scale : (r + 1) * scale,
                c * scale : (c + 1) * scale,
                :,
            ] = blockcolors[r, c]
    return image


def make_mosaic_image(matching_patch_indexes, patches, blocksize):
    rows, cols = matching_patch_indexes.shape
    image = np.zeros((rows * blocksize, cols * blocksize, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            image[
                r * blocksize : (r + 1) * blocksize,
                c * blocksize : (c + 1) * blocksize,
                :,
            ] = patches[matching_patch_indexes[r, c]]
    return image


patches_path = "Lab-1/input"
# inputimage_path = "Lab-1/input/0.jpg"
inputimage_path = "/mnt/c/Users/pl/Pictures/IMG_4547.jpg"
# inputimage_path = "/mnt/c/Users/pl/Pictures/oeil2300.jpg"
blocksize = 50


## patches
print("loading patches...")
t0 = time.perf_counter()

patches = load_patches(patches_path)

t1 = time.perf_counter()
print(f"{len(patches)} loaded in {t1-t0:.3f}s")

print(f"resizing patches to {blocksize}px...")
t0 = time.perf_counter()

patches = resize_patches_to_blocksize(patches, blocksize)

t1 = time.perf_counter()
print(f"{len(patches)} resized in {t1-t0:.3f}s")


print("computing patches mean colors")
patches_mean_colors = compute_patches_meancolors(patches)

## image

inputimage = load_inputimage(inputimage_path)

height, width, _ = inputimage.shape
rows = height // blocksize
cols = width // blocksize

blocks = slice_blocks(inputimage, blocksize)
blocks_mean_colors = compute_blocks_meancolors(blocks)
pixelimage = make_pixel_image(rows, cols, blocks_mean_colors, blocksize)

matching_patch_indexes = match_blocks_patches_indexes(
    blocks_mean_colors, patches_mean_colors
)

mosaicimage = make_mosaic_image(matching_patch_indexes, patches, blocksize)

write_image(pixelimage, "output/pixel.jpg")
write_image(mosaicimage, "output/mosaic.jpg")
