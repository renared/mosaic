import pyvips
import numpy as np

# from numba import njit


def info(img):
    fields = img.get_fields()
    print(fields)
    for field in fields:
        if not field.startswith("vips") and not field.startswith("jpeg"):
            val = img.get(field)
            print(field, val)


blocksize = 100
inputfilename = "/mnt/c/Users/pl/Pictures/IMG_4547.jpg"
# inputfilename = "Lab-1/input/0.jpg"

image = inputimage = pyvips.Image.new_from_file(
    inputfilename,
    access="sequential",
    memory=True,
)
inputimage = image.numpy()
# c is of shape (3) : an array [RED,GREEN,BLUE]
# inputimage is of shape (witdh, height, c)

# info(image)


## plutot que de faire ca on peut resize limage et voila

height, width, _ = inputimage.shape
rows = height // blocksize
cols = width // blocksize

blocks = inputimage.reshape(rows, blocksize, cols, blocksize, 3)
# blocks is of shape (rows, blocksize, cols, blocksize, 3)
blocks = blocks.swapaxes(1, 2)
# blocks is of shape (rows, cols, blocksize, blocksize, 3)
blockcolors = np.mean(blocks, axis=(2, 3))
# blockcolors is of shape (rows, cols, 3)


# @njit
def makeimg(rows, cols, blockcolors, blocksize):
    outputimage = np.zeros((height, width, 3))
    for r in range(rows):
        for c in range(cols):
            # colorsquare = np.full((blocksize, blocksize, 3), blockcolors[r, c])
            outputimage[
                r * blocksize : (r + 1) * blocksize,
                c * blocksize : (c + 1) * blocksize,
                :,
            ] = blockcolors[r, c]
    return outputimage


outputimage = makeimg(rows, cols, blockcolors, blocksize)

out = pyvips.Image.new_from_array(outputimage)

out.write_to_file(f"color.jpg")
