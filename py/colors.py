import pyvips
import time
import numpy as np


def info(img):
    fields = img.get_fields()
    print(fields)
    for field in fields:
        if not field.startswith("vips") and not field.startswith("jpeg"):
            val = img.get(field)
            print(field, val)


blocksize = 100
inputfilename = "/mnt/c/Users/pl/Pictures/IMG_4547.jpg"

image = inputimage = pyvips.Image.new_from_file(
    inputfilename,
    access="sequential",
    memory=True,
)
inputimage = image.numpy()
# c is of shape (3) : an array [RED,GREEN,BLUE]
# inputimage is of shape (witdh, height, c)

# info(image)


height, width, _ = inputimage.shape
rows = height // blocksize
cols = width // blocksize

blocks = inputimage.reshape(rows, blocksize, cols, blocksize, 3)
# blocks is of shape (rows, blocksize, cols, blocksize, 3)

blocks = blocks.swapaxes(1, 2)
# blocks is of shape (rows, cols, blocksize, blocksize, 3)

blocks = blocks.reshape(-1, blocksize, blocksize, 3)
# blocks is of shape (rows*cols , blocksize, blocksize, 3)


blockcolors = [np.mean(block, axis=(0, 1)) for block in blocks]  # .astype(int)

outputimage = pyvips.Image.black(width, height, bands=3).numpy()
# outputimage = pyvips.Image.new_from_array(inputimage)


# info(outputimage)

for r in range(rows):
    for c in range(cols):
        color = list(blockcolors[c + r * (width // blocksize)])
        # print(r, c, color, r * blocksize, c * blocksize)
        patch = np.full((blocksize, blocksize, 3), color)
        outputimage[
            r * blocksize : (r + 1) * blocksize,
            c * blocksize : (c + 1) * blocksize,
            :,
        ] = patch
        # outputimage = outputimage.draw_rect(
        #     color, r * blocksize, c * blocksize, blocksize, blocksize, fill=True
        # )

out = pyvips.Image.new_from_array(outputimage)

out.write_to_file(f"color.jpg")
