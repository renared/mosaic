from PIL import Image as pilimage
import mosaic
import numpy as np
from PIL import ImageDraw


patches_path = "Lab-1/input"
imagepath = "output/mosaic.jpg"
# imagepath = "Lab-1/input/0.jpg"
tilesize = 256
blocksize = 50

image = pilimage.open(imagepath)
w, h = (image.width, image.height)

# debug

patches = mosaic.load_patches(patches_path)
patches = mosaic.resize_patches_to_blocksize(patches, blocksize)
patches_mean_colors = mosaic.compute_patches_meancolors(patches)

inputimage = np.array(image)

height, width, _ = inputimage.shape
rows = height // blocksize
cols = width // blocksize

blocks = mosaic.slice_blocks(inputimage, blocksize)
blocks_mean_colors = mosaic.compute_blocks_meancolors(blocks)
pixelimage = mosaic.make_pixel_image(rows, cols, blocks_mean_colors, blocksize)

matching_patch_indexes = mosaic.match_blocks_patches_indexes(
    blocks_mean_colors, patches_mean_colors
)
print("indexes ready")


if __name__ == "__main__":
    z = 1
    x = 1
    y = 1

    z = z if z >= -2 else -2
    sf = int(tilesize / 2**z)

    dx = x * sf % w
    dy = y * sf % h

    r1 = int(np.floor(dx / blocksize))
    c1 = int(np.floor(dy / blocksize))
    r2 = int(np.ceil((dx + sf) / blocksize))
    c2 = int(np.ceil((dy + sf) / blocksize))

    tilecanvas = np.zeros((2 * sf, 2 * sf, 3), dtype=np.uint8)
    center = int(sf)

    deltax = r2 - r1
    deltay = c2 - c1

    xorigin = int(center - (sf / 2) - (dx % blocksize))
    yorigin = int(center - (sf / 2) - (dy % blocksize))

    for r in range(deltax):
        for c in range(deltay):

            tilecanvas[
                xorigin + r * blocksize : xorigin + (r + 1) * blocksize,
                yorigin + c * blocksize : yorigin + (c + 1) * blocksize,
                :,
            ] = patches[matching_patch_indexes[r + r1, c + c1]]

    tile = tilecanvas[
        int(center - sf / 2) : int(center + sf / 2),
        int(center - sf / 2) : int(center + sf / 2),
        :,
    ]

    out = pilimage.fromarray(tile)

    # draw = ImageDraw.Draw(out)
    # draw.crop(
    #     (
    #         (center - sf / 2, center - sf / 2),
    #         (center + sf / 2, center + sf / 2),
    #     ),
    #     fill=None,
    #     outline=0x0000FF,
    #     width=1,
    # )
    out.save("output/tile.jpg")
