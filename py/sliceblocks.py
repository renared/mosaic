import pyvips
import time

blocksize = 64

inputfilename = "Lab-1/input/1.jpg"

t0 = time.perf_counter()
inputimage = pyvips.Image.new_from_file(
    inputfilename, access="sequential", memory=True
).numpy()

t1 = time.perf_counter()
print(f"image loaded in {t1-t0:.5f}s")

height, width, _ = inputimage.shape
rows = height // blocksize
cols = width // blocksize

blocks = inputimage.reshape(rows, blocksize, cols, blocksize, 3)
# print(blocks.shape)

blocks = blocks.swapaxes(1, 2).reshape(-1, blocksize, blocksize, 3)
# print(blocks.shape)

t2 = time.perf_counter()
print(f"blocks sliced in {t2-t1:.5f}s")

blockimages = [pyvips.Image.new_from_array(block) for block in blocks]

t3 = time.perf_counter()
print(f"block images made in {t3-t2:.5f}s")

for i, block in enumerate(blockimages):
    block.write_to_file(f"output/block{i}.jpg")

t4 = time.perf_counter()
print(
    f"{len(blockimages)} block images (blocksize: {blocksize}px) saved in {t4-t3:.5f}s ({len(blockimages)/(t4-t3):.0f}/s)"
)
