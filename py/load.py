import time
import os
import skimage as ski
import pyvips

dir = "Lab-1/input"

patches = []
t0 = time.perf_counter()
for dirname, _, files in os.walk(dir, topdown=False):
    for filename in files:
        patches.append(
            pyvips.Image.new_from_file(
                os.path.join(dirname, filename), access="sequential", memory=True
            )
        )
t1 = time.perf_counter()
print(f"vips loaded {len(patches)} images in {t1-t0:.3f}s")


patches = []
t0 = time.perf_counter()
for dirname, _, files in os.walk(dir, topdown=False):
    for filename in files:
        patches.append(ski.io.imread(os.path.join(dirname, filename)))
t1 = time.perf_counter()
print(f"skimage loaded {len(patches)} images in {t1-t0:.3f}s")
