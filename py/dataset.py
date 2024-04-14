import os
import pyvips
import numpy as np

dir = "Lab-1/input"

patches = []

for dirname, _, files in os.walk(dir, topdown=False):
    for filename in files[:100]:
        patches.append(
            pyvips.Image.new_from_file(
                os.path.join(dirname, filename), access="sequential", memory=True
            )
        )

patches = np.array(patches)
# (N, x, y, c)

patchesmeancolors = np.mean(patches, axis=(1, 2))
# (N, meancolor)

color = np.array([123, 29, 234])

distances = np.sum((patchesmeancolors - color) ** 2)

image = patches[np.argmin(distances)]
