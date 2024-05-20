import os
from PIL import Image
import numpy as np
import argparse
from progress.bar import ChargingBar


def process_patches(inputpath: str, outputpath: str, size: int, verbose: bool):
    i = 0
    os.makedirs(outputpath, exist_ok=True)
    for dirname, _, files in os.walk(inputpath, topdown=False):
        for filename in files:
            if filename.endswith(".jpg"):
                patch = Image.open(os.path.join(dirname, filename))

                # patch.load()

                patch = square_image(patch)
                patch = resize_image(patch, size)

                out = os.path.join(outputpath, f"{i}.jpg")
                i += 1
                save_image(patch, out)

                if verbose:
                    print(f"{inputpath}/{filename} -> {out}")
            else:
                if verbose:
                    print(f"skipped {inputpath}/{filename}")


def square_image(image: Image.Image):
    x, y = image.width, image.height
    if x > y:
        return image.crop(
            (
                np.floor(x / 2 - y / 2),
                0,
                np.floor(x / 2 + y / 2),
                y,
            )
        )
    elif x < y:
        return image.crop(
            (
                0,
                np.floor(y / 2 - x / 2),
                x,
                np.floor(y / 2 + x / 2),
            )
        )
    return image


def resize_image(image: Image.Image, size: int):
    return image.resize((size, size))


def save_image(image: Image.Image, path: str):
    image.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-s", "--size", type=int, default=256)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    process_patches(args.input, args.output, args.size, args.verbose)
