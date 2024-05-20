import io
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from PIL import Image as pilimage
from PIL import ImageDraw, ImageFont
import mosaic
import numpy as np
import uvicorn


patches_path = "Lab-1/input"
imagepath = "input/neo.jpg"
tilesize = 256
blocksize = 20

image = pilimage.open(imagepath)
w, h = (image.width, image.height)

# debug
font = ImageFont.load_default(size=20)
color = 0x000000

app = FastAPI()

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

# extend left and down to fake a ring
matching_patch_indexes = np.concatenate(
    (matching_patch_indexes, matching_patch_indexes)
)
matching_patch_indexes = np.concatenate(
    (matching_patch_indexes, matching_patch_indexes), axis=1
)

print("indexes ready")


@app.get(
    "/tiles/{filename}_{z}_{x}_{y}",
    responses={200: {"Content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def root(filename: str, z: int, x: int, y: int):
    z = z if z >= -2 else -2
    sf = int(tilesize / 2**z)

    dx = x * sf % w
    dy = y * sf % h

    r1 = int(np.floor(dy / blocksize))
    r2 = int(np.ceil((dy + sf) / blocksize))
    c1 = int(np.floor(dx / blocksize))
    c2 = int(np.ceil((dx + sf) / blocksize))

    tilecanvas = np.zeros((3 * sf, 3 * sf, 3), dtype=np.uint8)
    center = int(sf)

    deltax = r2 - r1
    deltay = c2 - c1

    xorigin = int(center - (sf / 2) - (dy % blocksize))
    yorigin = int(center - (sf / 2) - (dx % blocksize))

    for r in range(deltax):
        for c in range(deltay):

            patch = patches[matching_patch_indexes[r + r1, c + c1]]
            # patch = patch.reshape((
            #     (xorigin + r * blocksize if xorigin + r * blocksize > 0 else 0) : (xorigin
            #     + (r + 1) * blocksize),
            #     (yorigin + c * blocksize if yorigin + c * blocksize > 0 else 0) : (yorigin
            #     + (c + 1) * blocksize),
            # :,
            # ))

            tilecanvas[
                xorigin + r * blocksize if xorigin + r * blocksize > 0 else 0 : xorigin
                + (r + 1) * blocksize,
                yorigin + c * blocksize if yorigin + c * blocksize > 0 else 0 : yorigin
                + (c + 1) * blocksize,
                :,
            ] = patch

    tile = tilecanvas[
        int(center - sf / 2) : int(center + sf / 2),
        int(center - sf / 2) : int(center + sf / 2),
        :,
    ]
    # tile = tilecanvas

    out = pilimage.fromarray(tile)

    # draw debug
    draw = ImageDraw.Draw(out)
    # draw.rectangle(
    #     (
    #         (center - sf / 2, center - sf / 2),
    #         (center + sf / 2, center + sf / 2),
    #     ),
    #     fill=None,
    #     outline=0x00FF00,
    #     width=3,
    # )

    out = out.resize((tilesize, tilesize))

    draw = ImageDraw.Draw(out)
    draw.text(
        (0, 0),
        f"{x},{y} z{z} .{sf}",
        color,
        font=font,
        stroke_fill=0xFFFFFF,
        stroke_width=1,
    )
    draw.text(
        (0, 20), f"{dx} {dy}", color, font=font, stroke_fill=0xFFFFFF, stroke_width=1
    )
    draw.text(
        (0, 40),
        f"{out.width} {out.height}",
        color,
        font=font,
        stroke_fill=0xFFFFFF,
        stroke_width=1,
    )
    draw.rectangle(
        ((0, 0), (out.width, out.height)),
        fill=None,
        outline=0x0000FF,
        width=1,
    )

    # send response
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
        },
    )


app.mount("/", StaticFiles(directory="public", html=True), name="public")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
