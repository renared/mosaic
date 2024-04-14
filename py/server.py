import io
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from PIL import Image as pilimage
from PIL import ImageDraw, ImageFont

imagepath = "output/mosaic.jpg"
# imagepath = "Lab-1/input/0.jpg"
image = pilimage.open(imagepath)
tilesize = 256
imageblocksize = 100

w, h = (image.width, image.height)

# debug
font = ImageFont.load_default(size=20)
color = 0x000000

app = FastAPI()


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

    tile = image.crop((dx, dy, dx + sf, dy + sf))

    # patch missing borders
    if dx + sf > w:
        opposite_x = image.crop((0, dy, abs(w - (dx + sf)), dy + sf))
        tile.paste(
            opposite_x,
            box=(
                int((sf - abs(w - (dx + sf)))),
                0,
            ),
        )
    if dy + sf > h:
        opposite_y = image.crop((dx, 0, dx + sf, abs(h - (dy + sf))))
        tile.paste(
            opposite_y,
            box=(
                0,
                int((sf - abs(h - (dy + sf)))),
            ),
        )
    if dx + sf > w and dy + sf > h:
        opposite_corner = image.crop(
            (
                0,
                0,
                abs(w - (dx + sf)),
                abs(h - (dy + sf)),
            )
        )
        tile.paste(
            opposite_corner,
            box=(
                int((sf - abs(w - (dx + sf)))),
                int((sf - abs(image.height - (dy + sf)))),
            ),
        )

    resized_tile = tile.resize((tilesize, tilesize))

    # draw debug
    draw = ImageDraw.Draw(resized_tile)
    draw.text(
        (0, 0),
        f"{x},{y} z{z} .{sf}",
        color,
        font=font,
        stroke_fill=0xFFFFFF,
        stroke_width=2,
    )
    draw.text(
        (0, 20), f"{dx} {dy}", color, font=font, stroke_fill=0xFFFFFF, stroke_width=2
    )
    draw.text(
        (0, 40),
        f"{resized_tile.width} {resized_tile.height}",
        color,
        font=font,
        stroke_fill=0xFFFFFF,
        stroke_width=2,
    )
    draw.rectangle(
        ((0, 0), (resized_tile.width, resized_tile.height)),
        fill=None,
        outline=0x0000FF,
        width=3,
    )

    # send response
    buf = io.BytesIO()
    resized_tile.save(buf, format="JPEG")
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
        },
    )


app.mount("/", StaticFiles(directory="public", html=True), name="public")
