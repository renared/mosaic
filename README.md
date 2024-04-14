## requirements

- libvips
- python3.10 & pipenv

## run

run tiles server

```shell
uvicorn py.server:app --reload
```

make mosaic

```shell
python py/mosaic.py
```
