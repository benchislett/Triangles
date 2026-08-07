# DiffVG triangle painting

`triangle_paint.py` optimizes ordered translucent triangles with DiffVG and
emits an RGB PNG compatible with TrianglePaintBench. DiffVG itself is not
vendored in this directory; clone and build it separately.

## Setup

Install PyTorch first, using a CUDA-enabled build on NVIDIA machines. A CUDA
build also needs the CUDA toolkit and `nvcc`.

```bash
git clone --recursive https://github.com/BachiLi/diffvg.git /path/to/diffvg
python -m pip install numpy pillow cssutils matplotlib pybind11 scikit-image svgpathtools
cd /path/to/diffvg
DIFFVG_CUDA=1 python setup.py build_ext --inplace  # use 0 for a CPU-only build
```

The local uncommitted `experiments/diff_vg` checkout contains compatibility
fixes for newer Python, CMake, pybind11, and macOS toolchains. Copy that checkout
separately if a fresh upstream clone does not build in the target environment.

## Current N=500 experiment

Run from the Triangles repository, placing the DiffVG checkout on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/diffvg python experiments/diffvg/triangle_paint.py \
  TrianglePaintBench/canonical/6f9a96514b68102a.png output.png \
  --device auto \
  --triangles 500 \
  --steps 400 \
  --resolution 512 \
  --point-lr 0.01 \
  --color-lr 0.01 \
  --full-resolution-steps 300 \
  --full-resolution-point-lr 0.001 \
  --full-resolution-color-lr 0.0005 \
  --full-resolution-warmup-steps 20 \
  --learning-rate-schedule cosine-to-zero
```

`--device auto` uses CUDA when PyTorch exposes it and otherwise uses CPU. Use
`--device cuda` or `--device cpu` to require one path. On multi-GPU systems,
select the GPU with `CUDA_VISIBLE_DEVICES`.
