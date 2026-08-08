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

You may need to apply additional patches to diffvg to get it working as it's a bit old

## Run the experiment

Run from the Triangles repository, placing the DiffVG checkout on `PYTHONPATH`.
For example, `steps-1000-n500` used:

```bash
PYTHONPATH=/path/to/diffvg python results/full/diffvg/sweep/triangle_paint.py \
  TrianglePaintBench/canonical/6f9a96514b68102a.png output.png \
  --device auto \
  --triangles 500 \
  --steps 1000 \
  --resolution 512 \
  --samples 1 \
  --background mean \
  --point-lr 0.004 \
  --color-lr 0.00392156862745098 \
  --full-resolution-steps 500 \
  --full-resolution-point-lr 0.0008 \
  --full-resolution-color-lr 0.000784313725490196 \
  --full-resolution-warmup-steps 20 \
  --learning-rate-schedule cosine-to-zero \
  --seed 0
```

`--device auto` uses CUDA when PyTorch exposes it and otherwise uses CPU. Use
`--device cuda` or `--device cpu` to require one path. On multi-GPU systems,
select the GPU with `CUDA_VISIBLE_DEVICES`.

## Sweep results

The sweep ran each configuration on all 75 `TrianglePaintBench/full` images.
Only the triangle count, step counts, and warmup varied as shown below; all
other parameters match the command above.

| Config | Triangles | Stage 1 steps | Full-resolution steps | Warmup steps | Score | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| steps-4000-n500 | 500 | 4000 | 2000 | 80 | 74.812524% | 24314.413s |
| steps-2000-n500 | 500 | 2000 | 1000 | 40 | 74.372928% | 11764.006s |
| steps-1000-n500 | 500 | 1000 | 500 | 20 | 73.731293% | 5942.403s |
| steps-500-n500 | 500 | 500 | 250 | 10 | 72.762652% | 3060.277s |
| steps-250-n500 | 500 | 250 | 125 | 5 | 70.764076% | 1596.291s |
| steps-4000-n100 | 100 | 4000 | 2000 | 80 | 65.351460% | 14886.101s |
| steps-2000-n100 | 100 | 2000 | 1000 | 40 | 64.903946% | 7180.798s |
| steps-1000-n100 | 100 | 1000 | 500 | 20 | 64.434699% | 3643.885s |
| steps-500-n100 | 100 | 500 | 250 | 10 | 63.421163% | 1885.056s |
| steps-250-n100 | 100 | 250 | 125 | 5 | 61.350672% | 1010.636s |
