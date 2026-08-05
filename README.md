# The Triangles Project

The goal of the Triangles project is to invent the best possible solution to the following optimization problem:

_You are given a "target" RGB image and a budget of N triangles._
_You must assign vertex positions and RGBA colours for each triangle, such that when rasterized on a blank canvas (in order) the collection is maximally similar to the "target" image._

Effectively, we are "Painting" a target image with triangles.

## Evaluation

To evaluate, we standardize the methodology:

### Budget

Submissions must use `N=100` (preferred) or `N=500` (optional) triangles in their submissions, for consistency. 

### Submission Format

Solutions must emit either a rasterized RGB-PNG submission image matching the resolution of the target image, or an ordered list of 10-tuples (v0x, v0y, v1x, v1y, v2x, v2y, r, g, b, a) with floating-point vertices where (0,0) is the upper-left image corner and (1, 1) is the bottom right image corner, and r, g, b, a are all [0, 1]. Such submissions will be rasterized by a reference rasterizer into a submission image.

### Scoring

To score, we use the Root Mean-Squared Error (RMSE). We compute the absolute RMSE (denoted "E") between the target and submission images. We then compute a "baseline" image as the simple global colour-mean of the target image and compute the RMSE, denoted "B".

The final score is a percentage: `100* (1 - E / B)`. An error of `E=0` gives 100% similarity. An error of `0.5B` (half as much error as the baseline) gives 50% similarity.

The reference scoring script applies this formula independently to each image,
using the per-channel global RGB mean for its baseline, and then averages the
75 image scores equally:

```bash
.venv/bin/python scripts/score_rmse.py \
  --submission-dir outputs/full/geometrize/triangles100/ \
  --triangle-budget 100 \
  --runtime-seconds 1234.567 \
  --json-output results/full/geometrize/triangles100.json
```

### Dataset

[TrianglePaintBench](https://huggingface.co/datasets/benchislett/TrianglePaintBench) is a diverse collection of 75 images curated by me for this task. It also contains a subset of 10 images selected for more rapid development testing. See the HF repository for more details.

The final score is averaged over all 75 samples in the dataset.

### Time Limit

Interactivity is a relevant factor to open-ended optimizations problems like this one. However, it is _not_ a priority of this project.

There is _no_ fixed time limit, but submissions should include the total E2E runtime so that speed can be plotted against quality.
This way we can make a _pareto_ plot of all modes and their runtimes