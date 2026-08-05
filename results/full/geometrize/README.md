# Evaluating Geometrize

Geometrize was evaluated on a range of shape-types, with both N=100 and N=500 total shapes.

In each run, we use `--candidates 100 --mutations 100 --alpha 128`.

Most runs took between 5 and 30 seconds per image.

## Results overview

- N=500 improved results significantly: +11% boost on average.
- Triangles produced the highest score at both budgets: 63.98% at N=100 and 74.55% at N=500.
- Rotated ellipses were the closest alternative to triangles, trailing by less
  than one point at both budgets while taking less time per image.
- Combining all shape types did not outperform the strongest individual
  primitive family.

## Full results

Sorted by N=100 score. A dash indicates that no completed result was recorded.

| Primitive set | N=100 score | N=100 sec/image | N=500 score | N=500 sec/image | Score gain |
|---|---:|---:|---:|---:|---:|
| Triangles | 63.98% | 10.32 | 74.55% | 22.81 | +10.57 pts |
| Rotated ellipses | 63.30% | 9.30 | 73.98% | 17.30 | +10.67 pts |
| All shapes | 62.12% | 8.36 | 73.46% | 16.65 | +11.35 pts |
| Rotated rectangles | 61.83% | 7.13 | 73.11% | 12.92 | +11.28 pts |
| Ellipses | 60.31% | 6.52 | 71.78% | 10.75 | +11.47 pts |
| Rectangles | 59.01% | 5.76 | 70.56% | 10.52 | +11.55 pts |
| Circles | 56.35% | 6.34 | 69.09% | 11.53 | +12.73 pts |

I did not run the line-based shapes (bezier / polyline), as they seem to have very low quality.