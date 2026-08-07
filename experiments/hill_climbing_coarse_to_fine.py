#!/usr/bin/env python3
"""Run Gaussian hill climbing from a coarse canvas to full resolution."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from hill_climbing import (
    choose_background_color,
    initialize_triangles,
    optimize,
    render,
)

SCALE_FACTORS = (16, 8, 4, 2, 1)


def rmse_score_percent(target: Image.Image, candidate: Image.Image) -> float:
    """TrianglePaintBench score: RMSE normalized against a mean-color baseline."""
    target_pixels = np.asarray(target, dtype=np.float64)
    candidate_pixels = np.asarray(candidate, dtype=np.float64)
    error_rmse = float(np.sqrt(np.mean(np.square(target_pixels - candidate_pixels))))
    mean_color = np.mean(target_pixels, axis=(0, 1), dtype=np.float64)
    baseline_rmse = float(np.sqrt(np.mean(np.square(target_pixels - mean_color))))
    if baseline_rmse == 0:
        return 100.0 if error_rmse == 0 else float("-inf")
    return 100.0 * (1.0 - error_rmse / baseline_rmse)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate one image by applying Gaussian hill climbing at "
            "progressively finer resolutions."
        )
    )
    parser.add_argument("target", type=Path, help="target image")
    parser.add_argument("output", type=Path, help="output RGB PNG")
    parser.add_argument(
        "-n",
        "--triangles",
        type=int,
        default=100,
        help="number of triangles (default: 100)",
    )
    parser.add_argument(
        "-s",
        "--seconds",
        type=float,
        default=60.0,
        help="total optimization runtime across all levels (default: 60)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--starting-scale",
        type=int,
        choices=SCALE_FACTORS,
        default=16,
        help="initial downsample factor; 1 runs a single full-resolution pass",
    )
    parser.add_argument(
        "--background-color-hint",
        type=str.lower,
        choices=("black", "white"),
        help="TrianglePaintBench background color hint; otherwise use target mean",
    )
    args = parser.parse_args()

    if args.triangles < 1:
        parser.error("--triangles must be at least 1")
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and non-negative")
    if args.output.suffix.lower() != ".png":
        parser.error("output must have a .png extension")
    return args


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    with Image.open(args.target) as source:
        target_image = source.convert("RGB")
    background_color = choose_background_color(
        target_image, args.background_color_hint
    )

    scale_factors = [s for s in SCALE_FACTORS if s <= args.starting_scale]
    total_weight = sum(s**2 for s in scale_factors)
    triangles = None
    previous_size = target_image.size
    results = []

    for scale_factor in scale_factors:
        stage_size = (
            max(1, round(target_image.width / scale_factor)),
            max(1, round(target_image.height / scale_factor)),
        )
        stage_target = (
            target_image
            if scale_factor == 1
            else target_image.resize(stage_size, Image.Resampling.LANCZOS)
        )

        if triangles is None:
            triangles = initialize_triangles(args.triangles, stage_size, rng)
        else:
            x_scale = stage_size[0] / previous_size[0]
            y_scale = stage_size[1] / previous_size[1]
            for triangle in triangles:
                triangle.points = [
                    (round(x * x_scale), round(y * y_scale))
                    for x, y in triangle.points
                ]
        previous_size = stage_size

        starting_score = rmse_score_percent(
            stage_target, render(triangles, stage_size, background_color)
        )
        result = optimize(
            triangles,
            stage_target,
            background_color,
            args.seconds * scale_factor**2 / total_weight,
            rng,
        )
        results.append(result)
        print(
            f"{scale_factor}x ({stage_size[0]}x{stage_size[1]}): "
            f"{result.mutations} mutations, {result.improvements} improvements, "
            f"RMSE score {starting_score:.2f}% -> "
            f"{rmse_score_percent(stage_target, result.image):.2f}%, "
            f"{result.elapsed:.2f}s, {result.mutations_per_second:.1f} mutations/s"
        )

    final_image = results[-1].image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(args.output, format="PNG")

    mutations = sum(r.mutations for r in results)
    elapsed = sum(r.elapsed for r in results)
    print(
        f"total: {args.triangles} triangles, {mutations} mutations, "
        f"{sum(r.improvements for r in results)} improvements, "
        f"{rmse_score_percent(target_image, final_image):.2f}% final RMSE score, "
        f"{elapsed:.2f}s, {mutations / elapsed if elapsed else 0.0:.1f} mutations/s"
    )


if __name__ == "__main__":
    main()
