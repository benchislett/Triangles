#!/usr/bin/env python3
"""Approximate one RGB image with translucent triangles using DiffVG."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pydiffvg
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="target RGB image")
    parser.add_argument("output", type=Path, help="output RGB PNG")
    parser.add_argument(
        "-n", "--triangles", type=int, default=100, help="triangle count (default: 100)"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Adam optimization steps (default: 200)"
    )
    parser.add_argument(
        "--full-resolution-steps",
        type=int,
        default=0,
        help="additional Adam steps at native resolution (default: 0)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="longest optimization-canvas side (default: 128)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="samples per axis while optimizing (default: 1)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="render device; auto uses CUDA when available (default: auto)",
    )
    parser.add_argument(
        "--point-lr",
        type=float,
        default=0.02,
        help="initial point learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--color-lr",
        type=float,
        default=0.01,
        help="initial color learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--full-resolution-point-lr",
        type=float,
        default=0.005,
        help="point learning rate after the native-stage optimizer reset (default: 0.005)",
    )
    parser.add_argument(
        "--full-resolution-color-lr",
        type=float,
        default=0.0025,
        help="color learning rate after the native-stage optimizer reset (default: 0.0025)",
    )
    parser.add_argument(
        "--full-resolution-warmup-steps",
        type=int,
        default=0,
        help="native-stage linear warmup from 10%% of its learning rates (default: 0)",
    )
    parser.add_argument(
        "--background",
        choices=("mean", "black", "white"),
        default="mean",
        help="opaque canvas color (default: target mean)",
    )
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "staged-decay", "cosine-to-zero"),
        default="constant",
        help="learning-rate schedule (default: constant)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    args = parser.parse_args()

    if args.triangles < 1:
        parser.error("--triangles must be at least 1")
    if args.steps < 0:
        parser.error("--steps cannot be negative")
    if args.full_resolution_steps < 0:
        parser.error("--full-resolution-steps cannot be negative")
    if args.resolution < 1:
        parser.error("--resolution must be at least 1")
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    for name in (
        "point_lr",
        "color_lr",
        "full_resolution_point_lr",
        "full_resolution_color_lr",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.full_resolution_warmup_steps < 0:
        parser.error("--full-resolution-warmup-steps cannot be negative")
    if args.full_resolution_warmup_steps:
        if args.learning_rate_schedule != "cosine-to-zero":
            parser.error(
                "--full-resolution-warmup-steps requires "
                "--learning-rate-schedule cosine-to-zero"
            )
        if args.full_resolution_warmup_steps >= args.full_resolution_steps:
            parser.error(
                "--full-resolution-warmup-steps must be less than "
                "--full-resolution-steps"
            )
    if args.output.suffix.lower() != ".png":
        parser.error("output must have a .png extension")
    return args


def resized_target(image: Image.Image, longest_side: int) -> Image.Image:
    scale = min(1.0, longest_side / max(image.size))
    size = tuple(max(1, round(dimension * scale)) for dimension in image.size)
    return image if size == image.size else image.resize(size, Image.Resampling.LANCZOS)


def background_color(target: torch.Tensor, choice: str) -> torch.Tensor:
    if choice == "black":
        return torch.zeros(3)
    if choice == "white":
        return torch.ones(3)
    return target.mean(dim=(0, 1))


def initialize_triangles(
    count: int, target: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.rand((count, 2), generator=generator)
    rotations = torch.rand((count, 1), generator=generator) * (2.0 * math.pi)
    angles = rotations + torch.arange(3) * (2.0 * math.pi / 3.0)
    radii = 0.05 + 0.25 * torch.rand((count, 3), generator=generator)
    offsets = (
        torch.stack((torch.cos(angles), torch.sin(angles)), dim=2) * radii[:, :, None]
    )
    points = (centers[:, None, :] + offsets).clamp(0.0, 1.0)

    height, width = target.shape[:2]
    pixel_x = (centers[:, 0] * (width - 1)).round().long()
    pixel_y = (centers[:, 1] * (height - 1)).round().long()
    colors = torch.cat((target[pixel_y, pixel_x], torch.full((count, 1), 0.35)), dim=1)
    return points.requires_grad_(), colors.requires_grad_()


def render(
    points: torch.Tensor,
    colors: torch.Tensor,
    size: tuple[int, int],
    canvas_color: torch.Tensor,
    samples: int,
    seed: int,
) -> torch.Tensor:
    width, height = size
    scale = torch.tensor([width, height])
    shapes = [
        pydiffvg.Polygon(points=triangle * scale, is_closed=True) for triangle in points
    ]
    groups = [
        pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([index]),
            fill_color=color,
        )
        for index, color in enumerate(colors)
    ]
    background = torch.empty((height, width, 4))
    background[:, :, :3] = canvas_color
    background[:, :, 3] = 1.0
    scene = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, groups)
    return pydiffvg.RenderFunction.apply(
        width,
        height,
        samples,
        samples,
        seed,
        background,
        *scene,
    )[:, :, :3]


def rmse_score(target: torch.Tensor, candidate: torch.Tensor) -> float:
    error = torch.sqrt(torch.mean((target - candidate) ** 2))
    baseline = torch.sqrt(torch.mean((target - target.mean(dim=(0, 1))) ** 2))
    if baseline == 0:
        return 100.0 if error == 0 else float("-inf")
    return 100.0 * (1.0 - error.item() / baseline.item())


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda requested, but CUDA is unavailable to this PyTorch install"
        )
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    pydiffvg.set_device(device)
    device_label = str(device)
    if use_cuda:
        device_label += f" ({torch.cuda.get_device_name(device)})"
    print(f"render device: {device_label}")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    with Image.open(args.target) as source:
        target_image = source.convert("RGB")
    optimization_image = resized_target(target_image, args.resolution)
    optimization_target = torch.from_numpy(
        np.array(optimization_image, dtype=np.float32, copy=True) / 255.0
    )
    native_target = torch.from_numpy(
        np.array(target_image, dtype=np.float32, copy=True) / 255.0
    )
    canvas_color = background_color(optimization_target, args.background)
    points, colors = initialize_triangles(
        args.triangles, optimization_target, generator
    )
    optimization_target = optimization_target.to(device)
    native_target = native_target.to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [points], "lr": args.point_lr},
            {"params": [colors], "lr": args.color_lr},
        ]
    )

    started = time.perf_counter()
    initial = render(
        points,
        colors,
        optimization_image.size,
        canvas_color,
        args.samples,
        args.seed,
    )
    initial_score = rmse_score(optimization_target, initial.detach())
    stages = [(optimization_image, optimization_target, args.steps)]
    if args.full_resolution_steps:
        stages.append((target_image, native_target, args.full_resolution_steps))

    completed_steps = 0
    for stage_index, (stage_image, stage_target, stage_steps) in enumerate(stages, 1):
        if args.learning_rate_schedule != "constant" and stage_index == 2:
            optimizer = torch.optim.Adam(
                [
                    {"params": [points], "lr": args.full_resolution_point_lr},
                    {"params": [colors], "lr": args.full_resolution_color_lr},
                ]
            )
        scheduler = None
        if args.learning_rate_schedule == "cosine-to-zero":
            if stage_index == 2 and args.full_resolution_warmup_steps:
                warmup_steps = args.full_resolution_warmup_steps
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=0.1,
                            end_factor=1.0,
                            total_iters=warmup_steps,
                        ),
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=stage_steps - warmup_steps,
                            eta_min=0.0,
                        ),
                    ],
                    milestones=[warmup_steps],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, stage_steps), eta_min=0.0
                )
        print(
            f"stage {stage_index}/{len(stages)}: {stage_steps} steps at "
            f"{stage_image.width}x{stage_image.height}"
        )
        report_interval = max(1, stage_steps // 10)
        for stage_step in range(stage_steps):
            if args.learning_rate_schedule == "staged-decay":
                if stage_index == 1:
                    decay_start = round(stage_steps * 0.6)
                    if stage_step < decay_start:
                        scale = 1.0
                    else:
                        progress = (stage_step - decay_start + 1) / max(
                            1, stage_steps - decay_start
                        )
                        scale = 0.2 + 0.8 * (1.0 + math.cos(math.pi * progress)) / 2.0
                    point_lr = args.point_lr * scale
                    color_lr = args.color_lr * scale
                else:
                    progress = stage_step / max(1, stage_steps - 1)
                    scale = 0.5 + 0.5 * (1.0 + math.cos(math.pi * progress)) / 2.0
                    point_lr = args.full_resolution_point_lr * scale
                    color_lr = args.full_resolution_color_lr * scale
                optimizer.param_groups[0]["lr"] = point_lr
                optimizer.param_groups[1]["lr"] = color_lr
            optimizer.zero_grad()
            candidate = render(
                points,
                colors,
                stage_image.size,
                canvas_color,
                args.samples,
                args.seed + completed_steps + 1,
            )
            loss = torch.mean((candidate - stage_target) ** 2)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                points.clamp_(0.0, 1.0)
                colors.clamp_(0.0, 1.0)
            completed_steps += 1
            if (stage_step + 1) % report_interval == 0 or stage_step + 1 == stage_steps:
                learning_rates = ""
                if args.learning_rate_schedule != "constant":
                    learning_rates = (
                        f", point lr {optimizer.param_groups[0]['lr']:.6f}, "
                        f"color lr {optimizer.param_groups[1]['lr']:.6f}"
                    )
                print(
                    f"stage {stage_index} step {stage_step + 1:4d}/{stage_steps}: "
                    f"loss {loss.item():.6f}{learning_rates}"
                )

    final = render(
        points.detach(),
        colors.detach(),
        target_image.size,
        canvas_color,
        max(2, args.samples),
        args.seed + completed_steps + 1,
    ).detach()
    output_pixels = (
        (final.clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_pixels, mode="RGB").save(args.output, format="PNG")

    elapsed = time.perf_counter() - started
    stage_summary = " -> ".join(
        f"{stage_steps}@{stage_image.width}x{stage_image.height}"
        for stage_image, _, stage_steps in stages
    )
    final_score = rmse_score(native_target, final)
    print(
        f"{args.triangles} triangles, {stage_summary}: RMSE score "
        f"{initial_score:.2f}% -> {final_score:.2f}%, {elapsed:.2f}s"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
