#!/usr/bin/env python3
"""Score RGB PNG submissions using TrianglePaintBench's normalized RMSE."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image, UnidentifiedImageError
except ModuleNotFoundError as error:
    raise SystemExit(
        "Pillow is required; install requirements-scoring.txt in a virtual environment"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SHAPE_TYPES = (
    "rectangle",
    "rotated_rectangle",
    "triangle",
    "ellipse",
    "rotated_ellipse",
    "circle",
    "line",
    "quadratic_bezier",
    "polyline",
)


class ScoreError(Exception):
    """Raised when benchmark inputs are invalid."""


@dataclass(frozen=True)
class SampleScore:
    name: str
    error_rmse: float
    baseline_rmse: float
    score: float


def load_rgb_png(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ScoreError(f"{path}: expected PNG data, found {image.format}")
            if image.mode != "RGB":
                raise ScoreError(f"{path}: expected RGB mode, found {image.mode}")
            return np.array(image, dtype=np.float64, copy=True)
    except (OSError, UnidentifiedImageError) as error:
        raise ScoreError(f"{path}: could not decode image: {error}") from error


def score_arrays(name: str, target: np.ndarray, submission: np.ndarray) -> SampleScore:
    if target.shape != submission.shape:
        raise ScoreError(
            f"{name}: dimensions differ: target {target.shape[1]}x{target.shape[0]}, "
            f"submission {submission.shape[1]}x{submission.shape[0]}"
        )

    error_rmse = float(np.sqrt(np.mean(np.square(target - submission))))
    mean_color = np.mean(target, axis=(0, 1), dtype=np.float64)
    baseline_rmse = float(np.sqrt(np.mean(np.square(target - mean_color))))

    if baseline_rmse == 0.0:
        if error_rmse == 0.0:
            score = 100.0
        else:
            raise ScoreError(
                f"{name}: target is constant, so normalized score is undefined for a non-perfect submission"
            )
    else:
        score = 100.0 * (1.0 - error_rmse / baseline_rmse)

    return SampleScore(name, error_rmse, baseline_rmse, score)


def png_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ScoreError(f"not a directory: {directory}")
    return {path.name: path for path in directory.iterdir() if path.suffix.lower() == ".png"}


def matched_files(
    target_dir: Path,
    submission_dir: Path,
    expected_count: int,
) -> list[tuple[str, Path, Path]]:
    targets = png_files(target_dir)
    submissions = png_files(submission_dir)

    if expected_count and len(targets) != expected_count:
        raise ScoreError(
            f"expected {expected_count} target PNGs in {target_dir}, found {len(targets)}"
        )

    target_names = set(targets)
    submission_names = set(submissions)
    missing = sorted(target_names - submission_names)
    extra = sorted(submission_names - target_names)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} submission(s): {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {len(extra)} submission(s): {', '.join(extra)}")
        raise ScoreError("; ".join(details))

    names = sorted(target_names & submission_names)
    if not names:
        raise ScoreError("no matching target and submission PNG filenames")

    return [(name, targets[name], submissions[name]) for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 100 * (1 - submission RMSE / mean-color baseline RMSE)."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=ROOT / "TrianglePaintBench/canonical",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=ROOT / "outputs/full/geometrize/triangles100",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=75,
        help="required number of target PNGs; use 0 to disable (default: 75)",
    )
    parser.add_argument(
        "--triangle-budget",
        type=int,
        choices=(100, 500),
        default=100,
        help="triangle budget recorded in JSON metadata (default: 100)",
    )
    parser.add_argument(
        "--shape-types",
        default="triangle",
        help="comma-separated primitive types recorded in JSON metadata (default: triangle)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="write a deterministic, version-control-friendly JSON score record",
    )
    parser.add_argument(
        "--runtime-seconds",
        type=float,
        help="include the full benchmark E2E wall time in the JSON submission record",
    )
    args = parser.parse_args()
    if args.expected_count < 0:
        parser.error("--expected-count cannot be negative")
    if args.runtime_seconds is not None and (
        not math.isfinite(args.runtime_seconds) or args.runtime_seconds < 0.0
    ):
        parser.error("--runtime-seconds must be finite and non-negative")
    if args.runtime_seconds is not None and args.json_output is None:
        parser.error("--runtime-seconds requires --json-output")
    try:
        args.shape_types = normalize_record_shape_types(args.shape_types)
    except ValueError as error:
        parser.error(str(error))
    return args


def normalize_record_shape_types(shape_types: str) -> tuple[str, ...]:
    selected = set()
    for raw_name in shape_types.split(","):
        name = raw_name.strip().lower().replace("-", "_")
        if not name:
            raise ValueError("shape type list contains an empty value")
        if name == "all":
            return SUPPORTED_SHAPE_TYPES
        if name not in SUPPORTED_SHAPE_TYPES:
            raise ValueError(f"unknown shape type: {name}")
        selected.add(name)
    return tuple(name for name in SUPPORTED_SHAPE_TYPES if name in selected)


def record_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_record(
    scores: list[SampleScore],
    target_dir: Path,
    submission_dir: Path,
    expected_count: int,
    triangle_budget: int = 100,
    shape_types: tuple[str, ...] = ("triangle",),
    runtime_seconds: float | None = None,
) -> dict:
    final_score = math.fsum(result.score for result in scores) / len(scores)
    submission = {
        "directory": record_path(submission_dir),
        "shape_count": triangle_budget,
        "shape_types": list(shape_types),
    }
    if runtime_seconds is not None:
        submission["runtime_seconds"] = round(runtime_seconds, 3)
        submission["runtime_seconds_per_sample"] = round(
            runtime_seconds / len(scores), 6
        )

    return {
        "submission": submission,
        "result": {
            "sample_count": len(scores),
            "score_percent": round(final_score, 6),
        },
        "benchmark": {
            "name": "TrianglePaintBench",
            "split": "full" if expected_count == 75 else "custom",
            "triangle_budget": triangle_budget,
            "target_directory": record_path(target_dir),
            "expected_sample_count": expected_count,
        },
        "samples": [
            {
                "id": Path(result.name).stem,
                "error_rmse": round(result.error_rmse, 6),
                "baseline_rmse": round(result.baseline_rmse, 6),
                "score_percent": round(result.score, 6),
            }
            for result in scores
        ],
    }


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        pairs = matched_files(
            args.target_dir,
            args.submission_dir,
            args.expected_count,
        )
        scores = [
            score_arrays(name, load_rgb_png(target), load_rgb_png(submission))
            for name, target, submission in pairs
        ]
    except ScoreError as error:
        raise SystemExit(f"score_rmse: {error}") from error

    print(f"{'sample':20} {'E RMSE':>12} {'B RMSE':>12} {'score':>12}")
    for result in scores:
        print(
            f"{result.name:20} {result.error_rmse:12.6f} "
            f"{result.baseline_rmse:12.6f} {result.score:11.6f}%"
        )

    final_score = math.fsum(result.score for result in scores) / len(scores)
    print(f"FINAL SCORE ({len(scores)} samples): {final_score:.6f}%")
    if args.json_output:
        write_record(
            args.json_output,
            build_record(
                scores,
                args.target_dir,
                args.submission_dir,
                args.expected_count,
                triangle_budget=args.triangle_budget,
                shape_types=args.shape_types,
                runtime_seconds=args.runtime_seconds,
            ),
        )
        print(f"JSON record: {args.json_output}")


if __name__ == "__main__":
    main()
