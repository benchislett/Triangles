"""Gaussian hill climbing over an ordered collection of translucent triangles."""

import random
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Triangle:
    points: list[tuple[int, int]]
    color: list[int]  # RGBA, each channel 0-255


@dataclass
class OptimizationResult:
    image: Image.Image
    error: int
    mutations: int
    improvements: int
    elapsed: float

    @property
    def mutations_per_second(self) -> float:
        return self.mutations / self.elapsed if self.elapsed else 0.0


def initialize_triangles(
    count: int, size: tuple[int, int], rng: random.Random
) -> list[Triangle]:
    width, height = size
    return [
        Triangle(
            points=[(rng.randint(0, width), rng.randint(0, height)) for _ in range(3)],
            color=[rng.randrange(256), rng.randrange(256), rng.randrange(256), 0],
        )
        for _ in range(count)
    ]


def choose_background_color(
    target: Image.Image, color_hint: str | None
) -> tuple[int, int, int]:
    if color_hint:
        channel = 0 if color_hint == "black" else 255
        return channel, channel, channel
    r, g, b = (int(round(float(c))) for c in np.mean(np.asarray(target), axis=(0, 1)))
    return r, g, b


def render(
    triangles: list[Triangle],
    size: tuple[int, int],
    background_color: tuple[int, int, int],
) -> Image.Image:
    image = Image.new("RGB", size, background_color)
    draw = ImageDraw.Draw(image, "RGBA")
    for triangle in triangles:
        draw.polygon(triangle.points, fill=tuple(triangle.color))
    return image


def pixel_error(target: np.ndarray, candidate: Image.Image) -> int:
    candidate_pixels = np.asarray(candidate, dtype=np.int16)
    return int(np.abs(target - candidate_pixels).sum(dtype=np.int64))


def gaussian_value(current: int, maximum: int, rng: random.Random) -> int:
    standard_deviation = max(maximum / 6, 1.0)
    while True:
        candidate = round(rng.gauss(current, standard_deviation))
        if 0 <= candidate <= maximum and candidate != current:
            return candidate


def mutate_gaussian(
    triangles: list[Triangle], size: tuple[int, int], rng: random.Random
) -> tuple[int, Triangle]:
    """Perturb one channel or one vertex coordinate; return how to undo it."""
    index = rng.randrange(len(triangles))
    original = triangles[index]
    mutated = Triangle(original.points.copy(), original.color.copy())

    if rng.random() < 0.5:
        channel = rng.randrange(4)
        mutated.color[channel] = gaussian_value(mutated.color[channel], 255, rng)
    else:
        point_index, axis = rng.randrange(3), rng.randrange(2)
        point = list(mutated.points[point_index])
        point[axis] = gaussian_value(point[axis], size[axis], rng)
        mutated.points[point_index] = (point[0], point[1])

    triangles[index] = mutated
    return index, original


def optimize(
    triangles: list[Triangle],
    target_image: Image.Image,
    background_color: tuple[int, int, int],
    seconds: float,
    rng: random.Random,
) -> OptimizationResult:
    """Accept only strictly error-reducing mutations until the time budget runs out."""
    target_pixels = np.asarray(target_image, dtype=np.int16)
    best_image = render(triangles, target_image.size, background_color)
    best_error = pixel_error(target_pixels, best_image)
    improvements = 0
    mutations = 0
    started = time.perf_counter()
    deadline = started + seconds

    while time.perf_counter() < deadline:
        index, original = mutate_gaussian(triangles, target_image.size, rng)
        candidate = render(triangles, target_image.size, background_color)
        candidate_error = pixel_error(target_pixels, candidate)
        if candidate_error < best_error:
            best_image = candidate
            best_error = candidate_error
            improvements += 1
        else:
            triangles[index] = original
        mutations += 1

    return OptimizationResult(
        image=best_image,
        error=best_error,
        mutations=mutations,
        improvements=improvements,
        elapsed=time.perf_counter() - started,
    )
