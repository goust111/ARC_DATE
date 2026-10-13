from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MIN_GRID_SIZE = 11
MAX_GRID_SIZE = 30
MIN_MARGIN = 1
MAX_CORE_DIM = 10
CENTER_JITTER = 2
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    corner_counts = Counter(corners).most_common()
    if len(corner_counts) == 1 or corner_counts[0][1] > corner_counts[1][1]:
        return int(corner_counts[0][0])
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_solid_rect(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _is_frame_rect(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    region = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(region, expected))


def _extract_layers(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) >= 2, "expected at least one frame and one core"

    layers: list[dict[str, object]] = []
    for color in colors:
        mask = g == int(color)
        coords = np.argwhere(mask)
        assert coords.size > 0
        bbox = _bbox(coords)
        if _is_solid_rect(mask, bbox):
            kind = "core"
        elif _is_frame_rect(mask, bbox):
            kind = "frame"
        else:
            raise AssertionError(f"color {color} is neither a frame nor a solid core")
        layers.append({"color": int(color), "bbox": bbox, "kind": kind})

    layers.sort(key=lambda layer: layer["bbox"])  # outer layers have the smallest top-left corner

    for i in range(len(layers) - 1):
        outer = layers[i]
        inner = layers[i + 1]
        assert outer["kind"] == "frame", "all non-innermost layers must be frames"
        r0, c0, r1, c1 = outer["bbox"]
        nr0, nc0, nr1, nc1 = inner["bbox"]
        assert (nr0, nc0, nr1, nc1) == (
            r0 + 2,
            c0 + 2,
            r1 - 2,
            c1 - 2,
        ), "nested layers must be separated by a one-cell background gap"

    assert layers[-1]["kind"] == "core", "innermost layer must be the solid core"
    return bg, layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the colors of nested rectangular frames and render them as centered
    staircase-like horizontal color bands from outermost to innermost.
    """
    bg, layers = _extract_layers(grid)
    depth = len(layers)
    r0, c0, r1, c1 = layers[0]["bbox"]
    outer_h = int(r1 - r0 + 1)
    outer_w = int(c1 - c0 + 1)
    out_w = max(outer_h, outer_w)
    assert depth <= out_w

    out = np.full((depth, out_w), bg, dtype=int)
    for row, layer in enumerate(layers):
        band_w = depth - row
        start = (out_w - band_w) // 2
        out[row, start : start + band_w] = int(layer["color"])
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _layer_options(max_h: int, max_w: int) -> dict[str, list[tuple[int, int]]]:
    options = {"square": [], "landscape": [], "portrait": []}
    for h in range(1, min(MAX_CORE_DIM, max_h) + 1):
        for w in range(1, min(MAX_CORE_DIM, max_w) + 1):
            if h == w:
                options["square"].append((h, w))
            elif h < w:
                options["landscape"].append((h, w))
            else:
                options["portrait"].append((h, w))
    return options


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    depth = rng.randint(2, 7)
    inset = 4 * (depth - 1)
    max_core_h = MAX_GRID_SIZE - 2 * MIN_MARGIN - inset
    max_core_w = MAX_GRID_SIZE - 2 * MIN_MARGIN - inset
    if max_core_h < 1 or max_core_w < 1:
        return None

    shape_options = _layer_options(max_core_h, max_core_w)
    non_empty_shapes = [name for name, dims in shape_options.items() if dims]
    if not non_empty_shapes:
        return None
    shape_name = rng.choice(non_empty_shapes)
    core_h, core_w = rng.choice(shape_options[shape_name])

    outer_h = core_h + inset
    outer_w = core_w + inset
    max_extra_h = MAX_GRID_SIZE - outer_h
    max_extra_w = MAX_GRID_SIZE - outer_w
    if max_extra_h < 2 or max_extra_w < 2:
        return None

    grid_h = rng.randint(max(MIN_GRID_SIZE, outer_h + 2), outer_h + max_extra_h)
    grid_w = rng.randint(max(MIN_GRID_SIZE, outer_w + 2), outer_w + max_extra_w)

    center_top = (grid_h - outer_h) // 2
    center_left = (grid_w - outer_w) // 2
    top = rng.randint(
        max(MIN_MARGIN, center_top - CENTER_JITTER),
        min(grid_h - outer_h - MIN_MARGIN, center_top + CENTER_JITTER),
    )
    left = rng.randint(
        max(MIN_MARGIN, center_left - CENTER_JITTER),
        min(grid_w - outer_w - MIN_MARGIN, center_left + CENTER_JITTER),
    )

    colors = rng.sample(COLOR_POOL, depth)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    r0, c0 = top, left
    r1, c1 = top + outer_h - 1, left + outer_w - 1
    for color in colors[:-1]:
        _draw_frame(grid, (r0, c0, r1, c1), color)
        r0 += 2
        c0 += 2
        r1 -= 2
        c1 -= 2
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(colors[-1])

    bg, layers = _extract_layers(grid)
    if bg != BG:
        return None
    if [int(layer["color"]) for layer in layers] != [int(color) for color in colors]:
        return None
    if grid_h < MIN_GRID_SIZE or grid_w < MIN_GRID_SIZE:
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
