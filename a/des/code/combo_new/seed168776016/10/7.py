from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_RETRIES = 300
K_BY_PARITY = {
    0: (4, 6),
    1: (3, 5),
}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_rect_frame(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = np.zeros_like(mask, dtype=bool)
    _draw_frame(expected, bbox, True)
    return np.array_equal(mask, expected)


def _extract_frames(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
    frames: list[tuple[tuple[int, int, int, int], int]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == BG:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "foreground color must appear at least once"
        bbox = _bbox(coords)
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        assert _is_rect_frame(mask, bbox), "every foreground object must be a hollow rectangle"
        frames.append((bbox, color))

    frames.sort(key=lambda item: (item[0][0], item[0][1], -item[0][2], -item[0][3]))
    assert len(frames) >= 3, "expected at least three nested frames"

    center_r = frames[0][0][0] + frames[0][0][2]
    center_c = frames[0][0][1] + frames[0][0][3]
    for idx in range(1, len(frames)):
        outer = frames[idx - 1][0]
        inner = frames[idx][0]
        assert center_r == inner[0] + inner[2] and center_c == inner[1] + inner[3], (
            "all frames must share a common center"
        )
        assert outer[0] < inner[0] < inner[2] < outer[2], "frames must be strictly nested by rows"
        assert outer[1] < inner[1] < inner[3] < outer[3], "frames must be strictly nested by columns"
        assert inner[0] - outer[0] >= 2 and inner[1] - outer[1] >= 2, (
            "there must be at least one background cell between adjacent frames"
        )

    return frames


def _paint_corridor(
    out: np.ndarray,
    grid: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    out[mask & (grid == BG)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill alternating nested-rectangle corridors using the innermost frame color.

    Input structure:
    - The background is black.
    - The foreground consists of one-pixel-thick hollow rectangular frames with distinct colors.
    - The frames are strictly nested, share a common center, and leave a one-cell black corridor between adjacent layers.
    - The innermost frame encloses a black interior.

    Rule:
    - Order the frames from outermost layer 1 to innermost layer K.
    - Let C be the color of the innermost frame.
    - For each corridor between layer i and layer i+1, fill that corridor with C iff corridor index i has the same parity as K.
    - Keep the other corridors black, preserve every frame border, and fill the innermost interior with C only when K is odd.

    Key insight:
    - The only color that propagates is the innermost frame color, and the target regions are determined by the parity pattern of the nesting depth rather than by local adjacency.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    k = len(frames)
    inner_bbox, inner_color = frames[-1]

    out = g.copy()
    for corridor_idx in range(1, k):
        if corridor_idx % 2 != k % 2:
            continue
        outer_bbox = frames[corridor_idx - 1][0]
        next_bbox = frames[corridor_idx][0]
        _paint_corridor(out, g, outer_bbox, next_bbox, inner_color)

    if k % 2 == 1:
        r0, c0, r1, c1 = inner_bbox
        interior = np.zeros_like(g, dtype=bool)
        interior[r0 + 1 : r1, c0 + 1 : c1] = True
        out[interior & (g == BG)] = inner_color

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray:
    parity = rng.randint(0, 1)
    k = rng.choice(K_BY_PARITY[parity])

    max_inner_h = 28 - 4 * (k - 1)
    max_inner_w = 28 - 4 * (k - 1)
    if max_inner_h < 1 or max_inner_w < 1:
        raise AssertionError("selected nesting depth does not fit the ARC size limit")

    inner_h = rng.randint(1, max_inner_h)
    inner_w = rng.randint(1, max_inner_w)

    outer_h = inner_h + 2 + 4 * (k - 1)
    outer_w = inner_w + 2 + 4 * (k - 1)
    assert outer_h <= 30 and outer_w <= 30

    extra_h = rng.randint(0, 30 - outer_h)
    extra_w = rng.randint(0, 30 - outer_w)
    top = rng.randint(0, extra_h)
    left = rng.randint(0, extra_w)
    grid_h = outer_h + extra_h
    grid_w = outer_w + extra_w

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    colors = rng.sample(PALETTE, k)
    for layer, color in enumerate(colors):
        inset = 2 * layer
        bbox = (
            top + inset,
            left + inset,
            top + outer_h - 1 - inset,
            left + outer_w - 1 - inset,
        )
        _draw_frame(grid, bbox, color)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
