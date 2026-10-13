from __future__ import annotations

import random
from typing import Iterable

import numpy as np


"""
ARC-visible variant of "expand each rectangle by one halo, keep only halo overlaps".

The original prompt describes choosing the overlap color by an invisible
color-name ordering. ARC grids do not expose color names, so this
implementation uses a visible ordering cue instead: for every halo-overlap
pixel, paint it with the color of the rectangle whose top-left corner comes
later in reading order `(row, col)`.
"""


BG = 0
COLOR_POOL = tuple(range(1, 10))

N_OBJECTS_RANGE = (3, 5)
RECT_H_RANGE = (2, 4)
RECT_W_RANGE = (2, 4)
PADDING_RANGE = (1, 4)
EXTRA_SLACK_MAX = 3
MAX_CONTENT_SPAN = 18
MAX_RETRIES = 400


def _present_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != BG)


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, tuple[int, int, int, int] | int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BG), "expected black background"

    colors = _present_colors(g)
    assert 3 <= len(colors) <= 5, "expected 3-5 foreground rectangles"

    objects = []
    for color in colors:
        rows, cols = np.where(g == color)
        assert rows.size > 0
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        assert window.shape[0] * window.shape[1] == rows.size, "object must be a solid rectangle"
        assert np.all(window == color), "each color must form one solid rectangle"
        objects.append({"color": color, "bbox": (r0, c0, r1, c1)})

    for i, first in enumerate(objects):
        fr0, fc0, fr1, fc1 = first["bbox"]
        for second in objects[i + 1 :]:
            sr0, sc0, sr1, sc1 = second["bbox"]
            row_gap = _interval_gap(fr0, fr1, sr0, sr1)
            col_gap = _interval_gap(fc0, fc1, sc0, sc1)
            assert not (row_gap == 0 and col_gap == 0), "rectangles must be separated by background"

    objects.sort(key=lambda item: item["bbox"])
    return objects


def _halo_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], bg_mask: np.ndarray) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    halo = np.zeros(shape, dtype=bool)
    halo[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True
    halo[r0 : r1 + 1, c0 : c1 + 1] = False
    halo &= bg_mask
    return halo


def _halo_covers(
    objects: list[dict[str, tuple[int, int, int, int] | int]], shape: tuple[int, int], bg_mask: np.ndarray
) -> np.ndarray:
    return np.stack([_halo_mask(shape, obj["bbox"], bg_mask) for obj in objects], axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the overlapping parts of the one-cell 8-neighborhood halos of
    distinct solid rectangles.

    Because color-name ordering is not observable in ARC, the visible ordering
    cue is rectangle reading order: when multiple halos cover the same
    background pixel, use the color of the rectangle whose top-left corner is
    later in reading order.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_rectangles(g)
    bg_mask = g == BG
    covers = _halo_covers(objects, g.shape, bg_mask)
    counts = covers.sum(axis=0)

    out = g.copy()
    order_keys = [obj["bbox"] for obj in objects]
    for r_raw, c_raw in np.argwhere(counts >= 2):
        r = int(r_raw)
        c = int(c_raw)
        coverers = np.flatnonzero(covers[:, r, c])
        winner = max(coverers, key=lambda idx: order_keys[int(idx)])
        out[r, c] = int(objects[int(winner)]["color"])
    return out


def _overlap_start(anchor0: int, anchor1: int, length: int) -> int:
    return random.randint(anchor0 - length + 1, anchor1)


def _propose_next(
    prev_bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    step: tuple[int, int],
    close: bool,
) -> tuple[int, int, int, int]:
    pr0, pc0, pr1, pc1 = prev_bbox
    dr, dc = step
    gap = 1 if close else random.randint(2, 3)

    if dr == 0:
        r0 = _overlap_start(pr0, pr1, height)
        c0 = pc1 + gap + 1 if dc > 0 else pc0 - width - gap
    elif dc == 0:
        r0 = pr1 + gap + 1 if dr > 0 else pr0 - height - gap
        c0 = _overlap_start(pc0, pc1, width)
    else:
        r0 = pr1 + gap + 1 if dr > 0 else pr0 - height - gap
        c0 = pc1 + gap + 1 if dc > 0 else pc0 - width - gap

    return (r0, c0, r0 + height - 1, c0 + width - 1)


def _step_choices(last_step: tuple[int, int] | None) -> list[tuple[int, int]]:
    steps = [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]
    if last_step is None:
        return steps
    reverse = (-last_step[0], -last_step[1])
    return [step for step in steps if step != reverse]


def _shift_rects(
    rects: Iterable[tuple[int, int, int, int]]
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    rects = list(rects)
    min_r = min(r0 for r0, _, _, _ in rects)
    min_c = min(c0 for _, c0, _, _ in rects)
    shifted = [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in rects]
    max_r = max(r1 for _, _, r1, _ in shifted)
    max_c = max(c1 for _, _, _, c1 in shifted)
    return shifted, (max_r + 1, max_c + 1)


def _paint_grid(
    rects: list[tuple[int, int, int, int]], shape: tuple[int, int], colors: list[int]
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _valid_geometry(rects: list[tuple[int, int, int, int]]) -> bool:
    for i, first in enumerate(rects):
        fr0, fc0, fr1, fc1 = first
        for second in rects[i + 1 :]:
            sr0, sc0, sr1, sc1 = second
            row_gap = _interval_gap(fr0, fr1, sr0, sr1)
            col_gap = _interval_gap(fc0, fc1, sc0, sc1)
            if row_gap == 0 and col_gap == 0:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(*N_OBJECTS_RANGE)
    sizes = [
        (
            random.randint(*RECT_H_RANGE),
            random.randint(*RECT_W_RANGE),
        )
        for _ in range(n_objects)
    ]

    close_flags = [random.random() < 0.6 for _ in range(n_objects - 1)]
    if close_flags and not any(close_flags):
        close_flags[random.randrange(len(close_flags))] = True

    rects = [(0, 0, sizes[0][0] - 1, sizes[0][1] - 1)]
    last_step: tuple[int, int] | None = None
    for index in range(1, n_objects):
        step = random.choice(_step_choices(last_step))
        rects.append(_propose_next(rects[-1], sizes[index][0], sizes[index][1], step, close_flags[index - 1]))
        last_step = step

    if not _valid_geometry(rects):
        return None

    rects, (content_h, content_w) = _shift_rects(rects)
    if content_h > MAX_CONTENT_SPAN or content_w > MAX_CONTENT_SPAN:
        return None

    top = random.randint(*PADDING_RANGE)
    left = random.randint(*PADDING_RANGE)
    bottom = random.randint(*PADDING_RANGE) + random.randint(0, EXTRA_SLACK_MAX)
    right = random.randint(*PADDING_RANGE) + random.randint(0, EXTRA_SLACK_MAX)
    rects = [(r0 + top, c0 + left, r1 + top, c1 + left) for r0, c0, r1, c1 in rects]
    grid_shape = (content_h + top + bottom, content_w + left + right)

    colors = random.sample(COLOR_POOL, n_objects)
    grid = _paint_grid(rects, grid_shape, colors)

    try:
        objects = _extract_rectangles(grid)
        covers = _halo_covers(objects, grid.shape, grid == BG)
        counts = covers.sum(axis=0)
        if not np.any(counts >= 2):
            return None
        if np.any(counts > 2):
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
