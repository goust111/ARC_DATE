from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BG = 0
ARC_MAX = 30
COLOR_POOL = tuple(range(1, 10))

OBJECT_COUNT_RANGE = (2, 4)
INNER_SIDE_RANGE = (3, 7)
SIZE_STEP_RANGE = (2, 3)
MAX_SCENE_SIDE = 24
MAX_MARGIN = 3

GENERATION_ATTEMPTS = 400
SIZE_ATTEMPTS = 300


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    h: int
    w: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    @property
    def area(self) -> int:
        return self.h * self.w


def _visible_rectangles(grid: np.ndarray) -> list[Rect]:
    rects: list[Rect] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, f"missing visible pixels for color {color}"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        rects.append(Rect(color=color, r0=r0, c0=c0, h=r1 - r0 + 1, w=c1 - c0 + 1))
    rects.sort(key=lambda rect: (-rect.area, rect.r0, rect.c0, rect.h, rect.w))
    return rects


def _pack_rectangles(rects: list[Rect]) -> np.ndarray:
    assert rects
    out_h = max(rect.h for rect in rects)
    out_w = sum(rect.w for rect in rects) + len(rects) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for rect in rects:
        out[out_h - rect.h :, c : c + rect.w] = rect.color
        c += rect.w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each overlapped color layer as a full rectangle and repack the layers by area.

    Input structure:
    - Black background.
    - Each foreground color belongs to one solid axis-aligned rectangle.
    - Rectangles may overlap, so lower layers can be partially hidden.
    - The visible pixels of every color still span that color's original rectangle bbox.

    Rule:
    - For each foreground color, take the bbox of all visible pixels and treat it as the full rectangle.
    - Compute each rectangle's area from that reconstructed width and height.
    - Sort the rectangles from largest area to smallest area.
    - Draw the full rectangles left-to-right on a black canvas, bottom-aligned, with one black column between neighbors.

    Key insight:
    - Overlap only removes interior pixels; the visible extent of each color still reveals the original rectangle size, so positions can be discarded after bbox recovery.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rects = _visible_rectangles(g)
    assert rects, "expected at least one foreground rectangle"
    return _pack_rectangles(rects)


def _sample_nested_sizes(rng: np.random.Generator, count: int) -> list[tuple[int, int]] | None:
    for _ in range(SIZE_ATTEMPTS):
        inner_h = int(rng.integers(INNER_SIDE_RANGE[0], INNER_SIDE_RANGE[1] + 1))
        inner_w = int(rng.integers(INNER_SIDE_RANGE[0], INNER_SIDE_RANGE[1] + 1))
        hs = [inner_h]
        ws = [inner_w]
        for _ in range(count - 1):
            hs.append(hs[-1] + int(rng.integers(SIZE_STEP_RANGE[0], SIZE_STEP_RANGE[1] + 1)))
            ws.append(ws[-1] + int(rng.integers(SIZE_STEP_RANGE[0], SIZE_STEP_RANGE[1] + 1)))

        hs = hs[::-1]
        ws = ws[::-1]
        if hs[0] > MAX_SCENE_SIDE or ws[0] > MAX_SCENE_SIDE:
            continue
        if sum(ws) + count - 1 > ARC_MAX:
            continue

        areas = {h * w for h, w in zip(hs, ws)}
        if len(areas) != count:
            continue
        return list(zip(hs, ws))
    return None


def _place_nested_stack(
    sizes: list[tuple[int, int]], colors: list[int], rng: np.random.Generator
) -> list[Rect]:
    rects: list[Rect] = [Rect(color=colors[0], r0=0, c0=0, h=sizes[0][0], w=sizes[0][1])]
    for idx, (h, w) in enumerate(sizes[1:], start=1):
        parent = rects[-1]
        row_slack = parent.h - h - 2
        col_slack = parent.w - w - 2
        assert row_slack >= 0 and col_slack >= 0
        r0 = parent.r0 + 1 + int(rng.integers(0, row_slack + 1))
        c0 = parent.c0 + 1 + int(rng.integers(0, col_slack + 1))
        rects.append(Rect(color=colors[idx], r0=r0, c0=c0, h=h, w=w))
    return rects


def _render_scene(rects: list[Rect], rng: np.random.Generator) -> np.ndarray:
    min_r = min(rect.r0 for rect in rects)
    min_c = min(rect.c0 for rect in rects)
    max_r = max(rect.r1 for rect in rects)
    max_c = max(rect.c1 for rect in rects)

    scene_h = max_r - min_r + 1
    scene_w = max_c - min_c + 1
    extra_h = ARC_MAX - scene_h
    extra_w = ARC_MAX - scene_w
    assert extra_h >= 0 and extra_w >= 0

    top = int(rng.integers(0, min(MAX_MARGIN, extra_h) + 1))
    left = int(rng.integers(0, min(MAX_MARGIN, extra_w) + 1))
    bottom = int(rng.integers(0, min(MAX_MARGIN, extra_h - top) + 1))
    right = int(rng.integers(0, min(MAX_MARGIN, extra_w - left) + 1))

    height = scene_h + top + bottom
    width = scene_w + left + right
    grid = np.full((height, width), BG, dtype=int)

    dr = top - min_r
    dc = left - min_c
    for rect in rects:
        grid[rect.r0 + dr : rect.r1 + dr + 1, rect.c0 + dc : rect.c1 + dc + 1] = rect.color
    return grid


def _apply_random_symmetry(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    variant = int(rng.integers(0, 4))
    if variant == 0:
        return grid
    if variant == 1:
        return np.flipud(grid)
    if variant == 2:
        return np.fliplr(grid)
    return np.flipud(np.fliplr(grid))


def _validate_scene(grid: np.ndarray, rects: list[Rect]) -> bool:
    visible = _visible_rectangles(grid)
    if len(visible) != len(rects):
        return False

    expected = sorted((rect.h, rect.w, rect.area) for rect in rects)
    actual = sorted((rect.h, rect.w, rect.area) for rect in visible)
    if expected != actual:
        return False

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > ARC_MAX:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    sizes = _sample_nested_sizes(rng, count)
    if sizes is None:
        return None

    colors = [int(c) for c in rng.choice(COLOR_POOL, size=count, replace=False)]
    rects = _place_nested_stack(sizes, colors, rng)
    grid = _render_scene(rects, rng)
    grid = _apply_random_symmetry(grid, rng).copy()

    if not _validate_scene(grid, rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapped-rectangle scene")
