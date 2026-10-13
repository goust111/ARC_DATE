from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
RECT_COLOR = 1
MARKER_COLOR = 2
INTERIOR_COLOR = 8

INPUT_SIDE_MIN = 7
INPUT_SIDE_MAX = 10
GENERATION_ATTEMPTS = 400
_SIDE_CYCLE = ("left", "right", "top", "bottom")
_side_cycle_index = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _determine_side(marker: tuple[int, int], box: tuple[int, int, int, int]) -> str:
    r, c = marker
    r0, c0, r1, c1 = box
    hits = []
    if r0 <= r <= r1 and c == c0 - 1:
        hits.append("left")
    if r0 <= r <= r1 and c == c1 + 1:
        hits.append("right")
    if c0 <= c <= c1 and r == r0 - 1:
        hits.append("top")
    if c0 <= c <= c1 and r == r1 + 1:
        hits.append("bottom")
    assert len(hits) == 1, "red marker must indicate exactly one rectangle side"
    return hits[0]


def _merged_mask(
    shape: tuple[int, int], box: tuple[int, int, int, int], side: str
) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1

    mask = np.zeros((h, w), dtype=bool)
    if side == "left":
        mask[r0 : r1 + 1, c0 - (rect_w - 1) : c1 + 1] = True
    elif side == "right":
        mask[r0 : r1 + 1, c0 : c1 + rect_w] = True
    elif side == "top":
        mask[r0 - (rect_h - 1) : r1 + 1, c0 : c1 + 1] = True
    else:
        assert side == "bottom"
        mask[r0 : r1 + rect_h, c0 : c1 + 1] = True
    return mask


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
            boundary[rr, cc] = True
            continue
        if not (mask[rr - 1, cc] and mask[rr + 1, cc] and mask[rr, cc - 1] and mask[rr, cc + 1]):
            boundary[rr, cc] = True
    return boundary


def _build_base_tile(n: int, box: tuple[int, int, int, int], side: str) -> np.ndarray:
    merged = _merged_mask((n, n), box, side)
    border = _boundary_mask(merged)
    interior = merged & ~border

    tile = np.zeros((n, n), dtype=int)
    tile[border] = RECT_COLOR
    tile[interior] = INTERIOR_COLOR
    return tile


def _pinwheel(tile: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [tile, np.rot90(tile, -1)],
            [np.rot90(tile, 1), np.rot90(tile, 2)],
        ]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the marked rectangle across its indicated side, recolor the merged interior, and assemble a 2x2 rotation pinwheel.

    Input structure:
    - The input is a square black grid containing one solid blue rectangle and one red marker pixel.
    - The blue rectangle is axis-aligned, at least 3x3, and does not touch the canvas border.
    - The red marker is 4-neighbor adjacent to exactly one side of the rectangle.

    Rule:
    - Detect which side of the rectangle the red marker touches.
    - Mirror the whole rectangle across that touched side so the mirrored copy shares the side with the original and covers the red marker.
    - Keep the merged shape's outer one-pixel border blue, recolor all strict interior pixels teal, then place the processed square tile into a 2x2 output mosaic with rotations: original, 90 clockwise, 90 counterclockwise, and 180 degrees.

    Key insight:
    - The red pixel is only a directional side marker: it disappears after the mirror step, but it determines the enlarged rectangle whose border and interior are then repeated in four rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, m = g.shape
    assert n == m, "expected a square grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, RECT_COLOR, MARKER_COLOR}), "unexpected colors"

    marker_coords = np.argwhere(g == MARKER_COLOR)
    assert len(marker_coords) == 1, "expected exactly one red marker"
    marker = tuple(map(int, marker_coords[0]))

    rect_mask = g == RECT_COLOR
    assert rect_mask.any(), "expected a blue rectangle"
    r0, c0, r1, c1 = _bbox(rect_mask)
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    assert rect_h >= 3 and rect_w >= 3 and rect_h != rect_w, "expected a non-square rectangle with interior"
    assert rect_mask.sum() == rect_h * rect_w, "blue object must be a solid rectangle"
    assert r0 > 0 and c0 > 0 and r1 < n - 1 and c1 < n - 1, "rectangle must stay away from the border"

    side = _determine_side(marker, (r0, c0, r1, c1))
    base_tile = _build_base_tile(n, (r0, c0, r1, c1), side)
    return _pinwheel(base_tile)


def _side_for_generation() -> str:
    global _side_cycle_index
    side = _SIDE_CYCLE[_side_cycle_index % len(_SIDE_CYCLE)]
    _side_cycle_index += 1
    return side


def _sample_rect(side: str, n: int, rng: random.Random) -> tuple[int, int, int, int] | None:
    if side in {"left", "right"}:
        max_w = n // 2
        if max_w < 3:
            return None
        widths = list(range(3, max_w + 1))
        rng.shuffle(widths)
        for rect_w in widths:
            heights = [h for h in range(3, n - 1) if h != rect_w]
            if not heights:
                continue
            rect_h = rng.choice(heights)
            r0 = rng.randint(1, n - rect_h - 1)
            if side == "left":
                c0_min = rect_w - 1
                c0_max = n - rect_w - 1
            else:
                c0_min = 1
                c0_max = n - 2 * rect_w + 1
            if c0_min > c0_max:
                continue
            c0 = rng.randint(c0_min, c0_max)
            return r0, c0, rect_h, rect_w

    max_h = n // 2
    if max_h < 3:
        return None
    heights = list(range(3, max_h + 1))
    rng.shuffle(heights)
    for rect_h in heights:
        widths = [w for w in range(3, n - 1) if w != rect_h]
        if not widths:
            continue
        rect_w = rng.choice(widths)
        c0 = rng.randint(1, n - rect_w - 1)
        if side == "top":
            r0_min = rect_h - 1
            r0_max = n - rect_h - 1
        else:
            r0_min = 1
            r0_max = n - 2 * rect_h + 1
        if r0_min > r0_max:
            continue
        r0 = rng.randint(r0_min, r0_max)
        return r0, c0, rect_h, rect_w
    return None


def _place_marker(
    grid: np.ndarray, side: str, r0: int, c0: int, rect_h: int, rect_w: int, rng: random.Random
) -> None:
    if side == "left":
        marker_r = rng.randint(r0, r0 + rect_h - 1)
        grid[marker_r, c0 - 1] = MARKER_COLOR
    elif side == "right":
        marker_r = rng.randint(r0, r0 + rect_h - 1)
        grid[marker_r, c0 + rect_w] = MARKER_COLOR
    elif side == "top":
        marker_c = rng.randint(c0, c0 + rect_w - 1)
        grid[r0 - 1, marker_c] = MARKER_COLOR
    else:
        marker_c = rng.randint(c0, c0 + rect_w - 1)
        grid[r0 + rect_h, marker_c] = MARKER_COLOR


def _attempt_generate(side: str, rng: random.Random) -> np.ndarray | None:
    n = rng.randint(INPUT_SIDE_MIN, INPUT_SIDE_MAX)
    rect = _sample_rect(side, n, rng)
    if rect is None:
        return None
    r0, c0, rect_h, rect_w = rect

    grid = np.zeros((n, n), dtype=int)
    grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = RECT_COLOR
    _place_marker(grid, side, r0, c0, rect_h, rect_w, rng)

    base_tile = _build_base_tile(n, (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1), side)
    if np.array_equal(base_tile, np.rot90(base_tile, -1)):
        return None

    try:
        expected = _pinwheel(base_tile)
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    side = _side_for_generation()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate(side, random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
