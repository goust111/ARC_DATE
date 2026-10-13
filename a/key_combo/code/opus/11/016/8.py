from __future__ import annotations

import numpy as np


RED = 2
GREEN = 3
TEAL = 8
MAROON = 9

RECT_SIZE = 3
MAX_GENERATION_ATTEMPTS = 200
ODD_SIZES = (9, 11, 13, 15)
SIDES = ("top", "bottom", "left", "right")
VERTICAL_SIDES = ("top", "bottom")
HORIZONTAL_SIDES = ("left", "right")


def _diagonal_background(shape: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices(shape)
    return np.where((rr + cc) % 2 == 0, TEAL, MAROON).astype(int)


def _rectangle_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    red = np.argwhere(grid == RED)
    assert red.shape[0] == RECT_SIZE * RECT_SIZE, "expected one 3x3 red rectangle"
    r0, c0 = red.min(axis=0)
    r1, c1 = red.max(axis=0)
    assert (int(r1) - int(r0) + 1, int(c1) - int(c0) + 1) == (RECT_SIZE, RECT_SIZE)
    assert np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == RED), "expected a solid rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _marked_projections(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[set[int], set[int], set[int], set[int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    markers = [tuple(map(int, coord)) for coord in np.argwhere(grid == GREEN)]
    assert 2 <= len(markers) <= 3, "expected 2-3 green control pixels"

    marked_top: set[int] = set()
    marked_bottom: set[int] = set()
    marked_left: set[int] = set()
    marked_right: set[int] = set()

    for r, c in markers:
        side_hits = 0
        if r < r0 and c0 <= c <= c1:
            marked_top.add(c)
            side_hits += 1
        if r > r1 and c0 <= c <= c1:
            marked_bottom.add(c)
            side_hits += 1
        if c < c0 and r0 <= r <= r1:
            marked_left.add(r)
            side_hits += 1
        if c > c1 and r0 <= r <= r1:
            marked_right.add(r)
            side_hits += 1
        assert side_hits == 1, "each marker must lie in exactly one rectangle side strip"

    return marked_top, marked_bottom, marked_left, marked_right, markers


def _paint_stripes(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    marked_top: set[int],
    marked_bottom: set[int],
    marked_left: set[int],
    marked_right: set[int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = grid.copy()

    for c in range(c0, c1 + 1):
        if c not in marked_top:
            out[:r0, c] = RED
        if c not in marked_bottom:
            out[r1 + 1 :, c] = RED

    for r in range(r0, r1 + 1):
        if r not in marked_left:
            out[r, :c0] = RED
        if r not in marked_right:
            out[r, c1 + 1 :] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the central red rectangle into side stripes while keeping marker-excluded projections as the visible diagonal background.

    Input structure:
    - The whole grid carries a period-2 diagonal background using teal and maroon.
    - A solid red 3x3 rectangle sits at the center area of the grid.
    - There are 2-3 green singleton control pixels, each in one of the four side strips aligned with the rectangle and never in a diagonal corner quadrant.

    Rule:
    - Preserve the red rectangle and every green control pixel.
    - For the three rectangle columns, extend red stripes upward and downward unless that side/column contains a green control pixel.
    - For the three rectangle rows, extend red stripes leftward and rightward unless that side/row contains a green control pixel.
    - Cells on excluded projections are not blanked; they remain whatever diagonal background color is already visible in the input.

    Key insight:
    - Each green pixel disables exactly one projected stripe on its own side, so the output is the overlay of red extensions on top of an already-present diagonal background rather than a simple fill-vs-empty pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bbox = _rectangle_bbox(g)
    marked_top, marked_bottom, marked_left, marked_right, markers = _marked_projections(g, bbox)
    out = _paint_stripes(g, bbox, marked_top, marked_bottom, marked_left, marked_right)
    for r, c in markers:
        out[r, c] = GREEN
    return out


def _side_cells_for_index(
    side: str,
    idx: int,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r, c0 + idx) for r in range(r0)]
    if side == "bottom":
        return [(r, c0 + idx) for r in range(r1 + 1, h)]
    if side == "left":
        return [(r0 + idx, c) for c in range(c0)]
    if side == "right":
        return [(r0 + idx, c) for c in range(c1 + 1, w)]
    raise AssertionError(f"unknown side: {side}")


def _choose_marker_specs(rng: np.random.Generator) -> list[tuple[str, int]]:
    count = 2 if rng.random() < 0.5 else 3

    if count == 2:
        family = int(rng.integers(4))
        if family == 0:
            side = str(rng.choice(SIDES))
            indices = rng.choice(RECT_SIZE, size=2, replace=False)
            return [(side, int(i)) for i in indices]
        if family == 1:
            return [(str(rng.choice(VERTICAL_SIDES)), int(rng.integers(RECT_SIZE))), (str(rng.choice(HORIZONTAL_SIDES)), int(rng.integers(RECT_SIZE)))]
        if family == 2:
            idx_a, idx_b = map(int, rng.integers(RECT_SIZE, size=2))
            return [("top", idx_a), ("bottom", idx_b)]
        idx_a, idx_b = map(int, rng.integers(RECT_SIZE, size=2))
        return [("left", idx_a), ("right", idx_b)]

    family = int(rng.integers(4))
    if family == 0:
        sides = list(map(str, rng.choice(SIDES, size=3, replace=False)))
        return [(side, int(rng.integers(RECT_SIZE))) for side in sides]
    if family == 1:
        side = str(rng.choice(VERTICAL_SIDES))
        side_indices = list(map(int, rng.choice(RECT_SIZE, size=2, replace=False)))
        other_side = str(rng.choice(HORIZONTAL_SIDES))
        return [(side, side_indices[0]), (side, side_indices[1]), (other_side, int(rng.integers(RECT_SIZE)))]
    if family == 2:
        side = str(rng.choice(HORIZONTAL_SIDES))
        side_indices = list(map(int, rng.choice(RECT_SIZE, size=2, replace=False)))
        other_side = str(rng.choice(VERTICAL_SIDES))
        return [(side, side_indices[0]), (side, side_indices[1]), (other_side, int(rng.integers(RECT_SIZE)))]
    distinct = [("top", int(rng.integers(RECT_SIZE))), ("bottom", int(rng.integers(RECT_SIZE)))]
    other = str(rng.choice(HORIZONTAL_SIDES))
    distinct.append((other, int(rng.integers(RECT_SIZE))))
    return distinct


def _place_markers(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    specs: list[tuple[str, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for side, idx in specs:
        candidates = _side_cells_for_index(side, idx, bbox, grid.shape)
        cell = candidates[int(rng.integers(len(candidates)))]
        markers.append(cell)
        grid[cell] = GREEN
    return markers


def _expected_output(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    specs: list[tuple[str, int]],
) -> np.ndarray:
    marked_top = {c0 + idx for side, idx in specs if side == "top" for c0 in [bbox[1]]}
    marked_bottom = {c0 + idx for side, idx in specs if side == "bottom" for c0 in [bbox[1]]}
    marked_left = {r0 + idx for side, idx in specs if side == "left" for r0 in [bbox[0]]}
    marked_right = {r0 + idx for side, idx in specs if side == "right" for r0 in [bbox[0]]}
    out = _paint_stripes(grid, bbox, marked_top, marked_bottom, marked_left, marked_right)
    out[grid == GREEN] = GREEN
    return out


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.choice(ODD_SIZES))
    w = int(rng.choice(ODD_SIZES))
    grid = _diagonal_background((h, w))

    r0 = h // 2 - 1
    c0 = w // 2 - 1
    bbox = (r0, c0, r0 + RECT_SIZE - 1, c0 + RECT_SIZE - 1)
    grid[r0 : r0 + RECT_SIZE, c0 : c0 + RECT_SIZE] = RED

    specs = _choose_marker_specs(rng)
    _place_markers(grid, bbox, specs, rng)

    expected = _expected_output(grid, bbox, specs)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(actual, grid):
        return None

    side_kinds = {side for side, _ in specs}
    if len(side_kinds) == 1 and len(specs) == 2 and len({idx for _, idx in specs}) == 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input for puzzle 8")
