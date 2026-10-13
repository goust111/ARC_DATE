from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
TEMP_SIZE = 30
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160
RECT_COUNT_VALUES = (1, 2, 3, 4)
RECT_COUNT_WEIGHTS = (1, 3, 3, 2)
HEIGHT_RANGE = (4, 8)
WIDTH_RANGE = (4, 8)
MARGIN_CAP = 3

DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast outward colored rays from grey framed rectangles using the count of interior markers as the ray length.

    Input structure:
    - The background is black.
    - Several one-pixel-thick grey rectangular frames enclose black interiors with zero or more colored marker pixels.
    - Each frame has one outward arrow on exactly one side: the border stays intact and one grey tip pixel protrudes one cell outside that side.
    - All non-black, non-grey marker pixels inside the same rectangle, when present, share one color.

    Rule:
    - Detect every complete grey rectangle and identify its unique arrow direction from the protruding grey tip.
    - Count the marker pixels N inside the rectangle and read their shared color.
    - If N is positive, paint the next N cells immediately beyond the arrow tip along the arrow direction with that marker color.
    - Keep the original frames, arrows, and marker pixels unchanged.

    Key insight:
    - The grey geometry only encodes where the ray points; the enclosed colored pixels determine both how far the ray extends and which color it uses.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    rectangles = _find_rectangles(g)
    assert rectangles, "expected at least one grey rectangle"

    for r0, c0, r1, c1 in rectangles:
        tip_r, tip_c, dr, dc = _find_arrow(g, r0, c0, r1, c1)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (inner != BLACK) & (inner != GREY)
        n_markers = int(marker_mask.sum())
        if n_markers == 0:
            continue

        colors = np.unique(inner[marker_mask])
        assert colors.size == 1, "markers inside one rectangle must share a single color"
        color = int(colors[0])

        rr = tip_r + dr
        cc = tip_c + dc
        for _ in range(n_markers):
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "ray left the grid"
            assert int(g[rr, cc]) == BLACK, "ray corridor must start on background"
            out[rr, cc] = color
            rr += dr
            cc += dc

    return out


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    rectangles: list[tuple[int, int, int, int]] = []
    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                if int(grid[r0, c0]) != GREY or int(grid[r1, c0]) != GREY:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(grid[r0, c1]) != GREY or int(grid[r1, c1]) != GREY:
                        continue
                    if not np.all(grid[r0, c0 : c1 + 1] == GREY):
                        continue
                    if not np.all(grid[r1, c0 : c1 + 1] == GREY):
                        continue
                    if not np.all(grid[r0 : r1 + 1, c0] == GREY):
                        continue
                    if not np.all(grid[r0 : r1 + 1, c1] == GREY):
                        continue
                    rectangles.append((r0, c0, r1, c1))
    rectangles.sort()
    return rectangles


def _find_arrow(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []

    if r0 > 0:
        for c in range(c0 + 1, c1):
            if int(grid[r0 - 1, c]) == GREY and _not_grey(grid, r0 - 2, c) and _not_grey(grid, r0 - 1, c - 1) and _not_grey(grid, r0 - 1, c + 1):
                candidates.append((r0 - 1, c, -1, 0))

    if r1 + 1 < grid.shape[0]:
        for c in range(c0 + 1, c1):
            if int(grid[r1 + 1, c]) == GREY and _not_grey(grid, r1 + 2, c) and _not_grey(grid, r1 + 1, c - 1) and _not_grey(grid, r1 + 1, c + 1):
                candidates.append((r1 + 1, c, 1, 0))

    if c0 > 0:
        for r in range(r0 + 1, r1):
            if int(grid[r, c0 - 1]) == GREY and _not_grey(grid, r, c0 - 2) and _not_grey(grid, r - 1, c0 - 1) and _not_grey(grid, r + 1, c0 - 1):
                candidates.append((r, c0 - 1, 0, -1))

    if c1 + 1 < grid.shape[1]:
        for r in range(r0 + 1, r1):
            if int(grid[r, c1 + 1]) == GREY and _not_grey(grid, r, c1 + 2) and _not_grey(grid, r - 1, c1 + 1) and _not_grey(grid, r + 1, c1 + 1):
                candidates.append((r, c1 + 1, 0, 1))

    assert len(candidates) == 1, f"expected one arrow for rectangle {(r0, c0, r1, c1)}"
    return candidates[0]


def _not_grey(grid: np.ndarray, r: int, c: int) -> bool:
    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        return int(grid[r, c]) != GREY
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray ARC input")


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    canvas_in = np.full((TEMP_SIZE, TEMP_SIZE), BLACK, dtype=int)
    canvas_out = canvas_in.copy()
    blocked = np.zeros((TEMP_SIZE, TEMP_SIZE), dtype=bool)
    future_ray = np.zeros((TEMP_SIZE, TEMP_SIZE), dtype=bool)

    rect_count = int(rng.choices(RECT_COUNT_VALUES, weights=RECT_COUNT_WEIGHTS, k=1)[0])
    zero_flags = [rng.random() < 0.5 for _ in range(rect_count)]
    if all(zero_flags):
        zero_flags[rng.randrange(rect_count)] = False
    if rect_count > 1 and not any(zero_flags) and rng.random() < 0.6:
        zero_flags[rng.randrange(rect_count)] = True

    specs = []
    color_pool = list(rng.sample(MARKER_COLORS, k=len(MARKER_COLORS)))
    color_index = 0
    for want_zero in zero_flags:
        direction = rng.choice(tuple(DIR_TO_DELTA))
        color = None
        if not want_zero:
            color = int(color_pool[color_index % len(color_pool)])
            color_index += 1
        specs.append({"direction": direction, "want_zero": want_zero, "color": color})

    specs.sort(key=lambda spec: spec["want_zero"])

    placed_any_positive = False
    for spec in specs:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            placement = _sample_placement(
                rng=rng,
                blocked=blocked,
                want_zero=bool(spec["want_zero"]),
                direction=str(spec["direction"]),
            )
            if placement is None:
                continue

            _stamp_rectangle(
                canvas_in,
                canvas_out,
                blocked,
                future_ray,
                placement,
                None if spec["color"] is None else int(spec["color"]),
                rng,
            )
            placed = True
            placed_any_positive = placed_any_positive or placement["n_markers"] > 0
            break
        if not placed:
            return None

    if not placed_any_positive:
        return None

    used = np.argwhere((canvas_in != BLACK) | future_ray)
    if used.size == 0:
        return None
    min_r, min_c = map(int, used.min(axis=0))
    max_r, max_c = map(int, used.max(axis=0))

    top_margin = rng.randint(0, min(MARGIN_CAP, min_r))
    left_margin = rng.randint(0, min(MARGIN_CAP, min_c))
    bottom_margin = rng.randint(0, min(MARGIN_CAP, TEMP_SIZE - 1 - max_r))
    right_margin = rng.randint(0, min(MARGIN_CAP, TEMP_SIZE - 1 - max_c))

    crop_r0 = min_r - top_margin
    crop_c0 = min_c - left_margin
    crop_r1 = max_r + bottom_margin
    crop_c1 = max_c + right_margin

    grid = canvas_in[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
    expected = canvas_out[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]

    if grid.ndim != 2 or max(grid.shape) > 30:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def _sample_placement(
    rng: random.Random,
    blocked: np.ndarray,
    want_zero: bool,
    direction: str,
) -> dict[str, object] | None:
    h = rng.randint(HEIGHT_RANGE[0], HEIGHT_RANGE[1])
    w = rng.randint(WIDTH_RANGE[0], WIDTH_RANGE[1])

    if h >= TEMP_SIZE - 2 or w >= TEMP_SIZE - 2:
        return None

    r0 = rng.randint(1, TEMP_SIZE - h - 1)
    c0 = rng.randint(1, TEMP_SIZE - w - 1)
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    if direction == "up":
        anchor = rng.randint(c0 + 1, c1 - 1)
        tip = (r0 - 1, anchor)
    elif direction == "down":
        anchor = rng.randint(c0 + 1, c1 - 1)
        tip = (r1 + 1, anchor)
    elif direction == "left":
        anchor = rng.randint(r0 + 1, r1 - 1)
        tip = (anchor, c0 - 1)
    else:
        anchor = rng.randint(r0 + 1, r1 - 1)
        tip = (anchor, c1 + 1)

    tip_r, tip_c = tip
    comp_r0 = min(r0, tip_r)
    comp_c0 = min(c0, tip_c)
    comp_r1 = max(r1, tip_r)
    comp_c1 = max(c1, tip_c)

    if comp_r0 < 1 or comp_c0 < 1 or comp_r1 > TEMP_SIZE - 2 or comp_c1 > TEMP_SIZE - 2:
        return None
    if blocked[comp_r0 - 1 : comp_r1 + 2, comp_c0 - 1 : comp_c1 + 2].any():
        return None

    dr, dc = DIR_TO_DELTA[direction]
    corridor = []
    rr, cc = tip_r + dr, tip_c + dc
    while 0 <= rr < TEMP_SIZE and 0 <= cc < TEMP_SIZE and not blocked[rr, cc]:
        corridor.append((rr, cc))
        rr += dr
        cc += dc

    inner_cells = [(rr0, cc0) for rr0 in range(r0 + 1, r1) for cc0 in range(c0 + 1, c1)]
    if not inner_cells:
        return None

    if want_zero:
        n_markers = 0
    else:
        cap = min(len(inner_cells), len(corridor))
        if cap <= 0:
            return None
        n_markers = rng.randint(1, cap)

    return {
        "bbox": (r0, c0, r1, c1),
        "tip": tip,
        "direction": direction,
        "n_markers": n_markers,
        "corridor": corridor,
        "inner_cells": inner_cells,
    }


def _stamp_rectangle(
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    blocked: np.ndarray,
    future_ray: np.ndarray,
    placement: dict[str, object],
    color: int | None,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = placement["bbox"]  # type: ignore[assignment]
    tip_r, tip_c = placement["tip"]  # type: ignore[assignment]
    n_markers = int(placement["n_markers"])
    inner_cells: list[tuple[int, int]] = list(placement["inner_cells"])  # type: ignore[assignment]
    corridor: list[tuple[int, int]] = list(placement["corridor"])  # type: ignore[assignment]

    for r, c in _perimeter_cells(r0, c0, r1, c1):
        canvas_in[r, c] = GREY
        canvas_out[r, c] = GREY
    canvas_in[tip_r, tip_c] = GREY
    canvas_out[tip_r, tip_c] = GREY

    if n_markers > 0:
        assert color is not None
        marker_cells = rng.sample(inner_cells, k=n_markers)
        for r, c in marker_cells:
            canvas_in[r, c] = color
            canvas_out[r, c] = color
        for r, c in corridor[:n_markers]:
            canvas_out[r, c] = color
            future_ray[r, c] = True

    comp_r0 = min(r0, tip_r)
    comp_c0 = min(c0, tip_c)
    comp_r1 = max(r1, tip_r)
    comp_c1 = max(c1, tip_c)
    blocked[comp_r0 - 1 : comp_r1 + 2, comp_c0 - 1 : comp_c1 + 2] = True
    for r, c in corridor[:n_markers]:
        blocked[r, c] = True


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> Iterable[tuple[int, int]]:
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)
