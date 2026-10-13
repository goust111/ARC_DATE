from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
MARKER_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_container(grid: np.ndarray, bg: int) -> dict:
    grey_coords = [tuple(map(int, pt)) for pt in np.argwhere(grid == GREY)]
    assert len(grey_coords) == 2, "expected exactly two grey corner markers"

    (gr0, gc0), (gr1, gc1) = sorted(grey_coords)
    if gr0 == gr1:
        orientation = "horizontal"
        floor_cols = tuple(sorted((gc0, gc1)))
        assert floor_cols[0] < floor_cols[1], "grey markers must span a horizontal edge"
    else:
        assert gc0 == gc1, "grey markers must share one row or one column"
        orientation = "vertical"
        floor_rows = tuple(sorted((gr0, gr1)))
        assert floor_rows[0] < floor_rows[1], "grey markers must span a vertical edge"

    candidates: list[tuple[int, list[tuple[int, int]], str]] = []
    for color, count in Counter(map(int, grid.flatten())).items():
        if color in (bg, GREY) or count != 2:
            continue
        coords = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
        (r0, c0), (r1, c1) = sorted(coords)
        if orientation == "horizontal":
            if r0 != r1:
                continue
            if tuple(sorted((c0, c1))) != floor_cols:
                continue
            side = "top" if gr0 < r0 else "bottom"
        else:
            if c0 != c1:
                continue
            if tuple(sorted((r0, r1))) != floor_rows:
                continue
            side = "left" if gc0 < c0 else "right"
        candidates.append((int(color), coords, side))

    assert len(candidates) == 1, "expected exactly one opposite edge marker pair"
    marker_color, other_coords, floor_side = candidates[0]

    all_rows = [r for r, _ in grey_coords + other_coords]
    all_cols = [c for _, c in grey_coords + other_coords]
    r0, r1 = min(all_rows), max(all_rows)
    c0, c1 = min(all_cols), max(all_cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "container must have a non-empty interior"

    return {
        "bbox": (r0, c0, r1, c1),
        "floor_side": floor_side,
        "marker_color": marker_color,
        "grey_coords": grey_coords,
        "other_coords": other_coords,
        "marker_coords": set(grey_coords) | set(other_coords),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the invisible rectangle from four corner markers and drop interior grains onto the grey-marked edge.

    Input structure:
    - The background is black.
    - Exactly four single-pixel corner markers define an axis-aligned rectangle: two grey markers share one edge and two markers of another color share the opposite edge.
    - Additional single-pixel colored grains, in colors different from the marker colors, are scattered both inside and outside the implied rectangle.

    Rule:
    - Reconstruct the hidden rectangle from the four corner markers and keep those markers fixed.
    - The grey pair indicates the floor edge, so every grain strictly inside the rectangle falls along the perpendicular lines toward that edge and stacks without crossing or changing order within each line.
    - Erase every grain outside the rectangle, and do not draw any rectangle edges in the output.

    Key insight:
    - The rectangle itself is invisible; the grey corner pair simultaneously identifies one side of the container and fixes the gravity direction for independent per-line stacking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    spec = _find_container(g, bg)
    r0, c0, r1, c1 = spec["bbox"]
    floor_side = spec["floor_side"]
    marker_coords = spec["marker_coords"]

    out = np.full_like(g, bg)
    for r, c in spec["grey_coords"] + spec["other_coords"]:
        out[r, c] = int(g[r, c])

    if floor_side in ("top", "bottom"):
        for c in range(c0 + 1, c1):
            grains = [
                int(g[r, c])
                for r in range(r0 + 1, r1)
                if (r, c) not in marker_coords and int(g[r, c]) != bg
            ]
            if floor_side == "top":
                dst_rows = range(r0 + 1, r0 + 1 + len(grains))
            else:
                dst_rows = range(r1 - len(grains), r1)
            for rr, color in zip(dst_rows, grains):
                out[rr, c] = color
    else:
        for r in range(r0 + 1, r1):
            grains = [
                int(g[r, c])
                for c in range(c0 + 1, c1)
                if (r, c) not in marker_coords and int(g[r, c]) != bg
            ]
            if floor_side == "left":
                dst_cols = range(c0 + 1, c0 + 1 + len(grains))
            else:
                dst_cols = range(c1 - len(grains), c1)
            for cc, color in zip(dst_cols, grains):
                out[r, cc] = color

    return out


def _marker_layout(side: str, r0: int, c0: int, r1: int, c1: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if side == "top":
        return [(r0, c0), (r0, c1)], [(r1, c0), (r1, c1)]
    if side == "bottom":
        return [(r1, c0), (r1, c1)], [(r0, c0), (r0, c1)]
    if side == "left":
        return [(r0, c0), (r1, c0)], [(r0, c1), (r1, c1)]
    if side == "right":
        return [(r0, c1), (r1, c1)], [(r0, c0), (r1, c0)]
    raise ValueError(f"unknown side: {side}")


def _sample_inside_positions(
    side: str,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    interior_rows = list(range(r0 + 1, r1))
    interior_cols = list(range(c0 + 1, c1))

    if side in ("top", "bottom"):
        line_count = random.randint(2, min(5, len(interior_cols)))
        cols = random.sample(interior_cols, line_count)
        forced_multi = random.choice(cols)
        for col in cols:
            low = 2 if col == forced_multi and len(interior_rows) >= 2 else 1
            high = min(len(interior_rows), 4)
            count = random.randint(low, high)
            rows = sorted(random.sample(interior_rows, count))
            positions.extend((row, col) for row in rows)
    else:
        line_count = random.randint(2, min(5, len(interior_rows)))
        rows = random.sample(interior_rows, line_count)
        forced_multi = random.choice(rows)
        for row in rows:
            low = 2 if row == forced_multi and len(interior_cols) >= 2 else 1
            high = min(len(interior_cols), 4)
            count = random.randint(low, high)
            cols = sorted(random.sample(interior_cols, count))
            positions.extend((row, col) for col in cols)

    return positions


def _sample_outside_positions(
    height: int,
    width: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in blocked and not (r0 <= r <= r1 and c0 <= c <= c1)
    ]
    assert candidates, "expected some space outside the container"
    count = random.randint(2, min(7, len(candidates)))
    return random.sample(candidates, count)


def _build_attempt() -> np.ndarray | None:
    height = random.randint(12, 30)
    width = random.randint(12, 30)
    max_rect_h = min(12, height - 2)
    max_rect_w = min(12, width - 2)
    if max_rect_h < 5 or max_rect_w < 5:
        return None

    rect_h = random.randint(5, max_rect_h)
    rect_w = random.randint(5, max_rect_w)
    r0 = random.randint(1, height - rect_h - 1)
    c0 = random.randint(1, width - rect_w - 1)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    side = random.choice(["top", "bottom", "left", "right"])
    other_marker_color = random.choice(MARKER_COLOR_POOL)
    grain_palette = [color for color in MARKER_COLOR_POOL if color != other_marker_color]
    random.shuffle(grain_palette)
    grain_palette = grain_palette[: random.randint(2, min(4, len(grain_palette)))]

    grid = np.full((height, width), BG, dtype=int)
    grey_coords, other_coords = _marker_layout(side, r0, c0, r1, c1)
    blocked = set(grey_coords) | set(other_coords)
    for r, c in grey_coords:
        grid[r, c] = GREY
    for r, c in other_coords:
        grid[r, c] = other_marker_color

    inside_positions = _sample_inside_positions(side, r0, c0, r1, c1)
    for r, c in inside_positions:
        grid[r, c] = random.choice(grain_palette)

    outside_positions = _sample_outside_positions(height, width, r0, c0, r1, c1, blocked | set(inside_positions))
    for r, c in outside_positions:
        grid[r, c] = random.choice(grain_palette)

    try:
        out = main(grid)
        spec = _find_container(grid, BG)
    except AssertionError:
        return None

    if spec["floor_side"] != side:
        return None
    if np.array_equal(grid, out):
        return None
    if not np.all(out[np.array(grey_coords + other_coords).T[0], np.array(grey_coords + other_coords).T[1]] != BG):
        return None
    if np.count_nonzero(grid != BG) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
