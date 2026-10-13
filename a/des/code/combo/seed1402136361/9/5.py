from __future__ import annotations

import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "expected a unique dominant background color"
    return int(winners[0])


def _quadrant_slices(shape: tuple[int, int]) -> tuple[tuple[slice, slice], ...]:
    height, width = shape
    top = slice(0, height // 2)
    bottom = slice((height + 1) // 2, height)
    left = slice(0, width // 2)
    right = slice((width + 1) // 2, width)
    return (
        (top, left),
        (top, right),
        (bottom, left),
        (bottom, right),
    )


def _quadrant_regions(shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], ...]:
    height, width = shape
    top_stop = height // 2
    bottom_start = (height + 1) // 2
    left_stop = width // 2
    right_start = (width + 1) // 2
    return (
        (1, top_stop, 1, left_stop),
        (1, top_stop, right_start, width - 1),
        (bottom_start, height - 1, 1, left_stop),
        (bottom_start, height - 1, right_start, width - 1),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the four corner colors as quadrant-specific filters.

    In each quadrant, keep only the pixels matching that quadrant's corner color.
    All other foreground pixels become background, and the corner markers are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 4 and width >= 4, "grid is too small for four quadrants"

    bg = _dominant_color(g)
    corners = (
        int(g[0, 0]),
        int(g[0, width - 1]),
        int(g[height - 1, 0]),
        int(g[height - 1, width - 1]),
    )
    assert len(set(corners)) == 4, "expected four distinct corner markers"
    assert bg not in corners, "corner markers must differ from the background"

    out = np.full_like(g, bg)
    for active_color, (row_slice, col_slice) in zip(corners, _quadrant_slices(g.shape)):
        view = g[row_slice, col_slice]
        out[row_slice, col_slice] = np.where(view == active_color, active_color, bg)

    out[0, 0] = bg
    out[0, width - 1] = bg
    out[height - 1, 0] = bg
    out[height - 1, width - 1] = bg
    return out


# --------------- generator ---------------

COLORS = tuple(range(10))
GRID_SIZE_RANGE = (10, 18)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80


def _neighbors8(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width:
                out.append((rr, cc))
    return out


def _shape_rect(max_h: int, max_w: int) -> set[tuple[int, int]]:
    h = random.randint(1, min(3, max_h))
    w = random.randint(1, min(3, max_w))
    return {(r, c) for r in range(h) for c in range(w)}


def _shape_line(max_h: int, max_w: int) -> set[tuple[int, int]]:
    if max_h == 1:
        horizontal = True
    elif max_w == 1:
        horizontal = False
    else:
        horizontal = random.random() < 0.5

    if horizontal:
        length = random.randint(1, min(4, max_w))
        return {(0, c) for c in range(length)}
    length = random.randint(1, min(4, max_h))
    return {(r, 0) for r in range(length)}


def _shape_l(max_h: int, max_w: int) -> set[tuple[int, int]]:
    h = random.randint(2, min(4, max_h))
    w = random.randint(2, min(4, max_w))
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        return {(0, c) for c in range(w)} | {(r, 0) for r in range(h)}
    if corner == "tr":
        return {(0, c) for c in range(w)} | {(r, w - 1) for r in range(h)}
    if corner == "bl":
        return {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)}
    return {(h - 1, c) for c in range(w)} | {(r, w - 1) for r in range(h)}


def _shape_blob(max_h: int, max_w: int) -> set[tuple[int, int]]:
    h = random.randint(2, min(4, max_h))
    w = random.randint(2, min(4, max_w))
    target = random.randint(3, min(6, h * w))
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))
    return cells


def _random_shape(max_h: int, max_w: int) -> set[tuple[int, int]]:
    choices = [_shape_rect, _shape_line]
    if max_h >= 2 and max_w >= 2:
        choices.extend([_shape_l, _shape_blob])
    return random.choice(choices)(max_h, max_w)


def _place_shape(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    region: tuple[int, int, int, int],
    color: int,
    keep: bool,
    bg: int,
) -> bool:
    row_start, row_stop, col_start, col_stop = region
    region_h = row_stop - row_start
    region_w = col_stop - col_start
    if region_h <= 0 or region_w <= 0:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        shape = _random_shape(region_h, region_w)
        shape_h = max(r for r, _c in shape) + 1
        shape_w = max(c for _r, c in shape) + 1
        if shape_h > region_h or shape_w > region_w:
            continue

        base_r = random.randint(row_start, row_stop - shape_h)
        base_c = random.randint(col_start, col_stop - shape_w)
        cells = [(base_r + r, base_c + c) for r, c in shape]
        if any(grid[r, c] != bg or blocked[r, c] for r, c in cells):
            continue

        for r, c in cells:
            grid[r, c] = color
            if keep:
                expected[r, c] = color
        for r, c in cells:
            for rr, cc in _neighbors8(r, c, grid.shape):
                blocked[rr, cc] = True
        return True
    return False


def _background_is_unique(grid: np.ndarray, bg: int) -> bool:
    colors, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[colors == bg][0])
    return all(bg_count > int(count) for color, count in zip(colors, counts) if int(color) != bg)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)

    bg = random.choice(COLORS)
    corner_colors = random.sample([c for c in COLORS if c != bg], 4)
    grid = np.full((height, width), bg, dtype=int)
    expected = np.full_like(grid, bg)
    blocked = np.zeros_like(grid, dtype=bool)

    corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
    for (r, c), color in zip(corners, corner_colors):
        grid[r, c] = color

    other_corner_colors = {
        idx: [corner_colors[j] for j in range(4) if j != idx]
        for idx in range(4)
    }
    regions = _quadrant_regions(grid.shape)
    for idx, region in enumerate(regions):
        active = corner_colors[idx]
        distractor = random.choice(other_corner_colors[idx])
        jobs: list[tuple[int, bool]] = [(active, True), (distractor, False)]
        if random.random() < 0.7:
            if random.random() < 0.5:
                jobs.append((active, True))
            else:
                palette = [c for c in COLORS if c not in (bg, active)]
                jobs.append((random.choice(palette), False))
        random.shuffle(jobs)

        for color, keep in jobs:
            if not _place_shape(grid, expected, blocked, region, color, keep, bg):
                return None

    if not _background_is_unique(grid, bg):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
