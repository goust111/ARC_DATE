from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

RULE_COLORS = (RED, BLUE, GREEN, YELLOW)
MIN_QUADRANT_SIDE = 7
MAX_QUADRANT_SIDE = 14
MIN_PANEL_SIDE = 5
FULL_ARM_PANEL_SIDE = 7
MAX_GENERATION_ATTEMPTS = 400


def _pattern_offsets(border_color: int) -> list[tuple[int, int]]:
    if border_color == RED:
        return [
            (0, 0),
            (-2, 0),
            (-1, 0),
            (1, 0),
            (2, 0),
            (0, -2),
            (0, -1),
            (0, 1),
            (0, 2),
        ]
    if border_color == BLUE:
        return [
            (0, 0),
            (-2, -2),
            (-1, -1),
            (-2, 2),
            (-1, 1),
            (1, -1),
            (2, -2),
            (1, 1),
            (2, 2),
        ]
    if border_color == GREEN:
        return [(dr, dc) for dr in range(-1, 2) for dc in range(-1, 2)]
    if border_color == YELLOW:
        return [(0, 0)]
    raise AssertionError("unexpected border color")


def _separator_indices(g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    grey_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    grey_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(grey_rows) == 1 and len(grey_cols) == 1
    sep_r = int(grey_rows[0])
    sep_c = int(grey_cols[0])
    assert 0 < sep_r < h - 1 and 0 < sep_c < w - 1
    return sep_r, sep_c


def _quadrant_slices(h: int, w: int, sep_r: int, sep_c: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, sep_r), slice(0, sep_c)),
        (slice(0, sep_r), slice(sep_c + 1, w)),
        (slice(sep_r + 1, h), slice(0, sep_c)),
        (slice(sep_r + 1, h), slice(sep_c + 1, w)),
    ]


def _extract_panel(g: np.ndarray, row_slice: slice, col_slice: slice) -> dict[str, int | tuple[int, int] | tuple[int, int, int, int]]:
    region = g[row_slice, col_slice]
    assert region.size > 0
    assert not np.any(region == GREY)

    coords = np.argwhere(region != BLACK)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    crop = region[r0 : r1 + 1, c0 : c1 + 1]

    colors = set(map(int, np.unique(crop)))
    colors.discard(BLACK)
    colors.discard(MAGENTA)
    assert len(colors) == 1
    border_color = int(next(iter(colors)))
    assert border_color in RULE_COLORS

    ch, cw = crop.shape
    assert ch >= 3 and cw >= 3
    frame = np.zeros((ch, cw), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    assert np.all(crop[frame] == border_color)
    assert not np.any(crop[1:-1, 1:-1] == border_color)
    assert np.all(np.isin(crop[~frame], [BLACK, MAGENTA]))

    seeds = np.argwhere(crop == MAGENTA)
    assert len(seeds) == 1
    seed_r, seed_c = map(int, seeds[0])
    assert 0 < seed_r < ch - 1 and 0 < seed_c < cw - 1

    abs_r0 = int(row_slice.start) + int(r0)
    abs_c0 = int(col_slice.start) + int(c0)
    abs_r1 = int(row_slice.start) + int(r1)
    abs_c1 = int(col_slice.start) + int(c1)
    return {
        "color": border_color,
        "bbox": (abs_r0, abs_c0, abs_r1, abs_c1),
        "seed": (abs_r0 + seed_r, abs_c0 + seed_c),
    }


def _apply_growth(out: np.ndarray, panel: dict[str, int | tuple[int, int] | tuple[int, int, int, int]]) -> None:
    color = int(panel["color"])
    r0, c0, r1, c1 = map(int, panel["bbox"])
    seed_r, seed_c = map(int, panel["seed"])
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1

    for dr, dc in _pattern_offsets(color):
        rr = seed_r + dr
        cc = seed_c + dc
        if inner_r0 <= rr <= inner_r1 and inner_c0 <= cc <= inner_c1:
            out[rr, cc] = MAGENTA


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each magenta seed inside its framed panel according to the panel's border color.

    Input structure:
    - Black background.
    - One full Grey row and one full Grey column divide the grid into four quadrants.
    - Each quadrant contains one one-cell-thick rectangular frame colored Red, Blue, Green, or Yellow.
    - Each frame encloses a single Magenta seed pixel on a Black interior.

    Rule:
    - Keep the grid size, separator lines, and panel borders unchanged.
    - In the Red frame, extend the seed two cells in each orthogonal direction to form a clipped plus.
    - In the Blue frame, extend the seed two cells in each diagonal direction to form a clipped X.
    - In the Green frame, expand the seed to a clipped 3x3 solid square.
    - In the Yellow frame, leave the seed as a single pixel.

    Key insight:
    - The border color is a visible selector: it chooses one fixed magenta growth stencil inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_r, sep_c = _separator_indices(g)
    panels = [_extract_panel(g, rs, cs) for rs, cs in _quadrant_slices(*g.shape, sep_r, sep_c)]
    assert len({int(panel["color"]) for panel in panels}) == 4

    out = g.copy()
    for panel in panels:
        _apply_growth(out, panel)
    return out


def _choose_panel_size(region_h: int, region_w: int, color: int) -> tuple[int, int]:
    if color in (RED, BLUE) and region_h >= FULL_ARM_PANEL_SIDE and region_w >= FULL_ARM_PANEL_SIDE and random.random() < 0.5:
        panel_h = random.randint(FULL_ARM_PANEL_SIDE, region_h)
        panel_w = random.randint(FULL_ARM_PANEL_SIDE, region_w)
    else:
        panel_h = random.randint(MIN_PANEL_SIDE, region_h)
        panel_w = random.randint(MIN_PANEL_SIDE, region_w)
    return panel_h, panel_w


def _distance_to_inner_edge(
    cell: tuple[int, int],
    inner_r0: int,
    inner_c0: int,
    inner_r1: int,
    inner_c1: int,
) -> int:
    r, c = cell
    return min(r - inner_r0, inner_r1 - r, c - inner_c0, inner_c1 - c)


def _choose_seed(r0: int, c0: int, r1: int, c1: int, color: int) -> tuple[int, int]:
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    cells = [(r, c) for r in range(inner_r0, inner_r1 + 1) for c in range(inner_c0, inner_c1 + 1)]
    assert cells

    if color in (RED, BLUE):
        full = [
            cell
            for cell in cells
            if _distance_to_inner_edge(cell, inner_r0, inner_c0, inner_r1, inner_c1) >= 2
        ]
        clipped = [
            cell
            for cell in cells
            if _distance_to_inner_edge(cell, inner_r0, inner_c0, inner_r1, inner_c1) == 0
        ]
        if full and clipped:
            return random.choice(full if random.random() < 0.5 else clipped)
        if full:
            return random.choice(full)
        if clipped and random.random() < 0.7:
            return random.choice(clipped)
    elif color == GREEN:
        full = [
            cell
            for cell in cells
            if _distance_to_inner_edge(cell, inner_r0, inner_c0, inner_r1, inner_c1) >= 1
        ]
        clipped = [
            cell
            for cell in cells
            if _distance_to_inner_edge(cell, inner_r0, inner_c0, inner_r1, inner_c1) == 0
        ]
        if full and clipped:
            return random.choice(full if random.random() < 0.5 else clipped)
        if full:
            return random.choice(full)
    return random.choice(cells)


def _draw_panel(
    grid: np.ndarray,
    top: int,
    left: int,
    region_h: int,
    region_w: int,
    color: int,
) -> dict[str, int | tuple[int, int] | tuple[int, int, int, int]]:
    panel_h, panel_w = _choose_panel_size(region_h, region_w, color)
    r0 = top + random.randint(0, region_h - panel_h)
    c0 = left + random.randint(0, region_w - panel_w)
    r1 = r0 + panel_h - 1
    c1 = c0 + panel_w - 1

    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    seed = _choose_seed(r0, c0, r1, c1, color)
    grid[seed] = MAGENTA
    return {"color": color, "bbox": (r0, c0, r1, c1), "seed": seed}


def _expected_output(inp: np.ndarray, panels: list[dict[str, int | tuple[int, int] | tuple[int, int, int, int]]]) -> np.ndarray:
    out = inp.copy()
    for panel in panels:
        _apply_growth(out, panel)
    return out


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(MIN_QUADRANT_SIDE, MAX_QUADRANT_SIDE)
    bottom_h = random.randint(MIN_QUADRANT_SIDE, MAX_QUADRANT_SIDE)
    left_w = random.randint(MIN_QUADRANT_SIDE, MAX_QUADRANT_SIDE)
    right_w = random.randint(MIN_QUADRANT_SIDE, MAX_QUADRANT_SIDE)

    h = top_h + 1 + bottom_h
    w = left_w + 1 + right_w
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[top_h, :] = GREY
    grid[:, left_w] = GREY

    colors = list(RULE_COLORS)
    random.shuffle(colors)
    quadrants = [
        (0, 0, top_h, left_w),
        (0, left_w + 1, top_h, right_w),
        (top_h + 1, 0, bottom_h, left_w),
        (top_h + 1, left_w + 1, bottom_h, right_w),
    ]

    panels: list[dict[str, int | tuple[int, int] | tuple[int, int, int, int]]] = []
    for color, (top, left, region_h, region_w) in zip(colors, quadrants, strict=False):
        panels.append(_draw_panel(grid, top, left, region_h, region_w, int(color)))

    expected = _expected_output(grid, panels)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
