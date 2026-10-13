from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
CONFLICT = 2
SOURCE_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _groups(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    return groups


def _layout(grid: np.ndarray, bg: int) -> tuple[int, tuple[slice, slice, slice], list[slice]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in [int(c) for c in np.unique(grid) if int(c) != bg]:
        full_cols = [int(c) for c in np.flatnonzero(np.all(grid == color, axis=0))]
        col_groups = _groups(full_cols)
        if len(col_groups) != 2:
            continue
        full_rows = [int(r) for r in np.flatnonzero(np.all(grid == color, axis=1))]
        row_groups = _groups(full_rows)
        candidates.append((color, row_groups, col_groups))

    assert len(candidates) == 1, "expected a unique separator color"
    sep, row_groups, col_groups = candidates[0]

    (c0a, c0b), (c1a, c1b) = col_groups
    left = slice(0, c0a)
    middle = slice(c0b + 1, c1a)
    right = slice(c1b + 1, w)
    region_widths = (left.stop - left.start, middle.stop - middle.start, right.stop - right.start)
    assert region_widths[0] > 0 and len(set(region_widths)) == 1, "three regions must have equal positive width"

    row_slices: list[slice] = []
    start = 0
    for r0, r1 in row_groups:
        if start < r0:
            row_slices.append(slice(start, r0))
        start = r1 + 1
    if start < h:
        row_slices.append(slice(start, h))
    assert row_slices, "expected at least one row band"
    return sep, (left, middle, right), row_slices


def _project_differences(left: np.ndarray, middle: np.ndarray, bg: int, conflict: int) -> np.ndarray:
    out = np.full_like(left, bg)
    same = left == middle
    left_only = (left != bg) & (middle == bg)
    middle_only = (left == bg) & (middle != bg)
    conflict_mask = (~same) & (left != bg) & (middle != bg)

    out[left_only] = left[left_only]
    out[middle_only] = middle[middle_only]
    out[conflict_mask] = conflict
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the aligned left and middle regions in each row band and paint a classified difference projection in the right region.

    Input structure:
    - Black background with two full-height grey separator columns, splitting the grid into three equal-width regions.
    - Optional full-width grey separator rows split the puzzle into independent horizontal bands.
    - In each band, the left and middle regions contain aligned multi-color patterns of the same size; the right region starts blank.

    Rule:
    - For every band, compare the left and middle regions cell by cell.
    - Equal colors stay black in the right region.
    - If exactly one side is non-background, copy that visible color into the right region.
    - If both sides are non-background but disagree, paint the right cell red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    assert bg != CONFLICT, "background must differ from the conflict color"
    sep, (left_cols, middle_cols, right_cols), row_slices = _layout(g, bg)
    assert sep != bg and sep != CONFLICT, "separator color must be distinct"

    out = g.copy()
    for rows in row_slices:
        left = g[rows, left_cols]
        middle = g[rows, middle_cols]
        right = g[rows, right_cols]

        assert left.shape == middle.shape == right.shape, "band regions must align"
        assert np.all(right == bg), "right region must start empty"

        out[rows, right_cols] = _project_differences(left, middle, bg, CONFLICT)

    return out


def _sample_patch(h: int, w: int) -> list[tuple[int, int]]:
    kind = random.choices(
        ["rect", "hline", "vline", "dot"],
        weights=[5, 3, 3, 1],
        k=1,
    )[0]

    if kind == "rect":
        ph = random.randint(1, min(3, h))
        pw = random.randint(1, min(4, w))
    elif kind == "hline":
        ph = 1
        pw = random.randint(2, w)
    elif kind == "vline":
        ph = random.randint(2, h)
        pw = 1
    else:
        ph = pw = 1

    r0 = random.randint(0, h - ph)
    c0 = random.randint(0, w - pw)
    return [(r0 + dr, c0 + dc) for dr in range(ph) for dc in range(pw)]


def _base_pattern(h: int, w: int, palette: list[int]) -> np.ndarray | None:
    area = h * w
    min_filled = max(6, area // 4)
    max_filled = max(min_filled, min(area - 4, (2 * area) // 3))
    target = random.randint(min_filled, max_filled)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    colors_used: set[int] = set()

    for _ in range(80):
        if int(np.count_nonzero(grid != BACKGROUND)) >= target:
            break
        cells = _sample_patch(h, w)
        if any(grid[r, c] != BACKGROUND for r, c in cells):
            continue

        unused = [color for color in palette if color not in colors_used]
        if unused and (len(colors_used) < min(2, len(palette)) or random.random() < 0.45):
            color = random.choice(unused)
        else:
            color = random.choice(palette)

        for r, c in cells:
            grid[r, c] = color
        colors_used.add(color)

    filled = int(np.count_nonzero(grid != BACKGROUND))
    if filled < min_filled:
        return None
    if len({int(v) for v in np.unique(grid) if int(v) != BACKGROUND}) < 2:
        return None
    return grid


def _pick_coord(coords: set[tuple[int, int]]) -> tuple[int, int]:
    return random.choice(tuple(coords))


def _different_color(palette: list[int], color: int) -> int:
    return random.choice([candidate for candidate in palette if candidate != color])


def _informative_pair(left: np.ndarray, middle: np.ndarray) -> bool:
    same_fg = bool(np.any((left == middle) & (left != BACKGROUND)))
    same_bg = bool(np.any((left == middle) & (left == BACKGROUND)))
    left_only = bool(np.any((left != BACKGROUND) & (middle == BACKGROUND)))
    middle_only = bool(np.any((left == BACKGROUND) & (middle != BACKGROUND)))
    conflict = bool(np.any((left != BACKGROUND) & (middle != BACKGROUND) & (left != middle)))
    left_colors = {int(v) for v in np.unique(left) if int(v) != BACKGROUND}
    middle_colors = {int(v) for v in np.unique(middle) if int(v) != BACKGROUND}
    return same_fg and same_bg and left_only and middle_only and conflict and len(left_colors) >= 2 and len(middle_colors) >= 2


def _make_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    palette = random.sample(SOURCE_COLORS, random.randint(2, min(5, len(SOURCE_COLORS))))
    base = _base_pattern(h, w, palette)
    if base is None:
        return None

    left = base.copy()
    middle = base.copy()

    area = h * w
    max_per_type = min(4, max(1, area // 8))
    counts = {
        "left_only": random.randint(1, max_per_type),
        "middle_only": random.randint(1, max_per_type),
        "conflict": random.randint(1, max_per_type),
    }

    occupied = {tuple(map(int, rc)) for rc in np.argwhere(base != BACKGROUND)}
    free = {tuple(map(int, rc)) for rc in np.argwhere(base == BACKGROUND)}

    for _ in range(counts["left_only"]):
        use_free = bool(free) and (not occupied or random.random() < 0.5)
        if use_free:
            r, c = _pick_coord(free)
            left[r, c] = random.choice(palette)
            free.remove((r, c))
        else:
            r, c = _pick_coord(occupied)
            middle[r, c] = BACKGROUND
            occupied.remove((r, c))

    for _ in range(counts["middle_only"]):
        use_free = bool(free) and (not occupied or random.random() < 0.5)
        if use_free:
            r, c = _pick_coord(free)
            middle[r, c] = random.choice(palette)
            free.remove((r, c))
        else:
            r, c = _pick_coord(occupied)
            left[r, c] = BACKGROUND
            occupied.remove((r, c))

    for _ in range(counts["conflict"]):
        use_free = bool(free) and (not occupied or random.random() < 0.35)
        if use_free:
            r, c = _pick_coord(free)
            left_color, middle_color = random.sample(palette, 2)
            left[r, c] = left_color
            middle[r, c] = middle_color
            free.remove((r, c))
        else:
            r, c = _pick_coord(occupied)
            base_color = int(base[r, c])
            if random.random() < 0.5:
                left[r, c] = base_color
                middle[r, c] = _different_color(palette, base_color)
            else:
                left[r, c] = _different_color(palette, base_color)
                middle[r, c] = base_color
            occupied.remove((r, c))

    if not _informative_pair(left, middle):
        return None
    return left, middle


def _assemble(bands: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    region_width = bands[0][0].shape[1]
    total_width = 3 * region_width + 2
    rows: list[np.ndarray] = []

    for idx, (left, middle) in enumerate(bands):
        band_h = left.shape[0]
        band = np.full((band_h, total_width), BACKGROUND, dtype=int)
        band[:, :region_width] = left
        band[:, region_width] = SEPARATOR
        band[:, region_width + 1 : 2 * region_width + 1] = middle
        band[:, 2 * region_width + 1] = SEPARATOR
        rows.append(band)
        if idx + 1 < len(bands):
            rows.append(np.full((1, total_width), SEPARATOR, dtype=int))

    return np.vstack(rows)


def _gen_attempt() -> np.ndarray | None:
    num_bands = random.randint(1, 4)
    region_width = random.randint(5, 10)

    bands: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(num_bands):
        band_h = random.randint(4, 8)
        pair = _make_pair(band_h, region_width)
        if pair is None:
            return None
        bands.append(pair)

    grid = _assemble(bands)

    try:
        out = main(grid)
    except AssertionError:
        return None

    right_start = 2 * region_width + 2
    if np.array_equal(out[:, right_start:], grid[:, right_start:]):
        return None
    if not np.any(out[:, right_start:] == CONFLICT):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
