from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(values.argmax())


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx_raw in indices[1:]:
        idx = int(idx_raw)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev + 1))
        start = prev = idx
    runs.append((start, prev + 1))
    return runs


def _intervals(length: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for start, end in runs:
        if cursor < start:
            intervals.append((cursor, start))
        cursor = end
    if cursor < length:
        intervals.append((cursor, length))
    assert intervals, "separator lines must leave at least one panel"
    sizes = {end - start for start, end in intervals}
    assert min(sizes) > 0 and len(sizes) == 1, "all panels must have the same size"
    return intervals


def _find_separator_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    best: tuple[int, int, list[int], list[int]] | None = None
    for color_raw in np.unique(grid):
        color = int(color_raw)
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not rows or not cols:
            continue
        score = len(rows) + len(cols)
        if best is None or score > best[0]:
            best = (score, color, rows, cols)
    assert best is not None, "failed to find the panel separator color"
    return best[1], best[2], best[3]


def _parse_panels(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    sep_color, row_lines, col_lines = _find_separator_color(grid)
    row_intervals = _intervals(grid.shape[0], _runs(row_lines))
    col_intervals = _intervals(grid.shape[1], _runs(col_lines))
    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "panel sizes must be regular"
    panel_size = next(iter(row_sizes))
    assert panel_size == next(iter(col_sizes)), "panels must be square"
    return sep_color, row_intervals, col_intervals, panel_size


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty foreground mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _extract_pattern(panel: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    fg_mask = panel != bg
    assert fg_mask.any(), "each panel must contain a foreground pattern"
    fg_colors = {int(color) for color in np.unique(panel[fg_mask])}
    assert len(fg_colors) == 1, "each panel pattern must be monochrome"
    color = next(iter(fg_colors))
    return _trim_mask(fg_mask), color


def _symmetry_order(mask: np.ndarray) -> int:
    tight = _trim_mask(mask)
    if tight.shape == np.rot90(tight, 1).shape and np.array_equal(tight, np.rot90(tight, 1)):
        return 4
    if np.array_equal(tight, np.rot90(tight, 2)):
        return 2
    return 1


def _transform_mask(mask: np.ndarray, order: int) -> np.ndarray:
    if order == 4:
        h, w = mask.shape
        quarter = mask[: max(1, h // 2), : max(1, w // 2)]
        return _trim_mask(quarter)
    if order == 2:
        return np.fliplr(mask)
    assert order == 1
    return np.rot90(mask, -1)


def _center_panel(mask: np.ndarray, color: int, bg: int, panel_shape: tuple[int, int]) -> np.ndarray:
    ph, pw = panel_shape
    mh, mw = mask.shape
    assert mh <= ph and mw <= pw, "transformed pattern must fit inside the panel"
    out = np.full((ph, pw), bg, dtype=int)
    r0 = (ph - mh) // 2
    c0 = (pw - mw) // 2
    patch = out[r0 : r0 + mh, c0 : c0 + mw]
    patch[mask] = color
    out[r0 : r0 + mh, c0 : c0 + mw] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose a transformation for each panel based on the rotational symmetry order of its centered pattern.

    4-fold rotational symmetry:
    keep the content of the pattern's upper-left quarter, then re-center it.
    2-fold rotational symmetry only:
    flip the pattern horizontally.
    No rotational symmetry:
    rotate the pattern 90 degrees clockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _sep_color, row_intervals, col_intervals, panel_size = _parse_panels(g)
    bg = _dominant_color(g)
    out = g.copy()

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            mask, color = _extract_pattern(panel, bg)
            order = _symmetry_order(mask)
            transformed = _transform_mask(mask, order)
            out[r0:r1, c0:c1] = _center_panel(transformed, color, bg, (panel_size, panel_size))

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
PANEL_SIZE_RANGE = (8, 12)
PANEL_COUNT_RANGE = (2, 3)
PALETTE = list(range(10))


def _grow_connected_mask(height: int, width: int, target_cells: int, start: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[start] = True
    while int(mask.sum()) < target_cells:
        frontier: list[tuple[int, int]] = []
        for r_raw, c_raw in np.argwhere(mask):
            r = int(r_raw)
            c = int(c_raw)
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
    return mask


def _generate_order4_mask(max_side: int) -> np.ndarray:
    quarter_max = max(2, max_side // 2)
    for _ in range(200):
        q = random.randint(2, quarter_max)
        max_cells = max(2, min(q * q - 1, q * q // 2 + q))
        target = random.randint(2, max_cells)
        seed = _grow_connected_mask(q, q, target, (0, 0))
        full = np.zeros((2 * q, 2 * q), dtype=bool)
        full[:q, :q] = seed
        full = full | np.rot90(full, 1) | np.rot90(full, 2) | np.rot90(full, 3)
        full = _trim_mask(full)
        if full.shape[0] != full.shape[1] or full.shape[0] > max_side:
            continue
        if full.shape[0] % 2 != 0 or full.sum() == full.size:
            continue
        if _symmetry_order(full) != 4:
            continue
        quarter = full[: full.shape[0] // 2, : full.shape[1] // 2]
        if not quarter.any():
            continue
        return full
    raise RuntimeError("failed to generate a 4-fold symmetric mask")


def _canonical_half_turn_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            rr = height - 1 - r
            cc = width - 1 - c
            if (r, c) <= (rr, cc):
                cells.append((r, c))
    return cells


def _generate_order2_mask(max_side: int) -> np.ndarray:
    for _ in range(300):
        height = random.randint(4, max_side)
        width = random.randint(4, max_side)
        if height == width:
            continue
        canon = _canonical_half_turn_cells(height, width)
        others = [(r, c) for r, c in canon if (r, c) != (0, 0)]
        if not others:
            continue
        count_hi = min(len(others), max(2, (height * width) // 4))
        count = random.randint(1, count_hi)
        chosen = {(0, 0), *random.sample(others, count)}
        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
            mask[height - 1 - r, width - 1 - c] = True
        mask = _trim_mask(mask)
        if _symmetry_order(mask) != 2:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError("failed to generate a 2-fold symmetric mask")


def _generate_order1_mask(max_side: int) -> np.ndarray:
    for _ in range(300):
        height = random.randint(4, max_side)
        width = random.randint(4, max_side)
        target_hi = min(height * width - 1, max(4, (height * width) // 2))
        target = random.randint(4, target_hi)
        start = (random.randrange(height), random.randrange(width))
        mask = _grow_connected_mask(height, width, target, start)
        mask = _trim_mask(mask)
        if mask.sum() < 4:
            continue
        if _symmetry_order(mask) != 1:
            continue
        return mask
    raise RuntimeError("failed to generate an order-1 mask")


def _branch_mask(branch: int, panel_size: int) -> np.ndarray:
    max_side = panel_size - 2
    if branch == 4:
        return _generate_order4_mask(max_side)
    if branch == 2:
        return _generate_order2_mask(max_side)
    assert branch == 1
    return _generate_order1_mask(max_side)


def _balanced_branches(panel_count: int) -> list[int]:
    branches = [4, 2, 1] * (panel_count // 3)
    remainder = [4, 2, 1]
    random.shuffle(remainder)
    branches.extend(remainder[: panel_count % 3])
    random.shuffle(branches)
    return branches


def _paint_separator_grid(
    row_panels: int,
    col_panels: int,
    panel_size: int,
    bg: int,
    sep: int,
) -> np.ndarray:
    height = row_panels * panel_size + (row_panels - 1)
    width = col_panels * panel_size + (col_panels - 1)
    grid = np.full((height, width), bg, dtype=int)
    for r in range(1, row_panels):
        grid[r * panel_size + (r - 1), :] = sep
    for c in range(1, col_panels):
        grid[:, c * panel_size + (c - 1)] = sep
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    if np.array_equal(out, grid):
        return False

    bg = _dominant_color(grid)
    _sep, row_intervals, col_intervals, _panel_size = _parse_panels(grid)
    orders = set()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            mask, _color = _extract_pattern(grid[r0:r1, c0:c1], bg)
            orders.add(_symmetry_order(mask))
    return orders == {1, 2, 4}


def _gen_attempt() -> np.ndarray | None:
    row_panels = random.randint(*PANEL_COUNT_RANGE)
    col_panels = random.randint(*PANEL_COUNT_RANGE)
    panel_size = random.randint(*PANEL_SIZE_RANGE)
    panel_count = row_panels * col_panels

    bg, sep = random.sample(PALETTE, 2)
    fg_candidates = [color for color in PALETTE if color not in {bg, sep}]
    fg_pool = random.sample(fg_candidates, random.randint(3, min(6, len(fg_candidates))))
    branches = _balanced_branches(panel_count)

    grid = _paint_separator_grid(row_panels, col_panels, panel_size, bg, sep)
    index = 0
    for pr in range(row_panels):
        for pc in range(col_panels):
            branch = branches[index]
            color = random.choice(fg_pool)
            mask = _branch_mask(branch, panel_size)
            panel = _center_panel(mask, color, bg, (panel_size, panel_size))
            r0 = pr * (panel_size + 1)
            c0 = pc * (panel_size + 1)
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel
            index += 1

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
