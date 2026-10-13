from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BAND_COLORS = (1, 2)
PATTERN_COLORS = (3, 4)
SWAP_PATTERN_COLOR = {3: 4, 4: 3}
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_band_rows(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    bands: list[tuple[int, int, int]] = []
    for r, row in enumerate(grid):
        gap_cols = np.flatnonzero(row == bg)
        if gap_cols.size != 1:
            continue
        non_bg = row[row != bg]
        if non_bg.size != row.size - 1:
            continue
        colors = np.unique(non_bg)
        if colors.size != 1:
            continue
        bands.append((int(r), int(gap_cols[0]), int(colors[0])))
    return bands


def _axis_from_bands(bands: list[tuple[int, int, int]], width: int) -> int:
    counts = np.zeros(width, dtype=int)
    for _r, gap_col, _color in bands:
        counts[gap_col] += 1
    return int(np.argmax(counts))


def _find_pattern_component(grid: np.ndarray) -> np.ndarray:
    mask = np.isin(grid, PATTERN_COLORS)
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one green/yellow pattern"
    coords = comps[0]
    colors = {int(grid[r, c]) for r, c in coords}
    assert colors == set(PATTERN_COLORS), "pattern must use both green and yellow"
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the mirror axis from the most frequent band-gap column, reflect the two-color motif, swap its original colors, and fill every gap.

    Input structure:
    - The background is black, with several full-width horizontal bands made of a single non-background color and exactly one black gap per band row.
    - The band rows use blue and red, while the background rows between them contain one 4-connected green/yellow motif on exactly one side of the hidden vertical axis.
    - One gap column occurs more often than every other gap column, and that most frequent column is the implicit mirror axis.

    Rule:
    - Detect every horizontal band row and count how many band gaps fall in each column; use the highest-frequency column, breaking ties to the left, as the reflection axis.
    - Find the single 4-connected motif made of green and yellow cells, then copy it across the axis while preserving the copied colors.
    - Swap green and yellow in place on the original motif, and finally fill each band gap with that row's band color.

    Key insight:
    - The axis is not drawn; it must be inferred from the global frequency pattern of the one-pixel gaps inside the horizontal bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    bands = _find_band_rows(g, bg)
    assert bands, "expected at least one horizontal band row"
    axis = _axis_from_bands(bands, w)

    pattern = _find_pattern_component(g)
    cols = pattern[:, 1]
    assert np.all(cols < axis) or np.all(cols > axis), "pattern must lie on one side of the axis"

    out = g.copy()
    for r, c in pattern:
        mirror_c = 2 * axis - int(c)
        assert 0 <= mirror_c < w, "reflected motif must stay in bounds"
        assert int(g[r, mirror_c]) == bg, "reflected target cells must be empty in the input"
        out[r, mirror_c] = int(g[r, c])
        out[r, c] = SWAP_PATTERN_COLOR[int(g[r, c])]

    for r, gap_col, color in bands:
        out[r, gap_col] = color
    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    frontier: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                frontier.append((r, c))
    return frontier


def _grow_connected_mask(height: int, width: int, count: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    while int(mask.sum()) < count:
        mask[random.choice(_frontier(mask))] = True
    return mask


def _sample_band_layout() -> tuple[int, list[int], int]:
    for _ in range(200):
        band_count = random.randint(5, 8)
        special_gap_index = random.randrange(band_count - 1)
        internal_gaps = [1 + int(random.random() < 0.35) for _ in range(band_count - 1)]
        internal_gaps[special_gap_index] = random.randint(2, 4)
        top_margin = random.randint(0, 1)
        bottom_margin = random.randint(0, 1)

        rows = [top_margin]
        for gap in internal_gaps:
            rows.append(rows[-1] + 1 + gap)
        height = rows[-1] + 1 + bottom_margin
        if 11 <= height <= 18:
            return height, rows, special_gap_index
    raise RuntimeError("failed to sample band rows")


def _sample_band_colors(count: int) -> list[int]:
    start = random.choice(BAND_COLORS)
    colors = [start]
    for _ in range(1, count):
        prev = colors[-1]
        if random.random() < 0.75:
            colors.append(1 if prev == 2 else 2)
        else:
            colors.append(prev)
    if len(set(colors)) == 1:
        colors[random.randrange(count)] = 1 if colors[0] == 2 else 2
    return colors


def _sample_gap_columns(width: int, band_count: int, axis: int) -> list[int]:
    dominant_count = random.randint(max(2, band_count // 3), min(band_count - 1, 4))
    cols = [axis] * dominant_count
    other_cols = [c for c in range(1, width - 1) if c != axis]
    counts = {axis: dominant_count}
    while len(cols) < band_count:
        col = random.choice(other_cols)
        if counts.get(col, 0) >= dominant_count - 1:
            continue
        cols.append(col)
        counts[col] = counts.get(col, 0) + 1
    random.shuffle(cols)
    return cols


def _render_pattern(
    grid: np.ndarray,
    axis: int,
    gap_top: int,
    gap_bottom: int,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    mirror_depth = min(axis, grid.shape[1] - 1 - axis)
    bbox_h = random.randint(2, min(4, gap_bottom - gap_top + 1))
    bbox_w = random.randint(2, min(4, mirror_depth))
    cell_count = random.randint(4, min(8, bbox_h * bbox_w))
    local_mask = _grow_connected_mask(bbox_h, bbox_w, cell_count)

    side = random.choice(("left", "right"))
    if side == "left":
        c0_min = axis - mirror_depth
        c0_max = axis - bbox_w
    else:
        c0_min = axis + 1
        c0_max = axis + mirror_depth - bbox_w + 1
    r0 = random.randint(gap_top, gap_bottom - bbox_h + 1)
    c0 = random.randint(c0_min, c0_max)

    coords = [(r0 + r, c0 + c) for r, c in np.argwhere(local_mask)]
    order = coords[:]
    random.shuffle(order)
    split = random.randint(1, len(order) - 1)
    color_map: dict[tuple[int, int], int] = {}
    for cell in order[:split]:
        color_map[cell] = PATTERN_COLORS[0]
    for cell in order[split:]:
        color_map[cell] = PATTERN_COLORS[1]

    for (r, c), color in color_map.items():
        mirror_c = 2 * axis - c
        assert int(grid[r, c]) == BG
        assert int(grid[r, mirror_c]) == BG
        grid[r, c] = color
    return coords, color_map


def _expected_output(
    grid: np.ndarray,
    bands: list[tuple[int, int, int]],
    axis: int,
    color_map: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for (r, c), color in color_map.items():
        mirror_c = 2 * axis - c
        out[r, mirror_c] = color
        out[r, c] = SWAP_PATTERN_COLOR[color]
    for r, gap_col, color in bands:
        out[r, gap_col] = color
    return out


def _axis_is_unique_max(bands: list[tuple[int, int, int]], width: int, axis: int) -> bool:
    counts = np.zeros(width, dtype=int)
    for _r, gap_col, _color in bands:
        counts[gap_col] += 1
    best = int(counts[axis])
    return best > 0 and all(best > int(counts[c]) for c in range(width) if c != axis)


def _gen_attempt() -> np.ndarray | None:
    height, band_rows, special_gap_index = _sample_band_layout()
    width = random.randint(13, 18)
    axis = random.randint(2, width - 3)
    if min(axis, width - 1 - axis) < 2:
        return None

    band_colors = _sample_band_colors(len(band_rows))
    gap_cols = _sample_gap_columns(width, len(band_rows), axis)
    bands = [(r, gap_col, color) for r, gap_col, color in zip(band_rows, gap_cols, band_colors)]
    if not _axis_is_unique_max(bands, width, axis):
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, gap_col, color in bands:
        grid[r, :] = color
        grid[r, gap_col] = BG

    gap_top = band_rows[special_gap_index] + 1
    gap_bottom = band_rows[special_gap_index + 1] - 1
    if gap_bottom - gap_top + 1 < 2:
        return None

    try:
        _coords, color_map = _render_pattern(grid, axis, gap_top, gap_bottom)
    except Exception:
        return None

    expected = _expected_output(grid, bands, axis, color_map)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
