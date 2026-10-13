from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400
ROW_CELL_OPTIONS = (1, 2, 3)
COL_CELL_OPTIONS = (1, 2, 3)
GAP_RANGE = (4, 7)
TOP_MARGIN_RANGE = (0, 4)
SIDE_MARGIN_RANGE = (0, 4)
BOTTOM_MARGIN_RANGE = (1, 4)
MAX_PALETTE_LEN = 7


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _read_palette(bottom_row: np.ndarray, bg: int) -> list[int]:
    palette: list[int] = []
    for value in bottom_row.tolist():
        value = int(value)
        if value == bg:
            break
        palette.append(value)
    assert len(palette) == len(set(palette)), "palette colors must be distinct"
    assert all(int(v) == bg for v in bottom_row[len(palette) :]), "palette must be a left-aligned strip"
    return palette


def _identify_line_and_islands(top: np.ndarray, bg: int) -> tuple[int, int | None]:
    colors = [int(v) for v in np.unique(top) if int(v) != bg]
    assert colors, "expected at least one non-background color above the palette"

    stats: list[tuple[int, int, int, bool]] = []
    for color in colors:
        comps = _components(top == color)
        sizes = [len(comp) for comp in comps]
        stats.append((color, max(sizes), sum(sizes), all(size == 1 for size in sizes)))

    line_color = max(stats, key=lambda item: (item[1], item[2]))[0]
    island_candidates = [color for color, _max_size, _total, all_singletons in stats if color != line_color and all_singletons]
    island_color = island_candidates[0] if island_candidates else None
    return int(line_color), None if island_color is None else int(island_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region according to how many isolated island pixels it contains.

    The bottom row is a left-aligned palette strip: the Nth palette color fills every
    non-line pixel of a closed region containing exactly N islands. Regions with no
    islands remain black, the line network stays unchanged, and the palette row is
    cleared in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a palette row"

    bg = _most_common_color(g)
    palette = _read_palette(g[-1], bg)
    top = g[:-1]
    line_color, island_color = _identify_line_and_islands(top, bg)

    out = g.copy()
    out[-1, :] = bg

    for comp in _components(top != line_color):
        if any(r == 0 or c == 0 or r == top.shape[0] - 1 or c == top.shape[1] - 1 for r, c in comp):
            continue
        island_count = 0
        if island_color is not None:
            island_count = sum(1 for r, c in comp if int(top[r, c]) == island_color)
        if island_count == 0:
            continue
        assert island_count <= len(palette), "palette must cover the largest island count"
        fill_color = int(palette[island_count - 1])
        for r, c in comp:
            out[r, c] = fill_color
    return out


def _slot_positions(r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int]]:
    rows = list(range(r0 + 2, r1 - 1, 2))
    cols = list(range(c0 + 2, c1 - 1, 2))
    return [(r, c) for r in rows for c in cols]


def _sample_region_counts(rng: np.random.Generator, capacities: list[int]) -> list[int]:
    n_regions = len(capacities)
    assert n_regions >= 2 and max(capacities) >= 1

    max_count = int(rng.integers(1, min(MAX_PALETTE_LEN, max(capacities)) + 1))
    counts = [0] * n_regions

    max_candidates = [idx for idx, cap in enumerate(capacities) if cap >= max_count]
    max_idx = int(rng.choice(max_candidates))
    counts[max_idx] = max_count

    for idx, cap in enumerate(capacities):
        if idx == max_idx:
            continue
        counts[idx] = int(rng.integers(0, min(cap, max_count) + 1))

    zero_candidates = [idx for idx in range(n_regions) if idx != max_idx]
    if zero_candidates and all(counts[idx] > 0 for idx in zero_candidates):
        zero_idx = int(rng.choice(zero_candidates))
        counts[zero_idx] = 0

    positive_candidates = [idx for idx, cap in enumerate(capacities) if idx != max_idx and cap >= 1]
    if positive_candidates and sum(count > 0 for count in counts) == 1:
        idx = int(rng.choice(positive_candidates))
        counts[idx] = int(rng.integers(1, min(capacities[idx], max_count) + 1))

    return counts


def _draw_network(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    line_color: int,
) -> None:
    for r in row_lines:
        grid[int(r), col_lines[0] : col_lines[-1] + 1] = int(line_color)
    for c in col_lines:
        grid[row_lines[0] : row_lines[-1] + 1, int(c)] = int(line_color)


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    shape_options = [
        (nr, nc)
        for nr in ROW_CELL_OPTIONS
        for nc in COL_CELL_OPTIONS
        if nr * nc >= 2
    ]
    n_row_cells, n_col_cells = shape_options[int(rng.integers(len(shape_options)))]

    row_gaps = [int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1)) for _ in range(n_row_cells)]
    col_gaps = [int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1)) for _ in range(n_col_cells)]

    top_margin = int(rng.integers(TOP_MARGIN_RANGE[0], TOP_MARGIN_RANGE[1] + 1))
    left_margin = int(rng.integers(SIDE_MARGIN_RANGE[0], SIDE_MARGIN_RANGE[1] + 1))
    right_margin = int(rng.integers(SIDE_MARGIN_RANGE[0], SIDE_MARGIN_RANGE[1] + 1))
    bottom_margin = int(rng.integers(BOTTOM_MARGIN_RANGE[0], BOTTOM_MARGIN_RANGE[1] + 1))

    top_height = top_margin + sum(row_gaps) + 1 + bottom_margin
    width = left_margin + sum(col_gaps) + 1 + right_margin
    if width < 2 or top_height < 2:
        return None

    row_lines = [top_margin]
    for gap in row_gaps:
        row_lines.append(row_lines[-1] + int(gap))
    col_lines = [left_margin]
    for gap in col_gaps:
        col_lines.append(col_lines[-1] + int(gap))

    regions: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []
    capacities: list[int] = []
    for i in range(n_row_cells):
        for j in range(n_col_cells):
            r0 = row_lines[i]
            r1 = row_lines[i + 1]
            c0 = col_lines[j]
            c1 = col_lines[j + 1]
            slots = _slot_positions(r0, r1, c0, c1)
            if not slots:
                return None
            regions.append((r0, r1, c0, c1, slots))
            capacities.append(len(slots))

    counts = _sample_region_counts(rng, capacities)
    palette_len = max(counts)
    if palette_len < 1 or palette_len > width:
        return None

    colors = rng.permutation(np.arange(1, 10))
    line_color = int(colors[0])
    island_color = int(colors[1])
    palette = [int(v) for v in colors[2 : 2 + palette_len]]

    inp = np.full((top_height + 1, width), BG, dtype=int)
    _draw_network(inp, row_lines, col_lines, line_color)

    expected = inp.copy()
    for (r0, r1, c0, c1, slots), count in zip(regions, counts, strict=False):
        if count > 0:
            chosen_idx = rng.choice(len(slots), size=count, replace=False)
            for idx in np.atleast_1d(chosen_idx).tolist():
                rr, cc = slots[int(idx)]
                inp[rr, cc] = island_color
            expected[r0 + 1 : r1, c0 + 1 : c1] = palette[count - 1]

    inp[-1, :palette_len] = palette
    expected[-1, :] = BG

    if np.array_equal(main(inp), expected):
        return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        built = _build_case(rng)
        if built is not None:
            inp, _expected = built
            return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
