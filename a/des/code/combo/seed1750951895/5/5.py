import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (2, 1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _region_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read top-row and left-column color markers, recover the grey-walled rectangular regions,
    then color each region from the row/column marker crossing.

    Simplified visible rule used here:
    - If a region contains exactly one left-edge marker row and exactly one top-edge marker column,
      fill the whole region with the row-marker color.
    - Then overwrite the region's center row with the column-marker color.
    - Regions without a unique row/column marker pair stay black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    out = g.copy()

    top_markers = [(0, c, int(g[0, c])) for c in range(1, w) if int(g[0, c]) not in (BG, GREY)]
    left_markers = [(r, 0, int(g[r, 0])) for r in range(1, h) if int(g[r, 0]) not in (BG, GREY)]
    assert top_markers and left_markers
    assert len({c for _r, c, _v in top_markers}) == len(top_markers)
    assert len({r for r, _c, _v in left_markers}) == len(left_markers)

    interior = g[1:, 1:]
    region_mask = interior == BG
    comps = _components(region_mask)
    assert comps, "expected at least one enclosed black region"

    for comp0 in comps:
        comp = [(r + 1, c + 1) for r, c in comp0]
        r0, c0, r1, c1 = _region_bbox(comp)
        row_hits = [color for r, _c, color in left_markers if r0 <= r <= r1]
        col_hits = [color for _r, c, color in top_markers if c0 <= c <= c1]
        if len(row_hits) != 1 or len(col_hits) != 1:
            continue

        row_color = int(row_hits[0])
        col_color = int(col_hits[0])
        for r, c in comp:
            out[r, c] = row_color

        center_row = (r0 + r1) // 2
        for r, c in comp:
            if r == center_row:
                out[r, c] = col_color

    return out


def _pick_partitions(total: int, n_cells: int, min_size: int) -> list[int] | None:
    free = total - (n_cells - 1) - n_cells * min_size
    if free < 0:
        return None
    sizes = [min_size] * n_cells
    for _ in range(free):
        sizes[random.randrange(n_cells)] += 1
    random.shuffle(sizes)
    return sizes


def _choose_marked_bands(n_bands: int, want_partial: bool) -> list[int]:
    if n_bands == 1:
        return [0]
    if want_partial:
        count = random.randint(1, n_bands - 1)
    else:
        count = n_bands
    return sorted(random.sample(range(n_bands), count))


def _band_ranges(sizes: list[int]) -> list[tuple[int, int]]:
    start = 1
    out: list[tuple[int, int]] = []
    for i, size in enumerate(sizes):
        out.append((start, start + size - 1))
        start += size
        if i != len(sizes) - 1:
            start += 1
    return out


def _draw_walls(grid: np.ndarray, row_sizes: list[int], col_sizes: list[int]) -> None:
    r = 1
    for i, size in enumerate(row_sizes[:-1]):
        r += size
        grid[r, 1:] = GREY
        r += 1
    c = 1
    for i, size in enumerate(col_sizes[:-1]):
        c += size
        grid[1:, c] = GREY
        c += 1


def _expected_from_layout(
    inp: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    row_markers: dict[int, tuple[int, int]],
    col_markers: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for ri, (r0, r1) in enumerate(row_ranges):
        for ci, (c0, c1) in enumerate(col_ranges):
            if ri not in row_markers or ci not in col_markers:
                continue
            row_color = row_markers[ri][1]
            col_color = col_markers[ci][1]
            out[r0 : r1 + 1, c0 : c1 + 1] = row_color
            out[(r0 + r1) // 2, c0 : c1 + 1] = col_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(9, 18)
    w = random.randint(9, 18)
    n_row_cells = random.randint(2, 4)
    n_col_cells = random.randint(2, 4)
    row_sizes = _pick_partitions(h - 1, n_row_cells, min_size=2)
    col_sizes = _pick_partitions(w - 1, n_col_cells, min_size=2)
    if row_sizes is None or col_sizes is None:
        return None

    row_ranges = _band_ranges(row_sizes)
    col_ranges = _band_ranges(col_sizes)

    make_partial = random.random() < 0.5 and (n_row_cells > 1 or n_col_cells > 1)
    row_marked = _choose_marked_bands(n_row_cells, want_partial=make_partial and random.random() < 0.5)
    col_marked = _choose_marked_bands(n_col_cells, want_partial=make_partial and random.random() < 0.5)
    if not row_marked or not col_marked:
        return None

    color_pool = list(PALETTE)
    random.shuffle(color_pool)
    needed = len(row_marked) + len(col_marked)
    if needed > len(color_pool):
        return None

    grid = np.full((h, w), BG, dtype=int)
    _draw_walls(grid, row_sizes, col_sizes)

    row_markers: dict[int, tuple[int, int]] = {}
    col_markers: dict[int, tuple[int, int]] = {}
    idx = 0
    for band in row_marked:
        r0, r1 = row_ranges[band]
        rr = random.randint(r0, r1)
        color = int(color_pool[idx])
        idx += 1
        grid[rr, 0] = color
        row_markers[band] = (rr, color)
    for band in col_marked:
        c0, c1 = col_ranges[band]
        cc = random.randint(c0, c1)
        color = int(color_pool[idx])
        idx += 1
        grid[0, cc] = color
        col_markers[band] = (cc, color)

    expected = _expected_from_layout(grid, row_ranges, col_ranges, row_markers, col_markers)
    if np.array_equal(expected, grid):
        return None

    both_count = len(row_markers) * len(col_markers)
    uncovered_count = n_row_cells * n_col_cells - both_count
    if make_partial:
        if uncovered_count == 0:
            return None
    else:
        if both_count == 0:
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
