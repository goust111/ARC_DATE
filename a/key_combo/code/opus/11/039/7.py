from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

WIDTH_OPTIONS = (11, 13, 15, 17)
SEPARATOR_COUNT_OPTIONS = (2, 3)
BAND_HEIGHT_OPTIONS = (2, 3, 4)
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _fill_gaps(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    fill_mask = np.zeros_like(g, dtype=bool)
    fill_mask[:, 1:-1] = (
        (g[:, 1:-1] == BLACK)
        & (g[:, :-2] == BLUE)
        & (g[:, 2:] == BLUE)
    )
    out[fill_mask] = RED
    return out


def _separator_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(np.isin(row, (BLUE, RED))):
            rows.append(int(r))
    return rows


def _band_ranges(height: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, height]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]


def _is_symmetric_band(grid: np.ndarray, r0: int, r1: int) -> bool:
    grey = grid == GREY
    width = grid.shape[1]
    coords = np.argwhere(grey[r0:r1, :])
    for rr, cc in coords:
        rr = int(rr + r0)
        cc = int(cc)
        if not grey[rr, width - 1 - cc]:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete dashed horizontal separators, then recolor symmetric grey patterns within each horizontal band.

    Input structure:
    - The grid has a black background, 2-3 full-width dashed blue horizontal rows, and grey pixels inside the bands between those dashed rows.
    - Each dashed separator spans the entire row from the left edge to the right edge, with only single black gaps between neighboring blue pixels.
    - The dashed rows divide the canvas into horizontal bands; grey pixels are interpreted only within their own band.

    Rule:
    - Recolor every black cell whose immediate left and right neighbors are blue to red, completing the dashed horizontal separators.
    - Treat the completed blue/red rows as full-width barriers and split the grid into the horizontal bands above, between, and below them.
    - For each band independently, test whether all grey pixels are horizontally symmetric about the grid's vertical midline; if so, recolor every grey pixel in that band to green, otherwise leave that band's grey pixels unchanged.
    - Keep the original blue pixels and the newly created red pixels unchanged.

    Key insight:
    - Filling the one-cell gaps is not the final goal; it is the step that makes the separator rows into true barriers, which changes the symmetry scope from one merged region into distinct horizontal bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h <= 30 and w <= 30, "ARC grids must be at most 30x30"
    assert w >= 3 and w % 2 == 1, "expected an odd width so dashed bars can span the row"
    assert _get_bg(g) == BLACK, "expected black background"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, GREY}), "input should contain only black, blue, and grey"

    blue_rows = [r for r in range(h) if np.any(g[r] == BLUE)]
    assert 2 <= len(blue_rows) <= 3, "expected 2-3 dashed separator rows"
    for r in blue_rows:
        row = g[r]
        assert not np.any(row == GREY), "separator rows cannot contain grey"
        assert row[0] == BLUE and row[-1] == BLUE, "dashed separator must span the full row"
        assert np.all(row[::2] == BLUE), "blue separator pixels must occupy alternating columns"
        assert np.all(row[1::2] == BLACK), "separator gaps must be one-cell black gaps"
    separators = sorted(int(r) for r in blue_rows)
    assert separators == blue_rows, "separator rows must be strictly ordered"

    filled = _fill_gaps(g)
    full_rows = _separator_rows(filled)
    assert full_rows == separators, "gap filling should complete exactly the separator rows"

    bands = _band_ranges(h, full_rows)
    assert len(bands) == len(separators) + 1
    assert all(r0 < r1 for r0, r1 in bands), "each horizontal band must be non-empty"

    grey_total = int(np.count_nonzero(g == GREY))
    assert grey_total > 0, "expected grey foreground pixels in the bands"

    out = filled.copy()
    for r0, r1 in bands:
        grey_count = int(np.count_nonzero(g[r0:r1] == GREY))
        assert grey_count >= 3, "each band should contain at least three grey pixels"
        if _is_symmetric_band(g, r0, r1):
            out[r0:r1][g[r0:r1] == GREY] = GREEN
        else:
            out[r0:r1][g[r0:r1] == GREY] = GREY
    return out


def _mirror_col(width: int, c: int) -> int:
    return width - 1 - c


def _sample_band_heights(band_count: int) -> list[int]:
    heights = [random.choice(BAND_HEIGHT_OPTIONS) for _ in range(band_count)]
    if random.random() < 0.35:
        idx = random.randrange(band_count)
        heights[idx] = 1
    return heights


def _band_rows_from_heights(heights: list[int]) -> tuple[int, list[int], list[tuple[int, int]]]:
    separators: list[int] = []
    bands: list[tuple[int, int]] = []
    cursor = 0
    for i, band_h in enumerate(heights):
        bands.append((cursor, cursor + band_h))
        cursor += band_h
        if i + 1 < len(heights):
            separators.append(cursor)
            cursor += 1
    return cursor, separators, bands


def _candidate_left_cells(r0: int, r1: int, width: int) -> list[tuple[int, int]]:
    mid = width // 2
    return [(r, c) for r in range(r0, r1) for c in range(mid)]


def _place_symmetric_band(grid: np.ndarray, r0: int, r1: int) -> bool:
    width = grid.shape[1]
    left_cells = _candidate_left_cells(r0, r1, width)
    if not left_cells:
        return False
    pair_count = random.randint(1, min(3, len(left_cells)))
    center_count = random.randint(0, 1)
    total = 2 * pair_count + center_count
    if total < 3:
        center_count = 1
        total += 1
    if total > 7:
        return False

    chosen = random.sample(left_cells, pair_count)
    for r, c in chosen:
        grid[r, c] = GREY
        grid[r, _mirror_col(width, c)] = GREY
    if center_count:
        center_candidates = [(r, width // 2) for r in range(r0, r1)]
        r, c = random.choice(center_candidates)
        grid[r, c] = GREY
    return 3 <= int(np.count_nonzero(grid[r0:r1] == GREY)) <= 7


def _place_asymmetric_band(grid: np.ndarray, r0: int, r1: int) -> bool:
    width = grid.shape[1]
    left_cells = _candidate_left_cells(r0, r1, width)
    if len(left_cells) < 2:
        return False

    pair_count = random.randint(1, min(2, len(left_cells)))
    chosen_pairs = random.sample(left_cells, pair_count)
    occupied = set()
    for r, c in chosen_pairs:
        mc = _mirror_col(width, c)
        occupied.add((r, c))
        occupied.add((r, mc))
        grid[r, c] = GREY
        grid[r, mc] = GREY

    if random.random() < 0.5:
        center_row = random.randrange(r0, r1)
        occupied.add((center_row, width // 2))
        grid[center_row, width // 2] = GREY

    unmatched_pool = []
    for r in range(r0, r1):
        for c in range(width):
            mc = _mirror_col(width, c)
            if (r, c) in occupied or (r, mc) in occupied:
                continue
            if c == width // 2:
                continue
            unmatched_pool.append((r, c))
    if not unmatched_pool:
        return False

    unmatched = random.choice(unmatched_pool)
    grid[unmatched] = GREY
    occupied.add(unmatched)

    if random.random() < 0.4:
        extra_pool = []
        for r, c in unmatched_pool:
            mc = _mirror_col(width, c)
            if (r, c) in occupied or (r, mc) in occupied:
                continue
            extra_pool.append((r, c))
        if extra_pool and int(np.count_nonzero(grid[r0:r1] == GREY)) < 7:
            extra = random.choice(extra_pool)
            grid[extra] = GREY

    band = grid[r0:r1]
    grey_count = int(np.count_nonzero(band == GREY))
    return 3 <= grey_count <= 7 and not _is_symmetric_band(grid, r0, r1)


def _wrong_component_scope_output(grid: np.ndarray) -> np.ndarray:
    filled = _fill_gaps(grid)
    passable = grid != BLUE
    out = filled.copy()
    for comp in _components(passable):
        grey_coords = [(int(r), int(c)) for r, c in comp if grid[int(r), int(c)] == GREY]
        if not grey_coords:
            continue
        symmetric = True
        for r, c in grey_coords:
            if grid[r, _mirror_col(grid.shape[1], c)] != GREY:
                symmetric = False
                break
        if symmetric:
            for r, c in grey_coords:
                out[r, c] = GREEN
    return out


def _wrong_pairwise_output(grid: np.ndarray) -> np.ndarray:
    filled = _fill_gaps(grid)
    grey = grid == GREY
    paired = grey & np.fliplr(grey)
    filled[paired] = GREEN
    return filled


def _generate_attempt() -> np.ndarray | None:
    width = random.choice(WIDTH_OPTIONS)
    separator_count = random.choice(SEPARATOR_COUNT_OPTIONS)
    band_count = separator_count + 1
    heights = _sample_band_heights(band_count)
    height, separators, bands = _band_rows_from_heights(heights)
    if height > 19:
        return None

    grid = np.zeros((height, width), dtype=int)
    for r in separators:
        grid[r, ::2] = BLUE

    symmetric_bands = set(random.sample(range(band_count), random.randint(1, band_count - 1)))
    if len(symmetric_bands) == band_count:
        return None

    for idx, (r0, r1) in enumerate(bands):
        ok = _place_symmetric_band(grid, r0, r1) if idx in symmetric_bands else _place_asymmetric_band(grid, r0, r1)
        if not ok:
            return None

    filled = _fill_gaps(grid)
    if _separator_rows(filled) != separators:
        return None

    band_syms = [_is_symmetric_band(grid, r0, r1) for r0, r1 in bands]
    if not any(band_syms) or all(band_syms):
        return None

    out = main(grid)
    if GREEN not in set(map(int, np.unique(out))):
        return None
    if GREY not in set(map(int, np.unique(out))):
        return None
    if RED not in set(map(int, np.unique(out))):
        return None

    if np.array_equal(out, _wrong_component_scope_output(grid)):
        return None
    if np.array_equal(out, _wrong_pairwise_output(grid)):
        return None

    recolored = [idx for idx, (r0, r1) in enumerate(bands) if np.any(out[r0:r1] == GREEN)]
    expected = [idx for idx, symmetric in enumerate(band_syms) if symmetric]
    if recolored != expected:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
