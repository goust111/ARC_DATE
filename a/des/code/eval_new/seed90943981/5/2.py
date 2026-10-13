import itertools
import random

import numpy as np


BG = 8
GRID = 5
MARK = 9
MOVABLE_COLORS = (0, 1, 2, 3, 4, 6, 7)

REGION_ROW_RANGE = (2, 4)
REGION_COL_RANGE = (2, 4)
REGION_SIZE_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400
REGION_FILL_RETRIES = 80


def _scan_order(length: int, toward_start: bool) -> range:
    if toward_start:
        return range(length)
    return range(length - 1, -1, -1)


def _pack_region(region: np.ndarray, to_top: bool, to_left: bool) -> np.ndarray:
    h, w = region.shape
    if h == 0 or w == 0:
        return region.copy()

    occupied = region != BG
    if not np.any(occupied):
        return region.copy()

    colors: list[int] = []
    for r in _scan_order(h, to_top):
        for c in _scan_order(w, to_left):
            value = int(region[r, c])
            if value != BG:
                colors.append(value)

    horiz = np.zeros_like(occupied, dtype=bool)
    for r in range(h):
        count = int(np.count_nonzero(occupied[r]))
        if count == 0:
            continue
        if to_left:
            horiz[r, :count] = True
        else:
            horiz[r, w - count :] = True

    final_mask = np.zeros_like(occupied, dtype=bool)
    for c in range(w):
        count = int(np.count_nonzero(horiz[:, c]))
        if count == 0:
            continue
        if to_top:
            final_mask[:count, c] = True
        else:
            final_mask[h - count :, c] = True

    out = np.full(region.shape, BG, dtype=int)
    targets: list[tuple[int, int]] = []
    for r in _scan_order(h, to_top):
        for c in _scan_order(w, to_left):
            if final_mask[r, c]:
                targets.append((r, c))

    assert len(colors) == len(targets), "packed region must preserve pixel count"
    for color, (r, c) in zip(colors, targets):
        out[r, c] = color
    return out


def _line_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [
            r
            for r in range(grid.shape[0])
            if np.all((grid[r] == GRID) | (grid[r] == MARK))
        ]
    return [
        c
        for c in range(grid.shape[1])
        if np.all((grid[:, c] == GRID) | (grid[:, c] == MARK))
    ]


def _gravity_corner(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> tuple[bool, bool]:
    corners = [
        (r0, c0, True, True),
        (r0, c1, True, False),
        (r1, c0, False, True),
        (r1, c1, False, False),
    ]
    marked = [(to_top, to_left) for r, c, to_top, to_left in corners if int(grid[r, c]) == MARK]
    assert len(marked) == 1, "each region must have exactly one maroon corner marker"
    return marked[0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    line_rows = _line_indices(g, axis=0)
    line_cols = _line_indices(g, axis=1)
    assert len(line_rows) >= 2 and len(line_cols) >= 2, "expected a grey grid lattice"
    assert line_rows[0] == 0 and line_rows[-1] == g.shape[0] - 1, "grid lattice must include top/bottom borders"
    assert line_cols[0] == 0 and line_cols[-1] == g.shape[1] - 1, "grid lattice must include left/right borders"

    out = g.copy()
    for rr in range(len(line_rows) - 1):
        r0, r1 = line_rows[rr], line_rows[rr + 1]
        if r1 - r0 <= 1:
            continue
        for cc in range(len(line_cols) - 1):
            c0, c1 = line_cols[cc], line_cols[cc + 1]
            if c1 - c0 <= 1:
                continue

            to_top, to_left = _gravity_corner(g, r0, r1, c0, c1)
            region = g[r0 + 1 : r1, c0 + 1 : c1]
            packed = _pack_region(region, to_top=to_top, to_left=to_left)
            out[r0 + 1 : r1, c0 + 1 : c1] = packed
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored pixels inside each grey-grid region toward the unique maroon-marked corner.

    Input structure:
    - The background is azure, and full grey rows/columns form a rectangular lattice of one-pixel-wide grid lines.
    - Each rectangular region between adjacent grey lines contains scattered colored pixels on azure cells.
    - Some grey line intersections are replaced by maroon pixels; for every region, exactly one of its four corner intersections is maroon.

    Rule:
    - For each region, read the unique maroon corner among its four lattice corners; that corner is the region's gravity source.
    - Repack the region's colored pixels by first compressing each row horizontally toward the gravity side, then compressing each column vertically toward the gravity side.
    - Preserve every pixel's color, but assign colors to the packed cells in the original scan order: rows are scanned from the gravity side first, and within each row columns are scanned from the gravity side first.
    - Leave the grey lattice and maroon intersection markers unchanged.

    Key insight:
    - The hidden maroon pixels on grey intersections choose a different gravity corner for each rectangle, and the final arrangement comes from a two-stage horizontal-then-vertical compaction plus direction-aware color ordering.
    """
    return _solve(grid)


def _row_compatibility(n_region_cols: int) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    rows = list(itertools.product((0, 1), repeat=n_region_cols + 1))
    compat: dict[tuple[int, ...], list[tuple[int, ...]]] = {row: [] for row in rows}
    for upper in rows:
        for lower in rows:
            ok = True
            for c in range(n_region_cols):
                if upper[c] + upper[c + 1] + lower[c] + lower[c + 1] != 1:
                    ok = False
                    break
            if ok:
                compat[upper].append(lower)
    return compat


def _cell_directions(marker_matrix: np.ndarray) -> list[tuple[bool, bool]]:
    dirs: list[tuple[bool, bool]] = []
    for r in range(marker_matrix.shape[0] - 1):
        for c in range(marker_matrix.shape[1] - 1):
            block = marker_matrix[r : r + 2, c : c + 2]
            total = int(block.sum())
            assert total == 1, "every region must see exactly one marker"
            if block[0, 0]:
                dirs.append((True, True))
            elif block[0, 1]:
                dirs.append((True, False))
            elif block[1, 0]:
                dirs.append((False, True))
            else:
                dirs.append((False, False))
    return dirs


def _sample_marker_matrix(n_region_rows: int, n_region_cols: int, rng: random.Random) -> np.ndarray | None:
    compat = _row_compatibility(n_region_cols)
    row_patterns = list(compat.keys())

    for _ in range(200):
        rows = [rng.choice(row_patterns)]
        ok = True
        for _step in range(n_region_rows):
            next_rows = compat[rows[-1]]
            if not next_rows:
                ok = False
                break
            rows.append(rng.choice(next_rows))
        if not ok:
            continue

        marker_matrix = np.array(rows, dtype=int)
        if rng.random() < 0.5:
            marker_matrix = np.flipud(marker_matrix)
        if rng.random() < 0.5:
            marker_matrix = np.fliplr(marker_matrix)

        direction_set = set(_cell_directions(marker_matrix))
        if len(direction_set) < min(4, n_region_rows * n_region_cols):
            continue
        return marker_matrix

    return None


def _sample_region_sizes(count: int, rng: random.Random) -> list[int] | None:
    low, high = REGION_SIZE_RANGE
    sizes = [rng.randint(low, high) for _ in range(count)]
    total = sum(sizes) + count + 1
    if total > 30:
        return None
    return sizes


def _line_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _random_region(rng: random.Random, h: int, w: int, to_top: bool, to_left: bool) -> np.ndarray | None:
    area = h * w
    coords = [(r, c) for r in range(h) for c in range(w)]

    min_pixels = 1
    max_pixels = min(area - 1, 7)
    if max_pixels < min_pixels:
        return None

    for _ in range(REGION_FILL_RETRIES):
        count = rng.randint(min_pixels, max_pixels)
        positions = rng.sample(coords, count)

        palette_size = rng.randint(1, min(4, len(MOVABLE_COLORS), count))
        palette = rng.sample(MOVABLE_COLORS, palette_size)

        region = np.full((h, w), BG, dtype=int)
        for r, c in positions:
            region[r, c] = rng.choice(palette)

        packed = _pack_region(region, to_top=to_top, to_left=to_left)
        if np.array_equal(region, packed):
            continue
        return region

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_region_rows = rng.randint(*REGION_ROW_RANGE)
    n_region_cols = rng.randint(*REGION_COL_RANGE)

    row_sizes = _sample_region_sizes(n_region_rows, rng)
    col_sizes = _sample_region_sizes(n_region_cols, rng)
    if row_sizes is None or col_sizes is None:
        return None

    marker_matrix = _sample_marker_matrix(n_region_rows, n_region_cols, rng)
    if marker_matrix is None:
        return None

    line_rows = _line_positions(row_sizes)
    line_cols = _line_positions(col_sizes)
    height = line_rows[-1] + 1
    width = line_cols[-1] + 1
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[line_rows, :] = GRID
    grid[:, line_cols] = GRID

    for mr in range(marker_matrix.shape[0]):
        for mc in range(marker_matrix.shape[1]):
            if marker_matrix[mr, mc]:
                grid[line_rows[mr], line_cols[mc]] = MARK

    changed_regions = 0
    for rr in range(n_region_rows):
        for cc in range(n_region_cols):
            to_top, to_left = _cell_directions(marker_matrix)[rr * n_region_cols + cc]
            region_h = row_sizes[rr]
            region_w = col_sizes[cc]
            region = _random_region(rng, region_h, region_w, to_top=to_top, to_left=to_left)
            if region is None:
                return None
            packed = _pack_region(region, to_top=to_top, to_left=to_left)
            if not np.array_equal(region, packed):
                changed_regions += 1
            r0 = line_rows[rr] + 1
            c0 = line_cols[cc] + 1
            grid[r0 : r0 + region_h, c0 : c0 + region_w] = region

    if changed_regions != n_region_rows * n_region_cols:
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-grid input")
