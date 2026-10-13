import random

import numpy as np


BG = 0
LEGEND_POOL = (1, 2, 3, 4, 7, 8)
GRID_MIN = 14
GRID_MAX = 22
GEN_RETRIES = 300


def _extract_cycle(grid: np.ndarray) -> list[int]:
    top = np.asarray(grid[0], dtype=int)
    nz = np.flatnonzero(top != BG)
    assert nz.size in (3, 4), "legend must contain 3 or 4 pixels"
    start, end = int(nz[0]), int(nz[-1])
    assert end - start + 1 == int(nz.size), "legend pixels must be contiguous"
    cycle = [int(v) for v in top[start : end + 1]]
    assert len(set(cycle)) == len(cycle), "legend colors must be distinct"
    return cycle


def _line_signatures(grid: np.ndarray, axis: int) -> list[frozenset[int]]:
    if axis == 0:
        return [frozenset(int(v) for v in np.unique(grid[r]) if int(v) != BG) for r in range(grid.shape[0])]
    return [frozenset(int(v) for v in np.unique(grid[:, c]) if int(v) != BG) for c in range(grid.shape[1])]


def _repair_signatures(signatures: list[frozenset[int]]) -> list[frozenset[int]]:
    fixed = list(signatures)
    for _ in range(2):
        changed = False
        for i in range(1, len(fixed) - 1):
            prev_sig = fixed[i - 1]
            cur_sig = fixed[i]
            next_sig = fixed[i + 1]
            if prev_sig and next_sig and prev_sig == next_sig and cur_sig and cur_sig < prev_sig:
                fixed[i] = prev_sig
                changed = True
        if not changed:
            break
    return fixed


def _bands_from_signatures(signatures: list[frozenset[int]]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = None
    current = None
    for idx, sig in enumerate(signatures):
        if not sig:
            if start is not None:
                bands.append((start, idx))
                start = None
                current = None
            continue
        if start is None:
            start = idx
            current = sig
            continue
        if sig != current:
            bands.append((start, idx))
            start = idx
            current = sig
    if start is not None:
        bands.append((start, len(signatures)))
    return bands


def _recover_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    body = np.asarray(grid[2:], dtype=int)
    row_sigs = _repair_signatures(_line_signatures(body, axis=0))
    col_sigs = _repair_signatures(_line_signatures(body, axis=1))
    row_bands = _bands_from_signatures(row_sigs)
    col_bands = _bands_from_signatures(col_sigs)
    assert row_bands and col_bands, "body must contain at least one row band and one column band"
    return row_bands, col_bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the block layout to a coarse grid and rotate each block's color
    forward in the top-row legend cycle by its hole count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3, "expected a 2D grid with legend and body"
    assert np.all(g[1] == BG), "row under the legend must be blank"

    cycle = _extract_cycle(g)
    color_to_idx = {color: idx for idx, color in enumerate(cycle)}
    body = g[2:]
    row_bands, col_bands = _recover_bands(g)

    out = np.full((len(row_bands), len(col_bands)), BG, dtype=int)
    for out_r, (r0, r1) in enumerate(row_bands):
        for out_c, (c0, c1) in enumerate(col_bands):
            block = body[r0:r1, c0:c1]
            colors = np.unique(block[block != BG])
            assert colors.size == 1, "each recovered block must be monochrome apart from holes"
            color = int(colors[0])
            assert color in color_to_idx, "block color must appear in the legend"
            holes = int(np.count_nonzero(block == BG))
            out[out_r, out_c] = cycle[(color_to_idx[color] + holes) % len(cycle)]
    return out


def _checkerboard_capacity(height: int, width: int) -> int:
    inner_h = max(0, height - 2)
    inner_w = max(0, width - 2)
    even = (inner_h * inner_w + 1) // 2
    odd = (inner_h * inner_w) // 2
    return max(even, odd)


def _sample_holes(height: int, width: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count == 0:
        return []
    groups = [[], []]
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            groups[(r + c) & 1].append((r, c))
    options = [cells for cells in groups if len(cells) >= count]
    assert options, "requested hole count exceeds non-adjacent interior capacity"
    return rng.sample(rng.choice(options), count)


def _band_sizes(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(4, 5) for _ in range(count)]


def _touch_pairs(row_seps: list[int], col_seps: list[int], rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(rows - 1):
        if row_seps[r] == 0:
            for c in range(cols):
                pairs.append(((r, c), (r + 1, c)))
    for c in range(cols - 1):
        if col_seps[c] == 0:
            for r in range(rows):
                pairs.append(((r, c), (r, c + 1)))
    return pairs


def _build_color_matrix(
    rng: random.Random,
    cycle: list[int],
    row_seps: list[int],
    col_seps: list[int],
    want_same_touch: bool,
) -> np.ndarray | None:
    rows = len(row_seps) + 1
    cols = len(col_seps) + 1
    visible_pairs = _touch_pairs(row_seps, col_seps, rows, cols)

    for _ in range(400):
        mat = np.array(
            [[rng.choice(cycle) for _ in range(cols)] for _ in range(rows)],
            dtype=int,
        )

        if want_same_touch and visible_pairs and not any(
            int(mat[a]) == int(mat[b]) for a, b in visible_pairs
        ):
            (r0, c0), (r1, c1) = rng.choice(visible_pairs)
            mat[r1, c1] = mat[r0, c0]

        if len(set(int(v) for v in mat.reshape(-1))) < min(len(cycle), 3):
            continue

        if want_same_touch:
            if visible_pairs and not any(int(mat[a]) == int(mat[b]) for a, b in visible_pairs):
                continue
        else:
            if any(int(mat[a]) == int(mat[b]) for a, b in visible_pairs):
                continue

        row_sigs = [frozenset(int(v) for v in mat[r]) for r in range(rows)]
        col_sigs = [frozenset(int(v) for v in mat[:, c]) for c in range(cols)]
        ok = True
        for r in range(rows - 1):
            if row_seps[r] == 0 and row_sigs[r] == row_sigs[r + 1]:
                ok = False
                break
        if not ok:
            continue
        for c in range(cols - 1):
            if col_seps[c] == 0 and col_sigs[c] == col_sigs[c + 1]:
                ok = False
                break
        if ok:
            return mat
    return None


def _axis_ranges(sizes: list[int], seps: list[int]) -> tuple[list[tuple[int, int]], int]:
    ranges: list[tuple[int, int]] = []
    pos = 0
    for idx, size in enumerate(sizes):
        ranges.append((pos, pos + size))
        pos += size
        if idx < len(seps):
            pos += seps[idx]
    return ranges, pos


def _hole_counts(
    rng: random.Random,
    row_sizes: list[int],
    col_sizes: list[int],
    cycle_len: int,
) -> np.ndarray:
    rows = len(row_sizes)
    cols = len(col_sizes)
    counts = np.zeros((rows, cols), dtype=int)
    max_counts: list[tuple[int, int, int]] = []
    for r, height in enumerate(row_sizes):
        for c, width in enumerate(col_sizes):
            cap = _checkerboard_capacity(height, width)
            max_k = min(cycle_len - 1, cap)
            max_counts.append((r, c, max_k))
            counts[r, c] = rng.randint(0, max_k)

    positives = [(r, c, max_k) for r, c, max_k in max_counts if max_k > 0]
    if positives and not np.any(counts):
        r, c, max_k = rng.choice(positives)
        counts[r, c] = rng.randint(1, max_k)

    high_targets = [(r, c) for r, c, max_k in max_counts if max_k >= cycle_len - 1]
    if cycle_len == 4 and high_targets and rng.random() < 0.5:
        r, c = rng.choice(high_targets)
        counts[r, c] = 3

    return counts


def _expected_output(cycle: list[int], colors: np.ndarray, hole_counts: np.ndarray) -> np.ndarray:
    color_to_idx = {color: idx for idx, color in enumerate(cycle)}
    out = np.zeros(colors.shape, dtype=int)
    for r in range(colors.shape[0]):
        for c in range(colors.shape[1]):
            color = int(colors[r, c])
            k = int(hole_counts[r, c])
            out[r, c] = cycle[(color_to_idx[color] + k) % len(cycle)]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cycle_len = 3 if rng.random() < 0.5 else 4
    cycle = rng.sample(list(LEGEND_POOL), cycle_len)

    rows = rng.choice((2, 3, 3, 4))
    cols = rng.choice((2, 3, 3, 4))
    row_sizes = _band_sizes(rng, rows)
    col_sizes = _band_sizes(rng, cols)

    if cycle_len == 4:
        row_sizes[rng.randrange(rows)] = 5
        col_sizes[rng.randrange(cols)] = 5

    row_seps = [rng.randint(0, 1) for _ in range(rows - 1)]
    col_seps = [rng.randint(0, 1) for _ in range(cols - 1)]

    want_same_touch = rng.random() < 0.5
    if want_same_touch and row_seps + col_seps and all(v == 1 for v in row_seps + col_seps):
        if row_seps and col_seps:
            if rng.random() < 0.5:
                row_seps[rng.randrange(len(row_seps))] = 0
            else:
                col_seps[rng.randrange(len(col_seps))] = 0
        elif row_seps:
            row_seps[rng.randrange(len(row_seps))] = 0
        else:
            col_seps[rng.randrange(len(col_seps))] = 0

    row_ranges, body_h = _axis_ranges(row_sizes, row_seps)
    col_ranges, body_w = _axis_ranges(col_sizes, col_seps)

    if body_h > GRID_MAX - 2 or body_w > GRID_MAX:
        return None

    height_min = max(GRID_MIN, 2 + body_h)
    width_min = max(GRID_MIN, body_w, cycle_len)
    if height_min > GRID_MAX or width_min > GRID_MAX:
        return None

    color_matrix = _build_color_matrix(rng, cycle, row_seps, col_seps, want_same_touch)
    if color_matrix is None:
        return None

    holes = _hole_counts(rng, row_sizes, col_sizes, cycle_len)
    expected = _expected_output(cycle, color_matrix, holes)

    height = rng.randint(height_min, GRID_MAX)
    width = rng.randint(width_min, GRID_MAX)
    body_left = rng.randint(0, width - body_w)
    legend_left = rng.randint(0, width - cycle_len)

    grid = np.full((height, width), BG, dtype=int)
    grid[0, legend_left : legend_left + cycle_len] = np.array(cycle, dtype=int)

    for r, (r0, r1) in enumerate(row_ranges):
        for c, (c0, c1) in enumerate(col_ranges):
            color = int(color_matrix[r, c])
            abs_r0 = 2 + r0
            abs_r1 = 2 + r1
            abs_c0 = body_left + c0
            abs_c1 = body_left + c1
            grid[abs_r0:abs_r1, abs_c0:abs_c1] = color
            for dr, dc in _sample_holes(r1 - r0, c1 - c0, int(holes[r, c]), rng):
                grid[abs_r0 + dr, abs_c0 + dc] = BG

    try:
        solved = main(grid)
        recovered_rows, recovered_cols = _recover_bands(grid)
    except AssertionError:
        return None

    if len(recovered_rows) != rows or len(recovered_cols) != cols:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
