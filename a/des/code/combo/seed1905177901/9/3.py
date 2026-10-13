import numpy as np


BACKGROUND = 0
CELL_SIDE = 3
SEPARATOR_THICKNESSES = (1, 2)
SEPARATOR_COLOR = 7
FOREGROUND_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
GRID_DIM_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 300


def _compress_runs(indices: list[int]) -> list[tuple[int, int]]:
    assert indices, "expected at least one separator index"
    runs: list[tuple[int, int]] = []
    start = end = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == end + 1:
            end = idx
        else:
            runs.append((start, end))
            start = end = idx
    runs.append((start, end))
    return runs


def _regular_intervals(length: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    prev_end = -1
    for start, end in runs:
        interval = (prev_end + 1, start)
        assert interval[0] < interval[1], "separator runs must leave non-empty cells"
        intervals.append(interval)
        prev_end = end
    interval = (prev_end + 1, int(length))
    assert interval[0] < interval[1], "separator runs must leave a final cell"
    intervals.append(interval)

    sizes = [b - a for a, b in intervals]
    assert len(set(sizes)) == 1, "all cell intervals must have the same size"
    return intervals


def _separator_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not rows or not cols:
            continue
        try:
            row_intervals = _regular_intervals(h, _compress_runs(rows))
            col_intervals = _regular_intervals(w, _compress_runs(cols))
        except AssertionError:
            continue
        if len(row_intervals) < 2 or len(col_intervals) < 2:
            continue
        row_size = row_intervals[0][1] - row_intervals[0][0]
        col_size = col_intervals[0][1] - col_intervals[0][0]
        if row_size != col_size:
            continue
        candidates.append((color, row_intervals, col_intervals))

    assert len(candidates) == 1, "expected exactly one regular separator lattice"
    return candidates[0]


def _nearest_indices(src_size: int, dst_size: int) -> np.ndarray:
    coords = ((np.arange(dst_size, dtype=float) + 0.5) * src_size / dst_size).astype(int)
    return np.clip(coords, 0, src_size - 1)


def _resize_nearest(block: np.ndarray, target_size: int) -> np.ndarray:
    src_h, src_w = block.shape
    rows = _nearest_indices(src_h, target_size)
    cols = _nearest_indices(src_w, target_size)
    return block[rows[:, None], cols[None, :]]


def _pack_row_strips(strips: list[list[np.ndarray]], bg: int) -> np.ndarray:
    row_heights = [max(block.shape[0] for block in row) for row in strips]
    row_widths = [sum(block.shape[1] for block in row) for row in strips]

    out = np.full((sum(row_heights), max(row_widths)), bg, dtype=int)
    top = 0
    for row_blocks, row_h in zip(strips, row_heights):
        left = 0
        for block in row_blocks:
            h, w = block.shape
            out[top : top + h, left : left + w] = block
            left += w
        top += row_h
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the foreground pixels in each separator-defined cell, resize that cell's
    full contents to an N x N square by nearest-neighbor sampling, then pack the
    scaled cells row by row without separator gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, row_intervals, col_intervals = _separator_structure(g)
    strips: list[list[np.ndarray]] = []

    for r0, r1 in row_intervals:
        row_blocks: list[np.ndarray] = []
        for c0, c1 in col_intervals:
            cell = g[r0:r1, c0:c1]
            colors = {int(x) for x in np.unique(cell) if int(x) not in {BACKGROUND, separator}}
            assert len(colors) == 1, "each cell must contain exactly one foreground color"
            pixel_count = int(np.count_nonzero((cell != BACKGROUND) & (cell != separator)))
            assert 1 <= pixel_count <= 9, "foreground pixel count must be in 1..9"
            row_blocks.append(_resize_nearest(cell, pixel_count))
        strips.append(row_blocks)

    return _pack_row_strips(strips, BACKGROUND)


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    stack = [tuple(map(int, points[0]))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                cell = (nr, nc)
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == len(points)


def _build_mask_pool() -> dict[int, list[np.ndarray]]:
    pool = {count: [] for count in range(1, CELL_SIDE * CELL_SIDE + 1)}
    for bits in range(1, 1 << (CELL_SIDE * CELL_SIDE)):
        count = bits.bit_count()
        mask = np.zeros((CELL_SIDE, CELL_SIDE), dtype=bool)
        for idx in range(CELL_SIDE * CELL_SIDE):
            if bits & (1 << idx):
                mask[idx // CELL_SIDE, idx % CELL_SIDE] = True
        if not _is_connected(mask):
            continue

        preview = np.zeros((CELL_SIDE, CELL_SIDE), dtype=int)
        preview[mask] = 1
        if not np.any(_resize_nearest(preview, count)):
            continue
        pool[count].append(mask)
    return pool


MASK_POOL = _build_mask_pool()


def _sample_counts(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray | None:
    total = rows * cols
    for _ in range(100):
        flat = [
            int(rng.choice((1, 2))),
            int(rng.choice((3, 4, 5, 6, 7))),
            int(rng.choice((8, 9))),
        ]
        while len(flat) < total:
            flat.append(int(rng.integers(1, 10)))
        rng.shuffle(flat)
        counts = np.array(flat[:total], dtype=int).reshape(rows, cols)
        if len(np.unique(counts)) < 3:
            continue
        if rows > 1:
            row_sums = {int(x) for x in counts.sum(axis=1)}
            row_maxes = {int(x) for x in counts.max(axis=1)}
            if len(row_sums) < 2 or len(row_maxes) < 2:
                continue
        if not any(len(set(map(int, row))) > 1 for row in counts):
            continue
        return counts
    return None


def _sample_colors(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    colors = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            forbidden = set()
            if r > 0:
                forbidden.add(int(colors[r - 1, c]))
            if c > 0:
                forbidden.add(int(colors[r, c - 1]))
            choices = [color for color in FOREGROUND_COLORS if color not in forbidden]
            colors[r, c] = int(rng.choice(choices))
    return colors


def _cell_origin(index: int, cell_side: int, sep: int) -> int:
    return index * (cell_side + sep)


def _render_input(counts: np.ndarray, colors: np.ndarray, sep: int, rng: np.random.Generator) -> np.ndarray:
    rows, cols = counts.shape
    height = rows * CELL_SIDE + (rows - 1) * sep
    width = cols * CELL_SIDE + (cols - 1) * sep
    grid = np.full((height, width), BACKGROUND, dtype=int)

    for r in range(1, rows):
        start = _cell_origin(r - 1, CELL_SIDE, sep) + CELL_SIDE
        grid[start : start + sep, :] = SEPARATOR_COLOR
    for c in range(1, cols):
        start = _cell_origin(c - 1, CELL_SIDE, sep) + CELL_SIDE
        grid[:, start : start + sep] = SEPARATOR_COLOR

    for r in range(rows):
        for c in range(cols):
            cell = np.full((CELL_SIDE, CELL_SIDE), BACKGROUND, dtype=int)
            mask = MASK_POOL[int(counts[r, c])][int(rng.integers(len(MASK_POOL[int(counts[r, c])])))]
            cell[mask] = int(colors[r, c])
            r0 = _cell_origin(r, CELL_SIDE, sep)
            c0 = _cell_origin(c, CELL_SIDE, sep)
            grid[r0 : r0 + CELL_SIDE, c0 : c0 + CELL_SIDE] = cell

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1] + 1))
    cols = int(rng.integers(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1] + 1))
    sep = int(rng.choice(SEPARATOR_THICKNESSES))

    counts = _sample_counts(rows, cols, rng)
    if counts is None:
        return None

    colors = _sample_colors(rows, cols, rng)
    if len(np.unique(colors)) < 3:
        return None

    grid = _render_input(counts, colors, sep, rng)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.size == 0:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
