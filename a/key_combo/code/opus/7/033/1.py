from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
TEAL = 8
PALETTE = (1, 2, 3, 4, 7)
MAX_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bands_from_separator(flags: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = None
    for idx, is_sep in enumerate(map(bool, flags)):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            bands.append((int(start), int(idx)))
            start = None
    if start is not None:
        bands.append((int(start), int(len(flags))))
    return bands


def _majority_color(block: np.ndarray) -> int:
    values = [int(v) for v in np.asarray(block).ravel()]
    counts = Counter(values)
    if counts.keys() == {BG}:
        return BG
    ranked = counts.most_common()
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _partition_spans(length: int, parts: int) -> list[tuple[int, int]]:
    base = int(length) // int(parts)
    extra = int(length) - base * int(parts)
    spans = []
    start = 0
    for idx in range(int(parts)):
        stop = start + base + (extra if idx == parts - 1 else 0)
        spans.append((int(start), int(stop)))
        start = stop
    return spans


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _offset_candidates(
    zero_positions: set[tuple[int, int]],
    hole_positions: set[tuple[int, int]],
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    candidates = []
    for dr in range(-(rows - 1), rows):
        for dc in range(-(cols - 1), cols):
            ok = True
            for hr, hc in hole_positions:
                zr = hr + dr
                zc = hc + dc
                if not (0 <= zr < rows and 0 <= zc < cols and (zr, zc) in zero_positions):
                    ok = False
                    break
            if ok:
                candidates.append((int(dr), int(dc)))
    return candidates


def _extract_lattice(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    grey_box = bbox(grid == GREY)
    assert grey_box is not None, "expected a grey-separated lattice"
    r0, c0, r1, c1 = grey_box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    sep_rows = np.all(patch == GREY, axis=1)
    sep_cols = np.all(patch == GREY, axis=0)
    row_bands = _bands_from_separator(sep_rows)
    col_bands = _bands_from_separator(sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a multi-cell lattice"

    matrix = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    for br, (rr0, rr1) in enumerate(row_bands):
        for bc, (cc0, cc1) in enumerate(col_bands):
            matrix[br, bc] = _majority_color(patch[rr0:rr1, cc0:cc1])
    return grey_box, matrix


def _shape_spans(grid: np.ndarray, rows: int, cols: int) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    shape_box = bbox(grid == TEAL)
    assert shape_box is not None, "expected one teal shape"
    r0, c0, r1, c1 = shape_box
    row_spans = _partition_spans(r1 - r0 + 1, rows)
    col_spans = _partition_spans(c1 - c0 + 1, cols)
    return shape_box, row_spans, col_spans


def _hole_positions(
    grid: np.ndarray,
    shape_box: tuple[int, int, int, int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = shape_box
    holes = set()
    for sr, (rr0, rr1) in enumerate(row_spans):
        for sc, (cc0, cc1) in enumerate(col_spans):
            region = grid[r0 + rr0 : r0 + rr1, c0 + cc0 : c0 + cc1]
            if not np.any(region == TEAL):
                holes.add((int(sr), int(sc)))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a partitioned teal shape using the majority-color summary of a grey-separated lattice.

    Input structure:
    - The canvas has a black background, a grey-separated color lattice on the left, and one teal foreground shape on the right.
    - The lattice is divided by complete grey rows and columns; each interior block is either fully black or has one unique majority color.
    - The teal shape's bounding box can be partitioned into the same row-by-column count as the lattice, and some of those subregions are already all-background holes.

    Rule:
    - Collapse the lattice to an R x C color matrix by taking the unique majority color of each block and recording fully black blocks as 0.
    - Partition the teal shape bbox into the same R x C subregions and locate the existing hole subregions.
    - Find the unique row/column offset whose translated hole pattern fits inside the lattice's 0-cells, then use that alignment to recolor every non-hole subregion: teal pixels become the aligned majority color, or are erased to black when the aligned matrix cell is 0.
    - Existing holes and the lattice area remain unchanged.

    Key insight:
    - The lattice's empty blocks act as both missing-color entries and as anchors for pattern matching, so majority-vote compression must happen before the teal shape can be aligned and repainted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _lattice_box, matrix = _extract_lattice(g)
    rows, cols = map(int, matrix.shape)

    shape_box, row_spans, col_spans = _shape_spans(g, rows, cols)
    holes = _hole_positions(g, shape_box, row_spans, col_spans)
    zero_positions = {
        (int(r), int(c))
        for r, c in np.argwhere(matrix == BG)
    }
    assert holes, "expected at least one existing hole"
    assert len(zero_positions) > len(holes), "expected additional empty lattice cells"

    candidates = _offset_candidates(zero_positions, holes, rows, cols)
    assert len(candidates) == 1, "expected a unique lattice/shape alignment"
    dr, dc = candidates[0]

    out = g.copy()
    sr0, sc0, _sr1, _sc1 = shape_box
    for r in range(rows):
        for c in range(cols):
            rr0, rr1 = row_spans[r]
            cc0, cc1 = col_spans[c]
            region = g[sr0 + rr0 : sr0 + rr1, sc0 + cc0 : sc0 + cc1]
            if not np.any(region == TEAL):
                continue
            mr = r + dr
            mc = c + dc
            if not (0 <= mr < rows and 0 <= mc < cols):
                continue
            color = int(matrix[mr, mc])
            dst = out[sr0 + rr0 : sr0 + rr1, sc0 + cc0 : sc0 + cc1]
            dst[region == TEAL] = color

    return out


def _choose_zero_pattern(rows: int, cols: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    interior = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]
    if len(interior) < 2:
        return None

    for _ in range(200):
        holes = set(random.sample(interior, 2))
        extra = random.choice([cell for cell in all_cells if cell not in holes])
        zeros = set(holes)
        zeros.add(extra)

        occupied_input = np.ones((rows, cols), dtype=bool)
        for r, c in holes:
            occupied_input[r, c] = False
        if not _is_connected(occupied_input):
            continue

        occupied_output = np.ones((rows, cols), dtype=bool)
        for r, c in zeros:
            occupied_output[r, c] = False
        if not _is_connected(occupied_output):
            continue

        if _offset_candidates(zeros, holes, rows, cols) != [(0, 0)]:
            continue
        return zeros, holes
    return None


def _make_majority_matrix(rows: int, cols: int) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]] | None:
    pattern = _choose_zero_pattern(rows, cols)
    if pattern is None:
        return None
    zeros, holes = pattern

    matrix = np.zeros((rows, cols), dtype=int)
    used_nonzero = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in zeros:
                continue
            color = random.choice(PALETTE)
            matrix[r, c] = int(color)
            used_nonzero.add(int(color))
    if len(used_nonzero) < 2:
        return None
    return matrix, zeros, holes


def _noisy_block(height: int, width: int, majority: int) -> np.ndarray:
    block = np.full((height, width), int(majority), dtype=int)
    cells = height * width
    max_noise = min(cells // 3, 4)
    noise_count = random.randint(1, max_noise) if max_noise else 0
    if noise_count:
        choices = [c for c in PALETTE if c != majority]
        picks = random.sample(range(cells), noise_count)
        for idx in picks:
            rr, cc = divmod(idx, width)
            block[rr, cc] = random.choice(choices)
    return block


def _render_lattice(grid: np.ndarray, top: int, left: int, matrix: np.ndarray, block_h: int, block_w: int) -> tuple[int, int]:
    rows, cols = map(int, matrix.shape)
    total_h = rows * block_h + rows + 1
    total_w = cols * block_w + cols + 1
    grid[top : top + total_h, left : left + total_w] = GREY
    for r in range(rows):
        for c in range(cols):
            rr0 = top + 1 + r * (block_h + 1)
            cc0 = left + 1 + c * (block_w + 1)
            color = int(matrix[r, c])
            if color == BG:
                grid[rr0 : rr0 + block_h, cc0 : cc0 + block_w] = BG
                continue
            grid[rr0 : rr0 + block_h, cc0 : cc0 + block_w] = _noisy_block(block_h, block_w, color)
    return total_h, total_w


def _render_shape(
    grid: np.ndarray,
    top: int,
    left: int,
    rows: int,
    cols: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    holes: set[tuple[int, int]],
) -> tuple[int, int]:
    shape_h = row_spans[-1][1]
    shape_w = col_spans[-1][1]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in holes:
                continue
            rr0, rr1 = row_spans[r]
            cc0, cc1 = col_spans[c]
            grid[top + rr0 : top + rr1, left + cc0 : left + cc1] = TEAL
    return shape_h, shape_w


def _expected_output(
    grid: np.ndarray,
    shape_top: int,
    shape_left: int,
    matrix: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    rows, cols = map(int, matrix.shape)
    for r in range(rows):
        for c in range(cols):
            rr0, rr1 = row_spans[r]
            cc0, cc1 = col_spans[c]
            region = grid[shape_top + rr0 : shape_top + rr1, shape_left + cc0 : shape_left + cc1]
            if not np.any(region == TEAL):
                continue
            dst = out[shape_top + rr0 : shape_top + rr1, shape_left + cc0 : shape_left + cc1]
            dst[region == TEAL] = int(matrix[r, c])
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice([(3, 4), (4, 3), (4, 4)])
    matrix_data = _make_majority_matrix(rows, cols)
    if matrix_data is None:
        return None
    matrix, _zeros, holes = matrix_data

    block_h = random.choice([3, 4])
    block_w = random.choice([3, 4])
    lattice_h = rows * block_h + rows + 1
    lattice_w = cols * block_w + cols + 1

    base_shape_h = random.choice([2, 3])
    base_shape_w = random.choice([2, 3])
    extra_h = random.choice([0, 1])
    extra_w = random.choice([0, 1])
    shape_h = rows * base_shape_h + extra_h
    shape_w = cols * base_shape_w + extra_w

    gap = random.randint(2, 4)
    width = lattice_w + gap + shape_w + random.randint(2, 4)
    height = max(lattice_h, shape_h) + random.randint(2, 5)
    if width > 30 or height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    lattice_top = random.randint(1, height - lattice_h - 1)
    lattice_left = random.randint(1, 2)
    _render_lattice(grid, lattice_top, lattice_left, matrix, block_h, block_w)

    row_spans = _partition_spans(shape_h, rows)
    col_spans = _partition_spans(shape_w, cols)
    shape_top = random.randint(1, height - shape_h - 1)
    shape_left = lattice_left + lattice_w + gap
    if shape_left + shape_w > width:
        return None
    _render_shape(grid, shape_top, shape_left, rows, cols, row_spans, col_spans, holes)

    if not _is_connected(grid == TEAL):
        return None

    expected = _expected_output(grid, shape_top, shape_left, matrix, row_spans, col_spans)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-1 input")
