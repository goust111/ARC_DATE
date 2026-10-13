from __future__ import annotations

from collections import Counter
import random

import numpy as np


GRIDLINE = 6
BG = 0
STRIPE_COLORS = (1, 2, 3, 4)
BASE_COLORS = (5, 7, 8, 9)
GEN_ATTEMPTS = 500
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _uniform_line_indices(grid: np.ndarray, axis: int, color: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == int(color)))]
    return [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == int(color)))]


def _gap_slices(lines: list[int]) -> list[tuple[int, int]]:
    return [(a + 1, b) for a, b in zip(lines, lines[1:]) if b > a + 1]


def _build_lattice(cells: list[list[np.ndarray]], bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_h, cell_w = map(int, cells[0][0].shape)
    out = np.full((rows * cell_h + rows + 1, cols * cell_w + cols + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        out[r * (cell_h + 1), :] = GRIDLINE
    for c in range(cols + 1):
        out[:, c * (cell_w + 1)] = GRIDLINE
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            r0 = 1 + r * (cell_h + 1)
            c0 = 1 + c * (cell_w + 1)
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell
    return out


def _extract_stripe(odd_cell: np.ndarray, majority_cell: np.ndarray, bg: int) -> dict[str, int]:
    diff = odd_cell != majority_cell
    candidates: list[dict[str, int]] = []
    h, w = map(int, odd_cell.shape)
    for r in range(h):
        c = 0
        while c < w:
            if diff[r, c] and int(odd_cell[r, c]) != int(bg):
                color = int(odd_cell[r, c])
                left = c
                while c < w and diff[r, c] and int(odd_cell[r, c]) == color:
                    c += 1
                length = c - left
                if length >= 2:
                    candidates.append({"v": r, "left": left, "length": length, "color": color})
            else:
                c += 1
    assert candidates, "expected one added stripe inside the odd cell"
    best = max(candidates, key=lambda item: item["length"])
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the odd stripe from each lattice row, then align every stripe to the median row and median length reference.

    Input structure:
    - The grid is partitioned by magenta gridlines into equal rectangular cells on a black background.
    - Within each lattice row, all but one cell share the same interior sprite; the remaining odd cell adds one colored horizontal stripe.
    - Odd-cell stripes may differ in color, vertical offset, length, and horizontal start position.

    Rule:
    - Split the lattice into cells and, for each row, find the cell whose full interior pattern appears exactly once.
    - Compare that odd cell against the row's repeated majority cell to isolate the added stripe and record its color, vertical offset, left start, and length.
    - Choose the stripe whose vertical offset is the median of all extracted offsets as the reference; keep its offset and length.
    - Build a single-column lattice with the same cell size, placing one stripe per row at the reference offset and reference length while preserving each stripe's own color and left start.

    Key insight:
    - The stripe is not identified in isolation; it is the extra structure revealed by subtracting the odd cell from its row's repeated template, and only after extracting all rows can the median offset reference be chosen.
    """
    g = np.asarray(grid, dtype=int)
    row_lines = _uniform_line_indices(g, axis=0, color=GRIDLINE)
    col_lines = _uniform_line_indices(g, axis=1, color=GRIDLINE)
    assert len(row_lines) >= 4 and len(col_lines) >= 4

    r_slices = _gap_slices(row_lines)
    c_slices = _gap_slices(col_lines)
    assert r_slices and c_slices

    heights = {rb - ra for ra, rb in r_slices}
    widths = {cb - ca for ca, cb in c_slices}
    assert len(heights) == 1 and len(widths) == 1
    cell_h = next(iter(heights))
    cell_w = next(iter(widths))

    non_grid = g[g != GRIDLINE]
    bg = int(Counter(map(int, non_grid.flat)).most_common(1)[0][0])

    stripes: list[dict[str, int]] = []
    for ra, rb in r_slices:
        cells = [g[ra:rb, ca:cb] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        counts = Counter(sigs)
        odd_indices = [idx for idx, sig in enumerate(sigs) if counts[sig] == 1]
        assert len(odd_indices) == 1
        majority_sig = next(sig for sig, count in counts.items() if count > 1)
        odd_cell = cells[odd_indices[0]]
        majority_cell = next(cell for cell, sig in zip(cells, sigs) if sig == majority_sig)
        stripes.append(_extract_stripe(odd_cell, majority_cell, bg))

    ref = sorted(stripes, key=lambda item: item["v"])[len(stripes) // 2]
    v_ref = int(ref["v"])
    l_ref = int(ref["length"])

    out_cells: list[list[np.ndarray]] = []
    for stripe in stripes:
        cell = np.full((cell_h, cell_w), int(bg), dtype=int)
        left = int(stripe["left"])
        assert left + l_ref <= cell_w
        cell[v_ref, left : left + l_ref] = int(stripe["color"])
        out_cells.append([cell])
    return _build_lattice(out_cells, bg)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _make_base_pattern(cell_h: int, cell_w: int, forbidden_row: int, color: int) -> np.ndarray | None:
    allowed = [(r, c) for r in range(cell_h) for c in range(cell_w) if r != forbidden_row]
    if len(allowed) < 3:
        return None
    for _ in range(80):
        target = random.randint(3, min(6, len(allowed)))
        seed = random.choice(allowed)
        cells = {seed}
        frontier = {
            nbr
            for nbr in _neighbors4(seed[0], seed[1], cell_h, cell_w)
            if nbr in allowed and nbr not in cells
        }
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nbr in _neighbors4(cell[0], cell[1], cell_h, cell_w):
                if nbr in allowed and nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != target:
            continue
        row_counts = Counter(r for r, _ in cells)
        if max(row_counts.values()) > 2:
            continue
        if len(row_counts) < 2:
            continue
        patch = np.full((cell_h, cell_w), BG, dtype=int)
        for r, c in cells:
            patch[r, c] = int(color)
        return patch
    return None


def _length_profiles(cell_w: int) -> list[tuple[int, int, int]]:
    profiles = [
        (2, 3, 4),
        (2, 3, 5),
        (2, 4, 5),
        (2, 4, 6),
        (3, 4, 6),
    ]
    return [profile for profile in profiles if profile[2] <= cell_w]


def _expected_output(stripes: list[dict[str, int]], cell_h: int, cell_w: int) -> np.ndarray:
    ref = sorted(stripes, key=lambda item: item["v"])[len(stripes) // 2]
    v_ref = int(ref["v"])
    l_ref = int(ref["length"])
    rows: list[list[np.ndarray]] = []
    for stripe in stripes:
        cell = np.full((cell_h, cell_w), BG, dtype=int)
        left = int(stripe["left"])
        if left + l_ref > cell_w:
            raise AssertionError("reference-aligned stripe does not fit in the output cell")
        cell[v_ref, left : left + l_ref] = int(stripe["color"])
        rows.append([cell])
    return _build_lattice(rows, BG)


def _gen_attempt() -> np.ndarray | None:
    n_rows = 3
    n_cols = random.choice((3, 4, 5))
    cell_h = random.randint(5, 8)
    max_cell_w = (30 - (n_cols + 1)) // n_cols
    if max_cell_w < 4:
        return None
    min_cell_w = 4 if n_cols == 5 else 5
    if min_cell_w > max_cell_w:
        return None
    cell_w = random.randint(min_cell_w, max_cell_w)

    profiles = _length_profiles(cell_w)
    if not profiles:
        return None
    short_len, ref_len, long_len = random.choice(profiles)

    v_values = random.sample(range(cell_h), n_rows)
    sorted_rows = sorted(range(n_rows), key=lambda idx: v_values[idx])
    ref_row = sorted_rows[1]

    row_lengths = [0] * n_rows
    row_lengths[ref_row] = ref_len
    other_rows = [idx for idx in range(n_rows) if idx != ref_row]
    if random.random() < 0.5:
        row_lengths[other_rows[0]] = short_len
        row_lengths[other_rows[1]] = long_len
    else:
        row_lengths[other_rows[0]] = long_len
        row_lengths[other_rows[1]] = short_len

    stripe_colors = random.sample(STRIPE_COLORS, n_rows)
    base_colors = random.sample(BASE_COLORS, n_rows)
    odd_cols = [random.randrange(n_cols) for _ in range(n_rows)]

    stripes: list[dict[str, int]] = []
    grid_rows: list[list[np.ndarray]] = []
    for row_idx in range(n_rows):
        base = _make_base_pattern(cell_h, cell_w, v_values[row_idx], base_colors[row_idx])
        if base is None:
            return None
        max_left = cell_w - max(ref_len, row_lengths[row_idx])
        if max_left < 0:
            return None
        left = random.randint(0, max_left)
        odd_cell = base.copy()
        odd_cell[v_values[row_idx], left : left + row_lengths[row_idx]] = stripe_colors[row_idx]
        row_cells = [base.copy() for _ in range(n_cols)]
        row_cells[odd_cols[row_idx]] = odd_cell
        grid_rows.append(row_cells)
        stripes.append(
            {
                "v": int(v_values[row_idx]),
                "left": int(left),
                "length": int(row_lengths[row_idx]),
                "color": int(stripe_colors[row_idx]),
            }
        )

    inp = _build_lattice(grid_rows, BG)
    if max(inp.shape) > 30:
        return None

    expected = _expected_output(stripes, cell_h, cell_w)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
