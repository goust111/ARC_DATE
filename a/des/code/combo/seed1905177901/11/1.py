from __future__ import annotations

import random

import numpy as np


BG = 0
FG = 1
RED = 2
GREY = 5
LINE_COLORS = {RED, GREY}
GRID_SIZES = (3, 4)
CELL_SIZE_RANGE = (4, 6)
MAX_GEN_ATTEMPTS = 400


def _line_indices(g: np.ndarray, axis: int) -> list[int]:
    indices: list[int] = []
    length = g.shape[axis]
    for idx in range(length):
        line = g[idx, :] if axis == 0 else g[:, idx]
        colors = {int(v) for v in np.unique(line)}
        if colors and colors <= LINE_COLORS:
            indices.append(int(idx))
    return indices


def _grid_structure(g: np.ndarray) -> tuple[list[int], list[int], int]:
    rows = _line_indices(g, axis=0)
    cols = _line_indices(g, axis=1)
    assert len(rows) >= 4 and len(cols) >= 4
    row_steps = [b - a for a, b in zip(rows, rows[1:])]
    col_steps = [b - a for a, b in zip(cols, cols[1:])]
    assert row_steps and col_steps
    assert len(set(row_steps)) == 1
    assert len(set(col_steps)) == 1
    step_r = row_steps[0]
    step_c = col_steps[0]
    assert step_r == step_c and step_r >= 2
    n_rows = len(rows) - 1
    n_cols = len(cols) - 1
    assert n_rows == n_cols
    assert n_rows in GRID_SIZES
    cell_size = step_r - 1
    assert cell_size >= 1
    return rows, cols, cell_size


def _find_template(rows: list[int], cols: list[int], g: np.ndarray) -> tuple[int, int]:
    templates: list[tuple[int, int]] = []
    for r_idx in range(len(rows) - 1):
        for c_idx in range(len(cols) - 1):
            r0, r1 = rows[r_idx], rows[r_idx + 1]
            c0, c1 = cols[c_idx], cols[c_idx + 1]
            top = g[r0, c0 : c1 + 1]
            bottom = g[r1, c0 : c1 + 1]
            left = g[r0 : r1 + 1, c0]
            right = g[r0 : r1 + 1, c1]
            if (
                np.all(top == RED)
                and np.all(bottom == RED)
                and np.all(left == RED)
                and np.all(right == RED)
            ):
                templates.append((r_idx, c_idx))
    assert len(templates) == 1
    return templates[0]


def _cell_view(g: np.ndarray, rows: list[int], cols: list[int], r_idx: int, c_idx: int) -> np.ndarray:
    r0, r1 = rows[r_idx], rows[r_idx + 1]
    c0, c1 = cols[c_idx], cols[c_idx + 1]
    return g[r0 + 1 : r1, c0 + 1 : c1]


def _transform(pattern: np.ndarray, dr: int, dc: int) -> np.ndarray:
    if dr == 0 and dc == 0:
        return pattern.copy()
    if dc == 0:
        return np.flipud(pattern)
    if dr == 0:
        return np.fliplr(pattern)
    return np.rot90(pattern, 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the red-bordered template cell as the transformation origin.

    Cells in the same column as the template get a vertical flip of the template pattern.
    Cells in the same row get a horizontal flip.
    Diagonal cells get a 180-degree rotation.
    The template itself stays unchanged, and its red border turns back to grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BG, FG, RED, GREY}

    rows, cols, cell_size = _grid_structure(g)
    template_r, template_c = _find_template(rows, cols, g)
    template = _cell_view(g, rows, cols, template_r, template_c).copy()
    assert template.shape == (cell_size, cell_size)
    assert set(map(int, np.unique(template))) <= {BG, FG}

    out = g.copy()
    out[out == RED] = GREY
    for r_idx in range(len(rows) - 1):
        for c_idx in range(len(cols) - 1):
            target = _transform(template, r_idx - template_r, c_idx - template_c)
            view = _cell_view(out, rows, cols, r_idx, c_idx)
            assert view.shape == target.shape
            view[:, :] = target
    return out


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < size and 0 <= cc < size:
            out.append((rr, cc))
    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, mask.shape[0]):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _variants(pattern: np.ndarray) -> list[np.ndarray]:
    return [
        pattern.copy(),
        np.flipud(pattern),
        np.fliplr(pattern),
        np.rot90(pattern, 2),
    ]


def _all_distinct(arrays: list[np.ndarray]) -> bool:
    return all(
        not np.array_equal(arrays[i], arrays[j])
        for i in range(len(arrays))
        for j in range(i + 1, len(arrays))
    )


def _random_pattern(cell_size: int) -> np.ndarray:
    for _ in range(200):
        target = random.randint(max(4, cell_size), min(cell_size * cell_size - 1, cell_size * cell_size // 2 + 3))
        mask = np.zeros((cell_size, cell_size), dtype=bool)
        seed = (random.randrange(cell_size), random.randrange(cell_size))
        mask[seed] = True
        frontier = set(_neighbors4(seed[0], seed[1], cell_size))
        while int(mask.sum()) < target and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            if mask[rr, cc]:
                continue
            mask[rr, cc] = True
            for nbr in _neighbors4(rr, cc, cell_size):
                if not mask[nbr]:
                    frontier.add(nbr)
        if int(mask.sum()) != target:
            continue
        if not _connected(mask):
            continue
        pattern = mask.astype(int) * FG
        if _all_distinct(_variants(pattern)):
            return pattern
    raise RuntimeError("failed to build asymmetric pattern")


def _template_positions(n: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]


def _expected_output(
    inp: np.ndarray,
    rows: list[int],
    cols: list[int],
    template_pos: tuple[int, int],
    template: np.ndarray,
) -> np.ndarray:
    out = inp.copy()
    out[out == RED] = GREY
    tr, tc = template_pos
    for r_idx in range(len(rows) - 1):
        for c_idx in range(len(cols) - 1):
            _cell_view(out, rows, cols, r_idx, c_idx)[:, :] = _transform(template, r_idx - tr, c_idx - tc)
    return out


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.choice(GRID_SIZES)
    cell_size = random.randint(*CELL_SIZE_RANGE)
    total = n * cell_size + (n + 1)
    rows = [i * (cell_size + 1) for i in range(n + 1)]
    cols = [i * (cell_size + 1) for i in range(n + 1)]

    inp = np.full((total, total), BG, dtype=int)
    inp[rows, :] = GREY
    inp[:, cols] = GREY

    template_pos = random.choice(_template_positions(n))
    tr, tc = template_pos
    r0, r1 = rows[tr], rows[tr + 1]
    c0, c1 = cols[tc], cols[tc + 1]
    inp[r0, c0 : c1 + 1] = RED
    inp[r1, c0 : c1 + 1] = RED
    inp[r0 : r1 + 1, c0] = RED
    inp[r0 : r1 + 1, c1] = RED

    canonical = _random_pattern(cell_size)
    template = random.choice(_variants(canonical)).copy()
    mismatch_count = 0
    for r_idx in range(n):
        for c_idx in range(n):
            desired = _transform(template, r_idx - tr, c_idx - tc)
            if (r_idx, c_idx) == template_pos:
                placed = template
            else:
                placed = random.choice(_variants(template)).copy()
                if np.array_equal(placed, desired):
                    alternatives = [v for v in _variants(template) if not np.array_equal(v, desired)]
                    placed = random.choice(alternatives).copy()
                mismatch_count += int(not np.array_equal(placed, desired))
            _cell_view(inp, rows, cols, r_idx, c_idx)[:, :] = placed

    assert mismatch_count > 0
    out = _expected_output(inp, rows, cols, template_pos, template)
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        pair = _build_candidate()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate valid puzzle input")
