from __future__ import annotations

import random

import numpy as np


BG = 0
TEMPLATE_COLORS = (1, 2)
CROSS_COLOR = 3
CONTROL_COLORS = (4, 5, 6, 7, 8, 9)
NON_TL_QUADRANTS = ((0, 1), (1, 0), (1, 1))
GEN_K_CHOICES = (4, 5, 6)
GEN_TEMPLATE_SIZE_RANGE = (3, 8)
GEN_RETRY_LIMIT = 500
_GEN_BRANCH_COUNTER = 0


def _flip_template(template: np.ndarray, quad_r: int, quad_c: int) -> np.ndarray:
    out = template
    if quad_r:
        out = np.flipud(out)
    if quad_c:
        out = np.fliplr(out)
    return out


def _recolor_from_controls(block: np.ndarray, transformed_template: np.ndarray) -> np.ndarray:
    out = np.full(block.shape, BG, dtype=int)
    mapping: dict[int, int] = {}
    controls = np.argwhere((block != BG) & (block != CROSS_COLOR))
    for r_raw, c_raw in controls:
        r = int(r_raw)
        c = int(c_raw)
        symbol = int(transformed_template[r, c])
        if symbol == BG:
            continue
        color = int(block[r, c])
        if symbol in mapping:
            assert mapping[symbol] == color, "each template symbol must map to a single control color"
        else:
            mapping[symbol] = color

    for symbol, color in mapping.items():
        out[transformed_template == symbol] = color
    return out


def _erase_gap_rows_and_cols(out: np.ndarray, cross_row: np.ndarray, cross_col: np.ndarray, mid: int) -> np.ndarray:
    for r in range(out.shape[0]):
        if r != mid and int(cross_col[r]) == BG:
            out[r, :mid] = BG
            out[r, mid + 1 :] = BG
    for c in range(out.shape[1]):
        if c != mid and int(cross_row[c]) == BG:
            out[:mid, c] = BG
            out[mid + 1 :, c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the top-left template into the other quadrants, recolor those copies from sparse controls, then erase rows and columns selected by cross gaps.

    Input structure:
    - The grid is an odd square with black background and a green central cross that divides it into four equal quadrants.
    - The top-left quadrant contains a complete blue/red template on black cells.
    - The other three quadrants contain only sparse non-black, non-green control pixels, while the central cross may contain black gaps.

    Rule:
    - Keep the top-left template unchanged, and mirror it into the other three quadrants by horizontal, vertical, and double-axis flips.
    - In each non-top-left quadrant, each control pixel assigns its own color to the transformed template symbol underneath it; all cells of unmapped template symbols become black.
    - After all four quadrants are built, every gap on the central column erases that full row inside the left and right quadrants, and every gap on the central row erases that full column inside the upper and lower quadrants.
    - The green cross itself, including its center and any existing black gaps on it, stays unchanged.

    Key insight:
    - The same central cross acts both as the mirror divider and as the erasure guide: its gaps selectively delete whole quadrant rows or columns after the mirrored recolored copies have been formed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1

    n = g.shape[0]
    mid = n // 2
    k = mid

    out = np.full_like(g, BG)
    out[mid, :] = g[mid, :]
    out[:, mid] = g[:, mid]

    template = g[:k, :k].copy()
    out[:k, :k] = template

    for quad_r, quad_c in NON_TL_QUADRANTS:
        transformed = _flip_template(template, quad_r, quad_c)
        r0 = 0 if quad_r == 0 else mid + 1
        c0 = 0 if quad_c == 0 else mid + 1
        block = g[r0 : r0 + k, c0 : c0 + k]
        out[r0 : r0 + k, c0 : c0 + k] = _recolor_from_controls(block, transformed)

    return _erase_gap_rows_and_cols(out, g[mid, :], g[:, mid], mid)


def _neighbors4(cell: tuple[int, int], k: int) -> list[tuple[int, int]]:
    r, c = cell
    nbrs: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < k and 0 <= nc < k:
            nbrs.append((nr, nc))
    return nbrs


def _generate_template(k: int) -> np.ndarray | None:
    area = random.randint(*GEN_TEMPLATE_SIZE_RANGE)
    start = (random.randrange(k), random.randrange(k))
    cells = {start}
    frontier = set(_neighbors4(start, k))

    while len(cells) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors4(cell, k):
            if nbr not in cells:
                frontier.add(nbr)

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None

    template = np.full((k, k), BG, dtype=int)
    n_red = random.randint(1, area - 1)
    red_cells = set(random.sample(tuple(cells), n_red))
    for cell in cells:
        template[cell] = TEMPLATE_COLORS[1] if cell in red_cells else TEMPLATE_COLORS[0]

    if np.array_equal(template, np.fliplr(template)) or np.array_equal(template, np.flipud(template)):
        return None

    return template


def _pick_control_positions(
    transformed: np.ndarray, mapped_symbols: list[int]
) -> dict[int, tuple[int, int]] | None:
    if len(mapped_symbols) == 1:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(transformed == mapped_symbols[0])]
        if not coords:
            return None
        return {mapped_symbols[0]: random.choice(coords)}

    coords_a = [tuple(map(int, rc)) for rc in np.argwhere(transformed == mapped_symbols[0])]
    coords_b = [tuple(map(int, rc)) for rc in np.argwhere(transformed == mapped_symbols[1])]
    pairs = [
        (a, b)
        for a in coords_a
        for b in coords_b
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
    ]
    if not pairs:
        return None
    pos_a, pos_b = random.choice(pairs)
    return {mapped_symbols[0]: pos_a, mapped_symbols[1]: pos_b}


def _build_pre_erase_pair(template: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    k = template.shape[0]
    n = 2 * k + 1
    mid = k
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)

    inp[mid, :] = CROSS_COLOR
    inp[:, mid] = CROSS_COLOR
    out[mid, :] = CROSS_COLOR
    out[:, mid] = CROSS_COLOR

    inp[:k, :k] = template
    out[:k, :k] = template

    full_map_quadrants = set(random.sample(NON_TL_QUADRANTS, random.choice((1, 2, 2, 3))))
    for quad_r, quad_c in NON_TL_QUADRANTS:
        transformed = _flip_template(template, quad_r, quad_c)
        mapped_symbols = list(TEMPLATE_COLORS if (quad_r, quad_c) in full_map_quadrants else (random.choice(TEMPLATE_COLORS),))
        control_positions = _pick_control_positions(transformed, mapped_symbols)
        if control_positions is None:
            return None

        control_colors = random.sample(CONTROL_COLORS, len(mapped_symbols))
        r0 = 0 if quad_r == 0 else mid + 1
        c0 = 0 if quad_c == 0 else mid + 1
        block_in = inp[r0 : r0 + k, c0 : c0 + k]
        block_out = out[r0 : r0 + k, c0 : c0 + k]

        for symbol, color in zip(mapped_symbols, control_colors):
            rr, cc = control_positions[symbol]
            block_in[rr, cc] = color
            block_out[transformed == symbol] = color

    return inp, out


def _choose_gap_indices(pre_out: np.ndarray, mid: int) -> tuple[list[int], list[int]] | None:
    global _GEN_BRANCH_COUNTER

    row_candidates = [
        r
        for r in range(pre_out.shape[0])
        if r != mid and (np.any(pre_out[r, :mid] != BG) or np.any(pre_out[r, mid + 1 :] != BG))
    ]
    col_candidates = [
        c
        for c in range(pre_out.shape[1])
        if c != mid and (np.any(pre_out[:mid, c] != BG) or np.any(pre_out[mid + 1 :, c] != BG))
    ]
    if not row_candidates or not col_candidates:
        return None

    branch = _GEN_BRANCH_COUNTER % 3
    _GEN_BRANCH_COUNTER += 1

    if branch == 0:
        row_gaps = random.sample(row_candidates, random.randint(1, min(2, len(row_candidates))))
        col_gaps: list[int] = []
    elif branch == 1:
        row_gaps = []
        col_gaps = random.sample(col_candidates, random.randint(1, min(2, len(col_candidates))))
    else:
        row_gaps = random.sample(row_candidates, 1)
        col_gaps = random.sample(col_candidates, 1)
        if len(row_candidates) > 1 and len(col_candidates) > 1 and random.random() < 0.35:
            if random.random() < 0.5:
                row_gaps += random.sample([r for r in row_candidates if r not in row_gaps], 1)
            else:
                col_gaps += random.sample([c for c in col_candidates if c not in col_gaps], 1)

    return sorted(row_gaps), sorted(col_gaps)


def _quadrant_nonzero_counts(grid: np.ndarray) -> list[int]:
    mid = grid.shape[0] // 2
    return [
        int(np.count_nonzero(grid[:mid, :mid] != BG)),
        int(np.count_nonzero(grid[:mid, mid + 1 :] != BG)),
        int(np.count_nonzero(grid[mid + 1 :, :mid] != BG)),
        int(np.count_nonzero(grid[mid + 1 :, mid + 1 :] != BG)),
    ]


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(GEN_K_CHOICES)
    template = _generate_template(k)
    if template is None:
        return None

    pair = _build_pre_erase_pair(template)
    if pair is None:
        return None
    inp, pre_out = pair
    mid = k

    gaps = _choose_gap_indices(pre_out, mid)
    if gaps is None:
        return None
    row_gaps, col_gaps = gaps

    inp[row_gaps, mid] = BG
    inp[mid, col_gaps] = BG

    out = main(inp)
    if np.array_equal(out, pre_out):
        return None

    counts = _quadrant_nonzero_counts(out)
    if counts[0] < 2:
        return None
    if sum(count > 0 for count in counts[1:]) < 2:
        return None

    if len(np.unique(inp)) < 4:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-cross-gap input")
