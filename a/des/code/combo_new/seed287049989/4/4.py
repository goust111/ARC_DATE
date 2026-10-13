from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5
LINE_COLORS = {BLUE, RED, GRAY}
PAYLOAD_COLORS = [3, 4, 6, 7, 8, 9]

CELL_COUNT_RANGE = (2, 4)
CELL_SIZE_MIN = 3
CELL_SIZE_MAX = 6
SEGMENT_RED_PROB = 1.0 - (0.5 ** 0.5)  # Makes each cell-axis flag red with probability 1/2.
MAX_GENERATION_ATTEMPTS = 400


def _apply_symmetry(mask: np.ndarray, need_ud: bool, need_lr: bool) -> np.ndarray:
    out = mask.copy()
    if need_ud:
        out |= np.flipud(mask)
    if need_lr:
        out |= np.fliplr(mask)
    if need_ud and need_lr:
        out |= np.flipud(np.fliplr(mask))
    return out


def _find_dividers(grid: np.ndarray) -> tuple[list[int], list[int], int]:
    h, w = grid.shape
    row_divs = [r for r in range(h) if all(int(v) in LINE_COLORS for v in grid[r, :])]
    col_divs = [c for c in range(w) if all(int(v) in LINE_COLORS for v in grid[:, c])]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "expected a framed lattice of cells"

    row_steps = {row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1)}
    col_steps = {col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1)}
    assert len(row_steps) == 1 and len(col_steps) == 1, "divider lines must be evenly spaced"

    row_step = row_steps.pop()
    col_step = col_steps.pop()
    assert row_step == col_step and row_step >= 2, "cell interiors must be equal square regions"
    assert row_divs == list(range(row_divs[0], row_divs[-1] + 1, row_step))
    assert col_divs == list(range(col_divs[0], col_divs[-1] + 1, col_step))
    return row_divs, col_divs, row_step - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each compartment's colored motif so it matches the mirror symmetry encoded by its border segments.

    Input structure:
    - Black background surrounds a lattice of equal square compartments.
    - One-cell-thick gray, red, and blue divider lines form the compartment borders; divider intersections stay gray.
    - Each compartment interior contains black plus a single non-divider color forming an incomplete motif.

    Rule:
    - Detect the compartment size from the evenly spaced full divider rows and columns.
    - For each compartment, a red top or bottom segment requires up-down mirror symmetry about the horizontal midline.
    - A red left or right segment requires left-right mirror symmetry about the vertical midline.
    - Preserve existing motif pixels and add same-color mirror counterparts until all red-indicated symmetries hold; blue and gray segments impose no extra filling.

    Key insight:
    The colored frame segments are local instructions: each compartment is solved independently by taking the mirror-closure of its visible motif under the axes marked in red.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    row_divs, col_divs, _cell_size = _find_dividers(g)

    for i in range(len(row_divs) - 1):
        r0, r1 = row_divs[i], row_divs[i + 1]
        for j in range(len(col_divs) - 1):
            c0, c1 = col_divs[j], col_divs[j + 1]
            cell = g[r0 + 1 : r1, c0 + 1 : c1]
            payload = [
                int(v)
                for v in np.unique(cell)
                if int(v) not in LINE_COLORS and int(v) != BLACK
            ]
            if not payload:
                continue
            color = max(payload, key=lambda v: int(np.count_nonzero(cell == v)))
            mask = cell == color

            need_ud = bool(np.any(g[r0, c0 + 1 : c1] == RED) or np.any(g[r1, c0 + 1 : c1] == RED))
            need_lr = bool(np.any(g[r0 + 1 : r1, c0] == RED) or np.any(g[r0 + 1 : r1, c1] == RED))

            filled = _apply_symmetry(mask, need_ud, need_lr)
            patch = out[r0 + 1 : r1, c0 + 1 : c1]
            patch[filled] = color

    return out


def _orbit_cells(r: int, c: int, size: int, need_ud: bool, need_lr: bool) -> list[tuple[int, int]]:
    pts = {(int(r), int(c))}
    if need_ud:
        pts.add((size - 1 - int(r), int(c)))
    if need_lr:
        pts.add((int(r), size - 1 - int(c)))
    if need_ud and need_lr:
        pts.add((size - 1 - int(r), size - 1 - int(c)))
    return sorted(pts)


def _filled_orbits(mask: np.ndarray, need_ud: bool, need_lr: bool) -> list[list[tuple[int, int]]]:
    size = mask.shape[0]
    visited: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        pt = (int(r), int(c))
        if pt in visited:
            continue
        orbit = _orbit_cells(pt[0], pt[1], size, need_ud, need_lr)
        orbits.append(orbit)
        visited.update(orbit)
    return orbits


def _grow_connected_mask(height: int, width: int, target: int, rng: random.Random) -> np.ndarray | None:
    if target < 1 or target > height * width:
        return None
    mask = np.zeros((height, width), dtype=bool)
    start = (rng.randrange(height), rng.randrange(width))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = rng.choice(tuple(frontier))
        mask[nr, nc] = True

    return mask


def _generate_complete_mask(size: int, need_ud: bool, need_lr: bool, rng: random.Random) -> np.ndarray | None:
    fund_h = (size + 1) // 2 if need_ud else size
    fund_w = (size + 1) // 2 if need_lr else size
    max_area = max(2, int(np.ceil(size * size * 0.6)))

    for _ in range(80):
        target = rng.randint(1, min(fund_h * fund_w, size + 1))
        base = _grow_connected_mask(fund_h, fund_w, target, rng)
        if base is None:
            continue

        seed = np.zeros((size, size), dtype=bool)
        seed[:fund_h, :fund_w] = base
        full = _apply_symmetry(seed, need_ud, need_lr)
        area = int(full.sum())
        if area < 2 or area > max_area or area >= size * size:
            continue

        if need_ud or need_lr:
            if not any(len(orbit) > 1 for orbit in _filled_orbits(full, need_ud, need_lr)):
                continue
        else:
            if (np.array_equal(full, np.flipud(full)) or np.array_equal(full, np.fliplr(full))) and rng.random() < 0.8:
                continue

        return full

    return None


def _make_partial_mask(full: np.ndarray, need_ud: bool, need_lr: bool, rng: random.Random) -> np.ndarray:
    if not need_ud and not need_lr:
        return full.copy()

    partial = np.zeros_like(full, dtype=bool)
    for orbit in _filled_orbits(full, need_ud, need_lr):
        if len(orbit) == 1:
            keep = orbit
        else:
            order = orbit[:]
            rng.shuffle(order)
            keep = order[: rng.randint(1, len(order) - 1)]
        for r, c in keep:
            partial[r, c] = True
    return partial


def _render_grid(
    n_rows: int,
    n_cols: int,
    cell_size: int,
    h_segments: np.ndarray,
    v_segments: np.ndarray,
    masks: list[list[np.ndarray]],
    colors: list[list[int]],
) -> np.ndarray:
    h = n_rows * cell_size + n_rows + 1
    w = n_cols * cell_size + n_cols + 1
    grid = np.full((h, w), BLACK, dtype=int)

    row_divs = [i * (cell_size + 1) for i in range(n_rows + 1)]
    col_divs = [j * (cell_size + 1) for j in range(n_cols + 1)]
    grid[row_divs, :] = GRAY
    grid[:, col_divs] = GRAY

    for i in range(n_rows + 1):
        r = row_divs[i]
        for j in range(n_cols):
            grid[r, col_divs[j] + 1 : col_divs[j + 1]] = int(h_segments[i, j])

    for i in range(n_rows):
        for j in range(n_cols + 1):
            c = col_divs[j]
            grid[row_divs[i] + 1 : row_divs[i + 1], c] = int(v_segments[i, j])

    for i in range(n_rows):
        for j in range(n_cols):
            patch = grid[
                row_divs[i] + 1 : row_divs[i + 1],
                col_divs[j] + 1 : col_divs[j + 1],
            ]
            patch[masks[i][j]] = int(colors[i][j])

    return grid


def _segment_layout(n_rows: int, n_cols: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        h_red = np.array(
            [[rng.random() < SEGMENT_RED_PROB for _ in range(n_cols)] for _ in range(n_rows + 1)],
            dtype=bool,
        )
        v_red = np.array(
            [[rng.random() < SEGMENT_RED_PROB for _ in range(n_cols + 1)] for _ in range(n_rows)],
            dtype=bool,
        )
        need_ud = h_red[:-1, :] | h_red[1:, :]
        need_lr = v_red[:, :-1] | v_red[:, 1:]
        kinds = {(bool(need_ud[i, j]), bool(need_lr[i, j])) for i in range(n_rows) for j in range(n_cols)}
        if n_rows * n_cols >= 4 and len(kinds) < 4:
            continue
        if not np.any(need_ud | need_lr):
            continue

        h_segments = np.full((n_rows + 1, n_cols), GRAY, dtype=int)
        v_segments = np.full((n_rows, n_cols + 1), GRAY, dtype=int)
        for i in range(n_rows + 1):
            for j in range(n_cols):
                h_segments[i, j] = RED if h_red[i, j] else rng.choice([BLUE, GRAY])
        for i in range(n_rows):
            for j in range(n_cols + 1):
                v_segments[i, j] = RED if v_red[i, j] else rng.choice([BLUE, GRAY])

        flat_segments = list(h_segments.ravel()) + list(v_segments.ravel())
        if BLUE not in flat_segments or GRAY not in flat_segments or RED not in flat_segments:
            continue
        return h_segments, v_segments

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(*CELL_COUNT_RANGE)
    n_cols = rng.randint(*CELL_COUNT_RANGE)
    max_size = min(
        CELL_SIZE_MAX,
        (30 - (n_rows + 1)) // n_rows,
        (30 - (n_cols + 1)) // n_cols,
    )
    if max_size < CELL_SIZE_MIN:
        return None
    cell_size = rng.randint(CELL_SIZE_MIN, max_size)

    segment_layout = _segment_layout(n_rows, n_cols, rng)
    if segment_layout is None:
        return None
    h_segments, v_segments = segment_layout

    need_ud = (h_segments[:-1, :] == RED) | (h_segments[1:, :] == RED)
    need_lr = (v_segments[:, :-1] == RED) | (v_segments[:, 1:] == RED)

    full_masks: list[list[np.ndarray]] = []
    partial_masks: list[list[np.ndarray]] = []
    colors: list[list[int]] = []
    changed_cells = 0

    for i in range(n_rows):
        full_row: list[np.ndarray] = []
        partial_row: list[np.ndarray] = []
        color_row: list[int] = []
        for j in range(n_cols):
            mask = _generate_complete_mask(cell_size, bool(need_ud[i, j]), bool(need_lr[i, j]), rng)
            if mask is None:
                return None
            partial = _make_partial_mask(mask, bool(need_ud[i, j]), bool(need_lr[i, j]), rng)
            if not np.array_equal(mask, partial):
                changed_cells += 1
            full_row.append(mask)
            partial_row.append(partial)
            color_row.append(rng.choice(PAYLOAD_COLORS))
        full_masks.append(full_row)
        partial_masks.append(partial_row)
        colors.append(color_row)

    if changed_cells == 0:
        return None

    inp = _render_grid(n_rows, n_cols, cell_size, h_segments, v_segments, partial_masks, colors)
    out = _render_grid(n_rows, n_cols, cell_size, h_segments, v_segments, full_masks, colors)
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
