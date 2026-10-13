from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

LATTICE_ROWS = 4
LATTICE_COLS = 3
CELL_SIZE = 3
CELL_GAP = 1
CELL_STRIDE = CELL_SIZE + CELL_GAP
LATTICE_H = LATTICE_ROWS * CELL_SIZE + (LATTICE_ROWS - 1) * CELL_GAP
LATTICE_W = LATTICE_COLS * CELL_SIZE + (LATTICE_COLS - 1) * CELL_GAP
CONTROL_H = LATTICE_ROWS
CONTROL_W = LATTICE_COLS

GRID_H_RANGE = (22, 30)  # high is exclusive
GRID_W_RANGE = (18, 30)
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_pair(sub: np.ndarray) -> tuple[int, int]:
    blue = np.argwhere(sub == BLUE)
    red = np.argwhere(sub == RED)
    assert blue.shape == (1, 2)
    assert red.shape == (1, 2)
    blue_idx = int(blue[0, 0] * CELL_SIZE + blue[0, 1])
    red_idx = int(red[0, 0] * CELL_SIZE + red[0, 1])
    return blue_idx, red_idx


def _is_template_component(g: np.ndarray, coords: np.ndarray) -> bool:
    if coords.shape[0] != CELL_SIZE * CELL_SIZE:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (CELL_SIZE, CELL_SIZE):
        return False
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub == BG):
        return False
    return (
        int(np.count_nonzero(sub == GRAY)) == 7
        and int(np.count_nonzero(sub == BLUE)) == 1
        and int(np.count_nonzero(sub == RED)) == 1
    )


def _find_template_lattice(
    g: np.ndarray,
) -> tuple[list[int], list[int], dict[tuple[int, int], np.ndarray]]:
    comps = [comp for comp in _components(g != BG) if _is_template_component(g, comp)]
    assert len(comps) == LATTICE_ROWS * LATTICE_COLS, "expected 12 template objects"

    by_anchor: dict[tuple[int, int], np.ndarray] = {}
    for comp in comps:
        r0, c0, _r1, _c1 = _bbox(comp)
        by_anchor[(r0, c0)] = comp

    rows = sorted({r for r, _c in by_anchor})
    cols = sorted({c for _r, c in by_anchor})
    assert len(rows) == LATTICE_ROWS and len(cols) == LATTICE_COLS, "template lattice must be 4x3"
    assert all(rows[i + 1] - rows[i] == CELL_STRIDE for i in range(len(rows) - 1))
    assert all(cols[i + 1] - cols[i] == CELL_STRIDE for i in range(len(cols) - 1))
    assert all((r, c) in by_anchor for r in rows for c in cols), "template lattice must be complete"
    return rows, cols, by_anchor


def _find_control_matrix(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = (g == GREEN) | (g == YELLOW)
    comps = _components(mask)
    candidates: list[tuple[int, int, int, int]] = []
    for comp in comps:
        if comp.shape[0] != CONTROL_H * CONTROL_W:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (CONTROL_H, CONTROL_W):
            continue
        if np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one 4x3 green/yellow control matrix"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a 4x3 control matrix to conditionally swap blue and red inside a matching 4x3 lattice of 3x3 templates.

    Input structure:
    - The background is black.
    - One 4x3 lattice of separated 3x3 template objects is present; each template is a full 3x3 block with exactly seven gray cells, one blue cell, and one red cell.
    - A separate dense 4x3 control matrix uses only green and yellow cells, with the same row/column indexing as the template lattice.

    Rule:
    - Match control cell `(i, j)` to the template object at lattice position `(i, j)`.
    - If that control cell is green, swap the template's blue and red cells in place.
    - If that control cell is yellow, leave the template unchanged.
    - Erase the entire control matrix to black, and keep every other pixel unchanged.

    Key insight:
    - The control colors are boolean instructions, not repaint colors: green means "swap blue/red here" and yellow means "do nothing here".
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    rows, cols, _by_anchor = _find_template_lattice(g)
    ctrl_r0, ctrl_c0, ctrl_r1, ctrl_c1 = _find_control_matrix(g)

    out = g.copy()
    for i, r0 in enumerate(rows):
        for j, c0 in enumerate(cols):
            if int(g[ctrl_r0 + i, ctrl_c0 + j]) == GREEN:
                sub = out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
                blue_mask = sub == BLUE
                red_mask = sub == RED
                assert int(np.count_nonzero(blue_mask)) == 1
                assert int(np.count_nonzero(red_mask)) == 1
                sub[blue_mask] = RED
                sub[red_mask] = BLUE
            else:
                assert int(g[ctrl_r0 + i, ctrl_c0 + j]) == YELLOW

    out[ctrl_r0 : ctrl_r1 + 1, ctrl_c0 : ctrl_c1 + 1] = BG
    return out


def _pair_pool() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for blue_idx in range(CELL_SIZE * CELL_SIZE):
        for red_idx in range(CELL_SIZE * CELL_SIZE):
            if blue_idx != red_idx:
                pairs.append((blue_idx, red_idx))
    return pairs


PAIR_POOL = _pair_pool()


def _pair_to_cells(pair: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    blue_idx, red_idx = pair
    return divmod(blue_idx, CELL_SIZE), divmod(red_idx, CELL_SIZE)


def _sample_control(rng: np.random.Generator) -> np.ndarray:
    while True:
        flat = np.array([GREEN] * 6 + [YELLOW] * 6, dtype=int)
        rng.shuffle(flat)
        ctrl = flat.reshape(CONTROL_H, CONTROL_W)
        if np.array_equal(ctrl, np.flipud(ctrl)):
            continue
        if np.array_equal(ctrl, np.fliplr(ctrl)):
            continue
        row_mixed = any(len(set(map(int, row))) == 2 for row in ctrl)
        col_mixed = any(len(set(map(int, col))) == 2 for col in ctrl.T)
        if row_mixed and col_mixed:
            return ctrl


def _sample_template_pairs(rng: np.random.Generator, control: np.ndarray) -> np.ndarray:
    green_slots = np.argwhere(control == GREEN)
    yellow_slots = np.argwhere(control == YELLOW)
    assert green_slots.size and yellow_slots.size

    pair_grid = np.empty((LATTICE_ROWS, LATTICE_COLS), dtype=object)
    reused = PAIR_POOL[int(rng.integers(0, len(PAIR_POOL)))]
    gpos = tuple(map(int, green_slots[int(rng.integers(0, len(green_slots)))]))
    ypos = tuple(map(int, yellow_slots[int(rng.integers(0, len(yellow_slots)))]))
    pair_grid[gpos] = reused
    pair_grid[ypos] = reused

    distinct_needed = {reused}
    for i in range(LATTICE_ROWS):
        for j in range(LATTICE_COLS):
            if pair_grid[i, j] is not None:
                continue
            pair = PAIR_POOL[int(rng.integers(0, len(PAIR_POOL)))]
            pair_grid[i, j] = pair
            distinct_needed.add(pair)

    if len(distinct_needed) < 4:
        empties = [(i, j) for i in range(LATTICE_ROWS) for j in range(LATTICE_COLS) if (i, j) not in {gpos, ypos}]
        rng.shuffle(empties)
        while len({pair_grid[i, j] for i in range(LATTICE_ROWS) for j in range(LATTICE_COLS)}) < 4:
            pos = empties.pop()
            pair_grid[pos] = PAIR_POOL[int(rng.integers(0, len(PAIR_POOL)))]

    return pair_grid


def _sample_layout(rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))

    max_lat_r = h - (LATTICE_H + CONTROL_H + 2)
    max_lat_c = w - (LATTICE_W + CONTROL_W + 2)
    if max_lat_r < 1 or max_lat_c < 1:
        raise ValueError("grid too small")

    lat_r0 = int(rng.integers(1, max_lat_r + 1))
    lat_c0 = int(rng.integers(1, max_lat_c + 1))

    ctrl_r_lo = lat_r0 + LATTICE_H + 1
    ctrl_c_lo = lat_c0 + LATTICE_W + 1
    ctrl_r0 = int(rng.integers(ctrl_r_lo, h - CONTROL_H))
    ctrl_c0 = int(rng.integers(ctrl_c_lo, w - CONTROL_W))
    return (h, w), (lat_r0, lat_c0), (ctrl_r0, ctrl_c0)


def _render_scene(
    shape: tuple[int, int],
    lattice_origin: tuple[int, int],
    control_origin: tuple[int, int],
    pairs: np.ndarray,
    control: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    lat_r0, lat_c0 = lattice_origin
    ctrl_r0, ctrl_c0 = control_origin
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    for i in range(LATTICE_ROWS):
        for j in range(LATTICE_COLS):
            r0 = lat_r0 + i * CELL_STRIDE
            c0 = lat_c0 + j * CELL_STRIDE
            inp[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = GRAY
            out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = GRAY

            blue_cell, red_cell = _pair_to_cells(pairs[i, j])
            inp[r0 + blue_cell[0], c0 + blue_cell[1]] = BLUE
            inp[r0 + red_cell[0], c0 + red_cell[1]] = RED

            if int(control[i, j]) == GREEN:
                out[r0 + blue_cell[0], c0 + blue_cell[1]] = RED
                out[r0 + red_cell[0], c0 + red_cell[1]] = BLUE
            else:
                out[r0 + blue_cell[0], c0 + blue_cell[1]] = BLUE
                out[r0 + red_cell[0], c0 + red_cell[1]] = RED

    inp[ctrl_r0 : ctrl_r0 + CONTROL_H, ctrl_c0 : ctrl_c0 + CONTROL_W] = control
    return inp, out


def _validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    lattice_origin: tuple[int, int],
    control_origin: tuple[int, int],
    pairs: np.ndarray,
    control: np.ndarray,
) -> bool:
    try:
        rows, cols, _by_anchor = _find_template_lattice(inp)
        ctrl_box = _find_control_matrix(inp)
        actual = main(inp)
    except Exception:
        return False

    lat_r0, lat_c0 = lattice_origin
    ctrl_r0, ctrl_c0 = control_origin
    if rows != [lat_r0 + i * CELL_STRIDE for i in range(LATTICE_ROWS)]:
        return False
    if cols != [lat_c0 + j * CELL_STRIDE for j in range(LATTICE_COLS)]:
        return False
    if ctrl_box != (ctrl_r0, ctrl_c0, ctrl_r0 + CONTROL_H - 1, ctrl_c0 + CONTROL_W - 1):
        return False
    if not np.array_equal(inp[ctrl_r0 : ctrl_r0 + CONTROL_H, ctrl_c0 : ctrl_c0 + CONTROL_W], control):
        return False
    if not np.array_equal(actual, expected):
        return False

    changed_templates = 0
    for i, r0 in enumerate(rows):
        for j, c0 in enumerate(cols):
            sub = inp[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            if _extract_pair(sub) != pairs[i, j]:
                return False
            out_sub = expected[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            if int(control[i, j]) == GREEN:
                changed_templates += 1
                if _extract_pair(out_sub) != (pairs[i, j][1], pairs[i, j][0]):
                    return False
            else:
                if _extract_pair(out_sub) != pairs[i, j]:
                    return False

    if changed_templates == 0 or changed_templates == LATTICE_ROWS * LATTICE_COLS:
        return False
    if len({pairs[i, j] for i in range(LATTICE_ROWS) for j in range(LATTICE_COLS)}) < 4:
        return False
    if int(np.count_nonzero(expected != inp)) != CONTROL_H * CONTROL_W + 2 * changed_templates:
        return False
    if set(map(int, np.unique(expected))) - {BG, BLUE, RED, GRAY}:
        return False

    duplicate_branch_hits = {}
    for i in range(LATTICE_ROWS):
        for j in range(LATTICE_COLS):
            duplicate_branch_hits.setdefault(pairs[i, j], set()).add(int(control[i, j]))
    if not any(branches == {GREEN, YELLOW} for branches in duplicate_branch_hits.values()):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    try:
        shape, lattice_origin, control_origin = _sample_layout(rng)
    except ValueError:
        return None
    control = _sample_control(rng)
    pairs = _sample_template_pairs(rng, control)
    inp, expected = _render_scene(shape, lattice_origin, control_origin, pairs, control)
    if not _validate_scene(inp, expected, lattice_origin, control_origin, pairs, control):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
