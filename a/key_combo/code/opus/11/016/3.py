from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

BLOCK_LAYOUT = 3
BLOCK_SIZE = 4
GRID_SIZE = BLOCK_LAYOUT * BLOCK_SIZE

SHIFT_MAPS = (
    {BLUE: BLUE, RED: RED, GREEN: GREEN},
    {BLUE: RED, RED: GREEN, GREEN: BLUE},
    {BLUE: GREEN, RED: BLUE, GREEN: RED},
)

MAX_GENERATION_ATTEMPTS = 400
MIN_FOREGROUND_CELLS = 6
MAX_FOREGROUND_CELLS = 10


def _recolor_motif(motif: np.ndarray, shift: int) -> np.ndarray:
    shifted = motif.copy()
    color_map = SHIFT_MAPS[int(shift) % 3]
    for color, mapped in color_map.items():
        shifted[motif == color] = mapped
    return shifted


def _stamp_grid(motif: np.ndarray) -> np.ndarray:
    n = int(motif.shape[0])
    out = np.zeros((n * BLOCK_LAYOUT, n * BLOCK_LAYOUT), dtype=int)
    for block_r in range(BLOCK_LAYOUT):
        for block_c in range(BLOCK_LAYOUT):
            rs = block_r * n
            cs = block_c * n
            out[rs : rs + n, cs : cs + n] = _recolor_motif(motif, (block_r + block_c) % 3)
    return out


def _apply_control_erasures(grid: np.ndarray, controls: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for r, c in controls:
        r = int(r)
        c = int(c)
        row_mask = out[r, :] != YELLOW
        col_mask = out[:, c] != YELLOW
        out[r, row_mask] = BACKGROUND
        out[col_mask, c] = BACKGROUND
    for r, c in controls:
        out[int(r), int(c)] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the top-left motif into every block, rotate its three colors by block diagonal residue, then erase control rows and columns.

    Input structure:
    - The grid is square and divided conceptually into a 3x3 block layout.
    - The top-left block contains the source motif, using black background plus blue(1), red(2), and green(3).
    - Yellow(4) singleton control pixels appear outside that source block on an otherwise black canvas.

    Rule:
    - Read the top-left block as the motif.
    - Paste that motif into all 9 blocks of the 3x3 layout.
    - For block position (br, bc), rotate motif colors by (br + bc) mod 3 using red->green->blue->red.
    - For every yellow control pixel, erase all non-yellow pixels in its row and column, then keep the yellow pixel itself.

    Key insight:
    - The task is an ordered composition: first build the full repeated stamp, then recolor each block by diagonal residue, and only after that remove the marked rows and columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    assert g.shape[0] % BLOCK_LAYOUT == 0, "expected a 3x3 block layout"

    n = g.shape[0] // BLOCK_LAYOUT
    motif = g[:n, :n].copy()
    assert not np.any(motif == YELLOW), "control pixels must be outside the motif block"

    controls = np.argwhere(g == YELLOW)
    stamped = _stamp_grid(motif)
    return _apply_control_erasures(stamped, controls)


def _make_motif() -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    # One cell per row and per column guarantees a full 4x4 bbox and non-empty row/col projections.
    perm = np.random.permutation(BLOCK_SIZE)
    for r, c in enumerate(perm):
        mask[r, int(c)] = True

    target = random.randint(MIN_FOREGROUND_CELLS, MAX_FOREGROUND_CELLS)
    remaining = np.argwhere(~mask)
    extra_count = target - BLOCK_SIZE
    if extra_count > 0:
        chosen = np.random.choice(len(remaining), size=extra_count, replace=False)
        mask[remaining[chosen, 0], remaining[chosen, 1]] = True

    motif = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    fg_cells = np.argwhere(mask)
    colors = [BLUE, BLUE, RED, RED, GREEN, GREEN]
    while len(colors) < len(fg_cells):
        colors.append(random.choice((BLUE, RED, GREEN)))
    random.shuffle(colors)
    for (r, c), color in zip(fg_cells, colors):
        motif[int(r), int(c)] = int(color)
    return motif


def _distinct_block_hits(stamped: np.ndarray, r: int, c: int) -> tuple[int, int]:
    row_blocks = {int(col // BLOCK_SIZE) for col in np.flatnonzero(stamped[r, :] != BACKGROUND)}
    col_blocks = {int(row // BLOCK_SIZE) for row in np.flatnonzero(stamped[:, c] != BACKGROUND)}
    return len(row_blocks), len(col_blocks)


def _pick_controls(motif: np.ndarray) -> list[tuple[int, int]] | None:
    stamped = _stamp_grid(motif)
    all_cells = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if not (r < BLOCK_SIZE and c < BLOCK_SIZE)
    ]
    row_shared = [(r, c) for r, c in all_cells if r < BLOCK_SIZE and c >= BLOCK_SIZE]
    col_shared = [(r, c) for r, c in all_cells if r >= BLOCK_SIZE and c < BLOCK_SIZE]
    far = [(r, c) for r, c in all_cells if r >= BLOCK_SIZE and c >= BLOCK_SIZE]

    valid_cells = []
    for r, c in all_cells:
        row_hits, col_hits = _distinct_block_hits(stamped, r, c)
        if row_hits >= 2 or col_hits >= 2:
            valid_cells.append((r, c))
    valid_set = set(valid_cells)
    row_shared = [pos for pos in row_shared if pos in valid_set]
    col_shared = [pos for pos in col_shared if pos in valid_set]
    far = [pos for pos in far if pos in valid_set]
    all_cells = valid_cells
    if not all_cells:
        return None

    count = random.choice((2, 3))
    mode = random.choice(("far", "row", "col", "mixed"))
    chosen: list[tuple[int, int]] = []

    def take(pool: list[tuple[int, int]]) -> bool:
        options = [pos for pos in pool if pos not in chosen]
        if not options:
            return False
        chosen.append(random.choice(options))
        return True

    if mode == "far":
        if not take(far):
            return None
    elif mode == "row":
        if not take(row_shared):
            return None
    elif mode == "col":
        if not take(col_shared):
            return None
    else:
        if not take(row_shared) or not take(col_shared):
            return None

    while len(chosen) < count:
        if not take(all_cells):
            return None

    rows = {r for r, _ in chosen}
    cols = {c for _, c in chosen}
    if len(rows) == 1 or len(cols) == 1:
        return None
    return chosen


def _visible_shift_counts(out: np.ndarray) -> list[int]:
    counts = [0, 0, 0]
    for block_r in range(BLOCK_LAYOUT):
        for block_c in range(BLOCK_LAYOUT):
            rs = block_r * BLOCK_SIZE
            cs = block_c * BLOCK_SIZE
            block = out[rs : rs + BLOCK_SIZE, cs : cs + BLOCK_SIZE]
            counts[(block_r + block_c) % 3] += int(np.count_nonzero((block != BACKGROUND) & (block != YELLOW)))
    return counts


def _visible_block_count(out: np.ndarray) -> int:
    count = 0
    for block_r in range(BLOCK_LAYOUT):
        for block_c in range(BLOCK_LAYOUT):
            rs = block_r * BLOCK_SIZE
            cs = block_c * BLOCK_SIZE
            block = out[rs : rs + BLOCK_SIZE, cs : cs + BLOCK_SIZE]
            if np.any((block != BACKGROUND) & (block != YELLOW)):
                count += 1
    return count


def _is_valid_example(inp: np.ndarray) -> bool:
    controls = np.argwhere(inp == YELLOW)
    if not (2 <= len(controls) <= 3):
        return False
    motif = inp[:BLOCK_SIZE, :BLOCK_SIZE]
    if np.count_nonzero(motif) < MIN_FOREGROUND_CELLS:
        return False
    if set(np.unique(motif)) - {BACKGROUND, BLUE, RED, GREEN}:
        return False
    if {BLUE, RED, GREEN} - set(np.unique(motif)):
        return False

    out = main(inp)
    non_marker = int(np.count_nonzero((out != BACKGROUND) & (out != YELLOW)))
    if non_marker < 24:
        return False
    if _visible_block_count(out) < 5:
        return False
    if min(_visible_shift_counts(out)) < 4:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    motif = _make_motif()
    controls = _pick_controls(motif)
    if controls is None:
        return None

    inp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    inp[:BLOCK_SIZE, :BLOCK_SIZE] = motif
    for r, c in controls:
        inp[int(r), int(c)] = YELLOW

    if not _is_valid_example(inp):
        return None

    solved = main(inp)
    if solved.shape != (GRID_SIZE, GRID_SIZE):
        return None
    if not np.array_equal(solved, main(inp.copy())):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-stamp input")
