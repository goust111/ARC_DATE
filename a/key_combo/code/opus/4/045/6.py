from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, paste


RED = Color.RED
BG = Color.BLACK
GEN_ATTEMPTS = 500
CORE_SIDE_OPTIONS = (4, 5)
ROW_REPEAT_OPTIONS = (3, 4)
COL_REPEAT_OPTIONS = (3, 4)
PALETTE = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)


def _row_keep_indices(grid: np.ndarray) -> list[int]:
    keep = [0]
    for r in range(1, grid.shape[0]):
        if not np.array_equal(grid[r], grid[r - 1]):
            keep.append(r)
    return keep


def _col_keep_indices(grid: np.ndarray) -> list[int]:
    keep = [0]
    for c in range(1, grid.shape[1]):
        if not np.array_equal(grid[:, c], grid[:, c - 1]):
            keep.append(c)
    return keep


def _compress_core(grid: np.ndarray) -> np.ndarray:
    rows = _row_keep_indices(grid)
    cols = _col_keep_indices(grid)
    return np.asarray(grid[np.ix_(rows, cols)], dtype=int)


def _run_lengths(items: list[np.ndarray]) -> list[int]:
    if not items:
        return []
    lengths = []
    start = 0
    for idx in range(1, len(items) + 1):
        if idx == len(items) or not np.array_equal(items[idx], items[idx - 1]):
            lengths.append(idx - start)
            start = idx
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a row/column-stretched hidden core pattern, then use its red cells as one-level recursive stamp controls.

    Input structure:
    - The background is black, and the visible grid is a smaller core pattern whose rows and columns were each repeated in contiguous runs.
    - Every row run has the same length, every column run has the same length, and those repeat lengths are encoded only by the visible repeated rows and columns.
    - The hidden core contains black cells, one to three non-touching red cells, and other non-red colors that define the template itself.

    Rule:
    - Scan top to bottom and keep the first row from each maximal run of identical consecutive rows; scan left to right and keep the first column from each maximal run of identical consecutive columns.
    - Intersect those kept rows and columns to recover the core pattern P of shape R x C.
    - Create an output grid of shape (R*R) x (C*C), viewed as an R x C array of blocks, each block having shape R x C.
    - For every red cell of P, copy the full pattern P into the corresponding output block. For every non-red cell of P, fill the corresponding output block uniformly with that cell's color.
    - Stamped copies are literal copies of P only; red cells inside a stamped block are not expanded again.

    Key insight:
    - The same compressed pattern simultaneously provides the hidden template, the red stamp-control positions, and the output scale, so the repeated input must be compressed before any stamping rule becomes visible.
    """
    g = np.asarray(grid, dtype=int)
    core = _compress_core(g)
    rows, cols = core.shape
    out = np.full((rows * rows, cols * cols), BG, dtype=int)

    for r in range(rows):
        for c in range(cols):
            r0 = r * rows
            c0 = c * cols
            if int(core[r, c]) == RED:
                paste(out, core, r0, c0)
            else:
                out[r0 : r0 + rows, c0 : c0 + cols] = int(core[r, c])

    return out


def _reds_are_separated(core: np.ndarray) -> bool:
    reds = [tuple(map(int, rc)) for rc in np.argwhere(core == RED)]
    for i, (r0, c0) in enumerate(reds):
        for r1, c1 in reds[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _core_rows_and_cols_unique(core: np.ndarray) -> bool:
    row_keys = {tuple(map(int, row)) for row in core}
    col_keys = {tuple(map(int, col)) for col in core.T}
    return len(row_keys) == core.shape[0] and len(col_keys) == core.shape[1]


def _core_is_rich_enough(core: np.ndarray) -> bool:
    counts: dict[int, int] = {}
    for value in core.ravel():
        counts[int(value)] = counts.get(int(value), 0) + 1

    non_red_colors = {color for color in counts if color != RED}
    colored_non_red = {color for color in non_red_colors if color != BG}
    if BG not in counts or len(colored_non_red) < 2:
        return False
    if counts[BG] <= max(counts[color] for color in counts if color != BG):
        return False
    if not (1 <= counts.get(RED, 0) <= 3):
        return False
    return True


def _sample_red_positions(side: int, red_count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(side) for c in range(side)]
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == red_count:
                return chosen
    return None


def _sample_core(side: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        palette_size = rng.choice((2, 3))
        colors = rng.sample(list(PALETTE), palette_size)
        red_count = rng.choice((1, 2, 3))
        red_positions = _sample_red_positions(side, red_count, rng)
        if red_positions is None:
            continue

        core = np.full((side, side), BG, dtype=int)
        for r, c in red_positions:
            core[r, c] = RED

        free_cells = [(r, c) for r in range(side) for c in range(side) if core[r, c] == BG]
        min_colored = max(2 * palette_size, side + 1)
        max_colored = max(min_colored, len(free_cells) // 2 + 1)
        if max_colored > len(free_cells) - 2:
            max_colored = len(free_cells) - 2
        if min_colored > max_colored:
            continue

        colored_count = rng.randint(min_colored, max_colored)
        chosen_cells = rng.sample(free_cells, colored_count)
        counts = [2] * palette_size
        for _ in range(colored_count - sum(counts)):
            counts[rng.randrange(palette_size)] += 1
        rng.shuffle(chosen_cells)
        cursor = 0
        for color, count in zip(colors, counts):
            for r, c in chosen_cells[cursor : cursor + count]:
                core[r, c] = color
            cursor += count

        if not _reds_are_separated(core):
            continue
        if not _core_rows_and_cols_unique(core):
            continue
        if not _core_is_rich_enough(core):
            continue
        if np.array_equal(core, np.rot90(core)) or np.array_equal(core, np.fliplr(core)):
            continue
        return core
    return None


def _stretch_core(core: np.ndarray, row_repeat: int, col_repeat: int) -> np.ndarray:
    return np.repeat(np.repeat(core, row_repeat, axis=0), col_repeat, axis=1)


def _validate_generated(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int(get_bg(inp)) != BG:
        return False

    row_runs = _run_lengths([inp[r].copy() for r in range(inp.shape[0])])
    col_runs = _run_lengths([inp[:, c].copy() for c in range(inp.shape[1])])
    if len(set(row_runs)) != 1 or len(set(col_runs)) != 1:
        return False
    if row_runs[0] not in ROW_REPEAT_OPTIONS or col_runs[0] not in COL_REPEAT_OPTIONS:
        return False

    core = _compress_core(inp)
    side_r, side_c = core.shape
    if side_r != side_c or side_r not in CORE_SIDE_OPTIONS:
        return False
    if not _reds_are_separated(core):
        return False
    if not _core_rows_and_cols_unique(core):
        return False
    if not _core_is_rich_enough(core):
        return False
    if not np.array_equal(inp, _stretch_core(core, row_runs[0], col_runs[0])):
        return False

    out = main(inp)
    if out.shape != (side_r * side_r, side_c * side_c):
        return False

    simple_scale = np.kron(core, np.ones((side_r, side_c), dtype=int))
    if np.array_equal(out, simple_scale):
        return False

    for r in range(side_r):
        for c in range(side_c):
            block = out[r * side_r : (r + 1) * side_r, c * side_c : (c + 1) * side_c]
            if int(core[r, c]) == RED:
                if not np.array_equal(block, core):
                    return False
            else:
                if not np.all(block == int(core[r, c])):
                    return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        side = rng.choice(CORE_SIDE_OPTIONS)
        core = _sample_core(side, rng)
        if core is None:
            continue
        row_repeat = rng.choice(ROW_REPEAT_OPTIONS)
        col_repeat = rng.choice(COL_REPEAT_OPTIONS)
        inp = _stretch_core(core, row_repeat, col_repeat)
        if _validate_generated(inp):
            return inp
    raise RuntimeError("failed to generate valid input for puzzle 6")
