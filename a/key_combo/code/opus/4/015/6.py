from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    _arc_utils = SourceFileLoader("_arc_utils_015_6", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
    paste = _arc_utils.paste
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
    ) -> np.ndarray:
        r0 = int(r0)
        c0 = int(c0)
        h, w = map(int, patch.shape)
        view = dst[r0 : r0 + h, c0 : c0 + w]
        if transparent is None:
            view[:, :] = patch
        else:
            mask = patch != int(transparent)
            view[mask] = patch[mask]
        return dst


BLOCK_SIZE = 3
COARSE_SIZE = 3
GRID_SIZE = BLOCK_SIZE * COARSE_SIZE

BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN

COARSE_CELLS = tuple((r, c) for r in range(COARSE_SIZE) for c in range(COARSE_SIZE))
BLUE_COUNT_CHOICES = tuple(range(2, 8))
MARK_COUNT_CHOICES = (2, 3, 4)
GEN_RETRY_LIMIT = 400


def _block_slice(block_r: int, block_c: int) -> tuple[slice, slice]:
    r0 = int(block_r) * BLOCK_SIZE
    c0 = int(block_c) * BLOCK_SIZE
    return slice(r0, r0 + BLOCK_SIZE), slice(c0, c0 + BLOCK_SIZE)


def _marker_offset(block_r: int, block_c: int) -> tuple[int, int]:
    # Fix the marker's in-block slot from the coarse coordinates so the generator
    # does not introduce irrelevant degrees of freedom that would collapse outputs.
    return int(block_r), int(block_c)


def _compress_blue_pattern(grid: np.ndarray) -> np.ndarray:
    pattern = np.zeros((COARSE_SIZE, COARSE_SIZE), dtype=bool)
    for block_r, block_c in COARSE_CELLS:
        rs, cs = _block_slice(block_r, block_c)
        pattern[block_r, block_c] = bool(np.any(grid[rs, cs] == BLUE))
    return pattern


def _marker_presence(grid: np.ndarray) -> np.ndarray:
    marks = np.zeros((COARSE_SIZE, COARSE_SIZE), dtype=bool)
    for block_r, block_c in COARSE_CELLS:
        rs, cs = _block_slice(block_r, block_c)
        block = grid[rs, cs]
        red_count = int(np.count_nonzero(block == RED))
        assert red_count <= 1, "expected at most one red marker per coarse block"
        marks[block_r, block_c] = red_count == 1
    return marks


def _template_patch(blue_pattern: np.ndarray) -> np.ndarray:
    return np.where(~blue_pattern, GREEN, BG).astype(int)


def _stamp_output(template_patch: np.ndarray, mark_mask: np.ndarray) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for block_r, block_c in COARSE_CELLS:
        if not mark_mask[block_r, block_c]:
            continue
        paste(
            out,
            template_patch,
            block_r * BLOCK_SIZE,
            block_c * BLOCK_SIZE,
            transparent=BG,
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the blue block layout, complement it, and stamp that template into every marked coarse cell.

    Input structure:
    - The input is a 9x9 black grid implicitly partitioned into a 3x3 array of 3x3 coarse blocks.
    - Some coarse blocks are fully blue except that a single red marker may replace one pixel inside the block.
    - Some other coarse blocks are otherwise black and may also contain one red marker pixel.

    Rule:
    - Compress the grid to a 3x3 coarse pattern: a coarse cell is filled iff its 3x3 block contains any blue pixel, ignoring red markers.
    - Complement that 3x3 pattern; complemented filled cells become the stamp template.
    - Detect which coarse blocks contain a red marker.
    - Produce a 9x9 black output and, for each marked coarse block, write the complemented 3x3 template into that block using green for filled template cells and black for empty ones.

    Key insight:
    - The blue distribution and the red markers encode different information streams in the same grid: blue defines the template, while red only selects the stamp locations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE), "expected a 9x9 ARC grid"
    assert set(map(int, np.unique(g))) <= {BG, BLUE, RED}, "unexpected input color"

    blue_pattern = _compress_blue_pattern(g)
    mark_mask = _marker_presence(g)

    blue_count = int(np.count_nonzero(blue_pattern))
    mark_count = int(np.count_nonzero(mark_mask))
    assert 0 < blue_count < blue_pattern.size, "coarse blue pattern must be mixed"
    assert 2 <= mark_count <= 4, "expected 2 to 4 markers"
    assert bool(np.any(mark_mask & blue_pattern)), "expected a marker in a blue coarse block"
    assert bool(np.any(mark_mask & (~blue_pattern))), "expected a marker in a black coarse block"

    template_patch = _template_patch(blue_pattern)
    return _stamp_output(template_patch, mark_mask)


def _sample_blue_pattern() -> np.ndarray:
    blue_count = random.choice(BLUE_COUNT_CHOICES)
    blue_cells = random.sample(COARSE_CELLS, blue_count)
    blue_pattern = np.zeros((COARSE_SIZE, COARSE_SIZE), dtype=bool)
    for block_r, block_c in blue_cells:
        blue_pattern[block_r, block_c] = True
    return blue_pattern


def _choose_marked_cells(blue_pattern: np.ndarray) -> list[tuple[int, int]] | None:
    blue_cells = [cell for cell in COARSE_CELLS if blue_pattern[cell]]
    black_cells = [cell for cell in COARSE_CELLS if not blue_pattern[cell]]
    mark_count = random.choice(MARK_COUNT_CHOICES)

    min_blue_marks = max(1, mark_count - len(black_cells))
    max_blue_marks = min(len(blue_cells), mark_count - 1)
    if min_blue_marks > max_blue_marks:
        return None

    blue_mark_count = random.randint(min_blue_marks, max_blue_marks)
    black_mark_count = mark_count - blue_mark_count
    marked = random.sample(blue_cells, blue_mark_count)
    marked += random.sample(black_cells, black_mark_count)
    random.shuffle(marked)
    return marked


def _render_input(blue_pattern: np.ndarray, marked_cells: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)

    for block_r, block_c in COARSE_CELLS:
        if not blue_pattern[block_r, block_c]:
            continue
        rs, cs = _block_slice(block_r, block_c)
        grid[rs, cs] = BLUE

    for block_r, block_c in marked_cells:
        local_r, local_c = _marker_offset(block_r, block_c)
        grid[block_r * BLOCK_SIZE + local_r, block_c * BLOCK_SIZE + local_c] = RED

    return grid


def _gen_attempt() -> np.ndarray | None:
    blue_pattern = _sample_blue_pattern()
    marked_cells = _choose_marked_cells(blue_pattern)
    if marked_cells is None:
        return None

    grid = _render_input(blue_pattern, marked_cells)
    out = main(grid)

    if np.count_nonzero(out == GREEN) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential stamp input")
