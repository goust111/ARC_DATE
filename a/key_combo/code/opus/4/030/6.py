import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
BLOCK = 3
PANEL = 9
GRID_SHAPE = (PANEL * 2 + 1, PANEL)
SIGNAL_COLORS = (1, 2, 3, 4, 6, 8, 9)
GEN_MAX_ATTEMPTS = 400


def _real_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != GREY)


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == ORANGE)]
    assert len(rows) == 1, "expected one full orange separator row"
    sep = rows[0]
    assert sep > 0 and sep < grid.shape[0] - 1
    assert sep == grid.shape[0] - sep - 1, "panels must have equal height"
    return sep


def _overlay_panels(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    assert top.shape == bottom.shape
    out = np.full_like(top, BLACK)

    top_real = _real_mask(top)
    bottom_real = _real_mask(bottom)
    assert not np.any(top_real & bottom_real), "panel signals must not collide"

    grey_mask = ((top == GREY) | (bottom == GREY)) & ~(top_real | bottom_real)
    out[grey_mask] = GREY
    out[bottom_real] = bottom[bottom_real]
    out[top_real] = top[top_real]
    return out


def _compress_block(block: np.ndarray) -> int:
    colors = [int(c) for c in np.unique(block) if int(c) not in (BLACK, GREY)]
    assert len(colors) <= 1, "each 3x3 block may contain at most one real color"
    if not colors:
        return BLACK
    color = colors[0]
    return color if int(np.count_nonzero(block == color)) >= 5 else BLACK


def _summarize(merged: np.ndarray) -> np.ndarray:
    h, w = merged.shape
    assert h % BLOCK == 0 and w % BLOCK == 0
    out = np.full((h // BLOCK, w // BLOCK), BLACK, dtype=int)
    for br in range(out.shape[0]):
        for bc in range(out.shape[1]):
            block = merged[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            out[br, bc] = _compress_block(block)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the two separated panels, then denoise-compress each 3x3 block to reveal the hidden pattern.

    Input structure:
    - The grid contains a full orange separator row splitting two equal-height black panels.
    - Each panel is partitioned into 3x3 blocks and contains sparse colored signal pixels plus grey noise pixels.
    - Matching top/bottom coordinates never contain conflicting non-black, non-grey colors.

    Rule:
    - Locate the orange separator row and overlay the top and bottom panels cellwise.
    - At each coordinate, keep a top real color if present, otherwise a bottom real color, otherwise grey if either panel has grey there, otherwise black.
    - Compress the merged panel by turning each 3x3 block into one output cell: ignore grey noise and output the block's unique real color only when that color forms the recovered signal majority (at least five cells); otherwise output black.

    Key insight:
    - Each panel holds only a sub-threshold fragment of the true 3x3 block signal, so compressing either panel alone loses the pattern; the hidden image appears only after overlay restores enough same-color cells inside every active block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape, "panels must be equal-sized"
    assert top.shape[0] % BLOCK == 0 and top.shape[1] % BLOCK == 0, "panel size must split into 3x3 blocks"

    merged = _overlay_panels(top, bottom)
    return _summarize(merged)


def _choose_output() -> np.ndarray:
    out = np.full((3, 3), BLACK, dtype=int)
    active_count = random.randint(4, 8)
    active_positions = random.sample(range(9), active_count)

    distinct_color_count = random.randint(2, min(5, active_count))
    palette = random.sample(SIGNAL_COLORS, distinct_color_count)
    assignments = palette.copy()
    while len(assignments) < active_count:
        assignments.append(random.choice(palette))
    random.shuffle(assignments)

    for index, color in zip(active_positions, assignments):
        out[index // 3, index % 3] = color
    return out


def _sample_signal_split(color: int) -> tuple[np.ndarray, np.ndarray]:
    top_block = np.full((BLOCK, BLOCK), BLACK, dtype=int)
    bottom_block = np.full((BLOCK, BLOCK), BLACK, dtype=int)
    cells = list(range(BLOCK * BLOCK))

    signal_size = random.randint(5, 7)
    signal_cells = set(random.sample(cells, signal_size))
    top_count = random.randint(max(1, signal_size - 4), min(4, signal_size - 1))
    top_signal = set(random.sample(sorted(signal_cells), top_count))
    bottom_signal = signal_cells - top_signal
    assert 1 <= len(top_signal) <= 4
    assert 1 <= len(bottom_signal) <= 4

    for idx in top_signal:
        top_block[idx // BLOCK, idx % BLOCK] = color
    for idx in bottom_signal:
        bottom_block[idx // BLOCK, idx % BLOCK] = color

    free_cells = sorted(set(cells) - signal_cells)
    top_noise_count = random.randint(0, min(2, len(free_cells)))
    top_noise = set(random.sample(free_cells, top_noise_count))
    bottom_noise_count = random.randint(0, min(2, len(free_cells)))
    bottom_noise = set(random.sample(free_cells, bottom_noise_count))

    for idx in top_noise:
        top_block[idx // BLOCK, idx % BLOCK] = GREY
    for idx in bottom_noise:
        bottom_block[idx // BLOCK, idx % BLOCK] = GREY
    return top_block, bottom_block


def _sample_background_split() -> tuple[np.ndarray, np.ndarray]:
    top_block = np.full((BLOCK, BLOCK), BLACK, dtype=int)
    bottom_block = np.full((BLOCK, BLOCK), BLACK, dtype=int)
    cells = list(range(BLOCK * BLOCK))

    top_noise_count = random.randint(0, 2)
    bottom_noise_count = random.randint(0, 2)
    top_noise = set(random.sample(cells, top_noise_count))
    bottom_noise = set(random.sample(cells, bottom_noise_count))

    for idx in top_noise:
        top_block[idx // BLOCK, idx % BLOCK] = GREY
    for idx in bottom_noise:
        bottom_block[idx // BLOCK, idx % BLOCK] = GREY
    return top_block, bottom_block


def _render_input(target: np.ndarray) -> np.ndarray:
    top = np.full((PANEL, PANEL), BLACK, dtype=int)
    bottom = np.full((PANEL, PANEL), BLACK, dtype=int)

    for br in range(3):
        for bc in range(3):
            color = int(target[br, bc])
            if color == BLACK:
                top_block, bottom_block = _sample_background_split()
            else:
                top_block, bottom_block = _sample_signal_split(color)
            r0 = br * BLOCK
            c0 = bc * BLOCK
            top[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = top_block
            bottom[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = bottom_block

    return np.vstack([top, np.full((1, PANEL), ORANGE, dtype=int), bottom])


def _validate_generated(grid: np.ndarray, target: np.ndarray) -> bool:
    if grid.shape != GRID_SHAPE:
        return False
    if _find_separator_row(grid) != PANEL:
        return False

    top = grid[:PANEL]
    bottom = grid[PANEL + 1 :]
    if np.any(_real_mask(top) & _real_mask(bottom)):
        return False

    merged = _overlay_panels(top, bottom)
    if not np.array_equal(_summarize(merged), target):
        return False
    if not np.array_equal(main(grid), target):
        return False

    non_black_colors = {int(c) for c in target.flat if int(c) != BLACK}
    if len(non_black_colors) < 2:
        return False

    active_blocks = 0
    grey_blocks = 0
    grey_background_blocks = 0
    for br in range(3):
        for bc in range(3):
            r0 = br * BLOCK
            c0 = bc * BLOCK
            merged_block = merged[r0 : r0 + BLOCK, c0 : c0 + BLOCK]
            top_block = top[r0 : r0 + BLOCK, c0 : c0 + BLOCK]
            bottom_block = bottom[r0 : r0 + BLOCK, c0 : c0 + BLOCK]
            target_color = int(target[br, bc])
            merged_colors = [int(c) for c in np.unique(merged_block) if int(c) not in (BLACK, GREY)]
            if len(merged_colors) > 1:
                return False

            if target_color == BLACK:
                if merged_colors:
                    return False
                if np.any(merged_block == GREY):
                    grey_background_blocks += 1
            else:
                active_blocks += 1
                if merged_colors != [target_color]:
                    return False
                merged_count = int(np.count_nonzero(merged_block == target_color))
                top_count = int(np.count_nonzero(top_block == target_color))
                bottom_count = int(np.count_nonzero(bottom_block == target_color))
                if not (5 <= merged_count <= 7):
                    return False
                if not (1 <= top_count <= 4 and 1 <= bottom_count <= 4):
                    return False
                if _compress_block(top_block) != BLACK or _compress_block(bottom_block) != BLACK:
                    return False
                if np.any(merged_block == GREY):
                    grey_blocks += 1

    if not (4 <= active_blocks <= 8):
        return False
    if grey_blocks == 0:
        return False
    if grey_background_blocks == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        target = _choose_output()
        grid = _render_input(target)
        if _validate_generated(grid, target):
            return grid
    raise RuntimeError("failed to generate a valid dual-panel overlay ARC input")
