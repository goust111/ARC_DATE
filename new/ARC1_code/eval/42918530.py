from collections import defaultdict

import numpy as np

from arc_utils import get_bg
import random


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    assert idx.size > 0

    runs = []
    start = prev = int(idx[0])
    for x in map(int, idx[1:]):
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev))
            start = prev = x
    runs.append((start, prev))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """Complete every same-colored frame with the union of that color's internal motif positions.

    Input structure
    The grid is a rectangular array of equal-size colored frames separated by background rows and columns.
    Each frame has a solid one-pixel border and may contain additional pixels of its own color inside.

    Rule
    For each frame color, collect all relative positions occupied by that color across every frame of that
    color, then repaint those positions into every frame with that color.

    Key insight
    Repeated frame colors are duplicate motif carriers: incomplete copies are completed by sharing the
    relative colored-cell mask within the color group.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    row_runs = _runs(np.any(g != bg, axis=1))
    col_runs = _runs(np.any(g != bg, axis=0))
    frame_h = row_runs[0][1] - row_runs[0][0] + 1
    frame_w = col_runs[0][1] - col_runs[0][0] + 1
    assert frame_h == frame_w
    assert frame_h >= 3
    assert all(r1 - r0 + 1 == frame_h for r0, r1 in row_runs)
    assert all(c1 - c0 + 1 == frame_w for c0, c1 in col_runs)

    masks_by_color: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros((frame_h, frame_w), dtype=bool)
    )
    frames: list[tuple[int, int, int]] = []

    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, block[block != bg]))
            assert len(colors) == 1
            color = colors.pop()
            assert np.all(block[0, :] == color)
            assert np.all(block[-1, :] == color)
            assert np.all(block[:, 0] == color)
            assert np.all(block[:, -1] == color)

            masks_by_color[color] |= block == color
            frames.append((r0, c0, color))

    out = g.copy()
    for r0, c0, color in frames:
        mask = masks_by_color[color]
        out[r0 : r0 + frame_h, c0 : c0 + frame_w][mask] = color
    return out

# --------------- generator (o5x) ---------------
# Frame lattice geometry.
BACKGROUND, FRAME_SIZE, INTERIOR_SIZE, CELL_STRIDE, GRID_PADDING = 0, 5, 3, 6, 1
# Canvas and palette sampling bounds.
FRAME_GRID_COUNTS, TOTAL_FRAME_CHOICES = range(2, 5), (6, 8, 9, 12, 16)
COLOR_COUNT_MIN, COLOR_COUNT_MAX, COLOR_POOL = 4, 7, range(1, 10)
# Color groups, motif density, and retries.
MIN_MULTI_GROUPS, MAX_MULTI_GROUPS, MAX_GROUP_SIZE, MAX_SINGLE_FRAME_COLORS = 1, 5, 5, 4
MOTIF_FILL_MIN, MOTIF_FILL_MAX, MAX_ATTEMPTS = 0, 5, 300
DIMS_BY_TOTAL = {
    t: [(r, c) for r in FRAME_GRID_COUNTS for c in FRAME_GRID_COUNTS if r * c == t] for t in TOTAL_FRAME_CHOICES
}

def _multi_group_bounds(total_frames: int, n_colors: int) -> tuple[int, int]:
    extra = total_frames - n_colors
    min_multi = max(MIN_MULTI_GROUPS, n_colors - MAX_SINGLE_FRAME_COLORS,
                    (extra + MAX_GROUP_SIZE - 2) // (MAX_GROUP_SIZE - 1))
    return min_multi, min(MAX_MULTI_GROUPS, n_colors, extra)

def _gen_group_sizes(total_frames: int, n_colors: int) -> list[int]:
    extra = total_frames - n_colors
    min_multi, max_multi = _multi_group_bounds(total_frames, n_colors)
    n_multi = random.randint(min_multi, max_multi)

    sizes = [1] * n_colors
    multi_idxs = random.sample(range(n_colors), n_multi)
    for idx in multi_idxs:
        sizes[idx] += 1
    extra -= n_multi

    while extra:
        expandable = [idx for idx in multi_idxs if sizes[idx] < MAX_GROUP_SIZE]
        idx = random.choice(expandable)
        add = random.randint(1, min(extra, MAX_GROUP_SIZE - sizes[idx]))
        sizes[idx] += add
        extra -= add

    random.shuffle(sizes)
    return sizes

def _gen_motif(fill_count: int) -> np.ndarray:
    motif = np.zeros((INTERIOR_SIZE, INTERIOR_SIZE), dtype=bool)
    motif.flat[random.sample(range(INTERIOR_SIZE * INTERIOR_SIZE), fill_count)] = True
    return motif

def _frame_origin(cell: tuple[int, int]) -> tuple[int, int]:
    return GRID_PADDING + cell[0] * CELL_STRIDE, GRID_PADDING + cell[1] * CELL_STRIDE

def _gen_paint_frame(grid: np.ndarray, r0: int, c0: int, color: int, motif: np.ndarray) -> None:
    r1, c1 = r0 + FRAME_SIZE, c0 + FRAME_SIZE
    grid[r0:r1, c0:c1] = BACKGROUND
    grid[r0, c0:c1] = color
    grid[r1 - 1, c0:c1] = color
    grid[r0:r1, c0] = color
    grid[r0:r1, c1 - 1] = color
    grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1][motif] = color

def _choose_layout() -> tuple[int, int, list[int], list[int]]:
    total_frames = random.choice(list(DIMS_BY_TOTAL))
    nr, nc = random.choice(DIMS_BY_TOTAL[total_frames])
    valid_color_counts = []
    for count in range(COLOR_COUNT_MIN, min(COLOR_COUNT_MAX, total_frames - 1) + 1):
        min_multi, max_multi = _multi_group_bounds(total_frames, count)
        if min_multi <= max_multi:
            valid_color_counts.append(count)
    n_colors = random.choice(valid_color_counts)
    colors = random.sample(COLOR_POOL, n_colors)
    return nr, nc, colors, _gen_group_sizes(total_frames, n_colors)

def _assign_colors(nr: int, nc: int, colors: list[int], sizes: list[int]):
    positions = [(i, j) for i in range(nr) for j in range(nc)]
    random.shuffle(positions)
    frames_by_color: dict[int, list[tuple[int, int]]] = {}
    assignment: dict[tuple[int, int], int] = {}
    cursor = 0
    for color, size in zip(colors, sizes):
        cells = positions[cursor : cursor + size]
        cursor += size
        frames_by_color[color] = cells
        for cell in cells:
            assignment[cell] = color
    return frames_by_color, assignment

def _choose_motifs(colors: list[int], frames_by_color: dict[int, list[tuple[int, int]]]):
    """Choose canonical motifs, forcing at least one repeated color to be informative."""
    fill_counts = {color: random.randint(MOTIF_FILL_MIN, MOTIF_FILL_MAX) for color in colors}
    repeated_colors = [color for color, cells in frames_by_color.items() if len(cells) > 1]
    if repeated_colors and all(fill_counts[color] == 0 for color in repeated_colors):
        fill_counts[random.choice(repeated_colors)] = random.randint(1, MOTIF_FILL_MAX)
    return {color: _gen_motif(fill_counts[color]) for color in colors}

def _build_input_output(nr: int, nc: int, assignment, frames_by_color, motifs):
    out = np.zeros((GRID_PADDING + nr * CELL_STRIDE, GRID_PADDING + nc * CELL_STRIDE), dtype=int)
    for cell, color in assignment.items():
        _gen_paint_frame(out, *_frame_origin(cell), color, motifs[color])
    inp = out.copy()
    for color, cells in frames_by_color.items():
        if len(cells) <= 1 or not motifs[color].any():
            continue
        carrier = random.choice(cells)
        for cell in cells:
            if cell == carrier:
                continue
            r0, c0 = _frame_origin(cell)
            inp[r0 + 1:r0 + 1 + INTERIOR_SIZE, c0 + 1:c0 + 1 + INTERIOR_SIZE] = BACKGROUND
    return inp, out

def _gen_attempt() -> np.ndarray | None:
    nr, nc, colors, sizes = _choose_layout()
    frames_by_color, assignment = _assign_colors(nr, nc, colors, sizes)
    motifs = _choose_motifs(colors, frames_by_color)
    inp, out = _build_input_output(nr, nc, assignment, frames_by_color, motifs)
    if not np.array_equal(main(inp), out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
