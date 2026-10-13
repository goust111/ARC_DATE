from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one copied value"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _separator_structure(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected exactly one separator color"
    color, rows, cols = candidates[0]
    return int(color), sorted(rows), sorted(cols)


def _content_bands(limit: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(set(map(int, separators)))
    bounds = [-1, *separators, int(limit)]
    bands: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        start = int(left + 1)
        end = int(right)
        assert end > start, "separator lines must leave nonempty blocks"
        bands.append((start, end))
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the one damaged copy of a repeated block motif and mark its wrong pixels red.

    Input structure:
    - Full same-color separator rows and columns partition the grid into equal-size blocks.
    - Every block is the same multicolor motif except exactly one block, where 1-3 cells differ.
    - The differing cells may be recolored or overwritten by the motif background.

    Rule:
    - Recover the consensus motif by taking the unique majority color at each local block coordinate.
    - Locate the single block that differs from that consensus.
    - Return the damaged block itself, but replace every differing coordinate with red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert not np.any(g == RED), "red is reserved for the output marker"

    separator, sep_rows, sep_cols = _separator_structure(g)
    row_bands = _content_bands(g.shape[0], sep_rows)
    col_bands = _content_bands(g.shape[1], sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected repeated block copies"

    block_heights = {r1 - r0 for r0, r1 in row_bands}
    block_widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(block_heights) == 1 and len(block_widths) == 1, "all blocks must have equal size"
    block_h = block_heights.pop()
    block_w = block_widths.pop()

    blocks = np.empty((len(row_bands), len(col_bands), block_h, block_w), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            block = g[r0:r1, c0:c1]
            assert block.shape == (block_h, block_w)
            blocks[br, bc] = block

    assert not np.any(blocks == separator), "separator color must stay out of block interiors"

    motif = np.empty((block_h, block_w), dtype=int)
    for rr in range(block_h):
        for cc in range(block_w):
            motif[rr, cc] = _unique_mode(blocks[:, :, rr, cc])

    diffs = blocks != motif
    diff_counts = diffs.reshape(len(row_bands), len(col_bands), block_h * block_w).sum(axis=2)
    damaged = np.argwhere(diff_counts > 0)
    assert damaged.shape[0] == 1, "expected exactly one damaged block"

    damaged_r, damaged_c = map(int, damaged[0])
    assert 1 <= int(diff_counts[damaged_r, damaged_c]) <= 3, "damage must affect 1-3 cells"

    out = blocks[damaged_r, damaged_c].copy()
    out[diffs[damaged_r, damaged_c]] = RED
    return out


# --------------- generator ---------------

COLOR_POOL = [0, 1, 3, 4, 5, 6, 7, 8, 9]
BLOCK_ROWS_RANGE = (2, 4)
BLOCK_COLS_RANGE = (2, 4)
BLOCK_H_RANGE = (3, 7)
BLOCK_W_RANGE = (3, 7)
NON_BG_COLOR_RANGE = (2, 4)
EXTRA_SHAPE_RANGE = (1, 3)
DAMAGE_COUNT_MAX = 3
GENERATION_ATTEMPTS = 400
MOTIF_ATTEMPTS = 80


def _draw_random_shape(canvas: np.ndarray, color: int) -> None:
    h, w = map(int, canvas.shape)
    shapes = ["filled_rect", "hseg", "vseg", "point"]
    if h >= 2 and w >= 2:
        shapes.append("outline")
    if h >= 3 or w >= 3:
        shapes.append("plus")

    shape = random.choice(shapes)
    if shape == "filled_rect":
        r0 = random.randrange(h)
        r1 = random.randrange(r0, h)
        c0 = random.randrange(w)
        c1 = random.randrange(c0, w)
        canvas[r0 : r1 + 1, c0 : c1 + 1] = color
        return

    if shape == "outline":
        r0 = random.randrange(h - 1)
        r1 = random.randrange(r0 + 1, h)
        c0 = random.randrange(w - 1)
        c1 = random.randrange(c0 + 1, w)
        canvas[r0 : r1 + 1, c0] = color
        canvas[r0 : r1 + 1, c1] = color
        canvas[r0, c0 : c1 + 1] = color
        canvas[r1, c0 : c1 + 1] = color
        return

    if shape == "hseg":
        r = random.randrange(h)
        c0 = random.randrange(w)
        c1 = random.randrange(c0, w)
        canvas[r, c0 : c1 + 1] = color
        return

    if shape == "vseg":
        c = random.randrange(w)
        r0 = random.randrange(h)
        r1 = random.randrange(r0, h)
        canvas[r0 : r1 + 1, c] = color
        return

    if shape == "plus":
        r = random.randrange(h)
        c = random.randrange(w)
        up = random.randint(0, r)
        down = random.randint(0, h - 1 - r)
        left = random.randint(0, c)
        right = random.randint(0, w - 1 - c)
        if up == down == left == right == 0:
            right = 1 if c + 1 < w else 0
            left = 1 if right == 0 and c > 0 else left
            down = 1 if right == left == 0 and r + 1 < h else down
            up = 1 if right == left == down == 0 and r > 0 else up
        canvas[r - up : r + down + 1, c] = color
        canvas[r, c - left : c + right + 1] = color
        return

    canvas[random.randrange(h), random.randrange(w)] = color


def _make_motif(block_h: int, block_w: int, bg: int, colors: list[int]) -> np.ndarray | None:
    area = int(block_h * block_w)
    for _ in range(MOTIF_ATTEMPTS):
        motif = np.full((block_h, block_w), bg, dtype=int)

        for color in colors:
            _draw_random_shape(motif, color)

        for _ in range(random.randint(*EXTRA_SHAPE_RANGE)):
            _draw_random_shape(motif, random.choice(colors))

        if random.random() < 0.45:
            _draw_random_shape(motif, bg)

        used = set(map(int, motif[motif != bg].reshape(-1)))
        non_bg = int(np.count_nonzero(motif != bg))
        touched_rows = int(np.count_nonzero(np.any(motif != bg, axis=1)))
        touched_cols = int(np.count_nonzero(np.any(motif != bg, axis=0)))

        if used != set(colors):
            continue
        if non_bg < max(5, len(colors) + 2):
            continue
        if non_bg > int(0.8 * area):
            continue
        if touched_rows < 2 or touched_cols < 2:
            continue

        return motif

    return None


def _render_grid(
    motif: np.ndarray,
    separator: int,
    block_rows: int,
    block_cols: int,
    damaged_pos: tuple[int, int],
    damaged_block: np.ndarray,
) -> np.ndarray:
    block_h, block_w = map(int, motif.shape)
    height = block_rows * block_h + (block_rows - 1)
    width = block_cols * block_w + (block_cols - 1)
    grid = np.full((height, width), separator, dtype=int)

    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = br * (block_h + 1)
            c0 = bc * (block_w + 1)
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = motif

    dr, dc = damaged_pos
    r0 = dr * (block_h + 1)
    c0 = dc * (block_w + 1)
    grid[r0 : r0 + block_h, c0 : c0 + block_w] = damaged_block
    return grid


def _damage_block(
    motif: np.ndarray,
    bg: int,
    non_bg_colors: list[int],
    damage_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    block = motif.copy()
    mask = np.zeros_like(motif, dtype=bool)

    all_cells = [(r, c) for r in range(motif.shape[0]) for c in range(motif.shape[1])]
    non_bg_cells = [(r, c) for r, c in all_cells if int(motif[r, c]) != bg]

    if damage_mode == "erase":
        assert non_bg_cells, "erase damage needs a non-background cell"
        damage_count = random.randint(1, min(DAMAGE_COUNT_MAX, len(non_bg_cells)))
        cells = random.sample(non_bg_cells, damage_count)
        for r, c in cells:
            block[r, c] = bg
            mask[r, c] = True
        return block, mask

    damage_count = random.randint(1, min(DAMAGE_COUNT_MAX, len(all_cells)))
    cells = random.sample(all_cells, damage_count)
    for r, c in cells:
        original = int(block[r, c])
        replacements = [color for color in non_bg_colors if color != original]
        assert replacements, "recolor damage needs an alternate visible color"
        block[r, c] = random.choice(replacements)
        mask[r, c] = True
    return block, mask


def _gen_attempt() -> np.ndarray | None:
    block_rows = random.randint(*BLOCK_ROWS_RANGE)
    block_cols = random.randint(*BLOCK_COLS_RANGE)
    block_h = random.randint(*BLOCK_H_RANGE)
    block_w = random.randint(*BLOCK_W_RANGE)

    non_bg_count = random.randint(*NON_BG_COLOR_RANGE)
    bg, separator, *motif_colors = random.sample(COLOR_POOL, 2 + non_bg_count)
    motif = _make_motif(block_h, block_w, bg, motif_colors)
    if motif is None:
        return None

    damage_mode = random.choice(["erase", "recolor"])
    if damage_mode == "erase" and int(np.count_nonzero(motif != bg)) == 0:
        return None

    damaged_block, damage_mask = _damage_block(motif, bg, motif_colors, damage_mode)
    if not np.any(damage_mask):
        return None

    expected = damaged_block.copy()
    expected[damage_mask] = RED

    damaged_pos = (random.randrange(block_rows), random.randrange(block_cols))
    inp = _render_grid(motif, separator, block_rows, block_cols, damaged_pos, damaged_block)

    try:
        actual = main(inp)
    except Exception:
        return None

    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
