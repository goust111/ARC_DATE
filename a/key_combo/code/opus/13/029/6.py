from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLOCK = 3
GREY = 5
MARK_GREEN = 3
PALETTE = (1, 2, 4)
SEPARATOR_CHOICES = (0, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 400


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    best = counts.most_common()
    return int(best[0][0])


def _compress_blocks(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h % BLOCK != 0 or w % BLOCK != 0:
        raise ValueError("grid must split into 3x3 blocks")

    out = np.zeros((h // BLOCK, w // BLOCK), dtype=int)
    for br in range(h // BLOCK):
        for bc in range(w // BLOCK):
            block = g[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            visible = block[block != GREY]
            out[br, bc] = 0 if visible.size == 0 else _mode_color(visible)
    return out


def _bands(separators: list[int], limit: int) -> list[tuple[int, int]] | None:
    if not separators:
        return None

    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators + [limit]:
        if sep <= start:
            return None
        bands.append((start, sep))
        start = sep + 1
    if start != limit + 1:
        return None

    sizes = {end - begin for begin, end in bands}
    if len(sizes) != 1:
        return None
    return bands


def _recover_consensus(low: np.ndarray) -> np.ndarray:
    candidates: list[tuple[int, int, np.ndarray]] = []
    h, w = low.shape

    for color in map(int, np.unique(low)):
        row_seps = [r for r in range(h) if np.all(low[r, :] == color)]
        col_seps = [c for c in range(w) if np.all(low[:, c] == color)]
        row_bands = _bands(row_seps, h)
        col_bands = _bands(col_seps, w)
        if row_bands is None or col_bands is None:
            continue
        if len(row_bands) < 2 or len(col_bands) < 2:
            continue

        tile_h = row_bands[0][1] - row_bands[0][0]
        tile_w = col_bands[0][1] - col_bands[0][0]
        blocks = np.empty((len(row_bands), len(col_bands), tile_h, tile_w), dtype=int)
        valid = True
        for br, (r0, r1) in enumerate(row_bands):
            for bc, (c0, c1) in enumerate(col_bands):
                block = low[r0:r1, c0:c1]
                if block.shape != (tile_h, tile_w):
                    valid = False
                    break
                blocks[br, bc] = block
            if not valid:
                break
        if not valid:
            continue

        motif = np.zeros((tile_h, tile_w), dtype=int)
        for rr in range(tile_h):
            for cc in range(tile_w):
                motif[rr, cc] = _mode_color(blocks[:, :, rr, cc])

        diffs = blocks != motif
        diff_counts = diffs.reshape(len(row_bands), len(col_bands), tile_h * tile_w).sum(axis=2)
        damaged = np.argwhere(diff_counts > 0)
        if damaged.shape[0] != 1:
            continue

        dr, dc = map(int, damaged[0])
        damage_mask = diffs[dr, dc]
        if not np.any(damage_mask):
            continue
        if not np.all(blocks[dr, dc][damage_mask] == 0):
            continue
        if not np.all(motif[damage_mask] != 0):
            continue

        block_count = len(row_bands) * len(col_bands)
        candidates.append((block_count, tile_h * tile_w, motif))

    if not candidates:
        raise ValueError("failed to recover repeated consensus tile")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _same_color_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords = [(r, c)]
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
                        coords.append((nr, nc))
            comps.append((color, coords))
    return comps


def _mark_small_components(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for _color, coords in _same_color_components(out):
        if len(coords) <= 2:
            for r, c in coords:
                out[r, c] = MARK_GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the noisy 3x3 block lattice, repair the repeated motif by majority vote, then recolor tiny components green.

    Input structure:
    - The grid dimensions are multiples of 3, so the input can be split into a lattice of 3x3 blocks.
    - Each block has one representative color, with optional grey noise pixels that must be ignored.
    - After compression, the low-resolution grid contains repeated equal-size motif copies separated by full solid rows and columns.
    - Exactly one compressed copy is damaged by replacing some non-black motif pixels with black.

    Rule:
    - Collapse every 3x3 block to one color by ignoring grey pixels and taking the block's majority visible color.
    - Detect the separator rows and columns in the compressed grid and slice out the equal-size motif copies.
    - Recover the consensus motif by majority vote at each local coordinate across all copies.
    - Find every same-color 4-connected non-black component in the repaired motif and recolor components of area 1 or 2 to green.

    Key insight:
    - The task is a three-stage pipeline: block-level denoising reveals the repeated structure, copy-level voting repairs the damaged instance, and only then does connected-component size become reliable enough for the final recoloring.
    """
    low = _compress_blocks(np.asarray(grid, dtype=int))
    motif = _recover_consensus(low)
    return _mark_small_components(motif)


def _random_motif(height: int, width: int, colors: tuple[int, ...], separator: int) -> np.ndarray | None:
    area = height * width
    fg_target = random.randint(max(5, area // 3), max(6, int(round(area * 0.65))))

    motif = np.zeros((height, width), dtype=int)
    fg_indices = random.sample(range(area), fg_target)
    for idx in fg_indices:
        r, c = divmod(idx, width)
        motif[r, c] = random.choice(colors)

    missing = [color for color in colors if not np.any(motif == color)]
    if missing:
        for color, idx in zip(missing, random.sample(range(area), len(missing))):
            r, c = divmod(idx, width)
            motif[r, c] = color

    comps = _same_color_components(motif)
    areas = [len(coords) for _color, coords in comps]
    if len(comps) < 2 or not any(a <= 2 for a in areas) or not any(a >= 3 for a in areas):
        return None
    if np.count_nonzero(motif) < 5:
        return None
    if separator == 0 and (np.any(np.all(motif == 0, axis=0)) or np.any(np.all(motif == 0, axis=1))):
        return None
    return motif


def _choose_motif() -> tuple[np.ndarray, int] | None:
    for _ in range(120):
        height = random.randint(3, 4)
        width = random.randint(3, 4)
        color_count = random.randint(2, 3)
        colors = tuple(random.sample(PALETTE, color_count))
        separator = random.choice(SEPARATOR_CHOICES)
        if separator in colors:
            continue
        motif = _random_motif(height, width, colors, separator)
        if motif is not None:
            return motif, separator
    return None


def _build_low_res(motif: np.ndarray, separator: int) -> tuple[np.ndarray, np.ndarray] | None:
    tile_h, tile_w = motif.shape
    copy_rows = 2
    copy_cols = 2
    low_h = copy_rows * tile_h + (copy_rows - 1)
    low_w = copy_cols * tile_w + (copy_cols - 1)
    if max(low_h * BLOCK, low_w * BLOCK) > 30:
        return None

    low = np.full((low_h, low_w), separator, dtype=int)
    for br in range(copy_rows):
        for bc in range(copy_cols):
            r0 = br * (tile_h + 1)
            c0 = bc * (tile_w + 1)
            low[r0 : r0 + tile_h, c0 : c0 + tile_w] = motif

    copies = [(0, 0), (0, 1), (1, 0), (1, 1)]
    damaged_copy = random.choice(copies)
    fg_cells = np.argwhere(motif != 0)
    damage_count = random.randint(1, min(3, len(fg_cells)))
    damage_pick = fg_cells[random.sample(range(len(fg_cells)), damage_count)]
    base_r = damaged_copy[0] * (tile_h + 1)
    base_c = damaged_copy[1] * (tile_w + 1)
    for dr, dc in damage_pick:
        low[base_r + int(dr), base_c + int(dc)] = 0

    expected = _mark_small_components(motif)
    return low, expected


def _expand_with_noise(low: np.ndarray) -> np.ndarray:
    h, w = low.shape
    big = np.repeat(np.repeat(low, BLOCK, axis=0), BLOCK, axis=1)
    block_cells = [(br, bc) for br in range(h) for bc in range(w)]
    random.shuffle(block_cells)
    noisy_blocks = random.randint(max(1, len(block_cells) // 5), max(2, len(block_cells) // 2))

    for br, bc in block_cells[:noisy_blocks]:
        r0 = br * BLOCK
        c0 = bc * BLOCK
        rr = random.randint(r0, r0 + BLOCK - 1)
        cc = random.randint(c0, c0 + BLOCK - 1)
        big[rr, cc] = GREY
    return big


def _gen_attempt() -> np.ndarray | None:
    chosen = _choose_motif()
    if chosen is None:
        return None
    motif, separator = chosen

    built = _build_low_res(motif, separator)
    if built is None:
        return None
    low, expected = built
    inp = _expand_with_noise(low)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
