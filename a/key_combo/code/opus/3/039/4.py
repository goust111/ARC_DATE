from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


BLOCK = 3
LOW_SIZE = 5
LOW_BG = Color.BLACK
NOISE = Color.GREY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]

MIN_COLORED = 8
MAX_COLORED = 12
GEN_RETRIES = 400


def _block_color(block: np.ndarray) -> int:
    counts = Counter(int(v) for v in block.ravel() if int(v) != NOISE)
    if not counts:
        return LOW_BG

    black_count = counts.get(LOW_BG, 0)
    color = LOW_BG
    best = black_count
    for candidate, count in counts.items():
        if candidate == LOW_BG:
            continue
        if count > best:
            color = int(candidate)
            best = int(count)
    return int(color)


def _compress_macroblocks(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % BLOCK == 0 and w % BLOCK == 0, "grid must decompose into 3x3 macroblocks"

    out = np.full((h // BLOCK, w // BLOCK), LOW_BG, dtype=int)
    for br in range(h // BLOCK):
        for bc in range(w // BLOCK):
            block = g[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            out[br, bc] = _block_color(block)
    return out


def _apply_gravity(lowres: np.ndarray) -> np.ndarray:
    low = np.asarray(lowres, dtype=int)
    h, w = low.shape
    out = np.full((h, w), LOW_BG, dtype=int)
    for c in range(w):
        falling = low[:, c][low[:, c] != LOW_BG]
        if falling.size:
            out[h - falling.size :, c] = falling
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise 3x3 macroblocks, compress them to a coarse grid, then drop the coarse pixels to the bottom of each column.

    Input structure:
    - The grid is a black-background picture whose height and width are multiples of 3, so it splits into 3x3 macroblocks.
    - Each macroblock is either a mostly-black empty block with optional grey noise, or a mostly-solid colored block with 1-2 grey noise pixels.
    - The real block colors are blue, red, green, or yellow, and colored macroblocks may float above empty macroblocks in the same coarse column.

    Rule:
    - For each 3x3 macroblock, ignore grey pixels and identify the dominant remaining color.
    - Compress the macroblock lattice to one coarse pixel per block: black if black is still dominant, otherwise the block's dominant non-black color.
    - In the compressed grid, move every non-black pixel downward within its own column, preserving top-to-bottom order and stacking at the bottom.

    Key insight:
    - Grey never represents a real coarse pixel; it only corrupts macroblocks before compression, and the actual motion happens only after the grid has been reduced to low resolution.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == LOW_BG, "expected black to remain the dominant background"
    return _apply_gravity(_compress_macroblocks(g))


def _sample_column_counts(rng: random.Random, total: int) -> tuple[list[int], int]:
    while True:
        active_cols = rng.randint(3, LOW_SIZE)
        cols = rng.sample(range(LOW_SIZE), active_cols)
        demo_col = rng.choice(cols)

        counts = [0] * LOW_SIZE
        max_demo = min(len(PALETTE), total - (active_cols - 1))
        if max_demo < 2:
            continue
        counts[demo_col] = rng.randint(2, max_demo)
        for c in cols:
            if c != demo_col:
                counts[c] = 1

        remaining = total - sum(counts)
        expandable = [c for c in cols if counts[c] < len(PALETTE)]
        while remaining > 0 and expandable:
            c = rng.choice(expandable)
            counts[c] += 1
            remaining -= 1
            if counts[c] == len(PALETTE):
                expandable.remove(c)

        if remaining == 0:
            return counts, demo_col


def _sample_rows(rng: random.Random, count: int, *, must_float: bool) -> list[int]:
    all_rows = [list(rows) for rows in itertools.combinations(range(LOW_SIZE), count)]
    if must_float:
        bottom_stack = list(range(LOW_SIZE - count, LOW_SIZE))
        all_rows = [rows for rows in all_rows if rows != bottom_stack]
    return list(rng.choice(all_rows))


def _build_lowres(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    total = rng.randint(MIN_COLORED, MAX_COLORED)
    counts, demo_col = _sample_column_counts(rng, total)

    low = np.full((LOW_SIZE, LOW_SIZE), LOW_BG, dtype=int)
    changed_cols = 0
    for c, count in enumerate(counts):
        if count == 0:
            continue
        rows = _sample_rows(rng, count, must_float=(c == demo_col))
        if rows != list(range(LOW_SIZE - count, LOW_SIZE)):
            changed_cols += 1
        colors = rng.sample(PALETTE, count)
        for r, color in zip(rows, colors):
            low[r, c] = int(color)

    out = _apply_gravity(low)
    assert changed_cols >= 1
    assert not np.array_equal(low, out)
    return low, out


def _noise_plan(low: np.ndarray, rng: random.Random) -> dict[tuple[int, int], int]:
    colored = [(r, c) for r in range(LOW_SIZE) for c in range(LOW_SIZE) if int(low[r, c]) != LOW_BG]
    empty = [(r, c) for r in range(LOW_SIZE) for c in range(LOW_SIZE) if int(low[r, c]) == LOW_BG]
    assert colored and empty

    grey_counts: dict[tuple[int, int], int] = {}

    color_two = rng.choice(colored)
    grey_counts[color_two] = 2
    if len(colored) > 1:
        color_one_choices = [pos for pos in colored if pos != color_two]
        grey_counts[rng.choice(color_one_choices)] = 1

    empty_two = rng.choice(empty)
    grey_counts[empty_two] = 2

    for pos in colored:
        grey_counts.setdefault(pos, rng.choice([1, 2]))
    for pos in empty:
        grey_counts.setdefault(pos, rng.choices([0, 1, 2], weights=[6, 3, 2], k=1)[0])

    if all(grey_counts[pos] > 0 for pos in empty):
        grey_counts[rng.choice(empty)] = 0
    return grey_counts


def _paint_block(color: int, grey_count: int, rng: random.Random) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), int(color), dtype=int)
    for idx in rng.sample(range(BLOCK * BLOCK), grey_count):
        block[idx // BLOCK, idx % BLOCK] = NOISE
    return block


def _expand_lowres(low: np.ndarray, rng: random.Random) -> np.ndarray:
    grey_counts = _noise_plan(low, rng)
    inp = np.full((LOW_SIZE * BLOCK, LOW_SIZE * BLOCK), LOW_BG, dtype=int)
    for r in range(LOW_SIZE):
        for c in range(LOW_SIZE):
            color = int(low[r, c])
            block = _paint_block(color, grey_counts[(r, c)], rng)
            inp[r * BLOCK : (r + 1) * BLOCK, c * BLOCK : (c + 1) * BLOCK] = block
    return inp


def _has_noisy_empty_block(grid: np.ndarray) -> bool:
    for br in range(LOW_SIZE):
        for bc in range(LOW_SIZE):
            block = grid[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            if np.count_nonzero(block == LOW_BG) >= 7 and np.count_nonzero(block == NOISE) > 0:
                return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    low, expected = _build_lowres(rng)
    inp = _expand_lowres(low, rng)
    if not _has_noisy_empty_block(inp):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid noisy gravity sample")
