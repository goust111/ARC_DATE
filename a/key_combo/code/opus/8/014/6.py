from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


BG = Color.BLACK
HEIGHT_RANGE = (12, 18)
WIDTH_RANGE = (9, 13)
BLANK_COUNT_RANGE = (2, 4)
PALETTE_COUNT_RANGE = (2, 3)
GEN_RETRY_LIMIT = 400
PALETTE_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]


def _cyclic_row(width: int, palette: list[int], run_len: int) -> np.ndarray:
    return np.array([palette[(c // run_len) % len(palette)] for c in range(width)], dtype=int)


def _blank_rows(grid: np.ndarray, bg: int) -> np.ndarray:
    return np.flatnonzero(np.all(grid == int(bg), axis=1))


def _palette_colors(grid: np.ndarray, bg: int) -> list[int]:
    return [int(v) for v in grid[:, -1] if int(v) != int(bg)]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill fully blank rows with horizontal palette runs whose run length equals the blank-row count.

    Input structure:
    - A black background grid contains scattered non-background pattern rows and a rightmost palette column.
    - The palette column holds 2-3 single non-black pixels from top to bottom; its other cells are black.
    - Some other rows are completely black across every column, including the palette column.

    Rule:
    - Identify every row that is entirely black, counting the palette column as part of the row test.
    - Let N be the number of such blank rows.
    - Read the non-black pixels in the rightmost column from top to bottom as the palette sequence.
    - Erase the rightmost column to black.
    - Repaint each formerly blank row from left to right by cycling through the palette, repeating each palette color for N consecutive cells before advancing to the next color.
    - Leave every non-blank row unchanged apart from the palette-column erase.

    Key insight:
    - The same set of all-black rows encodes both target locations and the horizontal run length for the cyclic fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    blank_rows = _blank_rows(g, bg)
    run_len = int(len(blank_rows))
    assert run_len > 0, "expected at least one fully blank row"

    palette = _palette_colors(g, bg)
    assert palette, "expected at least one palette color in the rightmost column"

    out = g.copy()
    out[:, -1] = bg
    fill_row = _cyclic_row(out.shape[1], palette, run_len)
    out[blank_rows, :] = fill_row
    return out


def _paint_pattern_row(row: np.ndarray, colors: list[int]) -> None:
    width = row.shape[0]
    num_segments = 1 if width <= 5 else random.randint(1, 2)
    occupied = np.zeros(width, dtype=bool)

    for _ in range(num_segments):
        options: list[tuple[int, int]] = []
        max_len = min(4, width)
        for seg_len in range(1, max_len + 1):
            for start in range(width - seg_len + 1):
                left_ok = start == 0 or not occupied[start - 1]
                right_ok = start + seg_len == width or not occupied[start + seg_len]
                if left_ok and right_ok and not occupied[start : start + seg_len].any():
                    options.append((start, seg_len))
        if not options:
            break
        start, seg_len = random.choice(options)
        seg_colors = random.sample(colors, k=min(len(colors), 2))
        if len(seg_colors) == 1 or random.random() < 0.55:
            row[start : start + seg_len] = seg_colors[0]
        else:
            for i in range(seg_len):
                row[start + i] = seg_colors[i % len(seg_colors)]
        occupied[start : start + seg_len] = True

    if not occupied.any():
        c = random.randrange(width)
        row[c] = random.choice(colors)


def _build_input() -> np.ndarray:
    height = random.randint(*HEIGHT_RANGE)
    blank_count = random.randint(*BLANK_COUNT_RANGE)
    min_width = max(WIDTH_RANGE[0], 2 * blank_count + 1)
    width = random.randint(min_width, WIDTH_RANGE[1])
    palette_count = random.randint(*PALETTE_COUNT_RANGE)

    color_order = random.sample(PALETTE_POOL, k=random.randint(5, len(PALETTE_POOL)))
    palette = color_order[:palette_count]
    pattern_colors = color_order[palette_count:]
    if len(pattern_colors) < 2:
        pattern_colors = [c for c in PALETTE_POOL if c not in palette][:2]

    grid = np.full((height, width), BG, dtype=int)

    blank_rows = set(random.sample(range(height), k=blank_count))
    nonblank_rows = [r for r in range(height) if r not in blank_rows]
    palette_rows = sorted(random.sample(nonblank_rows, k=palette_count))
    palette_only_count = random.randint(1, palette_count)
    palette_only_rows = set(random.sample(palette_rows, k=palette_only_count))

    for r, color in zip(palette_rows, palette):
        grid[r, -1] = int(color)

    for r in range(height):
        if r in blank_rows or r in palette_only_rows:
            continue
        _paint_pattern_row(grid[r, :-1], pattern_colors)

    return grid


def _expected_output(inp: np.ndarray) -> np.ndarray:
    bg = BG
    blank_rows = _blank_rows(inp, bg)
    palette = _palette_colors(inp, bg)
    out = inp.copy()
    out[:, -1] = bg
    out[blank_rows, :] = _cyclic_row(inp.shape[1], palette, len(blank_rows))
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    blank_rows = _blank_rows(inp, BG)
    palette = _palette_colors(inp, BG)
    if not (BLANK_COUNT_RANGE[0] <= len(blank_rows) <= BLANK_COUNT_RANGE[1]):
        return False
    if not (PALETTE_COUNT_RANGE[0] <= len(palette) <= PALETTE_COUNT_RANGE[1]):
        return False

    fill_row = _cyclic_row(inp.shape[1], palette, len(blank_rows))
    if len(set(map(int, fill_row.tolist()))) < 2:
        return False
    if not np.array_equal(out[blank_rows[0]], fill_row):
        return False

    palette_only_rows = np.flatnonzero((inp[:, -1] != BG) & np.all(inp[:, :-1] == BG, axis=1))
    if palette_only_rows.size == 0:
        return False
    wrong_blank_rows = np.flatnonzero(np.all(inp[:, :-1] == BG, axis=1))
    if wrong_blank_rows.size == blank_rows.size:
        return False
    wrong = inp.copy()
    wrong[:, -1] = BG
    wrong[wrong_blank_rows, :] = fill_row
    if np.array_equal(wrong, out):
        return False

    solid = out.copy()
    solid[blank_rows, :] = palette[0]
    if np.array_equal(solid, out):
        return False

    shifted_palette = palette[1:] + palette[:1]
    shifted = out.copy()
    shifted[blank_rows, :] = _cyclic_row(inp.shape[1], shifted_palette, len(blank_rows))
    if np.array_equal(shifted, out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _build_input()
        expected = _expected_output(inp)
        actual = main(inp)
        if np.array_equal(actual, expected) and _semantic_checks(inp, actual):
            return inp
    raise RuntimeError("failed to generate a valid input")
