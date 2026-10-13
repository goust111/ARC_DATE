from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


TILE_SHAPE = (4, 4)
STRIP_REPEATS = 5
MARKER_COUNT_CHOICES = (2, 3)
MARKER_COLORS = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
MAX_ATTEMPTS = 400


def _markers_in_reading_order(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != Color.BLACK)
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    markers.sort(key=lambda item: (item[0], item[1]))
    return markers


def _validate_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D tile"
    assert g.shape == TILE_SHAPE, f"expected a {TILE_SHAPE[0]}x{TILE_SHAPE[1]} tile"
    assert g.shape[0] <= 30 and g.shape[1] * STRIP_REPEATS <= 30, "output must stay within ARC limits"

    markers = _markers_in_reading_order(g)
    assert 2 <= len(markers) <= 3, "expected 2-3 marker pixels"

    rows = [r for r, _c, _color in markers]
    cols = [c for _r, c, _color in markers]
    colors = [color for _r, _c, color in markers]

    assert len(set(rows)) == len(rows), "marker rows must be distinct"
    assert len(set(cols)) == len(cols), "marker columns must be distinct"
    assert len(set(colors)) == len(colors), "marker colors must be distinct"
    assert all(color not in (Color.BLACK, Color.RED) for color in colors), "red is reserved for crossings"
    return g


def _build_strip(tile: np.ndarray) -> np.ndarray:
    mirrored = np.fliplr(tile)
    blocks = [tile if i % 2 == 0 else mirrored for i in range(STRIP_REPEATS)]
    return np.concatenate(blocks, axis=1)


def _paint_rays_and_crossings(strip: np.ndarray) -> np.ndarray:
    h, w = strip.shape
    markers = _markers_in_reading_order(strip)
    coverage = np.zeros((h, w), dtype=np.int64)

    for r, c, color in markers:
        bit = np.int64(1 << color)
        coverage[r, :] |= bit
        coverage[:, c] |= bit

    out = np.full_like(strip, Color.BLACK)
    for r in range(h):
        for c in range(w):
            bits = int(coverage[r, c])
            if bits == 0:
                continue
            if bits & (bits - 1):
                out[r, c] = Color.RED
            else:
                out[r, c] = bits.bit_length() - 1

    for r, c, color in markers:
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a sparse tile into an alternating mirrored strip, then color row/column ray crossings red.

    Input structure:
    - The input is a 4x4 black tile containing 2-3 single-cell marker pixels.
    - Marker pixels use distinct non-black, non-red colors and occupy distinct rows and columns.

    Rule:
    - Build a 5-block horizontal strip: blocks 1, 3, and 5 are the original tile, while blocks 2 and 4 are its left-right mirror.
    - In the strip, every marker extends through its full row and full column.
    - Any cell reached by markers of two or more different colors becomes red, while cells reached by exactly one color take that color.
    - All original and mirrored marker pixels keep their own colors.

    Key insight:
    - Horizontal mirroring changes only the copied marker columns, so duplicated markers create new column guides whose crossings with other colors reveal the final red pattern.
    """
    tile = _validate_tile(grid)
    strip = _build_strip(tile)
    return _paint_rays_and_crossings(strip)


def _choose_marker_columns(count: int) -> list[int]:
    if count == 2:
        symmetric_pairs = [(0, 3), (1, 2)]
        asymmetric_pairs = [
            pair for pair in itertools.combinations(range(TILE_SHAPE[1]), 2) if pair not in symmetric_pairs
        ]
        pair = random.choice(symmetric_pairs if random.random() < 0.5 else asymmetric_pairs)
        cols = list(pair)
    else:
        cols = random.sample(range(TILE_SHAPE[1]), count)
    random.shuffle(cols)
    return cols


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(MARKER_COUNT_CHOICES)
    rows = random.sample(range(TILE_SHAPE[0]), count)
    cols = _choose_marker_columns(count)
    colors = random.sample(MARKER_COLORS, count)
    random.shuffle(rows)
    random.shuffle(colors)

    tile = np.zeros(TILE_SHAPE, dtype=int)
    for r, c, color in zip(rows, cols, colors):
        tile[r, c] = color

    if np.array_equal(tile, np.fliplr(tile)):
        return None

    out = main(tile)
    if out.shape != (TILE_SHAPE[0], TILE_SHAPE[1] * STRIP_REPEATS):
        return None
    if not np.any(out == Color.RED):
        return None
    if not np.any(out == Color.BLACK):
        return None

    mirrored_block_1 = out[:, TILE_SHAPE[1] : 2 * TILE_SHAPE[1]]
    mirrored_block_2 = out[:, 3 * TILE_SHAPE[1] : 4 * TILE_SHAPE[1]]
    if not np.any(mirrored_block_1 == Color.RED):
        return None
    if not np.any(mirrored_block_2 == Color.RED):
        return None

    return tile


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        tile = _gen_attempt()
        if tile is not None:
            return tile
    raise RuntimeError("failed to generate a valid input for puzzle 6")
