from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox


PALETTE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GRID_COLOR = Color.GREY
INPUT_SIZE_RANGE = (8, 12)
SIDE_CHOICES = (2, 3)
K_CHOICES = (3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _read_palette_run(values: np.ndarray, start: int) -> list[int]:
    start = int(start)
    out: list[int] = []
    idx = start
    while idx < len(values) and int(values[idx]) != Color.BLACK:
        color = int(values[idx])
        assert color in PALETTE_COLORS, "palette colors must be blue/red/green/yellow"
        out.append(color)
        idx += 1
    assert out, "expected a non-empty contiguous palette run"
    assert np.all(values[idx:] == Color.BLACK), "palette run must be unique and contiguous"
    return out


def _parse_input(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_box = bbox(g == GRID_COLOR)
    assert grey_box is not None, "expected one grey anchor square"
    r0, c0, r1, c1 = grey_box
    assert (r0, c0) == (0, 0), "anchor square must touch the top-left corner"
    side = r1 - r0 + 1
    assert c1 - c0 + 1 == side, "anchor must be square"
    assert np.all(g[:side, :side] == GRID_COLOR), "anchor must be solid"
    assert int(np.count_nonzero(g == GRID_COLOR)) == side * side, "grey may only appear in the anchor square"

    col_palette = _read_palette_run(g[0], side)
    row_palette = _read_palette_run(g[:, 0], side)
    assert len(row_palette) == len(col_palette), "row and column palettes must have the same length"

    expected_nonzero = np.zeros_like(g, dtype=bool)
    expected_nonzero[:side, :side] = True
    expected_nonzero[0, side : side + len(col_palette)] = True
    expected_nonzero[side : side + len(row_palette), 0] = True
    assert np.all(g[~expected_nonzero] == Color.BLACK), "input may only contain the anchor and two palettes"

    return side, row_palette, col_palette


def _build_output(side: int, row_palette: list[int], col_palette: list[int]) -> np.ndarray:
    k = len(row_palette)
    assert k == len(col_palette) and k > 0

    # K chambers of size side, separated by K-1 one-cell grid lines.
    size = k * side + (k - 1)
    out = np.full((size, size), Color.BLACK, dtype=int)

    for line_idx in range(k - 1):
        line = side + line_idx * (side + 1)
        out[line, :] = GRID_COLOR
        out[:, line] = GRID_COLOR

    for r_idx, row_color in enumerate(row_palette):
        r0 = r_idx * (side + 1)
        for c_idx, col_color in enumerate(col_palette):
            if row_color != col_color:
                continue
            c0 = c_idx * (side + 1)
            out[r0 : r0 + side, c0 : c0 + side] = row_color

    return out


def _match_matrix(row_palette: list[int], col_palette: list[int]) -> np.ndarray:
    return np.array([[int(r == c) for c in col_palette] for r in row_palette], dtype=int)


def _valid_palettes(row_palette: list[int], col_palette: list[int]) -> bool:
    k = len(row_palette)
    if len(col_palette) != k or k not in K_CHOICES:
        return False
    if any(color not in PALETTE_COLORS for color in row_palette + col_palette):
        return False
    if len(set(row_palette)) < 2 or len(set(col_palette)) < 2:
        return False
    if len(set(row_palette)) == k and len(set(col_palette)) == k:
        return False

    matches = _match_matrix(row_palette, col_palette)
    total_matches = int(matches.sum())
    if total_matches <= 0 or total_matches >= k * k:
        return False
    if not any(0 < int(matches[r].sum()) < k for r in range(k)):
        return False
    if not any(0 < int(matches[:, c].sum()) < k for c in range(k)):
        return False

    filled_colors = {
        row_palette[r]
        for r in range(k)
        for c in range(k)
        if row_palette[r] == col_palette[c]
    }
    if len(filled_colors) < 2:
        return False

    return True


def _sample_palettes(k: int) -> tuple[list[int], list[int]]:
    for _ in range(300):
        active = random.sample(PALETTE_COLORS, k=random.choice((2, 3)))
        row_palette = [random.choice(active) for _ in range(k)]
        col_palette = [random.choice(active) for _ in range(k)]
        if _valid_palettes(row_palette, col_palette):
            return row_palette, col_palette
    raise RuntimeError("failed to sample non-degenerate palettes")


def _validate_generated_case(grid: np.ndarray) -> None:
    side, row_palette, col_palette = _parse_input(grid)
    assert side in SIDE_CHOICES
    assert _valid_palettes(row_palette, col_palette)

    out = _build_output(side, row_palette, col_palette)
    assert out.ndim == 2 and max(out.shape) <= 30

    k = len(row_palette)
    matches = _match_matrix(row_palette, col_palette)
    for r in range(k):
        for c in range(k):
            patch = out[r * (side + 1) : r * (side + 1) + side, c * (side + 1) : c * (side + 1) + side]
            if matches[r, c]:
                assert np.all(patch == row_palette[r])
            else:
                assert np.all(patch == Color.BLACK)

    line_positions = [side + i * (side + 1) for i in range(k - 1)]
    for pos in line_positions:
        assert np.all(out[pos, :] == GRID_COLOR)
        assert np.all(out[:, pos] == GRID_COLOR)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a grey lattice from the corner square and fill only row/column palette matches.

    Input structure:
    - A black information grid contains one solid grey square in the top-left corner.
    - Starting immediately to the right of that square, the top row contains a contiguous one-pixel column palette.
    - Starting immediately below the square, the left column contains a contiguous one-pixel row palette of the same length.

    Rule:
    - Read the grey square side length S and use grey as the lattice-line color.
    - Read the top-row palette from left to right and the left-column palette from top to bottom.
    - Create an output with K square chambers per side, each chamber sized SxS and separated by one-cell grey grid lines.
    - Fill chamber (i, j) with the shared palette color iff row_palette[i] == col_palette[j]; otherwise leave it black.

    Key insight:
    - The corner square controls the lattice geometry, but the chamber contents come from a 2D equality test between the row and column palettes.
    """
    side, row_palette, col_palette = _parse_input(grid)
    return _build_output(side, row_palette, col_palette)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        side = random.choice(SIDE_CHOICES)
        k_candidates = [k for k in K_CHOICES if k * side + (k - 1) <= 15]
        k = random.choice(k_candidates)
        row_palette, col_palette = _sample_palettes(k)

        min_h = max(INPUT_SIZE_RANGE[0], side + k)
        min_w = max(INPUT_SIZE_RANGE[0], side + k)
        height = random.randint(min_h, INPUT_SIZE_RANGE[1])
        width = random.randint(min_w, INPUT_SIZE_RANGE[1])

        grid = np.full((height, width), Color.BLACK, dtype=int)
        grid[:side, :side] = GRID_COLOR
        grid[0, side : side + k] = np.array(col_palette, dtype=int)
        grid[side : side + k, 0] = np.array(row_palette, dtype=int)

        try:
            _validate_generated_case(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
