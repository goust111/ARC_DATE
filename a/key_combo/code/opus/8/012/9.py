from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_ROOT = Path("/home/lixinyu/snap/new")
if str(_UTILS_ROOT) not in sys.path:
    sys.path.append(str(_UTILS_ROOT))

from solvers.arc_utils import Color, find_components


LEFT_SIZE = 5
LEFT_GAP = 1
BAR_COLOR = Color.PINK
PATTERN_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
HEIGHT_RANGE = (5, 14)
RIGHT_ZONE_WIDTH_RANGE = (4, 12)
GEN_RETRY_LIMIT = 300

CORNER_MASK = np.zeros((LEFT_SIZE, LEFT_SIZE), dtype=bool)
CORNER_MASK[0, 0] = True
CORNER_MASK[0, -1] = True
CORNER_MASK[-1, 0] = True
CORNER_MASK[-1, -1] = True

CROSS_MASK = np.zeros((LEFT_SIZE, LEFT_SIZE), dtype=bool)
CROSS_MASK[LEFT_SIZE // 2, :] = True
CROSS_MASK[:, LEFT_SIZE // 2] = True

FILL_MASK = ~(CORNER_MASK | CROSS_MASK)


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= LEFT_SIZE and w >= LEFT_SIZE + LEFT_GAP + 4, "grid too small"

    left = g[:LEFT_SIZE, :LEFT_SIZE]
    colors, counts = np.unique(left, return_counts=True)
    assert len(colors) == 3, "left pattern must use exactly three colors"

    freq_map = {int(color): int(count) for color, count in zip(colors, counts)}
    count_signature = sorted(freq_map.values())
    assert count_signature == [4, 9, 12], f"unexpected left frequencies: {count_signature}"

    role_r = next(color for color, count in freq_map.items() if count == 4)
    role_m = next(color for color, count in freq_map.items() if count == 9)
    role_c = next(color for color, count in freq_map.items() if count == 12)

    assert np.all(left[CORNER_MASK] == role_r), "rare role must occupy the four corners"
    assert np.all(left[CROSS_MASK] == role_m), "middle role must form the full 5x5 cross"
    assert np.all(left[FILL_MASK] == role_c), "common role must fill the remaining cells"

    assert np.all(g[:LEFT_SIZE, LEFT_SIZE] == Color.BLACK), "left/right separator must be black"

    bar_components = find_components(
        g,
        bg=Color.BLACK,
        colors=[BAR_COLOR],
        connectivity=4,
        monochromatic=True,
    )
    assert len(bar_components) == 2, "expected exactly two magenta bars"
    bar_components = sorted(bar_components, key=lambda comp: comp["bbox"][1])
    left_bar, right_bar = bar_components

    for comp in (left_bar, right_bar):
        r0, c0, r1, c1 = comp["bbox"]
        assert c0 == c1, "each bar must be one column wide"
        assert comp["area"] == r1 - r0 + 1, "each bar must be solid and vertical"

    lr0, lc0, lr1, lc1 = left_bar["bbox"]
    rr0, rc0, rr1, rc1 = right_bar["bbox"]
    assert (lr0, lr1) == (rr0, rr1), "bars must be row-aligned"
    assert lc0 > LEFT_SIZE, "left magenta bar must be inside the right zone"
    assert rc0 > lc0 + 1, "bars must leave room for horizontal rays"

    rare_rows = sorted({int(r) for r, _ in np.argwhere(CORNER_MASK)})
    global_rows = sorted({int(r) for r, _ in np.argwhere(left == role_r)})
    assert global_rows == rare_rows == [0, 4], "corner rows must be the top and bottom rows of the 5x5 block"
    assert lr0 <= global_rows[0] and global_rows[-1] <= lr1, "bar span must cover both rare-role rows"

    right_zone = g[:, lc0 : rc0 + 1]
    interior = right_zone[:, 1:-1]
    assert np.all(interior == Color.BLACK), "space between bars must be empty in the input"

    return {
        "height": h,
        "width": w,
        "role_r": int(role_r),
        "role_m": int(role_m),
        "role_c": int(role_c),
        "rows": global_rows,
        "left_bar_col": int(lc0),
        "right_bar_col": int(rc0),
    }


def _validate_output(inp: np.ndarray, out: np.ndarray, info: dict) -> None:
    assert out.shape == inp.shape, "output must keep the input shape"

    expected = np.asarray(inp, dtype=int).copy()
    expected_left = expected[:LEFT_SIZE, :LEFT_SIZE]
    expected_left[CORNER_MASK] = info["role_m"]
    ray_start = info["left_bar_col"] + 1
    ray_end = info["right_bar_col"]
    for row in info["rows"]:
        expected[row, ray_start:ray_end] = info["role_m"]

    assert np.array_equal(out, expected), "output does not match the required recolor-and-ray transformation"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the 5x5 pattern's frequency roles to recolor its corners and drive horizontal rays between the right-side bars.

    Input structure:
    - The upper-left 5x5 block contains exactly three non-black colors in a fixed corner / full-cross / filler layout.
    - A single black separator column follows that block.
    - In the right zone, exactly two magenta one-cell-wide vertical bars share the same row span, and the space between them is otherwise black.

    Rule:
    - Count the three colors inside the 5x5 block and rank them by frequency: 4 cells -> role R, 9 cells -> role M, 12 cells -> role C.
    - Take the global row indices occupied by role R; these are the top and bottom rows of the 5x5 block.
    - Recolor the four role-R corner pixels to role M.
    - On each selected row, draw a horizontal ray of color role M from the cell just right of the left magenta bar to the cell just left of the right magenta bar.

    Key insight:
    - The left motif is an implicit encoder: frequency reveals which color is the source of the row selection and which color supplies the ray and recolor value.
    """
    info = _analyze_input(grid)
    out = np.asarray(grid, dtype=int).copy()
    out_left = out[:LEFT_SIZE, :LEFT_SIZE]
    out_left[CORNER_MASK] = info["role_m"]
    ray_start = info["left_bar_col"] + 1
    ray_end = info["right_bar_col"]
    for row in info["rows"]:
        out[row, ray_start:ray_end] = info["role_m"]

    return out


def _build_input(
    height: int,
    right_zone_width: int,
    role_r: int,
    role_m: int,
    role_c: int,
) -> np.ndarray:
    width = LEFT_SIZE + LEFT_GAP + right_zone_width
    grid = np.zeros((height, width), dtype=int)

    left = np.full((LEFT_SIZE, LEFT_SIZE), role_c, dtype=int)
    left[CROSS_MASK] = role_m
    left[CORNER_MASK] = role_r
    grid[:LEFT_SIZE, :LEFT_SIZE] = left

    left_bar_col = LEFT_SIZE + LEFT_GAP
    right_bar_col = width - 1
    grid[:LEFT_SIZE, left_bar_col] = BAR_COLOR
    grid[:LEFT_SIZE, right_bar_col] = BAR_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    right_zone_width = random.randint(*RIGHT_ZONE_WIDTH_RANGE)
    role_r, role_m, role_c = random.sample(PATTERN_COLORS, 3)

    grid = _build_input(height, right_zone_width, role_r, role_m, role_c)
    info = _analyze_input(grid)
    out = main(grid)
    _validate_output(grid, out, info)

    assert not np.array_equal(grid, out), "task must change the grid"
    assert len(np.unique(grid[:LEFT_SIZE, :LEFT_SIZE])) == 3, "left motif must stay tri-color in the input"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
