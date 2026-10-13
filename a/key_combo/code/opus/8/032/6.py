from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import draw_line, get_bg


BG = 0
FILL = 4
INPUT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
CORNERS = ("tl", "tr", "bl", "br")
INFORMATIVE_CORNERS = {"tr", "bl"}
GEN_MIN_N = 4
GEN_MAX_N = 10
GEN_MAX_ATTEMPTS = 300


def _corner_l_mask(n: int, corner: str) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if corner == "tl":
        mask[0, : n - 1] = True
        mask[: n - 1, 0] = True
    elif corner == "tr":
        mask[0, 1:] = True
        mask[: n - 1, n - 1] = True
    elif corner == "bl":
        mask[n - 1, : n - 1] = True
        mask[1:, 0] = True
    elif corner == "br":
        mask[n - 1, 1:] = True
        mask[1:, n - 1] = True
    else:
        raise ValueError(f"unknown corner {corner}")
    return mask


def _match_corner(mask: np.ndarray) -> str:
    n = int(mask.shape[0])
    matches = [corner for corner in CORNERS if np.array_equal(mask, _corner_l_mask(n, corner))]
    assert len(matches) == 1, "expected exactly one corner L of leg length n-1"
    return matches[0]


def _triangle_geometry(out_size: int, side: int, corner: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int, int]:
    if corner == "tl":
        anchor = (0, 0)
        h_end = (0, side - 1)
        v_end = (side - 1, 0)
        return anchor, h_end, v_end, 1, 1
    if corner == "tr":
        anchor = (0, out_size - 1)
        h_end = (0, out_size - side)
        v_end = (side - 1, out_size - 1)
        return anchor, h_end, v_end, 1, -1
    if corner == "bl":
        anchor = (out_size - 1, 0)
        h_end = (out_size - 1, side - 1)
        v_end = (out_size - side, 0)
        return anchor, h_end, v_end, -1, 1
    if corner == "br":
        anchor = (out_size - 1, out_size - 1)
        h_end = (out_size - 1, out_size - side)
        v_end = (out_size - side, out_size - 1)
        return anchor, h_end, v_end, -1, -1
    raise ValueError(f"unknown corner {corner}")


def _expected_output(n: int, corner: str, fg: int) -> np.ndarray:
    out_size = 3 * n - 2
    side = 2 * (n - 1)
    out = np.full((out_size, out_size), BG, dtype=int)
    anchor, _h_end, _v_end, row_dir, col_dir = _triangle_geometry(out_size, side, corner)
    rv, cv = anchor

    for t in range(side):
        out[rv, cv + col_dir * t] = fg
        out[rv + row_dir * t, cv] = fg
        out[rv + row_dir * (side - 1 - t), cv + col_dir * t] = fg

    row_iter = range(0, side) if row_dir == 1 else range(out_size - side, out_size)
    col_iter = range(0, side) if col_dir == 1 else range(out_size - side, out_size)
    for r in row_iter:
        for c in col_iter:
            x = row_dir * (r - rv)
            y = col_dir * (c - cv)
            if x > 0 and y > 0 and x + y < side - 1 and (r - rv + c - cv) % 2 == 0:
                out[r, c] = FILL

    return out


def _origin_phase_output(n: int, corner: str, fg: int) -> np.ndarray:
    out = _expected_output(n, corner, fg)
    out[out == FILL] = BG

    out_size = int(out.shape[0])
    side = 2 * (n - 1)
    anchor, _h_end, _v_end, row_dir, col_dir = _triangle_geometry(out_size, side, corner)
    rv, cv = anchor
    row_iter = range(0, side) if row_dir == 1 else range(out_size - side, out_size)
    col_iter = range(0, side) if col_dir == 1 else range(out_size - side, out_size)
    for r in row_iter:
        for c in col_iter:
            x = row_dir * (r - rv)
            y = col_dir * (c - cv)
            if x > 0 and y > 0 and x + y < side - 1 and (r + c) % 2 == 0:
                out[r, c] = FILL
    return out


def _interior_counts(out: np.ndarray, n: int, corner: str) -> tuple[int, int]:
    out_size = int(out.shape[0])
    side = 2 * (n - 1)
    anchor, _h_end, _v_end, row_dir, col_dir = _triangle_geometry(out_size, side, corner)
    rv, cv = anchor
    yellow = 0
    black = 0
    row_iter = range(0, side) if row_dir == 1 else range(out_size - side, out_size)
    col_iter = range(0, side) if col_dir == 1 else range(out_size - side, out_size)
    for r in row_iter:
        for c in col_iter:
            x = row_dir * (r - rv)
            y = col_dir * (c - cv)
            if x > 0 and y > 0 and x + y < side - 1:
                if int(out[r, c]) == FILL:
                    yellow += 1
                elif int(out[r, c]) == BG:
                    black += 1
    return yellow, black


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a corner L into a same-color triangle outline and parity-fill the triangle interior from the right-angle vertex.

    Input structure:
    - The input is a square black grid with exactly one non-black foreground color.
    - The foreground is a one-pixel-thick L placed in exactly one corner.
    - Each L arm has length n-1 in an n x n grid, so it starts at the corner and stops one cell short of the opposite end.

    Rule:
    - Detect which corner the L occupies, then create a black output canvas of size (3n-2) x (3n-2).
    - In the same output corner, draw a same-color right-triangle outline whose two edge-aligned legs each have length 2(n-1).
    - Fill only the strict interior of that triangle: cells with even parity relative to the triangle's right-angle vertex become yellow, and odd-parity interior cells stay black.

    Key insight:
    - The L does double duty: it identifies both the triangle corner and the parity anchor, so changing the corner can change the checkerboard phase even when the triangle size stays the same.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n_rows, n_cols = map(int, g.shape)
    assert n_rows == n_cols, "expected a square grid"
    n = n_rows
    assert n >= GEN_MIN_N and 3 * n - 2 <= 30, "input size must expand to an ARC-valid output"

    bg = int(get_bg(g))
    assert bg == BG, "expected black background"
    colors = sorted(set(map(int, g.flatten())))
    assert FILL not in colors, "input must not contain yellow"
    fg_colors = [color for color in colors if color != BG]
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    mask = g == fg
    assert int(mask.sum()) == 2 * (n - 1) - 1, "expected the exact corner-L pixel count"
    corner = _match_corner(mask)

    out_size = 3 * n - 2
    side = 2 * (n - 1)
    out = np.full((out_size, out_size), BG, dtype=int)
    anchor, h_end, v_end, row_dir, col_dir = _triangle_geometry(out_size, side, corner)

    draw_line(out, anchor[0], anchor[1], end_r=h_end[0], end_c=h_end[1], color=fg)
    draw_line(out, anchor[0], anchor[1], end_r=v_end[0], end_c=v_end[1], color=fg)
    draw_line(out, h_end[0], h_end[1], end_r=v_end[0], end_c=v_end[1], color=fg)

    rv, cv = anchor
    row_iter = range(0, side) if row_dir == 1 else range(out_size - side, out_size)
    col_iter = range(0, side) if col_dir == 1 else range(out_size - side, out_size)
    for r in row_iter:
        for c in col_iter:
            x = row_dir * (r - rv)
            y = col_dir * (c - cv)
            if x > 0 and y > 0 and x + y < side - 1 and (r - rv + c - cv) % 2 == 0:
                out[r, c] = FILL

    return out


def _draw_input_l(n: int, fg: int, corner: str) -> np.ndarray:
    grid = np.full((n, n), BG, dtype=int)
    grid[_corner_l_mask(n, corner)] = fg
    return grid


def _sample_n(corner: str) -> int:
    if corner in INFORMATIVE_CORNERS:
        return random.choice([n for n in range(GEN_MIN_N, GEN_MAX_N + 1) if n % 2 == 0])
    return random.randint(GEN_MIN_N, GEN_MAX_N)


def _is_semantically_valid(inp: np.ndarray) -> bool:
    n = int(inp.shape[0])
    fg = int(next(iter(sorted(set(map(int, inp.flatten())) - {BG}))))
    corner = _match_corner(inp == fg)

    expected = _expected_output(n, corner, fg)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    yellow, black = _interior_counts(actual, n, corner)
    if yellow == 0 or black == 0:
        return False

    if corner in INFORMATIVE_CORNERS and np.array_equal(actual, _origin_phase_output(n, corner, fg)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        corner = random.choice(CORNERS)
        n = _sample_n(corner)
        fg = random.choice(INPUT_COLORS)
        inp = _draw_input_l(n, fg, corner)
        if _is_semantically_valid(inp):
            return inp
    raise RuntimeError("generate_input: failed to build a valid L-to-triangle example")
