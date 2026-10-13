from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

_CROSS_FILLS = {
    (0, "mid"): RED,
    ("mid", 0): YELLOW,
    ("mid", "mid"): PINK,
    ("mid", -1): GREEN,
    (-1, "mid"): BLUE,
}

_GEN_ATTEMPTS = 300
_VALID_N = (3, 4, 5)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_u_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, ORANGE, TEAL}, "unexpected colors"

    frame_box = _bbox(g == ORANGE)
    assert frame_box is not None, "expected one orange U frame"
    r0, c0, r1, c1 = frame_box

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert 6 <= h <= 8, "frame height out of range"
    assert 3 <= w <= 5, "frame width out of range"

    expected = np.zeros((h, w), dtype=bool)
    expected[:-1, 0] = True
    expected[:-1, -1] = True
    expected[-1, :] = True
    assert np.array_equal(patch == ORANGE, expected), "orange pixels must form one open-top U"

    assert np.all(g[:r0, :] == BLACK)
    assert np.all(g[r1 + 1 :, :] == BLACK)
    assert np.all(g[r0 : r1 + 1, :c0] == BLACK)
    assert np.all(g[r0 : r1 + 1, c1 + 1 :] == BLACK)

    interior = patch[:-1, 1:-1]
    assert interior.size > 0, "frame must have non-empty interior"
    assert set(map(int, np.unique(interior))) <= {BLACK, TEAL}, "interior can only contain black or teal"

    n = 0
    seen_teal = False
    for row in interior:
        is_black = bool(np.all(row == BLACK))
        is_teal = bool(np.all(row == TEAL))
        assert is_black or is_teal, "each interior row must be uniformly empty or filled"
        if is_black and not seen_teal:
            n += 1
        else:
            seen_teal = True
            assert is_teal, "black rows cannot appear below the liquid"

    assert n in _VALID_N, "top empty-row count must be 3, 4, or 5"
    assert n < interior.shape[0], "there must be at least one filled liquid row"
    return frame_box, n


def _build_output(n: int) -> np.ndarray:
    size = 2 * int(n) + 1
    out = np.full((size, size), MAROON, dtype=int)
    out[1::2, 1::2] = BLACK

    mid = n // 2
    for (rr, cc), color in _CROSS_FILLS.items():
        cell_r = mid if rr == "mid" else (n - 1 if rr == -1 else rr)
        cell_c = mid if cc == "mid" else (n - 1 if cc == -1 else cc)
        out[2 * cell_r + 1, 2 * cell_c + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the empty liquid headspace inside an orange U-frame and convert it into a colored partition grid.

    Input structure:
    - The grid is a black background containing exactly one orange U-shaped frame with wall thickness 1.
    - The frame is open at the top, closed on the left, right, and bottom, and its interior contains only black or teal.
    - Teal liquid occupies a contiguous block of full interior rows from the bottom upward, leaving 3 to 5 full black rows above it.

    Rule:
    - Find the orange U-frame and inspect only its interior cells (between the side walls and above the bottom wall).
    - Count the number N of consecutive black interior rows from the top before the teal liquid starts.
    - Create a (2N+1) by (2N+1) maroon separator grid, which defines an N by N lattice of 1 by 1 black cells.
    - Fill the lattice cross positions with fixed colors: top-middle red, middle-left yellow, center pink, middle-right green, and bottom-middle blue.

    Key insight:
    - The liquid height is not copied directly; its remaining empty headspace determines the lattice dimension, and then a fixed plus pattern is painted relative to that derived partition.
    """
    _frame_box, n = _parse_u_frame(grid)
    return _build_output(n)


def _render_input(n: int, frame_h: int, frame_w: int, grid_h: int, grid_w: int) -> np.ndarray:
    assert n in _VALID_N
    assert 6 <= frame_h <= 8
    assert 3 <= frame_w <= 5
    assert frame_h >= n + 2
    assert grid_h <= 30 and grid_w <= 30

    grid = np.zeros((grid_h, grid_w), dtype=int)
    r0 = random.randint(1, grid_h - frame_h - 1)
    c0 = random.randint(1, grid_w - frame_w - 1)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid[r0:r1, c0] = ORANGE
    grid[r0:r1, c1] = ORANGE
    grid[r1, c0 : c1 + 1] = ORANGE

    interior_h = frame_h - 1
    fill_rows = interior_h - n
    grid[r0 + n : r1, c0 + 1 : c1] = TEAL
    assert fill_rows >= 1
    return grid


def _check_output_semantics(out: np.ndarray, n: int) -> None:
    expected = _build_output(n)
    assert np.array_equal(out, expected), "output differs from the intended partition grid"

    mid = n // 2
    required = {
        (0, mid): RED,
        (mid, 0): YELLOW,
        (mid, mid): PINK,
        (mid, n - 1): GREEN,
        (n - 1, mid): BLUE,
    }
    cells = out[1::2, 1::2]
    for (r, c), color in required.items():
        assert int(cells[r, c]) == color
    for r in range(n):
        for c in range(n):
            if (r, c) not in required:
                assert int(cells[r, c]) == BLACK
    assert np.all(out[::2, :] == MAROON)
    assert np.all(out[:, ::2] == MAROON)


def _semantic_mutation_check(grid: np.ndarray, n: int) -> None:
    if n == min(_VALID_N):
        return

    g2 = grid.copy()
    frame_box, parsed_n = _parse_u_frame(g2)
    assert parsed_n == n
    r0, c0, _r1, c1 = frame_box
    mutate_row = r0 + n - 1
    g2[mutate_row, c0 + 1 : c1] = TEAL

    mutated_n = _parse_u_frame(g2)[1]
    assert mutated_n == n - 1, "changing the first liquid row should reduce N by one"
    assert np.array_equal(main(g2), _build_output(n - 1))


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_N)
    frame_h = random.randint(max(6, n + 2), 8)
    frame_w = random.randint(3, 5)

    grid_h = random.randint(frame_h + 2, min(12, 30))
    grid_w = random.randint(frame_w + 2, min(10, 30))
    if grid_h < frame_h + 2 or grid_w < frame_w + 2:
        return None

    inp = _render_input(n, frame_h, frame_w, grid_h, grid_w)
    try:
        frame_box, parsed_n = _parse_u_frame(inp)
        assert parsed_n == n
        r0, c0, r1, c1 = frame_box
        interior = inp[r0:r1, c0 + 1 : c1]
        assert interior.shape == (frame_h - 1, frame_w - 2)
        assert np.all(interior[:n, :] == BLACK)
        assert np.all(interior[n:, :] == TEAL)

        out = main(inp)
        _check_output_semantics(out, n)
        _semantic_mutation_check(inp, n)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
