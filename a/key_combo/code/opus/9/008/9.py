from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

_RNG = np.random.default_rng()


def _edge_coords(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blue = np.array([(r, 0) for r in range(n)], dtype=int)
    red = np.array([(k, k) for k in range(1, n)], dtype=int)
    green = np.array([(n - 1, c) for c in range(1, n - 1)], dtype=int)
    return blue, red, green


def _interior_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for r in range(1, n - 1):
        mask[r, 1:r] = True
    return mask


def _nearest_manhattan(cell: tuple[int, int], coords: np.ndarray) -> int:
    r, c = cell
    return int(np.min(np.abs(coords[:, 0] - r) + np.abs(coords[:, 1] - c)))


def _validate_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "expected a square grid"
    assert 6 <= h <= 30, "expected side length in [6, 30]"
    assert np.all(g[:, 0] == BLUE), "column 0 must be a full-height blue bar"
    assert np.all(g[:, 1:] == BG), "all non-bar cells must be background"
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the bar into a three-colored triangle and fill its interior by nearest-edge color.

    Input structure:
    - The grid is square with black background everywhere except column 0.
    - Column 0 is a full-height blue bar.
    - No other foreground objects or markers are present.

    Rule:
    - Preserve the blue left edge in column 0.
    - Draw a red main diagonal from the top-left corner to the bottom-right corner,
      but keep the shared top-left corner blue.
    - Draw a green base on the bottom row from column 1 to column n-2; the
      bottom-right corner stays red because it already belongs to the diagonal.
    - The strict interior of the resulting right triangle consists of cells with
      0 < c < r < n - 1.
    - For each interior cell, compute its minimum Manhattan distance to the blue
      edge pixels, the red diagonal pixels, and the green base pixels, then copy
      the color of the nearest edge; ties are resolved blue over red over green.

    Key insight:
    - The outline is only the first stage: the interior is partitioned by a
      nearest-edge Voronoi rule over the triangle's three sides, not by rows,
      columns, or a single flat fill color.
    """
    g = _validate_input(grid)
    n = g.shape[0]

    out = g.copy()
    diag = np.arange(1, n, dtype=int)
    out[diag, diag] = RED
    out[n - 1, 1 : n - 1] = GREEN

    blue_edge, red_edge, green_edge = _edge_coords(n)
    for r, c in np.argwhere(_interior_mask(n)):
        cell = (int(r), int(c))
        d_blue = _nearest_manhattan(cell, blue_edge)
        d_red = _nearest_manhattan(cell, red_edge)
        d_green = _nearest_manhattan(cell, green_edge)
        if d_blue <= d_red and d_blue <= d_green:
            out[r, c] = BLUE
        elif d_red <= d_green:
            out[r, c] = RED
        else:
            out[r, c] = GREEN

    return out


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> None:
    n = inp.shape[0]
    assert out.shape == inp.shape
    assert np.all(out[:, 0] == BLUE)
    assert out[0, 0] == BLUE
    assert out[n - 1, n - 1] == RED
    assert np.all(out[np.arange(1, n), np.arange(1, n)] == RED)
    assert np.all(out[n - 1, 1 : n - 1] == GREEN)
    assert np.all(out[0, 1:] == BG)

    interior = _interior_mask(n)
    blue_edge, red_edge, green_edge = _edge_coords(n)
    seen_inside = set()
    for r, c in np.argwhere(interior):
        cell = (int(r), int(c))
        d_blue = _nearest_manhattan(cell, blue_edge)
        d_red = _nearest_manhattan(cell, red_edge)
        d_green = _nearest_manhattan(cell, green_edge)
        expected = BLUE if d_blue <= d_red and d_blue <= d_green else RED if d_red <= d_green else GREEN
        assert int(out[r, c]) == expected
        seen_inside.add(int(out[r, c]))

    assert seen_inside == {BLUE, RED, GREEN}, "each edge color must own some interior cells"

    outside = ~(interior | (inp == BLUE) | (out == RED) | (out == GREEN))
    assert np.all(out[outside] == BG)


def _sample_size() -> int:
    bands = (
        list(range(6, 11)),
        list(range(11, 19)),
        list(range(19, 31)),
    )
    band = bands[int(_RNG.integers(0, len(bands)))]
    residue = int(_RNG.integers(0, 4))
    choices = [n for n in band if n % 4 == residue]
    if not choices:
        choices = [n for n in range(6, 31) if n % 4 == residue]
    return int(_RNG.choice(choices))


def generate_input() -> np.ndarray:
    n = _sample_size()
    inp = np.zeros((n, n), dtype=int)
    inp[:, 0] = BLUE

    out = main(inp)
    _semantic_check(inp, out)
    return inp
