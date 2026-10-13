import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        ORANGE = 7


BG = int(Color.BLACK)
ORANGE = int(Color.ORANGE)
PALETTE_COLORS = [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW)]
PALETTE_SET = set(PALETTE_COLORS)
CORNERS = ("tl", "tr", "bl", "br")


def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        err2 = 2 * err
        if err2 > -dc:
            err -= dc
            r += sr
        if err2 < dr:
            err += dr
            c += sc
    return points


def _candidate_l_coords(n: int, corner: str, leg_len: int) -> set[tuple[int, int]]:
    if corner == "tl":
        return {(0, c) for c in range(leg_len)} | {(r, 0) for r in range(leg_len)}
    if corner == "tr":
        return {(0, n - 1 - c) for c in range(leg_len)} | {(r, n - 1) for r in range(leg_len)}
    if corner == "bl":
        return {(n - 1, c) for c in range(leg_len)} | {(n - 1 - r, 0) for r in range(leg_len)}
    if corner == "br":
        return {(n - 1, n - 1 - c) for c in range(leg_len)} | {(n - 1 - r, n - 1) for r in range(leg_len)}
    raise ValueError(f"unknown corner {corner}")


def _decode_l_shape(grid: np.ndarray) -> tuple[str, int]:
    orange_coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)}
    assert orange_coords, "expected one orange L-shape"
    n = int(grid.shape[0])

    matches: list[tuple[str, int]] = []
    for corner in CORNERS:
        for leg_len in range(2, n):
            if orange_coords == _candidate_l_coords(n, corner, leg_len):
                matches.append((corner, leg_len))

    assert len(matches) == 1, "orange cells must form exactly one corner L-shape"
    return matches[0]


def _decode_palette(grid: np.ndarray, corner: str) -> tuple[int, ...]:
    palette_row = int(grid.shape[0] - 1 if corner.startswith("t") else 0)
    row = grid[palette_row]
    palette_positions = [idx for idx, color in enumerate(row) if int(color) in PALETTE_SET]
    assert palette_positions, "expected a palette row"

    start = palette_positions[0]
    end = palette_positions[-1]
    assert palette_positions == list(range(start, end + 1)), "palette row must be contiguous"
    palette = tuple(int(color) for color in row[start : end + 1])
    assert 2 <= len(palette) <= 4, "palette length must be 2..4"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert set(palette) <= PALETTE_SET, "palette colors must use 1..4"

    mask = np.ones_like(row, dtype=bool)
    mask[start : end + 1] = False
    assert np.all(row[mask] != ORANGE), "palette must sit on the edge opposite the L-shape"
    assert np.all((row[mask] == BG) | (row[mask] == ORANGE)), "unexpected non-palette colors on palette edge"
    return palette


def _decode_input(grid: np.ndarray) -> tuple[int, str, int, tuple[int, ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "input must be square"
    n = h
    assert 5 <= n <= 10, "unexpected input size"

    nonzero = set(map(int, np.unique(g))) - {BG}
    assert nonzero <= (PALETTE_SET | {ORANGE}), "input uses unexpected colors"

    corner, leg_len = _decode_l_shape(g)
    palette = _decode_palette(g, corner)
    assert leg_len >= 3, "leg length must leave visible layered interior"
    assert len(palette) <= leg_len - 1, "every palette color must appear in the fill"
    return n, corner, leg_len, palette


def _triangle_local(out_size: int, corner: str, r: int, c: int) -> tuple[int, int]:
    if corner == "tl":
        return r, c
    if corner == "tr":
        return r, out_size - 1 - c
    if corner == "bl":
        return out_size - 1 - r, c
    if corner == "br":
        return out_size - 1 - r, out_size - 1 - c
    raise ValueError(f"unknown corner {corner}")


def _triangle_outline_points(out_size: int, corner: str, side_len: int) -> set[tuple[int, int]]:
    if corner == "tl":
        diag_start = (0, side_len - 1)
        diag_end = (side_len - 1, 0)
        horiz = {(0, c) for c in range(side_len)}
        vert = {(r, 0) for r in range(side_len)}
    elif corner == "tr":
        diag_start = (0, out_size - side_len)
        diag_end = (side_len - 1, out_size - 1)
        horiz = {(0, out_size - 1 - c) for c in range(side_len)}
        vert = {(r, out_size - 1) for r in range(side_len)}
    elif corner == "bl":
        diag_start = (out_size - side_len, 0)
        diag_end = (out_size - 1, side_len - 1)
        horiz = {(out_size - 1, c) for c in range(side_len)}
        vert = {(out_size - 1 - r, 0) for r in range(side_len)}
    elif corner == "br":
        diag_start = (out_size - 1, out_size - side_len)
        diag_end = (out_size - side_len, out_size - 1)
        horiz = {(out_size - 1, out_size - 1 - c) for c in range(side_len)}
        vert = {(out_size - 1 - r, out_size - 1) for r in range(side_len)}
    else:
        raise ValueError(f"unknown corner {corner}")

    return horiz | vert | set(_bresenham_line(*diag_start, *diag_end))


def _render_from_params(n: int, corner: str, leg_len: int, palette: tuple[int, ...]) -> np.ndarray:
    out_size = 3 * n - 2
    side_len = 2 * leg_len
    out = np.full((out_size, out_size), BG, dtype=int)

    outline = _triangle_outline_points(out_size, corner, side_len)
    for r, c in outline:
        out[r, c] = ORANGE

    for r in range(out_size):
        for c in range(out_size):
            if out[r, c] == ORANGE:
                continue
            u, v = _triangle_local(out_size, corner, r, c)
            if 0 < u < side_len and 0 < v < side_len and u + v < side_len - 1:
                depth = min(u, v)
                out[r, c] = palette[(depth - 1) % len(palette)]
    return out


def _validate_semantics(grid: np.ndarray, out: np.ndarray) -> None:
    n, corner, leg_len, palette = _decode_input(grid)
    expected = _render_from_params(n, corner, leg_len, palette)
    assert out.shape == (3 * n - 2, 3 * n - 2), "wrong output shape"
    assert np.array_equal(out, expected), "output does not follow the triangle-fill rule"

    max_depth = leg_len - 1
    used_layers = {min(_triangle_local(out.shape[0], corner, r, c)) for r, c in np.argwhere((out != BG) & (out != ORANGE))}
    assert used_layers == set(range(1, max_depth + 1)), "every interior layer depth must appear"
    assert set(palette) <= set(map(int, np.unique(out))), "every palette color must be used"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a corner L-marker into a large outlined right triangle and cyclically color its interior by depth layers.

    Input structure:
    - The input is a small square grid on a black background.
    - Orange cells form a one-pixel-wide L whose equal legs lie on one horizontal edge and one vertical edge, identifying a corner.
    - A separate contiguous row of distinct colors 1..4 lies on the opposite horizontal edge and serves as the palette.

    Rule:
    - Detect which corner contains the orange L and measure its leg length.
    - Create a black output canvas of size (3n - 2) x (3n - 2), where n is the input size.
    - In the matching output corner, draw an orange right-triangle outline whose two axis-aligned legs have length 2 x leg_length and whose free ends are connected by an orange diagonal.
    - Fill only the interior of that triangle: a cell at depth d from the two right-angle legs gets palette[(d - 1) mod palette_length], while the outline stays orange and everything outside stays black.

    Key insight:
    - The orange L supplies the triangle's corner and scale, while the distant palette row supplies an independent cyclic color order for successive inward depth layers.
    """
    n, corner, leg_len, palette = _decode_input(np.asarray(grid, dtype=int))
    return _render_from_params(n, corner, leg_len, palette)


def _palette_start(n: int, corner: str, palette_len: int) -> int:
    return n - palette_len if "l" in corner else 0


def _build_input(n: int, corner: str, leg_len: int, palette: tuple[int, ...]) -> np.ndarray:
    grid = np.full((n, n), BG, dtype=int)
    for r, c in _candidate_l_coords(n, corner, leg_len):
        grid[r, c] = ORANGE

    row = n - 1 if corner.startswith("t") else 0
    start = _palette_start(n, corner, len(palette))
    grid[row, start : start + len(palette)] = np.array(palette, dtype=int)
    return grid


def _random_params() -> tuple[int, str, int, tuple[int, ...]]:
    n = int(np.random.randint(5, 8))
    corner = str(np.random.choice(CORNERS))
    leg_len = int(np.random.randint(3, n))
    max_palette_len = min(4, leg_len - 1)
    palette_len = int(np.random.randint(2, max_palette_len + 1))
    palette = tuple(int(c) for c in np.random.choice(PALETTE_COLORS, size=palette_len, replace=False))
    return n, corner, leg_len, palette


def generate_input() -> np.ndarray:
    for _ in range(200):
        n, corner, leg_len, palette = _random_params()
        grid = _build_input(n, corner, leg_len, palette)
        out = main(grid)
        _validate_semantics(grid, out)
        return grid
    raise RuntimeError("failed to generate a valid triangle-layer puzzle")
