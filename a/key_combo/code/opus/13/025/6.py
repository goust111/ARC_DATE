import random
from collections import Counter
from typing import Iterable

import numpy as np


BACKGROUND = 0
WALL = 5
SWATCH_POOL = (1, 2, 3, 4)
QUADRANT_SIZE_OPTIONS = (5, 6, 7)
GEN_ATTEMPTS = 400
WALL_ATTEMPTS = 80


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _flood_from_corner(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    assert bool(mask[0, 0]), "the quadrant corner must be open for flood-fill"

    seen = np.zeros_like(mask, dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return True
    start = tuple(map(int, coords[0]))
    seen = np.zeros_like(mask, dtype=bool)
    stack = [start]
    seen[start] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return np.array_equal(seen & mask, mask)


def _outside_mask(panel: np.ndarray, wall_color: int) -> np.ndarray:
    open_mask = panel != wall_color
    outside = _flood_from_corner(open_mask)
    inside_open = open_mask & ~outside
    assert inside_open.any(), "wall must leave a non-empty inner region"
    assert _single_component(inside_open), "wall must partition the quadrant into exactly two open regions"
    return outside


def _separator_index_rows(grid: np.ndarray, wall_color: int) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == wall_color)]
    assert len(rows) == 1, "expected exactly one full separator row"
    return rows[0]


def _separator_index_cols(grid: np.ndarray, wall_color: int) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == wall_color)]
    assert len(cols) == 1, "expected exactly one full separator column"
    return cols[0]


def _extract_layout(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wall_color = WALL
    sep_r = _separator_index_rows(grid, wall_color)
    sep_c = _separator_index_cols(grid, wall_color)
    tl = grid[:sep_r, :sep_c]
    tr = grid[:sep_r, sep_c + 1 :]
    bl = grid[sep_r + 1 :, :sep_c]
    br = grid[sep_r + 1 :, sep_c + 1 :]
    assert tl.shape == tr.shape == bl.shape == br.shape, "all quadrants must share one size"
    return sep_r, sep_c, tl, tr, bl, br


def _sorted_swatches(panel: np.ndarray, bg: int, wall_color: int) -> list[tuple[int, int, int]]:
    coords = [
        (int(r), int(c), int(panel[r, c]))
        for r, c in np.argwhere((panel != bg) & (panel != wall_color))
    ]
    assert len(coords) == 3, "expected exactly three swatches"
    assert len({r for r, _, _ in coords}) == 3, "swatches must occupy different rows"
    assert len({color for _, _, color in coords}) == 3, "swatch colors must be distinct"
    coords.sort()
    return coords


def _compose_output(
    grid: np.ndarray,
    outside_a: np.ndarray,
    outside_b: np.ndarray,
    swatch_colors: tuple[int, int, int],
    sep_r: int,
    sep_c: int,
) -> np.ndarray:
    color_ii, color_io, color_oi = swatch_colors
    out = np.array(grid, dtype=int, copy=True)
    quad = out[sep_r + 1 :, sep_c + 1 :]

    a_inner = ~outside_a
    b_inner = ~outside_b
    quad[:, :] = BACKGROUND
    quad[a_inner & b_inner] = color_ii
    quad[a_inner & outside_b] = color_io
    quad[outside_a & b_inner] = color_oi
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine two wall-defined flood-fill masks into a four-state color panel.

    Input structure:
    - The grid is split into four equal quadrants by one full gray separator row and one full gray separator column.
    - The top-left and top-right quadrants contain only black background plus a 1-pixel-wide gray wall path, each wall dividing its quadrant into exactly two open regions.
    - The bottom-left quadrant contains exactly three single-pixel color swatches on different rows, while the bottom-right quadrant is blank black.

    Rule:
    - Flood-fill from the top-left corner of the top-left quadrant, blocked by gray, to obtain the A-outside mask; every non-reached coordinate belongs to A-inside.
    - Flood-fill from the top-left corner of the top-right quadrant in the same way to obtain the B-outside mask and its complement B-inside.
    - Read the three bottom-left swatches from top to bottom as colors 1, 2, and 3.
    - For each coordinate in the bottom-right quadrant, use the corresponding coordinates from the two top quadrants: A-inside and B-inside paints swatch 1, A-inside and B-outside paints swatch 2, A-outside and B-inside paints swatch 3, and A-outside plus B-outside stays black. All other quadrants and separators remain unchanged.

    Key insight:
    - Each wall contributes one independent binary mask via flood-fill reachability, and only the Cartesian product of the two masks determines the final color at each output coordinate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    sep_r, sep_c, tl, tr, bl, _br = _extract_layout(g)
    assert bg == BACKGROUND, "expected black background"
    assert int(tl[0, 0]) == bg and int(tr[0, 0]) == bg, "flood-fill seeds must be open"
    assert set(map(int, np.unique(tl))) <= {bg, WALL}
    assert set(map(int, np.unique(tr))) <= {bg, WALL}

    outside_a = _outside_mask(tl, WALL)
    outside_b = _outside_mask(tr, WALL)
    swatches = _sorted_swatches(bl, bg, WALL)
    colors = (swatches[0][2], swatches[1][2], swatches[2][2])
    return _compose_output(g, outside_a, outside_b, colors, sep_r, sep_c)


def _build_vertical_wall(size: int) -> set[tuple[int, int]] | None:
    for _ in range(WALL_ATTEMPTS):
        col = random.randint(1, size - 2)
        cols = [col]
        for _row in range(1, size):
            candidates = [c for c in (cols[-1] - 1, cols[-1], cols[-1] + 1) if 1 <= c <= size - 2]
            cols.append(random.choice(candidates))

        wall: set[tuple[int, int]] = {(0, cols[0])}
        for r in range(1, size):
            lo = min(cols[r - 1], cols[r])
            hi = max(cols[r - 1], cols[r])
            for c in range(lo, hi + 1):
                wall.add((r, c))

        panel = np.zeros((size, size), dtype=int)
        for r, c in wall:
            panel[r, c] = WALL
        try:
            outside = _outside_mask(panel, WALL)
        except AssertionError:
            continue
        if int(np.count_nonzero(~outside)) < 3 or int(np.count_nonzero(outside)) < 3:
            continue
        return wall
    return None


def _build_horizontal_wall(size: int) -> set[tuple[int, int]] | None:
    for _ in range(WALL_ATTEMPTS):
        row = random.randint(1, size - 2)
        rows = [row]
        for _col in range(1, size):
            candidates = [r for r in (rows[-1] - 1, rows[-1], rows[-1] + 1) if 1 <= r <= size - 2]
            rows.append(random.choice(candidates))

        wall: set[tuple[int, int]] = {(rows[0], 0)}
        for c in range(1, size):
            lo = min(rows[c - 1], rows[c])
            hi = max(rows[c - 1], rows[c])
            for r in range(lo, hi + 1):
                wall.add((r, c))

        panel = np.zeros((size, size), dtype=int)
        for r, c in wall:
            panel[r, c] = WALL
        try:
            outside = _outside_mask(panel, WALL)
        except AssertionError:
            continue
        if int(np.count_nonzero(~outside)) < 3 or int(np.count_nonzero(outside)) < 3:
            continue
        return wall
    return None


def _state_counts(outside_a: np.ndarray, outside_b: np.ndarray) -> list[int]:
    a_inner = ~outside_a
    b_inner = ~outside_b
    return [
        int(np.count_nonzero(a_inner & b_inner)),
        int(np.count_nonzero(a_inner & outside_b)),
        int(np.count_nonzero(outside_a & b_inner)),
        int(np.count_nonzero(outside_a & outside_b)),
    ]


def _render_input(
    size: int,
    wall_a: set[tuple[int, int]],
    wall_b: set[tuple[int, int]],
    swatches: list[tuple[int, int, int]],
) -> np.ndarray:
    full = np.full((2 * size + 1, 2 * size + 1), BACKGROUND, dtype=int)
    full[size, :] = WALL
    full[:, size] = WALL

    for r, c in wall_a:
        full[r, c] = WALL
    for r, c in wall_b:
        full[r, size + 1 + c] = WALL
    for r, c, color in swatches:
        full[size + 1 + r, c] = color
    return full


def _interesting_output(inp: np.ndarray, out: np.ndarray, size: int) -> bool:
    br = out[size + 1 :, size + 1 :]
    colors = set(map(int, np.unique(br)))
    if colors != {BACKGROUND, *SWATCH_POOL} and len(colors) < 4:
        return False
    non_bg = int(np.count_nonzero(br != BACKGROUND))
    return size <= non_bg <= size * size - size


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(QUADRANT_SIZE_OPTIONS)
    wall_a = _build_vertical_wall(size)
    wall_b = _build_horizontal_wall(size)
    if wall_a is None or wall_b is None:
        return None

    panel_a = np.zeros((size, size), dtype=int)
    panel_b = np.zeros((size, size), dtype=int)
    for r, c in wall_a:
        panel_a[r, c] = WALL
    for r, c in wall_b:
        panel_b[r, c] = WALL

    outside_a = _outside_mask(panel_a, WALL)
    outside_b = _outside_mask(panel_b, WALL)
    counts = _state_counts(outside_a, outside_b)
    if min(counts) < 2:
        return None

    rows = sorted(random.sample(range(size), 3))
    cols = random.sample(range(size), 3)
    colors = random.sample(list(SWATCH_POOL), 3)
    swatches = [(rows[i], cols[i], colors[i]) for i in range(3)]
    inp = _render_input(size, wall_a, wall_b, swatches)

    swatch_colors = tuple(color for _, _, color in swatches)
    expected = _compose_output(inp, outside_a, outside_b, swatch_colors, size, size)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _interesting_output(inp, actual, size):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-wall mask input")
