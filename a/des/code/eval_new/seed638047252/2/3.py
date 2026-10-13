from __future__ import annotations

import numpy as np


BG = 0
VERTICAL_TOP = 1   # blue in ARC
HORIZONTAL_TOP = 2  # red in ARC
STRIP_COLORS = (3, 4, 5, 6, 7, 8, 9)

RNG = np.random.default_rng()
MAX_GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _find_template_size(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert h > 0 and w > 0 and int(grid[0, 0]) != BG

    size_row = 0
    while size_row < w and int(grid[0, size_row]) != BG:
        size_row += 1

    size_col = 0
    while size_col < h and int(grid[size_col, 0]) != BG:
        size_col += 1

    assert size_row == size_col and size_row >= 2
    return int(size_row)


def _detect_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    colors = [
        int(color)
        for color in np.unique(grid)
        if int(color) not in (BG, VERTICAL_TOP, HORIZONTAL_TOP)
    ]

    horizontal_bands: list[tuple[int, int, int]] = []
    vertical_bands: list[tuple[int, int, int]] = []

    for color in colors:
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height == 2 and width > 2:
            horizontal_bands.append((int(r0), int(c0), int(color)))
        elif width == 2 and height > 2:
            vertical_bands.append((int(c0), int(r0), int(color)))
        else:
            raise AssertionError("unexpected foreground object outside the weave template")

    horizontal_bands.sort(key=lambda item: item[0])
    vertical_bands.sort(key=lambda item: item[0])
    assert horizontal_bands and vertical_bands

    horizontal_rows = [(int(r0), int(color)) for r0, _c0, color in horizontal_bands]
    vertical_cols = [(int(c0), int(color)) for c0, _r0, color in vertical_bands]
    return horizontal_rows, vertical_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a small top-left weave template to decide which strip crossings show the vertical color.

    Input structure:
    - bg = black. The top-left corner contains a solid non-black square template made only of red and blue cells.
    - Elsewhere, the main area contains several horizontal strips of height 2 and several vertical strips of width 2.
    - Every strip has its own color distinct from red and blue, and strip crossings are initially painted with the horizontal-strip color.

    Rule:
    - Read the top-left square as a periodic weave template: red means the horizontal strip stays on top, blue means the vertical strip should show on top.
    - Order horizontal crossings from top to bottom and vertical crossings from left to right.
    - For crossing (i, j), look at template cell (i mod H, j mod W). If it is blue, repaint that 2x2 crossing with the corresponding vertical-strip color; otherwise leave it unchanged.
    - Non-crossing strip pixels and the template itself stay unchanged.

    Key insight:
    - The small red/blue square is not decoration: it is a tiled over-under control map for the larger woven strip grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template_size = _find_template_size(g)
    template = g[:template_size, :template_size]
    assert set(map(int, np.unique(template))) == {VERTICAL_TOP, HORIZONTAL_TOP}

    horizontal_rows, vertical_cols = _detect_bands(g)

    out = g.copy()
    for i, (row, _h_color) in enumerate(horizontal_rows):
        for j, (col, v_color) in enumerate(vertical_cols):
            if int(template[i % template_size, j % template_size]) == VERTICAL_TOP:
                out[row : row + 2, col : col + 2] = int(v_color)
    return out


def _used_template_values(template: np.ndarray, n_rows: int, n_cols: int) -> set[int]:
    used = {
        int(template[i % template.shape[0], j % template.shape[1]])
        for i in range(n_rows)
        for j in range(n_cols)
    }
    return used


def _random_template(size: int, n_rows: int, n_cols: int) -> np.ndarray:
    modes = ("checker", "row", "col", "random")
    for _ in range(100):
        mode = str(RNG.choice(modes))
        if mode == "checker":
            phase = int(RNG.integers(0, 2))
            mask = (np.add.outer(np.arange(size), np.arange(size)) + phase) % 2
        elif mode == "row":
            row_bits = RNG.integers(0, 2, size=size)
            mask = np.repeat(row_bits[:, None], size, axis=1)
        elif mode == "col":
            col_bits = RNG.integers(0, 2, size=size)
            mask = np.repeat(col_bits[None, :], size, axis=0)
        else:
            mask = RNG.integers(0, 2, size=(size, size))

        template = np.where(mask == 0, HORIZONTAL_TOP, VERTICAL_TOP).astype(int)
        used = _used_template_values(template, n_rows, n_cols)
        if used == {HORIZONTAL_TOP, VERTICAL_TOP}:
            return template
    raise RuntimeError("failed to sample a non-degenerate weave template")


def _band_starts(start: int, gaps: list[int]) -> list[int]:
    positions = [int(start)]
    for gap in gaps:
        positions.append(int(positions[-1] + 2 + gap))
    return positions


def _build_case(
    template: np.ndarray,
    area_r0: int,
    area_c0: int,
    h_rows: list[int],
    v_cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    t = int(template.shape[0])
    grid[:t, :t] = template

    area_r1 = int(h_rows[-1] + 2)
    area_c1 = int(v_cols[-1] + 2)

    for col, color in zip(v_cols, v_colors):
        grid[area_r0:area_r1, col : col + 2] = int(color)

    for row, color in zip(h_rows, h_colors):
        grid[row : row + 2, area_c0:area_c1] = int(color)

    out = grid.copy()
    for i, row in enumerate(h_rows):
        for j, col in enumerate(v_cols):
            if int(template[i % t, j % t]) == VERTICAL_TOP:
                out[row : row + 2, col : col + 2] = int(v_colors[j])
    return grid, out


def _gen_attempt() -> np.ndarray | None:
    template_size = int(RNG.integers(3, 6))

    for _ in range(120):
        n_h = int(RNG.integers(2, 5))
        n_v = int(RNG.integers(2, 5))
        if n_h + n_v <= len(STRIP_COLORS):
            break
    else:
        return None

    row_gaps = [int(RNG.integers(1, 4)) for _ in range(n_h - 1)]
    col_gaps = [int(RNG.integers(1, 4)) for _ in range(n_v - 1)]
    area_height = int(2 * n_h + sum(row_gaps))
    area_width = int(2 * n_v + sum(col_gaps))

    area_r0 = int(template_size + RNG.integers(2, 6))
    area_c0 = int(template_size + RNG.integers(2, 6))
    bottom_margin = int(RNG.integers(0, 4))
    right_margin = int(RNG.integers(0, 4))
    grid_h = int(area_r0 + area_height + bottom_margin)
    grid_w = int(area_c0 + area_width + right_margin)
    if max(grid_h, grid_w) > 30:
        return None

    h_rows = _band_starts(area_r0, row_gaps)
    v_cols = _band_starts(area_c0, col_gaps)
    template = _random_template(template_size, n_h, n_v)

    colors = list(RNG.permutation(np.array(STRIP_COLORS, dtype=int))[: n_h + n_v])
    h_colors = [int(color) for color in colors[:n_h]]
    v_colors = [int(color) for color in colors[n_h:]]

    inp, expected = _build_case(
        template=template,
        area_r0=area_r0,
        area_c0=area_c0,
        h_rows=h_rows,
        v_cols=v_cols,
        h_colors=h_colors,
        v_colors=v_colors,
        grid_shape=(grid_h, grid_w),
    )

    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving-template input")
