from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
STRIPE_WIDTHS = (1, 2, 3, 4)
MIN_GAP = 1
MAX_ATTEMPTS = 300


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask.tolist()):
        if value and start is None:
            start = int(i)
        elif not value and start is not None:
            runs.append((start, int(i) - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _visible_color(values: np.ndarray) -> int:
    colors = {int(v) for v in values.ravel() if int(v) not in (BLACK, GREY)}
    assert len(colors) == 1, f"expected one visible stripe color, got {colors}"
    return colors.pop()


def _horizontal_stripes(g: np.ndarray, vcol_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    full_rows = np.all(g != BLACK, axis=1)
    rows = np.flatnonzero(full_rows)
    assert len(rows) > 0, "expected at least one horizontal stripe"
    assert np.any(~vcol_mask), "expected visible horizontal pixels outside vertical stripes"

    row_colors = {int(r): _visible_color(g[int(r), ~vcol_mask]) for r in rows}
    stripes: list[tuple[int, int, int, int]] = []
    start = prev = int(rows[0])
    color = int(row_colors[prev])

    for row in map(int, rows[1:]):
        row_color = int(row_colors[row])
        if row == prev + 1 and row_color == color:
            prev = row
            continue
        width = prev - start + 1
        assert 1 <= width <= 4, f"horizontal stripe width out of range: {width}"
        stripes.append((start, prev, width, color))
        start = prev = row
        color = row_color

    width = prev - start + 1
    assert 1 <= width <= 4, f"horizontal stripe width out of range: {width}"
    stripes.append((start, prev, width, color))
    return stripes


def _vertical_stripes(g: np.ndarray, hrow_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    full_cols = np.all(g != BLACK, axis=0)
    cols = np.flatnonzero(full_cols)
    assert len(cols) > 0, "expected at least one vertical stripe"
    assert np.any(~hrow_mask), "expected visible vertical pixels outside horizontal stripes"

    col_colors = {int(c): _visible_color(g[~hrow_mask, int(c)]) for c in cols}
    stripes: list[tuple[int, int, int, int]] = []
    start = prev = int(cols[0])
    color = int(col_colors[prev])

    for col in map(int, cols[1:]):
        col_color = int(col_colors[col])
        if col == prev + 1 and col_color == color:
            prev = col
            continue
        width = prev - start + 1
        assert 1 <= width <= 4, f"vertical stripe width out of range: {width}"
        stripes.append((start, prev, width, color))
        start = prev = col
        color = col_color

    width = prev - start + 1
    assert 1 <= width <= 4, f"vertical stripe width out of range: {width}"
    stripes.append((start, prev, width, color))
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the over-under colors at grey crossings of woven horizontal and vertical stripes.

    Input structure:
    - Black background with full-width horizontal stripes and full-height vertical stripes.
    - Each stripe is a solid non-black, non-grey rectangle of width 1 to 4 pixels.
    - Every crossing rectangle between a horizontal and vertical stripe is masked in grey.

    Rule:
    - Copy the input grid.
    - Recover each horizontal stripe width and each vertical stripe color from the full row/column spans.
    - At every grey crossing, fill the crossing with the horizontal stripe color if the horizontal width is odd.
    - If the horizontal width is even, fill the crossing with the vertical stripe color instead.

    Key insight:
    - Grey cells are only crossing masks; the visible full-span stripes reveal the two stripe widths, and the horizontal width parity decides which stripe is on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    h, w = map(int, g.shape)
    assert h > 0 and w > 0
    assert np.any(g == GREY), "expected grey crossing masks"

    hrow_mask = np.all(g != BLACK, axis=1)
    vcol_mask = np.all(g != BLACK, axis=0)
    horizontals = _horizontal_stripes(g, vcol_mask)
    verticals = _vertical_stripes(g, hrow_mask)

    out = g.copy()
    for r0, r1, h_width, h_color in horizontals:
        for c0, c1, _v_width, v_color in verticals:
            crossing = g[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(crossing == GREY), "expected every stripe crossing to be grey"
            fill = int(h_color) if h_width % 2 == 1 else int(v_color)
            out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


# --------------- generator ---------------


def _split_extra(total: int, parts: int) -> list[int]:
    shares = [0] * int(parts)
    for _ in range(int(total)):
        shares[random.randrange(int(parts))] += 1
    return shares


def _place_bands(length: int, widths: list[int]) -> list[tuple[int, int, int]]:
    widths = list(map(int, widths))
    random.shuffle(widths)
    required = sum(widths) + MIN_GAP * (len(widths) - 1)
    assert required <= int(length)

    gaps = _split_extra(int(length) - required, len(widths) + 1)
    pos = gaps[0]
    bands: list[tuple[int, int, int]] = []
    for i, width in enumerate(widths):
        start = int(pos)
        end = int(start + width - 1)
        bands.append((start, end, int(width)))
        pos = end + 1
        if i + 1 < len(widths):
            pos += MIN_GAP + gaps[i + 1]
    return bands


def _horizontal_widths(count: int) -> list[int]:
    widths = [random.choice((1, 3)), random.choice((2, 4))]
    widths.extend(random.choice(STRIPE_WIDTHS) for _ in range(int(count) - 2))
    random.shuffle(widths)
    return [int(w) for w in widths]


def _render_input(
    shape: tuple[int, int],
    horizontal_bands: list[tuple[int, int, int]],
    vertical_bands: list[tuple[int, int, int]],
    horizontal_colors: list[int],
    vertical_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for (r0, r1, _width), color in zip(horizontal_bands, horizontal_colors):
        grid[r0 : r1 + 1, :] = int(color)
    for (c0, c1, _width), color in zip(vertical_bands, vertical_colors):
        grid[:, c0 : c1 + 1] = int(color)
    for r0, r1, _h_width in horizontal_bands:
        for c0, c1, _v_width in vertical_bands:
            grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return grid


def _expected_output(
    grid: np.ndarray,
    horizontal_bands: list[tuple[int, int, int]],
    vertical_bands: list[tuple[int, int, int]],
    horizontal_colors: list[int],
    vertical_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for (r0, r1, h_width), h_color in zip(horizontal_bands, horizontal_colors):
        for (c0, c1, _v_width), v_color in zip(vertical_bands, vertical_colors):
            fill = int(h_color) if int(h_width) % 2 == 1 else int(v_color)
            out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    n_h = random.randint(2, 4)
    n_v = random.randint(2, min(4, len(COLOR_POOL) - n_h))

    h_widths = _horizontal_widths(n_h)
    v_widths = [int(random.choice(STRIPE_WIDTHS)) for _ in range(n_v)]
    h_required = sum(h_widths) + MIN_GAP * (n_h - 1)
    v_required = sum(v_widths) + MIN_GAP * (n_v - 1)
    if h_required > 30 or v_required > 30:
        return None

    height = random.randint(max(8, h_required), 30)
    width = random.randint(max(8, v_required), 30)

    horizontal_bands = _place_bands(height, h_widths)
    vertical_bands = _place_bands(width, v_widths)

    colors = random.sample(COLOR_POOL, n_h + n_v)
    horizontal_colors = [int(c) for c in colors[:n_h]]
    vertical_colors = [int(c) for c in colors[n_h:]]

    grid = _render_input(
        (height, width),
        horizontal_bands,
        vertical_bands,
        horizontal_colors,
        vertical_colors,
    )
    expected = _expected_output(
        grid,
        horizontal_bands,
        vertical_bands,
        horizontal_colors,
        vertical_colors,
    )

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
