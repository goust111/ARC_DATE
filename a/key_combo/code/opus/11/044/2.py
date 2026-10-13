from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7
PALETTE_CHOICES = (2, 3, 4, 6, 8, 9)
GEN_RETRIES = 400


def _control_span(grid: np.ndarray) -> tuple[int, int]:
    row = np.asarray(grid[0], dtype=int)
    cols = np.flatnonzero(row == ORANGE)
    assert len(cols) >= 4, "expected an orange control bar"
    assert len(cols) <= 8, "control bar length must stay within the puzzle bounds"
    assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)), "control bar must be contiguous"
    assert cols[0] == 1, "control bar must start at column 1"
    assert np.all(np.isin(row, (BLACK, ORANGE))), "top row may only contain background and the control bar"
    return int(cols[0]), int(cols[-1])


def _palette(grid: np.ndarray) -> list[int]:
    col = np.asarray(grid[:, -1], dtype=int)
    assert col[0] == BLACK, "palette column must not occupy the top row"
    colors: list[int] = []
    for r in range(1, grid.shape[0]):
        value = int(col[r])
        if value == BLACK:
            break
        colors.append(value)
    assert 3 <= len(colors) <= 4, "palette length must be 3 or 4"
    assert len(set(colors)) == len(colors), "palette colors must be distinct"
    assert all(color in PALETTE_CHOICES for color in colors), "palette must use visible non-control colors"
    assert np.all(col[1 + len(colors) :] == BLACK), "palette cells must be tightly stacked at the top"
    return colors


def _blue_pixels(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(grid == BLUE)
    return [(int(r), int(c)) for r, c in coords]


def _distance_to_bar(grid: np.ndarray, r: int, c: int) -> int:
    assert grid[0, c] == ORANGE
    return int(np.count_nonzero(grid[1:r, c] == BLACK))


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    c0, c1 = _control_span(g)
    palette = _palette(g)
    out = g.copy()
    k = len(palette)
    for r, c in _blue_pixels(g):
        if c0 <= c <= c1:
            d = _distance_to_bar(g, r, c)
            out[r, c] = palette[d % k]
        else:
            out[r, c] = BLACK
    return out


def _wrong_ignore_coverage(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    palette = _palette(g)
    out = g.copy()
    k = len(palette)
    for r, c in _blue_pixels(g):
        if c == g.shape[1] - 1:
            continue
        d = int(np.count_nonzero(g[1:r, c] == BLACK))
        out[r, c] = palette[d % k]
    return out


def _wrong_constant_color(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    c0, c1 = _control_span(g)
    palette = _palette(g)
    out = g.copy()
    for r, c in _blue_pixels(g):
        out[r, c] = palette[0] if c0 <= c <= c1 else BLACK
    return out


def _wrong_no_wrap(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    c0, c1 = _control_span(g)
    palette = _palette(g)
    out = g.copy()
    for r, c in _blue_pixels(g):
        if c0 <= c <= c1:
            d = _distance_to_bar(g, r, c)
            out[r, c] = palette[min(d, len(palette) - 1)]
        else:
            out[r, c] = BLACK
    return out


def _validate_grid(grid: np.ndarray, expected: np.ndarray | None = None) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    c0, c1 = _control_span(g)
    palette = _palette(g)
    blues = _blue_pixels(g)
    assert 5 <= len(blues) <= 8, "need 5-8 blue pixels"
    assert all(r > 0 for r, _ in blues), "blue pixels must be below the control row"
    assert all(c < g.shape[1] - 1 for _, c in blues), "blue pixels may not overlap the palette column"
    assert len({c for _, c in blues}) == len(blues), "use one blue pixel per column for visual clarity"
    palette_rows = set(range(1, 1 + len(palette)))
    allowed_palette_cells = {(r, g.shape[1] - 1) for r in palette_rows}
    for r, c in np.argwhere(~np.isin(g, (BLACK, BLUE, ORANGE))):
        assert (int(r), int(c)) in allowed_palette_cells, "non-structural colors must belong only to the palette"

    inside = [(r, c) for r, c in blues if c0 <= c <= c1]
    outside = [(r, c) for r, c in blues if not (c0 <= c <= c1)]
    assert len(inside) >= 3, "at least three blue pixels must survive the column filter"
    assert len(outside) >= 1, "at least one blue pixel must be outside the covered columns"

    distances = [_distance_to_bar(g, r, c) for r, c in inside]
    assert len(set(distances)) >= 2, "distance must matter"
    assert any(d == 0 for d in distances), "cover the d=0 boundary case"
    assert any(d >= len(palette) for d in distances), "cover modulo wrapping"
    residues = {d % len(palette) for d in distances}
    assert len(residues) >= 2, "inside pixels should not all map to one palette color"

    out = main(g)
    oracle = _oracle_output(g) if expected is None else expected
    assert np.array_equal(out, oracle), "main() must match the intended rule"
    assert np.any((g == BLUE) & (out == BLACK)), "some blue pixels must be erased"
    assert np.any((g == BLUE) & (~np.isin(out, (BLACK, BLUE)))), "some blue pixels must be recolored by the palette"
    assert not np.array_equal(out, _wrong_ignore_coverage(g)), "column coverage must affect the result"
    assert not np.array_equal(out, _wrong_constant_color(g)), "distance-driven recoloring must affect the result"
    assert not np.array_equal(out, _wrong_no_wrap(g)), "cyclic palette wrapping must affect the result"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor surviving blue pixels by measuring upward background distance into a cyclic right-edge palette.

    Input structure:
    - The grid has black background, one contiguous orange control bar on row 0, and one palette column on the far right.
    - The control bar starts at column 1 and covers a contiguous span of columns.
    - The palette column contains 3 or 4 distinct non-black colors stacked from row 1 downward, with black below.
    - Blue foreground pixels are scattered below the bar; some lie under covered columns and some lie outside them.

    Rule:
    - Read the orange control bar to determine the covered column interval.
    - Read the rightmost palette column from top to bottom to obtain the cyclic palette sequence.
    - For each blue pixel outside the covered columns, erase it to black.
    - For each blue pixel inside the covered columns, count how many black cells lie strictly above it in the same column before the control bar, take that distance modulo the palette length, and repaint the pixel with the indexed palette color.
    - Preserve the grid size, the control bar, and the palette column.

    Key insight:
    - The control bar first decides which columns are active, then the upward black-cell distance in each active column becomes a cyclic palette index.
    """
    g = np.asarray(grid, dtype=int)
    c0, c1 = _control_span(g)
    palette = _palette(g)
    out = g.copy()
    k = len(palette)

    for r, c in _blue_pixels(g):
        if c0 <= c <= c1:
            d = _distance_to_bar(g, r, c)
            out[r, c] = palette[d % k]
        else:
            out[r, c] = BLACK
    return out


def _sample_inside_distances(height: int, palette_len: int, count: int, rng: random.Random) -> list[int]:
    max_d = height - 2
    assert max_d >= palette_len + 1

    wrap_bases = [d for d in range(1, max_d - palette_len + 1) if d % palette_len != palette_len - 1]
    assert wrap_bases
    base = rng.choice(wrap_bases)
    distances = [0, base, base + palette_len]
    if count >= 4:
        alt_choices = [d for d in range(max_d + 1) if d % palette_len != base % palette_len and d not in distances]
        distances.append(rng.choice(alt_choices))

    while len(distances) < count:
        candidate = rng.randint(0, max_d)
        if candidate not in distances:
            distances.append(candidate)
    distances = distances[:count]
    rng.shuffle(distances)
    return distances


def _build_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    height = rng.randint(10, 16)
    palette_len = rng.randint(3, 4)
    inside_count = rng.randint(3, 5)
    outside_count = rng.randint(1, min(3, 8 - inside_count))
    total_blue = inside_count + outside_count

    control_len = rng.randint(max(4, inside_count), 8)
    min_width = max(control_len + 4, total_blue + 3, 10)
    if min_width > 16:
        return None
    width = rng.randint(min_width, 16)

    grid = np.zeros((height, width), dtype=int)
    grid[0, 1 : 1 + control_len] = ORANGE

    palette_colors = rng.sample(PALETTE_CHOICES, palette_len)
    for idx, color in enumerate(palette_colors, start=1):
        grid[idx, width - 1] = color

    inside_cols = rng.sample(list(range(1, 1 + control_len)), inside_count)
    inside_distances = _sample_inside_distances(height, palette_len, inside_count, rng)
    for col, distance in zip(inside_cols, inside_distances):
        row = distance + 1
        if grid[row, col] != BLACK:
            return None
        grid[row, col] = BLUE

    outside_candidates = [0] + list(range(1 + control_len, width - 1))
    if len(outside_candidates) < outside_count:
        return None

    outside_cols: list[int] = []
    if outside_count >= 1 and 0 in outside_candidates and rng.random() < 0.65:
        outside_cols.append(0)
    remaining_cols = [c for c in outside_candidates if c not in outside_cols]
    while len(outside_cols) < outside_count:
        choice = rng.choice(remaining_cols)
        remaining_cols.remove(choice)
        outside_cols.append(choice)

    used_rows = {r for r, _ in _blue_pixels(grid)}
    for col in outside_cols:
        row_choices = [r for r in range(1, height) if grid[r, col] == BLACK]
        if not row_choices:
            return None
        prefer = [r for r in row_choices if r - 1 not in inside_distances and r not in used_rows]
        row = rng.choice(prefer or row_choices)
        grid[row, col] = BLUE
        used_rows.add(row)

    if len(_blue_pixels(grid)) != total_blue:
        return None

    expected = _oracle_output(grid)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = _build_attempt(rng)
        if built is None:
            continue
        grid, expected = built
        try:
            _validate_grid(grid, expected)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
