from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
MARKER_COLORS = (1, 2, 3)  # blue, red, green in the ARC palette
WIDTH_CHOICES = (11, 13, 15)
CASE_SPECS = (
    ("two_cross", 2, 1),
    ("two_disjoint", 2, 0),
    ("three_single_cross", 3, 1),
    ("three_triple_cross", 3, 3),
)
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()
_COLUMN_CACHE: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == 1, "expected a single 1xN marker row"
    n = int(g.shape[1])
    assert n % 2 == 1 and n <= 30, "expected an odd ARC-width row"

    markers = [(int(c), int(color)) for c, color in enumerate(g[0]) if int(color) != BLACK]
    assert 2 <= len(markers) <= 3, "expected two or three markers"

    cols = [c for c, _color in markers]
    colors = [color for _c, color in markers]
    assert cols == sorted(cols), "markers are read left-to-right on the input row"
    assert len(set(cols)) == len(cols), "marker columns must be distinct"
    assert len(set(colors)) == len(colors), "marker colors must be distinct"
    assert all(0 < c < n - 1 for c in cols), "markers must not sit on the border columns"
    assert all(cols[i + 1] - cols[i] >= 2 for i in range(len(cols) - 1)), "marker columns must be separated"

    return markers


def _v_coords(n: int, col: int) -> list[tuple[int, int]]:
    coords = [(0, int(col))]

    step = 1
    while col - step >= 0:
        coords.append((step, col - step))
        step += 1

    step = 1
    while col + step < n:
        coords.append((step, col + step))
        step += 1

    return coords


def _paint_v(canvas: np.ndarray, col: int, color: int) -> None:
    n = int(canvas.shape[0])
    for r, c in _v_coords(n, col):
        canvas[r, c] = int(color)


def _compose_output(markers: list[tuple[int, int]], n: int, *, reverse: bool = False) -> np.ndarray:
    out = np.full((n, n), BLACK, dtype=int)
    ordered = sorted(markers, reverse=reverse)
    for col, color in ordered:
        _paint_v(out, col, color)
    return out


def _crossings(markers: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    ordered = sorted(markers)
    hits: list[tuple[int, int, int]] = []
    for i, (left_col, _left_color) in enumerate(ordered):
        for right_col, right_color in ordered[i + 1 :]:
            gap = right_col - left_col
            if gap % 2 == 0:
                row = gap // 2
                col = left_col + row
                hits.append((row, col, right_color))
    return hits


def _crossing_count(cols: tuple[int, ...]) -> int:
    return sum((cols[j] - cols[i]) % 2 == 0 for i in range(len(cols)) for j in range(i + 1, len(cols)))


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each top-row marker into a downward V and overlay the V shapes from left to right.

    Input structure:
    - The input is a single 1xN row on a black background, with odd width N.
    - Two or three non-black marker pixels appear on that row in distinct interior columns.
    - Each marker's color identifies one V-shaped layer, and the marker columns encode the layer order.

    Rule:
    - Create an NxN black canvas.
    - For every marker at column c, draw both downward diagonals starting at (0,c) and continuing to the left and right canvas edges in the marker's color.
    - Overlay those V shapes in ascending column order, so later right-side markers overwrite earlier left-side markers at crossings while black stays transparent.

    Key insight:
    - The marker row simultaneously specifies the geometry and the z-order: apex positions define each V, and left-to-right marker order resolves every overlap.
    """
    markers = _extract_markers(np.asarray(grid, dtype=int))
    n = int(grid.shape[1])
    return _compose_output(markers, n)


def _candidate_columns(n: int, marker_count: int, crossing_count: int) -> list[tuple[int, ...]]:
    key = (n, marker_count, crossing_count)
    if key not in _COLUMN_CACHE:
        cols = range(2, n - 2)
        options = []
        for combo in itertools.combinations(cols, marker_count):
            if any(combo[i + 1] - combo[i] < 2 for i in range(len(combo) - 1)):
                continue
            if _crossing_count(combo) == crossing_count:
                options.append(combo)
        assert options, f"no columns available for {key}"
        _COLUMN_CACHE[key] = options
    return _COLUMN_CACHE[key]


def _make_input(n: int, cols: tuple[int, ...], colors: np.ndarray) -> np.ndarray:
    grid = np.full((1, n), BLACK, dtype=int)
    for col, color in zip(cols, colors):
        grid[0, int(col)] = int(color)
    return grid


def _validate_semantics(grid: np.ndarray, case_spec: tuple[str, int, int]) -> bool:
    _case_name, marker_count, crossing_count = case_spec
    markers = _extract_markers(grid)
    cols = tuple(col for col, _color in markers)
    colors = tuple(color for _col, color in markers)
    n = int(grid.shape[1])

    assert n in WIDTH_CHOICES
    assert len(markers) == marker_count
    assert set(colors).issubset(set(MARKER_COLORS))
    assert _crossing_count(cols) == crossing_count

    out = main(grid)
    expected = _compose_output(markers, n)
    assert out.shape == (n, n)
    assert np.array_equal(out, expected), "main() must match the explicit layer composition"

    # The top row reproduces the input exactly, which keeps the mapping injective.
    assert np.array_equal(out[:1, :], grid)

    # Each V must visibly reach both side boundaries.
    for col, color in markers:
        assert out[col, 0] == color
        assert out[n - 1 - col, n - 1] == color

    crossings = _crossings(markers)
    assert len(crossings) == crossing_count
    for row, col, top_color in crossings:
        assert out[row, col] == top_color, "crossings must be resolved by rightmost-layer overwrite"

    # Whenever crossings exist, reversing the layer order must change the answer.
    if crossings:
        reversed_out = _compose_output(markers, n, reverse=True)
        assert not np.array_equal(out, reversed_out), "crossing samples must make layer order observable"
    else:
        occupied: dict[tuple[int, int], int] = {}
        for col, color in markers:
            for coord in _v_coords(n, col):
                prev = occupied.setdefault(coord, color)
                assert prev == color, "disjoint samples must truly have no overlaps"

    return True


def _gen_attempt() -> tuple[np.ndarray, tuple[str, int, int]]:
    case_spec = CASE_SPECS[int(_RNG.integers(len(CASE_SPECS)))]
    _case_name, marker_count, crossing_count = case_spec
    n = int(_RNG.choice(WIDTH_CHOICES))
    cols = _candidate_columns(n, marker_count, crossing_count)
    chosen_cols = cols[int(_RNG.integers(len(cols)))]
    color_order = _RNG.permutation(np.array(MARKER_COLORS, dtype=int))[:marker_count]
    return _make_input(n, chosen_cols, color_order), case_spec


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, case_spec = _gen_attempt()
        if _validate_semantics(grid, case_spec):
            return grid
    raise RuntimeError("failed to generate a valid layered V-shape sample")
