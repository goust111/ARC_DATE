from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.insert(0, str(_TOOLBOX))

from arc_utils import Color, colors_present, get_bg


GREEN = int(Color.GREEN)
BG = int(Color.BLACK)
ACTUAL_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
GEN_RETRY_LIMIT = 300
INTERIOR_SIZE_RANGE = (6, 19)

MODE_PARTIAL = "partial"
MODE_A_CONTAINS_B = "a_contains_b"
MODE_B_CONTAINS_A = "b_contains_a"
MODE_EQUAL = "equal"
MODES = [MODE_PARTIAL, MODE_A_CONTAINS_B, MODE_B_CONTAINS_A, MODE_EQUAL]


def _randint_inclusive(lo: int, hi: int) -> int:
    return int(np.random.randint(int(lo), int(hi) + 1))


def _green_span(band: np.ndarray) -> tuple[int, int]:
    pos = np.flatnonzero(band == GREEN)
    assert pos.size > 0, "expected a nonempty green overlap segment"
    start = int(pos[0])
    stop = int(pos[-1])
    assert stop - start + 1 == int(pos.size), "green overlap must be one contiguous segment"
    return start, stop


def _render_band(length: int, span_a: tuple[int, int], span_b: tuple[int, int], color_a: int, color_b: int) -> np.ndarray:
    band = np.full(int(length), BG, dtype=int)
    a0, a1 = map(int, span_a)
    b0, b1 = map(int, span_b)
    band[a0 : a1 + 1] = int(color_a)
    band[b0 : b1 + 1] = int(color_b)
    overlap0 = max(a0, b0)
    overlap1 = min(a1, b1)
    assert overlap0 <= overlap1, "expected overlapping intervals"
    band[overlap0 : overlap1 + 1] = GREEN
    return band


def _infer_spans_from_band(
    band: np.ndarray, color_a: int, color_b: int
) -> dict[int, tuple[int, int]]:
    green0, green1 = _green_span(band)
    allowed = {BG, GREEN, int(color_a), int(color_b)}
    assert set(map(int, band.tolist())) <= allowed, "unexpected colors in projection band"

    spans: dict[int, tuple[int, int]] = {}
    for color in (int(color_a), int(color_b)):
        pos = np.flatnonzero(band == color)
        if pos.size == 0:
            span = (green0, green1)
        else:
            span = (min(int(pos[0]), green0), max(int(pos[-1]), green1))
        spans[color] = span

    rebuilt = _render_band(len(band), spans[int(color_a)], spans[int(color_b)], int(color_a), int(color_b))
    assert np.array_equal(rebuilt, band), "projection band does not match two-interval semantics"
    return spans


def _band_solutions(
    band: np.ndarray, color_a: int, color_b: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    length = int(len(band))
    sols: list[tuple[tuple[int, int], tuple[int, int]]] = []
    intervals = [(s, e) for s in range(length) for e in range(s, length)]
    for span_a in intervals:
        for span_b in intervals:
            if max(span_a[0], span_b[0]) > min(span_a[1], span_b[1]):
                continue
            if np.array_equal(_render_band(length, span_a, span_b, color_a, color_b), band):
                sols.append((span_a, span_b))
    return sols


def _sample_interval_pair(length: int, mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    n = int(length)
    assert n >= 4

    if mode == MODE_EQUAL:
        start = _randint_inclusive(0, n - 2)
        stop = _randint_inclusive(start, n - 1)
        return (start, stop), (start, stop)

    if mode == MODE_A_CONTAINS_B:
        a0 = _randint_inclusive(0, n - 3)
        a1 = _randint_inclusive(a0 + 2, n - 1)
        b0 = _randint_inclusive(a0 + 1, a1 - 1)
        b1 = _randint_inclusive(b0, a1 - 1)
        return (a0, a1), (b0, b1)

    if mode == MODE_B_CONTAINS_A:
        span_b, span_a = _sample_interval_pair(n, MODE_A_CONTAINS_B)
        return span_a, span_b

    assert mode == MODE_PARTIAL
    left_is_a = bool(np.random.randint(0, 2))
    overlap0 = _randint_inclusive(1, n - 2)
    overlap1 = _randint_inclusive(overlap0, n - 2)
    left0 = _randint_inclusive(0, overlap0 - 1)
    right1 = _randint_inclusive(overlap1 + 1, n - 1)
    if left_is_a:
        return (left0, overlap1), (overlap0, right1)
    return (overlap0, right1), (left0, overlap1)


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild two unseen solid rectangles from bottom/right projection bands and color their overlap green.

    Input structure:
    - The interior of the grid is entirely background; only the bottom row and the rightmost column act as projection bands.
    - Two hidden solid rectangles of distinct colors never appear directly in the input.
    - Each rectangle projects its column span onto the bottom band and its row span onto the right band; exclusive shadow cells show the rectangle color, while doubly covered shadow cells are green.

    Rule:
    - Collect the two actual rectangle colors from the non-green shadow cells across both bands.
    - On each band, recover each rectangle span as its visible color cells plus the contiguous green overlap segment; if a color is absent on that band, its span is exactly the green segment.
    - Use the bottom band for column spans and the right band for row spans, fill the two rectangles in the interior, and paint their interior intersection green while leaving both projection bands unchanged.

    Key insight:
    - Green is not a third object: it is the shared part of two one-dimensional shadows, and the same shared segment must be assigned to both rectangle spans before the 2D rectangles can be reconstructed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "expected at least a 2x2 interior plus two bands"

    bg = int(get_bg(g))
    assert np.all(g[:-1, :-1] == bg), "expected an empty interior input"

    bottom = g[-1, :-1]
    right = g[:-1, -1]
    assert int(g[-1, -1]) == bg, "expected the band corner to stay background"
    assert GREEN in set(map(int, bottom.tolist())), "expected green overlap on the bottom band"
    assert GREEN in set(map(int, right.tolist())), "expected green overlap on the right band"

    actual_colors = colors_present(g, exclude={bg, GREEN})
    assert len(actual_colors) == 2, "expected exactly two actual rectangle colors across both bands"
    color_a, color_b = map(int, actual_colors)

    col_spans = _infer_spans_from_band(bottom, color_a, color_b)
    row_spans = _infer_spans_from_band(right, color_a, color_b)

    out = g.copy()
    interior = out[:-1, :-1]
    mask_a = np.zeros_like(interior, dtype=bool)
    mask_b = np.zeros_like(interior, dtype=bool)

    ra0, ra1 = row_spans[color_a]
    ca0, ca1 = col_spans[color_a]
    rb0, rb1 = row_spans[color_b]
    cb0, cb1 = col_spans[color_b]
    mask_a[ra0 : ra1 + 1, ca0 : ca1 + 1] = True
    mask_b[rb0 : rb1 + 1, cb0 : cb1 + 1] = True

    interior[mask_a] = color_a
    interior[mask_b] = color_b
    interior[mask_a & mask_b] = GREEN
    return out


def _build_expected(
    interior_shape: tuple[int, int],
    color_a: int,
    color_b: int,
    row_a: tuple[int, int],
    col_a: tuple[int, int],
    row_b: tuple[int, int],
    col_b: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, interior_shape)
    inp = np.full((h + 1, w + 1), BG, dtype=int)
    out = inp.copy()

    bottom = _render_band(w, col_a, col_b, color_a, color_b)
    right = _render_band(h, row_a, row_b, color_a, color_b)
    inp[-1, :-1] = bottom
    inp[:-1, -1] = right
    out[-1, :-1] = bottom
    out[:-1, -1] = right

    mask_a = np.zeros((h, w), dtype=bool)
    mask_b = np.zeros((h, w), dtype=bool)
    mask_a[row_a[0] : row_a[1] + 1, col_a[0] : col_a[1] + 1] = True
    mask_b[row_b[0] : row_b[1] + 1, col_b[0] : col_b[1] + 1] = True
    out[:-1, :-1][mask_a] = color_a
    out[:-1, :-1][mask_b] = color_b
    out[:-1, :-1][mask_a & mask_b] = GREEN
    return inp, out


def _has_direct_visibility(span: tuple[int, int], other: tuple[int, int]) -> bool:
    a0, a1 = map(int, span)
    b0, b1 = map(int, other)
    return a0 < b0 or a1 > b1


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    color_a: int,
    color_b: int,
    row_a: tuple[int, int],
    col_a: tuple[int, int],
    row_b: tuple[int, int],
    col_b: tuple[int, int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    bottom = inp[-1, :-1]
    right = inp[:-1, -1]
    if len(_band_solutions(bottom, color_a, color_b)) != 1:
        return False
    if len(_band_solutions(right, color_a, color_b)) != 1:
        return False

    if _infer_spans_from_band(bottom, color_a, color_b) != {color_a: col_a, color_b: col_b}:
        return False
    if _infer_spans_from_band(right, color_a, color_b) != {color_a: row_a, color_b: row_b}:
        return False

    visible_a = _has_direct_visibility(col_a, col_b) or _has_direct_visibility(row_a, row_b)
    visible_b = _has_direct_visibility(col_b, col_a) or _has_direct_visibility(row_b, row_a)
    if not (visible_a and visible_b):
        return False

    overlap_rows = max(row_a[0], row_b[0]) <= min(row_a[1], row_b[1])
    overlap_cols = max(col_a[0], col_b[0]) <= min(col_a[1], col_b[1])
    return overlap_rows and overlap_cols


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = int(np.random.randint(*INTERIOR_SIZE_RANGE))
    w = int(np.random.randint(*INTERIOR_SIZE_RANGE))
    color_a, color_b = map(int, np.random.choice(ACTUAL_COLORS, size=2, replace=False))

    for _ in range(40):
        row_mode = MODES[int(np.random.randint(0, len(MODES)))]
        col_mode = MODES[int(np.random.randint(0, len(MODES)))]
        row_a, row_b = _sample_interval_pair(h, row_mode)
        col_a, col_b = _sample_interval_pair(w, col_mode)

        if not (_has_direct_visibility(col_a, col_b) or _has_direct_visibility(row_a, row_b)):
            continue
        if not (_has_direct_visibility(col_b, col_a) or _has_direct_visibility(row_b, row_a)):
            continue

        inp, expected = _build_expected((h, w), color_a, color_b, row_a, col_a, row_b, col_b)
        if _semantic_ok(inp, expected, color_a, color_b, row_a, col_a, row_b, col_b):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        sampled = _sample_case()
        if sampled is not None:
            return sampled[0]
    raise RuntimeError("failed to generate a valid dual-projection rectangle input")
