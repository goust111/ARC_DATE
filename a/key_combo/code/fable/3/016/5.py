from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, get_bg
except Exception:  # pragma: no cover - fallback for direct SourceFileLoader loading
    import sys

    _here = Path(__file__).resolve()
    for _p in _here.parents:
        _candidate = _p / "new" / "solvers" / "arc_utils.py"
        if _candidate.exists():
            sys.path.insert(0, str(_candidate.parent))
            break
    from arc_utils import Color, get_bg


_TICK_COLORS: tuple[int, int, int, int] = (Color.TEAL, Color.PINK, Color.YELLOW, Color.ORANGE)
_SWAP_BY_1: dict[int, int] = {
    Color.TEAL: Color.PINK,
    Color.PINK: Color.TEAL,
    Color.YELLOW: Color.ORANGE,
    Color.ORANGE: Color.YELLOW,
}
_SWAP_BY_2: dict[int, int] = {value: key for key, value in _SWAP_BY_1.items()}
_SWAP_MAP: dict[int, int] = {**_SWAP_BY_1, **_SWAP_BY_2}


# Generator params.
_BG = Color.BLACK
_GEN_H_RANGE = (10, 28)
_GEN_W_RANGE = (12, 28)
_GEN_ATTEMPTS = 600
_RECT_COUNT_CHOICES = np.array([1, 2, 3, 4], dtype=int)
_RECT_COUNT_PROBS = np.array([0.15, 0.35, 0.30, 0.20], dtype=float)
_RECT_COUNT_PROBS = _RECT_COUNT_PROBS / _RECT_COUNT_PROBS.sum()
_FILL_DENSITY_RANGE = (0.18, 0.52)


def _coverage_from_rectangles_loop(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    H, W = shape
    cov = np.zeros((H, W), dtype=np.int16)
    for r0, r1, c0, c1 in rects:
        cov[r0 : r1 + 1, c0 : c1 + 1] += 1
    return cov


def _coverage_from_rectangles_diff(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    H, W = shape
    marker = np.zeros((H + 1, W + 1), dtype=np.int16)
    for r0, r1, c0, c1 in rects:
        marker[r0, c0] += 1
        marker[r1 + 1, c0] -= 1
        marker[r0, c1 + 1] -= 1
        marker[r1 + 1, c1 + 1] += 1

    cov = marker[:-1, :-1]
    cov = np.cumsum(np.cumsum(cov, axis=0), axis=1)
    return cov


def _collect_tick_marks(g: np.ndarray, bg: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    top: dict[int, list[int]] = {}
    left: dict[int, list[int]] = {}

    H, W = g.shape
    for c in range(1, W):
        color = int(g[0, c])
        if color != bg:
            top.setdefault(color, []).append(c)

    for r in range(1, H):
        color = int(g[r, 0])
        if color != bg:
            left.setdefault(color, []).append(r)

    return top, left


def _rectangles_from_marks(
    g: np.ndarray,
    bg: int,
) -> list[tuple[int, int, int, int]]:
    top, left = _collect_tick_marks(g, bg)
    colors = sorted(set(top) & set(left))
    assert colors, "expected at least one tick color"
    assert set(top) == set(left), "tick colors must match on top and left edges"

    rects: list[tuple[int, int, int, int]] = []
    for color in colors:
        top_cols = top[color]
        left_rows = left[color]
        assert len(top_cols) == 2, "each tick color must appear twice on top"
        assert len(left_rows) == 2, "each tick color must appear twice on left"

        c0, c1 = sorted(top_cols)
        r0, r1 = sorted(left_rows)
        assert c1 - c0 >= 2, "top ticks for a color must leave at least one interior column"
        assert r1 - r0 >= 2, "left ticks for a color must leave at least one interior row"

        rects.append((r0 + 1, r1 - 1, c0 + 1, c1 - 1))

    return rects


def _expected_with_diff(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    rects = _rectangles_from_marks(g, bg)
    coverage = _coverage_from_rectangles_diff(g.shape, rects)
    out = g.copy()

    odd = coverage[1:, 1:] % 2 == 1

    for src, dst in _SWAP_MAP.items():
        mask = odd & (out[1:, 1:] == src)
        out[1:, 1:][mask] = dst

    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Toggle interior colors by odd coverage of implicit rectangles defined by matched top/left edge ticks.

    Input structure
    - The top row has paired marker pixels in several colors; each used color appears exactly twice.
    - The left column also has the same marker colors, each appearing exactly twice.
    - A pair (top positions + left positions) for one color defines an implicit rectangle whose boundary is not drawn.
    - The interior consists of black background and scattered cyan/pink/yellow/orange pixels.

    Rule
    - For every marker color, restore its implicit rectangle from its two top columns and two left rows.
    - Count how many rectangles cover each interior cell.
    - If coverage is odd, swap colors in pairs: cyan↔pink and yellow↔orange.
    - If coverage is even (including zero), keep the cell unchanged; top/left marker pixels stay unchanged.

    Key insight
    - The rectangle regions are implicit generators and compose by parity: odd overlay performs color swaps, even overlay cancels out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H >= 3 and W >= 3, "grid too small for top/left ticks"

    bg = int(get_bg(g))
    rects = _rectangles_from_marks(g, bg)
    coverage = _coverage_from_rectangles_loop((H, W), rects)

    out = g.copy()
    odd = coverage[1:, 1:] % 2 == 1

    for src, dst in _SWAP_MAP.items():
        mask = (out[1:, 1:] == src) & odd
        out[1:, 1:][mask] = dst

    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    return out


def _sample_tick_pair(size: int, rng: np.random.Generator) -> tuple[int, int] | None:
    lo, hi = 1, size - 2
    if hi - lo < 2:
        return None

    for _ in range(160):
        a = int(rng.integers(lo, hi + 1))
        b = int(rng.integers(lo, hi + 1))
        if abs(a - b) >= 2:
            if a < b:
                return a, b
            return b, a
    return None


def _rects_overlap(rects: list[tuple[int, int, int, int]]) -> bool:
    for i in range(len(rects)):
        a_r0, a_r1, a_c0, a_c1 = rects[i]
        for j in range(i + 1, len(rects)):
            b_r0, b_r1, b_c0, b_c1 = rects[j]
            if max(a_r0, b_r0) <= min(a_r1, b_r1) and max(a_c0, b_c0) <= min(a_c1, b_c1):
                return True
    return False


def _attempt_one(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(_GEN_H_RANGE[0], _GEN_H_RANGE[1] + 1))
    W = int(rng.integers(_GEN_W_RANGE[0], _GEN_W_RANGE[1] + 1))

    n_rect = int(rng.choice(_RECT_COUNT_CHOICES, p=_RECT_COUNT_PROBS))
    colors = tuple(int(c) for c in rng.choice(_TICK_COLORS, size=n_rect, replace=False))

    marker_defs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    rects: list[tuple[int, int, int, int]] = []
    for color in colors:
        top_pair = _sample_tick_pair(W, rng)
        left_pair = _sample_tick_pair(H, rng)
        if top_pair is None or left_pair is None:
            return None

        c0, c1 = top_pair
        r0, r1 = left_pair
        marker_defs.append((int(color), (c0, c1), (r0, r1)))
        rects.append((r0 + 1, r1 - 1, c0 + 1, c1 - 1))

    overlap_needed = n_rect >= 2 and float(rng.random()) < 0.75
    if overlap_needed and not _rects_overlap(rects):
        return None

    # fill interior
    grid = np.full((H, W), _BG, dtype=int)
    palette = np.array(_TICK_COLORS, dtype=int)

    density = float(rng.uniform(*_FILL_DENSITY_RANGE))
    paint_mask = rng.random((H - 1, W - 1)) < density
    paint_color = rng.choice(palette, size=(H - 1, W - 1))
    grid[1:, 1:] = np.where(paint_mask, paint_color, _BG)

    # place marker pixels on top and left
    for color, (c0, c1), (r0, r1) in marker_defs:
        grid[0, c0] = color
        grid[0, c1] = color
        grid[r0, 0] = color
        grid[r1, 0] = color

    coverage = _coverage_from_rectangles_diff((H, W), rects)
    odd = coverage[1:, 1:] % 2 == 1
    even = (coverage[1:, 1:] > 0) & (~odd)

    odd_positions = np.argwhere(odd)
    if odd_positions.size == 0:
        return None

    odd_non_bg = odd_positions[grid[1:, 1:][odd] != _BG]
    if odd_non_bg.shape[0] == 0:
        r, c = odd_positions[int(rng.integers(len(odd_positions))) ]
        grid[r + 1, c + 1] = int(rng.choice(palette))

    if overlap_needed:
        even_positions = np.argwhere(even)
        if even_positions.size == 0:
            return None

        even_non_bg = even_positions[grid[1:, 1:][even] != _BG]
        if even_non_bg.shape[0] == 0:
            r, c = even_positions[int(rng.integers(len(even_positions)))]
            grid[r + 1, c + 1] = int(rng.choice(palette))

    # semantic sanity check using the independent diff-based transform
    try:
        expected = _expected_with_diff(grid)
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(expected, actual):
        return None

    # structure check of generated markers (and implicit uniqueness assumptions)
    _ = _rectangles_from_marks(grid, _BG)

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_one(rng)
        if grid is not None:
            assert isinstance(grid, np.ndarray)
            return grid

    raise RuntimeError("failed to generate a valid input")
