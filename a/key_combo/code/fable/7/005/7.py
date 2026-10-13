from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


BLACK = int(Color.BLACK)
YELLOW = int(Color.YELLOW)
RECT_COLORS = [int(c) for c in Color.NOT_BLACK if int(c) != YELLOW]
MAX_GENERATION_ATTEMPTS = 400


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _boundary_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _parse_rectangles(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all(g[1:, 1:] == BLACK), "input interior must be black"
    assert int(g[0, 0]) == BLACK, "top-left corner must stay black"

    colors = sorted(int(c) for c in np.unique(g) if int(c) != BLACK)
    assert len(colors) >= 2, "expected at least two rectangle colors"
    assert np.count_nonzero(g[0]) == 2 * len(colors), "top row must contain exactly two markers per color"
    assert np.count_nonzero(g[:, 0]) == 2 * len(colors), "left column must contain exactly two markers per color"
    assert np.count_nonzero(g) == 4 * len(colors), "input must contain only edge markers"

    rects: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        top_cols = np.flatnonzero(g[0] == color)
        left_rows = np.flatnonzero(g[:, 0] == color)
        assert top_cols.size == 2 and left_rows.size == 2, "each color needs two top and two left markers"
        c0, c1 = map(int, np.sort(top_cols))
        r0, r1 = map(int, np.sort(left_rows))
        assert 1 <= r0 < r1 < g.shape[0] - 1, "rectangles must stay off the border"
        assert 1 <= c0 < c1 < g.shape[1] - 1, "rectangles must stay off the border"
        rects[int(color)] = (r0, c0, r1, c1)
    return rects


def _oracle_output(
    rects: dict[int, tuple[int, int, int, int]], shape: tuple[int, int]
) -> tuple[np.ndarray, dict[int, bool], dict[tuple[int, int], set[int]]]:
    out = np.zeros(shape, dtype=int)
    contained = {
        color: any(
            other_color != color and _strictly_contains(other_rect, rect)
            for other_color, other_rect in rects.items()
        )
        for color, rect in rects.items()
    }

    for color, rect in rects.items():
        if not contained[color]:
            continue
        r0, c0, r1, c1 = rect
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any((patch != BLACK) & (patch != color)), "overlapping solid rectangles are ambiguous"
        patch[:] = color

    outline_hits: dict[tuple[int, int], set[int]] = {}
    for color, rect in rects.items():
        if contained[color]:
            continue
        for cell in _boundary_cells(rect):
            outline_hits.setdefault(cell, set()).add(color)

    for (r, c), colors in outline_hits.items():
        if len(colors) >= 2:
            out[r, c] = YELLOW
        else:
            existing = int(out[r, c])
            only = next(iter(colors))
            assert existing in (BLACK, only), "outline/solid overlap is ambiguous"
            out[r, c] = only

    return out, contained, outline_hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each hidden rectangle from its top-edge and left-edge shadow markers.

    Input structure:
    - The grid interior is all black.
    - Every non-black color appears exactly twice on the top row and exactly twice on the left column.
    - Those four same-color edge markers encode one hidden rectangle whose sides do not touch the border.

    Rule:
    - For each color, use the two top markers as the rectangle's left/right columns and the two left markers as its top/bottom rows.
    - Remove all edge markers.
    - If a rectangle is strictly contained inside another color's rectangle, draw it as a solid filled rectangle; otherwise draw only its 1-pixel-thick outline.
    - Any cell that belongs to outlines of two or more different colors becomes yellow.

    Key insight:
    - The edge pixels are not rays; they are orthogonal projections of hidden rectangles, and the reconstructed rectangles then determine both containment-based filling and outline intersections.
    """
    rects = _parse_rectangles(np.asarray(grid, dtype=int))
    out, _contained, _outline_hits = _oracle_output(rects, grid.shape)
    return out


def _choose_pair(lo: int, hi: int, min_gap: int) -> tuple[int, int]:
    candidates = [
        (a, b)
        for a in range(lo, hi + 1)
        for b in range(a + min_gap, hi + 1)
    ]
    if not candidates:
        raise ValueError("no valid endpoint pair")
    return random.choice(candidates)


def _choose_cross_pair(
    r_lo: int, r_hi: int, c_lo: int, c_hi: int
) -> list[tuple[int, int, int, int]]:
    row_candidates = [
        (rb0, ra0, ra1, rb1)
        for rb0 in range(r_lo, r_hi + 1)
        for ra0 in range(rb0 + 2, r_hi + 1)
        for ra1 in range(ra0 + 3, r_hi + 1)
        for rb1 in range(ra1 + 2, r_hi + 1)
    ]
    col_candidates = [
        (ca0, cb0, cb1, ca1)
        for ca0 in range(c_lo, c_hi + 1)
        for cb0 in range(ca0 + 2, c_hi + 1)
        for cb1 in range(cb0 + 3, c_hi + 1)
        for ca1 in range(cb1 + 2, c_hi + 1)
    ]
    if not row_candidates or not col_candidates:
        raise ValueError("cross region too small")

    rb0, ra0, ra1, rb1 = random.choice(row_candidates)
    ca0, cb0, cb1, ca1 = random.choice(col_candidates)
    rect_a = (ra0, ca0, ra1, ca1)
    rect_b = (rb0, cb0, rb1, cb1)
    return [rect_a, rect_b]


def _choose_host_inner(
    r_lo: int, r_hi: int, c_lo: int, c_hi: int
) -> list[tuple[int, int, int, int]]:
    host_r0, host_r1 = _choose_pair(r_lo, r_hi, 6)
    host_c0, host_c1 = _choose_pair(c_lo, c_hi, 6)
    inner_r0, inner_r1 = _choose_pair(host_r0 + 2, host_r1 - 2, 2)
    inner_c0, inner_c1 = _choose_pair(host_c0 + 2, host_c1 - 2, 2)
    host = (host_r0, host_c0, host_r1, host_c1)
    inner = (inner_r0, inner_c0, inner_r1, inner_c1)
    return [host, inner]


def _choose_hollow(
    r_lo: int, r_hi: int, c_lo: int, c_hi: int
) -> tuple[int, int, int, int]:
    r0, r1 = _choose_pair(r_lo, r_hi, 3)
    c0, c1 = _choose_pair(c_lo, c_hi, 3)
    return (r0, c0, r1, c1)


def _flip_rect_v(rect: tuple[int, int, int, int], height: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (height - 1 - r1, c0, height - 1 - r0, c1)


def _flip_rect_h(rect: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (r0, width - 1 - c1, r1, width - 1 - c0)


def _transpose_rect(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (c0, r0, c1, r1)


def _render_input(
    shape: tuple[int, int], colored_rects: dict[int, tuple[int, int, int, int]]
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, (r0, c0, r1, c1) in colored_rects.items():
        grid[0, c0] = color
        grid[0, c1] = color
        grid[r0, 0] = color
        grid[r1, 0] = color
    return grid


def _sample_case() -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], dict[str, bool]]:
    scenario = random.choice(["cross_only", "contain_only", "mixed"])

    if scenario == "cross_only":
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        split_r = random.randint(9, h - 6)
        split_c = random.randint(9, w - 6)
        rects = _choose_cross_pair(1, split_r, 1, split_c)
        if random.random() < 0.65:
            rects.append(_choose_hollow(split_r + 1, h - 2, split_c + 1, w - 2))
        features = {"want_yellow": True, "want_contained": False}
    elif scenario == "contain_only":
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        split_r = random.randint(8, h - 6)
        split_c = random.randint(8, w - 6)
        rects = _choose_host_inner(1, split_r, 1, split_c)
        if random.random() < 0.75:
            rects.append(_choose_hollow(split_r + 1, h - 2, split_c + 1, w - 2))
        features = {"want_yellow": False, "want_contained": True}
    else:
        h = random.randint(22, 30)
        w = random.randint(22, 30)
        split_r = random.randint(9, h - 9)
        split_c = random.randint(9, w - 9)
        rects = _choose_cross_pair(1, split_r, 1, split_c) + _choose_host_inner(
            split_r + 1, h - 2, split_c + 1, w - 2
        )
        features = {"want_yellow": True, "want_contained": True}

    if random.random() < 0.5:
        rects = [_flip_rect_v(rect, h) for rect in rects]
    if random.random() < 0.5:
        rects = [_flip_rect_h(rect, w) for rect in rects]
    if random.random() < 0.5:
        rects = [_transpose_rect(rect) for rect in rects]
        h, w = w, h

    colors = random.sample(RECT_COLORS, len(rects))
    colored_rects = {color: rect for color, rect in zip(colors, rects)}
    grid = _render_input((h, w), colored_rects)
    return grid, colored_rects, features


def _validate_generated_case(
    grid: np.ndarray,
    planned_rects: dict[int, tuple[int, int, int, int]],
    features: dict[str, bool],
) -> bool:
    try:
        parsed = _parse_rectangles(grid)
    except AssertionError:
        return False
    if parsed != planned_rects:
        return False

    expected, contained, _outline_hits = _oracle_output(parsed, grid.shape)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    has_yellow = bool(np.any(expected == YELLOW))
    has_contained = any(contained.values())
    if features["want_yellow"] != has_yellow:
        return False
    if features["want_contained"] != has_contained:
        return False

    if any(contained.values()):
        for color, rect in parsed.items():
            if not contained[color]:
                continue
            r0, c0, r1, c1 = rect
            if not np.all(expected[r0 : r1 + 1, c0 : c1 + 1] == color):
                return False

    for color, rect in parsed.items():
        if contained[color]:
            continue
        own_color_or_yellow = {color, YELLOW}
        for r, c in _boundary_cells(rect):
            if int(expected[r, c]) not in own_color_or_yellow:
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, rects, features = _sample_case()
        if _validate_generated_case(grid, rects, features):
            return grid
    raise RuntimeError("failed to generate a valid shadow-reconstruction input")
