from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste
import random


def _assert_solid_rectangle(g: np.ndarray, color: int, component: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert int(component["area"]) == h * w, "foreground objects must be solid rectangles"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color)), "rectangle bbox must be filled"
    return r0, c0, r1, c1


def _facing_corners(
    upper: tuple[int, int, int, int],
    lower: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    _ur0, uc0, ur1, uc1 = upper
    lr0, lc0, _lr1, lc1 = lower
    assert ur1 < lr0, "paired rectangles must be vertically separated"

    if uc1 < lc0:
        return (ur1, uc1), (lr0, lc0)
    if lc1 < uc0:
        return (ur1, uc0), (lr0, lc1)
    raise AssertionError("paired rectangles must be diagonally separated")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the missing diagonal connector segment between each pair of same-color rectangular objects.

    Input structure:
    - The grid has one dominant background color.
    - Every foreground color appears as exactly two separated solid rectangular 4-connected objects.
    - The two rectangles of each color are diagonally offset with facing corners on a 45-degree line.

    Rule:
    - Keep all existing rectangles unchanged.
    - For each foreground color, connect the lower facing corner of the upper rectangle to the upper facing corner of the lower rectangle.
    - Paint only the intervening background pixels on that diagonal, excluding the two endpoint pixels already in the rectangles.

    Key insight:
    - Same-color rectangles define the endpoints; the output completes the open diagonal between their nearest facing corners.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    for color in colors_present(g, exclude={bg}):
        components = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[color])
        assert len(components) == 2, "each foreground color must have exactly two objects"

        rectangles = [_assert_solid_rectangle(g, int(color), component) for component in components]
        rectangles.sort(key=lambda b: b[0])
        upper, lower = rectangles
        start, end = _facing_corners(upper, lower)

        row_delta = int(end[0] - start[0])
        col_delta = int(end[1] - start[1])
        assert row_delta > 0, "connector must go downward from the upper rectangle"
        assert abs(row_delta) == abs(col_delta), "facing corners must be diagonally aligned"

        dr = 1
        dc = 1 if col_delta > 0 else -1
        r, c = int(start[0] + dr), int(start[1] + dc)
        while (r, c) != end:
            assert int(out[r, c]) == bg, "connector path must pass through background"
            out[r, c] = int(color)
            r += dr
            c += dc

    return out

# --------------- generator (att2, o5x) ---------------

# Canvas and palette.
BG, N = 7, 16
FOREGROUND_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# Pair sampling and rectangle spacing.
PAIR_COUNTS = [3, 4, 5]
PAIR_COUNT_WEIGHTS = [4, 10, 60]
LARGE_RECT_SIZE, SMALL_RECT_SIZE = 3, 2
MIN_RECT_GAP = 2

# Connector sampling and retry budget.
DIRECTIONS = [1, -1]
GAP_RANGE, MAX_ATTEMPTS = (2, 6), 300

Box = tuple[int, int, int, int]
Cells = list[tuple[int, int]]
Candidate = tuple[Box, Box, Cells]


def _bbox_gap_ok(a: Box, b: Box) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(ar0 - br1 - 1, br0 - ar1 - 1, 0)
    col_gap = max(ac0 - bc1 - 1, bc0 - ac1 - 1, 0)
    return max(row_gap, col_gap) >= MIN_RECT_GAP


def _rect_cells(box: Box) -> Cells:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _candidate(
    size: int,
    direction: int,
    gap: int,
    ur: int,
    uc: int,
) -> Candidate | None:
    delta = gap + 1
    upper = (ur, uc, ur + size - 1, uc + size - 1)
    if direction == 1:
        start = (upper[2], upper[3])
        end = (start[0] + delta, start[1] + delta)
        lower = (end[0], end[1], end[0] + size - 1, end[1] + size - 1)
    else:
        start = (upper[2], upper[1])
        end = (start[0] + delta, start[1] - delta)
        lower = (end[0], end[1] - size + 1, end[0] + size - 1, end[1])

    if min(*upper, *lower) < 0 or max(upper[2], upper[3], lower[2], lower[3]) >= N:
        return None
    if not _bbox_gap_ok(upper, lower):
        return None
    diag = [(start[0] + k, start[1] + direction * k) for k in range(1, gap + 1)]
    return upper, lower, diag


def _valid_candidates(
    occupied: np.ndarray,
    existing_rects: list[Box],
    size: int,
    direction: int,
    gap: int,
) -> list[Candidate]:
    out = []
    for ur in range(N - size + 1):
        for uc in range(N - size + 1):
            cand = _candidate(size, direction, gap, ur, uc)
            if cand is None:
                continue
            upper, lower, diag = cand
            if any(not _bbox_gap_ok(box, old) for box in (upper, lower) for old in existing_rects):
                continue
            cells = _rect_cells(upper) + _rect_cells(lower) + diag
            if any(bool(occupied[r, c]) for r, c in cells):
                continue
            out.append(cand)
    random.shuffle(out)
    return out


def _place_rect(grid: np.ndarray, box: Box, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _sample_plan(n_pairs: int) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    colors = random.sample(FOREGROUND_COLORS, n_pairs)
    sizes = [LARGE_RECT_SIZE] + [SMALL_RECT_SIZE] * (n_pairs - 1)
    random.shuffle(sizes)
    directions = [1, -1] + [random.choice(DIRECTIONS) for _ in range(n_pairs - 2)]
    random.shuffle(directions)
    gaps = [random.randint(*GAP_RANGE) for _ in range(n_pairs)]
    order = list(range(n_pairs))
    random.shuffle(order)
    order.sort(key=lambda i: (sizes[i], gaps[i]), reverse=True)
    return colors, sizes, directions, gaps, order


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choices(PAIR_COUNTS, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    colors, sizes, directions, gaps, order = _sample_plan(n_pairs)
    output = np.full((N, N), BG, dtype=int)
    occupied = np.zeros((N, N), dtype=bool)
    rects: list[Box] = []
    diagonals: list[Cells] = []

    for idx in order:
        cands = _valid_candidates(occupied, rects, sizes[idx], directions[idx], gaps[idx])
        if not cands:
            return None
        upper, lower, diag = random.choice(cands)
        color = colors[idx]
        _place_rect(output, upper, color)
        _place_rect(output, lower, color)
        for r, c in diag:
            output[r, c] = color
        for r, c in _rect_cells(upper) + _rect_cells(lower) + diag:
            occupied[r, c] = True
        rects.extend([upper, lower])
        diagonals.append(diag)

    inp = output.copy()
    for diag in diagonals:
        for r, c in diag:
            inp[r, c] = BG
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 342ae2ed input")
