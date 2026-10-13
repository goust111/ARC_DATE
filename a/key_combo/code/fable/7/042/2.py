from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
DRAW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _border_points(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    pts = {(0, c) for c in range(w)}
    pts |= {(h - 1, c) for c in range(w)}
    pts |= {(r, 0) for r in range(h)}
    pts |= {(r, w - 1) for r in range(h)}
    return sorted(pts)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0:
        step_r, step_c = 0, _sign(dc)
        steps = abs(dc)
    elif dc == 0:
        step_r, step_c = _sign(dr), 0
        steps = abs(dr)
    elif abs(dr) == abs(dc):
        step_r, step_c = _sign(dr), _sign(dc)
        steps = abs(dr)
    else:
        return None

    return [(r0 + k * step_r, c0 + k * step_c) for k in range(steps + 1)]


def _arm_clear(gray_points: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> bool:
    segment = _segment(start, end)
    assert segment is not None
    return all(cell not in gray_points for cell in segment[1:-1])


def _render_pair(
    out: np.ndarray,
    color: int,
    endpoints: tuple[tuple[int, int], tuple[int, int]],
    pivot: tuple[int, int] | None,
) -> None:
    if pivot is None:
        cells = _segment(endpoints[0], endpoints[1])
        assert cells is not None
        for r, c in cells:
            out[r, c] = color
        return

    first = _segment(endpoints[0], pivot)
    second = _segment(pivot, endpoints[1])
    assert first is not None and second is not None
    for r, c in first:
        if (r, c) != pivot:
            out[r, c] = color
    for r, c in second:
        if (r, c) != pivot:
            out[r, c] = color
    out[pivot] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each repeated border color by either one straight segment or one grey-pivot L bend.

    Input structure:
    - The grid has a black background, colored endpoint pixels on the outer border, and grey pixels in the interior.
    - Some non-grey border colors appear exactly twice; others appear exactly once as distractors.
    - A repeated color pair is either horizontally/vertically/diagonally aligned, or it has one unique grey pivot sharing a row with one endpoint and a column with the other.

    Rule:
    - Group non-grey border pixels by color.
    - If a color appears twice and its endpoints are aligned in one row, one column, or one 45-degree diagonal, fill the entire segment between them in that color.
    - Otherwise, find the unique valid grey pivot and draw the two orthogonal arms from the endpoints to that pivot, keeping the pivot grey.
    - Leave singleton border colors and unused grey pixels unchanged.

    Key insight:
    - The task is a two-stage selection problem: repeated border colors choose between straight and bent connectivity, and only one grey corner remains compatible with a non-straight pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    border_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in _border_points(g.shape):
        color = int(g[r, c])
        if color not in (BLACK, GREY):
            border_by_color[color].append((r, c))

    gray_points = {tuple(map(int, cell)) for cell in np.argwhere(g == GREY)}
    out = g.copy()

    for color, pts in border_by_color.items():
        assert len(pts) in (1, 2), "every border color should be singleton or paired"
        if len(pts) == 1:
            continue

        a, b = pts
        direct = _segment(a, b)
        if direct is not None:
            for r, c in direct:
                out[r, c] = color
            continue

        pivots = []
        candidate_corners = ((a[0], b[1]), (b[0], a[1]))
        for pivot in candidate_corners:
            if pivot not in gray_points:
                continue
            if _arm_clear(gray_points, a, pivot) and _arm_clear(gray_points, b, pivot):
                pivots.append(pivot)

        assert len(pivots) == 1, "expected one valid grey pivot for each non-straight pair"
        _render_pair(out, color, (a, b), pivots[0])

    return out


def _random_direct_types() -> tuple[str, ...]:
    return ("H", "V", "D")


def _sample_plan(mode: int) -> tuple[list[str], int]:
    direct_types = _random_direct_types()
    if mode == 0:
        pair_types = ["L", random.choice(direct_types)]
        pair_types += random.choices(direct_types + ("L",), k=random.randint(1, 2))
        singletons = random.randint(1, 2)
    elif mode == 1:
        pair_types = ["D", random.choice(("H", "V"))]
        if random.random() < 0.7:
            pair_types.append(random.choice(direct_types + ("L",)))
        singletons = random.randint(0, 2)
    else:
        pair_types = ["L", "L"]
        if random.random() < 0.8:
            pair_types.append(random.choice(direct_types))
        if random.random() < 0.45:
            pair_types.append(random.choice(direct_types + ("L",)))
        singletons = random.randint(0, 1)

    while len(pair_types) + singletons > len(DRAW_COLORS):
        if singletons > 0:
            singletons -= 1
        else:
            pair_types.pop()

    pair_types.sort(key={"H": 0, "V": 1, "D": 2, "L": 3}.__getitem__)
    return pair_types, singletons


def _place_horizontal(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    rows = list(range(1, h - 1))
    random.shuffle(rows)
    for r in rows:
        endpoints = ((r, 0), (r, w - 1))
        cells = _segment(*endpoints)
        assert cells is not None
        if all(cell not in occupied for cell in cells):
            return endpoints[0], endpoints[1], cells
    return None


def _place_vertical(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    cols = list(range(1, w - 1))
    random.shuffle(cols)
    for c in cols:
        endpoints = ((0, c), (h - 1, c))
        cells = _segment(*endpoints)
        assert cells is not None
        if all(cell not in occupied for cell in cells):
            return endpoints[0], endpoints[1], cells
    return None


def _place_diagonal(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    border = _border_points((h, w))
    candidates = []
    for i, a in enumerate(border):
        for b in border[i + 1 :]:
            if a[0] == b[0] or a[1] == b[1]:
                continue
            cells = _segment(a, b)
            if cells is None or len(cells) < 4:
                continue
            candidates.append((a, b, cells))

    random.shuffle(candidates)
    for a, b, cells in candidates:
        if all(cell not in occupied for cell in cells):
            return a, b, cells
    return None


def _place_l_shape(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    pivots = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(pivots)
    side_pairs = [("left", "top"), ("left", "bottom"), ("right", "top"), ("right", "bottom")]
    for pivot in pivots:
        r, c = pivot
        random.shuffle(side_pairs)
        for row_side, col_side in side_pairs:
            a = (r, 0) if row_side == "left" else (r, w - 1)
            b = (0, c) if col_side == "top" else (h - 1, c)
            first = _segment(a, pivot)
            second = _segment(pivot, b)
            assert first is not None and second is not None
            cells = list(dict.fromkeys(first + second))
            if all(cell not in occupied for cell in cells):
                return a, b, pivot, cells
    return None


def _add_decoy_greys(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    count: int,
) -> None:
    h, w = grid.shape
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if (r, c) not in occupied]
    random.shuffle(candidates)
    for cell in candidates[:count]:
        grid[cell] = GREY


def _add_singletons(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    colors: list[int],
) -> bool:
    border = [pt for pt in _border_points(grid.shape) if pt not in occupied]
    random.shuffle(border)
    if len(border) < len(colors):
        return False
    for color, cell in zip(colors, border):
        grid[cell] = color
        occupied.add(cell)
    return True


def _expected_output(grid: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        _render_pair(
            out,
            int(obj["color"]),
            obj["endpoints"],  # type: ignore[arg-type]
            obj.get("pivot"),  # type: ignore[arg-type]
        )
    return out


def _build_attempt(mode: int) -> np.ndarray | None:
    pair_types, singleton_count = _sample_plan(mode)
    min_size = max(12, 8 + 2 * len(pair_types))
    h = random.randint(min_size, 24)
    w = random.randint(min_size, 24)

    grid = np.full((h, w), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()
    objects: list[dict[str, object]] = []

    color_pool = list(DRAW_COLORS)
    random.shuffle(color_pool)
    pair_colors = color_pool[: len(pair_types)]
    singleton_colors = color_pool[len(pair_types) : len(pair_types) + singleton_count]

    for pair_type, color in zip(pair_types, pair_colors):
        if pair_type == "H":
            placed = _place_horizontal(h, w, occupied)
            if placed is None:
                return None
            a, b, cells = placed
            grid[a] = color
            grid[b] = color
            occupied.update(cells)
            objects.append({"color": color, "endpoints": (a, b), "pivot": None})
        elif pair_type == "V":
            placed = _place_vertical(h, w, occupied)
            if placed is None:
                return None
            a, b, cells = placed
            grid[a] = color
            grid[b] = color
            occupied.update(cells)
            objects.append({"color": color, "endpoints": (a, b), "pivot": None})
        elif pair_type == "D":
            placed = _place_diagonal(h, w, occupied)
            if placed is None:
                return None
            a, b, cells = placed
            grid[a] = color
            grid[b] = color
            occupied.update(cells)
            objects.append({"color": color, "endpoints": (a, b), "pivot": None})
        else:
            placed = _place_l_shape(h, w, occupied)
            if placed is None:
                return None
            a, b, pivot, cells = placed
            grid[a] = color
            grid[b] = color
            grid[pivot] = GREY
            occupied.update(cells)
            objects.append({"color": color, "endpoints": (a, b), "pivot": pivot})

    if not _add_singletons(grid, occupied, singleton_colors):
        return None

    decoy_count = random.randint(0, max(1, min(4, (h - 2) * (w - 2) // 25)))
    if mode != 1:
        decoy_count = max(1, decoy_count)
    _add_decoy_greys(grid, occupied, decoy_count)

    expected = _expected_output(grid, objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    unique, counts = np.unique(grid, return_counts=True)
    color_counts = {int(color): int(count) for color, count in zip(unique, counts)}
    for color in pair_colors:
        if color_counts.get(color, 0) != 2:
            return None
    for color in singleton_colors:
        if color_counts.get(color, 0) != 1:
            return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
