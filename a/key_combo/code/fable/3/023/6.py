from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_ATTEMPTS = 800


def _background(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    most = counts.most_common()
    assert most, "expected a non-empty grid"
    assert len(most) == 1 or most[0][1] > most[1][1], "expected a unique background color"
    return int(most[0][0])


def _axis_ranges(values: np.ndarray, bg: int, offset: int) -> dict[int, tuple[int, int]]:
    by_color: dict[int, list[int]] = {}
    for i, value in enumerate(values):
        color = int(value)
        if color == int(bg):
            continue
        by_color.setdefault(color, []).append(int(i + offset))
    return {color: (min(pos), max(pos)) for color, pos in by_color.items()}


def _rectangles_from_projections(g: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    top = _axis_ranges(g[0, 1:], bg, offset=1)
    left = _axis_ranges(g[1:, 0], bg, offset=1)
    paired = set(top) & set(left)
    assert len(paired) >= 2, "expected at least two paired projection colors"

    rects: dict[int, tuple[int, int, int, int]] = {}
    for color in paired:
        c0, c1 = top[color]
        r0, r1 = left[color]
        assert r0 <= r1 and c0 <= c1
        rects[int(color)] = (int(r0), int(r1), int(c0), int(c1))
    return rects


def _covers(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, r1, c0, c1 = rect
    return int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1)


def _covering_colors(rects: dict[int, tuple[int, int, int, int]], r: int, c: int) -> list[int]:
    return [color for color, rect in rects.items() if _covers(rect, r, c)]


def _unique_bottom_to_top(
    rects: dict[int, tuple[int, int, int, int]],
    probes: list[tuple[int, int, int]],
) -> list[int]:
    colors = set(rects)
    successors = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    assert probes, "expected probe pixels to determine the z-order"
    for r, c, visible in probes:
        assert visible in colors, "probe color must be one of the paired rectangle colors"
        covering = _covering_colors(rects, r, c)
        assert len(covering) >= 2, "each probe must lie in an overlap of at least two rectangles"
        assert visible in covering, "probe color must belong to a rectangle covering that cell"
        for lower in covering:
            if lower == visible:
                continue
            if visible not in successors[lower]:
                successors[lower].add(visible)
                indegree[visible] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1, "probe relations must imply a unique total z-order"
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for upper in successors[color]:
            indegree[upper] -= 1

    return order


def _render(
    shape: tuple[int, int],
    bg: int,
    rects: dict[int, tuple[int, int, int, int]],
    bottom_to_top: list[int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in bottom_to_top:
        r0, r1, c0, c1 = rects[int(color)]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct projected solid rectangles and render them in the probe-implied z-order.

    Input structure:
    - The background is the unique most common color.
    - The top border contains colored horizontal projection marks, and the left border
      contains vertical projection marks for the same rectangle colors.
    - Interior non-background pixels are probe pixels showing the final visible color at
      overlap cells; at most one projection color may appear on only one border.

    Rule:
    - Pair colors that occur on both borders.  A paired color defines one solid
      rectangle whose row span is taken from the left projection and whose column span
      is taken from the top projection; unpaired projection colors are ignored.
    - Each probe says its color is above every other reconstructed rectangle covering
      that same cell.  The accumulated occlusion relations must have one unique
      bottom-to-top topological order.
    - Clear all projections and probes, then draw the paired rectangles on a fresh
      background canvas from bottom to top.

    Key insight:
    - The rectangles are encoded only by two border projections, while their stacking
      order is encoded only by sparse visible probe pixels inside overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D ARC grid with borders"

    bg = _background(g)
    rects = _rectangles_from_projections(g, bg)
    probes = [
        (int(r), int(c), int(g[r, c]))
        for r in range(1, g.shape[0])
        for c in range(1, g.shape[1])
        if int(g[r, c]) != int(bg)
    ]
    order = _unique_bottom_to_top(rects, probes)
    return _render(g.shape, bg, rects, order)


def _strict_mono(values: list[int]) -> bool:
    return all(a < b for a, b in zip(values, values[1:])) or all(
        a > b for a, b in zip(values, values[1:])
    )


def _endpoints_are_unique(rects: dict[int, tuple[int, int, int, int]]) -> bool:
    top_points: set[tuple[int, int]] = set()
    left_points: set[tuple[int, int]] = set()
    for rect in rects.values():
        r0, r1, c0, c1 = rect
        for point in ((0, c0), (0, c1)):
            if point in top_points:
                return False
            top_points.add(point)
        for point in ((r0, 0), (r1, 0)):
            if point in left_points:
                return False
            left_points.add(point)
    return True


def _construct_rectangles(
    h: int,
    w: int,
    bottom_to_top: list[int],
) -> dict[int, tuple[int, int, int, int]] | None:
    n = len(bottom_to_top)
    heights = [random.randint(5, 8) for _ in range(n)]
    widths = [random.randint(5, 8) for _ in range(n)]

    row_offsets = [0]
    col_offsets = [0]
    for i in range(1, n):
        row_lo = -(heights[i] - 2)
        row_hi = heights[i - 1] - 2
        col_lo = -(widths[i] - 2)
        col_hi = widths[i - 1] - 2
        row_offsets.append(row_offsets[-1] + random.randint(row_lo, row_hi))
        col_offsets.append(col_offsets[-1] + random.randint(col_lo, col_hi))

    row_span0 = min(row_offsets)
    row_span1 = max(row_offsets[i] + heights[i] - 1 for i in range(n))
    col_span0 = min(col_offsets)
    col_span1 = max(col_offsets[i] + widths[i] - 1 for i in range(n))
    if row_span1 - row_span0 + 1 > h - 2 or col_span1 - col_span0 + 1 > w - 2:
        return None

    base_r_lo = 1 - row_span0
    base_r_hi = h - 1 - row_span1
    base_c_lo = 1 - col_span0
    base_c_hi = w - 1 - col_span1
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    base_r = random.randint(base_r_lo, base_r_hi)
    base_c = random.randint(base_c_lo, base_c_hi)

    rects: dict[int, tuple[int, int, int, int]] = {}
    for i, color in enumerate(bottom_to_top):
        r0 = base_r + row_offsets[i]
        c0 = base_c + col_offsets[i]
        rects[int(color)] = (int(r0), int(r0 + heights[i] - 1), int(c0), int(c0 + widths[i] - 1))

    if not _endpoints_are_unique(rects):
        return None

    row_starts = [rects[color][0] for color in bottom_to_top]
    col_starts = [rects[color][2] for color in bottom_to_top]
    areas = [
        (rects[color][1] - rects[color][0] + 1) * (rects[color][3] - rects[color][2] + 1)
        for color in bottom_to_top
    ]
    if _strict_mono(row_starts) or _strict_mono(col_starts) or _strict_mono(areas):
        return None

    return rects


def _probe_candidates(
    rects: dict[int, tuple[int, int, int, int]],
    bottom_to_top: list[int],
    lower_index: int,
    used: set[tuple[int, int]],
    require_spacing: bool,
) -> list[tuple[int, int]]:
    lower = bottom_to_top[lower_index]
    visible = bottom_to_top[lower_index + 1]
    higher = set(bottom_to_top[lower_index + 2 :])
    r0 = max(rects[lower][0], rects[visible][0])
    r1 = min(rects[lower][1], rects[visible][1])
    c0 = max(rects[lower][2], rects[visible][2])
    c1 = min(rects[lower][3], rects[visible][3])
    if r0 > r1 or c0 > c1:
        return []

    candidates = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in used:
                continue
            if require_spacing and any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in used):
                continue
            covering = set(_covering_colors(rects, r, c))
            if lower not in covering or visible not in covering:
                continue
            if covering & higher:
                continue
            candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _choose_chain_probes(
    rects: dict[int, tuple[int, int, int, int]],
    bottom_to_top: list[int],
) -> list[tuple[int, int, int]] | None:
    used: set[tuple[int, int]] = set()
    probes: list[tuple[int, int, int]] = []
    for i in range(len(bottom_to_top) - 1):
        candidates = _probe_candidates(rects, bottom_to_top, i, used, require_spacing=True)
        if not candidates:
            candidates = _probe_candidates(rects, bottom_to_top, i, used, require_spacing=False)
        if not candidates:
            return None
        r, c = random.choice(candidates)
        used.add((r, c))
        probes.append((int(r), int(c), int(bottom_to_top[i + 1])))
    return probes


def _add_extra_probes(
    probes: list[tuple[int, int, int]],
    rects: dict[int, tuple[int, int, int, int]],
    bottom_to_top: list[int],
) -> None:
    rank = {color: i for i, color in enumerate(bottom_to_top)}
    used = {(r, c) for r, c, _ in probes}
    cells: list[tuple[int, int, int]] = []
    all_rows = range(min(r[0] for r in rects.values()), max(r[1] for r in rects.values()) + 1)
    all_cols = range(min(r[2] for r in rects.values()), max(r[3] for r in rects.values()) + 1)
    for r in all_rows:
        for c in all_cols:
            if (r, c) in used:
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in used):
                continue
            covering = _covering_colors(rects, r, c)
            if len(covering) < 2:
                continue
            visible = max(covering, key=lambda color: rank[color])
            cells.append((int(r), int(c), int(visible)))
    random.shuffle(cells)
    for item in cells[: random.randint(1, min(3, max(1, len(cells))))]:
        probes.append(item)
        used.add((item[0], item[1]))


def _paint_projections(
    grid: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
    colors: list[int],
) -> None:
    order = list(colors)
    random.shuffle(order)
    for color in order:
        r0, r1, c0, c1 = rects[int(color)]
        grid[0, c0 : c1 + 1] = int(color)
        grid[r0 : r1 + 1, 0] = int(color)

    for color in colors:
        r0, r1, c0, c1 = rects[int(color)]
        grid[0, c0] = int(color)
        grid[0, c1] = int(color)
        grid[r0, 0] = int(color)
        grid[r1, 0] = int(color)


def _try_place_unmatched(grid: np.ndarray, color: int, protected: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    side = random.choice(("top", "left"))
    length = random.randint(2, 4)
    candidates: list[list[tuple[int, int]]] = []
    if side == "top":
        for c0 in range(1, w - length + 1):
            cells = [(0, c) for c in range(c0, c0 + length)]
            if not any(cell in protected for cell in cells):
                candidates.append(cells)
    else:
        for r0 in range(1, h - length + 1):
            cells = [(r, 0) for r in range(r0, r0 + length)]
            if not any(cell in protected for cell in cells):
                candidates.append(cells)
    if not candidates:
        return False

    bg_only = [cells for cells in candidates if all(int(grid[r, c]) == BG for r, c in cells)]
    cells = random.choice(bg_only if bg_only else candidates)
    for r, c in cells:
        grid[r, c] = int(color)
    return True


def _protected_endpoints(rects: dict[int, tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r0, r1, c0, c1 in rects.values():
        out.update({(0, c0), (0, c1), (r0, 0), (r1, 0)})
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(3, 5)
    colors = list(random.sample(COLORS, n_rects + 1))
    rect_colors = colors[:n_rects]
    unmatched_color = colors[-1]
    bottom_to_top = list(rect_colors)
    random.shuffle(bottom_to_top)

    h = random.randint(16, 30)
    w = random.randint(16, 30)
    rects = _construct_rectangles(h, w, bottom_to_top)
    if rects is None:
        return None

    probes = _choose_chain_probes(rects, bottom_to_top)
    if probes is None:
        return None
    _add_extra_probes(probes, rects, bottom_to_top)

    grid = np.full((h, w), BG, dtype=int)
    _paint_projections(grid, rects, rect_colors)

    if random.random() < 0.75:
        _try_place_unmatched(grid, unmatched_color, _protected_endpoints(rects))

    for r, c, color in probes:
        grid[int(r), int(c)] = int(color)

    try:
        actual = main(grid)
    except Exception:
        return None
    expected = _render(grid.shape, BG, rects, bottom_to_top)
    if not np.array_equal(actual, expected):
        return None
    if _background(grid) != BG:
        return None
    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection reconstruction input")
