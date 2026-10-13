from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, components_mask, find_components
import math
import random


def _largest_component_background(g: np.ndarray) -> int:
    candidates: list[tuple[int, int]] = []
    for color in colors_present(g):
        comps = components_mask(g == color, connectivity=4)
        assert comps, "present color must have at least one component"
        candidates.append((max(int(coords.shape[0]) for coords in comps), int(color)))

    largest = max(size for size, _color in candidates)
    winners = [color for size, color in candidates if size == largest]
    assert len(winners) == 1, f"ambiguous background by largest component: {candidates}"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete the rectangular foreground object that contains the fewest marker-color pixels.

    Input structure:
    - A two-color set of solid foreground rectangles lies on a single background color.
    - One foreground color fills most rectangle cells and the other is a sparse marker color.
    - The background is the color with the largest 4-connected component.

    Rule:
    - Infer the marker color as the less frequent non-background color.
    - Count marker-color pixels inside each 4-connected foreground rectangle.
    - Recolor the unique rectangle with the smallest marker count to background.

    Key insight:
    - The object to erase is not selected by position or size; it is the rectangle with
      the least internal marker evidence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _largest_component_background(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, f"expected exactly two foreground colors, got {fg_colors}"

    totals = {color: int(np.count_nonzero(g == color)) for color in fg_colors}
    marker_total = min(totals.values())
    marker_colors = [color for color, total in totals.items() if total == marker_total]
    assert len(marker_colors) == 1, f"ambiguous marker color counts: {totals}"
    marker = marker_colors[0]

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert objects, "expected at least one foreground object"

    scored = []
    for obj in objects:
        b = obj["bbox"]
        area = int(obj["area"])
        assert area == bbox_h(b) * bbox_w(b), f"foreground object is not a solid rectangle: {b}"
        coords = obj["coords"]
        marker_count = int(np.count_nonzero(g[coords[:, 0], coords[:, 1]] == marker))
        scored.append((marker_count, obj))

    fewest = min(score for score, _obj in scored)
    targets = [obj for score, obj in scored if score == fewest]
    assert len(targets) == 1, "expected a unique rectangle with the fewest marker pixels"

    out = g.copy()
    coords = targets[0]["coords"]
    out[coords[:, 0], coords[:, 1]] = bg
    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
_GEN_H, _GEN_W = 15, 15
_GEN_BG, _GEN_FILL, _GEN_MARKER = 0, 3, 9

# Rectangle count, dimensions, and area budget.
_GEN_RECT_COUNT_MIN, _GEN_RECT_COUNT_MAX = 4, 6
_GEN_SIDE_MIN, _GEN_SIDE_MAX = 2, 7
_GEN_RECT_AREA_MIN, _GEN_RECT_AREA_MAX = 6, 42
_GEN_AREA_BUDGET_BY_COUNT = {4: (68, 118), 5: (78, 128), 6: (88, 136)}
_GEN_VALID_DIMS = [
    (h, w)
    for h in range(_GEN_SIDE_MIN, _GEN_SIDE_MAX + 1)
    for w in range(_GEN_SIDE_MIN, _GEN_SIDE_MAX + 1)
    if _GEN_RECT_AREA_MIN <= h * w <= _GEN_RECT_AREA_MAX
]

# Marker density/count constraints.
_GEN_MARKER_DENSITY_MIN, _GEN_MARKER_DENSITY_MAX = 0.04, 0.33
_GEN_MARKER_COUNT_MIN, _GEN_MARKER_COUNT_MAX = 1, 7

# Layout constraints and edge-touch bias.
_GEN_PADDING = 1
_GEN_EDGE_TOUCH_MIN, _GEN_EDGE_TOUCH_MAX = 2, 4
_GEN_EDGE_TOUCH_BIAS = 0.65

# Rejection limit.
_GEN_MAX_ATTEMPTS = 300


def _gen_marker_bounds(area: int) -> tuple[int, int]:
    lo = max(_GEN_MARKER_COUNT_MIN, int(math.ceil(_GEN_MARKER_DENSITY_MIN * area)))
    hi = min(_GEN_MARKER_COUNT_MAX, int(math.floor(_GEN_MARKER_DENSITY_MAX * area)))
    return lo, hi


def _gen_choose_dims(n_rects: int) -> list[tuple[int, int]]:
    lo_budget, hi_budget = _GEN_AREA_BUDGET_BY_COUNT[n_rects]
    area_budget = random.randint(lo_budget, hi_budget)
    dims: list[tuple[int, int]] = []
    used_area = 0
    used_area6 = 0

    for i in range(n_rects):
        remaining = n_rects - i - 1
        max_area = area_budget - used_area - _GEN_RECT_AREA_MIN * remaining
        candidates = []
        for h, w in _GEN_VALID_DIMS:
            area = h * w
            if area > max_area:
                continue
            if area == _GEN_RECT_AREA_MIN and used_area6:
                continue
            candidates.append((h, w))
        if not candidates:
            return []

        h, w = random.choice(candidates)
        dims.append((h, w))
        used_area += h * w
        used_area6 += int(h * w == 6)

    random.shuffle(dims)
    return dims


def _gen_bg_is_connected(occupied: np.ndarray) -> bool:
    comps = components_mask(~occupied, connectivity=4)
    return len(comps) == 1


def _gen_touches_edge(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r + h == _GEN_H or c + w == _GEN_W


def _gen_candidate_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int, bool]]:
    positions: list[tuple[int, int, bool]] = []
    for r in range(_GEN_H - h + 1):
        for c in range(_GEN_W - w + 1):
            r0, c0 = max(0, r - _GEN_PADDING), max(0, c - _GEN_PADDING)
            r1 = min(_GEN_H, r + h + _GEN_PADDING)
            c1 = min(_GEN_W, c + w + _GEN_PADDING)
            if np.any(occupied[r0:r1, c0:c1]):
                continue

            test = occupied.copy()
            test[r : r + h, c : c + w] = True
            if not _gen_bg_is_connected(test):
                continue

            positions.append((r, c, _gen_touches_edge(r, c, h, w)))
    random.shuffle(positions)
    return positions


def _gen_filter_positions(
    positions: list[tuple[int, int, bool]], needed_edge_touches: int, remaining_rects: int
) -> list[tuple[int, int, bool]]:
    if needed_edge_touches >= remaining_rects:
        filtered = [pos for pos in positions if pos[2]]
    elif needed_edge_touches <= 0:
        filtered = [pos for pos in positions if not pos[2]]
    elif random.random() < _GEN_EDGE_TOUCH_BIAS:
        filtered = [pos for pos in positions if pos[2]]
    else:
        filtered = positions
    return filtered or positions


def _gen_place_rectangles(
    dims: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((_GEN_H, _GEN_W), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []
    ordered_dims = sorted(dims, key=lambda x: x[0] * x[1], reverse=True)
    desired_edge_touches = random.randint(_GEN_EDGE_TOUCH_MIN, min(_GEN_EDGE_TOUCH_MAX, len(ordered_dims)))
    edge_touches = 0

    for i, (h, w) in enumerate(ordered_dims):
        positions = _gen_candidate_positions(occupied, h, w)
        if not positions:
            return None

        remaining = len(ordered_dims) - i
        needed = desired_edge_touches - edge_touches
        filtered = _gen_filter_positions(positions, needed, remaining)
        r, c, touches_edge = random.choice(filtered)
        occupied[r : r + h, c : c + w] = True
        placed.append((r, c, h, w))
        edge_touches += int(touches_edge)

    if not _GEN_EDGE_TOUCH_MIN <= edge_touches <= _GEN_EDGE_TOUCH_MAX:
        return None
    return placed


def _gen_marker_counts(rects: list[tuple[int, int, int, int]]) -> tuple[int, list[int]] | None:
    bounds = [_gen_marker_bounds(h * w) for _r, _c, h, w in rects]
    options: list[tuple[int, int]] = []
    for target_i, (target_lo, target_hi) in enumerate(bounds):
        other_max = min(hi for j, (_lo, hi) in enumerate(bounds) if j != target_i)
        for k in range(target_lo, min(target_hi, other_max - 1) + 1):
            options.append((target_i, k))
    if not options:
        return None

    target_i, target_count = random.choice(options)
    counts: list[int] = []
    for i, (lo, hi) in enumerate(bounds):
        if i == target_i:
            counts.append(target_count)
        else:
            counts.append(random.randint(max(lo, target_count + 1), hi))
    return target_i, counts


def _gen_render_grid(
    rects: list[tuple[int, int, int, int]], marker_counts: list[int]
) -> np.ndarray:
    grid = np.full((_GEN_H, _GEN_W), _GEN_BG, dtype=int)
    for (r, c, h, w), marker_count in zip(rects, marker_counts):
        grid[r : r + h, c : c + w] = _GEN_FILL
        flat_indices = random.sample(range(h * w), marker_count)
        rr, cc = np.divmod(flat_indices, w)
        grid[r + rr, c + cc] = _GEN_MARKER
    return grid


def _gen_candidate_is_valid(
    grid: np.ndarray, target_rect: tuple[int, int, int, int]
) -> bool:
    if len(components_mask(grid == _GEN_BG, connectivity=4)) != 1:
        return False
    if np.count_nonzero(grid == _GEN_MARKER) >= np.count_nonzero(grid == _GEN_FILL):
        return False

    expected = grid.copy()
    r, c, h, w = target_rect
    expected[r : r + h, c : c + w] = _GEN_BG
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(_GEN_RECT_COUNT_MIN, _GEN_RECT_COUNT_MAX)
    dims = _gen_choose_dims(n_rects)
    if not dims:
        return None

    rects = _gen_place_rectangles(dims)
    if rects is None:
        return None

    marker_plan = _gen_marker_counts(rects)
    if marker_plan is None:
        return None
    target_i, marker_counts = marker_plan

    grid = _gen_render_grid(rects, marker_counts)
    if not _gen_candidate_is_valid(grid, rects[target_i]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 54db823b input")
