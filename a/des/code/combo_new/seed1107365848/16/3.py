from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_ARROW_PATTERNS = {
    "right": {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
    "left": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)},
    "up": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "down": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
}
_DIRECTIONS = tuple(_ARROW_PATTERNS)

_MAX_GENERATE_ATTEMPTS = 400
_PAIR_PLACEMENT_TRIES = 160
_EXTRA_PLACEMENT_TRIES = 120
_RECT_DIM_RANGE = (2, 5)
_EXTRA_RECT_DIM_RANGE = (2, 6)
_GAP_RANGE = (2, 5)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": _bbox_from_coords(coord_arr),
                    "area": len(coords),
                }
            )

    return components


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return h >= 2 and w >= 2 and int(component["area"]) == h * w


def _arrow_direction(component: dict) -> str | None:
    if int(component["area"]) != 5:
        return None
    r0, c0, r1, c1 = component["bbox"]
    if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != (3, 3):
        return None

    rel = {(int(r - r0), int(c - c0)) for r, c in component["coords"]}
    for direction, pattern in _ARROW_PATTERNS.items():
        if rel == pattern:
            return direction
    return None


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _target_distance(arrow: dict, rect: dict) -> int | None:
    ar0, ac0, ar1, ac1 = arrow["bbox"]
    rr0, rc0, rr1, rc1 = rect["bbox"]
    direction = arrow["direction"]

    if direction == "right":
        if rc0 > ac1 and _intervals_overlap(rr0, rr1, ar0, ar1):
            return int(rc0 - ac1)
    elif direction == "left":
        if rc1 < ac0 and _intervals_overlap(rr0, rr1, ar0, ar1):
            return int(ac0 - rc1)
    elif direction == "down":
        if rr0 > ar1 and _intervals_overlap(rc0, rc1, ac0, ac1):
            return int(rr0 - ar1)
    else:
        assert direction == "up"
        if rr1 < ar0 and _intervals_overlap(rc0, rc1, ac0, ac1):
            return int(ar0 - rr1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each arrow's color with the nearest solid rectangle in the direction it points.

    Input structure:
    - The background is black.
    - Foreground objects are monochrome 5-pixel arrows and solid monochrome rectangles of size at least 2x2.
    - Every arrow is a 3x3 T-shaped half-arrow that points up, down, left, or right.
    - Each arrow points toward exactly one nearest rectangle along its tip direction, and no two arrows share a target rectangle.

    Rule:
    - Identify every arrow and every solid rectangle.
    - For each arrow, look in the arrow-tip direction and select the nearest rectangle whose span overlaps the arrow's 3-cell tip projection.
    - Recolor the arrow to the target rectangle's original color.
    - Recolor the target rectangle to the arrow's original color, leaving all other rectangles unchanged.

    Key insight:
    - The arrow is not moving or painting a path; it defines a directional pairing, and the paired arrow and rectangle simply exchange colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_components(g, _BG)
    arrows = []
    rectangles = []
    for component in components:
        direction = _arrow_direction(component)
        if direction is not None:
            component = dict(component)
            component["direction"] = direction
            arrows.append(component)
        elif _is_solid_rectangle(component):
            rectangles.append(component)
        else:
            raise AssertionError("unexpected foreground object")

    assert arrows, "expected at least one arrow"
    assert rectangles, "expected at least one rectangle"

    assignments = []
    for arrow in arrows:
        candidates = []
        for rect_index, rect in enumerate(rectangles):
            dist = _target_distance(arrow, rect)
            if dist is not None:
                candidates.append((dist, rect_index))

        assert candidates, "each arrow must point to a rectangle"
        best_dist = min(dist for dist, _ in candidates)
        best = [rect_index for dist, rect_index in candidates if dist == best_dist]
        assert len(best) == 1, "arrow-to-rectangle matching must be unambiguous"

        rect = rectangles[best[0]]
        assert int(rect["color"]) != int(arrow["color"]), "arrow and target colors must differ"
        assignments.append((arrow, best[0]))

    target_ids = [rect_index for _arrow, rect_index in assignments]
    assert len(set(target_ids)) == len(target_ids), "no two arrows may share a target rectangle"

    out = g.copy()
    for arrow, rect_index in assignments:
        rect = rectangles[rect_index]
        arrow_color = int(arrow["color"])
        rect_color = int(rect["color"])
        for r, c in arrow["coords"]:
            out[int(r), int(c)] = rect_color
        rr0, rc0, rr1, rc1 = rect["bbox"]
        out[rr0 : rr1 + 1, rc0 : rc1 + 1] = arrow_color

    return out


def _arrow_cells(direction: str, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in _ARROW_PATTERNS[direction]]


def _rect_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _expanded_box(box: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _box_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _box_is_free(blocked: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_blocked(blocked: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _pair_footprint(spec: dict) -> int:
    rect_h = spec["rect_h"]
    rect_w = spec["rect_w"]
    gap = spec["gap"]
    if spec["direction"] in ("left", "right"):
        return int((max(3, rect_h) + 1) * (rect_w + gap + 3))
    return int((rect_h + gap + 3) * (max(3, rect_w) + 1))


def _sample_pair_specs(pair_count: int) -> list[dict]:
    specs = []
    for _ in range(pair_count):
        spec = {
            "direction": random.choice(_DIRECTIONS),
            "rect_h": random.randint(*_RECT_DIM_RANGE),
            "rect_w": random.randint(*_RECT_DIM_RANGE),
            "gap": random.randint(*_GAP_RANGE),
        }
        specs.append(spec)
    specs.sort(key=_pair_footprint, reverse=True)
    return specs


def _try_place_pair(blocked: np.ndarray, spec: dict) -> dict | None:
    h, w = blocked.shape
    direction = spec["direction"]
    rect_h = spec["rect_h"]
    rect_w = spec["rect_w"]
    gap = spec["gap"]

    for _ in range(_PAIR_PLACEMENT_TRIES):
        rect_r0 = random.randint(0, h - rect_h)
        rect_c0 = random.randint(0, w - rect_w)
        rect_r1 = rect_r0 + rect_h - 1
        rect_c1 = rect_c0 + rect_w - 1

        if direction == "right":
            if rect_c0 < gap + 3:
                continue
            row_lo = max(rect_r0, 1)
            row_hi = min(rect_r1, h - 2)
            if row_lo > row_hi:
                continue
            center_r = random.randint(row_lo, row_hi)
            arrow_top = center_r - 1
            arrow_left = rect_c0 - gap - 3
        elif direction == "left":
            if rect_c1 > w - gap - 4:
                continue
            row_lo = max(rect_r0, 1)
            row_hi = min(rect_r1, h - 2)
            if row_lo > row_hi:
                continue
            center_r = random.randint(row_lo, row_hi)
            arrow_top = center_r - 1
            arrow_left = rect_c1 + gap + 1
        elif direction == "down":
            if rect_r0 < gap + 3:
                continue
            col_lo = max(rect_c0, 1)
            col_hi = min(rect_c1, w - 2)
            if col_lo > col_hi:
                continue
            center_c = random.randint(col_lo, col_hi)
            arrow_top = rect_r0 - gap - 3
            arrow_left = center_c - 1
        else:
            assert direction == "up"
            if rect_r1 > h - gap - 4:
                continue
            col_lo = max(rect_c0, 1)
            col_hi = min(rect_c1, w - 2)
            if col_lo > col_hi:
                continue
            center_c = random.randint(col_lo, col_hi)
            arrow_top = rect_r1 + gap + 1
            arrow_left = center_c - 1

        arrow_cells = _arrow_cells(direction, arrow_top, arrow_left)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in arrow_cells):
            continue

        rect_box = (rect_r0, rect_c0, rect_r1, rect_c1)
        all_cells = arrow_cells + _rect_cells(rect_box)
        footprint = _expanded_box(_box_from_cells(all_cells), (h, w), pad=1)
        if not _box_is_free(blocked, footprint):
            continue

        return {
            "direction": direction,
            "arrow_cells": arrow_cells,
            "rect_box": rect_box,
            "blocked_box": footprint,
        }

    return None


def _try_place_extra_rectangle(blocked: np.ndarray) -> dict | None:
    h, w = blocked.shape
    for _ in range(_EXTRA_PLACEMENT_TRIES):
        rect_h = random.randint(*_EXTRA_RECT_DIM_RANGE)
        rect_w = random.randint(*_EXTRA_RECT_DIM_RANGE)
        rect_r0 = random.randint(0, h - rect_h)
        rect_c0 = random.randint(0, w - rect_w)
        rect_box = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)
        footprint = _expanded_box(rect_box, (h, w), pad=1)
        if _box_is_free(blocked, footprint):
            return {"rect_box": rect_box, "blocked_box": footprint}
    return None


def _paint_rectangle(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _paint_arrow(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sample_counts() -> tuple[int, int]:
    pair_count = random.choice([1, 2, 3])
    max_extra = min(2, 9 - 2 * pair_count)
    if max_extra == 0 or random.random() < 0.5:
        extra_count = 0
    else:
        extra_count = random.randint(1, max_extra)
    return pair_count, extra_count


def _gen_attempt() -> np.ndarray | None:
    pair_count, extra_count = _sample_counts()
    min_size = min(28, 18 + 2 * pair_count + extra_count)
    height = random.randint(min_size, 30)
    width = random.randint(min_size, 30)
    blocked = np.zeros((height, width), dtype=bool)

    pairs = []
    for spec in _sample_pair_specs(pair_count):
        placed = _try_place_pair(blocked, spec)
        if placed is None:
            return None
        _mark_blocked(blocked, placed["blocked_box"])
        pairs.append(placed)

    extras = []
    for _ in range(extra_count):
        placed = _try_place_extra_rectangle(blocked)
        if placed is None:
            return None
        _mark_blocked(blocked, placed["blocked_box"])
        extras.append(placed)

    palette = random.sample(list(range(1, 10)), 2 * pair_count + extra_count)
    arrow_colors = palette[:pair_count]
    rect_colors = palette[pair_count:]

    inp = np.zeros((height, width), dtype=int)
    for i, pair in enumerate(pairs):
        _paint_arrow(inp, pair["arrow_cells"], arrow_colors[i])
        _paint_rectangle(inp, pair["rect_box"], rect_colors[i])
    for j, extra in enumerate(extras):
        _paint_rectangle(inp, extra["rect_box"], rect_colors[pair_count + j])

    expected = inp.copy()
    for i, pair in enumerate(pairs):
        _paint_arrow(expected, pair["arrow_cells"], rect_colors[i])
        _paint_rectangle(expected, pair["rect_box"], arrow_colors[i])

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-swap input")
