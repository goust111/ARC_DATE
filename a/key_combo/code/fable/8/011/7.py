from __future__ import annotations

from collections import Counter
import random

import numpy as np


_DIR_RIGHT = "right"
_DIR_DOWN = "down"
_DIR_LEFT = "left"
_DIR_UP = "up"
_DIRECTIONS = (_DIR_RIGHT, _DIR_DOWN, _DIR_LEFT, _DIR_UP)

_BG = 0
_INDICATOR_COLOR = 2
_RECTANGLE_PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]

_GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _indicator_direction(r0: int, c0: int, r1: int, c1: int, ir: int, ic: int) -> str:
    sides = []
    if ir == r0:
        sides.append(_DIR_UP)
    if ir == r1:
        sides.append(_DIR_DOWN)
    if ic == c0:
        sides.append(_DIR_LEFT)
    if ic == c1:
        sides.append(_DIR_RIGHT)
    assert len(sides) == 1, "indicator must lie on exactly one rectangle side"
    return sides[0]


def _analyze_rectangles(grid: np.ndarray) -> tuple[list[dict], int]:
    bg = _get_bg(grid)
    comps = _components(grid != bg)
    assert comps, "expected at least one foreground rectangle"

    rectangles: list[dict] = []
    marked_index: int | None = None

    for coords in comps:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg), "each rectangle bbox must be fully filled"

        colors, counts = np.unique(patch, return_counts=True)
        colors = [int(x) for x in colors]
        counts = [int(x) for x in counts]
        area = patch.size

        rect: dict = {
            "bbox": (r0, c0, r1, c1),
            "marked": False,
        }

        if len(colors) == 1:
            rect["color"] = colors[0]
        else:
            assert len(colors) == 2, "marked rectangle must have exactly two colors"
            singleton_idx = min(range(2), key=lambda i: counts[i])
            assert counts[singleton_idx] == 1, "indicator must be a single pixel"
            rect_idx = 1 - singleton_idx
            assert counts[rect_idx] == area - 1, "host rectangle must be solid apart from the indicator"

            ir, ic = map(int, np.argwhere(patch == colors[singleton_idx])[0])
            ir += r0
            ic += c0
            direction = _indicator_direction(r0, c0, r1, c1, ir, ic)

            assert marked_index is None, "expected exactly one marked rectangle"
            marked_index = len(rectangles)
            rect["color"] = colors[rect_idx]
            rect["marked"] = True
            rect["indicator_color"] = colors[singleton_idx]
            rect["indicator"] = (ir, ic)
            rect["direction"] = direction

        rectangles.append(rect)

    assert marked_index is not None, "expected one marked rectangle"
    return rectangles, marked_index


def _find_next(rectangles: list[dict], current_idx: int, visited: set[int], direction: str) -> int | None:
    r0, c0, r1, c1 = rectangles[current_idx]["bbox"]
    best_idx: int | None = None
    best_distance: int | None = None

    for idx, rect in enumerate(rectangles):
        if idx == current_idx or idx in visited:
            continue
        nr0, nc0, nr1, nc1 = rect["bbox"]

        if direction == _DIR_RIGHT:
            if nc0 <= c1 or _interval_overlap(r0, r1, nr0, nr1) == 0:
                continue
            distance = nc0 - c1 - 1
        elif direction == _DIR_LEFT:
            if nc1 >= c0 or _interval_overlap(r0, r1, nr0, nr1) == 0:
                continue
            distance = c0 - nc1 - 1
        elif direction == _DIR_DOWN:
            if nr0 <= r1 or _interval_overlap(c0, c1, nc0, nc1) == 0:
                continue
            distance = nr0 - r1 - 1
        else:
            if nr1 >= r0 or _interval_overlap(c0, c1, nc0, nc1) == 0:
                continue
            distance = r0 - nr1 - 1

        assert distance >= 1, "rectangles must not touch"
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_idx = idx
        elif distance == best_distance:
            raise AssertionError("next relay target must be unique")

    return best_idx


def _extend_one(out: np.ndarray, rect: dict, next_rect: dict | None, direction: str) -> None:
    r0, c0, r1, c1 = rect["bbox"]
    color = int(rect["color"])
    h, w = out.shape

    if direction == _DIR_RIGHT:
        stop = w if next_rect is None else int(next_rect["bbox"][1]) - 1
        out[r0 : r1 + 1, c1 + 1 : stop] = color
    elif direction == _DIR_LEFT:
        start = 0 if next_rect is None else int(next_rect["bbox"][3]) + 2
        out[r0 : r1 + 1, start:c0] = color
    elif direction == _DIR_DOWN:
        stop = h if next_rect is None else int(next_rect["bbox"][0]) - 1
        out[r1 + 1 : stop, c0 : c1 + 1] = color
    else:
        start = 0 if next_rect is None else int(next_rect["bbox"][2]) + 2
        out[start:r0, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the marked rectangle and relay the same growth direction through every frontally aligned rectangle until the last one reaches the canvas edge.

    Input structure:
    - The background is black.
    - Every foreground object is a solid axis-aligned rectangle, and exactly one rectangle has a single indicator pixel embedded on one non-corner border cell.
    - Foreground rectangles are pairwise separated, use distinct rectangle colors, and are arranged so the marked rectangle points toward a unique relay chain of frontally aligned rectangles.

    Rule:
    - Infer the marked rectangle's growth direction from which side contains the indicator pixel.
    - Extend that rectangle only in that direction until exactly one empty cell remains before the nearest overlapping rectangle ahead.
    - Relay the same directional extension through the hit rectangle, then the next, and so on; when no rectangle remains ahead, extend the last rectangle all the way to the canvas edge.

    Key insight:
    - The indicator identifies only the first mover; the rest of the output is determined by repeatedly finding the nearest rectangle ahead whose perpendicular span overlaps the current one, with intermediate links stopping one cell short and only the final link reaching the boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles, marked_idx = _analyze_rectangles(g)
    direction = rectangles[marked_idx]["direction"]

    out = g.copy()
    visited: set[int] = set()
    current_idx: int | None = marked_idx
    while current_idx is not None:
        visited.add(current_idx)
        next_idx = _find_next(rectangles, current_idx, visited, direction)
        next_rect = None if next_idx is None else rectangles[next_idx]
        _extend_one(out, rectangles[current_idx], next_rect, direction)
        current_idx = next_idx

    assert len(visited) == len(rectangles), "all rectangles must belong to the same relay chain"
    return out


def _canonical_specs() -> tuple[tuple[int, int], list[dict]]:
    axis_budget = random.randint(16, 30)
    max_count = min(5, axis_budget // 5)
    count = random.randint(2, max_count)

    widths = [2] * count
    gaps = [3] * (count - 1)
    left_margin = 1
    right_margin = 2
    remaining = axis_budget - (sum(widths) + sum(gaps) + left_margin + right_margin)
    while remaining > 0:
        choices: list[tuple[str, int | None]] = []
        if left_margin < 4:
            choices.append(("left", None))
        choices.append(("right", None))
        for i in range(count):
            if widths[i] < 5:
                choices.append(("width", i))
        for i in range(count - 1):
            if gaps[i] < 6:
                choices.append(("gap", i))
        key, idx = random.choice(choices)
        if key == "left":
            left_margin += 1
        elif key == "right":
            right_margin += 1
        elif key == "width":
            widths[int(idx)] += 1
        else:
            gaps[int(idx)] += 1
        remaining -= 1

    h = random.randint(10, 20)
    heights = [random.randint(3, min(6, h - 1)) for _ in range(count)]
    anchor_row = random.randint(1, h - 2)

    colors = random.sample(_RECTANGLE_PALETTE, count)
    specs: list[dict] = []
    c = left_margin
    for i in range(count):
        height = heights[i]
        low = max(0, anchor_row - height + 2)
        high = min(anchor_row - 1, h - height)
        assert low <= high
        r0 = random.randint(low, high)
        width = widths[i]
        specs.append(
            {
                "bbox": (r0, c, r0 + height - 1, c + width - 1),
                "color": colors[i],
            }
        )
        c += width
        if i < count - 1:
            c += gaps[i]

    return (h, axis_budget), specs


def _render_canonical_pair(shape: tuple[int, int], specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.full((h, w), _BG, dtype=int)
    out = np.full((h, w), _BG, dtype=int)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = int(spec["color"])
        inp[r0 : r1 + 1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    host = specs[0]
    hr0, hc0, hr1, hc1 = host["bbox"]
    indicator_row = random.randint(hr0 + 1, hr1 - 1)
    inp[indicator_row, hc1] = _INDICATOR_COLOR
    out[indicator_row, hc1] = _INDICATOR_COLOR

    for i, spec in enumerate(specs):
        r0, c0, r1, c1 = spec["bbox"]
        color = int(spec["color"])
        if i + 1 < len(specs):
            next_c0 = int(specs[i + 1]["bbox"][1])
            out[r0 : r1 + 1, c1 + 1 : next_c0 - 1] = color
        else:
            out[r0 : r1 + 1, c1 + 1 : w] = color
    return inp, out


def _rotated_pair() -> tuple[np.ndarray, np.ndarray]:
    shape, specs = _canonical_specs()
    inp, out = _render_canonical_pair(shape, specs)
    k = random.randint(0, 3)
    return np.rot90(inp, k=k), np.rot90(out, k=k)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        inp, expected = _rotated_pair()
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError(f"failed to generate valid input after {_GEN_RETRIES} attempts")
