from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
RECT = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400

ARROW_PATTERNS: dict[str, frozenset[tuple[int, int]]] = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_arrow(coords: list[tuple[int, int]]) -> tuple[str, int, str] | None:
    if len(coords) != 5:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None
    norm = frozenset((r - r0, c - c0) for r, c in coords)
    for direction, pattern in ARROW_PATTERNS.items():
        if norm != pattern:
            continue
        if direction in ("up", "down"):
            return "col", c0 + 1, direction
        return "row", r0 + 1, direction
    return None


def _row_major_first(rect: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(rect == int(color))
    assert coords.size > 0
    return int(coords[0, 0]), int(coords[0, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored outside arrows to locate row-column intersections inside a gray rectangle.

    Input structure:
    - Black background surrounds one large gray rectangle whose interior contains a multicolor pattern.
    - Outside the rectangle are several 5-pixel monochrome arrows; each arrow is a T shape made from a 3-pixel shaft plus two tip pixels.
    - Vertical arrows sit above or below the rectangle and indicate one rectangle column; horizontal arrows sit left or right of the rectangle and indicate one rectangle row.
    - Every arrow color appears exactly twice outside the rectangle: once as a row arrow and once as a column arrow.

    Rule:
    - Find the gray rectangle, then detect every outside arrow by its 5-pixel T-shaped connected component.
    - Project each horizontal arrow onto the rectangle to select its row, and each vertical arrow to select its column.
    - For each arrow color, intersect the selected row and column; the color at that rectangle cell is the extracted color for that pair.
    - Output a single-row grid containing one pixel per pair, ordered by the extracted color's first appearance inside the rectangle in row-major order.

    Key insight:
    - The arrows do not directly name output colors; matching a same-colored row arrow with a same-colored column arrow yields a coordinate, and only that coordinate reveals the output color to sort by.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = Counter(map(int, g.ravel()))
    bg = int(counts.most_common(1)[0][0])

    rect_color = None
    rect_count = -1
    for color, count in counts.items():
        if int(color) == bg:
            continue
        if int(count) > rect_count:
            rect_color = int(color)
            rect_count = int(count)
    assert rect_color is not None

    rect_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == rect_color)]
    r0, c0, r1, c1 = _bbox(rect_cells)
    rect = g[r0 : r1 + 1, c0 : c1 + 1]

    outside = np.zeros_like(g, dtype=bool)
    outside[:r0, :] = g[:r0, :] != bg
    outside[r1 + 1 :, :] = g[r1 + 1 :, :] != bg
    outside[r0 : r1 + 1, :c0] = g[r0 : r1 + 1, :c0] != bg
    outside[r0 : r1 + 1, c1 + 1 :] = g[r0 : r1 + 1, c1 + 1 :] != bg

    arrows_by_color: dict[int, list[tuple[str, int, str]]] = {}
    for color in np.unique(g[outside]):
        color = int(color)
        for comp in _components(outside & (g == color)):
            parsed = _parse_arrow(comp)
            assert parsed is not None
            arrows_by_color.setdefault(color, []).append(parsed)

    extracted: list[int] = []
    for color, arrows in arrows_by_color.items():
        assert len(arrows) == 2
        row_arrow = next((arrow for arrow in arrows if arrow[0] == "row"), None)
        col_arrow = next((arrow for arrow in arrows if arrow[0] == "col"), None)
        assert row_arrow is not None and col_arrow is not None

        row = int(row_arrow[1])
        col = int(col_arrow[1])
        assert r0 <= row <= r1 and c0 <= col <= c1
        extracted.append(int(g[row, col]))

    ordered = sorted(extracted, key=lambda color: _row_major_first(rect, int(color)))
    return np.array([ordered], dtype=int)


def _valid_position_sets(values: list[int], count: int, min_gap: int = 3) -> list[tuple[int, ...]]:
    if count == 0:
        return [()]
    out: list[tuple[int, ...]] = []
    for combo in combinations(values, count):
        if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1)):
            out.append(tuple(int(v) for v in combo))
    return out


def _sample_side_positions(values: list[int], count_a: int, count_b: int) -> tuple[list[int], list[int]] | None:
    left_sets = _valid_position_sets(values, count_a)
    right_sets = _valid_position_sets(values, count_b)
    valid: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for lhs in left_sets:
        lhs_set = set(lhs)
        for rhs in right_sets:
            if lhs_set.isdisjoint(rhs):
                valid.append((lhs, rhs))
    if not valid:
        return None
    lhs, rhs = random.choice(valid)
    return list(lhs), list(rhs)


def _cell_index(cell: tuple[int, int], width: int) -> int:
    return int(cell[0] * width + cell[1])


def _assign_first_occurrences(
    inner_cells: list[tuple[int, int]],
    intersections: list[tuple[int, int]],
    width: int,
) -> dict[int, tuple[int, int]]:
    all_intersections = set(intersections)
    cells_by_index = sorted(inner_cells, key=lambda cell: _cell_index(cell, width))
    order = sorted(range(len(intersections)), key=lambda idx: _cell_index(intersections[idx], width))

    anchors: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    for idx in order:
        own = intersections[idx]
        own_idx = _cell_index(own, width)
        candidates = []
        for cell in cells_by_index:
            if _cell_index(cell, width) > own_idx or cell in used:
                continue
            if cell != own and cell in all_intersections:
                continue
            candidates.append(cell)
        assert candidates

        earlier = [cell for cell in candidates if cell != own]
        if earlier and random.random() < 0.75:
            chosen = random.choice(earlier[: max(1, len(earlier) // 2)])
        else:
            chosen = own if own in candidates else candidates[-1]
        anchors[idx] = chosen
        used.add(chosen)
    return anchors


def _draw_arrow(grid: np.ndarray, top: int, left: int, color: int, direction: str) -> None:
    for dr, dc in ARROW_PATTERNS[direction]:
        grid[top + dr, left + dc] = int(color)


def _decorate_rectangle(
    rect: np.ndarray,
    pair_count: int,
    intersections: list[tuple[int, int]],
    extracted_colors: list[int],
) -> dict[str, object]:
    rh, rw = rect.shape
    inner_cells = [(r, c) for r in range(1, rh - 1) for c in range(1, rw - 1)]
    anchors = _assign_first_occurrences(inner_cells, intersections, rw)

    reserved = set()
    for idx, color in enumerate(extracted_colors):
        anchor = anchors[idx]
        hit = intersections[idx]
        rect[anchor] = int(color)
        rect[hit] = int(color)
        reserved.add(anchor)
        reserved.add(hit)

    filler_budget = random.randint(pair_count + 2, pair_count + 8)
    filler_choices = inner_cells[:]
    random.shuffle(filler_choices)
    for cell in filler_choices:
        if filler_budget == 0 or cell in reserved:
            continue
        allowed = []
        cell_idx = _cell_index(cell, rw)
        for idx, color in enumerate(extracted_colors):
            if cell_idx > _cell_index(anchors[idx], rw):
                allowed.append(int(color))
        extras = [color for color in PALETTE if color not in extracted_colors]
        if allowed and random.random() < 0.55:
            rect[cell] = int(random.choice(allowed))
            reserved.add(cell)
            filler_budget -= 1
        elif extras and random.random() < 0.35:
            rect[cell] = int(random.choice(extras))
            reserved.add(cell)
            filler_budget -= 1

    expected = [int(extracted_colors[idx]) for idx in range(pair_count)]
    expected.sort(key=lambda color: _row_major_first(rect, int(color)))
    return {"anchors": anchors, "expected": expected}


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.randint(2, 4)

    rect_h = random.randint(8, 12)
    rect_w = random.randint(8, 12)
    inner_rows = list(range(1, rect_h - 1))
    inner_cols = list(range(1, rect_w - 1))

    row_sides = [random.choice(["left", "right"]) for _ in range(pair_count)]
    col_sides = [random.choice(["top", "bottom"]) for _ in range(pair_count)]

    left_rows = row_sides.count("left")
    right_rows = row_sides.count("right")
    top_cols = col_sides.count("top")
    bottom_cols = col_sides.count("bottom")

    row_sets = _sample_side_positions(inner_rows, left_rows, right_rows)
    col_sets = _sample_side_positions(inner_cols, top_cols, bottom_cols)
    if row_sets is None or col_sets is None:
        return None

    left_values, right_values = row_sets
    top_values, bottom_values = col_sets
    random.shuffle(left_values)
    random.shuffle(right_values)
    random.shuffle(top_values)
    random.shuffle(bottom_values)

    row_targets: list[int] = []
    col_targets: list[int] = []
    for row_side, col_side in zip(row_sides, col_sides):
        row_targets.append(left_values.pop() if row_side == "left" else right_values.pop())
        col_targets.append(top_values.pop() if col_side == "top" else bottom_values.pop())

    canvas_h = rect_h + 8 + random.randint(0, min(6, 30 - (rect_h + 8)))
    canvas_w = rect_w + 8 + random.randint(0, min(6, 30 - (rect_w + 8)))
    if canvas_h > 30 or canvas_w > 30:
        return None

    r0 = random.randint(4, canvas_h - rect_h - 4)
    c0 = random.randint(4, canvas_w - rect_w - 4)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    rect = np.full((rect_h, rect_w), RECT, dtype=int)

    pair_colors = random.sample(PALETTE, pair_count)
    extracted_colors = random.sample(PALETTE, pair_count)
    intersections = list(zip(row_targets, col_targets))
    deco = _decorate_rectangle(rect, pair_count, intersections, extracted_colors)

    grid[r0 : r1 + 1, c0 : c1 + 1] = rect

    for idx, pair_color in enumerate(pair_colors):
        row = r0 + row_targets[idx]
        col = c0 + col_targets[idx]

        if row_sides[idx] == "left":
            _draw_arrow(grid, row - 1, c0 - 4, pair_color, "right")
        else:
            _draw_arrow(grid, row - 1, c1 + 2, pair_color, "left")

        if col_sides[idx] == "top":
            _draw_arrow(grid, r0 - 4, col - 1, pair_color, "down")
        else:
            _draw_arrow(grid, r1 + 2, col - 1, pair_color, "up")

    expected = np.array([deco["expected"]], dtype=int)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_attempt()
        if built is None:
            continue
        grid, expected = built
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
