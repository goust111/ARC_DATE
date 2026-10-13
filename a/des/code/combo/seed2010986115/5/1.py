from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
AZURE = 8
SEED_COLORS = (1, 2, 3, 4, 6, 9)

HORIZONTAL = "horizontal"
VERTICAL = "vertical"

GEN_RETRIES = 300
PLACEMENT_TRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _pair_orientation(points: list[tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    assert len(points) == 2, "expected exactly two pixels in a marker or seed pair"
    (r0, c0), (r1, c1) = sorted(points)
    if r0 == r1 and abs(c0 - c1) == 1:
        return HORIZONTAL, (r0, min(c0, c1))
    if c0 == c1 and abs(r0 - r1) == 1:
        return VERTICAL, (min(r0, r1), c0)
    raise AssertionError("pair must be axis-aligned and adjacent")


def _extract_patterns(grid: np.ndarray) -> dict[str, tuple[int, int, set[tuple[int, int]]]]:
    patterns: dict[str, tuple[int, int, set[tuple[int, int]]]] = {}
    for comp in _components(grid != GREY):
        vals = {int(grid[r, c]) for r, c in comp}
        if ORANGE not in vals or AZURE not in vals:
            continue

        azure_points = [(r, c) for r, c in comp if int(grid[r, c]) == AZURE]
        orient, anchor = _pair_orientation(azure_points)
        r0, c0, r1, c1 = _bbox(comp)
        period_h = r1 - r0 + 1
        period_w = c1 - c0 + 1
        orange_offsets: set[tuple[int, int]] = set()
        for r, c in comp:
            if int(grid[r, c]) == ORANGE:
                orange_offsets.add(((r - anchor[0]) % period_h, (c - anchor[1]) % period_w))
        assert orange_offsets, "template room must contain orange pattern cells"
        patterns[orient] = (period_h, period_w, orange_offsets)

    assert set(patterns) == {HORIZONTAL, VERTICAL}, "must find both template patterns"
    return patterns


def _seed_room_info(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, str, tuple[int, int]] | None:
    colors = sorted({int(grid[r, c]) for r, c in comp if int(grid[r, c]) != BLACK})
    colors = [color for color in colors if color not in (ORANGE, AZURE)]
    if len(colors) != 1:
        return None
    seed_color = colors[0]
    seed_points = [(r, c) for r, c in comp if int(grid[r, c]) == seed_color]
    if len(seed_points) != 2:
        return None
    orient, anchor = _pair_orientation(seed_points)
    return seed_color, orient, anchor


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced region using the template pattern that matches its seed orientation.

    Grey pixels are immutable fences. Template rooms are recognized by the presence of
    an azure 2-pixel orientation marker together with orange pattern pixels. For every
    ordinary room, the two same-colored seed pixels determine both the fill color and
    whether the horizontal or vertical template should be tiled from the seed's anchor.
    """
    g = np.asarray(grid, dtype=int)
    patterns = _extract_patterns(g)

    out = g.copy()
    for comp in _components(g != GREY):
        info = _seed_room_info(g, comp)
        if info is None:
            continue
        seed_color, orient, anchor = info
        period_h, period_w, orange_offsets = patterns[orient]
        for r, c in comp:
            if int(g[r, c]) != BLACK:
                continue
            if ((r - anchor[0]) % period_h, (c - anchor[1]) % period_w) in orange_offsets:
                out[r, c] = seed_color
    return out


def _draw_box(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int = GREY) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _marker_cells(orient: str, anchor: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = anchor
    if orient == HORIZONTAL:
        return [(r, c), (r, c + 1)]
    return [(r, c), (r + 1, c)]


def _random_pattern(height: int, width: int, orient: str) -> set[tuple[int, int]]:
    reserved = {(0, 0), (0, 1)} if orient == HORIZONTAL else {(0, 0), (1, 0)}
    candidates = [(r, c) for r in range(height) for c in range(width) if (r, c) not in reserved]
    assert len(candidates) >= 2
    min_count = max(2, len(candidates) // 3)
    max_count = max(min_count, min(len(candidates) - 1, (2 * len(candidates)) // 3))
    count = random.randint(min_count, max_count)
    return set(random.sample(candidates, count))


def _apply_pattern(
    out: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    anchor: tuple[int, int],
    color: int,
    pattern: tuple[int, int, set[tuple[int, int]]],
) -> int:
    period_h, period_w, orange_offsets = pattern
    filled = 0
    for r in range(top, top + height):
        for c in range(left, left + width):
            if int(out[r, c]) != BLACK:
                continue
            if ((r - anchor[0]) % period_h, (c - anchor[1]) % period_w) in orange_offsets:
                out[r, c] = color
                filled += 1
    return filled


def _render_template_room(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    orient: str,
    orange_cells: set[tuple[int, int]],
) -> None:
    for r, c in _marker_cells(orient, (top, left)):
        grid[r, c] = AZURE
    for dr, dc in orange_cells:
        grid[top + dr, left + dc] = ORANGE


def _template_layout() -> tuple[
    tuple[int, int],
    list[tuple[int, int, int, int]],
    list[str],
    dict[str, tuple[int, int, set[tuple[int, int]]]],
]:
    side_by_side = random.choice([False, True])
    room_order = [HORIZONTAL, VERTICAL]
    random.shuffle(room_order)

    if side_by_side:
        room_h = random.randint(3, 6)
        widths = [random.randint(3, 6), random.randint(3, 6)]
        outer_h = room_h + 2
        outer_w = widths[0] + widths[1] + 3
        rooms = [
            (1, 1, room_h, widths[0]),
            (1, 2 + widths[0], room_h, widths[1]),
        ]
    else:
        room_w = random.randint(3, 6)
        heights = [random.randint(3, 6), random.randint(3, 6)]
        outer_h = heights[0] + heights[1] + 3
        outer_w = room_w + 2
        rooms = [
            (1, 1, heights[0], room_w),
            (2 + heights[0], 1, heights[1], room_w),
        ]

    patterns: dict[str, tuple[int, int, set[tuple[int, int]]]] = {}
    room_specs: list[tuple[int, int, int, int]] = []
    used_specs: set[tuple[int, int, frozenset[tuple[int, int]]]] = set()
    for orient, (r0, c0, h, w) in zip(room_order, rooms, strict=False):
        while True:
            orange_cells = _random_pattern(h, w, orient)
            sig = (h, w, frozenset(orange_cells))
            if sig not in used_specs:
                used_specs.add(sig)
                patterns[orient] = (h, w, orange_cells)
                room_specs.append((r0, c0, h, w))
                break

    return (outer_h, outer_w), room_specs, room_order, patterns


def _place_template(
    grid: np.ndarray,
    corner: str,
    outer_shape: tuple[int, int],
    room_specs: list[tuple[int, int, int, int]],
    room_order: list[str],
    patterns: dict[str, tuple[int, int, set[tuple[int, int]]]],
) -> tuple[int, int, int, int]:
    outer_h, outer_w = outer_shape
    h, w = grid.shape
    if corner == "TL":
        top, left = 0, 0
    elif corner == "TR":
        top, left = 0, w - outer_w
    elif corner == "BL":
        top, left = h - outer_h, 0
    else:
        top, left = h - outer_h, w - outer_w

    _draw_box(grid, top, left, outer_h, outer_w)
    if room_specs[0][0] == room_specs[1][0]:
        divider_col = left + room_specs[0][1] + room_specs[0][3]
        grid[top : top + outer_h, divider_col] = GREY
    else:
        divider_row = top + room_specs[0][0] + room_specs[0][2]
        grid[divider_row, left : left + outer_w] = GREY

    for orient, (r0, c0, hh, ww) in zip(room_order, room_specs, strict=False):
        pattern_h, pattern_w, orange_cells = patterns[orient]
        assert (pattern_h, pattern_w) == (hh, ww)
        _render_template_room(grid, top + r0, left + c0, hh, ww, orient, orange_cells)
    return top, left, top + outer_h - 1, left + outer_w - 1


def _sample_room_box(
    grid_h: int,
    grid_w: int,
    inner_h: int,
    inner_w: int,
    taken: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if grid_h - outer_h - 2 < 1 or grid_w - outer_w - 2 < 1:
        return None
    for _ in range(PLACEMENT_TRIES):
        top = random.randint(1, grid_h - outer_h - 2)
        left = random.randint(1, grid_w - outer_w - 2)
        box = (top, left, top + outer_h - 1, left + outer_w - 1)
        if any(_boxes_too_close(box, other) for other in taken):
            continue
        return box
    return None


def _seed_anchor(inner_top: int, inner_left: int, inner_h: int, inner_w: int, orient: str) -> tuple[int, int]:
    if orient == HORIZONTAL:
        return inner_top + random.randint(0, inner_h - 1), inner_left + random.randint(0, inner_w - 2)
    return inner_top + random.randint(0, inner_h - 2), inner_left + random.randint(0, inner_w - 1)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    room_count = random.randint(2, min(5, len(SEED_COLORS)))
    orientations = [HORIZONTAL, VERTICAL]
    while len(orientations) < room_count:
        orientations.append(random.choice([HORIZONTAL, VERTICAL]))
    random.shuffle(orientations)

    outer_shape, template_rooms, template_order, patterns = _template_layout()

    min_h = max(22, outer_shape[0] + 8)
    min_w = max(22, outer_shape[1] + 8)
    grid_h = random.randint(min_h, 32)
    grid_w = random.randint(min_w, 32)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    template_box = _place_template(
        grid,
        random.choice(["TL", "TR", "BL", "BR"]),
        outer_shape,
        template_rooms,
        template_order,
        patterns,
    )

    taken = [template_box]
    room_specs: list[tuple[int, int, int, int, int, str, tuple[int, int]]] = []
    colors = random.sample(SEED_COLORS, room_count)

    for color, orient in zip(colors, orientations, strict=False):
        inner_h = random.randint(4, 9)
        inner_w = random.randint(4, 9)
        box = _sample_room_box(grid_h, grid_w, inner_h, inner_w, taken)
        if box is None:
            return None
        top, left, bottom, right = box
        _draw_box(grid, top, left, bottom - top + 1, right - left + 1)
        inner_top = top + 1
        inner_left = left + 1
        anchor = _seed_anchor(inner_top, inner_left, inner_h, inner_w, orient)
        for r, c in _marker_cells(orient, anchor):
            grid[r, c] = color
        taken.append(box)
        room_specs.append((inner_top, inner_left, inner_h, inner_w, color, orient, anchor))

    expected = grid.copy()
    added = {HORIZONTAL: 0, VERTICAL: 0}
    for inner_top, inner_left, inner_h, inner_w, color, orient, anchor in room_specs:
        gained = _apply_pattern(expected, inner_top, inner_left, inner_h, inner_w, anchor, color, patterns[orient])
        added[orient] += gained

    if added[HORIZONTAL] == 0 or added[VERTICAL] == 0:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is not None:
            return case[0]
    raise AssertionError("failed to generate a valid input grid")
