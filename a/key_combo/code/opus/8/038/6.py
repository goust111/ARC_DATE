from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

RECT_COLORS = (BLUE, GREEN, YELLOW, TEAL)
GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (14, 18)
RECT_DIMS = ((4, 4), (4, 5), (5, 4), (4, 6), (6, 4), (5, 5))
TARGET_DIMS = ((4, 4), (4, 5), (5, 4), (4, 6), (6, 4))
WALL_LEN_RANGE = (3, 6)
MAX_ATTEMPTS = 500
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS = ("tl", "tr", "bl", "br")


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_corner_bracket(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 5:
        return False
    r0, c0, r1, c1 = _bbox(comp)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    pts = set(comp)
    candidates = [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]
    return any(pts == candidate for candidate in candidates)


def _find_bracket(grid: np.ndarray) -> dict:
    brackets = []
    for comp in _components_of_color(grid, RED):
        if _is_corner_bracket(comp):
            brackets.append(comp)
    assert len(brackets) == 1, "expected exactly one red 3x3 corner bracket"
    comp = brackets[0]
    return {"cells": set(comp), "bbox": _bbox(comp)}


def _find_marker(grid: np.ndarray, bracket: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = bracket["bbox"]
    bracket_cells = bracket["cells"]
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_cells and int(grid[r, c]) not in (ORANGE, RED)
    ]
    assert len(markers) == 1, "expected one non-orange marker inside the bracket"
    return markers[0]


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in RECT_COLORS:
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h >= 4 and w >= 4 and len(comp) == h * w:
                rectangles.append(
                    {
                        "color": color,
                        "cells": set(comp),
                        "bbox": (r0, c0, r1, c1),
                        "shape": (h, w),
                    }
                )
    assert 2 <= len(rectangles) <= 3, "expected 2-3 solid rectangles"
    return rectangles


def _rect_border(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    border = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                border.append((r, c))
    return border


def _rect_interior(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _reachable_maroons(grid: np.ndarray, target_box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    starts = []
    seen = set()
    for r, c in _rect_border(target_box):
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) in (ORANGE, MAROON) and (nr, nc) not in seen:
                seen.add((nr, nc))
                starts.append((nr, nc))
    assert starts, "target rectangle must border at least one passable cell"

    q = deque(starts)
    reachable_maroons: set[tuple[int, int]] = set()
    while q:
        r, c = q.popleft()
        if int(grid[r, c]) == MAROON:
            reachable_maroons.add((r, c))
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and (nr, nc) not in seen and int(grid[nr, nc]) in (ORANGE, MAROON):
                seen.add((nr, nc))
                q.append((nr, nc))
    return reachable_maroons


def _reachable_background(grid: np.ndarray, target_box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    starts = []
    seen = set()
    for r, c in _rect_border(target_box):
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) == ORANGE and (nr, nc) not in seen:
                seen.add((nr, nc))
                starts.append((nr, nc))

    q = deque(starts)
    cells = list(starts)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and (nr, nc) not in seen and int(grid[nr, nc]) == ORANGE:
                seen.add((nr, nc))
                q.append((nr, nc))
                cells.append((nr, nc))
    return cells


def _rank_key(cell: tuple[int, int], target_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r, c = cell
    r0, c0, r1, c1 = target_box
    return abs(2 * r - (r0 + r1)) + abs(2 * c - (c0 + c1)), r, c


def _analyze(grid: np.ndarray) -> dict:
    bracket = _find_bracket(grid)
    marker = _find_marker(grid, bracket)
    marker_color = int(grid[marker])
    rectangles = _find_rectangles(grid)
    targets = [rect for rect in rectangles if rect["color"] == marker_color]
    assert len(targets) == 1, "marker color must identify exactly one rectangle"
    target = targets[0]
    interior = _rect_interior(target["bbox"])
    capacity = len(interior)
    assert capacity > 0
    reachable = _reachable_maroons(grid, target["bbox"])
    ranked = sorted(reachable, key=lambda cell: _rank_key(cell, target["bbox"]))
    chosen = ranked[:capacity]
    maroons = {tuple(map(int, rc)) for rc in np.argwhere(grid == MAROON)}
    return {
        "bracket": bracket,
        "marker": marker,
        "marker_color": marker_color,
        "rectangles": rectangles,
        "target": target,
        "interior": interior,
        "capacity": capacity,
        "reachable": reachable,
        "ranked": ranked,
        "chosen": chosen,
        "unreachable": maroons - reachable,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the interior of the rectangle named by the bracket marker.

    Input structure:
    - The background is orange, with one red 3x3 corner bracket containing a single non-orange marker pixel.
    - The grid also contains 2-3 solid colored rectangles, several straight magenta wall segments, and scattered maroon pixels.
    - Exactly one rectangle has the same color as the bracket marker; that rectangle is the destination.

    Rule:
    - Read the marker color inside the bracket and select the solid rectangle of that same color.
    - Let N be the area of the selected rectangle after removing its one-pixel border.
    - From passable cells adjacent to the rectangle border, flood through orange and maroon cells only; reachable maroon pixels are the candidates.
    - Sort candidate maroons by Manhattan distance to the rectangle center, then by row, then by column; move the first N into the rectangle interior in row-major order, clearing their source cells to orange.

    Key insight:
    - The bracket does not choose what to erase; it chooses which rectangle becomes the gathering target, and walls can exclude nearby maroons that are geometrically close but unreachable.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze(g)

    out = g.copy()
    for r, c in analysis["chosen"]:
        out[r, c] = ORANGE
    for dest, src in zip(analysis["interior"], analysis["chosen"]):
        out[dest] = MAROON
    return out


def _box_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expanded_box(box: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - margin), max(0, c0 - margin), min(shape[0] - 1, r1 + margin), min(shape[1] - 1, c1 + margin)


def _make_bracket(h: int, w: int, corner: str) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]:
    if corner == "tl":
        cells = {(0, c) for c in range(3)} | {(r, 0) for r in range(3)}
        return cells, (1, 1), (0, 0, 2, 2)
    if corner == "tr":
        cells = {(0, c) for c in range(w - 3, w)} | {(r, w - 1) for r in range(3)}
        return cells, (1, w - 2), (0, w - 3, 2, w - 1)
    if corner == "bl":
        cells = {(h - 1, c) for c in range(3)} | {(r, 0) for r in range(h - 3, h)}
        return cells, (h - 2, 1), (h - 3, 0, h - 1, 2)
    cells = {(h - 1, c) for c in range(w - 3, w)} | {(r, w - 1) for r in range(h - 3, h)}
    return cells, (h - 2, w - 2), (h - 3, w - 3, h - 1, w - 1)


def _make_pocket_walls(h: int, w: int, corner: str, wall_h: int, wall_w: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    walls: set[tuple[int, int]] = set()
    pocket: set[tuple[int, int]] = set()
    if corner == "tl":
        walls |= {(r, wall_w) for r in range(wall_h)}
        walls |= {(wall_h, c) for c in range(wall_w)}
        pocket = {(r, c) for r in range(wall_h) for c in range(wall_w)}
    elif corner == "tr":
        vc = w - wall_w - 1
        walls |= {(r, vc) for r in range(wall_h)}
        walls |= {(wall_h, c) for c in range(w - wall_w, w)}
        pocket = {(r, c) for r in range(wall_h) for c in range(w - wall_w, w)}
    elif corner == "bl":
        hr = h - wall_h - 1
        walls |= {(r, wall_w) for r in range(h - wall_h, h)}
        walls |= {(hr, c) for c in range(wall_w)}
        pocket = {(r, c) for r in range(h - wall_h, h) for c in range(wall_w)}
    else:
        hr = h - wall_h - 1
        vc = w - wall_w - 1
        walls |= {(r, vc) for r in range(h - wall_h, h)}
        walls |= {(hr, c) for c in range(w - wall_w, w)}
        pocket = {(r, c) for r in range(h - wall_h, h) for c in range(w - wall_w, w)}
    return walls, pocket


def _target_candidates(
    shape: tuple[int, int],
    pocket_corner: str,
    wall_h: int,
    wall_w: int,
    rect_h: int,
    rect_w: int,
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    candidates: set[tuple[int, int, int, int]] = set()
    for gap in (0, 1):
        if pocket_corner == "tl":
            for r0 in range(0, max(0, min(h - rect_h, wall_h - 1)) + 1):
                c0 = wall_w + 1 + gap
                if c0 + rect_w <= w:
                    candidates.add((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
            r0 = wall_h + 1 + gap
            if r0 + rect_h <= h:
                for c0 in range(0, min(w - rect_w, wall_w - 1) + 1):
                    candidates.add((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
        elif pocket_corner == "tr":
            for r0 in range(0, max(0, min(h - rect_h, wall_h - 1)) + 1):
                c1 = w - wall_w - 2 - gap
                c0 = c1 - rect_w + 1
                if 0 <= c0:
                    candidates.add((r0, c0, r0 + rect_h - 1, c1))
            r0 = wall_h + 1 + gap
            if r0 + rect_h <= h:
                for c0 in range(max(0, w - wall_w - rect_w), w - rect_w + 1):
                    candidates.add((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
        elif pocket_corner == "bl":
            for r0 in range(max(0, h - wall_h - rect_h), h - rect_h + 1):
                c0 = wall_w + 1 + gap
                if c0 + rect_w <= w:
                    candidates.add((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
            r1 = h - wall_h - 2 - gap
            r0 = r1 - rect_h + 1
            if 0 <= r0:
                for c0 in range(0, min(w - rect_w, wall_w - 1) + 1):
                    candidates.add((r0, c0, r1, c0 + rect_w - 1))
        else:
            for r0 in range(max(0, h - wall_h - rect_h), h - rect_h + 1):
                c1 = w - wall_w - 2 - gap
                c0 = c1 - rect_w + 1
                if 0 <= c0:
                    candidates.add((r0, c0, r0 + rect_h - 1, c1))
            r1 = h - wall_h - 2 - gap
            r0 = r1 - rect_h + 1
            if 0 <= r0:
                for c0 in range(max(0, w - wall_w - rect_w), w - rect_w + 1):
                    candidates.add((r0, c0, r1, c0 + rect_w - 1))
    return sorted(candidates)


def _box_ok(box: tuple[int, int, int, int], forbidden: set[tuple[int, int]], shape: tuple[int, int], margin: int) -> bool:
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= shape[0] or c1 >= shape[1]:
        return False
    er0, ec0, er1, ec1 = _expanded_box(box, margin, shape)
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if (r, c) in forbidden:
                return False
    return True


def _extra_wall_candidates(grid: np.ndarray, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[list[tuple[int, int]]] = []
    for length in range(WALL_LEN_RANGE[0], WALL_LEN_RANGE[1] + 1):
        for r in range(h):
            for c0 in range(w - length + 1):
                seg = [(r, c) for c in range(c0, c0 + length)]
                if any(cell in forbidden for cell in seg):
                    continue
                if any(int(grid[cell]) != ORANGE for cell in seg):
                    continue
                if any(
                    _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) == PINK and (nr, nc) not in seg
                    for cell in seg
                    for nr, nc in _neighbors4(*cell)
                ):
                    continue
                candidates.append(seg)
        for c in range(w):
            for r0 in range(h - length + 1):
                seg = [(r, c) for r in range(r0, r0 + length)]
                if any(cell in forbidden for cell in seg):
                    continue
                if any(int(grid[cell]) != ORANGE for cell in seg):
                    continue
                if any(
                    _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) == PINK and (nr, nc) not in seg
                    for cell in seg
                    for nr, nc in _neighbors4(*cell)
                ):
                    continue
                candidates.append(seg)
    return candidates


def _place_extra_walls(grid: np.ndarray, forbidden: set[tuple[int, int]]) -> None:
    for _ in range(random.choice((0, 1, 1, 2))):
        candidates = _extra_wall_candidates(grid, forbidden)
        if not candidates:
            return
        seg = random.choice(candidates)
        for cell in seg:
            grid[cell] = PINK
        forbidden.update(seg)
        for r, c in seg:
            forbidden.update((nr, nc) for nr, nc in _neighbors4(r, c) if _in_bounds(grid.shape, nr, nc))


def _straight_wall_components_ok(grid: np.ndarray) -> bool:
    comps = _components_of_color(grid, PINK)
    if len(comps) < 2:
        return False
    for comp in comps:
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(comp) < 3 or len(comp) > 6:
            return False
        if len(rows) != 1 and len(cols) != 1:
            return False
    return True


def _choose_reachable_positions(
    cells: list[tuple[int, int]],
    target_box: tuple[int, int, int, int],
    capacity: int,
    want_tie: bool,
) -> list[tuple[int, int]] | None:
    ranked = sorted(cells, key=lambda cell: _rank_key(cell, target_box))
    if len(ranked) <= capacity:
        return None

    min_count = capacity + 2
    max_count = min(len(ranked), capacity + 4)
    if want_tie:
        buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for cell in ranked:
            buckets[_rank_key(cell, target_box)[0]].append(cell)
        pre = 0
        for dist in sorted(buckets):
            bucket = buckets[dist]
            if pre < capacity < pre + len(bucket):
                cutoff = pre + len(bucket)
                tail = random.randint(0, max(0, min(2, len(ranked) - cutoff)))
                count = cutoff + tail
                if min_count <= count <= max_count:
                    return ranked[:count]
            pre += len(bucket)
    return ranked[: random.randint(min_count, max_count)]


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze(grid)
        out = main(grid)
    except Exception:
        return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if len(analysis["reachable"]) <= analysis["capacity"]:
        return False
    if len(analysis["chosen"]) != analysis["capacity"]:
        return False
    if len(analysis["unreachable"]) == 0:
        return False
    if not _straight_wall_components_ok(grid):
        return False

    chosen = analysis["chosen"]
    ranked = analysis["ranked"]
    chosen_max = max(_rank_key(cell, analysis["target"]["bbox"])[0] for cell in chosen)
    unreachable_min = min(_rank_key(cell, analysis["target"]["bbox"])[0] for cell in analysis["unreachable"])
    if unreachable_min > chosen_max:
        return False

    interior = analysis["interior"]
    if any(int(grid[cell]) != analysis["marker_color"] for cell in interior):
        return False
    if not all(int(out[cell]) == MAROON for cell in interior):
        return False
    if any(int(out[cell]) != ORANGE for cell in chosen):
        return False
    if any(int(out[cell]) != MAROON for cell in analysis["unreachable"]):
        return False
    if np.array_equal(grid, out):
        return False

    has_cutoff_tie = False
    if len(ranked) > analysis["capacity"]:
        left = _rank_key(ranked[analysis["capacity"] - 1], analysis["target"]["bbox"])[0]
        right = _rank_key(ranked[analysis["capacity"]], analysis["target"]["bbox"])[0]
        has_cutoff_tie = left == right

    maroon_count = int(np.count_nonzero(grid == MAROON))
    if not (8 <= maroon_count <= 15):
        return False
    if has_cutoff_tie:
        return True
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        grid = np.full((h, w), ORANGE, dtype=int)

        pocket_corner = random.choice(CORNERS)
        bracket_corner = random.choice([corner for corner in CORNERS if corner != pocket_corner])
        wall_h = random.randint(*WALL_LEN_RANGE)
        wall_w = random.randint(*WALL_LEN_RANGE)

        bracket_cells, marker_cell, bracket_box = _make_bracket(h, w, bracket_corner)
        walls, pocket = _make_pocket_walls(h, w, pocket_corner, wall_h, wall_w)
        if bracket_cells & pocket or bracket_cells & walls or marker_cell in pocket or marker_cell in walls:
            continue

        reserved = set(bracket_cells) | _box_cells(bracket_box) | pocket | walls
        rect_colors = random.sample(RECT_COLORS, random.choice((2, 3)))
        target_color = random.choice(rect_colors)
        target_h, target_w = random.choice(TARGET_DIMS)

        target_boxes = [
            box
            for box in _target_candidates((h, w), pocket_corner, wall_h, wall_w, target_h, target_w)
            if _box_ok(box, reserved, (h, w), margin=0)
        ]
        if not target_boxes:
            continue
        target_box = random.choice(target_boxes)

        boxes = {target_color: target_box}
        forbidden = reserved | _box_cells(_expanded_box(target_box, 1, (h, w)))
        other_colors = [color for color in rect_colors if color != target_color]
        failed = False
        for color in other_colors:
            dims = list(RECT_DIMS)
            random.shuffle(dims)
            placed = False
            for rect_h, rect_w in dims:
                candidates = []
                for r0 in range(h - rect_h + 1):
                    for c0 in range(w - rect_w + 1):
                        box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                        if _box_ok(box, forbidden, (h, w), margin=1):
                            candidates.append(box)
                if candidates:
                    box = random.choice(candidates)
                    boxes[color] = box
                    forbidden |= _box_cells(_expanded_box(box, 1, (h, w)))
                    placed = True
                    break
            if not placed:
                failed = True
                break
        if failed:
            continue

        for cell in walls:
            grid[cell] = PINK
        for color, box in boxes.items():
            r0, c0, r1, c1 = box
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
        for cell in bracket_cells:
            grid[cell] = RED
        grid[marker_cell] = target_color

        wall_forbidden = set(bracket_cells) | _box_cells(bracket_box) | pocket
        for box in boxes.values():
            wall_forbidden |= _box_cells(_expanded_box(box, 1, (h, w)))
        _place_extra_walls(grid, wall_forbidden)

        try:
            analysis = _analyze(grid)
        except Exception:
            continue
        if analysis["target"]["bbox"] != target_box:
            continue

        capacity = analysis["capacity"]
        want_tie = random.random() < 0.45
        reachable_positions = _choose_reachable_positions(_reachable_background(grid, target_box), target_box, capacity, want_tie)
        if reachable_positions is None:
            continue

        unreachable_pool = sorted(
            [cell for cell in pocket if int(grid[cell]) == ORANGE],
            key=lambda cell: _rank_key(cell, target_box),
        )
        if not unreachable_pool:
            continue

        max_unreachable = min(len(unreachable_pool), 4)
        min_unreachable = 2 if capacity <= 6 else 1
        unreachable_count = random.randint(min_unreachable, max_unreachable)
        unreachable_positions = unreachable_pool[:unreachable_count]

        total_maroons = len(reachable_positions) + len(unreachable_positions)
        if total_maroons < 8 or total_maroons > 15:
            continue

        for cell in reachable_positions + unreachable_positions:
            grid[cell] = MAROON

        if not _validate_generated(grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
