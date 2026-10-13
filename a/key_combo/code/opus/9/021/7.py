from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

DIAG_DOWN = "down"  # "\" : r-c is constant
DIAG_UP = "up"  # "/" : r+c is constant


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def _diag_value(coord: tuple[int, int], orientation: str) -> int:
    r, c = coord
    return r - c if orientation == DIAG_DOWN else r + c


def _line_cells(shape: tuple[int, int], orientation: str, value: int) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        c = r - value if orientation == DIAG_DOWN else value - r
        if 0 <= c < w:
            cells.append((r, c))
    return cells


def _draw_full_diagonal(grid: np.ndarray, orientation: str, value: int, color: int) -> None:
    for r, c in _line_cells(grid.shape, orientation, value):
        grid[r, c] = color


def _parse_original_diagonal(grid: np.ndarray) -> tuple[str, int, list[tuple[int, int]]]:
    green = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert green, "expected a green diagonal"
    down_values = {_diag_value(cell, DIAG_DOWN) for cell in green}
    up_values = {_diag_value(cell, DIAG_UP) for cell in green}
    if len(down_values) == 1:
        orientation = DIAG_DOWN
        value = next(iter(down_values))
    else:
        assert len(up_values) == 1, "green pixels must form one 45-degree diagonal"
        orientation = DIAG_UP
        value = next(iter(up_values))
    expected = set(_line_cells(grid.shape, orientation, value))
    assert set(green) == expected, "original diagonal must be a maximal border-to-border segment"
    return orientation, int(value), green


def _guide_tip(guide: list[tuple[int, int]], orientation: str, base_value: int) -> tuple[int, int]:
    distances = [abs(_diag_value(cell, orientation) - base_value) for cell in guide]
    max_distance = max(distances)
    assert max_distance > 0, "guide must sit off the original diagonal"
    tips = [cell for cell, dist in zip(guide, distances, strict=False) if dist == max_distance]
    assert len(tips) == 1, "guide must have a unique far tip"
    return tips[0]


def _extract_plus_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for comp in _components_of_color(grid, BLUE):
        assert len(comp) == 5, "each blue object must be a 5-pixel plus"
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 == 2 and c1 - c0 == 2, "plus bbox must be 3x3"
        center = (r0 + 1, c0 + 1)
        assert set(comp) == _plus_cells(center), "blue object must be a plus"
        centers.append(center)
    assert 4 <= len(centers) <= 6, "expected 4-6 pluses"
    return sorted(centers)


def _interior_diagonal(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    dr = r2 - r1
    dc = c2 - c1
    if abs(dr) != abs(dc) or dr == 0:
        return []
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    return [(r1 + step * sr, c1 + step * sc) for step in range(1, abs(dr))]


def _build_construction(grid: np.ndarray) -> tuple[np.ndarray, str, int, set[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    orientation, base_value, _ = _parse_original_diagonal(g)
    guides = _components_of_color(g, GREY)
    assert 2 <= len(guides) <= 3, "expected 2-3 grey guides"
    plus_centers = _extract_plus_centers(g)

    line_values = {base_value}
    out = g.copy()
    tips: list[tuple[int, int]] = []
    for guide in guides:
        tip = _guide_tip(guide, orientation, base_value)
        tips.append(tip)
        value = _diag_value(tip, orientation)
        line_values.add(value)
        _draw_full_diagonal(out, orientation, value, GREEN)

    out[out == GREY] = BLACK
    return out, orientation, base_value, line_values, plus_centers, tips


def _solve_pairs(
    base: np.ndarray,
    orientation: str,
    line_values: set[int],
    centers: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    red_cells: set[tuple[int, int]] = set()
    participants: set[tuple[int, int]] = set()
    connected_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for a, b in combinations(centers, 2):
        interior = _interior_diagonal(a, b)
        if not interior:
            continue
        if any(int(base[r, c]) not in (BLACK, GREEN) for r, c in interior):
            continue
        if not any(int(base[r, c]) == GREEN for r, c in interior):
            continue
        newly_red = {(r, c) for r, c in interior if int(base[r, c]) == BLACK}
        if not newly_red:
            continue
        red_cells |= newly_red
        participants.add(a)
        participants.add(b)
        connected_pairs.append((a, b))

    return red_cells, participants, connected_pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate parallel green diagonals from the grey guides, then use those diagonals to enable diagonal plus-to-plus links.

    Input structure:
    - A black canvas contains one green 45-degree diagonal segment, 2-3 grey guide triangles, and 4-6 blue 5-pixel pluses.
    - Each grey guide is a small right-triangle component touching the original green diagonal and has one unique pixel farthest from that diagonal.
    - Pluses do not overlap the original diagonal or the guides.

    Rule:
    - Detect the orientation of the original green diagonal.
    - For each grey guide, find its unique farthest pixel and draw a full border-to-border green diagonal through that pixel, parallel to the original one; after all such lines are drawn, erase every grey pixel to black.
    - Check every diagonally aligned plus-center pair. A pair is enabled only when every interior pixel on its center-to-center 45-degree segment is black or green, the segment uses at least one old/new green diagonal pixel as support, and at least one interior black pixel remains to be repainted red.
    - For every enabled pair, repaint the black interior pixels red while keeping existing green pixels unchanged.
    - Recolor every plus that participates in at least one such red connector from blue to orange.

    Key insight:
    - The grey triangles are construction guides: their far tips create the green diagonal support lines that make some diagonal center-to-center paths eligible while similar all-black diagonal gaps stay inactive.
    """
    base, orientation, _base_value, line_values, centers, _tips = _build_construction(grid)
    red_cells, participants, _pairs = _solve_pairs(base, orientation, line_values, centers)

    out = base.copy()
    for r, c in red_cells:
        out[r, c] = RED
    for center in participants:
        for r, c in _plus_cells(center):
            out[r, c] = ORANGE
    return out


def _guide_cells(orientation: str, delta: int, tip: tuple[int, int]) -> set[tuple[int, int]]:
    size = abs(delta)
    tr, tc = tip
    cells: set[tuple[int, int]] = set()
    if orientation == DIAG_DOWN:
        if delta > 0:
            r0, c0 = tr - size + 1, tc
            for i in range(size):
                for j in range(i + 1):
                    cells.add((r0 + i, c0 + j))
        else:
            r0, c0 = tr, tc - size + 1
            for i in range(size):
                for j in range(i, size):
                    cells.add((r0 + i, c0 + j))
    else:
        if delta > 0:
            r0, c0 = tr - size + 1, tc - size + 1
            for i in range(size):
                for j in range(size):
                    if i + j >= size - 1:
                        cells.add((r0 + i, c0 + j))
        else:
            r0, c0 = tr, tc
            for i in range(size):
                for j in range(size):
                    if i + j <= size - 1:
                        cells.add((r0 + i, c0 + j))
    return cells


def _cells_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _cells_touch(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return True
    return False


def _can_place_plus(
    center: tuple[int, int],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    existing_pluses: list[set[tuple[int, int]]],
) -> bool:
    h, w = shape
    r, c = center
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    cells = _plus_cells(center)
    if any(cell in blocked for cell in cells):
        return False
    for other in existing_pluses:
        if _cells_touch(cells, other):
            return False
    return True


def _shuffled(seq: list) -> list:
    out = list(seq)
    random.shuffle(out)
    return out


def _choose_base_value(shape: tuple[int, int], orientation: str) -> int:
    h, w = shape
    if orientation == DIAG_DOWN:
        candidates = [
            value
            for value in range(-(w - 1), h)
            if len(_line_cells(shape, orientation, value)) >= 9
        ]
    else:
        candidates = [
            value
            for value in range(h + w - 1)
            if len(_line_cells(shape, orientation, value)) >= 9
        ]
    return random.choice(candidates)


def _place_active_guide(
    shape: tuple[int, int],
    orientation: str,
    line_value: int,
    delta_offset: int,
    blocked: set[tuple[int, int]],
    existing_pluses: list[set[tuple[int, int]]],
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    line = _line_cells(shape, orientation, line_value)
    index_by_cell = {cell: idx for idx, cell in enumerate(line)}
    candidates = [
        cell
        for cell in line
        if 1 <= cell[0] < shape[0] - 1 and 1 <= cell[1] < shape[1] - 1
    ]
    random.shuffle(candidates)

    for tip in candidates:
        guide = _guide_cells(orientation, delta_offset, tip)
        if not _cells_in_bounds(shape, guide) or guide & blocked:
            continue

        tip_idx = index_by_cell[tip]
        left_cells = [cell for cell in line[: max(0, tip_idx - 2)]]
        right_cells = [cell for cell in line[tip_idx + 3 :]]
        left_cells = _shuffled(left_cells)
        right_cells = _shuffled(right_cells)
        if not left_cells or not right_cells:
            continue

        blocked_now = blocked | guide
        for left in left_cells:
            if not _can_place_plus(left, shape, blocked_now, existing_pluses):
                continue
            left_plus = _plus_cells(left)
            for right in right_cells:
                if not _can_place_plus(right, shape, blocked_now, existing_pluses + [left_plus]):
                    continue
                interior = _interior_diagonal(left, right)
                if tip not in interior:
                    continue
                return guide, tip, left, right
    return None


def _place_inactive_guide(
    shape: tuple[int, int],
    orientation: str,
    line_value: int,
    delta_offset: int,
    blocked: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    line = _line_cells(shape, orientation, line_value)
    candidates = [
        cell
        for cell in line
        if 1 <= cell[0] < shape[0] - 1 and 1 <= cell[1] < shape[1] - 1
    ]
    random.shuffle(candidates)
    for tip in candidates:
        guide = _guide_cells(orientation, delta_offset, tip)
        if _cells_in_bounds(shape, guide) and not (guide & blocked):
            return guide, tip
    return None


def _opposite_orientation(orientation: str) -> str:
    return DIAG_UP if orientation == DIAG_DOWN else DIAG_DOWN


def _support_grid(
    shape: tuple[int, int],
    orientation: str,
    base_value: int,
    guides: list[set[tuple[int, int]]],
) -> tuple[np.ndarray, set[int]]:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in _line_cells(shape, orientation, base_value):
        grid[r, c] = GREEN
    for guide in guides:
        for r, c in guide:
            grid[r, c] = GREY

    line_values = {base_value}
    out = grid.copy()
    for guide in guides:
        tip = _guide_tip(sorted(guide), orientation, base_value)
        value = _diag_value(tip, orientation)
        line_values.add(value)
        _draw_full_diagonal(out, orientation, value, GREEN)
    out[out == GREY] = BLACK
    return out, line_values


def _pair_support_values(
    support: np.ndarray,
    green_orientation: str,
    line_values: set[int],
    interior: list[tuple[int, int]],
) -> set[int]:
    used = set()
    for cell in interior:
        if int(support[cell]) == GREEN:
            value = _diag_value(cell, green_orientation)
            if value in line_values:
                used.add(value)
    return used


def _place_supported_pair(
    shape: tuple[int, int],
    green_orientation: str,
    support: np.ndarray,
    line_values: set[int],
    target_new_value: int,
    blocked: set[tuple[int, int]],
    existing_pluses: list[set[tuple[int, int]]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    pair_orientation = _opposite_orientation(green_orientation)
    candidates: list[tuple[int, int]] = []
    for r in range(1, shape[0] - 1):
        for c in range(1, shape[1] - 1):
            center = (r, c)
            if _can_place_plus(center, shape, blocked, existing_pluses):
                candidates.append(center)

    by_value: dict[int, list[tuple[int, int]]] = {}
    for center in candidates:
        by_value.setdefault(_diag_value(center, pair_orientation), []).append(center)

    values = list(by_value)
    random.shuffle(values)
    for value in values:
        cells = _shuffled(by_value[value])
        for a, b in combinations(cells, 2):
            interior = _interior_diagonal(a, b)
            if not interior:
                continue
            if any(cell in existing for existing in existing_pluses for cell in interior):
                continue
            if any(int(support[r, c]) not in (BLACK, GREEN) for r, c in interior):
                continue
            if not any(int(support[r, c]) == BLACK for r, c in interior):
                continue
            used_values = _pair_support_values(support, green_orientation, line_values, interior)
            if target_new_value not in used_values:
                continue
            return a, b
    return None


def _place_black_pair(
    shape: tuple[int, int],
    green_orientation: str,
    support: np.ndarray,
    blocked: set[tuple[int, int]],
    existing_pluses: list[set[tuple[int, int]]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    pair_orientation = _opposite_orientation(green_orientation)
    candidates = []
    for r in range(1, shape[0] - 1):
        for c in range(1, shape[1] - 1):
            center = (r, c)
            if _can_place_plus(center, shape, blocked, existing_pluses):
                candidates.append(center)

    by_value: dict[int, list[tuple[int, int]]] = {}
    for center in candidates:
        by_value.setdefault(_diag_value(center, pair_orientation), []).append(center)

    values = list(by_value)
    random.shuffle(values)
    for value in values:
        cells = _shuffled(by_value[value])
        for a, b in combinations(cells, 2):
            interior = _interior_diagonal(a, b)
            if not interior:
                continue
            if any(cell in existing for existing in existing_pluses for cell in interior):
                continue
            if all(int(support[r, c]) == BLACK for r, c in interior):
                return a, b
    return None


def _place_isolated_plus(
    shape: tuple[int, int],
    orientation: str,
    used_values: set[int],
    blocked: set[tuple[int, int]],
    existing_pluses: list[set[tuple[int, int]]],
) -> tuple[int, int] | None:
    candidates = []
    for r in range(1, shape[0] - 1):
        for c in range(1, shape[1] - 1):
            center = (r, c)
            if _diag_value(center, orientation) in used_values:
                continue
            if _can_place_plus(center, shape, blocked, existing_pluses):
                candidates.append(center)
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _render_input(
    shape: tuple[int, int],
    orientation: str,
    base_value: int,
    guides: list[set[tuple[int, int]]],
    centers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in _line_cells(shape, orientation, base_value):
        grid[r, c] = GREEN
    for guide in guides:
        for r, c in guide:
            grid[r, c] = GREY
    for center in centers:
        for r, c in _plus_cells(center):
            grid[r, c] = BLUE
    return grid


def _semantic_summary(grid: np.ndarray) -> dict:
    base, orientation, base_value, line_values, centers, tips = _build_construction(grid)
    red_cells, participants, connected_pairs = _solve_pairs(base, orientation, line_values, centers)

    off_line_clear_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for a, b in combinations(centers, 2):
        interior = _interior_diagonal(a, b)
        if interior and all(int(base[r, c]) == BLACK for r, c in interior):
            off_line_clear_pairs.append((a, b))

    active_new_values: set[int] = set()
    for a, b in connected_pairs:
        interior = _interior_diagonal(a, b)
        used = _pair_support_values(base, orientation, line_values, interior)
        active_new_values |= {value for value in used if value != base_value}

    return {
        "orientation": orientation,
        "base_value": base_value,
        "line_values": line_values,
        "centers": centers,
        "tips": tips,
        "red_cells": red_cells,
        "participants": participants,
        "connected_pairs": connected_pairs,
        "active_new_values": active_new_values,
        "off_line_clear_pairs": off_line_clear_pairs,
        "output": main(grid),
    }


def generate_input() -> np.ndarray:
    branch_choices = [
        {"guides": 2, "active": 1, "extra_isolated": False},
        {"guides": 3, "active": 1, "extra_isolated": True},
        {"guides": 3, "active": 2, "extra_isolated": False},
    ]

    for _attempt in range(600):
        h = random.randint(16, 22)
        w = random.randint(16, 22)
        shape = (h, w)
        orientation = random.choice([DIAG_DOWN, DIAG_UP])
        base_value = _choose_base_value(shape, orientation)
        branch = random.choice(branch_choices)

        deltas = random.sample([-3, -2, 2, 3], branch["guides"])

        original_cells = set(_line_cells(shape, orientation, base_value))
        blocked = set(original_cells)
        guides: list[set[tuple[int, int]]] = []
        plus_centers: list[tuple[int, int]] = []
        plus_shapes: list[set[tuple[int, int]]] = []

        failed = False
        new_values: list[int] = []
        for delta_offset in _shuffled(list(deltas)):
            new_value = base_value + delta_offset
            placed = _place_inactive_guide(shape, orientation, new_value, delta_offset, blocked)
            if placed is None:
                failed = True
                break
            guide, _tip = placed
            guides.append(guide)
            blocked |= guide
            new_values.append(new_value)
        if failed:
            continue

        support, line_values = _support_grid(shape, orientation, base_value, guides)
        active_values = random.sample(new_values, branch["active"])

        for target_new_value in _shuffled(active_values):
            pair = _place_supported_pair(
                shape,
                orientation,
                support,
                line_values,
                target_new_value,
                blocked,
                plus_shapes,
            )
            if pair is None:
                failed = True
                break
            a, b = pair
            plus_centers.extend([a, b])
            plus_shapes.extend([_plus_cells(a), _plus_cells(b)])
            blocked |= _plus_cells(a)
            blocked |= _plus_cells(b)
        if failed:
            continue

        distractor = _place_black_pair(shape, orientation, support, blocked, plus_shapes)
        if distractor is None:
            continue
        a, b = distractor
        plus_centers.extend([a, b])
        plus_shapes.extend([_plus_cells(a), _plus_cells(b)])
        blocked |= _plus_cells(a)
        blocked |= _plus_cells(b)

        if branch["extra_isolated"]:
            pair_orientation = _opposite_orientation(orientation)
            used_values = {_diag_value(center, pair_orientation) for center in plus_centers}
            extra = _place_isolated_plus(shape, pair_orientation, used_values, blocked, plus_shapes)
            if extra is None:
                continue
            plus_centers.append(extra)
            plus_shapes.append(_plus_cells(extra))
            blocked |= _plus_cells(extra)

        grid = _render_input(shape, orientation, base_value, guides, plus_centers)
        summary = _semantic_summary(grid)

        if not (4 <= len(summary["centers"]) <= 6):
            continue
        if len(summary["active_new_values"]) < 1:
            continue
        if not summary["red_cells"]:
            continue
        if not summary["participants"]:
            continue
        if len(summary["participants"]) >= len(summary["centers"]):
            continue
        if not summary["off_line_clear_pairs"]:
            continue
        if len(summary["connected_pairs"]) < branch["active"]:
            continue

        output = summary["output"]
        if not np.any(output == RED):
            continue
        if not np.any(output == ORANGE):
            continue
        if not np.any(output == BLUE):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
