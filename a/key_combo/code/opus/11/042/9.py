from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

import numpy as np


YELLOW = 4
WINNER_COLORS = (1, 2, 3)
GEN_ATTEMPTS = 600
GRID_CHOICES = (15, 17, 19)
SMALL_INTERIORS = ((2, 2), (2, 3), (3, 2))
WIDE_INTERIORS = ((3, 4), (2, 4), (3, 5))
TALL_INTERIORS = ((4, 3), (4, 2), (5, 3))
OVERLAP_SHAPES = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
OVERLAP_MODES = ("nw", "ne", "sw", "se")
MARKER_PATTERNS = ((2, 1), (3, 1), (2, 1, 1))


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _interior_from_rect(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _rect_from_interior(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - 1, c0 - 1, r1 + 1, c1 + 1


def _corners(rect: tuple[int, int, int, int]) -> frozenset[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return frozenset(((r0, c0), (r0, c1), (r1, c0), (r1, c1)))


def _box_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _box_overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _box_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _full_rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _rect_center_distance(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    grid_center = (shape[0] // 2, shape[1] // 2)
    rect_center = (_round_half_up((rect[0] + rect[2]) / 2.0), _round_half_up((rect[1] + rect[3]) / 2.0))
    return abs(rect_center[0] - grid_center[0]) + abs(rect_center[1] - grid_center[1])


def _candidate_rectangles(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], frozenset[tuple[int, int]]]]:
    yellows = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    rows = sorted({r for r, _c in yellows})
    cols = sorted({c for _r, c in yellows})
    candidates: list[tuple[tuple[int, int, int, int], frozenset[tuple[int, int]]]] = []
    for r0, r1 in combinations(rows, 2):
        if r1 - r0 < 3:
            continue
        for c0, c1 in combinations(cols, 2):
            if c1 - c0 < 3:
                continue
            rect_corners = frozenset(((r0, c0), (r0, c1), (r1, c0), (r1, c1)))
            if rect_corners <= yellows:
                candidates.append(((r0, c0, r1, c1), rect_corners))
    return candidates


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    yellows = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    assert len(yellows) == 12, "expected exactly three groups of four yellow corners"

    candidates = _candidate_rectangles(grid)
    by_corner: dict[tuple[int, int], list[int]] = {corner: [] for corner in yellows}
    for idx, (_rect, corners) in enumerate(candidates):
        for corner in corners:
            by_corner[corner].append(idx)

    solutions: list[list[int]] = []

    def search(remaining: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(chosen.copy())
            return
        if len(chosen) >= 4 or len(remaining) % 4:
            return
        pivot = min(remaining)
        for idx in by_corner[pivot]:
            corners = candidates[idx][1]
            if corners <= remaining:
                chosen.append(idx)
                search(remaining - set(corners), chosen)
                chosen.pop()

    search(set(yellows), [])
    assert len(solutions) == 1 and len(solutions[0]) == 3, "yellow corners must define a unique set of three rectangles"

    rects = [candidates[idx][0] for idx in solutions[0]]
    rects.sort()
    return rects


def _vote_color(grid: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    ir0, ic0, ir1, ic1 = _interior_from_rect(rect)
    interior = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
    counts = Counter(int(value) for value in interior.flatten() if value not in (0, YELLOW))
    assert counts, "each rectangle must contain internal vote markers"
    ranking = counts.most_common()
    assert len(ranking) == 1 or ranking[0][1] > ranking[1][1], "rectangle vote must have a unique winner"
    return int(ranking[0][0])


def _render_with_order(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    fill_colors: dict[tuple[int, int, int, int], int],
    order: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        for r, c in _corners(rect):
            out[r, c] = YELLOW
    for rect in order:
        ir0, ic0, ir1, ic1 = _interior_from_rect(rect)
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = int(fill_colors[rect])
    for rect in rects:
        for r, c in _corners(rect):
            out[r, c] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each yellow-corner rectangle with the color that wins its internal marker vote, resolving overlaps by center distance.

    Input structure:
    - A black background contains exactly three hidden rectangles whose four corners are marked by yellow pixels.
    - Each rectangle is defined by four yellow pixels sharing two rows and two columns, and its open interior may overlap other rectangles.
    - Inside each rectangle's interior are a few blue, red, or green marker pixels, while extra same-color distractor markers may appear outside all rectangles.

    Rule:
    - Group the yellow pixels into the unique three rectangles they define, then inspect only the strict interior of each rectangle.
    - For each rectangle, count the non-black, non-yellow marker colors in that interior; the unique majority color is that rectangle's fill color.
    - Compute each rectangle center by averaging its corner rows and columns and rounding to the nearest cell, then sort rectangles by Manhattan distance from the grid center from farthest to nearest.
    - Fill rectangle interiors in that order so later, nearer rectangles overwrite earlier fills in overlaps; keep yellow corners unchanged and erase all non-corner markers outside the final fills.

    Key insight:
    - The colored markers are votes local to each rectangle, but overlap resolution is global: the final color of shared cells depends on which voted rectangle lies closer to the grid center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rects = _find_rectangles(g)
    fill_colors = {rect: _vote_color(g, rect) for rect in rects}
    distances = {rect: _rect_center_distance(rect, g.shape) for rect in rects}
    assert len(set(distances.values())) == len(rects), "rectangle center distances must be unique"
    ordered = sorted(rects, key=lambda rect: distances[rect], reverse=True)
    return _render_with_order(g.shape, rects, fill_colors, ordered)


def _random_box(shape: tuple[int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = size
    max_r0 = shape[0] - h - 1
    max_c0 = shape[1] - w - 1
    r0 = random.randint(1, max_r0)
    c0 = random.randint(1, max_c0)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _central_box(shape: tuple[int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = size
    center_r = shape[0] // 2 + random.randint(-1, 1)
    center_c = shape[1] // 2 + random.randint(-1, 1)
    r0 = min(max(1, center_r - h // 2), shape[0] - h - 1)
    c0 = min(max(1, center_c - w // 2), shape[1] - w - 1)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _overlap_box(
    ref: tuple[int, int, int, int],
    size: tuple[int, int],
    overlap: tuple[int, int],
    mode: str,
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    h, w = size
    oh, ow = overlap
    rr0, rc0, rr1, rc1 = ref
    if mode == "nw":
        r1 = rr0 + oh - 1
        c1 = rc0 + ow - 1
        r0 = r1 - h + 1
        c0 = c1 - w + 1
    elif mode == "ne":
        r1 = rr0 + oh - 1
        c0 = rc1 - ow + 1
        r0 = r1 - h + 1
        c1 = c0 + w - 1
    elif mode == "sw":
        r0 = rr1 - oh + 1
        c1 = rc0 + ow - 1
        r1 = r0 + h - 1
        c0 = c1 - w + 1
    else:
        r0 = rr1 - oh + 1
        c0 = rc1 - ow + 1
        r1 = r0 + h - 1
        c1 = c0 + w - 1
    if not (1 <= r0 <= r1 <= shape[0] - 2 and 1 <= c0 <= c1 <= shape[1] - 2):
        return None
    return (r0, c0, r1, c1)


def _exclusive_cells(index: int, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    own = _box_cells(boxes[index])
    other = set()
    for j, box in enumerate(boxes):
        if j != index:
            other |= _box_cells(box)
    cells = sorted(own - other)
    random.shuffle(cells)
    return cells


def _corners_are_clear(rects: list[tuple[int, int, int, int]], boxes: list[tuple[int, int, int, int]]) -> bool:
    all_corners: set[tuple[int, int]] = set()
    for i, rect in enumerate(rects):
        rect_corners = _corners(rect)
        if len(rect_corners) != 4 or all_corners & rect_corners:
            return False
        all_corners |= rect_corners
        for j, box in enumerate(boxes):
            if i == j:
                continue
            if any(box[0] <= r <= box[2] and box[1] <= c <= box[3] for r, c in rect_corners):
                return False
    return True


def _pick_boxes(shape: tuple[int, int]) -> list[tuple[int, int, int, int]] | None:
    topology = random.choice(("single", "double"))
    if topology == "double":
        sizes = [random.choice(((2, 3), (3, 2))), random.choice(WIDE_INTERIORS), random.choice(TALL_INTERIORS)]
    else:
        sizes = [random.choice(SMALL_INTERIORS), random.choice(WIDE_INTERIORS), random.choice(TALL_INTERIORS)]
    if topology == "double":
        sizes.sort(key=lambda size: size[0] * size[1], reverse=True)
    else:
        random.shuffle(sizes)

    for _ in range(120):
        boxes: list[tuple[int, int, int, int]] = []
        if topology == "double":
            boxes.append(_central_box(shape, sizes[0]))
            mode_pairs = [("nw", "se"), ("ne", "sw"), ("nw", "ne"), ("sw", "se"), ("nw", "sw"), ("ne", "se")]
            random.shuffle(mode_pairs)
            built = False
            for mode_ab, mode_ac in mode_pairs:
                overlap_ab = random.choice([ov for ov in OVERLAP_SHAPES if ov[0] <= sizes[0][0] and ov[1] <= sizes[0][1] and ov[0] <= sizes[1][0] and ov[1] <= sizes[1][1]])
                overlap_ac = random.choice([ov for ov in OVERLAP_SHAPES if ov[0] <= sizes[0][0] and ov[1] <= sizes[0][1] and ov[0] <= sizes[2][0] and ov[1] <= sizes[2][1]])
                box_b = _overlap_box(boxes[0], sizes[1], overlap_ab, mode_ab, shape)
                box_c = _overlap_box(boxes[0], sizes[2], overlap_ac, mode_ac, shape)
                if box_b is None or box_c is None:
                    continue
                if _box_overlap_area(box_b, box_c) != 0:
                    continue
                boxes.extend((box_b, box_c))
                built = True
                break
            if not built:
                continue
        else:
            boxes.append(_random_box(shape, sizes[0]))
            overlap_ab = random.choice([ov for ov in OVERLAP_SHAPES if ov[0] <= sizes[0][0] and ov[1] <= sizes[0][1] and ov[0] <= sizes[1][0] and ov[1] <= sizes[1][1]])
            mode_ab = random.choice(OVERLAP_MODES)
            box_b = _overlap_box(boxes[0], sizes[1], overlap_ab, mode_ab, shape)
            if box_b is None:
                continue
            boxes.append(box_b)
            for _inner in range(120):
                candidate = _random_box(shape, sizes[2])
                if _box_overlap_area(candidate, boxes[0]) or _box_overlap_area(candidate, boxes[1]):
                    continue
                boxes.append(candidate)
                break
            if len(boxes) != 3:
                continue

        pair_overlaps = [_box_overlap_area(boxes[i], boxes[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        if topology == "single":
            if sum(area > 0 for area in pair_overlaps) != 1:
                continue
        else:
            if pair_overlaps[1] == 0 or pair_overlaps[0] == 0 or pair_overlaps[2] != 0:
                continue
        if any(area not in (0, 2, 3, 4, 6) for area in pair_overlaps):
            continue

        rects = [_rect_from_interior(box) for box in boxes]
        if not _corners_are_clear(rects, boxes):
            continue
        if any(len(_exclusive_cells(i, boxes)) < 3 for i in range(3)):
            continue
        distances = [_rect_center_distance(rect, shape) for rect in rects]
        if len(set(distances)) != 3:
            continue
        return boxes
    return None


def _place_internal_markers(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    winners: list[int],
) -> dict[int, list[tuple[int, int, int]]]:
    placements: dict[int, list[tuple[int, int, int]]] = {}
    for idx, winner in enumerate(winners):
        cells = _exclusive_cells(idx, boxes)
        loser_colors = [color for color in WINNER_COLORS if color != winner]
        choices = [pattern for pattern in MARKER_PATTERNS if sum(pattern) <= len(cells)]
        pattern = random.choice(choices)
        colors: list[int]
        if len(pattern) == 2:
            colors = [winner, random.choice(loser_colors)]
        else:
            random.shuffle(loser_colors)
            colors = [winner, loser_colors[0], loser_colors[1]]
        rect_marks: list[tuple[int, int, int]] = []
        offset = 0
        for color, count in zip(colors, pattern):
            for cell in cells[offset : offset + count]:
                rect_marks.append((cell[0], cell[1], color))
                grid[cell] = color
            offset += count
        placements[idx] = rect_marks
    return placements


def _place_distractors(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int]]:
    forbidden = set()
    for rect in rects:
        forbidden |= _full_rect_cells(rect)
    available = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if (r, c) not in forbidden]
    random.shuffle(available)
    count = random.randint(3, 5)
    distractors: list[tuple[int, int, int]] = []
    for r, c in available[:count]:
        color = random.choice(WINNER_COLORS)
        grid[r, c] = color
        distractors.append((r, c, color))
    return distractors


def _validate_generated(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    winners: list[int],
    marker_map: dict[int, list[tuple[int, int, int]]],
    distractors: list[tuple[int, int, int]],
) -> bool:
    rects = [_rect_from_interior(box) for box in boxes]
    found = _find_rectangles(grid)
    if {tuple(rect) for rect in found} != {tuple(rect) for rect in rects}:
        return False

    voted = {_vote_color(grid, rect) for rect in found}
    if voted != set(WINNER_COLORS):
        return False

    for idx, box in enumerate(boxes):
        own = _box_cells(box)
        others = set()
        for j, other_box in enumerate(boxes):
            if j != idx:
                others |= _box_cells(other_box)
        for r, c, _color in marker_map[idx]:
            if (r, c) not in own or (r, c) in others:
                return False
    for r, c, _color in distractors:
        if any(rect[0] <= r <= rect[2] and rect[1] <= c <= rect[3] for rect in rects):
            return False

    distances = {_rect_center_distance(rect, grid.shape) for rect in rects}
    if len(distances) != 3:
        return False

    expected = main(grid)
    if expected.shape != grid.shape:
        return False

    fill_colors = {rect: _vote_color(grid, rect) for rect in rects}
    ordered = sorted(rects, key=lambda rect: _rect_center_distance(rect, grid.shape), reverse=True)
    reversed_order = list(reversed(ordered))
    if np.array_equal(_render_with_order(grid.shape, rects, fill_colors, ordered), _render_with_order(grid.shape, rects, fill_colors, reversed_order)):
        return False

    pair_overlaps = [_box_overlap_area(boxes[i], boxes[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    if not any(area in (2, 3, 4, 6) for area in pair_overlaps):
        return False

    if set(winners) != set(WINNER_COLORS):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shape = (random.choice(GRID_CHOICES), random.choice(GRID_CHOICES))
        boxes = _pick_boxes(shape)
        if boxes is None:
            continue

        grid = np.zeros(shape, dtype=int)
        rects = [_rect_from_interior(box) for box in boxes]
        for rect in rects:
            for r, c in _corners(rect):
                grid[r, c] = YELLOW

        winners = list(WINNER_COLORS)
        random.shuffle(winners)
        marker_map = _place_internal_markers(grid, boxes, winners)
        distractors = _place_distractors(grid, rects)

        try:
            if _validate_generated(grid, boxes, winners, marker_map, distractors):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 9")
