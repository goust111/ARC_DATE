import random

import numpy as np


NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGH8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
]

COLOR_VALUES = tuple(range(10))
GENERATION_ATTEMPTS = 400
SEGMENT_ATTEMPTS = 120
OBJECT_ATTEMPTS = 80
HORIZONTAL_SPAN = (2, 4)
VERTICAL_SPAN = (2, 4)
CANVAS_LIMIT = 30


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, "expected a unique dominant background color"
    return int(winners[0])


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    neighbors = NEIGH4 if connectivity == 4 else NEIGH8
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []

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
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(comp)
    return result


def _reading_order_points(grid: np.ndarray, color: int) -> list[tuple[int, int]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    if not (4 <= len(coords) <= 8):
        return None
    comps = _components(grid == color, connectivity=8)
    if len(comps) != len(coords) or any(len(comp) != 1 for comp in comps):
        return None
    return sorted(coords)


def _neighbors_in_grid(point: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = point
    result = []
    for dr, dc in NEIGH4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc))
    return result


def _identify_scene(grid: np.ndarray, bg: int) -> tuple[int, int, list[tuple[int, int]]]:
    h, w = grid.shape
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []

    for marker_color in colors:
        markers = _reading_order_points(grid, marker_color)
        if markers is None:
            continue

        first = markers[0]
        adjacent_colors = {
            int(grid[nr, nc])
            for nr, nc in _neighbors_in_grid(first, h, w)
            if int(grid[nr, nc]) not in (bg, marker_color)
        }
        if len(adjacent_colors) != 1:
            continue

        track_color = adjacent_colors.pop()
        comps = _components(grid == track_color, connectivity=4)
        if len(comps) != 1 or not (2 <= len(comps[0]) <= 4):
            continue
        if not any(abs(r - first[0]) + abs(c - first[1]) == 1 for r, c in comps[0]):
            continue
        candidates.append((marker_color, track_color, markers))

    assert len(candidates) == 1, "expected one marker color and one adjacent color object"
    return candidates[0]


def _rail_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    cells: set[tuple[int, int]] = set()

    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1, step):
            cells.add((r0, c))
        return cells

    if c0 == c1:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1, step):
            cells.add((r, c0))
        return cells

    step_h = 1 if c1 > c0 else -1
    step_v = 1 if r1 > r0 else -1

    for c in range(c0 + step_h, c1 + step_h, step_h):
        cells.add((r0, c))
    for r in range(r0 + step_v, r1, step_v):
        cells.add((r, c1))

    inner_row = r0 + step_v
    inner_col = c1 - step_h
    for c in range(c0, c1 + step_h, step_h):
        cells.add((inner_row, c))
    for r in range(r0, r1 + step_v, step_v):
        cells.add((r, inner_col))

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Lay a colored rail through the reading-order marker sequence, widening only at turning segments.

    Input structure:
    - The background is the dominant color.
    - One foreground color appears as 4-8 isolated single-pixel markers.
    - These markers are read in row-major order (top to bottom, then left to right).
    - A small 2-4 pixel object of another color touches the first marker in the 4-neighborhood.

    Rule:
    - Identify the singleton-marker color and sort its pixels by reading order.
    - Connect each consecutive marker pair with a path that goes horizontally first, then vertically.
    - Paint that path in the small object's color: straight segments stay 1 pixel wide, while turning segments gain one parallel inner-side rail to become 2 pixels wide.
    - Keep the original markers and the original small object unchanged.

    Key insight:
    The singleton markers are not separate objects to copy; they are ordered waypoints that define a horizontal-then-vertical route whose width depends on whether each hop turns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _majority_color(g)
    marker_color, track_color, markers = _identify_scene(g, bg)

    out = g.copy()
    for start, end in zip(markers, markers[1:]):
        for r, c in _rail_cells(start, end):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if int(out[r, c]) == marker_color:
                continue
            if int(out[r, c]) in (bg, track_color):
                out[r, c] = track_color
    return out


def _translate(points: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in points}


def _neighbor_set(points: set[tuple[int, int]], connectivity: int) -> set[tuple[int, int]]:
    if not points:
        return set()
    offsets = NEIGH4 if connectivity == 4 else NEIGH8
    result = set(points)
    for r, c in points:
        for dr, dc in offsets:
            result.add((r + dr, c + dc))
    return result


def _sample_segment_types(count: int) -> list[str]:
    if count % 2 == 0:
        n_turns = count // 2
    else:
        n_turns = random.choice([count // 2, count // 2 + 1])
    turn_slots = set(random.sample(range(count), n_turns))
    kinds = []
    for idx in range(count):
        if idx in turn_slots:
            kinds.append(random.choice(["turn_left", "turn_right"]))
        else:
            kinds.append(random.choice(["straight_h", "straight_v"]))
    assert any(kind.startswith("turn") for kind in kinds)
    assert any(kind.startswith("straight") for kind in kinds)
    return kinds


def _sample_delta(kind: str) -> tuple[int, int]:
    if kind == "straight_h":
        return 0, random.randint(*HORIZONTAL_SPAN)
    if kind == "straight_v":
        return random.randint(*VERTICAL_SPAN), 0
    if kind == "turn_right":
        return random.randint(*VERTICAL_SPAN), random.randint(*HORIZONTAL_SPAN)
    if kind == "turn_left":
        return random.randint(*VERTICAL_SPAN), -random.randint(*HORIZONTAL_SPAN)
    raise ValueError(kind)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _track_clear(
    cells: set[tuple[int, int]],
    previous_markers: list[tuple[int, int]],
    previous_track: set[tuple[int, int]],
) -> bool:
    if cells & previous_track:
        return False
    buffered_track = _neighbor_set(previous_track, connectivity=4)
    if cells & buffered_track:
        return False
    for marker in previous_markers[:-1]:
        for cell in cells:
            if abs(cell[0] - marker[0]) + abs(cell[1] - marker[1]) <= 1:
                return False
    return True


def _build_route() -> tuple[list[tuple[int, int]], set[tuple[int, int]], list[str]] | None:
    n_markers = random.randint(4, 8)
    kinds = _sample_segment_types(n_markers - 1)
    markers = [(0, 0)]
    track: set[tuple[int, int]] = set()

    for kind in kinds:
        placed = False
        for _ in range(SEGMENT_ATTEMPTS):
            dr, dc = _sample_delta(kind)
            current = markers[-1]
            nxt = (current[0] + dr, current[1] + dc)
            if any(_chebyshev(nxt, marker) <= 1 for marker in markers):
                continue
            cells = _rail_cells(current, nxt)
            if not cells:
                continue
            if any(cell in markers for cell in cells):
                continue
            if not _track_clear(cells, markers, track):
                continue
            markers.append(nxt)
            track |= cells
            placed = True
            break
        if not placed:
            return None

    return markers, track, kinds


def _grow_object(
    first_marker: tuple[int, int],
    other_markers: set[tuple[int, int]],
    track: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    track_forbidden = _neighbor_set(track, connectivity=4)
    other_marker_forbidden = _neighbor_set(other_markers, connectivity=4)
    directions = NEIGH4[:]
    random.shuffle(directions)

    for dr, dc in directions:
        anchor = (first_marker[0] + dr, first_marker[1] + dc)
        if anchor in track_forbidden or anchor in other_markers or anchor in other_marker_forbidden:
            continue
        for _ in range(OBJECT_ATTEMPTS):
            target_size = random.randint(2, 4)
            cells = {anchor}
            while len(cells) < target_size:
                frontier: list[tuple[tuple[int, int], int]] = []
                for r, c in cells:
                    for sdr, sdc in NEIGH4:
                        cand = (r + sdr, c + sdc)
                        if cand in cells or cand in track_forbidden or cand in other_markers or cand in other_marker_forbidden:
                            continue
                        if max(abs(cand[0] - first_marker[0]), abs(cand[1] - first_marker[1])) > 3:
                            continue
                        score = (cand[0] - first_marker[0]) * dr + (cand[1] - first_marker[1]) * dc
                        frontier.append((cand, score))
                if not frontier:
                    break
                best_score = max(score for _, score in frontier)
                best = [cand for cand, score in frontier if score == best_score]
                cells.add(random.choice(best))
            if len(cells) == target_size:
                return cells
    return None


def _bbox(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _split_extra(extra: int) -> tuple[int, int]:
    if extra <= 0:
        return 0, 0
    low = 2 if extra >= 2 else 1
    total = random.randint(low, min(8, extra))
    first = random.randint(0, total)
    return first, total - first


def _make_scene() -> np.ndarray | None:
    bg, marker_color, track_color = random.sample(COLOR_VALUES, 3)
    built = _build_route()
    if built is None:
        return None
    markers, track, _kinds = built

    obj = _grow_object(markers[0], set(markers[1:]), track)
    if obj is None:
        return None

    all_points = set(markers) | track | obj
    min_r, min_c, max_r, max_c = _bbox(all_points)
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1
    if box_h > CANVAS_LIMIT or box_w > CANVAS_LIMIT:
        return None

    top, bottom = _split_extra(CANVAS_LIMIT - box_h)
    left, right = _split_extra(CANVAS_LIMIT - box_w)
    height = box_h + top + bottom
    width = box_w + left + right
    if height > CANVAS_LIMIT or width > CANVAS_LIMIT:
        return None

    shift_r = top - min_r
    shift_c = left - min_c
    shifted_markers = _translate(set(markers), shift_r, shift_c)
    shifted_track = _translate(track, shift_r, shift_c)
    shifted_obj = _translate(obj, shift_r, shift_c)

    grid = np.full((height, width), bg, dtype=int)
    for r, c in shifted_markers:
        grid[r, c] = marker_color
    for r, c in shifted_obj:
        grid[r, c] = track_color

    expected = grid.copy()
    for r, c in shifted_track:
        if expected[r, c] != marker_color:
            expected[r, c] = track_color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _make_scene()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATION_ATTEMPTS} attempts")
