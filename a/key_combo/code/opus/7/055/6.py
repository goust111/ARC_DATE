from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in _neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        next_cells = [
            (nr, nc)
            for nr, nc in _neighbors4(*cur)
            if (nr, nc) in pts and (nr, nc) != prev
        ]
        assert len(next_cells) == 1, (cur, next_cells)
        prev, cur = cur, next_cells[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _cycle_turns(coords: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    path = _trace_cycle(coords)
    turn_signs: dict[tuple[int, int], int] = {}
    counts: Counter[int] = Counter()
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            continue
        sign = 1 if cross > 0 else -1
        turn_signs[(r, c)] = sign
        counts[sign] += 1

    assert counts[1] > 0 and counts[-1] > 0
    convex_sign = 1 if counts[1] > counts[-1] else -1
    convex = {cell for cell, sign in turn_signs.items() if sign == convex_sign}
    concave = set(turn_signs) - convex
    return convex, concave


def _bar_info(grid: np.ndarray) -> list[tuple[int, int, int]]:
    cols = np.flatnonzero(np.any(grid == GREY, axis=0))
    bars = []
    for c in cols:
        rows = np.flatnonzero(grid[:, c] == GREY)
        if rows.size == 0:
            continue
        r0 = int(rows[0])
        r1 = int(rows[-1])
        height = int(rows.size)
        assert height == r1 - r0 + 1
        assert np.all(grid[r0 : r1 + 1, c] == GREY)
        bars.append((height, r0, int(c)))
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Use the tallest and shortest grey bars to frame the grid, then classify only the enclosed green loops.

    Input structure:
    - Black background contains 5-7 grey bottom-aligned vertical bars with distinct heights.
    - Green pixels form one or more one-cell-wide 4-connected closed orthogonal loops.
    - The tallest and shortest bar heights are unique and determine a rectangle.

    Rule:
    - Find the unique tallest grey bar and the unique shortest grey bar.
    - Recolor the tallest bar blue and the shortest bar red; erase every other grey bar.
    - The rectangle uses the two selected bar columns as left/right edges, the tallest bar top as its top edge, and the bottom row as its bottom edge.
    - Any green loop whose bounding box lies strictly inside that rectangle is traced: convex bends become yellow, concave bends become red, and straight pixels stay green.
    - Any green loop not strictly inside the rectangle is erased, and the rectangle edges are drawn on top at the end.

    Key insight:
    - The bars do not directly mark the answer; their relative extreme heights implicitly define a frame that gates which loops deserve the bend analysis.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bars = _bar_info(g)
    heights = [height for height, _r0, _c in bars]
    tallest_height = max(heights)
    shortest_height = min(heights)
    tallest_matches = [bar for bar in bars if bar[0] == tallest_height]
    shortest_matches = [bar for bar in bars if bar[0] == shortest_height]
    assert len(tallest_matches) == 1 and len(shortest_matches) == 1
    tallest = tallest_matches[0]
    shortest = shortest_matches[0]

    tall_height, tall_r0, tall_c = tallest
    short_height, short_r0, short_c = shortest
    top = tall_r0
    bottom = h - 1
    left = min(tall_c, short_c)
    right = max(tall_c, short_c)

    out = np.full_like(g, BLACK)
    selected_mask = np.zeros_like(g, dtype=bool)
    out[top : bottom + 1, tall_c] = BLUE
    selected_mask[top : bottom + 1, tall_c] = True
    out[short_r0 : bottom + 1, short_c] = RED
    selected_mask[short_r0 : bottom + 1, short_c] = True

    for coords in _components(g == GREEN):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 > top and r1 < bottom and c0 > left and c1 < right:
            convex, concave = _cycle_turns(coords)
            for r, c in map(tuple, coords):
                out[r, c] = GREEN
            for r, c in convex:
                out[r, c] = YELLOW
            for r, c in concave:
                out[r, c] = RED

    for r in range(top, bottom + 1):
        if not selected_mask[r, left]:
            out[r, left] = BLUE
        if not selected_mask[r, right]:
            out[r, right] = BLUE
    for c in range(left, right + 1):
        if not selected_mask[top, c]:
            out[top, c] = RED
        if not selected_mask[bottom, c]:
            out[bottom, c] = RED

    out[top, left] = BLUE
    out[top, right] = BLUE
    out[bottom, left] = BLUE
    out[bottom, right] = BLUE
    return out


def _draw_segment(path: list[tuple[int, int]], a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[0] == cells[-1]:
        cells = cells[:-1]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _rotate_coords(coords: set[tuple[int, int]], turns: int) -> set[tuple[int, int]]:
    pts = set(coords)
    turns %= 4
    for _ in range(turns):
        height = max(r for r, _ in pts) + 1
        pts = {(c, height - 1 - r) for r, c in pts}
    return pts


def _flip_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    width = max(c for _r, c in coords) + 1
    return {(r, width - 1 - c) for r, c in coords}


def _normalize_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    arr = np.array(sorted((r - r0, c - c0) for r, c in coords), dtype=int)
    return arr


def _make_cycle_patch(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 4 or max_w < 5:
        return None
    for _ in range(80):
        h = random.randint(4, max_h)
        w = random.randint(5, max_w)
        notch_left = random.randint(1, w - 4)
        notch_right = random.randint(notch_left + 2, w - 2)
        depth = random.randint(1, h - 3)
        vertices = [
            (0, 0),
            (0, notch_left),
            (depth, notch_left),
            (depth, notch_right),
            (0, notch_right),
            (0, w - 1),
            (h - 1, w - 1),
            (h - 1, 0),
            (0, 0),
        ]

        path: list[tuple[int, int]] = []
        for a, b in zip(vertices, vertices[1:]):
            _draw_segment(path, a, b)
        cells = set(path)
        cells = _rotate_coords(cells, random.randint(0, 3))
        if random.random() < 0.5:
            cells = _flip_coords(cells)
        arr = _normalize_coords(cells)
        ph = int(arr[:, 0].max()) + 1
        pw = int(arr[:, 1].max()) + 1
        if ph > max_h or pw > max_w:
            continue
        try:
            convex, concave = _cycle_turns(arr)
        except AssertionError:
            continue
        if len(convex) == 0 or len(concave) == 0 or len(arr) < 8:
            continue
        return arr
    return None


def _strictly_inside(box: tuple[int, int, int, int], frame: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    top, left, bottom, right = frame
    return r0 > top and r1 < bottom and c0 > left and c1 < right


def _intersects_frame(box: tuple[int, int, int, int], frame: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    top, left, bottom, right = frame
    return not (r1 < top or r0 > bottom or c1 < left or c0 > right)


def _can_place(coords: np.ndarray, r0: int, c0: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in map(tuple, coords):
        rr = r0 + r
        cc = c0 + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        for dr, dc in DIRS4 + [(0, 0)]:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _place_loop(grid: np.ndarray, occupied: np.ndarray, coords: np.ndarray, positions: list[tuple[int, int]]) -> bool:
    if not positions:
        return False
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _can_place(coords, r0, c0, occupied):
            continue
        for r, c in map(tuple, coords):
            rr = r0 + r
            cc = c0 + c
            grid[rr, cc] = GREEN
            occupied[rr, cc] = True
        return True
    return False


def _candidate_positions(
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    status: str,
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch_shape
    top, left, bottom, right = frame
    positions = []
    for r0 in range(0, h - ph + 1):
        for c0 in range(0, w - pw + 1):
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            inside = _strictly_inside(box, frame)
            if status == "inside" and inside:
                positions.append((r0, c0))
            elif status == "outside_any" and not inside:
                positions.append((r0, c0))
            elif status == "outside_crossing" and (not inside) and _intersects_frame(box, frame):
                positions.append((r0, c0))
    return positions


def _sample_bar_columns(width: int, count: int) -> list[int] | None:
    layouts = []
    for start in range(1, 4):
        cols = [start + 3 * i for i in range(count)]
        if cols[-1] < width - 1:
            layouts.append(cols)
    if not layouts:
        return None
    return random.choice(layouts)


def _build_input() -> np.ndarray | None:
    h = random.randint(20, 24)
    w = random.randint(20, 24)

    feasible_counts = [count for count in (5, 6, 7) if count <= 6 or w >= 21 + 3 * (count - 6)]
    if not feasible_counts:
        return None
    weights = [4 if count == 5 else 3 if count == 6 else 1 for count in feasible_counts]
    bar_count = random.choices(feasible_counts, weights=weights)[0]
    cols = _sample_bar_columns(w, bar_count)
    if cols is None:
        return None

    tall_i, short_i = (0, bar_count - 1) if random.random() < 0.5 else (bar_count - 1, 0)

    max_tall = min(15, h - 4)
    if max_tall < 10:
        return None
    tall_height = random.randint(10, max_tall)
    short_height = random.randint(3, 5)
    remaining_pool = [x for x in range(4, min(9, tall_height - 1)) if x not in {tall_height, short_height}]
    if len(remaining_pool) < bar_count - 2:
        return None
    middle_heights = random.sample(remaining_pool, bar_count - 2)

    heights = [0] * bar_count
    heights[tall_i] = tall_height
    heights[short_i] = short_height
    for idx, height in zip([i for i in range(bar_count) if i not in {tall_i, short_i}], sorted(middle_heights)):
        heights[idx] = height

    top = h - tall_height
    left = min(cols[tall_i], cols[short_i])
    right = max(cols[tall_i], cols[short_i])
    bottom = h - 1
    frame = (top, left, bottom, right)
    interior_h = bottom - top - 1
    interior_w = right - left - 1
    if interior_h < 5 or interior_w < 6:
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for col, height in zip(cols, heights):
        r0 = h - height
        grid[r0:h, col] = GREY
        occupied[r0:h, col] = True

    loop_count = random.choices([2, 3, 4], weights=[4, 3, 1])[0]
    inside_count = 1 if loop_count == 2 or random.random() < 0.8 else 2
    inside_count = min(inside_count, loop_count - 1)
    outside_statuses = [random.choice(["outside_any", "outside_crossing"]) for _ in range(loop_count - inside_count)]
    statuses = ["inside"] * inside_count + outside_statuses

    middle_max = max((heights[i] for i in range(bar_count) if i not in {tall_i, short_i}), default=0)
    safe_inside_bottom = min(bottom - 1, h - middle_max - 2)
    for status in statuses:
        if status == "inside":
            max_ph = min(6, interior_h)
            max_pw = min(7, interior_w)
        else:
            max_ph = min(8, h - 2)
            max_pw = min(9, w - 2)
        patch = _make_cycle_patch(max_ph, max_pw)
        if patch is None:
            return None
        ph = int(patch[:, 0].max()) + 1
        pw = int(patch[:, 1].max()) + 1
        positions = _candidate_positions((h, w), (ph, pw), frame, status)
        if status == "inside":
            positions = [(r0, c0) for r0, c0 in positions if r0 + ph - 1 <= safe_inside_bottom]
        if status == "outside_crossing" and not positions:
            positions = _candidate_positions((h, w), (ph, pw), frame, "outside_any")
        if not _place_loop(grid, occupied, patch, positions):
            if status == "outside_crossing":
                fallback = _candidate_positions((h, w), (ph, pw), frame, "outside_any")
                if _place_loop(grid, occupied, patch, fallback):
                    continue
            return None

    if not _valid_input(grid):
        return None
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False

    bars = _bar_info(grid)
    if not (5 <= len(bars) <= 7):
        return False
    heights = [bar[0] for bar in bars]
    if len(set(heights)) != len(heights):
        return False
    cols = sorted(bar[2] for bar in bars)
    if any(b - a < 3 for a, b in zip(cols, cols[1:])):
        return False

    tallest = max(bars, key=lambda x: x[0])
    shortest = min(bars, key=lambda x: x[0])
    if sum(bar[0] == tallest[0] for bar in bars) != 1:
        return False
    if sum(bar[0] == shortest[0] for bar in bars) != 1:
        return False
    frame = (tallest[1], min(tallest[2], shortest[2]), h - 1, max(tallest[2], shortest[2]))

    loops = _components(grid == GREEN)
    if not (2 <= len(loops) <= 4):
        return False

    inside = 0
    outside = 0
    for coords in loops:
        try:
            convex, concave = _cycle_turns(coords)
        except AssertionError:
            return False
        if len(convex) == 0 or len(concave) == 0:
            return False
        box = _bbox(coords)
        if _strictly_inside(box, frame):
            inside += 1
        else:
            outside += 1
    if inside == 0 or outside == 0:
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
