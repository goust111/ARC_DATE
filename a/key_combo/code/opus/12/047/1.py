from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

LAYOUTS = [
    (25, [5, 11, 17]),
    (27, [6, 12, 18]),
    (29, [5, 11, 17, 23]),
    (29, [8, 14, 20]),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        comp = set()
        q = deque([(r, c)])
        seen[r, c] = True
        while q:
            cr, cc = q.popleft()
            comp.add((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bar_bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotated_bar_pixels(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bar_bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = len(coords)
    assert height == 1 or width == 1
    length = max(height, width)
    assert area == length and length % 2 == 1

    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    half = length // 2
    if height == 1:
        return {(cr + d, cc) for d in range(-half, half + 1)}
    return {(cr, cc + d) for d in range(-half, half + 1)}


def _extract_bars(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    bars = _component_sets(grid == GREY)
    for bar in bars:
        r0, c0, r1, c1 = _bar_bbox(bar)
        assert r0 == r1 or c0 == c1
        assert len(bar) % 2 == 1
        if r0 == r1:
            assert bar == {(r0, c) for c in range(c0, c1 + 1)}
        else:
            assert bar == {(r, c0) for r in range(r0, r1 + 1)}
    return bars


def _distances(passable: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    q = deque([start])
    dist = {start: 0}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                q.append(nxt)
    return dist


def _count_simple_paths(
    passable: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    limit: int = 2,
) -> int:
    count = 0
    stack = [(start, {start})]
    while stack:
        node, seen = stack.pop()
        if node == end:
            count += 1
            if count >= limit:
                return count
            continue
        r, c = node
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and nxt not in seen:
                stack.append((nxt, seen | {nxt}))
    return count


def _shortest_path_nodes(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    q = deque([start])
    parent = {start: None}
    while q:
        node = q.popleft()
        if node == end:
            break
        r, c = node
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and nxt not in parent:
                parent[nxt] = node
                q.append(nxt)
    assert end in parent
    path = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _track_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == TEAL) | (grid == MAROON)


def _mark_forbidden(
    shape: tuple[int, int], obstacle_pixels: set[tuple[int, int]]
) -> np.ndarray:
    forbidden = np.zeros(shape, dtype=bool)
    for r, c in obstacle_pixels:
        if _in_bounds(shape, r, c):
            forbidden[r, c] = True
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(shape, nr, nc):
                forbidden[nr, nc] = True
    return forbidden


def _forbidden_layers(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[set[tuple[int, int]]], list[tuple[int, int]]]:
    bars = _extract_bars(grid)
    oranges = [tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)]

    orange_layers = [_mark_forbidden(grid.shape, {pix}) for pix in oranges]
    bar_layers = [_mark_forbidden(grid.shape, _rotated_bar_pixels(bar)) for bar in bars]

    total = np.zeros(grid.shape, dtype=bool)
    for layer in orange_layers + bar_layers:
        total |= layer
    return total, orange_layers, bar_layers, bars, oranges


def _main_logic(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    endpoints = [tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)]
    assert len(endpoints) == 2

    forbidden, _orange_layers, _bar_layers, _bars, _oranges = _forbidden_layers(g)
    track = _track_mask(g)
    assert all(not forbidden[r, c] for r, c in endpoints)
    passable = track & ~forbidden
    assert _count_simple_paths(passable, endpoints[0], endpoints[1], limit=2) == 1

    path = _shortest_path_nodes(passable, endpoints[0], endpoints[1])
    out = g.copy()
    out[out == GREY] = BLACK
    for r, c in path:
        if out[r, c] == TEAL:
            out[r, c] = MAROON
    return out, path, passable


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each gray bar into its true obstacle position, then recolor the sole safe route.

    Input structure:
    - The canvas is a black background containing one teal 4-connected track with
      exactly two maroon endpoints.
    - Orange singleton blockers sit near the track.
    - Grey one-cell-thick odd bars of length 3-7 sit off the track; each bar is
      horizontal or vertical and should be interpreted by its center and length.

    Rule:
    - Clear the original grey bars and rotate each one 90 degrees around its own
      center; the rotated grey pixels define hidden blocker locations.
    - Treat every orange blocker and every rotated grey blocker pixel together
      with their 8-neighborhoods as forbidden for track travel.
    - In the remaining teal/maroon 4-connected subgraph, there is exactly one
      simple path between the two maroon endpoints.
    - Recolor the teal pixels on that unique safe path to maroon; keep the
      endpoints and orange blockers unchanged, and do not redraw the grey bars.

    Key insight:
    - The visible grey bars are decoys: their current orientation is irrelevant,
      and only their mentally rotated positions reveal which apparent track loops
      are actually blocked.
    """
    return _main_logic(grid)[0]


def _build_snake(height: int, width: int, lane_rows: list[int]):
    c_left = 2
    c_right = width - 3
    track: set[tuple[int, int]] = set()
    for row in lane_rows:
        for c in range(c_left, c_right + 1):
            track.add((row, c))

    for i in range(len(lane_rows) - 1):
        row0 = lane_rows[i]
        row1 = lane_rows[i + 1]
        col = c_right if i % 2 == 0 else c_left
        for r in range(row0, row1 + 1):
            track.add((r, col))

    start = (lane_rows[0], 0)
    for c in range(0, c_left + 1):
        track.add((lane_rows[0], c))

    if (len(lane_rows) - 1) % 2 == 0:
        end = (lane_rows[-1], width - 1)
        for c in range(c_right, width):
            track.add((lane_rows[-1], c))
    else:
        end = (lane_rows[-1], 0)
        for c in range(0, c_left + 1):
            track.add((lane_rows[-1], c))

    return track, start, end, c_left, c_right


def _module_clearance(height: int, lane_rows: list[int], idx: int, side: str) -> int:
    row = lane_rows[idx]
    if side == "above":
        prev_row = lane_rows[idx - 1] if idx > 0 else -1
        return row - prev_row - 1
    next_row = lane_rows[idx + 1] if idx + 1 < len(lane_rows) else height
    return next_row - row - 1


def _orange_module(row: int, side: str, mid: int):
    sign = -1 if side == "above" else 1
    detour = set()
    left = mid - 1
    right = mid + 1
    detour.add((row + sign, left))
    detour.add((row + sign * 2, left))
    detour.add((row + sign * 3, left))
    detour.add((row + sign * 3, mid))
    detour.add((row + sign * 3, right))
    detour.add((row + sign * 2, right))
    detour.add((row + sign, right))
    obstacle = {(row + sign * 2, mid)}
    return detour, obstacle


def _gray_module(row: int, side: str, mid: int, length: int):
    sign = -1 if side == "above" else 1
    half = length // 2
    detour = set()
    left = mid - 1
    right = mid + 1
    detour.add((row + sign, left))
    detour.add((row + sign * 2, left))
    detour.add((row + sign * 2, mid))
    detour.add((row + sign * 2, right))
    detour.add((row + sign, right))

    bar_row = row + sign * (half + 2)
    bar = {(bar_row, mid + d) for d in range(-half, half + 1)}
    return detour, bar


def _apply_random_transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if random.random() < 0.5:
        out = out.T.copy()
    if random.random() < 0.5:
        out = np.flipud(out).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    return out


def _path_signature(path: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(path))


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    if not colors <= {BLACK, GREY, ORANGE, TEAL, MAROON}:
        return False

    endpoints = [tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)]
    if len(endpoints) != 2:
        return False

    try:
        bars = _extract_bars(g)
    except AssertionError:
        return False
    if not (3 <= len(bars) <= 4):
        return False
    if not all(3 <= len(bar) <= 7 and len(bar) % 2 == 1 for bar in bars):
        return False

    oranges = [tuple(map(int, rc)) for rc in np.argwhere(g == ORANGE)]
    if not (2 <= len(oranges) <= 3):
        return False

    track = _track_mask(g)
    if _count_simple_paths(track, endpoints[0], endpoints[1], limit=2) < 2:
        return False

    try:
        out, safe_path, passable = _main_logic(g)
    except AssertionError:
        return False

    if np.array_equal(out, g):
        return False
    if not any(g[r, c] == TEAL for r, c in safe_path):
        return False

    try:
        total_forbidden, orange_layers, bar_layers, _, _ = _forbidden_layers(g)
    except AssertionError:
        return False
    if any(not np.any(track & layer) for layer in orange_layers + bar_layers):
        return False

    path_sig = _path_signature(safe_path)

    forbidden_with_original_bars = np.zeros(g.shape, dtype=bool)
    for pos in oranges:
        forbidden_with_original_bars |= _mark_forbidden(g.shape, {pos})
    for bar in bars:
        forbidden_with_original_bars |= _mark_forbidden(g.shape, bar)
    passable_original = track & ~forbidden_with_original_bars
    count_original = _count_simple_paths(passable_original, endpoints[0], endpoints[1], limit=2)
    if count_original == 1:
        original_path = _shortest_path_nodes(passable_original, endpoints[0], endpoints[1])
        if _path_signature(original_path) == path_sig:
            return False

    all_layers = orange_layers + bar_layers
    for i, removed in enumerate(all_layers):
        forbid = np.zeros(g.shape, dtype=bool)
        for j, layer in enumerate(all_layers):
            if i != j:
                forbid |= layer
        candidate = track & ~forbid
        if _count_simple_paths(candidate, endpoints[0], endpoints[1], limit=2) == 1:
            return False

    return np.count_nonzero(passable) >= 12


def _interval_conflict(existing: list[tuple[int, int]], a: int, b: int, pad: int = 2) -> bool:
    for x0, x1 in existing:
        if not (b + pad < x0 or a - pad > x1):
            return True
    return False


def _attempt_base_grid() -> np.ndarray | None:
    height, lane_rows = random.choice([(27, [7, 13, 19]), (29, [8, 14, 20])])
    width = random.choice([29, 29, 27])

    base_track, start, end, c_left, c_right = _build_snake(height, width, lane_rows)
    occupied_track = set(base_track)
    occupied_grey: set[tuple[int, int]] = set()
    occupied_orange: set[tuple[int, int]] = set()
    slot_template = [7, 13, 19] if width == 27 else [7, 13, 19, 24]
    if width == 29 and random.random() < 0.55:
        placements = [
            (0, "above", slot_template[0]),
            (0, "above", slot_template[2]),
            (0, "above", slot_template[3]),
            (len(lane_rows) - 1, "below", slot_template[0]),
            (len(lane_rows) - 1, "below", slot_template[1]),
            (len(lane_rows) - 1, "below", slot_template[2]),
        ]
        gray_count, orange_count = random.choice([(4, 2), (3, 3)])
    else:
        placements = [
            (0, "above", slot_template[0]),
            (0, "above", slot_template[2]),
            (len(lane_rows) - 1, "below", slot_template[0]),
            (len(lane_rows) - 1, "below", slot_template[1]),
            (len(lane_rows) - 1, "below", slot_template[2]),
        ]
        gray_count, orange_count = 3, 2

    module_types = ["gray"] * gray_count + ["orange"] * orange_count
    random.shuffle(module_types)

    for module_type, (lane_idx, side, mid_base) in zip(module_types, placements):
        row = lane_rows[lane_idx]
        mid = mid_base + random.choice([-1, 0, 1])

        if module_type == "orange":
            detour, obstacle = _orange_module(row, side, mid)
            true_layer = _mark_forbidden((height, width), obstacle)
            grey_bar = set()
            orange_pix = set(obstacle)
        else:
            clearance = row if side == "above" else height - 1 - row
            feasible_lengths = [L for L in (3, 5, 7) if clearance >= (L // 2 + 2)]
            if not feasible_lengths:
                return None
            length = random.choice(feasible_lengths)
            detour, grey_bar = _gray_module(row, side, mid, length)
            true_layer = _mark_forbidden((height, width), _rotated_bar_pixels(grey_bar))
            orange_pix = set()

        if not all(_in_bounds((height, width), r, c) for r, c in detour | grey_bar | orange_pix):
            return None
        if any(true_layer[r, c] for r, c in base_track):
            return None
        if detour & occupied_track:
            return None
        if detour & occupied_grey:
            return None
        if detour & occupied_orange:
            return None
        if grey_bar & occupied_track:
            return None
        if grey_bar & occupied_grey:
            return None
        if orange_pix & occupied_track:
            return None
        if orange_pix & occupied_orange:
            return None
        if orange_pix & occupied_grey:
            return None
        if not any(true_layer[r, c] for r, c in detour):
            return None

        occupied_track |= detour
        occupied_grey |= grey_bar
        occupied_orange |= orange_pix

    grid = np.full((height, width), BLACK, dtype=int)
    for r, c in occupied_track:
        grid[r, c] = TEAL
    for r, c in occupied_grey:
        grid[r, c] = GREY
    for r, c in occupied_orange:
        grid[r, c] = ORANGE
    grid[start] = MAROON
    grid[end] = MAROON

    try:
        forbidden, _, _, _, _ = _forbidden_layers(grid)
    except AssertionError:
        return None
    passable = _track_mask(grid) & ~forbidden
    if {tuple(map(int, rc)) for rc in np.argwhere(passable)} != base_track:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        base = _attempt_base_grid()
        if base is None:
            continue
        grid = _apply_random_transform(base)
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
