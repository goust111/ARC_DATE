from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
AZURE = 8
PALETTE_POOL = (1, 2, 3, 4, 6, 9)

GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 30
PALETTE_LEN_RANGE = (2, 5)
PLAN_H_RANGE = (10, 18)
PLAN_W_RANGE = (10, 18)
MIN_ROOM_H = 3
MIN_ROOM_W = 3
MAX_TARGET_ROOMS = 7
PLACEMENT_MARGIN = 1
SEPARATION_GAP = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_palette(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, list[int], np.ndarray]] = []

    for comp in _components(grid == ORANGE):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bw != 3 or bh < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(sub == ORANGE, border):
            continue

        colors = [int(v) for v in sub[1:-1, 1].tolist()]
        if not colors:
            continue

        corner_distance = min(
            r0 + c0,
            r0 + (w - 1 - c1),
            (h - 1 - r1) + c0,
            (h - 1 - r1) + (w - 1 - c1),
        )
        mask = np.zeros((h, w), dtype=bool)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        candidates.append((corner_distance, -len(colors), colors, mask))

    assert candidates, "expected an orange-bordered vertical palette"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _corner_distance, _neg_len, colors, mask = candidates[0]
    return colors, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with the palette color indexed by its azure-window count.

    Input structure:
    - Black background.
    - Grey axis-aligned walls partition a floorplan into closed rectangular rooms.
    - Azure pixels on walls are windows.
    - An orange-bordered vertical palette strip lists colors from top to bottom.

    Rule:
    - Read the palette as C_0, C_1, ...
    - For each enclosed black room, count the unique azure pixels adjacent to its boundary.
    - Fill the room interior with C_W when W is in range; otherwise keep it black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    palette, palette_mask = _find_palette(g)
    out = g.copy()

    room_mask = (g == BLACK) & (~palette_mask)
    for comp in _components(room_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        windows = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == AZURE:
                    windows.add((int(nr), int(nc)))

        count = len(windows)
        if count < len(palette):
            fill = int(palette[count])
            for r, c in comp:
                out[r, c] = fill

    return out


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _draw_border(grid: np.ndarray, top: int, left: int, bottom: int, right: int, color: int) -> None:
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _splittable(room: tuple[int, int, int, int]) -> list[str]:
    r0, r1, c0, c1 = room
    choices = []
    if r1 - r0 + 1 >= 2 * MIN_ROOM_H + 1:
        choices.append("h")
    if c1 - c0 + 1 >= 2 * MIN_ROOM_W + 1:
        choices.append("v")
    return choices


def _partition_plan(
    grid: np.ndarray,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> list[tuple[int, int, int, int]]:
    _draw_border(grid, top, left, bottom, right, GREY)
    rooms = [(top + 1, bottom - 1, left + 1, right - 1)]

    interior_area = (bottom - top - 1) * (right - left - 1)
    room_cap = max(2, min(MAX_TARGET_ROOMS, interior_area // 14))
    target_rooms = random.randint(2, room_cap)

    while len(rooms) < target_rooms:
        indices = [idx for idx, room in enumerate(rooms) if _splittable(room)]
        if not indices:
            break
        idx = random.choice(indices)
        room = rooms.pop(idx)
        r0, r1, c0, c1 = room
        orientation = random.choice(_splittable(room))
        if orientation == "v":
            split_c = random.randint(c0 + MIN_ROOM_W, c1 - MIN_ROOM_W)
            grid[r0 - 1 : r1 + 2, split_c] = GREY
            rooms.append((r0, r1, c0, split_c - 1))
            rooms.append((r0, r1, split_c + 1, c1))
        else:
            split_r = random.randint(r0 + MIN_ROOM_H, r1 - MIN_ROOM_H)
            grid[split_r, c0 - 1 : c1 + 2] = GREY
            rooms.append((r0, split_r - 1, c0, c1))
            rooms.append((split_r + 1, r1, c0, c1))

    return rooms


def _room_segments(
    room: tuple[int, int, int, int],
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> list[list[tuple[int, int]]]:
    r0, r1, c0, c1 = room
    segments: list[list[tuple[int, int]]] = []

    if r0 == top + 1 and c1 - c0 + 1 >= 3:
        segments.append([(r0 - 1, c) for c in range(c0 + 1, c1)])
    if r1 == bottom - 1 and c1 - c0 + 1 >= 3:
        segments.append([(r1 + 1, c) for c in range(c0 + 1, c1)])
    if c0 == left + 1 and r1 - r0 + 1 >= 3:
        segments.append([(r, c0 - 1) for r in range(r0 + 1, r1)])
    if c1 == right - 1 and r1 - r0 + 1 >= 3:
        segments.append([(r, c1 + 1) for r in range(r0 + 1, r1)])

    return [segment for segment in segments if segment]


def _segment_capacity(segment: list[tuple[int, int]]) -> int:
    return (len(segment) + 1) // 2


def _choose_non_adjacent(segment: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    assert 0 <= count <= _segment_capacity(segment)
    if count == 0:
        return []
    bases = sorted(random.sample(range(len(segment) - count + 1), count))
    return [segment[base + offset] for offset, base in enumerate(bases)]


def _choose_windows(segments: list[list[tuple[int, int]]], count: int) -> list[tuple[int, int]]:
    capacities = [_segment_capacity(segment) for segment in segments]
    assert sum(capacities) >= count

    order = list(range(len(segments)))
    random.shuffle(order)
    take = [0] * len(segments)
    remaining = count

    for pos, idx in enumerate(order):
        future_cap = sum(capacities[j] for j in order[pos + 1 :])
        minimum_here = max(0, remaining - future_cap)
        maximum_here = min(capacities[idx], remaining)
        take[idx] = random.randint(minimum_here, maximum_here)
        remaining -= take[idx]

    assert remaining == 0

    chosen: list[tuple[int, int]] = []
    for segment, amount in zip(segments, take, strict=False):
        chosen.extend(_choose_non_adjacent(segment, amount))
    return chosen


def _pick_target_counts(room_caps: list[int], palette_len: int) -> list[int] | None:
    n = len(room_caps)
    if n == 0:
        return None

    targets = [0] * n

    in_range = [i for i, cap in enumerate(room_caps) if cap >= 1]
    overflow = [i for i, cap in enumerate(room_caps) if cap >= palette_len]
    if not in_range or not overflow:
        return None

    idx_in = random.choice(in_range)
    targets[idx_in] = random.randint(1, min(room_caps[idx_in], palette_len - 1))

    idx_over = random.choice(overflow)
    if idx_over == idx_in and len(overflow) > 1:
        idx_over = random.choice([i for i in overflow if i != idx_in])
    targets[idx_over] = random.randint(palette_len, room_caps[idx_over])

    for i, cap in enumerate(room_caps):
        classes = ["zero"]
        if cap >= 1:
            classes.append("in")
        if cap >= palette_len:
            classes.append("over")
        choice = random.choice(classes)
        if choice == "zero":
            continue
        if choice == "in":
            targets[i] = random.randint(1, min(cap, palette_len - 1))
        else:
            targets[i] = random.randint(palette_len, cap)

    if not any(count == 0 for count in targets):
        targets[random.randrange(n)] = 0
    if not any(0 < count < palette_len for count in targets):
        idx = random.choice(in_range)
        targets[idx] = random.randint(1, min(room_caps[idx], palette_len - 1))
    if not any(count >= palette_len for count in targets):
        idx = random.choice(overflow)
        targets[idx] = random.randint(palette_len, room_caps[idx])

    return targets


def _place_palette(
    grid: np.ndarray,
    colors: list[int],
) -> tuple[int, int, int, int]:
    h, w = map(int, grid.shape)
    ph, pw = len(colors) + 2, 3
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        r0, c0 = 1, 1
    elif corner == "tr":
        r0, c0 = 1, w - pw - 1
    elif corner == "bl":
        r0, c0 = h - ph - 1, 1
    else:
        r0, c0 = h - ph - 1, w - pw - 1

    _draw_border(grid, r0, c0, r0 + ph - 1, c0 + pw - 1, ORANGE)
    for i, color in enumerate(colors):
        grid[r0 + 1 + i, c0 + 1] = int(color)
    return r0, c0, r0 + ph - 1, c0 + pw - 1


def _attempt() -> np.ndarray | None:
    palette_len = random.randint(*PALETTE_LEN_RANGE)
    palette = random.sample(PALETTE_POOL, palette_len)

    h = random.randint(max(GRID_MIN, palette_len + 8), GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)

    palette_bbox = _place_palette(grid, palette)

    plan_h_max = min(PLAN_H_RANGE[1], h - 2 * PLACEMENT_MARGIN - 1)
    plan_w_max = min(PLAN_W_RANGE[1], w - 2 * PLACEMENT_MARGIN - 1)
    if plan_h_max < PLAN_H_RANGE[0] or plan_w_max < PLAN_W_RANGE[0]:
        return None

    plan_h = random.randint(PLAN_H_RANGE[0], plan_h_max)
    plan_w = random.randint(PLAN_W_RANGE[0], plan_w_max)

    plan_bbox = None
    for _ in range(80):
        top = random.randint(PLACEMENT_MARGIN, h - plan_h - PLACEMENT_MARGIN)
        left = random.randint(PLACEMENT_MARGIN, w - plan_w - PLACEMENT_MARGIN)
        candidate = (top, left, top + plan_h - 1, left + plan_w - 1)
        if _rects_separated(candidate, palette_bbox, SEPARATION_GAP):
            plan_bbox = candidate
            break
    if plan_bbox is None:
        return None

    top, left, bottom, right = plan_bbox
    rooms = _partition_plan(grid, top, left, bottom, right)
    if len(rooms) < 2:
        return None

    segments = [_room_segments(room, top, left, bottom, right) for room in rooms]
    capacities = [sum(_segment_capacity(segment) for segment in room_segments) for room_segments in segments]
    if any(cap == 0 for cap in capacities):
        return None

    targets = _pick_target_counts(capacities, palette_len)
    if targets is None:
        return None

    for room, room_segments, target in zip(rooms, segments, targets, strict=False):
        for wr, wc in _choose_windows(room_segments, target):
            grid[wr, wc] = AZURE
    expected = grid.copy()
    for room, target in zip(rooms, targets, strict=False):
        if target < palette_len:
            r0, r1, c0, c1 = room
            expected[r0 : r1 + 1, c0 : c1 + 1] = int(palette[target])

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-window palette puzzle")
