from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400
MAX_ABS_STEPS = 4
CORNER_OFFSETS = ((0, 0), (0, 2), (2, 0), (2, 2))
BASE_SHAPES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (1, -1)),
    ((-1, 0), (0, 0), (1, 0), (0, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((-1, 1), (0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((-1, 0), (0, 0), (0, 1), (1, 1), (1, 2)),
)


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _as_coord(coord: np.ndarray | tuple[int, int]) -> tuple[int, int]:
    r, c = coord
    return int(r), int(c)


def _axis_distance(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    if a[0] == b[0]:
        d = abs(a[1] - b[1])
        return d if d > 0 else None
    if a[1] == b[1]:
        d = abs(a[0] - b[0])
        return d if d > 0 else None
    return None


def _candidate_steps(points: list[tuple[int, int]]) -> list[int]:
    vals = set()
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            d = _axis_distance(a, b)
            if d is not None:
                vals.add(int(d))
    return sorted(vals)


def _path_from_points(
    points: list[tuple[int, int]],
    anchor: tuple[int, int],
    blue: tuple[int, int],
    step: int,
) -> dict | None:
    n = len(points)
    if n < 2 or step <= 0 or len(set(points)) != n:
        return None

    adj = [[] for _ in range(n)]
    for i, a in enumerate(points):
        for j in range(i + 1, n):
            b = points[j]
            if _axis_distance(a, b) == step:
                adj[i].append(j)
                adj[j].append(i)

    degrees = [len(nei) for nei in adj]
    if any(d not in (1, 2) for d in degrees):
        return None
    endpoints = [i for i, d in enumerate(degrees) if d == 1]
    if len(endpoints) != 2 or sum(degrees) // 2 != n - 1:
        return None

    seen = set()
    stack = [0]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj[cur])
    if len(seen) != n:
        return None

    blue_hits = []
    for idx in endpoints:
        nei = adj[idx][0]
        dr = _sgn(points[idx][0] - points[nei][0])
        dc = _sgn(points[idx][1] - points[nei][1])
        expected_blue = (points[idx][0] + dr, points[idx][1] + dc)
        if expected_blue == blue:
            blue_hits.append(idx)
    if len(blue_hits) != 1:
        return None

    blue_idx = blue_hits[0]
    start_idx = endpoints[0] if endpoints[1] == blue_idx else endpoints[1]
    order_idx = [start_idx]
    prev = -1
    cur = start_idx
    while cur != blue_idx:
        nxt = [v for v in adj[cur] if v != prev]
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]
        order_idx.append(cur)
    if len(order_idx) != n:
        return None

    order = [points[i] for i in order_idx]
    return {
        "order": order,
        "step": int(step),
        "anchor": anchor,
        "current_index": int(order.index(anchor)),
        "blue_index": int(len(order) - 1),
    }


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == Color.BLACK

    blue_coords = np.argwhere(g == Color.BLUE)
    assert blue_coords.shape == (1, 2), "there must be exactly one blue endpoint marker"
    blue = _as_coord(blue_coords[0])

    red_components = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, colors=[Color.RED]
    )
    assert len(red_components) == 1, "there must be exactly one red object"
    red_coords = red_components[0]["coords"].astype(int)
    red_cells = [_as_coord(cell) for cell in red_coords]

    green_cells = [_as_coord(cell) for cell in np.argwhere(g == Color.GREEN)]
    assert len(green_cells) >= 4, "track must expose multiple visible slots"

    solutions = {}
    for anchor in red_cells:
        points = list(green_cells) + [anchor]
        for step in _candidate_steps(points):
            sol = _path_from_points(points, anchor, blue, step)
            if sol is None:
                continue
            key = (sol["anchor"], sol["step"], tuple(sol["order"]))
            solutions[key] = sol

    assert len(solutions) == 1, "track anchor and spacing must be uniquely determined"
    sol = next(iter(solutions.values()))
    track_set = set(sol["order"])
    assert sum(cell in track_set for cell in red_cells) == 1, "red object must cover one slot"

    return {
        "bg": bg,
        "blue": blue,
        "red_coords": red_coords,
        "red_cells": red_cells,
        "order": sol["order"],
        "step": sol["step"],
        "current_index": sol["current_index"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move the red object by a signed number of slots along the green dotted bent track.

    Input structure:
    - A black background contains a sequence of equally spaced green single-pixel track slots,
      optionally with one right-angle bend.
    - One red connected object covers exactly one hidden track slot, and exactly one track end
      has a blue marker one cell beyond it.
    - Yellow and teal single-pixel tally markers sit away from the track and are used only as clues.

    Rule:
    - Reconstruct the ordered slot sequence of the track, with positive direction pointing toward
      the end next to the blue marker.
    - Compute the signed displacement as `#yellow - #teal`; positive moves toward blue and negative
      moves away from blue.
    - Redraw the full green track, then translate the entire red object so that its covered slot lands
      on the destination slot while preserving the object's internal pose.

    Key insight:
    - The movement is a step count on the slot order of a bent polyline, not a direct geometric shift,
      and the hidden current slot must be inferred from the unique red pixel that completes that path.
    """
    details = _solve_details(grid)
    order = details["order"]
    current_index = int(details["current_index"])
    signed_steps = int(np.sum(grid == Color.YELLOW) - np.sum(grid == Color.TEAL))
    target_index = current_index + signed_steps
    assert 0 <= target_index < len(order), "signed movement must stay on the track"

    current_anchor = np.array(order[current_index], dtype=int)
    target_anchor = np.array(order[target_index], dtype=int)
    delta = target_anchor - current_anchor
    shifted_red = details["red_coords"] + delta
    H, W = grid.shape
    assert (
        shifted_red[:, 0].min() >= 0
        and shifted_red[:, 0].max() < H
        and shifted_red[:, 1].min() >= 0
        and shifted_red[:, 1].max() < W
    )

    track_set = set(order)
    shifted_cells = {_as_coord(cell) for cell in shifted_red}
    assert sum(cell in track_set for cell in shifted_cells) == 1, "shifted object must cover one slot"

    out = np.full_like(grid, details["bg"])
    for r, c in order:
        out[r, c] = Color.GREEN
    out[shifted_red[:, 0], shifted_red[:, 1]] = Color.RED
    return out


def _transform_offsets(offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    k = random.randint(0, 3)
    flip = random.random() < 0.5
    out = []
    for r, c in offsets:
        rr, cc = int(r), int(c)
        if flip:
            cc = -cc
        for _ in range(k):
            rr, cc = cc, -rr
        out.append((rr, cc))
    return tuple(sorted(set(out)))


def _choose_object_offsets() -> np.ndarray:
    base = random.choice(BASE_SHAPES)
    coords = _transform_offsets(base)
    assert (0, 0) in coords
    arr = np.array(coords, dtype=int)
    assert arr.shape[0] >= 3
    return arr


def _turn_choices(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(0, -1), (0, 1)] if dr != 0 else [(-1, 0), (1, 0)]


def _generate_order() -> tuple[list[tuple[int, int]], int | None]:
    step = random.choice([2, 3])
    bent = random.random() < 0.75
    direction = random.choice(DIRS4)

    if bent:
        max_seg = 4 if step == 3 else 5
        len1 = random.randint(3, max_seg)
        len2 = random.randint(3, max_seg)
        turn = random.choice(_turn_choices(direction))
        order = [(direction[0] * step * k, direction[1] * step * k) for k in range(len1)]
        bend = order[-1]
        for k in range(1, len2):
            order.append((bend[0] + turn[0] * step * k, bend[1] + turn[1] * step * k))
        bend_index = len1 - 1
    else:
        max_len = 7 if step == 3 else 8
        length = random.randint(5, max_len)
        order = [(direction[0] * step * k, direction[1] * step * k) for k in range(length)]
        bend_index = None

    if random.random() < 0.5:
        order = list(reversed(order))
        if bend_index is not None:
            bend_index = len(order) - 1 - bend_index
    return order, bend_index


def _blue_marker(order: list[tuple[int, int]]) -> tuple[int, int]:
    last = order[-1]
    prev = order[-2]
    dr = _sgn(last[0] - prev[0])
    dc = _sgn(last[1] - prev[1])
    return last[0] + dr, last[1] + dc


def _crosses_bend(current: int, target: int, bend_index: int | None) -> bool:
    if bend_index is None:
        return False
    return (current < bend_index < target) or (target < bend_index < current)


def _choose_indices(order_len: int, bend_index: int | None) -> tuple[int, int]:
    require_cross = bend_index is not None and 0 < bend_index < order_len - 1 and random.random() < 0.6

    for _ in range(300):
        sign = random.choices([-1, 0, 1], weights=[4, 2, 4], k=1)[0]
        if require_cross and sign == 0:
            continue

        if sign > 0:
            current = random.randint(0, order_len - 2)
            max_mag = min(MAX_ABS_STEPS, order_len - 1 - current)
            if max_mag < 1:
                continue
            target = current + random.randint(1, max_mag)
        elif sign < 0:
            current = random.randint(1, order_len - 1)
            max_mag = min(MAX_ABS_STEPS, current)
            if max_mag < 1:
                continue
            target = current - random.randint(1, max_mag)
        else:
            current = random.randint(0, order_len - 1)
            target = current

        if require_cross and not _crosses_bend(current, target, bend_index):
            continue
        return current, target

    raise RuntimeError("failed to choose valid source/target indices")


def _marker_counts(steps: int) -> tuple[int, int]:
    base = random.randint(1, 3)
    if steps >= 0:
        return base + steps, base
    return base, base - steps


def _shift_coords(coords: np.ndarray | list[tuple[int, int]], dr: int, dc: int) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    return arr + np.array([int(dr), int(dc)], dtype=int)


def _corner_slots(H: int, W: int) -> list[tuple[int, int]]:
    anchors = [(0, 0), (0, W - 3), (H - 3, 0), (H - 3, W - 3)]
    slots = []
    for r0, c0 in anchors:
        for dr, dc in CORNER_OFFSETS:
            slots.append((r0 + dr, c0 + dc))
    return slots


def _paint_track(grid: np.ndarray, order: list[tuple[int, int]], omit: tuple[int, int] | None = None) -> None:
    for cell in order:
        if omit is not None and cell == omit:
            continue
        grid[cell] = Color.GREEN


def _paint_object(grid: np.ndarray, obj_cells: np.ndarray) -> None:
    grid[obj_cells[:, 0], obj_cells[:, 1]] = Color.RED


def _build_output(shape: tuple[int, int], order: list[tuple[int, int]], obj_cells: np.ndarray) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    _paint_track(out, order)
    _paint_object(out, obj_cells)
    return out


def _gen_attempt() -> np.ndarray | None:
    order_local, bend_index = _generate_order()
    current_index, target_index = _choose_indices(len(order_local), bend_index)
    signed_steps = target_index - current_index
    yellow_count, teal_count = _marker_counts(signed_steps)
    if yellow_count + teal_count > 16:
        return None

    object_offsets = _choose_object_offsets()
    current_anchor = np.array(order_local[current_index], dtype=int)
    target_anchor = np.array(order_local[target_index], dtype=int)
    current_obj = object_offsets + current_anchor
    target_obj = object_offsets + target_anchor
    blue_local = np.array(_blue_marker(order_local), dtype=int)

    track_local = np.asarray(order_local, dtype=int)
    everything = np.vstack([track_local, current_obj, target_obj, blue_local[None, :]])
    r0, c0 = everything.min(axis=0)
    r1, c1 = everything.max(axis=0)
    top = random.randint(3, 4)
    bottom = random.randint(3, 4)
    left = random.randint(3, 4)
    right = random.randint(3, 4)
    H = int(r1 - r0 + 1 + top + bottom)
    W = int(c1 - c0 + 1 + left + right)
    if H > 30 or W > 30:
        return None

    shift = np.array([top - r0, left - c0], dtype=int)
    order = [_as_coord(cell + shift) for cell in track_local]
    current_anchor_s = _as_coord(current_anchor + shift)
    target_anchor_s = _as_coord(target_anchor + shift)
    current_obj_s = current_obj + shift
    target_obj_s = target_obj + shift
    blue = _as_coord(blue_local + shift)

    track_set = set(order)
    current_cells = {_as_coord(cell) for cell in current_obj_s}
    target_cells = {_as_coord(cell) for cell in target_obj_s}

    if current_anchor_s not in current_cells or target_anchor_s not in target_cells:
        return None
    if sum(cell in track_set for cell in current_cells) != 1:
        return None
    if sum(cell in track_set for cell in target_cells) != 1:
        return None
    if blue in current_cells or blue in target_cells or blue in track_set:
        return None

    marker_slots = _corner_slots(H, W)
    if len(marker_slots) < yellow_count + teal_count:
        return None
    random.shuffle(marker_slots)
    yellow_slots = marker_slots[:yellow_count]
    teal_slots = marker_slots[yellow_count : yellow_count + teal_count]
    marker_cells = set(yellow_slots) | set(teal_slots)
    if current_cells & marker_cells or target_cells & marker_cells or track_set & marker_cells:
        return None
    if blue in marker_cells:
        return None

    inp = np.full((H, W), Color.BLACK, dtype=int)
    _paint_track(inp, order, omit=current_anchor_s)
    inp[blue] = Color.BLUE
    for cell in yellow_slots:
        inp[cell] = Color.YELLOW
    for cell in teal_slots:
        inp[cell] = Color.TEAL
    _paint_object(inp, current_obj_s)

    expected = _build_output(inp.shape, order, target_obj_s)

    try:
        details = _solve_details(inp)
    except Exception:
        return None

    if details["order"] != order or int(details["current_index"]) != current_index:
        return None
    if np.sum(inp == Color.YELLOW) - np.sum(inp == Color.TEAL) != signed_steps:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if bend_index is not None and random.random() < 0.2:
        if not _crosses_bend(current_index, target_index, bend_index):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bent-track stepping puzzle")
