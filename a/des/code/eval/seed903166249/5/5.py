import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (17, 28)
GRID_W_RANGE = (17, 28)
ROOT_LEN_RANGE = (3, 7)
CHILD_LEN_RANGE = (2, 6)
TARGET_SEGMENT_RANGE = (4, 8)
MAX_DEPTH = 5
ACTION_PLAN_TRIES = 40
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _corner_marker_cells(h: int, w: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if corner == "tr":
        return [(0, w - 2), (0, w - 1), (1, w - 2), (1, w - 1)]
    if corner == "bl":
        return [(h - 2, 0), (h - 2, 1), (h - 1, 0), (h - 1, 1)]
    if corner == "br":
        return [(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)]
    raise ValueError(f"unknown corner {corner}")


def _find_marker_block(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]], str]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, str], int, list[tuple[int, int]], str]] = []
    for corner in ("tl", "tr", "bl", "br"):
        cells = _corner_marker_cells(h, w, corner)
        vals = [int(grid[r, c]) for r, c in cells]
        if len(set(vals)) == 1 and vals[0] != bg:
            color = int(vals[0])
            total = int(np.count_nonzero(grid == color))
            candidates.append(((total != 4, total, corner), color, cells, corner))
    if not candidates:
        raise ValueError("missing corner marker block")
    candidates.sort(key=lambda x: x[0])
    _key, color, cells, corner = candidates[0]
    return int(color), cells, corner


def _leaf_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    leaves: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        deg = 0
        for rr, cc in _neighbors4(int(r), int(c), h, w):
            if bool(mask[rr, cc]):
                deg += 1
        if deg == 1:
            leaves.append((int(r), int(c)))
    return leaves


def _closest_leaf(mask: np.ndarray, marker_cells: list[tuple[int, int]]) -> tuple[int, int]:
    leaves = _leaf_cells(mask)
    if not leaves:
        raise ValueError("path mask has no leaf")

    def dist(p: tuple[int, int]) -> tuple[int, int, int]:
        best = min(abs(p[0] - mr) + abs(p[1] - mc) for mr, mc in marker_cells)
        return (int(best), int(p[0]), int(p[1]))

    return min(leaves, key=dist)


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    marker_color, marker_cells, _corner = _find_marker_block(g, bg)

    path_mask = g != bg
    for r, c in marker_cells:
        path_mask[r, c] = False

    root = _closest_leaf(path_mask, marker_cells)

    out = g.copy()
    seen: set[tuple[int, int]] = set()
    stack: list[tuple[tuple[int, int], tuple[int, int] | None, int]] = [(root, None, marker_color)]

    while stack:
        cell, parent, cur_color = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        out[r, c] = int(cur_color)

        prev_dir = None
        if parent is not None:
            prev_dir = (int(r - parent[0]), int(c - parent[1]))

        for rr, cc in _neighbors4(r, c, h, w):
            if not bool(path_mask[rr, cc]) or (parent is not None and (rr, cc) == parent):
                continue
            move_dir = (int(rr - r), int(cc - c))
            next_color = int(cur_color) if prev_dir is None or move_dir == prev_dir else int(g[rr, cc])
            stack.append(((rr, cc), (r, c), next_color))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor an orthogonal path tree by direction-sensitive inheritance.

    Input structure:
    - Black background.
    - A single 1-pixel-wide orthogonal tree made of colored straight segments.
    - A 2x2 solid marker block in one corner with a color distinct from the path.

    Rule:
    - The root path starts at the leaf nearest the corner marker.
    - The root segment is recolored to the marker color.
    - Along the tree, continuing in the same direction inherits the current color.
    - Turning into a perpendicular segment resets to that segment's original color.
    """
    return _transform_grid(np.asarray(grid, dtype=int))


def _inside_path_area(r: int, c: int, h: int, w: int) -> bool:
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _build_root_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(int(r + k * dr), int(c + k * dc)) for k in range(length)]


def _build_child_cells(node: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = node
    dr, dc = direction
    return [(int(r + k * dr), int(c + k * dc)) for k in range(1, length + 1)]


def _valid_cells(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
    first_allowed: tuple[int, int] | None,
) -> bool:
    for i, (r, c) in enumerate(cells):
        if not _inside_path_area(r, c, h, w):
            return False
        if (r, c) in blocked or (r, c) in occupied:
            return False

        allowed: set[tuple[int, int]] = set()
        if i == 0:
            if first_allowed is not None:
                allowed.add(first_allowed)
        else:
            allowed.add(cells[i - 1])

        for rr, cc in _neighbors4(r, c, h, w):
            if (rr, cc) in blocked:
                return False
            if (rr, cc) in occupied and (rr, cc) not in allowed:
                return False
    return True


def _turn_dirs(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _choose_root_start(h: int, w: int, corner: str, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner == "tl":
        if rng.random() < 0.5:
            return (2, rng.randint(2, min(4, w - 3))), (1, 0)
        return (rng.randint(2, min(4, h - 3)), 2), (0, 1)
    if corner == "tr":
        if rng.random() < 0.5:
            return (2, rng.randint(max(2, w - 5), w - 3)), (1, 0)
        return (rng.randint(2, min(4, h - 3)), w - 3), (0, -1)
    if corner == "bl":
        if rng.random() < 0.5:
            return (h - 3, rng.randint(2, min(4, w - 3))), (-1, 0)
        return (rng.randint(max(2, h - 5), h - 3), 2), (0, 1)
    if corner == "br":
        if rng.random() < 0.5:
            return (h - 3, rng.randint(max(2, w - 5), w - 3)), (-1, 0)
        return (rng.randint(max(2, h - 5), h - 3), w - 3), (0, -1)
    raise ValueError(f"unknown corner {corner}")


def _plan_children(
    node: tuple[int, int],
    direction: tuple[int, int],
    action: str,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
    rng: random.Random,
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]] | None:
    turns = _turn_dirs(direction)
    if action == "bend":
        dir_builder = lambda: [rng.choice(turns)]
    elif action == "straight_turn":
        dir_builder = lambda: [direction, rng.choice(turns)]
    elif action == "two_turns":
        dir_builder = lambda: turns[:]
    else:
        return None

    lengths = list(range(CHILD_LEN_RANGE[0], CHILD_LEN_RANGE[1] + 1))
    for _ in range(ACTION_PLAN_TRIES):
        child_dirs = dir_builder()
        order = child_dirs[:]
        rng.shuffle(order)
        temp_occ = set(occupied)
        plans: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        ok = True
        for child_dir in order:
            rng.shuffle(lengths)
            placed = False
            for length in lengths:
                cells = _build_child_cells(node, child_dir, length)
                if _valid_cells(cells, temp_occ, blocked, h, w, node):
                    plans.append((child_dir, cells))
                    temp_occ.update(cells)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return plans
    return None


def _pick_action(depth: int, seg_count: int, target_segments: int, rng: random.Random) -> str:
    if depth >= MAX_DEPTH:
        return "stop"
    if depth >= MAX_DEPTH - 1:
        return rng.choices(["stop", "bend"], weights=[0.6, 0.4], k=1)[0]
    if seg_count < target_segments - 2:
        return rng.choices(
            ["bend", "straight_turn", "two_turns", "stop"],
            weights=[0.35, 0.35, 0.15, 0.15],
            k=1,
        )[0]
    return rng.choices(
        ["stop", "bend", "straight_turn", "two_turns"],
        weights=[0.45, 0.35, 0.15, 0.05],
        k=1,
    )[0]


def _assign_segment_colors(segments: list[dict], marker_color: int, rng: random.Random) -> None:
    base_pool = [c for c in PALETTE if c != marker_color]
    active_count = rng.randint(3, min(6, len(base_pool)))
    active_pool = rng.sample(base_pool, active_count)

    for seg_id, seg in enumerate(segments):
        parent = seg["parent"]
        if parent is None:
            seg["color"] = int(rng.choice(active_pool))
            continue

        exclude = {int(segments[parent]["color"])}
        grand = segments[parent]["parent"]
        if grand is not None and rng.random() < 0.5:
            exclude.add(int(segments[grand]["color"]))

        choices = [c for c in active_pool if c not in exclude]
        if not choices:
            choices = [c for c in base_pool if c not in exclude]
        if not choices:
            choices = base_pool[:]
        seg["color"] = int(rng.choice(choices))


def _connected_size(cells: set[tuple[int, int]], start: tuple[int, int], h: int, w: int) -> int:
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return int(len(seen))


def _paint_grid(
    h: int,
    w: int,
    segments: list[dict],
    marker_cells: list[tuple[int, int]],
    marker_color: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in marker_cells:
        grid[r, c] = int(marker_color)
    for seg in segments:
        color = int(seg["color"])
        for r, c in seg["cells"]:
            grid[r, c] = color
    return grid


def _segment_changed(grid: np.ndarray, out: np.ndarray, seg: dict) -> bool:
    return any(int(grid[r, c]) != int(out[r, c]) for r, c in seg["cells"])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    marker_color = rng.choice(PALETTE)
    marker_cells = _corner_marker_cells(h, w, corner)

    blocked = set(marker_cells)
    for r, c in marker_cells:
        for rr, cc in _neighbors4(r, c, h, w):
            blocked.add((rr, cc))

    root_start, root_dir = _choose_root_start(h, w, corner, rng)
    root_len = rng.randint(*ROOT_LEN_RANGE)
    root_cells = _build_root_cells(root_start, root_dir, root_len)
    if not _valid_cells(root_cells, set(), blocked, h, w, None):
        return None

    occupied = set(root_cells)
    segments: list[dict] = [{"parent": None, "dir": root_dir, "cells": root_cells, "children": []}]
    frontiers: list[dict] = []

    root_split = _plan_children(root_cells[-1], root_dir, "straight_turn", occupied, blocked, h, w, rng)
    if root_split is None:
        return None

    for child_dir, cells in root_split:
        seg_id = len(segments)
        segments[0]["children"].append(seg_id)
        segments.append({"parent": 0, "dir": child_dir, "cells": cells, "children": []})
        occupied.update(cells)
        frontiers.append({"seg_id": seg_id, "node": cells[-1], "dir": child_dir, "depth": 1})

    target_segments = rng.randint(*TARGET_SEGMENT_RANGE)
    while frontiers and len(segments) < target_segments:
        idx = rng.randrange(len(frontiers))
        frontier = frontiers.pop(idx)
        action = _pick_action(int(frontier["depth"]), len(segments), target_segments, rng)
        if action == "stop":
            continue

        plans = _plan_children(
            frontier["node"],
            frontier["dir"],
            action,
            occupied,
            blocked,
            h,
            w,
            rng,
        )
        if plans is None:
            continue

        for child_dir, cells in plans:
            seg_id = len(segments)
            segments[int(frontier["seg_id"])]["children"].append(seg_id)
            segments.append(
                {
                    "parent": int(frontier["seg_id"]),
                    "dir": child_dir,
                    "cells": cells,
                    "children": [],
                }
            )
            occupied.update(cells)
            frontiers.append(
                {
                    "seg_id": seg_id,
                    "node": cells[-1],
                    "dir": child_dir,
                    "depth": int(frontier["depth"]) + 1,
                }
            )

    if len(segments) < 4:
        return None

    _assign_segment_colors(segments, marker_color, rng)
    grid = _paint_grid(h, w, segments, marker_cells, marker_color)

    path_mask = grid != 0
    for r, c in marker_cells:
        path_mask[r, c] = False

    if int(path_mask.sum()) != len(occupied):
        return None
    if _connected_size(set(occupied), root_start, h, w) != len(occupied):
        return None

    leaves = _leaf_cells(path_mask)
    if len(leaves) < 3:
        return None

    def leaf_key(p: tuple[int, int]) -> tuple[int, int, int]:
        d = min(abs(p[0] - mr) + abs(p[1] - mc) for mr, mc in marker_cells)
        return (int(d), int(p[0]), int(p[1]))

    leaf_order = sorted((leaf_key(p), p) for p in leaves)
    if leaf_order[0][1] != root_start:
        return None
    if len(leaf_order) > 1 and leaf_order[1][0][0] == leaf_order[0][0][0]:
        return None

    for r, c in occupied:
        deg = sum((rr, cc) in occupied for rr, cc in _neighbors4(r, c, h, w))
        if deg < 1 or deg > 3:
            return None

    out = _transform_grid(grid)
    if np.array_equal(out, grid):
        return None

    straight_changed = False
    turned_unchanged = False
    changed_segments = 0
    for seg_id, seg in enumerate(segments):
        changed = _segment_changed(grid, out, seg)
        if changed:
            changed_segments += 1
        parent = seg["parent"]
        if parent is None:
            continue
        if tuple(seg["dir"]) == tuple(segments[parent]["dir"]) and changed:
            straight_changed = True
        if tuple(seg["dir"]) != tuple(segments[parent]["dir"]) and not changed:
            turned_unchanged = True

    if changed_segments < 2 or not straight_changed or not turned_unchanged:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
