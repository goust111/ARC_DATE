from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIR_BY_MARKER = {
    RED: (1, 0),      # down
    GREEN: (0, -1),   # left
    YELLOW: (0, 1),   # right
    ORANGE: (-1, 0),  # up
}
MARKER_COLORS = tuple(DIR_BY_MARKER)

BASE_SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 2)),
]

GENERATE_ATTEMPTS = 500


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def normalize_shape(shape) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    out = sorted((r - min_r, c - min_c) for r, c in pts)
    return tuple(out)


def rotate_shape(shape) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    max_c = max(c for _, c in pts)
    return normalize_shape((c, max_c - r) for r, c in pts)


def flip_shape(shape) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    max_c = max(c for _, c in pts)
    return normalize_shape((r, max_c - c) for r, c in pts)


def random_shape(*, min_size: int | None = None, max_size: int | None = None) -> tuple[tuple[int, int], ...]:
    pool = []
    for base in BASE_SHAPES:
        area = len(base)
        if min_size is not None and area < min_size:
            continue
        if max_size is not None and area > max_size:
            continue
        pool.append(base)
    if not pool:
        raise ValueError("no shapes match requested size bounds")
    shape = tuple(random.choice(pool))
    if random.random() < 0.5:
        shape = flip_shape(shape)
    for _ in range(random.randrange(4)):
        shape = rotate_shape(shape)
    return normalize_shape(shape)


def shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in shape), 1 + max(c for _, c in shape)


def place_shape(shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in shape}


def bbox_from_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def blue_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLUE:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = {(r, c)}
            while stack:
                cr, cc = stack.pop()
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == BLUE:
                        seen[nr, nc] = True
                        comp.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def marker_counts(grid: np.ndarray, component: set[tuple[int, int]]) -> Counter:
    counts = Counter({RED: 0, GREEN: 0, YELLOW: 0, ORANGE: 0})
    for r, c in component:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc):
                color = int(grid[nr, nc])
                if color in DIR_BY_MARKER:
                    counts[color] += 1
    return counts


def component_direction(grid: np.ndarray, component: set[tuple[int, int]]) -> int:
    counts = marker_counts(grid, component)
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert top >= 1, "each component needs adjacent markers"
    assert len(winners) == 1, "marker majority must be unique"
    return winners[0]


def solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = blue_components(g)
    info = []
    for comp in components:
        direction_color = component_direction(g, comp)
        info.append(
            {
                "cells": comp,
                "area": len(comp),
                "bbox": bbox_from_cells(comp),
                "direction_color": direction_color,
                "direction": DIR_BY_MARKER[direction_color],
            }
        )
    info.sort(key=lambda item: (-item["area"], item["bbox"][0], item["bbox"][1]))

    out = g.copy()
    out[out == BLUE] = BG
    occupied = np.zeros_like(out, dtype=bool)
    occupied[out == GREY] = True

    traces: list[dict] = []
    for item in info:
        cells = sorted(item["cells"])
        dr, dc = item["direction"]
        shift = 0
        while True:
            moved = {(r + (shift + 1) * dr, c + (shift + 1) * dc) for r, c in cells}
            if all(in_bounds(out.shape, r, c) and not occupied[r, c] for r, c in moved):
                shift += 1
                continue
            break

        final_cells = {(r + shift * dr, c + shift * dc) for r, c in cells}
        stop_causes = set()
        for r, c in final_cells:
            nr, nc = r + dr, c + dc
            if not in_bounds(out.shape, nr, nc):
                stop_causes.add("edge")
            elif occupied[nr, nc]:
                stop_causes.add("deposit" if int(out[nr, nc]) == BLUE else "wall")
        for r, c in final_cells:
            out[r, c] = BLUE
            occupied[r, c] = True
        traces.append(
            {
                "area": item["area"],
                "direction_color": item["direction_color"],
                "shift": shift,
                "stop_causes": stop_causes,
                "start_bbox": item["bbox"],
                "final_bbox": bbox_from_cells(final_cells),
            }
        )

    marker_mask = np.isin(g, MARKER_COLORS)
    out[marker_mask] = g[marker_mask]
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let each blue component vote for its own gravity direction using adjacent marker colors, then slide that component until blocked.

    Input structure:
    - The grid has black background, 2-4 disjoint blue 4-connected components, isolated colored marker pixels, and grey wall pixels.
    - Marker colors encode directions: red means down, green means left, yellow means right, and orange means up.
    - Each marker is side-adjacent to exactly one blue component, and each blue component has a unique majority marker color.

    Rule:
    - For every blue component, count side-adjacent marker colors, where each shared edge contributes one vote.
    - The unique majority marker color chooses that component's gravity direction.
    - Process components from largest to smallest, breaking ties by the top-left corner of the input component bbox.
    - Remove the moving component from its start position, then shift the whole component one cell at a time in its chosen direction until the next step would leave the grid or hit a grey wall or an already-deposited blue component.
    - Keep grey walls and markers unchanged, and leave the component blue at its deposited position.

    Key insight:
    - The adjacency vote does not recolor a component; it chooses a per-component motion direction, and the processing order makes earlier deposits become later obstacles.
    """
    return solve_with_trace(grid)[0]


def interior_positions(limit: int) -> list[int]:
    if limit <= 1:
        return [0]
    if limit == 2:
        return [0, 1]
    return list(range(1, limit))


def build_wall_segment(shape: tuple[tuple[int, int], ...], final_anchor: tuple[int, int], direction_color: int) -> set[tuple[int, int]]:
    h, w = shape_dims(shape)
    fr, fc = final_anchor
    if direction_color == RED:
        return {(fr + h, fc + c) for c in range(w)}
    if direction_color == ORANGE:
        return {(fr - 1, fc + c) for c in range(w)}
    if direction_color == YELLOW:
        return {(fr + r, fc + w) for r in range(h)}
    return {(fr + r, fc - 1) for r in range(h)}


def sweep_bbox(start_cells: set[tuple[int, int]], final_cells: set[tuple[int, int]], extras: set[tuple[int, int]] | None = None):
    pts = set(start_cells) | set(final_cells)
    if extras:
        pts |= set(extras)
    return bbox_from_cells(pts)


def expanded_bbox(box: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def bbox_is_free(reserved: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any())


def reserve_bbox(reserved: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def plan_independent(shape: tuple[tuple[int, int], ...], direction_color: int, grid_shape: tuple[int, int], reserved: np.ndarray, must_use_wall: bool) -> dict | None:
    h, w = grid_shape
    sh, sw = shape_dims(shape)
    stop_kinds = ["wall", "edge"] if must_use_wall else random.sample(["wall", "edge"], k=2)
    for stop_kind in stop_kinds:
        for _ in range(80):
            if direction_color in (RED, ORANGE):
                anchor_c = random.choice(interior_positions(w - sw))
            else:
                anchor_c = None
            if direction_color in (GREEN, YELLOW):
                anchor_r = random.choice(interior_positions(h - sh))
            else:
                anchor_r = None

            if direction_color == RED:
                if stop_kind == "edge":
                    final_r = h - sh
                else:
                    choices = list(range(2, h - sh))
                    if not choices:
                        continue
                    final_r = random.choice(choices)
                max_dist = final_r - 1
                if max_dist < 2:
                    continue
                dist = random.randint(2, min(6, max_dist))
                start_anchor = (final_r - dist, anchor_c)
                final_anchor = (final_r, anchor_c)
            elif direction_color == ORANGE:
                if stop_kind == "edge":
                    final_r = 0
                else:
                    choices = list(range(1, h - sh - 1))
                    if not choices:
                        continue
                    final_r = random.choice(choices)
                max_dist = h - sh - 1 - final_r
                if max_dist < 2:
                    continue
                dist = random.randint(2, min(6, max_dist))
                start_anchor = (final_r + dist, anchor_c)
                final_anchor = (final_r, anchor_c)
            elif direction_color == YELLOW:
                if stop_kind == "edge":
                    final_c = w - sw
                else:
                    choices = list(range(2, w - sw))
                    if not choices:
                        continue
                    final_c = random.choice(choices)
                max_dist = final_c - 1
                if max_dist < 2:
                    continue
                dist = random.randint(2, min(6, max_dist))
                start_anchor = (anchor_r, final_c - dist)
                final_anchor = (anchor_r, final_c)
            else:
                if stop_kind == "edge":
                    final_c = 0
                else:
                    choices = list(range(1, w - sw - 1))
                    if not choices:
                        continue
                    final_c = random.choice(choices)
                max_dist = w - sw - 1 - final_c
                if max_dist < 2:
                    continue
                dist = random.randint(2, min(6, max_dist))
                start_anchor = (anchor_r, final_c + dist)
                final_anchor = (anchor_r, final_c)

            start_cells = place_shape(shape, start_anchor)
            final_cells = place_shape(shape, final_anchor)
            wall_cells = build_wall_segment(shape, final_anchor, direction_color) if stop_kind == "wall" else set()
            full_box = sweep_bbox(start_cells, final_cells, wall_cells)
            full_box = expanded_bbox(full_box, grid_shape, pad=1)
            if not bbox_is_free(reserved, full_box):
                continue
            return {
                "shape": shape,
                "direction_color": direction_color,
                "start_anchor": start_anchor,
                "final_anchor": final_anchor,
                "start_cells": start_cells,
                "final_cells": final_cells,
                "stop_kind": stop_kind,
                "wall_cells": wall_cells,
                "reserve_box": full_box,
            }
    return None


def plan_pair(direction_color: int, grid_shape: tuple[int, int], reserved: np.ndarray) -> list[dict] | None:
    for _ in range(150):
        leader_shape = random_shape(min_size=5)
        follower_shape = random_shape(max_size=max(4, len(leader_shape) - 1))
        if len(follower_shape) >= len(leader_shape):
            continue

        leader = plan_independent(leader_shape, direction_color, grid_shape, reserved, must_use_wall=random.random() < 0.7)
        if leader is None:
            continue

        lr0, lc0, lr1, lc1 = bbox_from_cells(leader["start_cells"])
        fh, fw = shape_dims(follower_shape)
        if direction_color == RED:
            follower_anchor = (lr0 - fh - random.randint(2, 4), leader["start_anchor"][1])
        elif direction_color == ORANGE:
            follower_anchor = (lr1 + 1 + random.randint(2, 4), leader["start_anchor"][1])
        elif direction_color == YELLOW:
            follower_anchor = (leader["start_anchor"][0], lc0 - fw - random.randint(2, 4))
        else:
            follower_anchor = (leader["start_anchor"][0], lc1 + 1 + random.randint(2, 4))

        follower_cells = place_shape(follower_shape, follower_anchor)
        if not all(in_bounds(grid_shape, r, c) for r, c in follower_cells):
            continue
        if follower_cells & leader["start_cells"]:
            continue

        combined_box = sweep_bbox(
            leader["start_cells"] | follower_cells,
            leader["final_cells"],
            leader["wall_cells"],
        )
        combined_box = expanded_bbox(combined_box, grid_shape, pad=1)
        if not bbox_is_free(reserved, combined_box):
            continue

        leader["pair_role"] = "leader"
        leader["reserve_box"] = combined_box
        follower = {
            "shape": follower_shape,
            "direction_color": direction_color,
            "start_anchor": follower_anchor,
            "start_cells": follower_cells,
            "stop_kind": "deposit",
            "wall_cells": set(),
            "pair_role": "follower",
            "reserve_box": combined_box,
        }
        return [leader, follower]
    return None


def add_component_markers(grid: np.ndarray, plan: dict) -> bool:
    comp = set(plan["start_cells"])
    direction_color = plan["direction_color"]
    r0, c0, r1, c1 = bbox_from_cells(comp)
    candidates = set()
    for r, c in comp:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) != BG:
                continue
            if direction_color == RED and nr > r1:
                continue
            if direction_color == ORANGE and nr < r0:
                continue
            if direction_color == GREEN and nc < c0:
                continue
            if direction_color == YELLOW and nc > c1:
                continue
            blue_neighbors = {(ar, ac) for ar, ac in neighbors4(nr, nc) if in_bounds(grid.shape, ar, ac) and int(grid[ar, ac]) == BLUE}
            if len(blue_neighbors) != 1 or not blue_neighbors <= comp:
                continue
            ok = True
            for ar, ac in neighbors4(nr, nc):
                if not in_bounds(grid.shape, ar, ac):
                    continue
                val = int(grid[ar, ac])
                if val in MARKER_COLORS or val == GREY:
                    ok = False
                    break
                if val == BLUE and (ar, ac) not in comp:
                    ok = False
                    break
            if ok:
                candidates.add((nr, nc))
    candidates = list(candidates)
    if len(candidates) < 2:
        return False

    random.shuffle(candidates)
    total = min(len(candidates), random.randint(2, min(4, len(candidates))))
    if total == 4:
        majority_count = 3
    elif total == 3:
        majority_count = 2 if random.random() < 0.8 else 3
    else:
        majority_count = 2
    chosen = candidates[:total]
    majority_cells = chosen[:majority_count]
    minority_cells = chosen[majority_count:]
    for r, c in majority_cells:
        grid[r, c] = direction_color
    other_colors = [color for color in MARKER_COLORS if color != direction_color]
    for r, c in minority_cells:
        grid[r, c] = random.choice(other_colors)
    return True


def build_grid_from_plans(grid_shape: tuple[int, int], plans: list[dict]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for plan in plans:
        for r, c in plan["wall_cells"]:
            grid[r, c] = GREY
    for plan in plans:
        for r, c in plan["start_cells"]:
            grid[r, c] = BLUE
    for plan in plans:
        if not add_component_markers(grid, plan):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        h = random.randint(14, 18)
        w = random.randint(14, 18)
        grid_shape = (h, w)
        reserved = np.zeros(grid_shape, dtype=bool)
        plans: list[dict] = []

        n_components = random.randint(2, 4)
        use_pair = n_components >= 2 and random.random() < 0.6
        if use_pair:
            pair_direction = random.choice(MARKER_COLORS)
            pair_plans = plan_pair(pair_direction, grid_shape, reserved)
            if pair_plans is None:
                continue
            reserve_bbox(reserved, pair_plans[0]["reserve_box"])
            plans.extend(pair_plans)

        remaining = n_components - len(plans)
        target_dirs = []
        palette = list(MARKER_COLORS)
        random.shuffle(palette)
        if remaining > 0:
            while len(target_dirs) < remaining:
                target_dirs.append(palette[len(target_dirs) % len(palette)])
            random.shuffle(target_dirs)

        wall_used = any(plan["wall_cells"] for plan in plans)
        ok = True
        for i in range(remaining):
            shape = random_shape()
            direction_color = target_dirs[i]
            plan = plan_independent(shape, direction_color, grid_shape, reserved, must_use_wall=not wall_used)
            if plan is None:
                ok = False
                break
            reserve_bbox(reserved, plan["reserve_box"])
            wall_used = wall_used or bool(plan["wall_cells"])
            plans.append(plan)
        if not ok or not wall_used:
            continue

        grid = build_grid_from_plans(grid_shape, plans)
        if grid is None:
            continue

        traces_out = solve_with_trace(grid)
        out, traces = traces_out
        if any(trace["shift"] < 1 for trace in traces):
            continue
        if np.array_equal(out, grid):
            continue
        if len({trace["direction_color"] for trace in traces}) < min(2, len(traces)):
            continue
        if use_pair and not any("deposit" in trace["stop_causes"] for trace in traces):
            continue
        if np.count_nonzero(grid == GREY) < 3:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
