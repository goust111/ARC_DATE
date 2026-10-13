from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


GREY = 5
MAROON = 9
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_BOUNCES = 2
MAX_GENERATE_ATTEMPTS = 400


@dataclass
class Segment:
    cells: tuple[tuple[int, int], ...]
    orientation: str
    endpoints: tuple[tuple[int, int], tuple[int, int]]
    marker_pos: tuple[int, int]
    marker_color: int
    towers: dict[tuple[int, int], int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(mask.shape, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_receivers(grid: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], np.ndarray]:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    tops: list[tuple[int, int]] = []
    receiver_by_cell: dict[tuple[int, int], int] = {}

    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r : r + 2, c : c + 2] == MAROON):
                block = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                if any(mask[rr, cc] for rr, cc in block):
                    continue
                rid = len(tops)
                tops.append((r, c))
                for rr, cc in block:
                    mask[rr, cc] = True
                    receiver_by_cell[(rr, cc)] = rid

    assert np.array_equal(grid == MAROON, mask), "maroon pixels must form disjoint 2x2 receivers"
    return tops, receiver_by_cell, mask


def _build_segment(grid: np.ndarray, coords: list[tuple[int, int]]) -> Segment:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    assert len(rows) == 1 or len(cols) == 1, "wall components must be straight segments"

    if len(rows) == 1:
        orientation = "horizontal"
        ordered = tuple(sorted(coords, key=lambda rc: rc[1]))
    else:
        orientation = "vertical"
        ordered = tuple(sorted(coords, key=lambda rc: rc[0]))

    assert len(ordered) >= 3, "segments need interior space for one marker"
    endpoints = (ordered[0], ordered[-1])
    interior = ordered[1:-1]

    marker_cells = [cell for cell in interior if int(grid[cell]) != GREY]
    assert len(marker_cells) == 1, "each wall segment needs exactly one interior marker"
    marker_pos = marker_cells[0]
    marker_color = int(grid[marker_pos])

    for cell in interior:
        if cell != marker_pos:
            assert int(grid[cell]) == GREY, "only one interior wall cell may be colored"

    towers: dict[tuple[int, int], int] = {}
    for endpoint in endpoints:
        value = int(grid[endpoint])
        if value != GREY:
            towers[endpoint] = value

    return Segment(
        cells=ordered,
        orientation=orientation,
        endpoints=endpoints,
        marker_pos=marker_pos,
        marker_color=marker_color,
        towers=towers,
    )


def _parse_scene(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], dict[tuple[int, int], int], list[Segment], dict[tuple[int, int], Segment]]:
    bg = _most_common_color(grid)
    receivers, receiver_by_cell, receiver_mask = _find_receivers(grid)
    wall_mask = (grid != bg) & ~receiver_mask
    components = _components(wall_mask)
    assert components, "expected at least one wall segment"

    segments: list[Segment] = []
    cell_to_segment: dict[tuple[int, int], Segment] = {}
    for coords in components:
        seg = _build_segment(grid, coords)
        segments.append(seg)
        for cell in seg.cells:
            cell_to_segment[cell] = seg

    return bg, receivers, receiver_by_cell, segments, cell_to_segment


def _same_axis(direction: tuple[int, int], orientation: str) -> bool:
    dr, dc = direction
    return (orientation == "horizontal" and dc != 0) or (orientation == "vertical" and dr != 0)


def _turn_toward_marker(segment: Segment, current_pos: tuple[int, int]) -> tuple[int, int]:
    r, c = current_pos
    if segment.orientation == "vertical":
        marker_r = segment.marker_pos[0]
        if marker_r < r:
            return (-1, 0)
        if marker_r > r:
            return (1, 0)
        top_r = segment.endpoints[0][0]
        bottom_r = segment.endpoints[1][0]
        return (-1, 0) if (r - top_r) >= (bottom_r - r) else (1, 0)

    marker_c = segment.marker_pos[1]
    if marker_c < c:
        return (0, -1)
    if marker_c > c:
        return (0, 1)
    left_c = segment.endpoints[0][1]
    right_c = segment.endpoints[1][1]
    return (0, -1) if (c - left_c) >= (right_c - c) else (0, 1)


def _trace_ray(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    start_color: int,
    cell_to_segment: dict[tuple[int, int], Segment],
) -> list[tuple[int, int, int, int]]:
    r, c = start
    dr, dc = direction
    color = start_color
    bounces = 0
    events: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc, bounces)
        if state in seen:
            break
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            break

        hit = cell_to_segment.get((nr, nc))
        if hit is not None:
            if _same_axis((dr, dc), hit.orientation):
                break
            bounces += 1
            if bounces > MAX_BOUNCES:
                break
            color = hit.marker_color
            dr, dc = _turn_toward_marker(hit, (r, c))
            continue

        r, c = nr, nc
        events.append((r, c, bounces, color))

    return events


def _trace_all_rays(grid: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int, int]]], dict[int, tuple[int, int]]]:
    _bg, receivers, receiver_by_cell, segments, cell_to_segment = _parse_scene(grid)
    towers: list[tuple[tuple[int, int], int]] = []
    for seg in segments:
        for pos, color in seg.towers.items():
            towers.append((pos, color))
    towers.sort(key=lambda item: (item[0][0], item[0][1], item[1]))
    assert towers, "expected at least one tower"

    cell_hits: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    receiver_hits: dict[int, tuple[int, int]] = {}

    ray_id = 0
    for start, tower_color in towers:
        for direction in DIRS:
            events = _trace_ray(grid.shape, start, direction, tower_color, cell_to_segment)
            for r, c, bounces, color in events:
                cell_hits.setdefault((r, c), []).append((ray_id, bounces, color))
                rid = receiver_by_cell.get((r, c))
                if rid is None:
                    continue
                prev = receiver_hits.get(rid)
                if prev is None or bounces < prev[0]:
                    receiver_hits[rid] = (bounces, color)
            ray_id += 1

    return receivers, cell_hits, receiver_hits


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    receivers, _cell_hits, receiver_hits = _trace_all_rays(g)
    out = g.copy()
    for rid, (r, c) in enumerate(receivers):
        if rid in receiver_hits:
            out[r : r + 2, c : c + 2] = receiver_hits[rid][1]
    return out


def _add_dir(a: tuple[int, int], b: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * scale, a[1] + b[1] * scale


def _perpendicular_directions(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _segment_adjacency(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in cells:
        for dr, dc in DIRS:
            expanded.add((r + dr, c + dc))
    return expanded


def _segment_ok(cells: list[tuple[int, int]], wall_cells: set[tuple[int, int]], path_cells: set[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    if cell_set & wall_cells:
        return False
    if cell_set & path_cells:
        return False
    for r, c in cell_set:
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (nr, nc) in wall_cells:
                return False
    return True


def _build_stub(direction: tuple[int, int], tower_color: int, marker_color: int) -> dict:
    length = random.randint(4, 6)
    opposite = (-direction[0], -direction[1])
    cells = [_add_dir((0, 0), opposite, step) for step in range(length)]
    marker_index = random.randint(1, length - 2)
    return {
        "cells": cells,
        "tower": ((0, 0), tower_color),
        "marker": (cells[marker_index], marker_color),
    }


def _build_bounce_wall(hit_cell: tuple[int, int], next_dir: tuple[int, int], marker_color: int) -> dict:
    major = random.randint(2, 4)
    minor = random.randint(1, 3)

    if next_dir[0] != 0:
        low = minor if next_dir[0] > 0 else major
        high = major if next_dir[0] > 0 else minor
        cells = [(hit_cell[0] + delta, hit_cell[1]) for delta in range(-low, high + 1)]
        marker_offset = random.randint(1, major - 1)
        marker = (hit_cell[0] + next_dir[0] * marker_offset, hit_cell[1])
    else:
        low = minor if next_dir[1] > 0 else major
        high = major if next_dir[1] > 0 else minor
        cells = [(hit_cell[0], hit_cell[1] + delta) for delta in range(-low, high + 1)]
        marker_offset = random.randint(1, major - 1)
        marker = (hit_cell[0], hit_cell[1] + next_dir[1] * marker_offset)

    return {"cells": cells, "tower": None, "marker": (marker, marker_color)}


def _build_relative_plan(bounce_count: int, tower_color: int, marker_colors: list[int]) -> dict | None:
    for _ in range(120):
        direction = random.choice(DIRS)
        segments: list[dict] = []
        wall_cells: set[tuple[int, int]] = set()
        path_cells: set[tuple[int, int]] = set()

        stub = _build_stub(direction, tower_color, marker_colors[0])
        if not _segment_ok(stub["cells"], wall_cells, path_cells):
            continue
        segments.append(stub)
        wall_cells.update(stub["cells"])

        current = (0, 0)
        current_dir = direction
        ok = True

        for bounce_index in range(bounce_count + 1):
            leg_length = random.randint(4, 8)
            leg_cells = [_add_dir(current, current_dir, step) for step in range(1, leg_length + 1)]
            if any(cell in wall_cells or cell in path_cells for cell in leg_cells):
                ok = False
                break
            path_cells.update(leg_cells)
            current = leg_cells[-1]

            if bounce_index == bounce_count:
                continue

            next_dir = random.choice(_perpendicular_directions(current_dir))
            hit_cell = _add_dir(current, current_dir)
            wall = _build_bounce_wall(hit_cell, next_dir, marker_colors[bounce_index + 1])
            if not _segment_ok(wall["cells"], wall_cells, path_cells):
                ok = False
                break
            segments.append(wall)
            wall_cells.update(wall["cells"])
            current_dir = next_dir

        if not ok:
            continue

        all_cells = wall_cells | path_cells
        return {"segments": segments, "wall_cells": wall_cells, "path_cells": path_cells, "all_cells": all_cells}

    return None


def _transform_plan(plan: dict, offset: tuple[int, int]) -> dict:
    dr, dc = offset
    segments = []
    for seg in plan["segments"]:
        new_seg = {
            "cells": [(r + dr, c + dc) for r, c in seg["cells"]],
            "tower": None,
            "marker": ((seg["marker"][0][0] + dr, seg["marker"][0][1] + dc), seg["marker"][1]),
        }
        if seg["tower"] is not None:
            new_seg["tower"] = ((seg["tower"][0][0] + dr, seg["tower"][0][1] + dc), seg["tower"][1])
        segments.append(new_seg)
    return {
        "segments": segments,
        "wall_cells": {(r + dr, c + dc) for r, c in plan["wall_cells"]},
        "path_cells": {(r + dr, c + dc) for r, c in plan["path_cells"]},
        "all_cells": {(r + dr, c + dc) for r, c in plan["all_cells"]},
    }


def _place_plan(
    shape: tuple[int, int],
    plan: dict,
    global_walls: set[tuple[int, int]],
    global_wall_adj: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
) -> dict | None:
    rows = [r for r, _ in plan["all_cells"]]
    cols = [c for _, c in plan["all_cells"]]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    margin = 2
    r_lo = margin - min_r
    r_hi = shape[0] - margin - 1 - max_r
    c_lo = margin - min_c
    c_hi = shape[1] - margin - 1 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(80):
        offset = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
        placed = _transform_plan(plan, offset)
        if placed["path_cells"] & global_walls:
            continue
        if placed["wall_cells"] & reserved_paths:
            continue
        if any(cell in global_wall_adj for cell in placed["wall_cells"]):
            continue
        return placed

    return None


def _render_base_grid(shape: tuple[int, int], bg: int, placed_plans: list[dict]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for plan in placed_plans:
        for seg in plan["segments"]:
            for r, c in seg["cells"]:
                grid[r, c] = GREY
    for plan in placed_plans:
        for seg in plan["segments"]:
            marker_pos, marker_color = seg["marker"]
            grid[marker_pos] = marker_color
            if seg["tower"] is not None:
                tower_pos, tower_color = seg["tower"]
                grid[tower_pos] = tower_color
    return grid


def _receiver_block(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = top_left
    return [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]


def _receiver_guard(top_left: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = top_left
    cells = set()
    for rr in range(r - 1, r + 3):
        for cc in range(c - 1, c + 3):
            if _in_bounds(shape, rr, cc):
                cells.add((rr, cc))
    return cells


def _collect_candidates(grid: np.ndarray, cell_hits: dict[tuple[int, int], list[tuple[int, int, int]]]) -> tuple[list[dict], list[tuple[int, int]]]:
    bg = _most_common_color(grid)
    h, w = grid.shape
    hit_candidates: list[dict] = []
    unhit_candidates: list[tuple[int, int]] = []

    for r in range(h - 1):
        for c in range(w - 1):
            block = _receiver_block((r, c))
            if any(int(grid[cell]) != bg for cell in block):
                continue

            by_ray: dict[int, tuple[int, int]] = {}
            for cell in block:
                for ray_id, bounces, color in cell_hits.get(cell, []):
                    prev = by_ray.get(ray_id)
                    if prev is None or bounces < prev[0]:
                        by_ray[ray_id] = (bounces, color)

            if not by_ray:
                unhit_candidates.append((r, c))
                continue

            best_bounce = min(value[0] for value in by_ray.values())
            best_colors = {value[1] for value in by_ray.values() if value[0] == best_bounce}
            if len(best_colors) != 1:
                continue

            hit_candidates.append(
                {
                    "top_left": (r, c),
                    "best_bounce": best_bounce,
                    "color": next(iter(best_colors)),
                    "priority": any(value[0] > best_bounce for value in by_ray.values()),
                    "ray_count": len(by_ray),
                }
            )

    return hit_candidates, unhit_candidates


def _pick_receivers(shape: tuple[int, int], candidates: list[dict], unhit_candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    chosen: list[tuple[int, int]] = []
    prefer_priority = random.choice([0, 1, 2])

    for bounce in [0, 1, 2]:
        eligible = [
            cand
            for cand in candidates
            if cand["best_bounce"] == bounce
            and not (_receiver_guard(cand["top_left"], shape) & blocked)
        ]
        if not eligible:
            return []
        random.shuffle(eligible)
        eligible.sort(key=lambda cand: (bounce == prefer_priority and not cand["priority"], random.random()))
        top_left = eligible[0]["top_left"]
        chosen.append(top_left)
        blocked |= _receiver_guard(top_left, shape)

    priority_free = [
        cand
        for cand in candidates
        if cand["priority"] and not (_receiver_guard(cand["top_left"], shape) & blocked)
    ]
    if priority_free and random.random() < 0.55:
        random.shuffle(priority_free)
        top_left = priority_free[0]["top_left"]
        chosen.append(top_left)
        blocked |= _receiver_guard(top_left, shape)

    free_unhit = [pos for pos in unhit_candidates if not (_receiver_guard(pos, shape) & blocked)]
    if free_unhit and random.random() < 0.35:
        top_left = random.choice(free_unhit)
        chosen.append(top_left)

    return chosen


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8])
    palette = [color for color in range(10) if color not in {bg, GREY, MAROON}]
    random.shuffle(palette)

    active_bounces = [0, 1, 2]
    if random.random() < 0.6:
        active_bounces.append(random.choice([0, 1, 2]))

    tower_colors = palette[: len(active_bounces)]
    remaining = [color for color in palette if color not in tower_colors]
    if len(remaining) < 2:
        return None

    wall_cells: set[tuple[int, int]] = set()
    wall_adj: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    placed_plans: list[dict] = []

    for index, bounce_count in enumerate(active_bounces):
        marker_pool = [color for color in remaining if color != tower_colors[index]]
        if not marker_pool:
            marker_pool = remaining[:]
        marker_colors = [random.choice(marker_pool) for _ in range(bounce_count + 1)]
        plan = _build_relative_plan(bounce_count, tower_colors[index], marker_colors)
        if plan is None:
            return None
        placed = _place_plan((h, w), plan, wall_cells, wall_adj, reserved_paths)
        if placed is None:
            return None
        placed_plans.append(placed)
        wall_cells |= placed["wall_cells"]
        reserved_paths |= placed["path_cells"]
        wall_adj |= _segment_adjacency(placed["wall_cells"])

    grid = _render_base_grid((h, w), bg, placed_plans)
    _receivers, cell_hits, _receiver_hits = _trace_all_rays(grid)
    candidates, unhit_candidates = _collect_candidates(grid, cell_hits)
    if not candidates:
        return None

    chosen_receivers = _pick_receivers((h, w), candidates, unhit_candidates)
    if len(chosen_receivers) < 3:
        return None

    input_grid = grid.copy()
    for top_left in chosen_receivers:
        for cell in _receiver_block(top_left):
            input_grid[cell] = MAROON

    output_grid = main(input_grid)
    if np.array_equal(output_grid, input_grid):
        return None

    receiver_colors = []
    changed = 0
    for top_left in chosen_receivers:
        block_in = input_grid[top_left[0] : top_left[0] + 2, top_left[1] : top_left[1] + 2]
        block_out = output_grid[top_left[0] : top_left[0] + 2, top_left[1] : top_left[1] + 2]
        if not np.all(block_in == MAROON):
            return None
        receiver_colors.append(int(block_out[0, 0]))
        if not np.all(block_out == block_out[0, 0]):
            return None
        if int(block_out[0, 0]) != MAROON:
            changed += 1

    if changed < 3:
        return None
    if len({color for color in receiver_colors if color != MAROON}) < 2:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
