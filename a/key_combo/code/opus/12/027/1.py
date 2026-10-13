from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME = 1
OBSTACLE = 2
FILL = 3

FRAME_SIZE = 4
FRAME_INTERIOR = 2
GAP_OFFSETS = (1, 2)
SIDES = ("top", "bottom", "left", "right")
DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 600
FRAME_ATTEMPTS = 220
PATH_ATTEMPTS = 90
OUTWARD_MARGIN = 7
SIDE_MARGIN = 2


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def bbox_coords(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def expand_cells(
    cells: set[tuple[int, int]], shape: tuple[int, int], radius: int
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    h, w = map(int, shape)
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def frame_blue_cells(
    r0: int, c0: int, side: str, gap_offset: int
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]:
    r1 = r0 + FRAME_SIZE - 1
    c1 = c0 + FRAME_SIZE - 1
    bbox_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if side == "top":
        gap = (r0, c0 + gap_offset)
    elif side == "bottom":
        gap = (r1, c0 + gap_offset)
    elif side == "left":
        gap = (r0 + gap_offset, c0)
    else:
        gap = (r0 + gap_offset, c1)
    blue = border - {gap}
    interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    return blue, gap, DIR_BY_SIDE[side], interior, bbox_cells


def detect_frames(grid: np.ndarray) -> list[dict]:
    comps = component_sets(grid == FRAME)
    frames: list[dict] = []
    for comp in comps:
        assert len(comp) == 11, "each blue object must be a 4x4 one-gap frame"
        r0, c0, r1, c1 = bbox_coords(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (4, 4), "frame bbox must be 4x4"
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        gaps = sorted(border - comp)
        assert len(gaps) == 1, "frame must have exactly one gap"
        gap = gaps[0]
        gr, gc = gap
        if gr == r0:
            assert c0 < gc < c1
            side = "top"
        elif gr == r1:
            assert c0 < gc < c1
            side = "bottom"
        elif gc == c0:
            assert r0 < gr < r1
            side = "left"
        else:
            assert gc == c1 and r0 < gr < r1
            side = "right"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "dir": DIR_BY_SIDE[side],
                "side": side,
                "cells": comp,
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    assert 2 <= len(frames) <= 3, "expected 2-3 frames"
    return frames


def obstacle_segment(
    grid: np.ndarray, hit: tuple[int, int], direction: tuple[int, int]
) -> dict:
    h, w = map(int, grid.shape)
    hr, hc = hit
    dr, dc = direction
    assert grid[hr, hc] == OBSTACLE
    if dr != 0:
        left = hc
        right = hc
        while left - 1 >= 0 and grid[hr, left - 1] == OBSTACLE:
            left -= 1
        while right + 1 < w and grid[hr, right + 1] == OBSTACLE:
            right += 1
        return {
            "axis": "horizontal",
            "row": hr,
            "left": left,
            "right": right,
            "length": right - left + 1,
        }
    top = hr
    bottom = hr
    while top - 1 >= 0 and grid[top - 1, hc] == OBSTACLE:
        top -= 1
    while bottom + 1 < h and grid[bottom + 1, hc] == OBSTACLE:
        bottom += 1
    return {
        "axis": "vertical",
        "col": hc,
        "top": top,
        "bottom": bottom,
        "length": bottom - top + 1,
    }


def trace_ray(
    grid: np.ndarray, gap: tuple[int, int], direction: tuple[int, int]
) -> tuple[set[tuple[int, int]], int]:
    h, w = map(int, grid.shape)
    dr, dc = direction
    r = gap[0] + dr
    c = gap[1] + dc
    path: set[tuple[int, int]] = set()
    hits = 0
    seen_turns: set[tuple[int, int]] = set()

    while in_bounds((h, w), r, c):
        assert grid[r, c] != FRAME and grid[r, c] != OBSTACLE
        path.add((r, c))
        nr = r + dr
        nc = c + dc
        if not in_bounds((h, w), nr, nc):
            break
        if grid[nr, nc] != OBSTACLE:
            r, c = nr, nc
            continue

        hits += 1
        seg = obstacle_segment(grid, (nr, nc), direction)
        if dr != 0:
            options: list[tuple[int, int, int]] = []
            if seg["left"] > 0:
                left_target = seg["left"] - 1
                options.append((abs(c - left_target), left_target, -1))
            if seg["right"] < w - 1:
                right_target = seg["right"] + 1
                options.append((abs(c - right_target), right_target, 1))
            assert options
            _, target_c, step = min(options, key=lambda item: (item[0], item[1]))
            for cc in range(c + step, target_c + step, step):
                assert grid[r, cc] != FRAME and grid[r, cc] != OBSTACLE
                path.add((r, cc))
            c = target_c
        else:
            options = []
            if seg["top"] > 0:
                top_target = seg["top"] - 1
                options.append((abs(r - top_target), top_target, -1))
            if seg["bottom"] < h - 1:
                bottom_target = seg["bottom"] + 1
                options.append((abs(r - bottom_target), bottom_target, 1))
            assert options
            _, target_r, step = min(options, key=lambda item: (item[0], item[1]))
            for rr in range(r + step, target_r + step, step):
                assert grid[rr, c] != FRAME and grid[rr, c] != OBSTACLE
                path.add((rr, c))
            r = target_r

        state = (r, c)
        assert state not in seen_turns, "detour loop detected"
        seen_turns.add(state)

    return path, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each one-gap blue frame and route a green ray around red blocker segments.

    Input structure:
    - The background is black, with 2-3 blue 4x4 rectangular frame borders that each miss one non-corner border pixel.
    - The missing pixel is the gap; red pixels form separate horizontal or vertical obstacle segments away from the frames.
    - Each frame's 2x2 interior is empty, and the gap direction points toward at least one obstacle segment before the grid edge.

    Rule:
    - Keep all blue frame pixels and all red obstacle pixels unchanged.
    - For each frame, paint its 2x2 interior and its gap pixel green.
    - Start from the cell just outside the gap and extend a green ray in the gap's outward direction.
    - When the next step would enter a red obstacle segment, detour on the row or column just before that segment toward the nearest outside end of the segment, using the smaller row or column on ties, then continue in the original direction until the boundary; repeat for later obstacles.

    Key insight:
    - The gap fixes the ray direction, while each encountered perpendicular obstacle segment shifts the ray sideways to the nearest free end before the ray resumes unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    frames = detect_frames(g)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                assert g[r, c] == BG
                out[r, c] = FILL
        gr, gc = frame["gap"]
        assert g[gr, gc] == BG
        out[gr, gc] = FILL
        path, _ = trace_ray(g, frame["gap"], frame["dir"])
        for r, c in path:
            assert g[r, c] == BG
            out[r, c] = FILL

    return out


def min_chebyshev_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            best = min(best, max(abs(r0 - r1), abs(c0 - c1)))
    return best


def sample_frame_top_left(h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        r0 = random.randint(OUTWARD_MARGIN, h - FRAME_SIZE - SIDE_MARGIN)
        c0 = random.randint(SIDE_MARGIN, w - FRAME_SIZE - SIDE_MARGIN)
    elif side == "bottom":
        r0 = random.randint(SIDE_MARGIN, h - FRAME_SIZE - OUTWARD_MARGIN)
        c0 = random.randint(SIDE_MARGIN, w - FRAME_SIZE - SIDE_MARGIN)
    elif side == "left":
        r0 = random.randint(SIDE_MARGIN, h - FRAME_SIZE - SIDE_MARGIN)
        c0 = random.randint(OUTWARD_MARGIN, w - FRAME_SIZE - SIDE_MARGIN)
    else:
        r0 = random.randint(SIDE_MARGIN, h - FRAME_SIZE - SIDE_MARGIN)
        c0 = random.randint(SIDE_MARGIN, w - FRAME_SIZE - OUTWARD_MARGIN)
    return r0, c0


def boundary_path_cells(
    start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = direction
    while in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def obstacle_options(
    shape: tuple[int, int],
    current: tuple[int, int],
    hit: tuple[int, int],
    direction: tuple[int, int],
    input_occ: set[tuple[int, int]],
    path_occ: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
    local_obstacles: set[tuple[int, int]],
) -> list[tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int], bool]]:
    h, w = map(int, shape)
    cur_r, cur_c = current
    hit_r, hit_c = hit
    dr, dc = direction
    blocked_input = expand_cells(input_occ | local_obstacles, shape, 1)
    results: list[tuple[set[tuple[int, int]], list[tuple[int, int]], tuple[int, int], bool]] = []

    if dr != 0:
        for left in range(max(0, hit_c - 5), hit_c + 1):
            for right in range(hit_c, min(w - 1, hit_c + 5) + 1):
                if not (3 <= right - left + 1 <= 6):
                    continue
                seg_cells = {(hit_r, cc) for cc in range(left, right + 1)}
                if seg_cells & blocked_input:
                    continue
                if seg_cells & (path_occ | local_path):
                    continue
                options = []
                if left > 0 and (hit_r, left - 1) not in input_occ | local_obstacles | path_occ | local_path:
                    options.append((abs(cur_c - (left - 1)), left - 1, -1))
                if right < w - 1 and (hit_r, right + 1) not in input_occ | local_obstacles | path_occ | local_path:
                    options.append((abs(cur_c - (right + 1)), right + 1, 1))
                if not options:
                    continue
                tie = len(options) == 2 and options[0][0] == options[1][0]
                _, target_c, step = min(options, key=lambda item: (item[0], item[1]))
                detour = [(cur_r, cc) for cc in range(cur_c + step, target_c + step, step)]
                detour_set = set(detour)
                if detour_set & (input_occ | local_obstacles | path_occ | local_path):
                    continue
                next_cell = (cur_r + dr, target_c)
                if not in_bounds(shape, *next_cell):
                    continue
                if next_cell in input_occ | local_obstacles | path_occ | local_path:
                    continue
                results.append((seg_cells, detour, next_cell, tie))
    else:
        for top in range(max(0, hit_r - 5), hit_r + 1):
            for bottom in range(hit_r, min(h - 1, hit_r + 5) + 1):
                if not (3 <= bottom - top + 1 <= 6):
                    continue
                seg_cells = {(rr, hit_c) for rr in range(top, bottom + 1)}
                if seg_cells & blocked_input:
                    continue
                if seg_cells & (path_occ | local_path):
                    continue
                options = []
                if top > 0 and (top - 1, hit_c) not in input_occ | local_obstacles | path_occ | local_path:
                    options.append((abs(cur_r - (top - 1)), top - 1, -1))
                if bottom < h - 1 and (bottom + 1, hit_c) not in input_occ | local_obstacles | path_occ | local_path:
                    options.append((abs(cur_r - (bottom + 1)), bottom + 1, 1))
                if not options:
                    continue
                tie = len(options) == 2 and options[0][0] == options[1][0]
                _, target_r, step = min(options, key=lambda item: (item[0], item[1]))
                detour = [(rr, cur_c) for rr in range(cur_r + step, target_r + step, step)]
                detour_set = set(detour)
                if detour_set & (input_occ | local_obstacles | path_occ | local_path):
                    continue
                next_cell = (target_r, cur_c + dc)
                if not in_bounds(shape, *next_cell):
                    continue
                if next_cell in input_occ | local_obstacles | path_occ | local_path:
                    continue
                results.append((seg_cells, detour, next_cell, tie))
    return results


def plan_ray_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    obstacle_count: int,
    input_occ: set[tuple[int, int]],
    path_occ: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], list[set[tuple[int, int]]]] | None:
    for _ in range(PATH_ATTEMPTS):
        cursor = start
        local_path: set[tuple[int, int]] = set()
        obstacles: list[set[tuple[int, int]]] = []
        local_obstacles: set[tuple[int, int]] = set()
        tie_needed = random.random() < 0.35
        tie_seen = False
        ok = True

        for idx in range(obstacle_count):
            placed = False
            for _try in range(60):
                straight_len = random.randint(1, 4)
                straight = [
                    (cursor[0] + t * direction[0], cursor[1] + t * direction[1])
                    for t in range(straight_len)
                ]
                if not all(in_bounds(shape, r, c) for r, c in straight):
                    continue
                straight_set = set(straight)
                if straight_set & (input_occ | path_occ | local_path | local_obstacles):
                    continue
                current = straight[-1]
                hit = (current[0] + direction[0], current[1] + direction[1])
                if not in_bounds(shape, *hit):
                    continue
                opts = obstacle_options(
                    shape, current, hit, direction, input_occ, path_occ, local_path | straight_set, local_obstacles
                )
                if not opts:
                    continue
                if tie_needed and not tie_seen:
                    tie_opts = [item for item in opts if item[3]]
                    if tie_opts:
                        opts = tie_opts
                seg_cells, detour, next_cell, tie = random.choice(opts)
                if idx < obstacle_count - 1 and not in_bounds(shape, *next_cell):
                    continue
                local_path |= straight_set
                local_path |= set(detour)
                obstacles.append(seg_cells)
                local_obstacles |= seg_cells
                cursor = next_cell
                tie_seen = tie_seen or tie
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        final_cells = boundary_path_cells(cursor, direction, shape)
        final_set = set(final_cells)
        if not final_cells:
            continue
        if final_set & (input_occ | path_occ | local_path | local_obstacles):
            continue
        local_path |= final_set
        if tie_needed and not tie_seen:
            continue
        return local_path, obstacles
    return None


def validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, FRAME, OBSTACLE}:
        return False

    try:
        frames = detect_frames(grid)
    except AssertionError:
        return False

    obstacle_comps = component_sets(grid == OBSTACLE)
    if not (2 <= len(obstacle_comps) <= 4):
        return False
    for comp in obstacle_comps:
        r0, c0, r1, c1 = bbox_coords(comp)
        if not (r0 == r1 or c0 == c1):
            return False
        if len(comp) != max(r1 - r0 + 1, c1 - c0 + 1):
            return False
        if not (3 <= len(comp) <= 6):
            return False

    frame_cells = [frame["cells"] for frame in frames]
    frame_boxes = [
        {(r, c) for r in range(frame["bbox"][0], frame["bbox"][2] + 1) for c in range(frame["bbox"][1], frame["bbox"][3] + 1)}
        for frame in frames
    ]
    for i in range(len(frame_boxes)):
        for j in range(i + 1, len(frame_boxes)):
            if min_chebyshev_distance(frame_boxes[i], frame_boxes[j]) < 4:
                return False

    all_obstacles = [set(comp) for comp in obstacle_comps]
    for frame_box in frame_boxes:
        for segment in all_obstacles:
            if min_chebyshev_distance(frame_box, segment) < 2:
                return False
    for i in range(len(all_obstacles)):
        for j in range(i + 1, len(all_obstacles)):
            if min_chebyshev_distance(all_obstacles[i], all_obstacles[j]) < 2:
                return False

    paths: list[set[tuple[int, int]]] = []
    total_hits = 0
    for frame in frames:
        try:
            path, hits = trace_ray(grid, frame["gap"], frame["dir"])
        except AssertionError:
            return False
        if hits < 1:
            return False
        total_hits += hits
        for other in paths:
            if path & other:
                return False
        paths.append(path)

    if total_hits < len(frames):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.shape != grid.shape:
        return False
    if FILL not in set(map(int, np.unique(out))):
        return False
    return True


def generate_attempt() -> np.ndarray | None:
    h = random.randint(20, 26)
    w = random.randint(21, 28)
    frame_count = random.randint(2, 3)
    extra_obstacle = 1 if random.random() < 0.65 else 0
    if frame_count == 3 and extra_obstacle == 1:
        total_obstacles = 4
    else:
        total_obstacles = frame_count + extra_obstacle
    obstacle_counts = [1] * frame_count
    if total_obstacles > frame_count:
        obstacle_counts[random.randrange(frame_count)] += 1

    if frame_count == 2 and random.random() < 0.6:
        sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
        random.shuffle(sides)
    else:
        sides = random.sample(SIDES, k=frame_count) if random.random() < 0.55 else [random.choice(SIDES) for _ in range(frame_count)]
        if len(set(sides)) == 1:
            sides[-1] = random.choice([side for side in SIDES if side != sides[0]])

    grid = np.zeros((h, w), dtype=int)
    blue_occ: set[tuple[int, int]] = set()
    red_occ: set[tuple[int, int]] = set()
    path_occ: set[tuple[int, int]] = set()
    frame_forbidden: set[tuple[int, int]] = set()

    for idx in range(frame_count):
        placed = False
        for _ in range(FRAME_ATTEMPTS):
            side = sides[idx]
            gap_offset = random.choice(GAP_OFFSETS)
            r0, c0 = sample_frame_top_left(h, w, side)
            blue, gap, direction, interior, bbox_cells = frame_blue_cells(r0, c0, side, gap_offset)
            if bbox_cells & frame_forbidden:
                continue
            if blue & (blue_occ | red_occ):
                continue
            start = (gap[0] + direction[0], gap[1] + direction[1])
            if not in_bounds((h, w), *start):
                continue
            seed_path = set(interior) | {gap}
            if seed_path & (blue_occ | red_occ | path_occ):
                continue
            planned = plan_ray_path(
                (h, w),
                start,
                direction,
                obstacle_counts[idx],
                blue_occ | red_occ | blue,
                path_occ | seed_path,
            )
            if planned is None:
                continue
            ray_path, new_obstacles = planned

            blue_occ |= blue
            path_occ |= seed_path
            path_occ |= ray_path
            for seg in new_obstacles:
                red_occ |= seg
            frame_forbidden |= expand_cells(bbox_cells, (h, w), 2)
            frame_forbidden |= expand_cells(red_occ, (h, w), 2)
            frame_forbidden |= expand_cells(path_occ, (h, w), 1)
            placed = True
            break
        if not placed:
            return None

    grid.fill(BG)
    for r, c in red_occ:
        grid[r, c] = OBSTACLE
    for r, c in blue_occ:
        assert grid[r, c] == BG
        grid[r, c] = FRAME
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = generate_attempt()
        if grid is not None and validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
