from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

CARDINAL_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GRID_MIN = 18
GRID_MAX = 24
ANCHOR_HEIGHTS = (2, 3)
ANCHOR_WIDTHS = (3, 4)
STEM_LENGTHS = (2, 3, 4, 5)
MARKER_COUNTS = (4, 5, 6)
RETRY_LIMIT = 500

# Base shapes face right. Each attachment pixel lies on the right boundary and
# has a non-collinear body branch, which makes the stem/body split identifiable.
BASE_SHAPES = [
    {"cells": {(0, 0), (0, 1), (1, 1)}, "attach": (0, 1)},
    {"cells": {(0, 1), (1, 0), (1, 1)}, "attach": (1, 1)},
    {"cells": {(0, 1), (1, 0), (1, 1), (2, 1)}, "attach": (1, 1)},
    {"cells": {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)}, "attach": (1, 1)},
    {"cells": {(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)}, "attach": (1, 1)},
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = {start}
        remaining.remove(start)
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _anchor_is_ahead(
    tip: tuple[int, int],
    direction: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
) -> bool:
    r, c = tip
    r0, c0, r1, c1 = anchor_bbox
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return r0 <= r <= r1 and c0 - c in (1, 2)
    if (dr, dc) == (0, -1):
        return r0 <= r <= r1 and c - c1 in (1, 2)
    if (dr, dc) == (1, 0):
        return c0 <= c <= c1 and r0 - r in (1, 2)
    if (dr, dc) == (-1, 0):
        return c0 <= c <= c1 and r - r1 in (1, 2)
    return False


def _trace_stem_candidate(
    blue_cells: set[tuple[int, int]],
    tip: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
) -> dict | None:
    tip_neighbors = [cell for cell in _neighbors4(*tip) if cell in blue_cells]
    if len(tip_neighbors) != 1:
        return None

    step_back = (tip_neighbors[0][0] - tip[0], tip_neighbors[0][1] - tip[1])
    if step_back not in CARDINAL_DIRS.values():
        return None

    stem_cells = [tip]
    prev = tip
    current = tip_neighbors[0]
    while True:
        current_neighbors = [cell for cell in _neighbors4(*current) if cell in blue_cells]
        if len(current_neighbors) != 2:
            break
        next_cells = [cell for cell in current_neighbors if cell != prev]
        if len(next_cells) != 1:
            break
        nxt = next_cells[0]
        if (nxt[0] - current[0], nxt[1] - current[1]) != step_back:
            break
        stem_cells.append(current)
        prev, current = current, nxt

    if len(stem_cells) < 2:
        return None

    stem = set(stem_cells)
    body = blue_cells - stem
    if current not in body:
        return None
    if not (3 <= len(body) <= 5):
        return None
    if len(_components(body)) != 1:
        return None
    if _is_solid_rectangle(body):
        return None

    stem_dir = (-step_back[0], -step_back[1])
    if not _anchor_is_ahead(tip, stem_dir, anchor_bbox):
        return None

    body_neighbors = [cell for cell in _neighbors4(*current) if cell in body]
    backward = (-stem_dir[0], -stem_dir[1])
    if not any((cell[0] - current[0], cell[1] - current[1]) != backward for cell in body_neighbors):
        return None

    return {"body": body, "stem": stem, "direction": stem_dir, "length": len(stem)}


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == RED)}
    blue_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == BLUE)}
    assert anchor_cells, "expected a red anchor"
    assert blue_cells, "expected a blue sprite-plus-stem object"
    assert len(_components(anchor_cells)) == 1, "expected one red anchor object"
    assert len(_components(blue_cells)) == 1, "expected one blue connected object"
    assert _is_solid_rectangle(anchor_cells), "anchor must be a solid rectangle"

    anchor_bbox = _bbox(anchor_cells)
    candidates = []
    for tip in sorted(blue_cells):
        candidate = _trace_stem_candidate(blue_cells, tip, anchor_bbox)
        if candidate is not None:
            key = (
                tuple(sorted(candidate["body"])),
                tuple(sorted(candidate["stem"])),
                candidate["direction"],
            )
            if key not in {
                (
                    tuple(sorted(prev["body"])),
                    tuple(sorted(prev["stem"])),
                    prev["direction"],
                )
                for prev in candidates
            }:
                candidates.append(candidate)

    assert len(candidates) == 1, "blue object must split uniquely into body and stem"
    result = candidates[0]
    result["anchor"] = anchor_cells
    return result


def _move_body(
    body: set[tuple[int, int]],
    direction: tuple[int, int],
    length: int,
) -> set[tuple[int, int]]:
    dr, dc = direction
    return {(r + dr * length, c + dc * length) for r, c in body}


def _cast_rays(
    out: np.ndarray,
    moved_body: set[tuple[int, int]],
    anchor: set[tuple[int, int]],
) -> np.ndarray:
    h, w = out.shape
    blockers = set(moved_body)
    for r, c in sorted(moved_body):
        for dr, dc in DIAGONAL_DIRS:
            nr, nc = r + dr, c + dc
            while _in_bounds((h, w), nr, nc):
                if (nr, nc) in blockers or (nr, nc) in anchor:
                    break
                if out[nr, nc] == GREEN:
                    out[nr, nc] = YELLOW
                elif out[nr, nc] == BG:
                    out[nr, nc] = BLUE
                nr += dr
                nc += dc
    return out


def _simulate_from_scene(grid: np.ndarray, scene: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == BLUE] = BG
    moved_body = _move_body(scene["body"], scene["direction"], scene["length"])
    for r, c in moved_body:
        assert _in_bounds(out.shape, r, c), "moved body must remain in bounds"
        assert out[r, c] == BG, "moved body must land on cleared background"
        out[r, c] = BLUE
    return _cast_rays(out, moved_body, scene["anchor"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the blue sprite along its stem toward the red anchor, then fire diagonal rays that recolor hit markers.

    Input structure:
    - A black background contains one red solid rectangular anchor, one blue 4-connected sprite-with-stem object, and several green single-pixel markers.
    - The blue object consists of a small non-rectangular sprite body plus a one-cell-wide straight stem pointing toward the anchor.
    - The stem length determines how far the sprite body will slide, and the anchor remains fixed.

    Rule:
    - Identify the red rectangle, the blue sprite body, the straight blue stem, and the green markers.
    - Erase the entire blue object, then translate only the sprite body forward by exactly the stem length in the stem direction.
    - From every pixel of the translated sprite, extend rays along the four diagonal directions until the next cell would be outside the grid, inside the red anchor, or inside the translated sprite.
    - Ray cells turn black background to blue, and any green marker hit by a ray turns yellow while the ray keeps going.

    Key insight:
    - The red anchor has two coupled roles: it fixes the landing position of the sliding sprite and also acts as a geometric blocker that truncates the later diagonal rays.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _simulate_from_scene(np.asarray(grid, dtype=int), scene)


def _rotate_cw(cells: set[tuple[int, int]], attach: tuple[int, int]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rotated_cells = {(c, -r) for r, c in cells}
    rotated_attach = (attach[1], -attach[0])
    min_r = min(r for r, _ in rotated_cells)
    min_c = min(c for _, c in rotated_cells)
    norm_cells = {(r - min_r, c - min_c) for r, c in rotated_cells}
    norm_attach = (rotated_attach[0] - min_r, rotated_attach[1] - min_c)
    return norm_cells, norm_attach


def _oriented_shape(direction: str) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    spec = random.choice(BASE_SHAPES)
    cells = set(spec["cells"])
    attach = spec["attach"]
    rotations = {"right": 0, "down": 1, "left": 2, "up": 3}[direction]
    for _ in range(rotations):
        cells, attach = _rotate_cw(cells, attach)
    return cells, attach


def _translated(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _ray_geometry(
    moved_body: set[tuple[int, int]],
    anchor: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = shape
    path: set[tuple[int, int]] = set()
    shadow: set[tuple[int, int]] = set()
    body = set(moved_body)
    for r, c in moved_body:
        for dr, dc in DIAGONAL_DIRS:
            nr, nc = r + dr, c + dc
            blocked = False
            while _in_bounds((h, w), nr, nc):
                if (nr, nc) in body:
                    break
                if (nr, nc) in anchor:
                    blocked = True
                    nr += dr
                    nc += dc
                    continue
                if blocked:
                    shadow.add((nr, nc))
                else:
                    path.add((nr, nc))
                nr += dr
                nc += dc
    return path, shadow - path


def _pick_spread_cells(
    candidates: list[tuple[int, int]],
    count: int,
    chosen: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    selected = [] if chosen is None else list(chosen)
    random.shuffle(candidates)
    original_len = len(selected)
    for cell in candidates:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in selected):
            selected.append(cell)
            if len(selected) == original_len + count:
                return selected[original_len:]
    for cell in candidates:
        if cell not in selected:
            selected.append(cell)
            if len(selected) == original_len + count:
                return selected[original_len:]
    return None


def _build_expected(
    grid: np.ndarray,
    body: set[tuple[int, int]],
    direction: tuple[int, int],
    stem_length: int,
    anchor: set[tuple[int, int]],
) -> np.ndarray:
    scene = {
        "body": set(body),
        "direction": direction,
        "length": stem_length,
        "anchor": set(anchor),
    }
    return _simulate_from_scene(grid, scene)


def _build_candidate_input() -> tuple[np.ndarray, dict] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    direction_name = random.choice(tuple(CARDINAL_DIRS))
    direction = CARDINAL_DIRS[direction_name]
    stem_length = random.choice(STEM_LENGTHS)
    gap = random.randint(0, 1)
    marker_total = random.choice(MARKER_COUNTS)
    anchor_h = random.choice(ANCHOR_HEIGHTS)
    anchor_w = random.choice(ANCHOR_WIDTHS)
    local_body, local_attach = _oriented_shape(direction_name)
    body_h = max(r for r, _ in local_body) + 1
    body_w = max(c for _, c in local_body) + 1

    for _ in range(120):
        body_top = random.randint(1, h - body_h - 2)
        body_left = random.randint(1, w - body_w - 2)
        moved_body = _translated(local_body, body_top, body_left)
        attach_final = (body_top + local_attach[0], body_left + local_attach[1])
        dr, dc = direction

        anchor_options = []
        if direction_name == "right":
            anchor_left = body_left + body_w + gap
            if anchor_left + anchor_w <= w - 1:
                top_lo = max(1, attach_final[0] - anchor_h + 1)
                top_hi = min(h - anchor_h - 1, attach_final[0] + 1)
                anchor_options = [(r0, anchor_left) for r0 in range(top_lo, top_hi + 1)]
        elif direction_name == "left":
            anchor_left = body_left - gap - anchor_w - 1
            if anchor_left >= 1:
                top_lo = max(1, attach_final[0] - anchor_h + 1)
                top_hi = min(h - anchor_h - 1, attach_final[0] + 1)
                anchor_options = [(r0, anchor_left) for r0 in range(top_lo, top_hi + 1)]
        elif direction_name == "down":
            anchor_top = body_top + body_h + gap
            if anchor_top + anchor_h <= h - 1:
                left_lo = max(1, attach_final[1] - anchor_w + 1)
                left_hi = min(w - anchor_w - 1, attach_final[1] + 1)
                anchor_options = [(anchor_top, c0) for c0 in range(left_lo, left_hi + 1)]
        else:
            anchor_top = body_top - gap - anchor_h - 1
            if anchor_top >= 1:
                left_lo = max(1, attach_final[1] - anchor_w + 1)
                left_hi = min(w - anchor_w - 1, attach_final[1] + 1)
                anchor_options = [(anchor_top, c0) for c0 in range(left_lo, left_hi + 1)]

        random.shuffle(anchor_options)
        for anchor_top, anchor_left in anchor_options:
            anchor = {
                (r, c)
                for r in range(anchor_top, anchor_top + anchor_h)
                for c in range(anchor_left, anchor_left + anchor_w)
            }

            input_body = _translated(moved_body, -dr * stem_length, -dc * stem_length)
            if not all(_in_bounds((h, w), r, c) for r, c in input_body):
                continue
            if any(r in {0, h - 1} or c in {0, w - 1} for r, c in input_body):
                continue
            input_attach = (attach_final[0] - dr * stem_length, attach_final[1] - dc * stem_length)
            stem = {(input_attach[0] + dr * k, input_attach[1] + dc * k) for k in range(1, stem_length + 1)}
            input_blue = input_body | stem
            if input_blue & anchor:
                continue
            if len(_components(input_blue)) != 1:
                continue

            path, shadow = _ray_geometry(moved_body, anchor, (h, w))
            shadow_only = sorted(shadow - input_blue - anchor - moved_body)
            hit_candidates = sorted(path - input_blue - anchor - moved_body)
            if len(hit_candidates) < 2 or not shadow_only:
                continue

            all_cells = {(r, c) for r in range(h) for c in range(w)}
            forbidden = input_blue | moved_body | anchor
            off_path = sorted(all_cells - forbidden - path)
            hit_count = min(random.randint(2, 3), marker_total - 2)
            hit_markers = _pick_spread_cells(hit_candidates[:], hit_count)
            if hit_markers is None:
                continue
            blocked_marker = _pick_spread_cells(shadow_only[:], 1, chosen=hit_markers)
            if blocked_marker is None:
                continue
            remaining = marker_total - len(hit_markers) - 1
            miss_markers = _pick_spread_cells(off_path[:], remaining, chosen=hit_markers + blocked_marker)
            if miss_markers is None:
                continue
            markers = set(hit_markers + blocked_marker + miss_markers)

            grid = np.zeros((h, w), dtype=int)
            for r, c in anchor:
                grid[r, c] = RED
            for r, c in input_blue:
                grid[r, c] = BLUE
            for r, c in markers:
                grid[r, c] = GREEN

            expected = _build_expected(grid, input_body, direction, stem_length, anchor)
            try:
                scene = _extract_scene(grid)
            except AssertionError:
                continue
            if scene["body"] != input_body or scene["stem"] != stem or scene["direction"] != direction:
                continue
            actual = main(grid)
            if not np.array_equal(actual, expected):
                continue

            yellow_cells = {tuple(map(int, cell)) for cell in np.argwhere(actual == YELLOW)}
            if len(yellow_cells) < 2:
                continue
            if not set(hit_markers).issubset(yellow_cells):
                continue
            if blocked_marker[0] in yellow_cells:
                continue
            if not miss_markers:
                continue
            if not any(actual[r, c] == GREEN for r, c in blocked_marker + miss_markers):
                continue
            if len({tuple(map(int, cell)) for cell in np.argwhere(actual == BLUE)} - moved_body) < 4:
                continue

            meta = {
                "input_body": input_body,
                "stem": stem,
                "direction": direction,
                "stem_length": stem_length,
                "moved_body": moved_body,
                "anchor": anchor,
                "hit_markers": set(hit_markers),
                "blocked_marker": set(blocked_marker),
                "miss_markers": set(miss_markers),
            }
            return grid, meta
    return None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        built = _build_candidate_input()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid sliding-anchor-ray input")
