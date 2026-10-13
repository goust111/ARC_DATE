from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
EMITTER_COLORS = (2, 3, 4, 6, 7)

MIN_SIZE = 16
MAX_SIZE = 24
MIN_ROOM_SPAN = 5
TARGET_ROOM_COUNTS = (3, 4, 5)
MAX_LAYOUT_ATTEMPTS = 300
MAX_MARKER_ATTEMPTS = 400
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _flood_components(passable: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = passable.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or labels[r, c] >= 0:
                continue
            label = len(components)
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = label
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            components.append(comp)
    return components, labels


def _analyze_chambers(grid: np.ndarray) -> list[dict]:
    wall_mask = (grid == BLUE) | (grid == GREY)
    passable = ~wall_mask
    components, labels = _flood_components(passable)
    chambers: list[dict] = []
    for chamber_id, cells in enumerate(components):
        emitters = [(r, c) for r, c in cells if int(grid[r, c]) not in (BLACK,)]
        emitters = [(r, c) for r, c in emitters if int(grid[r, c]) not in (BLUE, GREY)]
        if not emitters:
            continue
        assert len(emitters) == 1, "each chamber must contain exactly one emitter"
        er, ec = emitters[0]
        marker_cells: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
                    marker_cells.add((nr, nc))
        chambers.append(
            {
                "id": chamber_id,
                "cells": cells,
                "labels": labels,
                "emitter": (er, ec),
                "color": int(grid[er, ec]),
                "grey_count": len(marker_cells),
            }
        )
    return chambers


def main(grid: np.ndarray) -> np.ndarray:
    """Extend four chamber rays with alternating emitter/grey colors, flipping parity when the chamber wall has an odd number of grey markers.

    Input structure:
    - Black is the background inside rectilinear chambers bounded by blue wall segments; some wall cells are replaced by grey markers.
    - Each chamber contains exactly one non-wall emitter pixel, with a distinct color chosen from the non-grey warm/bright palette.
    - Grey markers lie on the chamber walls, still behave as walls, and never occupy wall intersections.

    Rule:
    - Treat blue and grey cells as blocking walls, then flood-fill the remaining cells into chambers.
    - For each chamber, count how many grey marker cells are 4-neighbor-adjacent to that chamber's interior.
    - From the chamber's emitter, extend rays up, down, left, and right until the next wall cell or the grid boundary; the emitter stays unchanged.
    - With even grey-marker count, odd ray distances become grey and even distances become the emitter color; with odd grey-marker count, swap that phase.

    Key insight:
    - The ray geometry comes from the chamber layout and emitter position, but the visible stripe phase is controlled by a second signal: the parity of grey wall markers surrounding that chamber.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_mask = (g == BLUE) | (g == GREY)
    out = g.copy()
    for chamber in _analyze_chambers(g):
        er, ec = chamber["emitter"]
        color = chamber["color"]
        flipped = chamber["grey_count"] % 2 == 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            dist = 1
            r = er + dr
            c = ec + dc
            while _in_bounds(g.shape, r, c) and not wall_mask[r, c]:
                use_emitter = (dist % 2 == 1) if flipped else (dist % 2 == 0)
                out[r, c] = color if use_emitter else GREY
                r += dr
                c += dc
                dist += 1
    return out


def _can_split(rect: tuple[int, int, int, int]) -> tuple[list[str], list[int], list[int]]:
    top, bottom, left, right = rect
    vertical_positions = list(range(left + MIN_ROOM_SPAN + 1, right - MIN_ROOM_SPAN))
    horizontal_positions = list(range(top + MIN_ROOM_SPAN + 1, bottom - MIN_ROOM_SPAN))
    orientations: list[str] = []
    if vertical_positions:
        orientations.append("v")
    if horizontal_positions:
        orientations.append("h")
    return orientations, vertical_positions, horizontal_positions


def _partition_rectangles(rng: np.random.Generator) -> tuple[int, int, list[tuple[int, int, int, int]], list[tuple[str, int, int, int]]]:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        h = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
        w = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
        target = int(rng.choice(np.array(TARGET_ROOM_COUNTS, dtype=int)))
        rects = [(0, h - 1, 0, w - 1)]
        walls: list[tuple[str, int, int, int]] = []
        ok = True
        while len(rects) < target:
            splittable: list[tuple[int, list[str], list[int], list[int]]] = []
            for idx, rect in enumerate(rects):
                orientations, vertical_positions, horizontal_positions = _can_split(rect)
                if orientations:
                    splittable.append((idx, orientations, vertical_positions, horizontal_positions))
            if not splittable:
                ok = False
                break
            idx, orientations, vertical_positions, horizontal_positions = splittable[
                int(rng.integers(len(splittable)))
            ]
            top, bottom, left, right = rects.pop(idx)
            orientation = str(rng.choice(np.array(orientations)))
            if orientation == "v":
                cut = int(rng.choice(np.array(vertical_positions, dtype=int)))
                rects.append((top, bottom, left, cut))
                rects.append((top, bottom, cut, right))
                walls.append(("v", cut, top, bottom))
            else:
                cut = int(rng.choice(np.array(horizontal_positions, dtype=int)))
                rects.append((top, cut, left, right))
                rects.append((cut, bottom, left, right))
                walls.append(("h", cut, left, right))
        if not ok:
            continue
        rects.sort()
        return h, w, rects, walls
    raise RuntimeError("failed to partition the canvas into valid chambers")


def _render_wall_grid(
    h: int,
    w: int,
    rects: list[tuple[int, int, int, int]],
    walls: list[tuple[str, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = BLUE
    grid[h - 1, :] = BLUE
    grid[:, 0] = BLUE
    grid[:, w - 1] = BLUE
    for orientation, axis, start, end in walls:
        if orientation == "v":
            grid[start : end + 1, axis] = BLUE
        else:
            grid[axis, start : end + 1] = BLUE
    chamber_ids = np.full((h, w), -1, dtype=int)
    for idx, (top, bottom, left, right) in enumerate(rects):
        chamber_ids[top + 1 : bottom, left + 1 : right] = idx
    return grid, chamber_ids


def _wall_is_intersection(wall_mask: np.ndarray, r: int, c: int) -> bool:
    up = _in_bounds(wall_mask.shape, r - 1, c) and bool(wall_mask[r - 1, c])
    down = _in_bounds(wall_mask.shape, r + 1, c) and bool(wall_mask[r + 1, c])
    left = _in_bounds(wall_mask.shape, r, c - 1) and bool(wall_mask[r, c - 1])
    right = _in_bounds(wall_mask.shape, r, c + 1) and bool(wall_mask[r, c + 1])
    return (up or down) and (left or right)


def _collect_marker_groups(grid: np.ndarray, chamber_ids: np.ndarray) -> list[dict]:
    wall_mask = grid == BLUE
    h, w = grid.shape
    meta = np.empty((h, w), dtype=object)
    for r in range(h):
        for c in range(w):
            if not wall_mask[r, c] or _wall_is_intersection(wall_mask, r, c):
                continue
            adj = sorted(
                {
                    int(chamber_ids[nr, nc])
                    for nr, nc in _neighbors4(r, c)
                    if _in_bounds(grid.shape, nr, nc) and int(chamber_ids[nr, nc]) >= 0
                }
            )
            if not adj:
                continue
            up = _in_bounds(grid.shape, r - 1, c) and bool(wall_mask[r - 1, c])
            down = _in_bounds(grid.shape, r + 1, c) and bool(wall_mask[r + 1, c])
            left = _in_bounds(grid.shape, r, c - 1) and bool(wall_mask[r, c - 1])
            right = _in_bounds(grid.shape, r, c + 1) and bool(wall_mask[r, c + 1])
            if left or right:
                orientation = "h"
            elif up or down:
                orientation = "v"
            else:
                continue
            meta[r, c] = (orientation, tuple(adj))

    groups: list[dict] = []
    visited: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            cell_meta = meta[r, c]
            if cell_meta is None or (r, c) in visited:
                continue
            orientation, adj = cell_meta
            cells: list[tuple[int, int]] = []
            if orientation == "h":
                cc = c
                while cc - 1 >= 0 and meta[r, cc - 1] == cell_meta:
                    cc -= 1
                while cc < w and meta[r, cc] == cell_meta:
                    visited.add((r, cc))
                    cells.append((r, cc))
                    cc += 1
            else:
                rr = r
                while rr - 1 >= 0 and meta[rr - 1, c] == cell_meta:
                    rr -= 1
                while rr < h and meta[rr, c] == cell_meta:
                    visited.add((rr, c))
                    cells.append((rr, c))
                    rr += 1
            groups.append({"cells": cells, "adj": adj, "orientation": orientation})
    return groups


def _choose_marker_groups(
    rng: np.random.Generator,
    groups: list[dict],
    room_count: int,
) -> tuple[list[int], list[int]] | None:
    shared = [idx for idx, group in enumerate(groups) if len(group["adj"]) == 2]
    outer = [idx for idx, group in enumerate(groups) if len(group["adj"]) == 1]
    if not shared or not outer:
        return None
    for _ in range(MAX_MARKER_ATTEMPTS):
        counts = [0] * room_count
        selected: list[int] = []

        shared_pick = int(rng.integers(1, min(3, len(shared)) + 1))
        outer_pick = int(rng.integers(1, min(3, len(outer)) + 1))
        for pool, count in ((shared, shared_pick), (outer, outer_pick)):
            for idx in rng.choice(np.array(pool, dtype=int), size=count, replace=False):
                gid = int(idx)
                adj = groups[gid]["adj"]
                if all(counts[ch] < 4 for ch in adj):
                    selected.append(gid)
                    for ch in adj:
                        counts[ch] += 1

        rest = list(range(len(groups)))
        rng.shuffle(rest)
        pick_prob = float(rng.uniform(0.15, 0.5))
        for gid in rest:
            if gid in selected or rng.random() >= pick_prob:
                continue
            adj = groups[gid]["adj"]
            if any(counts[ch] >= 4 for ch in adj):
                continue
            selected.append(gid)
            for ch in adj:
                counts[ch] += 1

        if not any(v % 2 == 0 for v in counts):
            continue
        if not any(v % 2 == 1 for v in counts):
            continue
        if max(counts) > 4:
            continue
        if len(selected) < 2:
            continue
        return sorted(selected), counts
    return None


def _marker_cell_for_group(group: dict) -> tuple[int, int]:
    cells = group["cells"]
    if len(cells) <= 2:
        return cells[len(cells) // 2]
    return cells[len(cells) // 2]


def _place_emitters(
    rng: np.random.Generator,
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
) -> dict[int, tuple[int, int, int]]:
    room_count = len(rects)
    colors = rng.choice(np.array(EMITTER_COLORS, dtype=int), size=room_count, replace=False)
    emitters: dict[int, tuple[int, int, int]] = {}
    for room_id, (top, bottom, left, right) in enumerate(rects):
        row = int(rng.integers(top + 2, bottom - 1))
        col = int(rng.integers(left + 2, right - 1))
        color = int(colors[room_id])
        grid[row, col] = color
        emitters[room_id] = (row, col, color)
    return emitters


def _expected_from_metadata(
    grid: np.ndarray,
    emitters: dict[int, tuple[int, int, int]],
    counts: list[int],
) -> np.ndarray:
    out = grid.copy()
    wall_mask = (grid == BLUE) | (grid == GREY)
    for room_id, (er, ec, color) in emitters.items():
        flipped = counts[room_id] % 2 == 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            dist = 1
            r = er + dr
            c = ec + dc
            while _in_bounds(grid.shape, r, c) and not wall_mask[r, c]:
                use_emitter = (dist % 2 == 1) if flipped else (dist % 2 == 0)
                out[r, c] = color if use_emitter else GREY
                dist += 1
                r += dr
                c += dc
    return out


def _expected_without_flip(
    grid: np.ndarray,
    emitters: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    wall_mask = (grid == BLUE) | (grid == GREY)
    for _room_id, (er, ec, color) in emitters.items():
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            dist = 1
            r = er + dr
            c = ec + dc
            while _in_bounds(grid.shape, r, c) and not wall_mask[r, c]:
                out[r, c] = color if dist % 2 == 0 else GREY
                dist += 1
                r += dr
                c += dc
    return out


def _validate_generated_case(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    emitters: dict[int, tuple[int, int, int]],
    selected_groups: list[int],
    group_defs: list[dict],
    counts: list[int],
) -> bool:
    chambers = _analyze_chambers(grid)
    if len(chambers) != len(rects):
        return False
    by_color = {int(info["color"]): info for info in chambers}
    if len(by_color) != len(rects):
        return False

    wall_mask = (grid == BLUE) | (grid == GREY)
    for room_id, (er, ec, color) in emitters.items():
        if color not in by_color:
            return False
        info = by_color[color]
        if tuple(info["emitter"]) != (er, ec):
            return False
        if info["grey_count"] != counts[room_id]:
            return False
        for nr, nc in _neighbors4(er, ec):
            if wall_mask[nr, nc]:
                return False

    for gid in selected_groups:
        mr, mc = _marker_cell_for_group(group_defs[gid])
        if _wall_is_intersection(grid == BLUE, mr, mc):
            return False
        if int(grid[mr, mc]) != GREY:
            return False

    if not any(len(group_defs[gid]["adj"]) == 2 for gid in selected_groups):
        return False
    if not any(len(group_defs[gid]["adj"]) == 1 for gid in selected_groups):
        return False
    if not any(count % 2 == 0 for count in counts):
        return False
    if not any(count % 2 == 1 for count in counts):
        return False

    expected = _expected_from_metadata(grid, emitters, counts)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, _expected_without_flip(grid, emitters)):
        return False
    if np.array_equal(actual, grid):
        return False
    if len({tuple(count % 2 for count in counts), tuple(counts)}) < 2:
        return False
    return True


def _generate_one(rng: np.random.Generator) -> np.ndarray | None:
    h, w, rects, walls = _partition_rectangles(rng)
    grid, chamber_ids = _render_wall_grid(h, w, rects, walls)
    group_defs = _collect_marker_groups(grid, chamber_ids)
    choice = _choose_marker_groups(rng, group_defs, len(rects))
    if choice is None:
        return None
    selected_groups, counts = choice

    for gid in selected_groups:
        r, c = _marker_cell_for_group(group_defs[gid])
        grid[r, c] = GREY

    emitters = _place_emitters(rng, grid, rects)
    if not _validate_generated_case(grid, rects, emitters, selected_groups, group_defs, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chamber-ray input")
