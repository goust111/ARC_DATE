from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_RETRIES = 300
PLACEMENT_RETRIES = 120
SHAPE_RETRIES = 80


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(grid: np.ndarray, bg: int = BG) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _marker_candidates(
    cells: set[tuple[int, int]], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    dr, dc = direction
    candidates = []
    for r, c in cells:
        if dr == -1 and r == rmin and cmin < c < cmax:
            candidates.append((r, c))
        elif dr == 1 and r == rmax and cmin < c < cmax:
            candidates.append((r, c))
        elif dc == -1 and c == cmin and rmin < r < rmax:
            candidates.append((r, c))
        elif dc == 1 and c == cmax and rmin < r < rmax:
            candidates.append((r, c))
    return candidates


def _grow_polyomino(target_area: int, box_h: int, box_w: int) -> set[tuple[int, int]]:
    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return _normalize(cells)


def _make_random_object(
    main_color: int, marker_color: int, direction: tuple[int, int]
) -> dict[str, object] | None:
    for _ in range(SHAPE_RETRIES):
        box_h = random.randint(3, 5)
        box_w = random.randint(3, 5)
        min_area = max(5, (box_h * box_w + 1) // 2)
        area = random.randint(min_area, box_h * box_w)
        cells = _grow_polyomino(area, box_h, box_w)
        if len(cells) < 5:
            continue
        marker_options = _marker_candidates(cells, direction)
        if not marker_options:
            continue
        marker = random.choice(marker_options)
        h, w = _bbox(cells)
        return {
            "cells": cells,
            "marker": marker,
            "h": h,
            "w": w,
            "main_color": int(main_color),
            "marker_color": int(marker_color),
            "direction": (int(direction[0]), int(direction[1])),
        }
    return None


def _make_dense_rect_object(
    main_color: int, marker_color: int, direction: tuple[int, int]
) -> dict[str, object]:
    h = random.randint(3, 4)
    w = random.randint(3, 4)
    cells = {(r, c) for r in range(h) for c in range(w)}
    dr, dc = direction
    if dr != 0:
        marker = (0 if dr == -1 else h - 1, w // 2)
    else:
        marker = (h // 2, 0 if dc == -1 else w - 1)
    return {
        "cells": cells,
        "marker": marker,
        "h": h,
        "w": w,
        "main_color": int(main_color),
        "marker_color": int(marker_color),
        "direction": (int(dr), int(dc)),
    }


def _shift_cells(cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in cells}


def _place_ok(
    abs_cells: set[tuple[int, int]], occupied: set[tuple[int, int]]
) -> bool:
    for r, c in abs_cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _render_input(placements: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for placement in placements:
        spec = placement["spec"]
        top = int(placement["top"])
        left = int(placement["left"])
        cells = spec["cells"]
        marker = spec["marker"]
        for r, c in cells:
            rr = top + r
            cc = left + c
            color = int(spec["marker_color"]) if (r, c) == marker else int(spec["main_color"])
            grid[rr, cc] = color
    return grid


def _parse_objects(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _connected_components(grid, bg=BG)
    objects: list[dict[str, object]] = []
    for coords in comps:
        colors = [int(grid[r, c]) for r, c in coords]
        counts = Counter(colors)
        singleton_colors = [color for color, count in counts.items() if int(count) == 1]
        assert len(singleton_colors) == 1, "each object must contain exactly one singleton marker color"
        marker_color = int(singleton_colors[0])
        main_colors = [color for color in counts if int(color) != int(marker_color)]
        assert len(main_colors) == 1, "each object must have one main color and one marker color"
        main_color = int(main_colors[0])
        assert int(counts[main_color]) >= 2, "main body must contain at least two cells"

        marker_abs = None
        for r, c in coords:
            if int(grid[r, c]) == int(marker_color):
                marker_abs = (int(r), int(c))
                break
        assert marker_abs is not None

        rmin = int(coords[:, 0].min())
        rmax = int(coords[:, 0].max())
        cmin = int(coords[:, 1].min())
        cmax = int(coords[:, 1].max())
        sides = []
        if marker_abs[0] == rmin:
            sides.append((-1, 0))
        if marker_abs[0] == rmax:
            sides.append((1, 0))
        if marker_abs[1] == cmin:
            sides.append((0, -1))
        if marker_abs[1] == cmax:
            sides.append((0, 1))
        assert len(sides) == 1, "marker must lie on exactly one non-corner bbox side"
        direction = sides[0]

        rel_cells = {(int(r) - rmin, int(c) - cmin) for r, c in coords}
        rel_marker = (int(marker_abs[0]) - rmin, int(marker_abs[1]) - cmin)
        color_map = {
            (int(r) - rmin, int(c) - cmin): int(grid[int(r), int(c)]) for r, c in coords
        }
        objects.append(
            {
                "coords": coords.copy(),
                "direction": (int(direction[0]), int(direction[1])),
                "main_color": int(main_color),
                "marker_color": int(marker_color),
                "source_marker": (int(marker_abs[0]), int(marker_abs[1])),
                "cells": rel_cells,
                "marker": rel_marker,
                "h": int(rmax - rmin + 1),
                "w": int(cmax - cmin + 1),
                "color_map": color_map,
            }
        )
    assert objects, "expected at least one foreground object"
    return objects


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    objects = _parse_objects(g)

    current = [obj["coords"].astype(int, copy=True) for obj in objects]
    directions = [obj["direction"] for obj in objects]
    moved_steps = [0 for _ in objects]

    max_rounds = h + w + 5
    for _ in range(max_rounds):
        occupied = np.full((h, w), -1, dtype=int)
        for idx, coords in enumerate(current):
            occupied[coords[:, 0], coords[:, 1]] = int(idx)

        movable = []
        for idx, coords in enumerate(current):
            dr, dc = directions[idx]
            shifted = coords + np.array([dr, dc], dtype=int)
            if np.any(shifted[:, 0] < 0) or np.any(shifted[:, 0] >= h) or np.any(shifted[:, 1] < 0) or np.any(shifted[:, 1] >= w):
                movable.append(False)
                continue
            other_hits = occupied[shifted[:, 0], shifted[:, 1]]
            can_move = bool(np.all((other_hits == -1) | (other_hits == int(idx))))
            movable.append(can_move)

        if not any(movable):
            break

        for idx, can_move in enumerate(movable):
            if can_move:
                dr, dc = directions[idx]
                current[idx] = current[idx] + np.array([dr, dc], dtype=int)
                moved_steps[idx] += 1
    else:
        raise AssertionError("movement did not stabilize")

    out = np.zeros_like(g)
    for obj, coords in zip(objects, current):
        rmin = int(coords[:, 0].min())
        cmin = int(coords[:, 1].min())
        for (lr, lc), color in obj["color_map"].items():
            out[rmin + int(lr), cmin + int(lc)] = int(color)

    for obj in objects:
        sr, sc = obj["source_marker"]
        out[int(sr), int(sc)] = int(obj["marker_color"])

    if not return_meta:
        return out

    final_occ = np.full((h, w), -1, dtype=int)
    for idx, coords in enumerate(current):
        final_occ[coords[:, 0], coords[:, 1]] = int(idx)

    meta_objects = []
    for idx, (obj, coords, steps) in enumerate(zip(objects, current, moved_steps)):
        dr, dc = directions[idx]
        shifted = coords + np.array([dr, dc], dtype=int)
        if np.any(shifted[:, 0] < 0) or np.any(shifted[:, 0] >= h) or np.any(shifted[:, 1] < 0) or np.any(shifted[:, 1] >= w):
            stop_reason = "boundary"
        else:
            stop_reason = "object"
        meta_objects.append(
            {
                "source_marker": obj["source_marker"],
                "direction": obj["direction"],
                "steps": int(steps),
                "stop_reason": stop_reason,
                "final_coords": coords.copy(),
                "span": int(obj["w"] if obj["direction"][1] != 0 else obj["h"]),
            }
        )
    return out, meta_objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each direction-marked object to the furthest reachable position on its own ray, leaving only the source marker behind.

    Input structure:
    - The background is black.
    - Each foreground component is one 4-connected object with one main color and one different-colored singleton marker pixel.
    - The marker lies on exactly one non-corner side of the object's bounding box, which encodes up, down, left, or right.

    Rule:
    - Read each object's direction from the bounding-box side that contains its marker pixel.
    - Move all objects step by step along their own directions; an object may advance one cell only when the shifted shape stays in bounds and does not hit another object's current position.
    - Repeat until no object can move, then draw every full object at its resting position with its original colors.
    - Clear each source body and restore only the original marker pixel at that source site.

    Key insight: The marker's location inside the object, not its color id, determines the motion ray; the final layout comes from simulating those directed slides while keeping the source marker as a visible launch trace.
    """
    return _solve(np.asarray(grid, dtype=int), return_meta=False)


def _random_free_position(
    spec: dict[str, object],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    obj_h = int(spec["h"])
    obj_w = int(spec["w"])
    dr, dc = spec["direction"]
    span = int(obj_w if dc != 0 else obj_h)

    if dr == -1:
        row_lo, row_hi = span, h - obj_h - 2
        col_lo, col_hi = 1, w - obj_w - 2
    elif dr == 1:
        row_lo, row_hi = 1, h - obj_h - span - 1
        col_lo, col_hi = 1, w - obj_w - 2
    elif dc == -1:
        row_lo, row_hi = 1, h - obj_h - 2
        col_lo, col_hi = span, w - obj_w - 2
    else:
        row_lo, row_hi = 1, h - obj_h - 2
        col_lo, col_hi = 1, w - obj_w - span - 1

    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(row_lo, row_hi)
        left = random.randint(col_lo, col_hi)
        abs_cells = _shift_cells(spec["cells"], top, left)
        if _place_ok(abs_cells, occupied):
            return top, left
    return None


def _build_free_case(shape: tuple[int, int]) -> np.ndarray | None:
    obj_count = random.randint(2, 4)
    palette = random.sample(list(range(1, 10)), 2 * obj_count)
    directions = random.sample(DIRS, k=min(4, obj_count))
    while len(directions) < obj_count:
        directions.append(random.choice(DIRS))
    random.shuffle(directions)

    specs = []
    for idx in range(obj_count):
        spec = _make_random_object(palette[2 * idx], palette[2 * idx + 1], directions[idx])
        if spec is None:
            return None
        specs.append(spec)

    occupied: set[tuple[int, int]] = set()
    placements = []
    for spec in specs:
        pos = _random_free_position(spec, shape, occupied)
        if pos is None:
            return None
        top, left = pos
        abs_cells = _shift_cells(spec["cells"], top, left)
        occupied |= abs_cells
        placements.append({"spec": spec, "top": int(top), "left": int(left)})

    return _render_input(placements, shape)


def _build_blocked_case(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    obj_count = random.randint(2, 4)
    palette = random.sample(list(range(1, 10)), 2 * obj_count)
    pair_dir = random.choice(DIRS)

    leader = _make_dense_rect_object(palette[0], palette[1], pair_dir)
    follower = _make_dense_rect_object(palette[2], palette[3], pair_dir)

    placements = []
    occupied: set[tuple[int, int]] = set()

    dr, dc = pair_dir
    h1, w1 = int(leader["h"]), int(leader["w"])
    h2, w2 = int(follower["h"]), int(follower["w"])

    if dr == 0:
        lane_h = max(h1, h2)
        if h - lane_h - 2 < 1:
            return None
        top = random.randint(1, h - lane_h - 2)
        front_gap = random.randint(w1, w1 + 3)
        gap = random.randint(2, 4)
        if dc == 1:
            leader_left = w - w1 - front_gap
            follower_left = leader_left - gap - w2
        else:
            leader_left = front_gap
            follower_left = leader_left + w1 + gap
        if leader_left < 1 or follower_left < 1 or leader_left + w1 > w - 1 or follower_left + w2 > w - 1:
            return None
        positions = [(leader, top, leader_left), (follower, top, follower_left)]
    else:
        lane_w = max(w1, w2)
        if w - lane_w - 2 < 1:
            return None
        left = random.randint(1, w - lane_w - 2)
        front_gap = random.randint(h1, h1 + 3)
        gap = random.randint(2, 4)
        if dr == 1:
            leader_top = h - h1 - front_gap
            follower_top = leader_top - gap - h2
        else:
            leader_top = front_gap
            follower_top = leader_top + h1 + gap
        if leader_top < 1 or follower_top < 1 or leader_top + h1 > h - 1 or follower_top + h2 > h - 1:
            return None
        positions = [(leader, leader_top, left), (follower, follower_top, left)]

    for spec, top, left in positions:
        abs_cells = _shift_cells(spec["cells"], int(top), int(left))
        if not _place_ok(abs_cells, occupied):
            return None
        occupied |= abs_cells
        placements.append({"spec": spec, "top": int(top), "left": int(left)})

    other_dirs = random.sample(DIRS, k=min(4, obj_count + 1))
    while len(other_dirs) < obj_count - 2:
        other_dirs.append(random.choice(DIRS))
    random.shuffle(other_dirs)

    for idx in range(2, obj_count):
        direction = other_dirs[idx - 2]
        spec = _make_random_object(palette[2 * idx], palette[2 * idx + 1], direction)
        if spec is None:
            return None
        pos = _random_free_position(spec, shape, occupied)
        if pos is None:
            return None
        top, left = pos
        abs_cells = _shift_cells(spec["cells"], top, left)
        occupied |= abs_cells
        placements.append({"spec": spec, "top": int(top), "left": int(left)})

    return _render_input(placements, shape)


def _validate_case(grid: np.ndarray, blocked_case: bool) -> bool:
    try:
        out, meta = _solve(grid, return_meta=True)
    except AssertionError:
        return False

    if int(np.count_nonzero(grid)) == 0 or int(np.count_nonzero(out)) == 0:
        return False

    if not all(int(info["steps"]) >= 1 for info in meta):
        return False

    source_markers = {tuple(map(int, info["source_marker"])) for info in meta}
    final_cells = set()
    for info in meta:
        final_cells |= {tuple(map(int, rc)) for rc in info["final_coords"]}
    if source_markers & final_cells:
        return False

    if blocked_case:
        if not any(info["stop_reason"] == "object" for info in meta):
            return False
    else:
        if any(info["stop_reason"] != "boundary" for info in meta):
            return False

    directions = {tuple(map(int, info["direction"])) for info in meta}
    if len(meta) >= 3 and len(directions) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    blocked_case = bool(random.getrandbits(1))
    for _ in range(GENERATE_RETRIES):
        shape = (random.randint(14, 24), random.randint(14, 24))
        grid = _build_blocked_case(shape) if blocked_case else _build_free_case(shape)
        if grid is None:
            continue
        if _validate_case(grid, blocked_case):
            return grid
    raise AssertionError("failed to generate a valid input")
