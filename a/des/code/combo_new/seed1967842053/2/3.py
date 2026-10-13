import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
MAROON = 9
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8]


def _in_bounds(shape, cell):
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _step(cell, direction, amount=1):
    return cell[0] + direction[0] * amount, cell[1] + direction[1] * amount


def _opposite(direction):
    return -direction[0], -direction[1]


def _start_cells(center):
    r, c = center
    return {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}


def _end_frame_cells(center):
    r, c = center
    return {
        (rr, cc)
        for rr in range(r - 1, r + 2)
        for cc in range(c - 1, c + 2)
        if (rr, cc) != (r, c)
    }


def _arrow_cells(center, direction):
    r, c = center
    dr, dc = direction
    cells = {(r, c), (r + dr, c + dc)}
    if dr:
        cells.add((r, c - 1))
        cells.add((r, c + 1))
    else:
        cells.add((r - 1, c))
        cells.add((r + 1, c))
    return cells


def _find_start(grid):
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] in (BLACK, MAROON):
                continue
            cells = _start_cells((r, c))
            border = cells - {(r, c)}
            if all(grid[rr, cc] == MAROON for rr, cc in border):
                return {
                    "kind": "start",
                    "center": (r, c),
                    "cells": cells,
                    "color": int(grid[r, c]),
                }
    return None


def _find_end(grid):
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BLACK:
                continue
            frame = _end_frame_cells((r, c))
            if all(grid[rr, cc] == GRAY for rr, cc in frame):
                return {"kind": "end", "center": (r, c), "cells": frame}
    return None


def _parse_arrow(cells):
    if len(cells) != 4:
        return None
    for r, c in cells:
        neighbors = set()
        for direction in DIRS:
            nbr = _step((r, c), direction)
            if nbr in cells:
                neighbors.add(direction)
        if len(neighbors) != 3:
            continue
        if (0, -1) in neighbors and (0, 1) in neighbors:
            shaft = [d for d in neighbors if d[1] == 0]
            if len(shaft) == 1 and _arrow_cells((r, c), shaft[0]) == cells:
                return (r, c), shaft[0]
        if (-1, 0) in neighbors and (1, 0) in neighbors:
            shaft = [d for d in neighbors if d[0] == 0]
            if len(shaft) == 1 and _arrow_cells((r, c), shaft[0]) == cells:
                return (r, c), shaft[0]
    return None


def _find_arrows(grid):
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    arrows = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (BLACK, GRAY, MAROON):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = set()
            while q:
                cell = q.popleft()
                cells.add(cell)
                for direction in DIRS:
                    nbr = _step(cell, direction)
                    if (
                        _in_bounds(grid.shape, nbr)
                        and not seen[nbr]
                        and int(grid[nbr]) == color
                    ):
                        seen[nbr] = True
                        q.append(nbr)
            parsed = _parse_arrow(cells)
            if parsed is not None:
                center, direction = parsed
                arrows.append(
                    {
                        "kind": "arrow",
                        "center": center,
                        "cells": cells,
                        "direction": direction,
                        "color": color,
                    }
                )
    return arrows


def _object_map(objects):
    mapping = {}
    for idx, obj in enumerate(objects):
        for cell in obj["cells"]:
            mapping[cell] = idx
    return mapping


def _ray_first(shape, center, direction, current_cells, mapping):
    cell = _step(center, direction)
    while _in_bounds(shape, cell):
        if cell not in current_cells:
            idx = mapping.get(cell)
            if idx is not None:
                return idx
        cell = _step(cell, direction)
    return None


def _draw_segment(out, prev_obj, next_obj, direction, color):
    cell = _step(prev_obj["center"], direction)
    while _in_bounds(out.shape, cell):
        if cell in prev_obj["cells"]:
            cell = _step(cell, direction)
            continue
        if cell in next_obj["cells"]:
            break
        if out[cell] == BLACK:
            out[cell] = color
        cell = _step(cell, direction)


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the arrow chain, repainting each travelled segment with the carried color.

    Input structure:
    The background is black. One 3x3 maroon start block has a non-maroon center
    color. Several distinct-color four-cell T arrows sit elsewhere, and one gray
    hollow 3x3 frame is the endpoint.
    Rule:
    Begin at the start center with its color. Find the only first arrow visible in
    a cardinal ray from the start. For each arrow reached, draw the just-travelled
    straight gap using the currently carried color, then update the carried color
    to that arrow's color and continue in the direction encoded by its T shape.
    When the gray frame is reached, draw the final gap and fill only the frame
    center with the carried color. Existing start, arrow, and frame pixels remain
    unchanged.
    Key insight:
    The T shape encodes the next direction, while segment colors are delayed: a
    segment uses the color carried before arriving at the arrow, not the arrow's
    own color.
    """
    out = np.array(grid, copy=True)
    start = _find_start(grid)
    end = _find_end(grid)
    if start is None or end is None:
        return out

    arrows = _find_arrows(grid)
    objects = [start] + arrows + [end]
    mapping = _object_map(objects)

    first_options = []
    for direction in DIRS:
        idx = _ray_first(grid.shape, start["center"], direction, start["cells"], mapping)
        if idx is not None and objects[idx]["kind"] == "arrow":
            first_options.append((direction, idx))
    if not first_options:
        return out

    direction, _ = first_options[0]
    current_idx = 0
    carry = start["color"]
    visited = set()
    for _ in range(len(arrows) + 2):
        current = objects[current_idx]
        next_idx = _ray_first(grid.shape, current["center"], direction, current["cells"], mapping)
        if next_idx is None:
            break
        nxt = objects[next_idx]
        _draw_segment(out, current, nxt, direction, carry)
        if nxt["kind"] == "end":
            out[nxt["center"]] = carry
            break
        if nxt["kind"] != "arrow" or next_idx in visited:
            break
        visited.add(next_idx)
        carry = nxt["color"]
        direction = nxt["direction"]
        current_idx = next_idx
    return out


def _path_cells(shape, prev_obj, next_obj, direction):
    cells = []
    cell = _step(prev_obj["center"], direction)
    while _in_bounds(shape, cell):
        if cell in prev_obj["cells"]:
            cell = _step(cell, direction)
            continue
        if cell in next_obj["cells"]:
            return cells
        cells.append(cell)
        cell = _step(cell, direction)
    return None


def _scan_for_generation(shape, obj, direction, mapping):
    return _ray_first(shape, obj["center"], direction, obj["cells"], mapping)


def _build_candidate():
    h = random.randint(15, 30)
    w = random.randint(15, 30)
    arrow_count = random.randint(1, 5)
    margin = 2
    min_dist = 5
    max_dist = 9

    start = (random.randint(margin, h - margin - 1), random.randint(margin, w - margin - 1))
    centers = [start]
    directions = []
    prev_direction = None
    for _ in range(arrow_count + 1):
        choices = []
        for direction in DIRS:
            if prev_direction is not None and direction == _opposite(prev_direction):
                continue
            for dist in range(min_dist, max_dist + 1):
                nxt = _step(centers[-1], direction, dist)
                if margin <= nxt[0] < h - margin and margin <= nxt[1] < w - margin:
                    choices.append((direction, nxt))
        if not choices:
            return None
        direction, nxt = random.choice(choices)
        directions.append(direction)
        centers.append(nxt)
        prev_direction = direction

    arrow_colors = random.sample(ARROW_COLORS, arrow_count)
    remaining = [c for c in ARROW_COLORS if c not in arrow_colors]
    start_color = random.choice(remaining or ARROW_COLORS)

    objects = [
        {
            "kind": "start",
            "center": centers[0],
            "cells": _start_cells(centers[0]),
            "color": start_color,
        }
    ]
    for idx in range(arrow_count):
        direction = directions[idx + 1]
        objects.append(
            {
                "kind": "arrow",
                "center": centers[idx + 1],
                "cells": _arrow_cells(centers[idx + 1], direction),
                "direction": direction,
                "color": arrow_colors[idx],
            }
        )
    objects.append({"kind": "end", "center": centers[-1], "cells": _end_frame_cells(centers[-1])})

    reserved = set()
    mapping = {}
    for idx, obj in enumerate(objects):
        obj_cells = set(obj["cells"])
        if obj["kind"] == "end":
            obj_cells.add(obj["center"])
        if reserved & obj_cells:
            return None
        reserved |= obj_cells
        for cell in obj["cells"]:
            mapping[cell] = idx

    if _scan_for_generation((h, w), objects[0], directions[0], mapping) != 1:
        return None
    for direction in DIRS:
        if direction == directions[0]:
            continue
        if _scan_for_generation((h, w), objects[0], direction, mapping) is not None:
            return None

    used_path = set()
    for idx in range(arrow_count + 1):
        prev_obj = objects[idx]
        next_obj = objects[idx + 1]
        if idx > 0 and prev_obj["direction"] != directions[idx]:
            return None
        if _scan_for_generation((h, w), prev_obj, directions[idx], mapping) != idx + 1:
            return None
        cells = _path_cells((h, w), prev_obj, next_obj, directions[idx])
        if cells is None or not cells:
            return None
        cell_set = set(cells)
        if cell_set & reserved or cell_set & used_path:
            return None
        used_path |= cell_set

    grid = np.zeros((h, w), dtype=int)
    for cell in objects[0]["cells"]:
        grid[cell] = MAROON
    grid[objects[0]["center"]] = start_color
    for obj in objects[1:-1]:
        for cell in obj["cells"]:
            grid[cell] = obj["color"]
    for cell in objects[-1]["cells"]:
        grid[cell] = GRAY
    return grid


def generate_input() -> np.ndarray:
    for _ in range(2000):
        grid = _build_candidate()
        if grid is not None:
            return grid
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0],
            [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
