import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

COLOR_ORDER = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE]
DIRS = {
    "right": (0, 1),
    "up": (-1, 0),
    "left": (0, -1),
    "down": (1, 0),
}
ARROW_SHAPES = {
    "right": {(-1, 1), (0, -1), (0, 0), (0, 1), (1, 1)},
    "up": {(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)},
    "left": {(-1, -1), (0, -1), (0, 0), (0, 1), (1, -1)},
    "down": {(-1, 0), (0, 0), (1, -1), (1, 0), (1, 1)},
}
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _arrow_cells(center, direction):
    r, c = center
    return {(r + dr, c + dc) for dr, dc in ARROW_SHAPES[direction]}


def _component_cells(mask):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _detect_arrows(grid):
    g = np.asarray(grid, dtype=int)
    arrows = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY, MAROON)):
        for cells in _component_cells(g == color):
            if len(cells) != 5:
                continue
            pts = np.array(cells, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if r1 - r0 != 2 or c1 - c0 != 2:
                continue
            rel = {(int(r - (r0 + 1)), int(c - (c0 + 1))) for r, c in cells}
            direction = None
            for name, shape in ARROW_SHAPES.items():
                if rel == shape:
                    direction = name
                    break
            if direction is None:
                continue
            arrows.append(
                {
                    "color": int(color),
                    "direction": direction,
                    "center": (int(r0 + 1), int(c0 + 1)),
                    "top_left": (int(r0), int(c0)),
                    "cells": tuple(sorted((int(r), int(c)) for r, c in cells)),
                }
            )
    color_rank = {color: idx for idx, color in enumerate(COLOR_ORDER)}
    arrows.sort(key=lambda a: (color_rank[a["color"]], a["top_left"][0], a["top_left"][1]))
    return arrows


def _find_target_box(grid, arrow):
    h, w = grid.shape
    r, c = arrow["center"]
    dr, dc = DIRS[arrow["direction"]]
    rr, cc = r + 2 * dr, c + 2 * dc
    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value == MAROON:
            return None
        if value == GREY:
            return (int(rr), int(cc))
        rr += dr
        cc += dc
    return None


def _simulate(grid, collect_events=False):
    out = np.array(grid, dtype=int, copy=True)
    arrows = _detect_arrows(grid)
    transformed_cells = set()
    events = []

    for arrow in arrows:
        start = _find_target_box(out, arrow)
        event = {
            "color": arrow["color"],
            "direction": arrow["direction"],
            "center": arrow["center"],
            "found_box": start is not None,
            "moved": False,
            "stop_reason": None,
        }
        if start is not None:
            dr, dc = DIRS[arrow["direction"]]
            cur_r, cur_c = start
            while True:
                nr, nc = cur_r + dr, cur_c + dc
                if not (0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]):
                    event["stop_reason"] = "boundary"
                    break
                next_value = int(out[nr, nc])
                if next_value == GREY:
                    event["stop_reason"] = "box"
                    break
                if next_value == MAROON:
                    event["stop_reason"] = "transformed_arrow" if (nr, nc) in transformed_cells else "wall"
                    break
                cur_r, cur_c = nr, nc
            out[start] = BLACK
            out[cur_r, cur_c] = GREY
            event["moved"] = (cur_r, cur_c) != start
            event["box_start"] = start
            event["box_end"] = (int(cur_r), int(cur_c))
        for cell in arrow["cells"]:
            out[cell] = MAROON
            transformed_cells.add(cell)
        events.append(event)

    if collect_events:
        return out, arrows, events
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Execute colored arrow pushes in fixed color order, turning each arrow into a maroon wall."""
    return _simulate(grid)


def _can_place(grid, new_cells, pad=1):
    h, w = grid.shape
    new_cells = {(int(r), int(c)) for r, c in new_cells}
    for r, c in new_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    if pad <= 0:
        return True
    for r, c in new_cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                if (rr, cc) in new_cells:
                    continue
                if int(grid[rr, cc]) != BLACK:
                    return False
    return True


def _paint_cells(grid, cells, color):
    for r, c in cells:
        grid[r, c] = int(color)


def _sample_lane_rows(height, count):
    candidates = np.arange(2, height - 2, dtype=int)
    if len(candidates) < count:
        return None
    for _ in range(200):
        chosen = sorted(int(x) for x in _RNG.choice(candidates, size=count, replace=False))
        if all(chosen[i + 1] - chosen[i] >= 4 for i in range(len(chosen) - 1)):
            return chosen
    return None


def _edge_segment_cells(shape, edge, start, length):
    h, w = shape
    if edge == "left":
        return [(start + i, 0) for i in range(length)]
    if edge == "right":
        return [(start + i, w - 1) for i in range(length)]
    if edge == "top":
        return [(0, start + i) for i in range(length)]
    return [(h - 1, start + i) for i in range(length)]


def _rows_for_segment(height, row, length):
    low = max(0, row - length + 1)
    high = min(row, height - length)
    if low > high:
        return None
    return int(_RNG.integers(low, high + 1))


def _place_chain_scenario(grid, row, early_color, late_color):
    h, w = grid.shape
    if w < 18:
        return False
    for _ in range(200):
        direction = str(_RNG.choice(["right", "left"]))
        blocker_dir = str(_RNG.choice(["up", "down"]))
        if direction == "right":
            pusher_col = int(_RNG.integers(2, w - 8))
            box_col = int(_RNG.integers(pusher_col + 4, w - 4))
            blocker_col = int(_RNG.integers(box_col + 2, w - 1))
        else:
            pusher_col = int(_RNG.integers(7, w - 1))
            box_col = int(_RNG.integers(3, pusher_col - 3))
            blocker_col = int(_RNG.integers(1, box_col - 1))
        blocker_cells = _arrow_cells((row, blocker_col), blocker_dir)
        pusher_cells = _arrow_cells((row, pusher_col), direction)
        target_box = {(row, box_col)}
        if not _can_place(grid, blocker_cells | pusher_cells | target_box, pad=1):
            continue
        probe = grid.copy()
        _paint_cells(probe, blocker_cells, early_color)
        _paint_cells(probe, pusher_cells, late_color)
        _paint_cells(probe, target_box, GREY)
        _, _, events = _simulate(probe, collect_events=True)
        if not any(e["stop_reason"] == "transformed_arrow" for e in events):
            continue
        _paint_cells(grid, blocker_cells, early_color)
        _paint_cells(grid, pusher_cells, late_color)
        _paint_cells(grid, target_box, GREY)
        return True
    return False


def _place_push_scenario(grid, row, color, obstacle_kind):
    h, w = grid.shape
    if w < 16:
        return False
    for _ in range(200):
        direction = str(_RNG.choice(["right", "left"]))
        new_cells = set()
        if direction == "right":
            if obstacle_kind == "box":
                pusher_col = int(_RNG.integers(2, w - 10))
                box_col = int(_RNG.integers(pusher_col + 4, w - 5))
                blocker_box_col = int(_RNG.integers(box_col + 2, w))
                new_cells |= {(row, box_col), (row, blocker_box_col)}
            else:
                pusher_col = int(_RNG.integers(2, w - 8))
                box_col = int(_RNG.integers(pusher_col + 4, w - 2))
                new_cells.add((row, box_col))
        else:
            if obstacle_kind == "box":
                pusher_col = int(_RNG.integers(9, w - 1))
                box_col = int(_RNG.integers(4, pusher_col - 3))
                blocker_box_col = int(_RNG.integers(0, box_col - 1))
                new_cells |= {(row, box_col), (row, blocker_box_col)}
            else:
                pusher_col = int(_RNG.integers(7, w - 1))
                box_col = int(_RNG.integers(1, pusher_col - 3))
                new_cells.add((row, box_col))
        pusher_cells = _arrow_cells((row, pusher_col), direction)
        new_cells |= pusher_cells

        wall_cells = set()
        if obstacle_kind == "wall":
            length = int(_RNG.integers(2, min(5, h) + 1))
            start = _rows_for_segment(h, row, length)
            if start is None:
                continue
            edge = "right" if direction == "right" else "left"
            wall_cells = set(_edge_segment_cells(grid.shape, edge, start, length))
            new_cells |= wall_cells

        if not _can_place(grid, new_cells, pad=1):
            continue

        probe = grid.copy()
        _paint_cells(probe, pusher_cells, color)
        for cell in new_cells - pusher_cells - wall_cells:
            probe[cell] = GREY
        _paint_cells(probe, wall_cells, MAROON)
        _, arrows, events = _simulate(probe, collect_events=True)
        matched = False
        for arrow, event in zip(arrows, events):
            if set(arrow["cells"]) == pusher_cells and event["found_box"] and event["moved"]:
                matched = True
                break
        if not matched:
            continue

        _paint_cells(grid, pusher_cells, color)
        for cell in new_cells - pusher_cells - wall_cells:
            grid[cell] = GREY
        _paint_cells(grid, wall_cells, MAROON)
        return True
    return False


def _place_no_box_arrow(grid, row, color):
    w = grid.shape[1]
    for _ in range(200):
        direction = str(_RNG.choice(["right", "left"]))
        if direction == "right":
            center_col = int(_RNG.integers(2, w - 2))
        else:
            center_col = int(_RNG.integers(1, w - 1))
        cells = _arrow_cells((row, center_col), direction)
        if not _can_place(grid, cells, pad=1):
            continue
        probe = grid.copy()
        _paint_cells(probe, cells, color)
        arrows = _detect_arrows(probe)
        placed = next((a for a in arrows if a["color"] == color and set(a["cells"]) == cells), None)
        if placed is None:
            continue
        if _find_target_box(probe, placed) is not None:
            continue
        _paint_cells(grid, cells, color)
        return True
    return False


def _add_decorative_walls(grid, lane_rows):
    h, w = grid.shape
    extra_count = int(_RNG.integers(1, 4))
    forbidden_rows = set(int(r) for r in lane_rows)
    for _ in range(extra_count):
        for _ in range(80):
            edge = str(_RNG.choice(["left", "right", "top", "bottom"]))
            if edge in ("left", "right"):
                length = int(_RNG.integers(2, min(6, h) + 1))
                possible = []
                for start in range(0, h - length + 1):
                    rows = {start + i for i in range(length)}
                    if rows & forbidden_rows:
                        continue
                    possible.append(start)
                if not possible:
                    break
                start = int(_RNG.choice(possible))
            else:
                length = int(_RNG.integers(2, min(6, w) + 1))
                start = int(_RNG.integers(0, w - length + 1))
            cells = set(_edge_segment_cells(grid.shape, edge, start, length))
            if _can_place(grid, cells, pad=0):
                _paint_cells(grid, cells, MAROON)
                break


def _interesting(grid):
    out, arrows, events = _simulate(grid, collect_events=True)
    if len(arrows) < 4:
        return False
    if len({a["color"] for a in arrows}) < 2:
        return False
    if not any(e["found_box"] and e["moved"] for e in events):
        return False
    if not any(not e["found_box"] for e in events):
        return False
    if not any(e["stop_reason"] == "transformed_arrow" for e in events):
        return False
    if int(np.count_nonzero(grid == GREY)) != int(np.count_nonzero(out == GREY)):
        return False
    if np.array_equal(grid, out):
        return False
    return True


def _gen_attempt():
    h = int(_RNG.integers(18, 27))
    w = int(_RNG.integers(18, 29))
    lane_count = int(_RNG.integers(3, min(6, (h - 2) // 4 + 1)))
    lane_rows = _sample_lane_rows(h, lane_count)
    if lane_rows is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)

    early_idx = int(_RNG.integers(0, len(COLOR_ORDER) - 1))
    late_idx = int(_RNG.integers(early_idx + 1, len(COLOR_ORDER)))
    if not _place_chain_scenario(grid, lane_rows[0], COLOR_ORDER[early_idx], COLOR_ORDER[late_idx]):
        return None

    obstacle_kind = str(_RNG.choice(["boundary", "wall", "box"]))
    push_color = int(_RNG.choice(COLOR_ORDER))
    if not _place_push_scenario(grid, lane_rows[1], push_color, obstacle_kind):
        return None

    no_box_color = int(_RNG.choice(COLOR_ORDER))
    if not _place_no_box_arrow(grid, lane_rows[2], no_box_color):
        return None

    for row in lane_rows[3:]:
        extra_color = int(_RNG.choice(COLOR_ORDER))
        if not _place_no_box_arrow(grid, row, extra_color):
            return None

    _add_decorative_walls(grid, lane_rows)

    if not _interesting(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
