import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
DOOR_COLORS = [1, 2, 3, 4, 6, 8]
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _neighbors4(r, c, h, w):
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(grid, color):
    h, w = grid.shape
    seen = set()
    comps = []
    for r, c in zip(*np.where(grid == color)):
        start = (int(r), int(c))
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        cells = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, h, w):
                if (nr, nc) not in seen and grid[nr, nc] == color:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        comps.append(set(cells))
    return comps


def _is_door_component(cells):
    if len(cells) != 2:
        return False
    (r0, c0), (r1, c1) = sorted(cells)
    return abs(r0 - r1) + abs(c0 - c1) == 1


def _open_side(grid, cells):
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1:
        candidates = [(-1, 0), (1, 0)]
    else:
        candidates = [(0, -1), (0, 1)]
    h, w = grid.shape
    best = candidates[0]
    best_score = -1
    for dr, dc in candidates:
        open_count = 0
        wall_count = 0
        for r, c in cells:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                wall_count += 1
            elif grid[nr, nc] == BG:
                open_count += 1
            elif grid[nr, nc] == GREY:
                wall_count += 1
        score = open_count * 3 - wall_count
        if open_count == len(cells) and score > best_score:
            best = (dr, dc)
            best_score = score
    return best


def _doors(grid):
    doors = []
    for color in DOOR_COLORS:
        for comp in _components(grid, color):
            if _is_door_component(comp):
                cells = set(comp)
                doors.append({"color": color, "cells": cells, "open": _open_side(grid, cells)})

    cell_to_door = {}
    by_color = defaultdict(list)
    for i, door in enumerate(doors):
        by_color[door["color"]].append(i)
        for cell in door["cells"]:
            cell_to_door[cell] = i
    pairs = {}
    for color, ids in by_color.items():
        if len(ids) == 2:
            pairs[ids[0]] = ids[1]
            pairs[ids[1]] = ids[0]
    return doors, cell_to_door, pairs


def _moving_blocks(grid):
    orange = {tuple(map(int, cell)) for cell in np.argwhere(grid == ORANGE)}
    blocks = []
    for comp in _components(grid, MAROON):
        votes = defaultdict(int)
        marker_cells = set()
        for r, c in comp:
            for dr, dc in DIRS:
                cell = (r + dr, c + dc)
                if cell in orange:
                    votes[(dr, dc)] += 1
                    marker_cells.add(cell)
        if votes:
            direction = max(votes, key=lambda d: (votes[d], -DIRS.index(d)))
            blocks.append({"cells": set(comp), "dir": direction, "markers": marker_cells})
    return blocks


def _exit_position(cells, exit_door, direction):
    dr, dc = direction
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    offsets = [(r - min_r, c - min_c) for r, c in cells]
    height = max(r for r, _ in offsets) + 1
    width = max(c for _, c in offsets) + 1
    door_rows = [r for r, _ in exit_door["cells"]]
    door_cols = [c for _, c in exit_door["cells"]]
    if dc == 1:
        base_r = min(door_rows)
        base_c = max(door_cols) + 1
    elif dc == -1:
        base_r = min(door_rows)
        base_c = min(door_cols) - width
    elif dr == 1:
        base_r = max(door_rows) + 1
        base_c = min(door_cols)
    else:
        base_r = min(door_rows) - height
        base_c = min(door_cols)
    return {(base_r + r, base_c + c) for r, c in offsets}


def _in_bounds(cells, h, w):
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _simulate_block(grid, cells, direction, doors, cell_to_door, pairs, used_colors):
    h, w = grid.shape
    dr, dc = direction
    pos = set(cells)
    for _ in range(h * w * 3):
        moved = {(r + dr, c + dc) for r, c in pos}
        if not _in_bounds(moved, h, w):
            break
        if any(grid[r, c] == GREY for r, c in moved):
            break
        hit_ids = [cell_to_door[cell] for cell in moved if cell in cell_to_door]
        if hit_ids:
            hit = hit_ids[0]
            color = doors[hit]["color"]
            if color in used_colors or hit not in pairs:
                break
            used_colors.add(color)
            pos = _exit_position(pos, doors[pairs[hit]], direction)
            if not _in_bounds(pos, h, w):
                break
            continue
        pos = moved
    return pos


def main(grid: np.ndarray) -> np.ndarray:
    """Move maroon blocks through paired colored door frames.

    Input structure:
    Maroon asymmetric blocks each touch one orange direction marker.  Colored
    two-pixel door frames occur in same-color pairs, with grey wall pixels or an
    edge behind one side and black space on the usable side.

    Rule:
    Erase orange markers and each block's original position.  Move every block
    along the marker direction until its next step hits a grey wall or edge.  If
    the next step enters a colored door frame, move the block to the paired door's
    open side, preserve its shape and direction, and keep moving.  Door frames and
    grey walls remain fixed.

    Key insight:
    The orange pixel supplies the movement vector while equal-colored two-cell
    door frames form teleport pairs; the two encodings combine in the motion path.
    """
    out = np.array(grid, copy=True)
    doors, cell_to_door, pairs = _doors(grid)
    blocks = _moving_blocks(grid)
    for block in blocks:
        for r, c in block["cells"] | block["markers"]:
            out[r, c] = BG

    used_colors = set()
    for block in blocks:
        final_cells = _simulate_block(grid, block["cells"], block["dir"], doors, cell_to_door, pairs, used_colors)
        for r, c in final_cells:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and out[r, c] == BG:
                out[r, c] = MAROON
    return out


SHAPE = {(0, 0), (1, 0), (1, 1)}


def _can_paint(grid, cells):
    h, w = grid.shape
    return all(0 <= r < h and 0 <= c < w and grid[r, c] == BG for r, c in cells)


def _paint_cells(grid, cells, color):
    if not _can_paint(grid, cells):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _paint_block(grid, top, left, direction):
    cells = {(top + r, left + c) for r, c in SHAPE}
    dr, dc = direction
    if direction == (0, 1):
        marker = (top + 1, left + 2)
    elif direction == (0, -1):
        marker = (top, left - 1)
    elif direction == (1, 0):
        marker = (top + 2, left)
    else:
        marker = (top - 1, left)
    if not _paint_cells(grid, cells, MAROON):
        return False
    if not _paint_cells(grid, {marker}, ORANGE):
        return False
    return True


def _paint_vertical_door(grid, top, col, color, open_side):
    cells = {(top, col), (top + 1, col)}
    wall_col = col - 1 if open_side == (0, 1) else col + 1
    wall = {(top, wall_col), (top + 1, wall_col)}
    return _paint_cells(grid, wall, GREY) and _paint_cells(grid, cells, color)


def _paint_horizontal_door(grid, row, left, color, open_side):
    cells = {(row, left), (row, left + 1)}
    wall_row = row - 1 if open_side == (1, 0) else row + 1
    wall = {(wall_row, left), (wall_row, left + 1)}
    return _paint_cells(grid, wall, GREY) and _paint_cells(grid, cells, color)


def _paint_stop_wall(grid, direction, anchor, span_start):
    dr, dc = direction
    if dc:
        col = anchor
        cells = {(span_start, col), (span_start + 1, col)}
    else:
        row = anchor
        cells = {(row, span_start), (row, span_start + 1)}
    return _paint_cells(grid, cells, GREY)


def _horizontal_case(direction):
    h = random.randint(18, 29)
    w = random.randint(20, 29)
    grid = np.zeros((h, w), dtype=int)
    color = random.choice(DOOR_COLORS)
    entry_top = random.randint(4, h - 6)
    exit_top = random.randint(3, h - 5)
    if direction == (0, 1):
        entry_col = random.randint(7, w - 11)
        exit_col = random.randint(entry_col + 4, w - 7)
        start_left = random.randint(1, entry_col - 4)
        stop_col = random.randint(exit_col + 4, w - 2)
        if not _paint_block(grid, entry_top, start_left, direction):
            return None
        if not _paint_vertical_door(grid, entry_top, entry_col, color, (0, -1)):
            return None
        if not _paint_vertical_door(grid, exit_top, exit_col, color, (0, 1)):
            return None
        if not _paint_stop_wall(grid, direction, stop_col, exit_top):
            return None
    else:
        entry_col = random.randint(9, w - 8)
        exit_col = random.randint(3, entry_col - 5)
        start_left = random.randint(entry_col + 3, w - 3)
        stop_col = random.randint(0, max(0, exit_col - 3))
        if not _paint_block(grid, entry_top, start_left, direction):
            return None
        if not _paint_vertical_door(grid, entry_top, entry_col, color, (0, 1)):
            return None
        if not _paint_vertical_door(grid, exit_top, exit_col, color, (0, -1)):
            return None
        if not _paint_stop_wall(grid, direction, stop_col, exit_top):
            return None
    return grid


def _vertical_case(direction):
    h = random.randint(20, 29)
    w = random.randint(18, 29)
    grid = np.zeros((h, w), dtype=int)
    color = random.choice(DOOR_COLORS)
    entry_left = random.randint(4, w - 6)
    exit_left = random.randint(3, w - 5)
    if direction == (1, 0):
        entry_row = random.randint(7, h - 11)
        exit_row = random.randint(entry_row + 4, h - 7)
        start_top = random.randint(1, entry_row - 4)
        stop_row = random.randint(exit_row + 4, h - 2)
        if not _paint_block(grid, start_top, entry_left, direction):
            return None
        if not _paint_horizontal_door(grid, entry_row, entry_left, color, (-1, 0)):
            return None
        if not _paint_horizontal_door(grid, exit_row, exit_left, color, (1, 0)):
            return None
        if not _paint_stop_wall(grid, direction, stop_row, exit_left):
            return None
    else:
        entry_row = random.randint(9, h - 8)
        exit_row = random.randint(3, entry_row - 5)
        start_top = random.randint(entry_row + 3, h - 3)
        stop_row = random.randint(0, max(0, exit_row - 3))
        if not _paint_block(grid, start_top, entry_left, direction):
            return None
        if not _paint_horizontal_door(grid, entry_row, entry_left, color, (1, 0)):
            return None
        if not _paint_horizontal_door(grid, exit_row, exit_left, color, (-1, 0)):
            return None
        if not _paint_stop_wall(grid, direction, stop_row, exit_left):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        direction = random.choice(DIRS)
        if direction[0] == 0:
            grid = _horizontal_case(direction)
        else:
            grid = _vertical_case(direction)
        if grid is None:
            continue
        if not np.array_equal(main(grid), grid):
            return grid
    raise RuntimeError("failed to generate a valid paired-door motion puzzle")
