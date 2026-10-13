from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
ROOM_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_GEN_ATTEMPTS = 400
LAYOUT_CHOICES = [(2, 2), (2, 3), (3, 2), (3, 3), (1, 4), (4, 1)]
ROOM_COUNT_RANGE = (3, 5)
CELL_INNER_RANGE = (4, 7)
FRAGMENT_COUNT_RANGE = (4, 8)


def neighbors4(r: int, c: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _door_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    door = np.zeros_like(g, dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BG:
                continue
            vertical_gap = (
                int(g[r - 1, c]) == WALL
                and int(g[r + 1, c]) == WALL
                and int(g[r, c - 1]) != WALL
                and int(g[r, c + 1]) != WALL
            )
            horizontal_gap = (
                int(g[r, c - 1]) == WALL
                and int(g[r, c + 1]) == WALL
                and int(g[r - 1, c]) != WALL
                and int(g[r + 1, c]) != WALL
            )
            if vertical_gap or horizontal_gap:
                door[r, c] = True
    return door


def _room_data(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    doors = _door_mask(g)
    passable = (g != WALL) & (~doors)
    comps = components_mask(passable)
    room_id = -np.ones(g.shape, dtype=int)
    rooms = []

    for idx, comp in enumerate(comps):
        for r, c in comp:
            room_id[r, c] = idx

        colors = [int(g[r, c]) for r, c in comp if int(g[r, c]) not in (BG, WALL)]
        assert len(set(colors)) >= 2, "each room must contain at least two fragment colors"
        counts = Counter(colors).most_common()
        assert counts[0][1] > counts[1][1], "room majority color must be unique"

        black_count = sum(int(g[r, c]) == BG for r, c in comp)
        rooms.append(
            {
                "cells": comp,
                "camp": counts[0][0],
                "black_count": black_count,
            }
        )

    return rooms, room_id, doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vote within each walled room, unify its fragments, and color every doorway by the winning adjacent camp.

    Input structure:
    - Black background with grey wall pixels forming multiple rooms and narrow wall gaps.
    - Each room contains scattered single-pixel colored fragments from at least two colors.
    - A doorway is a one-cell black gap in an otherwise straight grey wall, connecting two rooms.

    Rule:
    - Temporarily treat each doorway gap as a wall to recover the separate rooms.
    - In each room, count fragment colors; the unique majority color is that room's camp.
    - Recolor every non-black, non-grey fragment in the room to the camp color.
    - For each doorway, if both adjacent rooms share the same camp, paint the doorway that color; otherwise paint it with the camp color of the adjacent room that has more black interior cells.

    Key insight:
    The black doorway cells are not part of any room vote; they must first be recognized as wall gaps so the room-majority vote and the doorway size comparison can both be computed correctly.
    """
    g = np.asarray(grid, dtype=int)
    rooms, room_id, door_mask = _room_data(g)

    out = g.copy()
    for room in rooms:
        camp = int(room["camp"])
        for r, c in room["cells"]:
            if int(g[r, c]) not in (BG, WALL):
                out[r, c] = camp

    for r, c in np.argwhere(door_mask):
        adjacent = {
            int(room_id[nr, nc])
            for nr, nc in neighbors4(int(r), int(c))
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and room_id[nr, nc] >= 0
        }
        assert len(adjacent) == 2, "each doorway must connect exactly two rooms"
        a, b = sorted(adjacent)
        room_a = rooms[a]
        room_b = rooms[b]

        if int(room_a["camp"]) == int(room_b["camp"]):
            out[int(r), int(c)] = int(room_a["camp"])
            continue

        count_a = int(room_a["black_count"])
        count_b = int(room_b["black_count"])
        assert count_a != count_b, "different-camp doorway requires unequal room black areas"
        out[int(r), int(c)] = int(room_a["camp"] if count_a > count_b else room_b["camp"])

    return out


def _cell_boxes(heights: list[int], widths: list[int]):
    boxes = {}
    row_starts = []
    cursor = 1
    for h in heights:
        row_starts.append(cursor)
        cursor += h + 1
    col_starts = []
    cursor = 1
    for w in widths:
        col_starts.append(cursor)
        cursor += w + 1

    for r, top in enumerate(row_starts):
        for c, left in enumerate(col_starts):
            boxes[(r, c)] = (top, left, top + heights[r] - 1, left + widths[c] - 1)
    return boxes


def _grid_dims(heights: list[int], widths: list[int]) -> tuple[int, int]:
    return sum(heights) + len(heights) + 1, sum(widths) + len(widths) + 1


def _random_path(rows: int, cols: int, length: int, rng: random.Random):
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(80):
        path = [rng.choice(cells)]
        used = {path[0]}
        while len(path) < length:
            r, c = path[-1]
            neighbors = []
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in used:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            nxt = rng.choice(neighbors)
            used.add(nxt)
            path.append(nxt)
        if len(path) == length:
            return path
    return None


def _majority_sequence(length: int, rng: random.Random) -> list[int]:
    palette = rng.sample(ROOM_COLORS, k=rng.randint(3, min(5, len(ROOM_COLORS))))
    relation_count = length - 1
    while True:
        relations = [rng.choice(["same", "diff"]) for _ in range(relation_count)]
        if "same" in relations and "diff" in relations:
            break

    majors = [rng.choice(palette)]
    for rel in relations:
        if rel == "same":
            majors.append(majors[-1])
        else:
            choices = [color for color in palette if color != majors[-1]]
            majors.append(rng.choice(choices))
    return majors


def _fragment_color_multiset(majority: int, total: int, rng: random.Random) -> list[int]:
    other_colors = [c for c in ROOM_COLORS if c != majority]
    minority_k = 1 if total <= 5 else rng.choice([1, 2])

    if minority_k == 1:
        majority_count = rng.randint(max(2, total // 2 + 1), total - 1)
        counts = [majority_count, total - majority_count]
    else:
        majority_count = rng.randint(max(3, total // 2 + 1), total - 2)
        rem = total - majority_count
        split = rng.randint(1, rem - 1)
        counts = [majority_count, split, rem - split]

    colors = [majority] * counts[0]
    minority_colors = rng.sample(other_colors, k=minority_k)
    for color, count in zip(minority_colors, counts[1:]):
        colors.extend([color] * count)
    rng.shuffle(colors)
    return colors


def _door_cell(a: tuple[int, int, int, int], b: tuple[int, int, int, int], rng: random.Random):
    top_a, left_a, bottom_a, right_a = a
    top_b, left_b, bottom_b, right_b = b
    if right_a + 2 == left_b or right_b + 2 == left_a:
        wall_col = right_a + 1 if right_a < left_b else right_b + 1
        row_lo = max(top_a, top_b) + 1
        row_hi = min(bottom_a, bottom_b) - 1
        return rng.randint(row_lo, row_hi), wall_col
    wall_row = bottom_a + 1 if bottom_a < top_b else bottom_b + 1
    col_lo = max(left_a, left_b) + 1
    col_hi = min(right_a, right_b) - 1
    return wall_row, rng.randint(col_lo, col_hi)


def _gen_attempt(rng: random.Random):
    rows, cols = rng.choice(LAYOUT_CHOICES)
    room_count = rng.randint(ROOM_COUNT_RANGE[0], min(ROOM_COUNT_RANGE[1], rows * cols))

    heights = [rng.randint(*CELL_INNER_RANGE) for _ in range(rows)]
    widths = [rng.randint(*CELL_INNER_RANGE) for _ in range(cols)]
    h, w = _grid_dims(heights, widths)
    if h > 30 or w > 30:
        return None

    path = _random_path(rows, cols, room_count, rng)
    if path is None:
        return None

    majors = _majority_sequence(room_count, rng)
    boxes = _cell_boxes(heights, widths)

    grid = np.full((h, w), WALL, dtype=int)
    room_cells = []
    for cell in path:
        top, left, bottom, right = boxes[cell]
        grid[top : bottom + 1, left : right + 1] = BG
        room_cells.append([(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)])

    edges = list(zip(path, path[1:]))
    for a, b in edges:
        dr = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if dr != 1:
            return None
        door_r, door_c = _door_cell(boxes[a], boxes[b], rng)
        grid[door_r, door_c] = BG

    room_fragment_totals = []
    for i, cells in enumerate(room_cells):
        capacity = len(cells)
        max_fragments = min(FRAGMENT_COUNT_RANGE[1], capacity - 4)
        min_fragments = min(FRAGMENT_COUNT_RANGE[0], max_fragments)
        if max_fragments < 4:
            return None
        fragment_total = rng.randint(min_fragments, max_fragments)
        room_fragment_totals.append(fragment_total)

        coords = rng.sample(cells, k=fragment_total)
        colors = _fragment_color_multiset(majors[i], fragment_total, rng)
        for (r, c), color in zip(coords, colors):
            grid[r, c] = color

    for i, (a, b) in enumerate(edges):
        ia = path.index(a)
        ib = path.index(b)
        if majors[ia] != majors[ib]:
            black_a = len(room_cells[ia]) - room_fragment_totals[ia]
            black_b = len(room_cells[ib]) - room_fragment_totals[ib]
            if black_a == black_b:
                return None

    try:
        rooms, _, door_mask = _room_data(grid)
    except AssertionError:
        return None

    if len(rooms) != room_count or int(np.sum(door_mask)) != len(edges):
        return None

    door_pairs = []
    room_id = -np.ones(grid.shape, dtype=int)
    passable = (grid != WALL) & (~door_mask)
    for idx, comp in enumerate(components_mask(passable)):
        for r, c in comp:
            room_id[r, c] = idx
    for r, c in np.argwhere(door_mask):
        adjacent = {
            int(room_id[nr, nc])
            for nr, nc in neighbors4(int(r), int(c))
            if 0 <= nr < h and 0 <= nc < w and room_id[nr, nc] >= 0
        }
        if len(adjacent) != 2:
            return None
        door_pairs.append(tuple(sorted(adjacent)))

    same_branch = False
    diff_branch = False
    for a, b in door_pairs:
        if int(rooms[a]["camp"]) == int(rooms[b]["camp"]):
            same_branch = True
        else:
            diff_branch = True
    if not (same_branch and diff_branch):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-voting puzzle")
