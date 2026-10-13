import random
from collections import defaultdict

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

PORTAL_ORDER = {RED: 0, ORANGE: 1, YELLOW: 2, GREEN: 3, BLUE: 4}
TEMPLATE_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, AZURE]

PATTERNS = [
    [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    [(0, 0), (-1, -1), (-1, 0), (0, 1), (1, 1)],
    [(0, 0), (-1, 0), (0, 1), (1, 0)],
    [(0, 0), (0, -1), (0, 1), (1, -1), (1, 1)],
    [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)],
]


def _separator_lines(grid):
    h, w = grid.shape
    sep_colors = set([GREY, *PORTAL_ORDER.keys()])
    rows = []
    for r in range(h):
        count = sum(int(v) in sep_colors for v in grid[r])
        if count >= max(3, int(0.75 * w)):
            rows.append(r)
    cols = []
    for c in range(w):
        count = sum(int(v) in sep_colors for v in grid[:, c])
        if count >= max(3, int(0.75 * h)):
            cols.append(c)
    return rows, cols


def _rooms(grid):
    h, w = grid.shape
    sep_rows, sep_cols = _separator_lines(grid)
    row_edges = [-1] + sep_rows + [h]
    col_edges = [-1] + sep_cols + [w]
    result = []
    for ri in range(len(row_edges) - 1):
        r0, r1 = row_edges[ri] + 1, row_edges[ri + 1]
        if r1 <= r0:
            continue
        for ci in range(len(col_edges) - 1):
            c0, c1 = col_edges[ci] + 1, col_edges[ci + 1]
            if c1 <= c0:
                continue
            result.append(
                {
                    "idx": len(result),
                    "ri": ri,
                    "ci": ci,
                    "bounds": (r0, r1, c0, c1),
                }
            )
    return result, sep_rows, sep_cols, row_edges, col_edges


def _room_lookup(rooms):
    by_pos = {}
    for room in rooms:
        by_pos[(room["ri"], room["ci"])] = room["idx"]
    return by_pos


def _portal_priorities(grid, rooms, sep_rows, sep_cols, row_edges, col_edges):
    by_pos = _room_lookup(rooms)
    priorities = defaultdict(list)
    portal_cells = []

    sep_row_to_i = {r: i for i, r in enumerate(sep_rows)}
    sep_col_to_i = {c: i for i, c in enumerate(sep_cols)}

    for r, c in zip(*np.where(np.isin(grid, list(PORTAL_ORDER.keys())))):
        color = int(grid[r, c])
        rank = PORTAL_ORDER[color]
        portal_cells.append((r, c))
        if r in sep_row_to_i:
            wall_i = sep_row_to_i[r]
            ci = None
            for k in range(len(col_edges) - 1):
                if col_edges[k] < c < col_edges[k + 1]:
                    ci = k
                    break
            if ci is not None:
                for ri in (wall_i, wall_i + 1):
                    idx = by_pos.get((ri, ci))
                    if idx is not None:
                        priorities[idx].append(rank)
        if c in sep_col_to_i:
            wall_i = sep_col_to_i[c]
            ri = None
            for k in range(len(row_edges) - 1):
                if row_edges[k] < r < row_edges[k + 1]:
                    ri = k
                    break
            if ri is not None:
                for ci in (wall_i, wall_i + 1):
                    idx = by_pos.get((ri, ci))
                    if idx is not None:
                        priorities[idx].append(rank)

    return priorities, portal_cells


def _template_and_anchors(grid, room):
    r0, r1, c0, c1 = room["bounds"]
    corner_r1 = min(r1, r0 + 3)
    corner_c1 = min(c1, c0 + 3)
    cells = []
    color_counts = defaultdict(int)
    for r in range(r0, corner_r1):
        for c in range(c0, corner_c1):
            color = int(grid[r, c])
            if color not in (BLACK, GREY, MAROON) and color not in PORTAL_ORDER:
                cells.append((r, c, color))
                color_counts[color] += 1
            elif color in TEMPLATE_COLORS:
                cells.append((r, c, color))
                color_counts[color] += 1
    if not color_counts:
        return None, []

    template_color = max(color_counts, key=color_counts.get)
    template_cells = [(r, c) for r, c, color in cells if color == template_color]
    rows = [r for r, _ in template_cells]
    cols = [c for _, c in template_cells]
    center = ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)
    offsets = [(r - center[0], c - center[1]) for r, c in template_cells]

    anchors = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if int(grid[r, c]) == MAROON:
                anchors.append((r, c))
    return (template_color, offsets), anchors


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    rooms, sep_rows, sep_cols, row_edges, col_edges = _rooms(grid)
    priorities, portal_cells = _portal_priorities(
        grid, rooms, sep_rows, sep_cols, row_edges, col_edges
    )

    def room_key(room):
        ranks = priorities.get(room["idx"])
        return (min(ranks) if ranks else 99, room["idx"])

    original_mutable = (grid == BLACK) | (grid == MAROON)
    for room in sorted(rooms, key=room_key):
        template, anchors = _template_and_anchors(grid, room)
        if template is None:
            continue
        color, offsets = template
        for ar, ac in anchors:
            for dr, dc in offsets:
                rr, cc = ar + dr, ac + dc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    if original_mutable[rr, cc]:
                        out[rr, cc] = color

    for r, c in portal_cells:
        out[r, c] = GREY
    return out


def _draw_template(grid, r0, c0, color, pattern):
    center = (r0 + 1, c0 + 1)
    for dr, dc in pattern:
        grid[center[0] + dr, center[1] + dc] = color


def generate_input() -> np.ndarray:
    rows = random.choice([2, 3])
    cols = random.choice([2, 3])
    room_h = random.randint(6, 8)
    room_w = random.randint(6, 9)
    h = rows * room_h + (rows - 1)
    w = cols * room_w + (cols - 1)
    grid = np.zeros((h, w), dtype=int)

    for r in range(room_h, h, room_h + 1):
        grid[r, :] = GREY
    for c in range(room_w, w, room_w + 1):
        grid[:, c] = GREY

    room_bounds = []
    for ri in range(rows):
        r0 = ri * (room_h + 1)
        r1 = r0 + room_h
        for ci in range(cols):
            c0 = ci * (room_w + 1)
            c1 = c0 + room_w
            room_bounds.append((r0, r1, c0, c1))

    colors = [random.choice(TEMPLATE_COLORS) for _ in room_bounds]
    for idx, (r0, r1, c0, c1) in enumerate(room_bounds):
        pattern = random.choice(PATTERNS)
        _draw_template(grid, r0, c0, colors[idx], pattern)
        anchor_count = random.randint(1, 3)
        used = set(zip(*np.where(grid != BLACK)))
        for _ in range(anchor_count):
            for _attempt in range(100):
                ar = random.randint(r0 + 2, r1 - 2)
                ac = random.randint(c0 + 2, c1 - 2)
                if (ar, ac) not in used:
                    grid[ar, ac] = MAROON
                    used.add((ar, ac))
                    break

    portal_colors = random.sample(list(PORTAL_ORDER), random.randint(1, min(4, len(PORTAL_ORDER))))
    for color in portal_colors:
        if random.random() < 0.5 and rows > 1:
            wall_ri = random.randrange(rows - 1)
            ci = random.randrange(cols)
            r = (wall_ri + 1) * room_h + wall_ri
            c0 = ci * (room_w + 1)
            c = random.randint(c0 + 1, c0 + room_w - 2)
        else:
            wall_ci = random.randrange(cols - 1)
            ri = random.randrange(rows)
            c = (wall_ci + 1) * room_w + wall_ci
            r0 = ri * (room_h + 1)
            r = random.randint(r0 + 1, r0 + room_h - 2)
        grid[r, c] = color

    return grid


if __name__ == "__main__":
    for _ in range(100):
        sample = generate_input()
        result = main(sample)
        assert isinstance(sample, np.ndarray)
        assert isinstance(result, np.ndarray)
