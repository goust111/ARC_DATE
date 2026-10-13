from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
START = 2
END = 4
PATH = 5
AZURE = 8
BRANCH_PALETTE = (1, 3, 6, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
GENERATE_RETRIES = 400
MAX_GRID_SIDE = 30


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in DIRS:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _neighbors(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append((nr, nc))
    return out


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, str], np.ndarray]:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)
    turn_map: dict[int, str] = {}

    for comp in _components(grid == AZURE):
        assert int(comp.shape[0]) == 3, "legend arrow must be a 3-pixel L"
        r0 = int(comp[:, 0].min())
        r1 = int(comp[:, 0].max())
        c0 = int(comp[:, 1].min())
        c1 = int(comp[:, 1].max())
        assert r1 - r0 == 1 and c1 - c0 == 1, "legend arrow must fit a 2x2 box"

        block = np.zeros((2, 2), dtype=bool)
        for r, c in comp:
            block[int(r) - r0, int(c) - c0] = True
        missing = np.argwhere(~block)
        assert missing.shape == (1, 2), "legend arrow must have one missing corner"
        turn = "L" if int(missing[0, 1]) == 0 else "R"

        candidates: list[tuple[int, int, int]] = []
        rr0 = max(0, r0)
        rr1 = min(h - 1, r1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w - 1, c1 + 1)
        for rr in range(rr0, rr1 + 1):
            for cc in range(cc0, cc1 + 1):
                val = int(grid[rr, cc])
                if val != int(bg) and val != AZURE:
                    candidates.append((rr, cc, val))
        assert len(candidates) == 1, "each legend arrow must have one nearby color marker"
        rr, cc, color = candidates[0]
        if color in turn_map:
            assert turn_map[color] == turn, "duplicate legend color must agree"
        turn_map[color] = turn

        legend_mask[comp[:, 0], comp[:, 1]] = True
        legend_mask[rr, cc] = True

    assert turn_map, "legend must define at least one branch color"
    return turn_map, legend_mask


def _trace_route(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    turn_map, legend_mask = _parse_legend(g, bg)
    maze_mask = (g != int(bg)) & (~legend_mask)

    starts = np.argwhere((g == START) & maze_mask)
    ends = np.argwhere((g == END) & maze_mask)
    assert starts.shape[0] == 1 and ends.shape[0] == 1, "maze must have one red start and one yellow end"
    start = (int(starts[0, 0]), int(starts[0, 1]))
    end = (int(ends[0, 0]), int(ends[0, 1]))

    assert len(_neighbors(maze_mask, start)) == 1, "start must be a maze endpoint"
    assert len(_neighbors(maze_mask, end)) == 1, "end must be a maze endpoint"

    visited_gray: list[tuple[int, int]] = []
    branch_colors: list[int] = []
    prev: tuple[int, int] | None = None
    curr = start
    seen_states: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()

    while curr != end:
        state = (prev, curr)
        assert state not in seen_states, "route must not loop forever"
        seen_states.add(state)

        nbrs = _neighbors(maze_mask, curr)
        if prev is None:
            assert len(nbrs) == 1
            nxt = nbrs[0]
        else:
            forward = [p for p in nbrs if p != prev]
            cell_color = int(g[curr])
            if cell_color in turn_map and cell_color not in (START, END):
                branch_colors.append(cell_color)
                dr = int(curr[0] - prev[0])
                dc = int(curr[1] - prev[1])
                direction = DIR_INDEX[(dr, dc)]
                want = (direction - 1) % 4 if turn_map[cell_color] == "L" else (direction + 1) % 4
                nr = int(curr[0] + DIRS[want][0])
                nc = int(curr[1] + DIRS[want][1])
                nxt = (nr, nc)
                assert nxt in forward, "branch rule must pick a valid left/right continuation"
            else:
                assert len(forward) == 1, "non-branch maze cell must have a single forward continuation"
                nxt = forward[0]

        if curr != start and curr != end and int(g[curr]) == PATH:
            visited_gray.append(curr)
        prev, curr = curr, nxt

    return visited_gray, branch_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the colored-branch maze from the red endpoint to the yellow endpoint,
    using the corner legend to interpret each branch color as a relative left or
    right turn, and recolor only the traversed gray path pixels red.
    """
    g = np.asarray(grid, dtype=int)
    traced_gray, _ = _trace_route(g)
    out = g.copy()
    for r, c in traced_gray:
        out[int(r), int(c)] = START
    return out


def _assign_turns(colors: list[int]) -> dict[int, str]:
    if len(colors) == 2:
        turns = ["L", "R"]
    else:
        turns = ["L", "R", random.choice(["L", "R"])]
    random.shuffle(turns)
    return {int(color): turns[i] for i, color in enumerate(colors)}


def _choose_gadget_colors(colors: list[int]) -> list[int]:
    gadget_count = len(colors) + random.choice([0, 1 if len(colors) == 2 else 0])
    picked = colors[:]
    while len(picked) < gadget_count:
        picked.append(random.choice(colors))
    random.shuffle(picked)
    return [int(x) for x in picked]


def _choose_gadget_types(count: int) -> list[int]:
    types = [random.choice([3, 4]) for _ in range(count)]
    if count >= 2 and len(set(types)) == 1:
        types[random.randrange(count)] = 7 - types[0]
    return types


def _build_local_maze() -> tuple[np.ndarray, dict[int, str], list[int], list[int]]:
    num_colors = random.choice([2, 3])
    colors = random.sample(BRANCH_PALETTE, num_colors)
    turn_map = _assign_turns(colors)
    gadget_colors = _choose_gadget_colors(colors)
    gadget_types = _choose_gadget_types(len(gadget_colors))
    side_spans = [random.randint(2, 4) for _ in gadget_colors]
    vspans = [random.randint(2, 3) for _ in gadget_colors]
    top_gap = random.randint(1, 2)
    bottom_gap = random.randint(1, 2)
    inter_gaps = [random.randint(1, 2) for _ in range(len(gadget_colors) - 1)]

    center = max(side_spans)
    width = 2 * center + 1
    top_rows: list[int] = []
    bottom_rows: list[int] = []
    y = top_gap + 1
    for i, vspan in enumerate(vspans):
        top_rows.append(y)
        x = y + vspan
        bottom_rows.append(x)
        if i < len(vspans) - 1:
            y = x + inter_gaps[i] + 1
    red_row = bottom_rows[-1] + bottom_gap + 1

    local = np.full((red_row + 1, width), BG, dtype=int)
    local[0, center] = END
    local[red_row, center] = START
    local[1 : top_rows[0], center] = PATH
    for i in range(len(gadget_colors) - 1):
        local[bottom_rows[i] + 1 : top_rows[i + 1], center] = PATH
    local[bottom_rows[-1] + 1 : red_row, center] = PATH

    for i, color in enumerate(gadget_colors):
        y = top_rows[i]
        x = bottom_rows[i]
        span = side_spans[i]
        local[y, center - span : center + span + 1] = PATH
        local[x, center - span : center + span + 1] = PATH
        local[y : x + 1, center - span] = PATH
        local[y : x + 1, center + span] = PATH
        if gadget_types[i] == 4:
            local[y : x + 1, center] = PATH
        local[y, center] = color
        local[x, center] = color

    return local, turn_map, top_rows, bottom_rows


def _legend_size(count: int) -> tuple[int, int]:
    return 3 * count - 1, 3


def _corner_margins(corner: str, legend_h: int, legend_w: int) -> tuple[int, int, int, int]:
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(2, 4)
    right = random.randint(2, 4)
    if corner.startswith("t"):
        top = random.randint(legend_h, legend_h + 1)
    else:
        bottom = random.randint(legend_h, legend_h + 1)
    if corner.endswith("l"):
        left = random.randint(legend_w + 1, legend_w + 3)
    else:
        right = random.randint(legend_w + 1, legend_w + 3)
    return top, bottom, left, right


def _draw_legend(grid: np.ndarray, turn_map: dict[int, str], corner: str) -> None:
    h, w = grid.shape
    colors = list(turn_map.keys())
    random.shuffle(colors)
    legend_h, legend_w = _legend_size(len(colors))
    base_r = 0 if corner.startswith("t") else h - legend_h
    base_c = 0 if corner.endswith("l") else w - legend_w

    for i, color in enumerate(colors):
        r0 = base_r + 3 * i
        c0 = base_c
        color_row = r0 + random.choice([0, 1])
        grid[color_row, c0] = int(color)
        miss_row = random.choice([0, 1])
        miss_col = 0 if turn_map[int(color)] == "L" else 1
        for dr in range(2):
            for dc in range(2):
                if dr == miss_row and dc == miss_col:
                    continue
                grid[r0 + dr, c0 + 1 + dc] = AZURE


def _degree_counts(mask: np.ndarray) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        degrees[cell] = len(_neighbors(mask, cell))
    return degrees


def _validate_scene(grid: np.ndarray, intended_turns: dict[int, str]) -> bool:
    if max(grid.shape) > MAX_GRID_SIDE:
        return False

    parsed_turns, legend_mask = _parse_legend(grid, BG)
    if parsed_turns != intended_turns:
        return False

    maze_mask = (grid != BG) & (~legend_mask)
    degrees = _degree_counts(maze_mask)
    endpoints = []
    branch_degrees = Counter()

    for cell, degree in degrees.items():
        r, c = cell
        color = int(grid[r, c])
        if color in (START, END):
            if degree != 1:
                return False
            endpoints.append(cell)
        elif color == PATH:
            if degree != 2:
                return False
        else:
            if color not in intended_turns or degree not in (3, 4):
                return False
            branch_degrees[degree] += 1

    if len(endpoints) != 2:
        return False
    if branch_degrees[3] == 0 or branch_degrees[4] == 0:
        return False

    recolored, route_colors = _trace_route(grid)
    if set(route_colors) != set(intended_turns):
        return False
    if not recolored:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any((grid == PATH) & (out == START)):
        return False
    if not np.any(out == PATH):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    local, turn_map, _top_rows, _bottom_rows = _build_local_maze()
    legend_h, legend_w = _legend_size(len(turn_map))
    corner = random.choice(["tl", "tr", "bl", "br"])
    top, bottom, left, right = _corner_margins(corner, legend_h, legend_w)
    h = top + local.shape[0] + bottom
    w = left + local.shape[1] + right
    if h > MAX_GRID_SIDE or w > MAX_GRID_SIDE:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[top : top + local.shape[0], left : left + local.shape[1]] = local
    _draw_legend(grid, turn_map, corner)

    if not _validate_scene(grid, turn_map):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid maze-turn ARC input")
