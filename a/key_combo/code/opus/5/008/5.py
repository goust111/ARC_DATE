from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS) not in sys.path:
    sys.path.append(str(_UTILS))

from arc_utils import Color, find_components, get_bg, neighbors4

Coord = tuple[int, int]

BG = Color.BLACK
WALL = Color.GREY
PALETTE = [c for c in range(10) if c not in (BG, WALL)]

_BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
]


def _normalize_shape(cells: tuple[Coord, ...]) -> tuple[Coord, ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_variants(cells: tuple[Coord, ...]) -> list[tuple[Coord, ...]]:
    out: set[tuple[Coord, ...]] = set()
    pts = list(cells)
    for transpose in (False, True):
        for flip_r in (-1, 1):
            for flip_c in (-1, 1):
                cur = []
                for r, c in pts:
                    rr, cc = (c, r) if transpose else (r, c)
                    cur.append((rr * flip_r, cc * flip_c))
                out.add(_normalize_shape(tuple(cur)))
    return sorted(out)


SHAPE_LIBRARY = sorted({variant for shape in _BASE_SHAPES for variant in _shape_variants(shape)})


def _component_key(coords: np.ndarray) -> tuple[int, int]:
    key = tuple(map(int, coords.min(axis=0)))
    return key[0], key[1]


def _gap_orientations(grid: np.ndarray) -> dict[Coord, str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gaps: dict[Coord, str] = {}
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != BG:
                continue
            vertical = 0 < r < h - 1 and int(g[r - 1, c]) == WALL and int(g[r + 1, c]) == WALL
            horizontal = 0 < c < w - 1 and int(g[r, c - 1]) == WALL and int(g[r, c + 1]) == WALL
            if vertical ^ horizontal:
                gaps[(r, c)] = "v" if vertical else "h"
    return gaps


def _ordered_components(grid: np.ndarray) -> tuple[list[int], dict[int, list[np.ndarray]]]:
    bg = int(get_bg(grid))
    color_to_comps: dict[int, list[np.ndarray]] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {bg, WALL}):
        comps = [
            comp["coords"]
            for comp in find_components(
                grid,
                bg=bg,
                connectivity=4,
                monochromatic=True,
                colors=[color],
            )
        ]
        comps.sort(key=_component_key)
        if comps:
            color_to_comps[color] = comps
    colors = sorted(color_to_comps, key=lambda color: _component_key(color_to_comps[color][0]))
    return colors, color_to_comps


def _ordered_neighbors(cur: Coord, goal_hint: Coord) -> list[Coord]:
    out = list(neighbors4(cur[0], cur[1]))
    out.sort(key=lambda p: (abs(p[0] - goal_hint[0]) + abs(p[1] - goal_hint[1]), p[0], p[1]))
    return out


def _shortest_path(comp_a: np.ndarray, comp_b: np.ndarray, canvas: np.ndarray, bg: int, color: int) -> list[Coord]:
    h, w = canvas.shape
    starts = sorted({tuple(map(int, pt)) for pt in comp_a})
    goals = {tuple(map(int, pt)) for pt in comp_b}
    goal_hint = min(goals)

    q = deque(starts)
    prev: dict[Coord, Coord | None] = {start: None for start in starts}

    while q:
        cur = q.popleft()
        for nr, nc in _ordered_neighbors(cur, goal_hint):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if int(canvas[nr, nc]) not in (bg, color) and nxt not in goals:
                continue
            prev[nxt] = cur
            if nxt in goals:
                path = [nxt]
                step = cur
                while step is not None:
                    path.append(step)
                    step = prev[step]
                path.reverse()
                return path
            q.append(nxt)

    raise AssertionError(f"no clear path for color {color}")


def _expand_wedge(canvas: np.ndarray, bg: int, color: int, exit_cell: Coord, forward: Coord) -> list[Coord]:
    h, w = canvas.shape
    er, ec = exit_cell
    dr, dc = forward
    out: list[Coord] = []
    depth = 0
    while True:
        if dr != 0:
            row = er + dr * depth
            c0 = ec - depth
            c1 = ec + depth
            if not (0 <= row < h and 0 <= c0 and c1 < w):
                break
            strip = [(row, c) for c in range(c0, c1 + 1)]
        else:
            col = ec + dc * depth
            r0 = er - depth
            r1 = er + depth
            if not (0 <= col < w and 0 <= r0 and r1 < h):
                break
            strip = [(r, col) for r in range(r0, r1 + 1)]

        if any(int(canvas[r, c]) not in (bg, color) for r, c in strip):
            break
        out.extend(strip)
        depth += 1
    return out


def _solve(
    grid: np.ndarray,
    *,
    color_order: list[int] | None = None,
    wedge_enabled: bool = True,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    natural_order, color_to_comps = _ordered_components(g)
    if color_order is None:
        order = natural_order
    else:
        seen = set(natural_order)
        order = [int(color) for color in color_order if int(color) in seen]
        for color in natural_order:
            if color not in order:
                order.append(color)

    gaps = _gap_orientations(g)
    events = []
    for color in order:
        comps = color_to_comps[color]
        for pair_index, (comp_a, comp_b) in enumerate(zip(comps, comps[1:])):
            path = _shortest_path(comp_a, comp_b, out, bg, color)
            for r, c in path:
                if int(out[r, c]) not in (bg, color):
                    raise AssertionError("connector collided with another obstacle")
                out[r, c] = color

            wedge_cells: list[Coord] = []
            if wedge_enabled:
                seen_wedge = set()
                for idx in range(1, len(path) - 1):
                    gap = path[idx]
                    orientation = gaps.get(gap)
                    if orientation is None:
                        continue
                    prev_pt = path[idx - 1]
                    next_pt = path[idx + 1]
                    if orientation == "v":
                        if not (prev_pt[0] == gap[0] == next_pt[0] and prev_pt[1] != next_pt[1]):
                            continue
                        forward = (0, 1 if next_pt[1] > gap[1] else -1)
                    else:
                        if not (prev_pt[1] == gap[1] == next_pt[1] and prev_pt[0] != next_pt[0]):
                            continue
                        forward = (1 if next_pt[0] > gap[0] else -1, 0)
                    new_cells = _expand_wedge(out, bg, color, next_pt, forward)
                    for cell in new_cells:
                        out[cell] = color
                        if cell not in seen_wedge:
                            seen_wedge.add(cell)
                            wedge_cells.append(cell)

            events.append(
                {
                    "color": int(color),
                    "pair_index": int(pair_index),
                    "path": list(path),
                    "wedge_cells": list(wedge_cells),
                }
            )

    if return_trace:
        return out, {"color_order": list(order), "events": events}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored components through wall gaps, then grow obstacle wedges on the exit side of each crossed gap.

    Input structure:
    - The background is black, and grey cells form 1-pixel-thick horizontal or vertical wall segments, with one or two walls containing a single-cell gap.
    - Two or three other colors each appear as two or three separate 4-connected components.
    - Components are scattered on opposite sides of the walls so that at least one same-color chain must cross a wall gap.

    Rule:
    - Sort colors by the top-left position of their top-left component, using row-major order.
    - For each color, sort its original 4-connected components in row-major order and connect each consecutive pair by the shortest clear 4-neighbor path through background or the same color only.
    - Whenever such a path crosses a 1-cell wall gap, keep the path and also grow a same-colored wedge from the first exit cell beyond the gap, expanding by one cell on both sides at each step until the next strip would hit the grid edge or a non-background, non-same-color obstacle.
    - Previously painted wedges remain fixed and therefore block later colors; walls and untouched background stay unchanged.

    Key insight:
    - The hidden rule is the color-processing order: an early gap crossing creates a wedge that becomes a brand-new obstacle, so later colors must detour differently than they would on the raw input.
    """
    return _solve(grid)


def _transform_grid(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if mode == "identity":
        return g.copy()
    if mode == "mirror":
        return g[:, ::-1].copy()
    if mode == "transpose":
        return g.T.copy()
    if mode == "transpose_mirror":
        return g.T[:, ::-1].copy()
    raise ValueError(mode)


def _random_shape(rng: random.Random) -> tuple[Coord, ...]:
    candidates = [shape for shape in SHAPE_LIBRARY if 2 <= len(shape) <= 5]
    return candidates[rng.randrange(len(candidates))]


def _can_place_component(grid: np.ndarray, cells: list[Coord]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cell_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cell_set:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) != BG:
                return False
    return True


def _paint_cells(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _place_component_in_box(
    grid: np.ndarray,
    rng: random.Random,
    color: int,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    *,
    shape: tuple[Coord, ...] | None = None,
    attempts: int = 40,
) -> list[Coord] | None:
    r0, r1 = row_range
    c0, c1 = col_range
    if r0 > r1 or c0 > c1:
        return None
    for _ in range(attempts):
        cur_shape = _random_shape(rng) if shape is None else shape
        sh = 1 + max(r for r, _ in cur_shape)
        sw = 1 + max(c for _, c in cur_shape)
        if r0 + sh - 1 > r1 or c0 + sw - 1 > c1:
            if shape is not None:
                return None
            continue
        top = rng.randint(r0, r1 - sh + 1)
        left = rng.randint(c0, c1 - sw + 1)
        cells = [(top + r, left + c) for r, c in cur_shape]
        if _can_place_component(grid, cells):
            _paint_cells(grid, cells, color)
            return cells
    return None


def _gapped_wall_cells(
    orientation: str,
    start: Coord,
    len_a: int,
    len_b: int,
) -> tuple[list[Coord], Coord]:
    r0, c0 = start
    cells: list[Coord] = []
    if orientation == "v":
        for rr in range(r0, r0 + len_a):
            cells.append((rr, c0))
        gap = (r0 + len_a, c0)
        for rr in range(r0 + len_a + 1, r0 + len_a + 1 + len_b):
            cells.append((rr, c0))
        return cells, gap
    for cc in range(c0, c0 + len_a):
        cells.append((r0, cc))
    gap = (r0, c0 + len_a)
    for cc in range(c0 + len_a + 1, c0 + len_a + 1 + len_b):
        cells.append((r0, cc))
    return cells, gap


def _can_place_wall(grid: np.ndarray, cells: list[Coord]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cell_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    return True


def _paint_wall(grid: np.ndarray, cells: list[Coord]) -> bool:
    if not _can_place_wall(grid, cells):
        return False
    for r, c in cells:
        grid[r, c] = WALL
    return True


def _place_optional_gap_wall(grid: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    for _ in range(30):
        orientation = rng.choice(["h", "v"])
        len_a = rng.randint(4, 6)
        len_b = rng.randint(4, 6)
        if orientation == "h":
            row = rng.randint(max(1, h - 6), h - 3)
            c_min = max(4, w // 2)
            c_max = w - (len_a + len_b + 2)
            if c_min > c_max:
                continue
            c0 = rng.randint(c_min, c_max)
            cells, _gap = _gapped_wall_cells("h", (row, c0), len_a, len_b)
        else:
            if w - 5 > w - 3:
                continue
            col = rng.randint(w - 5, w - 3)
            r_min = 2
            r_max = max(2, h // 2 - 1)
            if r_min > r_max or r_max + len_a + len_b + 1 >= h:
                continue
            r0 = rng.randint(r_min, r_max)
            cells, _gap = _gapped_wall_cells("v", (r0, col), len_a, len_b)
        if _paint_wall(grid, cells):
            return True
    return False


def _build_canonical_input(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(17, 21)
    w = rng.randint(19, 23)
    grid = np.full((h, w), BG, dtype=int)

    colors = rng.sample(PALETTE, rng.choice([2, 3]))
    color_a, color_b = colors[0], colors[1]
    color_c = colors[2] if len(colors) == 3 else None

    wall_col = rng.randint(8, min(w // 2 + 1, w - 9))
    wedge_depth = rng.randint(3, 4)
    gap_r_min = wedge_depth + 6
    gap_r_max = h - wedge_depth - 8
    if gap_r_min > gap_r_max:
        return None
    gap_r = rng.randint(gap_r_min, gap_r_max)

    top_len = rng.randint(4, min(6, gap_r - 2))
    bot_len = rng.randint(4, min(6, h - gap_r - 3))
    main_wall, _gap = _gapped_wall_cells("v", (gap_r - top_len, wall_col), top_len, bot_len)
    if not _paint_wall(grid, main_wall):
        return None

    stop_top = rng.choice([True, False])
    stop_row = gap_r - wedge_depth if stop_top else gap_r + wedge_depth
    stop_col0 = wall_col + wedge_depth + 2
    stop_len = rng.randint(4, 6)
    stop_cells = [(stop_row, c) for c in range(stop_col0, stop_col0 + stop_len)]
    if stop_row <= 0 or stop_row >= h - 1 or stop_col0 + stop_len >= w:
        return None
    if not _paint_wall(grid, stop_cells):
        return None

    if rng.random() < 0.5:
        plain_len = rng.randint(4, 6)
        plain_col = rng.randint(wall_col + 4, w - 3)
        plain_row0 = rng.randint(1, max(1, gap_r - plain_len - 2))
        plain_cells = [(plain_row0 + k, plain_col) for k in range(plain_len)]
        _paint_wall(grid, plain_cells)

    if rng.random() < 0.5:
        _place_optional_gap_wall(grid, rng)

    if _place_component_in_box(grid, rng, color_a, (1, 3), (1, 3)) is None:
        return None
    if _place_component_in_box(grid, rng, color_a, (gap_r - 1, gap_r + 1), (wall_col - 7, wall_col - 5)) is None:
        return None
    if _place_component_in_box(grid, rng, color_a, (gap_r - 1, gap_r + 1), (wall_col + 2, wall_col + 4)) is None:
        return None

    b_col = min(w - 5, wall_col + wedge_depth)
    b1 = _place_component_in_box(
        grid,
        rng,
        color_b,
        (gap_r - wedge_depth - 3, gap_r - wedge_depth - 2),
        (b_col - 1, b_col + 1),
    )
    if b1 is None:
        return None
    b2 = _place_component_in_box(
        grid,
        rng,
        color_b,
        (gap_r + wedge_depth + 2, gap_r + wedge_depth + 3),
        (b_col - 1, b_col + 1),
    )
    if b2 is None:
        return None
    if rng.random() < 0.5:
        if _place_component_in_box(
            grid,
            rng,
            color_b,
            (min(h - 5, gap_r + wedge_depth + 4), h - 3),
            (min(w - 6, b_col + 2), w - 3),
        ) is None:
            return None

    if color_c is not None:
        if _place_component_in_box(
            grid,
            rng,
            color_c,
            (h - 7, h - 5),
            (wall_col + 2, min(w - 8, wall_col + 5)),
        ) is None:
            return None
        if _place_component_in_box(
            grid,
            rng,
            color_c,
            (h - 5, h - 3),
            (w - 7, w - 4),
        ) is None:
            return None
        if rng.random() < 0.5:
            if _place_component_in_box(
                grid,
                rng,
                color_c,
                (max(3, h - 10), h - 8),
                (w - 6, w - 3),
            ) is None:
                return None

    mode = rng.choice(["identity", "mirror", "transpose", "transpose_mirror"])
    return _transform_grid(grid, mode)


def _validate_scene(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    if max(inp.shape) > 30:
        return False
    if int(get_bg(inp)) != BG:
        return False

    gaps = _gap_orientations(inp)
    if not (1 <= len(gaps) <= 2):
        return False

    colors, color_to_comps = _ordered_components(inp)
    if len(colors) not in (2, 3):
        return False
    for color in colors:
        comps = color_to_comps[color]
        if not (2 <= len(comps) <= 3):
            return False
        if any(not (2 <= len(comp) <= 5) for comp in comps):
            return False

    try:
        out, trace = _solve(inp, return_trace=True)
        out_nowedge, trace_nowedge = _solve(inp, wedge_enabled=False, return_trace=True)
    except AssertionError:
        return False

    if np.array_equal(out, inp):
        return False
    if np.array_equal(out, out_nowedge):
        return False

    if not any(event["wedge_cells"] for event in trace["events"]):
        return False

    changed_late_path = False
    seen_wedge = False
    baseline = {(event["color"], event["pair_index"]): event["path"] for event in trace_nowedge["events"]}
    for event in trace["events"]:
        if seen_wedge:
            if baseline[(event["color"], event["pair_index"])] != event["path"]:
                changed_late_path = True
                break
        if event["wedge_cells"]:
            seen_wedge = True
    if not changed_late_path:
        return False

    try:
        reversed_out = _solve(inp, color_order=list(reversed(trace["color_order"])))
        order_matters = not np.array_equal(out, reversed_out)
    except AssertionError:
        order_matters = True
    if not order_matters:
        return False

    return True


def generate_input() -> np.ndarray:
    for seed in range(800):
        rng = random.Random(random.randrange(1 << 30) ^ (seed * 1315423911))
        grid = _build_canonical_input(rng)
        if grid is None:
            continue
        if _validate_scene(grid):
            return grid
    raise RuntimeError("failed to generate a valid wedge-gap ARC input")
