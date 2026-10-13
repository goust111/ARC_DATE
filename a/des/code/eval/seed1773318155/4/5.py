from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
PIPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _most_common_non_bg(grid: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(grid[grid != bg], return_counts=True)
    assert vals.size > 0, "expected at least one non-background color"
    return int(vals[counts.argmax()])


def _side_of(r: int, c: int, h: int, w: int) -> str | None:
    if r == 0 and 0 < c < w - 1:
        return "top"
    if r == h - 1 and 0 < c < w - 1:
        return "bottom"
    if c == 0 and 0 < r < h - 1:
        return "left"
    if c == w - 1 and 0 < r < h - 1:
        return "right"
    return None


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a == b:
        return [a]
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment endpoints must be aligned")


def _bfs_path(
    start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray
) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None
    if not (0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if not bool(walkable[start]) or not bool(walkable[goal]):
        return None

    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in _neighbors4(*cur):
            nr, nc = nxt
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in prev or not bool(walkable[nr, nc]):
                continue
            prev[nxt] = cur
            q.append(nxt)

    if goal not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _find_release_start(
    grid: np.ndarray, exit_cell: tuple[int, int], side: str, grey: int
) -> tuple[int, int] | None:
    h, w = grid.shape
    dr, dc = SIDE_TO_DIR[side]
    r, c = exit_cell[0] + dr, exit_cell[1] + dc
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == grey:
        r += dr
        c += dc
    if not (0 <= r < h and 0 <= c < w):
        return None
    return (r, c)


def _release_cells_from_path(
    path: list[tuple[int, int]], exit_side: str
) -> list[tuple[int, int]]:
    dr, dc = SIDE_TO_DIR[exit_side]
    release = [path[-1]]
    idx = len(path) - 1
    while idx - 1 >= 0:
        rr, cc = path[idx]
        pr, pc = path[idx - 1]
        if (pr, pc) != (rr + dr, cc + dc):
            break
        release.append((pr, pc))
        idx -= 1
    return release


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored corridor from its entry to its matching exit, collect the
    single-pixel objects encountered along the way, clear them, then release the
    collected sequence back along the straight inward tail behind the exit cap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    grey = _most_common_non_bg(g, bg)

    border_by_color: dict[int, list[tuple[int, int, str]]] = {}
    for r in range(h):
        for c in range(w):
            side = _side_of(r, c, h, w)
            if side is None:
                continue
            v = int(g[r, c])
            if v in (bg, grey):
                continue
            border_by_color.setdefault(v, []).append((r, c, side))

    endpoint_cells = {
        (r, c)
        for pts in border_by_color.values()
        for r, c, _side in pts
    }
    walkable = g != grey
    for r, c in endpoint_cells:
        walkable[r, c] = False

    out = g.copy()

    for color, pts in border_by_color.items():
        if len(pts) != 2:
            continue

        entry: tuple[int, int, str] | None = None
        exit_cell: tuple[int, int, str] | None = None
        for r, c, side in pts:
            dr, dc = SIDE_TO_DIR[side]
            ir, ic = r + dr, c + dc
            if not (0 <= ir < h and 0 <= ic < w):
                continue
            if int(g[ir, ic]) == grey:
                exit_cell = (r, c, side)
            else:
                entry = (r, c, side)

        assert entry is not None and exit_cell is not None, f"color {color} must have one entry and one exit"

        er, ec, entry_side = entry
        dr_in, dc_in = SIDE_TO_DIR[entry_side]
        start = (er + dr_in, ec + dc_in)

        xr, xc, exit_side = exit_cell
        goal = _find_release_start(g, (xr, xc), exit_side, grey)
        assert goal is not None, f"exit for color {color} must lead to a release tail"

        path = _bfs_path(start, goal, walkable)
        assert path is not None, f"no corridor path found for color {color}"

        collected: list[int] = []
        object_cells: list[tuple[int, int]] = []
        for r, c in path:
            v = int(out[r, c])
            if v != bg:
                collected.append(v)
                object_cells.append((r, c))

        for r, c in object_cells:
            out[r, c] = bg

        release_cells = _release_cells_from_path(path, exit_side)
        assert len(release_cells) >= len(collected), f"release tail too short for color {color}"
        for (r, c), obj_color in zip(release_cells, collected, strict=False):
            out[r, c] = int(obj_color)

    return out


def _entry_cells(side: str, idx: int, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    dr, dc = SIDE_TO_DIR[side]
    if side == "top":
        border = (0, idx)
    elif side == "bottom":
        border = (h - 1, idx)
    elif side == "left":
        border = (idx, 0)
    else:
        border = (idx, w - 1)
    start = (border[0] + dr, border[1] + dc)
    hub = (start[0] + dr, start[1] + dc)
    return border, start, hub


def _exit_cells(
    side: str, idx: int, h: int, w: int, tail_len: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    dr, dc = SIDE_TO_DIR[side]
    if side == "top":
        border = (0, idx)
    elif side == "bottom":
        border = (h - 1, idx)
    elif side == "left":
        border = (idx, 0)
    else:
        border = (idx, w - 1)

    cap = (border[0] + dr, border[1] + dc)
    goal = (cap[0] + dr, cap[1] + dc)
    tail_end = (goal[0] + dr * (tail_len - 1), goal[1] + dc * (tail_len - 1))
    return border, cap, goal, tail_end


def _route_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert points
    path = [points[0]]
    for a, b in zip(points, points[1:], strict=False):
        if a == b:
            continue
        choices: list[list[tuple[int, int]]] = []
        if a[0] == b[0] or a[1] == b[1]:
            choices.append(_segment(a, b))
        else:
            c1 = (a[0], b[1])
            c2 = (b[0], a[1])
            choices.append(_segment(a, c1) + _segment(c1, b)[1:])
            choices.append(_segment(a, c2) + _segment(c2, b)[1:])
        seg = random.choice(choices)
        path.extend(seg[1:])
    return path


def _valid_pipe_path(
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    h: int,
    w: int,
) -> bool:
    if not path or path[0] != start or path[-1] != goal:
        return False

    seen: set[tuple[int, int]] = set()
    for idx, (r, c) in enumerate(path):
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        if idx != 0 and (r in (1, h - 2) or c in (1, w - 2)):
            return False
        if (r, c) in seen:
            return False
        seen.add((r, c))

    path_set = set(path)
    for idx, (r, c) in enumerate(path):
        expected = set()
        if idx > 0:
            expected.add(path[idx - 1])
        if idx + 1 < len(path):
            expected.add(path[idx + 1])
        actual = {(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in path_set}
        if actual != expected:
            return False
    return True


def _wall_cells(path: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    path_set = set(path)
    walls: set[tuple[int, int]] = set()
    for r, c in path:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in path_set:
                walls.add((nr, nc))
    return walls


def _pipe_metadata(
    h: int,
    w: int,
    color: int,
    object_palette: list[int],
    occupied: set[tuple[int, int]],
) -> dict[str, object] | None:
    side_in, side_out = random.sample(list(SIDES), 2)
    object_count = random.randint(1, min(4, len(object_palette)))
    tail_len = random.randint(object_count, object_count + 3)

    edge_margin = 2
    if h < 9 or w < 9:
        return None

    if side_in in ("left", "right"):
        in_idx = random.randint(edge_margin, h - edge_margin - 1)
    else:
        in_idx = random.randint(edge_margin, w - edge_margin - 1)

    if side_out in ("left", "right"):
        if tail_len > w - 3:
            return None
        out_idx = random.randint(edge_margin, h - edge_margin - 1)
    else:
        if tail_len > h - 3:
            return None
        out_idx = random.randint(edge_margin, w - edge_margin - 1)

    entry_border, start, hub = _entry_cells(side_in, in_idx, h, w)
    exit_border, cap, goal, tail_end = _exit_cells(side_out, out_idx, h, w, tail_len)

    for cell in (start, hub, cap, goal, tail_end):
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            return None
    if hub == tail_end:
        return None

    for _ in range(120):
        via_count = random.choice((0, 1, 2))
        vias: list[tuple[int, int]] = []
        for _ in range(via_count):
            vias.append((random.randint(2, h - 3), random.randint(2, w - 3)))

        middle = _route_points([hub] + vias + [tail_end])
        path = [start] + middle
        path.extend(_segment(tail_end, goal)[1:])

        if not _valid_pipe_path(path, start, goal, h, w):
            continue

        walls = _wall_cells(path, h, w)
        footprint = set(path) | walls | {entry_border, exit_border, cap}
        if footprint & occupied:
            continue

        internal = path[1:-1]
        if len(internal) < object_count:
            continue

        object_cells = random.sample(internal, object_count)
        order = {cell: idx for idx, cell in enumerate(path)}
        object_cells.sort(key=order.get)
        object_colors = random.sample(object_palette, object_count)
        release_cells = [(goal[0] + i * SIDE_TO_DIR[side_out][0], goal[1] + i * SIDE_TO_DIR[side_out][1]) for i in range(tail_len)]

        return {
            "color": color,
            "entry_side": side_in,
            "exit_side": side_out,
            "entry_border": entry_border,
            "exit_border": exit_border,
            "cap": cap,
            "goal": goal,
            "path": path,
            "footprint": footprint,
            "walls": walls,
            "object_cells": object_cells,
            "object_colors": object_colors,
            "release_cells": release_cells,
        }

    return None


def _render_scene(h: int, w: int, pipes: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    for pipe in pipes:
        for r, c in pipe["walls"]:
            grid[r, c] = GREY
            expected[r, c] = GREY
        cr, cc = pipe["cap"]
        grid[cr, cc] = GREY
        expected[cr, cc] = GREY

    for pipe in pipes:
        er, ec = pipe["entry_border"]
        xr, xc = pipe["exit_border"]
        color = int(pipe["color"])
        grid[er, ec] = color
        grid[xr, xc] = color
        expected[er, ec] = color
        expected[xr, xc] = color

        for (r, c), obj_color in zip(pipe["object_cells"], pipe["object_colors"], strict=False):
            grid[r, c] = int(obj_color)

        order = {cell: idx for idx, cell in enumerate(pipe["path"])}
        ordered_objects = sorted(
            zip(pipe["object_cells"], pipe["object_colors"], strict=False),
            key=lambda item: order[item[0]],
        )
        for obj_idx, (_cell, obj_color) in enumerate(ordered_objects):
            rr, cc = pipe["release_cells"][obj_idx]
            expected[rr, cc] = int(obj_color)

    return grid, expected


def _valid_scene(grid: np.ndarray, expected: np.ndarray, pipes: list[dict[str, object]]) -> bool:
    if _most_common_color(grid) != BG:
        return False
    if _most_common_non_bg(grid, BG) != GREY:
        return False

    counts = Counter(int(v) for v in grid.ravel())
    if counts[GREY] <= max(counts[c] for c in PIPE_COLORS if c != GREY):
        return False

    for pipe in pipes:
        color = int(pipe["color"])
        locs = np.argwhere(grid == color)
        border_hits = [
            (int(r), int(c))
            for r, c in locs
            if _side_of(int(r), int(c), grid.shape[0], grid.shape[1]) is not None
        ]
        if len(border_hits) != 2:
            return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    pipe_count = random.randint(1, 3)

    pipe_colors = random.sample(list(PIPE_COLORS), pipe_count)
    object_palette = [c for c in PIPE_COLORS if c not in pipe_colors]
    if len(object_palette) < 2:
        return None

    occupied: set[tuple[int, int]] = set()
    pipes: list[dict[str, object]] = []

    for color in pipe_colors:
        pipe = None
        for _ in range(120):
            pipe = _pipe_metadata(h, w, color, object_palette, occupied)
            if pipe is not None:
                break
        if pipe is None:
            return None
        occupied |= set(pipe["footprint"])
        pipes.append(pipe)

    grid, expected = _render_scene(h, w, pipes)
    if not _valid_scene(grid, expected, pipes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
