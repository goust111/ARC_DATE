from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
WALL = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _same_color_components(grid: np.ndarray, mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_doors(grid: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    colored = (grid != BLACK) & (grid != WALL)
    doors: dict[tuple[int, int], int] = {}
    for comp in _same_color_components(grid, colored):
        if len(comp) != 1:
            continue
        r, c = comp[0]
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        vertical = (
            int(grid[r - 1, c]) == WALL
            and int(grid[r + 1, c]) == WALL
            and int(grid[r, c - 1]) != WALL
            and int(grid[r, c + 1]) != WALL
        )
        horizontal = (
            int(grid[r, c - 1]) == WALL
            and int(grid[r, c + 1]) == WALL
            and int(grid[r - 1, c]) != WALL
            and int(grid[r + 1, c]) != WALL
        )
        if vertical or horizontal:
            doors[(r, c)] = int(grid[r, c])
    return doors


def _touches_border(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each room interior by the color of the door used to enter it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    doors = _find_doors(g)
    door_mask = np.zeros((h, w), dtype=bool)
    for r, c in doors:
        door_mask[r, c] = True

    passable = (g != WALL) & ~door_mask
    room_comps = [comp for comp in _components(passable) if not _touches_border(comp, h, w)]
    assert room_comps, "expected enclosed room interiors"

    room_of: dict[tuple[int, int], int] = {}
    for rid, comp in enumerate(room_comps):
        for cell in comp:
            room_of[cell] = rid

    source_rooms = []
    source_color = None
    for rid, comp in enumerate(room_comps):
        colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != BLACK}
        if colors:
            assert len(colors) == 1, "source room should have one fill color"
            source_rooms.append(rid)
            source_color = next(iter(colors))
    assert len(source_rooms) == 1, "expected exactly one initially filled source room"
    assert source_color is not None
    source_room = source_rooms[0]

    graph: dict[int, list[tuple[int, int]]] = {rid: [] for rid in range(len(room_comps))}
    for (r, c), door_color in doors.items():
        neigh_rooms = {
            room_of[(nr, nc)]
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in room_of
        }
        assert len(neigh_rooms) == 2, "each door must separate exactly two rooms"
        a, b = tuple(neigh_rooms)
        graph[a].append((b, door_color))
        graph[b].append((a, door_color))

    room_colors = {source_room: source_color}
    q = deque([source_room])
    while q:
        room = q.popleft()
        for nxt, door_color in graph[room]:
            if nxt not in room_colors:
                room_colors[nxt] = door_color
                q.append(nxt)

    out = g.copy()
    for rid, color in room_colors.items():
        for r, c in room_comps[rid]:
            out[r, c] = color
    return out


def _choice_in_span(rng: random.Random, start: int, length: int) -> int:
    vals = list(range(start, start + length))
    inner = vals[1:-1] if len(vals) >= 3 else vals
    return rng.choice(inner)


def _random_tree(rows: int, cols: int, rng: random.Random) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    def nbrs(node: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = node
        out = []
        if r > 0:
            out.append((r - 1, c))
        if r + 1 < rows:
            out.append((r + 1, c))
        if c > 0:
            out.append((r, c - 1))
        if c + 1 < cols:
            out.append((r, c + 1))
        return out

    start = (rng.randrange(rows), rng.randrange(cols))
    seen = {start}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(seen) < rows * cols:
        frontier = [(u, v) for u in seen for v in nbrs(u) if v not in seen]
        u, v = rng.choice(frontier)
        edges.append((u, v))
        seen.add(v)
    return edges


def _room_starts(sizes: list[int], offset: int) -> list[int]:
    starts = []
    cur = offset + 1
    for size in sizes:
        starts.append(cur)
        cur += size + 1
    return starts


def _build_sample(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    room_rows = rng.randint(2, 4)
    room_cols = rng.randint(2, 4)
    heights = [rng.randint(2, 5) for _ in range(room_rows)]
    widths = [rng.randint(2, 5) for _ in range(room_cols)]

    block_h = sum(heights) + room_rows + 1
    block_w = sum(widths) + room_cols + 1

    flush_side = rng.choice(["top", "bottom", "left", "right"])
    top_margin = 0 if flush_side == "top" else rng.randint(0, 3)
    bottom_margin = 0 if flush_side == "bottom" else rng.randint(0, 3)
    left_margin = 0 if flush_side == "left" else rng.randint(0, 3)
    right_margin = 0 if flush_side == "right" else rng.randint(0, 3)

    h = block_h + top_margin + bottom_margin
    w = block_w + left_margin + right_margin
    grid = np.full((h, w), BLACK, dtype=int)

    row_starts = _room_starts(heights, top_margin)
    col_starts = _room_starts(widths, left_margin)
    wall_rows = [top_margin] + [row_starts[i] + heights[i] for i in range(room_rows)]
    wall_cols = [left_margin] + [col_starts[j] + widths[j] for j in range(room_cols)]

    for wr in wall_rows:
        grid[wr, left_margin : left_margin + block_w] = WALL
    for wc in wall_cols:
        grid[top_margin : top_margin + block_h, wc] = WALL

    rooms = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    if flush_side == "top":
        source_room = (0, rng.randrange(room_cols))
    elif flush_side == "bottom":
        source_room = (room_rows - 1, rng.randrange(room_cols))
    elif flush_side == "left":
        source_room = (rng.randrange(room_rows), 0)
    else:
        source_room = (rng.randrange(room_rows), room_cols - 1)

    source_color = rng.choice(COLOR_POOL)
    door_colors = [color for color in COLOR_POOL if color != source_color]
    edges = _random_tree(room_rows, room_cols, rng)

    room_color_truth: dict[tuple[int, int], int] = {source_room: source_color}
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {room: [] for room in rooms}
    for a, b in edges:
        door_color = rng.choice(door_colors)
        graph[a].append((b, door_color))
        graph[b].append((a, door_color))
        if a[0] == b[0]:
            row = a[0]
            left_col = min(a[1], b[1])
            wall_c = col_starts[left_col] + widths[left_col]
            door_r = _choice_in_span(rng, row_starts[row], heights[row])
            grid[door_r, wall_c] = door_color
        else:
            col = a[1]
            top_row = min(a[0], b[0])
            wall_r = row_starts[top_row] + heights[top_row]
            door_c = _choice_in_span(rng, col_starts[col], widths[col])
            grid[wall_r, door_c] = door_color

    rq = deque([source_room])
    while rq:
        room = rq.popleft()
        for nxt, door_color in graph[room]:
            if nxt not in room_color_truth:
                room_color_truth[nxt] = door_color
                rq.append(nxt)

    sr, sc = source_room
    grid[
        row_starts[sr] : row_starts[sr] + heights[sr],
        col_starts[sc] : col_starts[sc] + widths[sc],
    ] = source_color

    expected = grid.copy()
    for rr, cc in rooms:
        color = room_color_truth[(rr, cc)]
        expected[
            row_starts[rr] : row_starts[rr] + heights[rr],
            col_starts[cc] : col_starts[cc] + widths[cc],
        ] = color
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid, expected = _build_sample(rng)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid room-door sample")
