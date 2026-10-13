from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

SIDES = ("up", "right", "down", "left")
DELTA_TO_SIDE = {
    (-1, 0): "up",
    (0, 1): "right",
    (1, 0): "down",
    (0, -1): "left",
}

LAYOUTS = (
    ((0, 0), (0, 1), (1, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 0), (1, 0)),
)

GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 80
DFS_ATTEMPTS = 18

Cell = tuple[int, int]


@dataclass(frozen=True)
class Room:
    bbox: tuple[int, int, int, int]
    path_color: int
    path_cells: tuple[Cell, ...]
    ports: tuple[tuple[int, Cell], ...]


def _neighbors4(r: int, c: int) -> tuple[Cell, Cell, Cell, Cell]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[Cell]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[Cell]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(int(r), int(c))])
            comp: list[Cell] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(cells: list[Cell]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _ceil_to_multiple(value: int, base: int) -> int:
    return ((value + base - 1) // base) * base


def _find_divider_col(g: np.ndarray) -> int:
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == MAROON)]
    assert cols, "expected a maroon divider column"
    assert all(cols[i] + 1 == cols[i + 1] for i in range(len(cols) - 1))
    return cols[len(cols) // 2]


def _extract_rooms(left: np.ndarray) -> list[Room]:
    rooms: list[Room] = []
    for comp in _components(left != BLACK):
        r0, c0, r1, c1 = _bbox(comp)
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 3 and w >= 3, "room must have a border and an interior"

        inner = patch[1:-1, 1:-1]
        path_colors = sorted(set(map(int, inner.flatten())) - {BLACK})
        assert len(path_colors) == 1, "each room must contain exactly one path color"
        path_color = path_colors[0]

        path_cells = {(int(r), int(c)) for r, c in np.argwhere(patch == path_color)}
        ports: list[tuple[int, Cell]] = []
        for rr in range(h):
            for cc in range(w):
                on_border = rr in (0, h - 1) or cc in (0, w - 1)
                if not on_border:
                    continue
                val = int(patch[rr, cc])
                if val == GREY:
                    continue
                assert val not in (BLACK, MAROON, path_color), "border cells may only be grey or ports"
                adj = [
                    (nr, nc)
                    for nr, nc in _neighbors4(rr, cc)
                    if 0 <= nr < h and 0 <= nc < w and int(patch[nr, nc]) == path_color
                ]
                assert len(adj) == 1, "each port must touch one path endpoint"
                ports.append((val, (rr, cc)))
                path_cells.add((rr, cc))

        assert len(ports) == 2, "each room must expose exactly two ports"
        rooms.append(
            Room(
                bbox=(r0, c0, r1, c1),
                path_color=path_color,
                path_cells=tuple(sorted(path_cells)),
                ports=tuple(sorted(ports, key=lambda item: item[1])),
            )
        )

    rooms.sort(key=lambda room: room.bbox[:2])
    assert rooms, "expected at least one room"
    return rooms


def _assemble_components(rooms: list[Room]) -> list[list[Cell]]:
    by_color: dict[int, list[tuple[int, Cell]]] = {}
    for idx, room in enumerate(rooms):
        for color, local in room.ports:
            by_color.setdefault(color, []).append((idx, local))

    adjacency: dict[int, list[tuple[int, Cell, Cell]]] = {idx: [] for idx in range(len(rooms))}
    for color, items in by_color.items():
        assert len(items) == 2, f"port color {color} must appear exactly twice"
        (a_idx, a_local), (b_idx, b_local) = items
        adjacency[a_idx].append((b_idx, a_local, b_local))
        adjacency[b_idx].append((a_idx, b_local, a_local))

    translations: dict[int, Cell] = {}
    components: list[tuple[tuple[int, int], list[Cell]]] = []
    for root in range(len(rooms)):
        if root in translations:
            continue

        translations[root] = (0, 0)
        queue = deque([root])
        comp_indices: list[int] = []
        while queue:
            cur = queue.popleft()
            comp_indices.append(cur)
            base_r, base_c = translations[cur]
            for nxt, own_local, other_local in adjacency[cur]:
                nxt_shift = (
                    base_r + own_local[0] - other_local[0],
                    base_c + own_local[1] - other_local[1],
                )
                if nxt in translations:
                    assert translations[nxt] == nxt_shift, "port alignments must be consistent"
                    continue
                translations[nxt] = nxt_shift
                queue.append(nxt)

        cells: set[Cell] = set()
        for idx in comp_indices:
            dr, dc = translations[idx]
            for rr, cc in rooms[idx].path_cells:
                cells.add((dr + rr, dc + cc))
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        normalized = sorted((r - min_r, c - min_c) for r, c in cells)
        order_key = min((rooms[idx].bbox[0], rooms[idx].bbox[1]) for idx in comp_indices)
        components.append((order_key, normalized))

    components.sort(key=lambda item: item[0])
    return [cells for _, cells in components]


def _draw_components(shape: tuple[int, int], components: list[list[Cell]], color: int) -> np.ndarray:
    panel = np.full(shape, BLACK, dtype=int)
    h, w = shape
    cur_r = 0
    cur_c = 0
    shelf_h = 0

    for cells in components:
        comp_h = max(r for r, _ in cells) + 1
        comp_w = max(c for _, c in cells) + 1
        if cur_c > 0 and cur_c + comp_w > w:
            cur_r += shelf_h + 1
            cur_c = 0
            shelf_h = 0
        assert cur_r + comp_h <= h and cur_c + comp_w <= w, "assembled path must fit the right panel"
        for rr, cc in cells:
            panel[cur_r + rr, cur_c + cc] = color
        cur_c += comp_w + 1
        shelf_h = max(shelf_h, comp_h)
    return panel


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble room-local path segments by matching equal-colored ports, then draw the assembled path on the right panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider = _find_divider_col(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    assert left.size > 0 and right.size > 0

    rooms = _extract_rooms(left)
    path_colors = {room.path_color for room in rooms}
    assert len(path_colors) == 1, "all room segments must share one path color"
    path_color = next(iter(path_colors))

    components = _assemble_components(rooms)
    assembled = _draw_components(right.shape, components, path_color)

    out = g.copy()
    out[:, divider + 1 :] = assembled
    return out


def _inner_endpoint(n: int, side: str) -> Cell:
    mid = n // 2
    if side == "left":
        return (mid, 0)
    if side == "right":
        return (mid, n - 1)
    if side == "up":
        return (0, mid)
    if side == "down":
        return (n - 1, mid)
    raise ValueError(side)


def _port_cell(n: int, side: str) -> Cell:
    mid = n // 2 + 1
    if side == "left":
        return (mid, 0)
    if side == "right":
        return (mid, n + 1)
    if side == "up":
        return (0, mid)
    if side == "down":
        return (n + 1, mid)
    raise ValueError(side)


def _reachable_info(n: int, current: Cell, end: Cell, visited: set[Cell]) -> tuple[bool, int, int]:
    queue = deque([current])
    seen = {current}
    dist = {current: 0}
    while queue:
        rr, cc = queue.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            if nxt in visited and nxt != end:
                continue
            seen.add(nxt)
            dist[nxt] = dist[(rr, cc)] + 1
            queue.append(nxt)
    return end in seen, dist.get(end, 10**9), len(seen)


def _fallback_path(start: Cell, end: Cell) -> list[Cell]:
    r0, c0 = start
    r1, c1 = end
    cells: list[Cell] = [(r0, c0)]
    if random.random() < 0.5:
        c_step = 1 if c1 >= c0 else -1
        for c in range(c0 + c_step, c1 + c_step, c_step):
            cells.append((r0, c))
        if r1 != r0:
            r_step = 1 if r1 >= r0 else -1
            for r in range(r0 + r_step, r1 + r_step, r_step):
                cells.append((r, c1))
    else:
        r_step = 1 if r1 >= r0 else -1
        for r in range(r0 + r_step, r1 + r_step, r_step):
            cells.append((r, c0))
        if c1 != c0:
            c_step = 1 if c1 >= c0 else -1
            for c in range(c0 + c_step, c1 + c_step, c_step):
                cells.append((r1, c))
    return cells


def _random_interior_path(n: int, start: Cell, end: Cell) -> list[Cell]:
    dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
    min_len = dist + 1
    max_len = min(n * n, min_len + n + 2)
    candidates = [length for length in range(min_len, max_len + 1) if (length - min_len) % 2 == 0]
    random.shuffle(candidates)
    if random.random() < 0.6:
        candidates.sort(reverse=True)

    for target_len in candidates[: min(len(candidates), 5)]:
        for _ in range(DFS_ATTEMPTS):
            path = [start]
            visited = {start}

            def dfs(cur: Cell) -> bool:
                if len(path) == target_len:
                    return cur == end

                remaining = target_len - len(path)
                ok, d_to_end, reach_count = _reachable_info(n, cur, end, visited)
                if not ok or d_to_end > remaining or (remaining - d_to_end) % 2 != 0:
                    return False
                if reach_count < remaining + 1:
                    return False

                moves: list[tuple[int, float, Cell]] = []
                for nr, nc in _neighbors4(cur[0], cur[1]):
                    nxt = (nr, nc)
                    if not (0 <= nr < n and 0 <= nc < n) or nxt in visited:
                        continue
                    d = abs(nr - end[0]) + abs(nc - end[1])
                    slack = remaining - 1 - d
                    if d > remaining - 1 or slack % 2 != 0:
                        continue
                    moves.append((d, random.random(), nxt))
                moves.sort(key=lambda item: (item[0], item[1]))
                if random.random() < 0.4:
                    random.shuffle(moves)

                for _, _, nxt in moves:
                    visited.add(nxt)
                    path.append(nxt)
                    if dfs(nxt):
                        return True
                    path.pop()
                    visited.remove(nxt)
                return False

            if dfs(start):
                return path

    return _fallback_path(start, end)


def _make_room_patch(
    n: int,
    path_color: int,
    side_a: str,
    side_b: str,
    port_color_a: int,
    port_color_b: int,
) -> tuple[np.ndarray, tuple[Cell, ...]]:
    patch = np.full((n + 2, n + 2), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    start_inner = _inner_endpoint(n, side_a)
    end_inner = _inner_endpoint(n, side_b)
    inner_path = _random_interior_path(n, start_inner, end_inner)

    path_cells: set[Cell] = set()
    for rr, cc in inner_path:
        full = (rr + 1, cc + 1)
        path_cells.add(full)
        patch[full] = path_color

    port_a = _port_cell(n, side_a)
    port_b = _port_cell(n, side_b)
    patch[port_a] = port_color_a
    patch[port_b] = port_color_b
    path_cells.add(port_a)
    path_cells.add(port_b)
    return patch, tuple(sorted(path_cells))


def _side_for_neighbor(tile: Cell, other: Cell) -> str:
    return DELTA_TO_SIDE[(other[0] - tile[0], other[1] - tile[1])]


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    rs = max(0, r0 - gap)
    re = min(used.shape[0], r0 + h + gap)
    cs = max(0, c0 - gap)
    ce = min(used.shape[1], c0 + w + gap)
    return not np.any(used[rs:re, cs:ce])


def _place_patches(patches: list[np.ndarray], h: int, w: int) -> list[tuple[np.ndarray, Cell]] | None:
    order = list(range(len(patches)))
    random.shuffle(order)
    used = np.zeros((h, w), dtype=bool)
    placements: list[tuple[np.ndarray, Cell]] = []
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        coords = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(coords)
        for r0, c0 in coords:
            if _can_place(used, r0, c0, ph, pw, gap=1):
                used[r0 : r0 + ph, c0 : c0 + pw] = patch != BLACK
                placements.append((patch, (r0, c0)))
                break
        else:
            return None
    return placements


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(4, 6)
    layout = random.choice(LAYOUTS)
    room_count = len(layout)
    step = n + 1
    path_color = random.choice(PALETTE)
    port_colors = random.sample([c for c in PALETTE if c != path_color], room_count)

    room_patches: list[np.ndarray] = []
    union_cells: set[Cell] = set()
    for idx, tile in enumerate(layout):
        prev_tile = layout[idx - 1]
        next_tile = layout[(idx + 1) % room_count]
        side_prev = _side_for_neighbor(tile, prev_tile)
        side_next = _side_for_neighbor(tile, next_tile)
        port_prev = port_colors[idx - 1]
        port_next = port_colors[idx]
        patch, path_cells = _make_room_patch(n, path_color, side_prev, side_next, port_prev, port_next)
        room_patches.append(patch)

        base_r = tile[0] * step
        base_c = tile[1] * step
        for rr, cc in path_cells:
            union_cells.add((base_r + rr, base_c + cc))

    min_r = min(r for r, _ in union_cells)
    min_c = min(c for _, c in union_cells)
    union_cells = {(r - min_r, c - min_c) for r, c in union_cells}

    out_h = max(r for r, _ in union_cells) + 1
    out_w = max(c for _, c in union_cells) + 1

    left_needed_h = (2 if room_count == 4 else 3) * (n + 2) + 6
    base_h = max(out_h, left_needed_h) + random.randint(0, n)
    grid_h = _ceil_to_multiple(base_h, n)
    right_w = _ceil_to_multiple(out_w + random.choice((0, n)), n)

    base_left_w = (2 if room_count == 4 else 3) * (n + 2) + 6
    left_w = base_left_w + random.randint(0, n + 2)

    placements = _place_patches(room_patches, grid_h, left_w)
    if placements is None:
        return None

    grid = np.full((grid_h, left_w + 1 + right_w), BLACK, dtype=int)
    divider = left_w
    grid[:, divider] = MAROON
    for patch, (r0, c0) in placements:
        ph, pw = patch.shape
        mask = patch != BLACK
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]

    expected = grid.copy()
    for rr, cc in union_cells:
        expected[rr, divider + 1 + cc] = path_color

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is None:
            continue
        inp, expected = case
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
