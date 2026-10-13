from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_WALL = 1
_SEED_COLORS = (2, 3, 4, 7)
_PALETTE_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
_PITCH = 6
_GEN_RETRIES = 400
_RNG = random.Random()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    labels = -np.ones((h, w), dtype=int)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            idx = len(comps)
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                labels[rr, cc] = idx
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps, labels


def _find_palette(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]]]] = []

    def add_run(cells: list[tuple[int, int]]) -> None:
        vals = [int(g[r, c]) for r, c in cells]
        if len(vals) < 3:
            return
        if any(v in (_BG, _WALL) for v in vals):
            return
        if len(set(vals)) != len(vals):
            return
        candidates.append((len(vals), cells))

    for row in (0, h - 1):
        c = 0
        while c < w:
            if int(g[row, c]) in (_BG, _WALL):
                c += 1
                continue
            c0 = c
            while c < w and int(g[row, c]) not in (_BG, _WALL):
                c += 1
            add_run([(row, cc) for cc in range(c0, c)])
    for col in (0, w - 1):
        r = 0
        while r < h:
            if int(g[r, col]) in (_BG, _WALL):
                r += 1
                continue
            r0 = r
            while r < h and int(g[r, col]) not in (_BG, _WALL):
                r += 1
            add_run([(rr, col) for rr in range(r0, r)])

    if not candidates:
        raise AssertionError("palette strip not found")

    cells = max(candidates, key=lambda item: item[0])[1]
    mask = np.zeros_like(g, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    palette = [int(g[r, c]) for r, c in cells]
    return palette, mask


def _detect_rooms(grid: np.ndarray, palette_mask: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    open_mask = (g != _WALL) & ~palette_mask
    roomish = np.zeros((h, w), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if bool(open_mask[r : r + 2, c : c + 2].all()):
                roomish[r : r + 2, c : c + 2] = True

    comps, _ = _components(roomish)
    rooms: list[dict] = []
    room_mask = np.zeros((h, w), dtype=bool)
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if hh < 2 or ww < 2 or int(coords.shape[0]) != hh * ww:
            continue
        if not bool(open_mask[r0 : r1 + 1, c0 : c1 + 1].all()):
            continue
        side_hits = 0
        if np.any(g[r0 - 1, c0:c1 + 1] == _WALL):
            side_hits += 1
        if np.any(g[r1 + 1, c0:c1 + 1] == _WALL):
            side_hits += 1
        if np.any(g[r0:r1 + 1, c0 - 1] == _WALL):
            side_hits += 1
        if np.any(g[r0:r1 + 1, c1 + 1] == _WALL):
            side_hits += 1
        if side_hits < 4:
            continue
        room_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        rooms.append({"coords": coords, "bbox": (r0, c0, r1, c1)})

    corridor_mask = open_mask & ~room_mask
    return rooms, room_mask, corridor_mask


def _build_room_graph(
    rooms: list[dict], corridor_labels: np.ndarray, palette_mask: np.ndarray
) -> tuple[list[set[int]], list[set[int]]]:
    h, w = corridor_labels.shape
    room_to_corr = [set() for _ in rooms]
    corr_count = int(corridor_labels.max()) + 1 if corridor_labels.size else 0
    corr_to_room = [set() for _ in range(max(0, corr_count))]

    for rid, room in enumerate(rooms):
        r0, c0, r1, c1 = room["bbox"]
        for c in range(c0, c1 + 1):
            for rr, cc in ((r0 - 1, c), (r1 + 1, c)):
                if 0 <= rr < h and 0 <= cc < w and not palette_mask[rr, cc]:
                    cid = int(corridor_labels[rr, cc])
                    if cid >= 0:
                        room_to_corr[rid].add(cid)
                        corr_to_room[cid].add(rid)
        for r in range(r0, r1 + 1):
            for rr, cc in ((r, c0 - 1), (r, c1 + 1)):
                if 0 <= rr < h and 0 <= cc < w and not palette_mask[rr, cc]:
                    cid = int(corridor_labels[rr, cc])
                    if cid >= 0:
                        room_to_corr[rid].add(cid)
                        corr_to_room[cid].add(rid)

    return room_to_corr, corr_to_room


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood door-linked rooms and corridors from seeded rooms, advancing by one palette step whenever the flow enters a new room.

    Input structure:
    - Background is black. Blue wall pixels form axis-aligned room shells and thin corridor passages.
    - A palette strip lies on one grid edge as a single row or column of distinct non-wall colors; its left-to-right or top-to-bottom order defines the color cycle.
    - Some enclosed rectangular room interiors contain one seed pixel in red, green, yellow, or orange; other rooms are empty black interiors.
    - Doorways are 1-cell openings in the room walls that connect rooms to corridor cells.

    Rule:
    - Fill each seeded room interior with its seed color.
    - Let that color flow out through every adjacent doorway and flood the connected corridor cells without changing color.
    - Whenever a flow enters another room through a doorway, switch to the next palette color before filling that room, then continue outward from that room's other doors with the switched color.
    - If several seeded flows could reach a place, the one requiring fewer room entries wins; unreachable corridor cells stay black.
    - Preserve the blue walls and the palette strip unchanged.

    Key insight:
    - The correct color of a cell depends on the number of door crossings from the nearest seed room in the room-corridor graph, not on raw geometric distance inside the corridor lines.
    """
    g = np.asarray(grid, dtype=int)
    palette, palette_mask = _find_palette(g)
    color_to_idx = {int(color): idx for idx, color in enumerate(palette)}

    rooms, room_mask, corridor_mask = _detect_rooms(g, palette_mask)
    corridors, corridor_labels = _components(corridor_mask)
    room_to_corr, corr_to_room = _build_room_graph(rooms, corridor_labels, palette_mask)

    inf = 10 ** 9
    best_room_dist = [inf] * len(rooms)
    best_room_color = [-1] * len(rooms)
    best_corr_dist = [inf] * len(corridors)
    best_corr_color = [-1] * len(corridors)
    dq: deque[tuple[str, int]] = deque()

    for rid, room in enumerate(rooms):
        vals = {
            int(g[r, c])
            for r, c in room["coords"]
            if int(g[r, c]) not in (_BG, _WALL)
        }
        seed_colors = [color for color in vals if color in color_to_idx]
        if len(seed_colors) != 1:
            continue
        color_idx = int(color_to_idx[seed_colors[0]])
        best_room_dist[rid] = 0
        best_room_color[rid] = color_idx
        dq.appendleft(("room", rid))

    while dq:
        kind, idx = dq.popleft()
        if kind == "room":
            dist = int(best_room_dist[idx])
            color_idx = int(best_room_color[idx])
            for cid in room_to_corr[idx]:
                if dist < best_corr_dist[cid]:
                    best_corr_dist[cid] = dist
                    best_corr_color[cid] = color_idx
                    dq.appendleft(("corr", cid))
        else:
            dist = int(best_corr_dist[idx])
            color_idx = int(best_corr_color[idx])
            next_idx = (color_idx + 1) % len(palette)
            for rid in corr_to_room[idx]:
                ndist = dist + 1
                if ndist < best_room_dist[rid]:
                    best_room_dist[rid] = ndist
                    best_room_color[rid] = next_idx
                    dq.append(("room", rid))

    out = g.copy()
    for cid, coords in enumerate(corridors):
        if best_corr_color[cid] < 0:
            continue
        out[coords[:, 0], coords[:, 1]] = int(palette[best_corr_color[cid]])
    for rid, room in enumerate(rooms):
        if best_room_color[rid] < 0:
            continue
        coords = room["coords"]
        out[coords[:, 0], coords[:, 1]] = int(palette[best_room_color[rid]])

    out[g == _WALL] = _WALL
    out[palette_mask] = g[palette_mask]
    return out


def _grid_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
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


def _grow_tree(rows: int, cols: int, target_rooms: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    start = (_RNG.randrange(rows), _RNG.randrange(cols))
    chosen = {start}
    order = [start]
    edges: list[tuple[int, int]] = []
    cell_to_idx = {start: 0}
    frontier = [start]

    while len(order) < target_rooms:
        src = _RNG.choice(frontier)
        candidates = [nb for nb in _grid_neighbors(src, rows, cols) if nb not in chosen]
        if not candidates:
            frontier = [cell for cell in frontier if any(nb not in chosen for nb in _grid_neighbors(cell, rows, cols))]
            if not frontier:
                break
            continue
        dst = _RNG.choice(candidates)
        chosen.add(dst)
        cell_to_idx[dst] = len(order)
        order.append(dst)
        edges.append((cell_to_idx[src], cell_to_idx[dst]))
        frontier.append(dst)
    return order, edges


def _room_layout(room_cell: tuple[int, int], box_r0: int, box_c0: int) -> tuple[int, int]:
    rr, cc = room_cell
    center_r = int(box_r0 + 3 + rr * _PITCH)
    center_c = int(box_c0 + 3 + cc * _PITCH)
    return center_r, center_c


def _room_bbox_from_center(center_r: int, center_c: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    outer_h = int(inner_h + 2)
    outer_w = int(inner_w + 2)
    r0 = int(center_r - (outer_h - 1) // 2)
    c0 = int(center_c - (outer_w - 1) // 2)
    r1 = int(r0 + outer_h - 1)
    c1 = int(c0 + outer_w - 1)
    return r0, c0, r1, c1


def _all_shortest_room_distances(adj: list[list[int]], src: int) -> list[int]:
    dist = [-1] * len(adj)
    dist[src] = 0
    dq = deque([src])
    while dq:
        node = dq.popleft()
        for nb in adj[node]:
            if dist[nb] >= 0:
                continue
            dist[nb] = dist[node] + 1
            dq.append(nb)
    return dist


def _paint_expected_output(
    grid: np.ndarray,
    room_infos: list[dict],
    adj: list[list[int]],
    edges: list[tuple[int, int]],
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]],
    seed_rooms: list[int],
    seed_colors: list[int],
    palette: list[int],
    palette_cells: list[tuple[int, int]],
) -> np.ndarray | None:
    if not seed_rooms:
        return None

    color_to_idx = {int(color): idx for idx, color in enumerate(palette)}
    dist_maps = [_all_shortest_room_distances(adj, src) for src in seed_rooms]

    room_owner: list[int] = []
    room_dist: list[int] = []
    room_fill: list[int] = []
    for rid in range(len(room_infos)):
        dvals = [int(dist_map[rid]) for dist_map in dist_maps]
        best = min(dvals)
        winners = [idx for idx, dval in enumerate(dvals) if dval == best]
        if len(winners) != 1:
            return None
        owner = winners[0]
        room_owner.append(owner)
        room_dist.append(best)
        room_fill.append(int(palette[(color_to_idx[seed_colors[owner]] + best) % len(palette)]))

    for u, v in edges:
        if room_dist[u] == room_dist[v]:
            return None

    out = np.array(grid, dtype=int, copy=True)
    for rid, info in enumerate(room_infos):
        r0, c0, r1, c1 = info["inner_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(room_fill[rid])
    for u, v in edges:
        key = (min(u, v), max(u, v))
        carrier = u if room_dist[u] < room_dist[v] else v
        fill = int(room_fill[carrier])
        for r, c in edge_cells[key]:
            out[r, c] = fill
    for r, c in palette_cells:
        out[r, c] = int(grid[r, c])
    return out


def _place_palette(grid: np.ndarray, palette: list[int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    side = _RNG.choice(("top", "bottom", "left", "right"))
    cells: list[tuple[int, int]] = []
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        start = _RNG.randint(1, w - len(palette) - 1)
        for i, color in enumerate(palette):
            cell = (row, start + i)
            grid[cell] = int(color)
            cells.append(cell)
    else:
        col = 0 if side == "left" else w - 1
        start = _RNG.randint(1, h - len(palette) - 1)
        for i, color in enumerate(palette):
            cell = (start + i, col)
            grid[cell] = int(color)
            cells.append(cell)
    return cells


def _gen_attempt() -> np.ndarray | None:
    rows = _RNG.randint(2, 4)
    cols = _RNG.randint(2, 4)
    room_cap = min(rows * cols, 8)
    room_count = _RNG.randint(4, room_cap)

    box_h = 6 * (rows - 1) + 7
    box_w = 6 * (cols - 1) + 7
    if box_h + 4 > 30 or box_w + 4 > 30:
        return None

    grid_h = box_h + 4
    grid_w = box_w + 4
    box_r0 = 2
    box_c0 = 2
    box_r1 = box_r0 + box_h - 1
    box_c1 = box_c0 + box_w - 1

    room_cells, edges = _grow_tree(rows, cols, room_count)
    if len(room_cells) != room_count:
        return None

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    grid[box_r0 : box_r1 + 1, box_c0 : box_c1 + 1] = _WALL

    room_infos: list[dict] = []
    for cell in room_cells:
        center_r, center_c = _room_layout(cell, box_r0, box_c0)
        inner_h = _RNG.choice((2, 3))
        inner_w = _RNG.choice((2, 3))
        r0, c0, r1, c1 = _room_bbox_from_center(center_r, center_c, inner_h, inner_w)
        if not (box_r0 < r0 < r1 < box_r1 and box_c0 < c0 < c1 < box_c1):
            return None
        grid[r0 + 1 : r1, c0 + 1 : c1] = _BG
        room_infos.append(
            {
                "cell": cell,
                "center": (center_r, center_c),
                "outer_bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )

    adj = [[] for _ in room_infos]
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        ur0, uc0, ur1, uc1 = room_infos[u]["outer_bbox"]
        vr0, vc0, vr1, vc1 = room_infos[v]["outer_bbox"]
        ucr, ucc = room_infos[u]["center"]
        vcr, vcc = room_infos[v]["center"]
        cells: list[tuple[int, int]] = []
        if ucr == vcr:
            door_r = int(ucr)
            if ucc < vcc:
                start = int(uc1)
                end = int(vc0)
            else:
                start = int(vc1)
                end = int(uc0)
            for c in range(start, end + 1):
                grid[door_r, c] = _BG
                cells.append((door_r, c))
        elif ucc == vcc:
            door_c = int(ucc)
            if ucr < vcr:
                start = int(ur1)
                end = int(vr0)
            else:
                start = int(vr1)
                end = int(ur0)
            for r in range(start, end + 1):
                grid[r, door_c] = _BG
                cells.append((r, door_c))
        else:
            return None
        edge_cells[(min(u, v), max(u, v))] = cells

    if max(len(nbs) for nbs in adj) < 2:
        return None

    leaves = [idx for idx, nbs in enumerate(adj) if len(nbs) == 1]
    seed_count = 1 if _RNG.random() < 0.5 else 2
    if len(leaves) < seed_count:
        return None
    seed_rooms = _RNG.sample(leaves, seed_count)
    if max(_all_shortest_room_distances(adj, seed_rooms[0])) < 2:
        return None

    seed_colors = _RNG.sample(list(_SEED_COLORS), seed_count)
    palette_len = _RNG.randint(max(3, seed_count + 1), 5)
    extras = [color for color in _PALETTE_POOL if color not in seed_colors]
    _RNG.shuffle(extras)
    palette = list(seed_colors) + extras[: palette_len - seed_count]
    _RNG.shuffle(palette)

    for rid, color in zip(seed_rooms, seed_colors):
        r0, c0, r1, c1 = room_infos[rid]["inner_bbox"]
        sr = _RNG.randint(r0, r1)
        sc = _RNG.randint(c0, c1)
        grid[sr, sc] = int(color)

    palette_cells = _place_palette(grid, palette)

    expected = _paint_expected_output(
        grid=grid,
        room_infos=room_infos,
        adj=adj,
        edges=edges,
        edge_cells=edge_cells,
        seed_rooms=seed_rooms,
        seed_colors=seed_colors,
        palette=palette,
        palette_cells=palette_cells,
    )
    if expected is None or np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid room-corridor instance")
