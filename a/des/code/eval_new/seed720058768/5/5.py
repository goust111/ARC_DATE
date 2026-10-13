from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BG = 0
GRAY = 5
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
OBJECT_COLORS = (6, 7, 8, 9)
TRANSFORM_COLORS = (RED, GREEN, BLUE, YELLOW)
MAX_RETRIES = 500
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _apply_transform(mask: np.ndarray, color: int) -> np.ndarray:
    if color == RED:
        return np.fliplr(mask)
    if color == GREEN:
        return np.flipud(mask)
    if color == BLUE:
        return np.rot90(mask, -1)
    if color == YELLOW:
        return mask.T.copy()
    return mask.copy()


def _order_path_cells(
    cells: list[tuple[int, int]],
    sender_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    if len(cells) == 1:
        return list(cells)

    cell_set = set(cells)
    degrees = {
        cell: sum((nr, nc) in cell_set for nr, nc in _neighbors4(cell[0], cell[1], h, w))
        for cell in cells
    }

    def touches_sender(cell: tuple[int, int]) -> bool:
        return any((nr, nc) in sender_cells for nr, nc in _neighbors4(cell[0], cell[1], h, w))

    endpoints = [cell for cell in cells if degrees[cell] <= 1]
    starts = [cell for cell in endpoints if touches_sender(cell)]
    if not starts:
        starts = [cell for cell in cells if touches_sender(cell)]
    start = min(starts, key=lambda x: (degrees.get(x, 0), x[0], x[1]))

    ordered = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start
    while len(ordered) < len(cells):
        next_cells = [
            (nr, nc)
            for nr, nc in _neighbors4(cur[0], cur[1], h, w)
            if (nr, nc) in cell_set and (nr, nc) != prev and (nr, nc) not in visited
        ]
        if not next_cells:
            break
        nxt = min(next_cells)
        ordered.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    if len(ordered) == len(cells):
        return ordered

    remaining = [cell for cell in cells if cell not in visited]
    return ordered + sorted(remaining)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each same-color object pair linked by a gray path, read transform stations along the path, and redraw the transformed sender shape at the receiver anchor.

    Input structure:
    - The background is black.
    - Colored solid objects are monochrome 4-connected regions with area at least 2.
    - Some colors appear exactly twice; each such pair is connected by a one-pixel-wide gray path.
    - Path cells may contain 0-3 single-pixel non-black non-gray stations: red, green, blue, or yellow.

    Rule:
    - For every color that has exactly two objects, use the smaller object as sender and the larger as receiver; ties use the earlier bounding-box top-left as sender.
    - Crop the sender to its tight bounding box and traverse the connecting path from sender to receiver.
    - Apply station transforms in path order: red = horizontal flip, green = vertical flip, blue = 90-degree clockwise rotation, yellow = transpose.
    - Erase the sender, receiver, path, and stations, then draw the final transformed sender shape in the shared object color with the receiver bounding-box top-left as anchor.

    Key insight:
    - The gray connector is both a pairing cue and an ordered instruction tape, so solving requires combining object pairing, sender/receiver disambiguation, and path-ordered dihedral transforms.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    objects = []
    cell_to_object: dict[tuple[int, int], int] = {}
    station_cells: set[tuple[int, int]] = set()

    for color in sorted(int(x) for x in np.unique(g)):
        if color in (BG, GRAY):
            continue
        for coords in _components(g == color):
            if len(coords) == 1:
                station_cells.add(coords[0])
                continue
            mask, bbox = _tight_mask(coords)
            obj = {
                "id": len(objects),
                "color": color,
                "coords": coords,
                "coord_set": set(coords),
                "area": len(coords),
                "bbox": bbox,
                "mask": mask,
            }
            for cell in coords:
                cell_to_object[cell] = obj["id"]
            objects.append(obj)

    color_to_ids: dict[int, list[int]] = defaultdict(list)
    for obj in objects:
        color_to_ids[int(obj["color"])].append(int(obj["id"]))

    path_mask = g == GRAY
    for r, c in station_cells:
        path_mask[r, c] = True

    pair_to_path: dict[frozenset[int], list[tuple[int, int]]] = {}
    for cells in _components(path_mask):
        adjacent = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) in cell_to_object:
                    adjacent.add(cell_to_object[(nr, nc)])
        if len(adjacent) != 2:
            continue
        a, b = sorted(adjacent)
        if int(objects[a]["color"]) != int(objects[b]["color"]):
            continue
        key = frozenset((a, b))
        if key not in pair_to_path or len(cells) > len(pair_to_path[key]):
            pair_to_path[key] = cells

    for color, ids in color_to_ids.items():
        if len(ids) != 2:
            continue
        key = frozenset(ids)
        if key not in pair_to_path:
            continue

        first, second = (objects[ids[0]], objects[ids[1]])
        sender, receiver = sorted(
            (first, second),
            key=lambda obj: (int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])),
        )
        path_cells = pair_to_path[key]
        ordered_path = _order_path_cells(path_cells, sender["coord_set"], (h, w))

        pattern = np.array(sender["mask"], dtype=bool)
        for r, c in ordered_path:
            cell_color = int(g[r, c])
            if cell_color not in (BG, GRAY):
                pattern = _apply_transform(pattern, cell_color)

        for r, c in sender["coords"]:
            out[r, c] = BG
        for r, c in receiver["coords"]:
            out[r, c] = BG
        for r, c in path_cells:
            out[r, c] = BG

        r0, c0, _r1, _c1 = receiver["bbox"]
        ph, pw = pattern.shape
        assert r0 + ph <= h and c0 + pw <= w
        for dr in range(ph):
            for dc in range(pw):
                if bool(pattern[dr, dc]):
                    out[r0 + dr, c0 + dc] = color

    return out


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask(area: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        h = int(_RNG.integers(2, max_h + 1))
        w = int(_RNG.integers(2, max_w + 1))
        if h * w < area:
            continue
        start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], h, w))
        while len(cells) < area and frontier:
            choices = sorted(frontier)
            idx = int(_RNG.integers(len(choices)))
            cell = choices[idx]
            frontier.remove(cell)
            cells.add(cell)
            for nxt in _neighbors4(cell[0], cell[1], h, w):
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return _crop_bool(mask)
    raise RuntimeError("failed to sample connected mask")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                grid[top + r, left + c] = color


def _line_horizontal(row: int, c0: int, c1: int) -> list[tuple[int, int]]:
    step = 1 if c1 >= c0 else -1
    return [(row, c) for c in range(c0, c1 + step, step)]


def _line_vertical(col: int, r0: int, r1: int) -> list[tuple[int, int]]:
    step = 1 if r1 >= r0 else -1
    return [(r, col) for r in range(r0, r1 + step, step)]


def _build_pair_cluster(color: int, station_count: int, equal_area: bool, ops_seed: list[int]) -> np.ndarray:
    sender_area = int(_RNG.integers(2, 7))
    sender_mask = _random_connected_mask(sender_area, 4, 4)

    if equal_area:
        receiver_area = sender_area
    else:
        receiver_area = int(_RNG.integers(sender_area + 1, min(10, sender_area + 5) + 1))

    for _ in range(150):
        receiver_mask = _random_connected_mask(receiver_area, 5, 5)
        if not equal_area or receiver_mask.shape != sender_mask.shape or not np.array_equal(receiver_mask, sender_mask):
            break
    else:
        raise RuntimeError("failed to sample receiver mask")

    ops: list[int] = []
    pattern = np.array(sender_mask, copy=True)
    for _ in range(station_count):
        if not ops_seed:
            ops_seed.extend(list(_RNG.permutation(TRANSFORM_COLORS)))
        op = int(ops_seed.pop())
        ops.append(op)
        pattern = _apply_transform(pattern, op)

    sender_h, sender_w = sender_mask.shape
    receiver_h, receiver_w = receiver_mask.shape
    reserve_h = max(receiver_h, pattern.shape[0])
    reserve_w = max(receiver_w, pattern.shape[1])

    gap = int(_RNG.integers(max(4, station_count + 2), 8))
    cluster_h = max(sender_h, reserve_h) + 4
    sender_top = int(_RNG.integers(1, cluster_h - sender_h - 1))
    if equal_area:
        receiver_top = sender_top
    else:
        receiver_top = int(_RNG.integers(1, cluster_h - reserve_h - 1))
    sender_left = 1
    receiver_left = sender_left + sender_w + gap + 1
    cluster_w = receiver_left + reserve_w + 2
    cluster = np.full((cluster_h, cluster_w), BG, dtype=int)

    _paint_mask(cluster, sender_mask, sender_top, sender_left, color)
    _paint_mask(cluster, receiver_mask, receiver_top, receiver_left, color)

    sender_ports = [(r, c) for r in range(sender_h) for c in range(sender_w) if sender_mask[r, c] and c == sender_w - 1]
    receiver_ports = [(r, c) for r in range(receiver_h) for c in range(receiver_w) if receiver_mask[r, c] and c == 0]
    sp_r, _sp_c = sender_ports[int(_RNG.integers(len(sender_ports)))]
    rp_r, _rp_c = receiver_ports[int(_RNG.integers(len(receiver_ports)))]

    start = (sender_top + sp_r, sender_left + sender_w)
    end = (receiver_top + rp_r, receiver_left - 1)
    if start[1] + 1 < end[1]:
        mid_col = int(_RNG.integers(start[1] + 1, end[1]))
    else:
        mid_col = start[1]

    path = _line_horizontal(start[0], start[1], mid_col)
    vertical = _line_vertical(mid_col, start[0], end[0])
    if len(vertical) > 1:
        path.extend(vertical[1:])
    tail = _line_horizontal(end[0], mid_col, end[1])
    if len(tail) > 1:
        path.extend(tail[1:])

    if len(path) < station_count + 2:
        raise RuntimeError("path too short for requested stations")

    station_positions = []
    if station_count:
        interior = np.arange(1, len(path) - 1)
        chosen = _RNG.choice(interior, size=station_count, replace=False)
        station_positions = sorted(int(x) for x in chosen)

    for idx, (r, c) in enumerate(path):
        cluster[r, c] = GRAY
        if idx in station_positions:
            cluster[r, c] = ops[station_positions.index(idx)]

    return cluster


def _build_unpaired_cluster(color: int) -> np.ndarray:
    area = int(_RNG.integers(2, 8))
    mask = _random_connected_mask(area, 5, 5)
    k = int(_RNG.integers(4))
    if k:
        mask = np.rot90(mask, k)
    grid = np.full(mask.shape, BG, dtype=int)
    _paint_mask(grid, mask, 0, 0, color)
    return grid


def _augment_cluster(cluster: np.ndarray, allow_reorient: bool) -> np.ndarray:
    out = np.array(cluster, copy=True)
    if allow_reorient:
        k = int(_RNG.integers(4))
        if k:
            out = np.rot90(out, k)
        if bool(_RNG.integers(2)):
            out = np.fliplr(out)
    return out


def _pack_clusters(clusters: list[np.ndarray]) -> np.ndarray | None:
    order = list(_RNG.permutation(len(clusters)))
    max_w = max(int(clusters[i].shape[1]) for i in order)
    target_w = int(_RNG.integers(max(max_w + 2, 12), 31))

    placements: dict[int, tuple[int, int]] = {}
    x = 1
    y = 1
    row_h = 0
    used_w = 0
    for idx in order:
        h, w = map(int, clusters[idx].shape)
        if x > 1 and x + w + 1 > target_w:
            y += row_h + int(_RNG.integers(2, 5))
            x = 1
            row_h = 0
        placements[idx] = (y, x)
        x += w + int(_RNG.integers(2, 5))
        row_h = max(row_h, h)
        used_w = max(used_w, x)

    used_h = y + row_h + 1
    if used_h > 30 or used_w > 30:
        return None

    final_h = int(_RNG.integers(max(used_h, 10), 31))
    final_w = int(_RNG.integers(max(used_w, 10), 31))
    y_shift = int(_RNG.integers(0, final_h - used_h + 1))
    x_shift = int(_RNG.integers(0, final_w - used_w + 1))

    grid = np.full((final_h, final_w), BG, dtype=int)
    for idx, cluster in enumerate(clusters):
        top, left = placements[idx]
        top += y_shift
        left += x_shift
        h, w = cluster.shape
        patch = grid[top : top + h, left : left + w]
        patch[cluster != BG] = cluster[cluster != BG]
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.integers(1, 4))
    chosen_colors = list(_RNG.permutation(OBJECT_COLORS))
    pair_colors = chosen_colors[:pair_count]
    remaining = chosen_colors[pair_count:]
    unmatched_count = 0 if not remaining else int(_RNG.integers(0, min(2, len(remaining)) + 1))
    unmatched_colors = remaining[:unmatched_count]

    op_seed = list(_RNG.permutation(TRANSFORM_COLORS))
    pair_clusters = []
    for color in pair_colors:
        station_count = int(_RNG.integers(0, 4))
        equal_area = bool(_RNG.integers(5) == 0)
        cluster = _build_pair_cluster(int(color), station_count, equal_area, op_seed)
        pair_clusters.append(_augment_cluster(cluster, allow_reorient=not equal_area))

    clusters = pair_clusters[:]
    for color in unmatched_colors:
        clusters.append(_augment_cluster(_build_unpaired_cluster(int(color)), allow_reorient=True))
    if not clusters:
        return None
    _RNG.shuffle(clusters)

    grid = _pack_clusters(clusters)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
