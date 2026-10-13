import random
from collections import Counter

import numpy as np


PATH_COLOR = 7
ROTATE_COLOR = 3
FLIP_COLOR = 2
PATH_COLORS = {PATH_COLOR, ROTATE_COLOR, FLIP_COLOR}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _trace_path(path_mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [p for p in _neighbors4(cur[0], cur[1], path_mask.shape) if path_mask[p]]
        if prev is not None:
            nxts = [p for p in nxts if p != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order


def _apply_rotate(items: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return [(c, -r, color) for r, c, color in items]


def _apply_flip(items: list[tuple[int, int, int]], horizontal_path: bool) -> list[tuple[int, int, int]]:
    if horizontal_path:
        return [(-r, c, color) for r, c, color in items]
    return [(r, -c, color) for r, c, color in items]


def _item_signature(items: list[tuple[int, int, int]]) -> frozenset[tuple[int, int, int]]:
    return frozenset((int(r), int(c), int(color)) for r, c, color in items)


def _path_station_ops(grid: np.ndarray, order: list[tuple[int, int]]) -> list[tuple[str, bool]]:
    ops: list[tuple[str, bool]] = []
    for idx in range(1, len(order) - 1):
        r, c = order[idx]
        color = int(grid[r, c])
        if color not in (ROTATE_COLOR, FLIP_COLOR):
            continue
        pr, pc = order[idx - 1]
        nr, nc = order[idx + 1]
        horizontal = pr == nr
        vertical = pc == nc
        assert horizontal ^ vertical
        ops.append(("rot", horizontal) if color == ROTATE_COLOR else ("flip", horizontal))
    return ops


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    path_mask = np.isin(g, list(PATH_COLORS))
    path_comps = _components(path_mask)
    assert len(path_comps) == 1
    path_comp = path_comps[0]
    path_set = set(path_comp)

    degrees = {}
    for cell in path_comp:
        degrees[cell] = sum((nbr in path_set) for nbr in _neighbors4(cell[0], cell[1], g.shape))
        assert degrees[cell] <= 2
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2

    shape_mask = (g != bg) & (~path_mask)
    shape_comps = _components(shape_mask)
    assert len(shape_comps) == 1
    shape_cells = shape_comps[0]
    shape_set = set(shape_cells)

    endpoint_to_adj = {}
    for endpoint in endpoints:
        adj = [nbr for nbr in _neighbors4(endpoint[0], endpoint[1], g.shape) if nbr in shape_set]
        endpoint_to_adj[endpoint] = adj
    start_candidates = [cell for cell, adj in endpoint_to_adj.items() if adj]
    assert len(start_candidates) == 1
    start = start_candidates[0]
    end = endpoints[0] if endpoints[1] == start else endpoints[1]
    assert len(endpoint_to_adj[start]) == 1
    anchor_in = endpoint_to_adj[start][0]
    side = (anchor_in[0] - start[0], anchor_in[1] - start[1])

    order = _trace_path(path_mask, start)
    assert len(order) == len(path_comp)
    assert order[-1] == end

    items = [(r - anchor_in[0], c - anchor_in[1], int(g[r, c])) for r, c in shape_cells]
    for op, horizontal in _path_station_ops(g, order):
        items = _apply_rotate(items) if op == "rot" else _apply_flip(items, horizontal)

    out = g.copy()
    for r, c in path_comp:
        out[r, c] = bg
    for r, c in shape_cells:
        out[r, c] = bg

    anchor_out = (end[0] - side[0], end[1] - side[1])
    for dr, dc, color in items:
        rr = anchor_out[0] + dr
        cc = anchor_out[1] + dc
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
        assert out[rr, cc] == bg
        out[rr, cc] = color

    return out


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _can_add_segment(
    occupied: set[tuple[int, int]],
    prev: tuple[int, int],
    cells: list[tuple[int, int]],
) -> bool:
    temp = set(occupied)
    cur_prev = prev
    for cell in cells:
        if cell in temp:
            return False
        for dr, dc in DIRS4:
            nbr = (cell[0] + dr, cell[1] + dc)
            if nbr in temp and nbr != cur_prev:
                return False
        temp.add(cell)
        cur_prev = cell
    return True


def _generate_path() -> list[tuple[int, int]] | None:
    num_segments = random.randint(4, 6)
    path = [(0, 0)]
    occupied = {(0, 0)}
    prev_direction = None

    for seg_idx in range(num_segments):
        if prev_direction is None:
            direction_options = DIRS4[:]
        else:
            direction_options = _perpendiculars(prev_direction)
        random.shuffle(direction_options)
        placed = False
        for direction in direction_options:
            lengths = list(range(2, 6))
            random.shuffle(lengths)
            for length in lengths:
                cells = _segment_cells(path[-1], direction, length)
                if not _can_add_segment(occupied, path[-1], cells):
                    continue
                path.extend(cells)
                occupied.update(cells)
                prev_direction = direction
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if abs(path[-1][0] - path[0][0]) + abs(path[-1][1] - path[0][1]) < 4:
        return None
    return path


def _straight_indices(path: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    horizontal = []
    vertical = []
    for idx in range(1, len(path) - 1):
        pr, pc = path[idx - 1]
        nr, nc = path[idx + 1]
        if pr == nr:
            horizontal.append(idx)
        elif pc == nc:
            vertical.append(idx)
    return horizontal, vertical


def _sample_station_indices(path: list[tuple[int, int]]) -> dict[int, int] | None:
    horizontal, vertical = _straight_indices(path)
    all_straight = sorted(horizontal + vertical)
    if len(horizontal) == 0 or len(vertical) == 0 or len(all_straight) < 2:
        return None

    for _ in range(80):
        station_count = random.randint(2, min(5, len(all_straight)))
        kinds = random.choices(["rot", "hflip", "vflip"], k=station_count)
        if len(set(kinds)) < 2:
            continue
        h_need = kinds.count("hflip")
        v_need = kinds.count("vflip")
        r_need = kinds.count("rot")
        if h_need > len(horizontal) or v_need > len(vertical):
            continue
        chosen: dict[int, int] = {}
        h_pick = random.sample(horizontal, h_need)
        v_pick = random.sample(vertical, v_need)
        for idx in h_pick:
            chosen[idx] = FLIP_COLOR
        for idx in v_pick:
            chosen[idx] = FLIP_COLOR
        remaining = [idx for idx in all_straight if idx not in chosen]
        if r_need > len(remaining):
            continue
        for idx in random.sample(remaining, r_need):
            chosen[idx] = ROTATE_COLOR
        assert len(chosen) == station_count
        return chosen
    return None


def _generate_local_shape() -> list[tuple[int, int]] | None:
    target = random.randint(4, 7)
    max_depth = random.randint(2, 4)
    max_side = random.randint(2, 3)
    cells = {(0, 0)}

    while len(cells) < target:
        frontier = set()
        for u, v in cells:
            for du, dv in DIRS4:
                nu, nv = u + du, v + dv
                if nu < 0 or nu > max_depth or abs(nv) > max_side:
                    continue
                if (nu, nv) not in cells:
                    frontier.add((nu, nv))
        if not frontier:
            return None
        cells.add(random.choice(list(frontier)))

    us = {u for u, _v in cells}
    vs = {v for _u, v in cells}
    if len(us) < 2 or len(vs) < 2:
        return None
    return sorted(cells)


def _local_to_global(
    local_cells: list[tuple[int, int]],
    away: tuple[int, int],
    lateral: tuple[int, int],
) -> list[tuple[int, int]]:
    rel = []
    for u, v in local_cells:
        dr = u * away[0] + v * lateral[0]
        dc = u * away[1] + v * lateral[1]
        rel.append((dr, dc))
    return rel


def _color_items(rel_coords: list[tuple[int, int]], bg: int) -> list[tuple[int, int, int]]:
    color_count = random.choices([1, 2, 3], weights=[1, 4, 3])[0]
    palette_pool = [color for color in [0, 1, 4, 5, 6, 8, 9] if color != bg]
    palette = random.sample(palette_pool, color_count)
    ordering_keys = [
        lambda item: (item[0], item[1]),
        lambda item: (item[1], item[0]),
        lambda item: (item[0] + item[1], item[0], item[1]),
        lambda item: (item[0] - item[1], item[0], item[1]),
    ]
    coords = sorted(rel_coords, key=random.choice(ordering_keys))
    items = []
    for rank, (dr, dc) in enumerate(coords):
        color_idx = min(color_count - 1, rank * color_count // len(coords))
        items.append((dr, dc, palette[color_idx]))
    return items


def _station_ops_for_generator(
    path: list[tuple[int, int]],
    station_map: dict[int, int],
) -> list[tuple[str, bool]]:
    ops = []
    for idx in sorted(station_map):
        pr, pc = path[idx - 1]
        nr, nc = path[idx + 1]
        horizontal = pr == nr
        if station_map[idx] == ROTATE_COLOR:
            ops.append(("rot", horizontal))
        else:
            ops.append(("flip", horizontal))
    return ops


def _visible_sequence(
    items: list[tuple[int, int, int]],
    ops: list[tuple[str, bool]],
) -> list[tuple[int, int, int]] | None:
    current = items
    for op, horizontal in ops:
        nxt = _apply_rotate(current) if op == "rot" else _apply_flip(current, horizontal)
        if _item_signature(nxt) == _item_signature(current):
            return None
        current = nxt
    if _item_signature(current) == _item_signature(items):
        return None
    return current


def _adjacent_to_path_only_at_start(
    path_set: set[tuple[int, int]],
    start: tuple[int, int],
    shape_abs: set[tuple[int, int]],
) -> bool:
    for cell in shape_abs:
        for dr, dc in DIRS4:
            nbr = (cell[0] + dr, cell[1] + dc)
            if nbr in path_set and nbr != start:
                return False
    return True


def _build_grid(
    bg: int,
    path: list[tuple[int, int]],
    station_map: dict[int, int],
    source_abs: list[tuple[int, int, int]],
    final_abs: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    coords = [(r, c) for r, c in path]
    coords.extend((r, c) for r, c, _color in source_abs)
    coords.extend((r, c) for r, c, _color in final_abs)
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)

    h = max_r - min_r + 1 + top + bottom
    w = max_c - min_c + 1 + left + right
    row_shift = top - min_r
    col_shift = left - min_c

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    for idx, (r, c) in enumerate(path):
        color = station_map.get(idx, PATH_COLOR)
        inp[r + row_shift, c + col_shift] = color
    for r, c, color in source_abs:
        inp[r + row_shift, c + col_shift] = color
    for r, c, color in final_abs:
        out[r + row_shift, c + col_shift] = color

    return inp, out


def _make_sample() -> tuple[np.ndarray, np.ndarray] | None:
    path = _generate_path()
    if path is None:
        return None
    station_map = _sample_station_indices(path)
    if station_map is None:
        return None

    start = path[0]
    end = path[-1]
    start_dir = (path[1][0] - start[0], path[1][1] - start[1])
    end_in = (end[0] - path[-2][0], end[1] - path[-2][1])
    path_set = set(path)
    ops = _station_ops_for_generator(path, station_map)
    bg = random.choice([0, 1, 4, 5, 6, 8, 9])

    side_options = [d for d in DIRS4 if d != start_dir and d != end_in]
    random.shuffle(side_options)

    for _ in range(80):
        local_shape = _generate_local_shape()
        if local_shape is None:
            continue
        for side in side_options:
            lateral_options = _perpendiculars(side)
            random.shuffle(lateral_options)
            for lateral in lateral_options:
                rel_coords = _local_to_global(local_shape, side, lateral)
                items = _color_items(rel_coords, bg)
                transformed = _visible_sequence(items, ops)
                if transformed is None:
                    continue

                anchor_in = (start[0] + side[0], start[1] + side[1])
                source_abs = [(anchor_in[0] + dr, anchor_in[1] + dc, color) for dr, dc, color in items]
                source_cells = {(r, c) for r, c, _color in source_abs}
                if source_cells & path_set:
                    continue
                if len(source_cells) != len(source_abs):
                    continue
                if not _adjacent_to_path_only_at_start(path_set, start, source_cells):
                    continue

                anchor_out = (end[0] - side[0], end[1] - side[1])
                final_abs = [(anchor_out[0] + dr, anchor_out[1] + dc, color) for dr, dc, color in transformed]
                final_cells = {(r, c) for r, c, _color in final_abs}
                if len(final_cells) != len(final_abs):
                    continue
                if final_cells & path_set:
                    continue
                if final_cells & source_cells:
                    continue

                inp, expected = _build_grid(bg, path, station_map, source_abs, final_abs)
                try:
                    actual = main(inp)
                except Exception:
                    continue
                if not np.array_equal(actual, expected):
                    continue
                return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        sample = _make_sample()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid sample")
