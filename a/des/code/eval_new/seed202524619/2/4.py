import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


OPS_2D: Tuple[str, ...] = ("flip_v", "flip_h", "rot90", "rot180", "rot270")


def _normalize_points(points: Iterable[Tuple[int, int]]) -> Tuple[Set[Tuple[int, int]], Tuple[int, int]]:
    rs, cs = zip(*points)
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    return {(r - min_r, c - min_c) for r, c in points}, (max_r - min_r + 1, max_c - min_c + 1)


def _transform_points(
    points: Iterable[Tuple[int, int]],
    h: int,
    w: int,
    op: str,
) -> Tuple[Set[Tuple[int, int]], Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for r, c in points:
        if op == "flip_v":
            tr, tc = r, w - 1 - c
            nh, nw = h, w
        elif op == "flip_h":
            tr, tc = h - 1 - r, c
            nh, nw = h, w
        elif op == "rot90":
            tr, tc = c, h - 1 - r
            nh, nw = w, h
        elif op == "rot180":
            tr, tc = h - 1 - r, w - 1 - c
            nh, nw = h, w
        elif op == "rot270":
            tr, tc = w - 1 - c, r
            nh, nw = w, h
        else:
            raise ValueError(f"unsupported op: {op}")
        out.add((tr, tc))
    return _normalize_points(out)[0], (nh, nw)


def _infer_transform_from_pair(
    src: Iterable[Tuple[int, int]],
    dst: Iterable[Tuple[int, int]],
) -> Optional[str]:
    src_n, src_shape = _normalize_points(tuple(src))
    dst_n, _ = _normalize_points(tuple(dst))
    for op in OPS_2D:
        trans_n, _ = _transform_points(src_n, src_shape[0], src_shape[1], op)
        if trans_n == dst_n:
            return op
    return None


def _iter_components(
    room: np.ndarray,
    banned: Set[int],
) -> List[Tuple[int, Set[Tuple[int, int]]]]:
    h, w = room.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Tuple[int, Set[Tuple[int, int]]]] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(h):
        for c in range(w):
            color = int(room[r, c])
            if color in banned or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pts: Set[Tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                pts.add((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if visited[nr, nc] or int(room[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, pts))
    return comps


def _components_with_map(room: np.ndarray, color: int):
    h, w = room.shape
    visited = np.zeros((h, w), dtype=bool)
    comp_list: List[Set[Tuple[int, int]]] = []
    boxes: List[Tuple[int, int, int, int]] = []
    id_map: Dict[Tuple[int, int], int] = {}
    comp_id = 0
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(h):
        for c in range(w):
            if int(room[r, c]) != color or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pts: Set[Tuple[int, int]] = set()
            min_r = min_c = 10**9
            max_r = max_c = -1
            while stack:
                cr, cc = stack.pop()
                pts.add((cr, cc))
                id_map[(cr, cc)] = comp_id
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if visited[nr, nc] or int(room[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comp_list.append(pts)
            boxes.append((min_r, min_c, max_r, max_c))
            comp_id += 1

    return comp_list, id_map, boxes


def _room_slices(length: int, cuts: Iterable[int]) -> List[Tuple[int, int]]:
    sorted_cuts = sorted({x for x in cuts if 0 <= x < length})
    if not sorted_cuts:
        return [(0, length - 1)]
    intervals: List[Tuple[int, int]] = []
    start = 0
    for cut in sorted_cuts:
        if start < cut:
            intervals.append((start, cut - 1))
        start = cut + 1
    if start < length:
        intervals.append((start, length - 1))
    return intervals


def _detect_separator_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    row_count: Dict[int, int] = {}
    col_count: Dict[int, int] = {}
    for r in range(h):
        color = int(grid[r, 0])
        if np.all(grid[r] == color):
            row_count[color] = row_count.get(color, 0) + 1
    for c in range(w):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color):
            col_count[color] = col_count.get(color, 0) + 1
    best: Optional[int] = None
    best_score = -1
    for color in set(row_count).intersection(col_count):
        if color == bg:
            continue
        score = row_count[color] + col_count[color]
        if score > best_score:
            best_score = score
            best = color
    if best is not None:
        return best
    # Fallback: pick the color with most full lines.
    all_count: Dict[int, int] = {}
    for k, v in row_count.items():
        all_count[k] = all_count.get(k, 0) + v
    for k, v in col_count.items():
        all_count[k] = all_count.get(k, 0) + v
    if not all_count:
        return bg
    return max(all_count.items(), key=lambda kv: kv[1])[0]


def _find_room_transform(
    room: np.ndarray,
    bg: int,
    sep: int,
) -> Optional[Tuple[int, int, str]]:
    colors = {int(c) for c in np.unique(room)} - {bg, sep}
    for maroon in colors:
        comps, id_map, boxes = _components_with_map(room, maroon)
        if len(comps) < 2:
            continue
        if not id_map:
            continue
        for orange in colors:
            if orange == maroon:
                continue
            for r in range(1, room.shape[0] - 1):
                for c in range(1, room.shape[1] - 1):
                    if int(room[r, c]) != orange:
                        continue
                    # Horizontal teacher layout: source is left, target is right.
                    if room[r, c - 1] == maroon and room[r, c + 1] == maroon:
                        id_a = id_map.get((r, c - 1))
                        id_b = id_map.get((r, c + 1))
                        if id_a is None or id_b is None or id_a == id_b:
                            continue
                        b1, b2 = boxes[id_a], boxes[id_b]
                        if b1[1] <= b2[1]:
                            src_comp, dst_comp = comps[id_a], comps[id_b]
                        else:
                            src_comp, dst_comp = comps[id_b], comps[id_a]
                        op = _infer_transform_from_pair(src_comp, dst_comp)
                        if op is not None:
                            return maroon, orange, op
                    # Vertical teacher layout: source is top, target is bottom.
                    if room[r - 1, c] == maroon and room[r + 1, c] == maroon:
                        id_a = id_map.get((r - 1, c))
                        id_b = id_map.get((r + 1, c))
                        if id_a is None or id_b is None or id_a == id_b:
                            continue
                        b1, b2 = boxes[id_a], boxes[id_b]
                        if b1[0] <= b2[0]:
                            src_comp, dst_comp = comps[id_a], comps[id_b]
                        else:
                            src_comp, dst_comp = comps[id_b], comps[id_a]
                        op = _infer_transform_from_pair(src_comp, dst_comp)
                        if op is not None:
                            return maroon, orange, op
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Apply the classroom-specific transformation taught by maroon teacher pairs to all student patterns.
Input structure:
The grid is black background divided by grey full separator lines into equal rectangular classrooms. Each classroom contains one maroon teacher pair (before/after), with a single orange pixel between them, plus other colored student objects.
Rule:
For each classroom, infer the same geometric rule from the maroon pair (flips and 90-degree rotations), apply it to every student object in that room, and erase the teacher pair/connector in the output. Use the transformed student objects as output colors at transformed positions.
Key insight:
Each room has an independent rule, so the transform must be solved locally from the teacher pair before transforming any objects in that room.
"""

    grid = np.array(grid, dtype=np.int64)
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    sep = _detect_separator_color(grid, bg)

    row_lines = [r for r in range(h) if np.all(grid[r] == sep)]
    col_lines = [c for c in range(w) if np.all(grid[:, c] == sep)]
    row_rooms = _room_slices(h, row_lines)
    col_rooms = _room_slices(w, col_lines)

    out = grid.copy()
    for r0, r1 in row_rooms:
        for c0, c1 in col_rooms:
            room = grid[r0 : r1 + 1, c0 : c1 + 1]
            inferred = _find_room_transform(room, bg, sep)
            if inferred is None:
                continue
            maroon, orange, op = inferred

            out_room = out[r0 : r1 + 1, c0 : c1 + 1]
            room_h, room_w = out_room.shape
            blocked = {bg, sep, maroon, orange}
            out_room[out_room != sep] = bg
            comps = _iter_components(room, blocked)
            for color, pixels in comps:
                if color in blocked:
                    continue
                rs = [p[0] for p in pixels]
                cs = [p[1] for p in pixels]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                rel = [(r - min_r, c - min_c) for r, c in pixels]
                h0 = max_r - min_r + 1
                w0 = max_c - min_c + 1
                dst_points, (h1, w1) = _transform_points(rel, h0, w0, op)

                if min_r + h1 > room_h or min_c + w1 > room_w:
                    # If the inferred transform does not fit in this room, keep no pixels for safety.
                    continue
                for tr, tc in dst_points:
                    out_room[min_r + tr, min_c + tc] = color
    return out


def _transform_size_for_op(h: int, w: int, op: str) -> Tuple[int, int]:
    if op in {"flip_v", "flip_h", "rot180"}:
        return h, w
    if op in {"rot90", "rot270"}:
        return w, h
    raise ValueError(f"unsupported op: {op}")


def _random_connected_in_box(h: int, w: int, min_cells: int = 2) -> Set[Tuple[int, int]]:
    max_cells = h * w
    target = random.randint(min_cells, max_cells)
    start = (random.randrange(h), random.randrange(w))
    pts: Set[Tuple[int, int]] = {start}
    frontier = [start]
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while len(pts) < target:
        cr, cc = random.choice(frontier)
        dr, dc = random.choice(dirs)
        nr, nc = cr + dr, cc + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            continue
        if (nr, nc) in pts:
            continue
        pts.add((nr, nc))
        frontier.append((nr, nc))
    return pts


def _random_square_shape(side: int, min_area: int = 3, max_area: Optional[int] = None) -> Set[Tuple[int, int]]:
    if max_area is None:
        max_area = side * side
    max_area = max(min_area, min(max_area, side * side))
    target = random.randint(min_area, max_area)
    while True:
        cells = _random_connected_in_box(side, side, min_cells=min(2, target))
        if len(cells) >= min_area:
            return cells


def _random_student_shape() -> Tuple[Set[Tuple[int, int]], Tuple[int, int]]:
    h = random.randint(1, 3)
    w = random.randint(1, 3)
    target_min = min(2, h * w)
    cells = _random_connected_in_box(h, w, min_cells=target_min)
    if len(cells) == h * w:
        return cells, (h, w)
    return _normalize_points(tuple(cells))


def generate_input() -> np.ndarray:
    """Generate one valid puzzle input with per-room geometric rules and student objects."""
    bg = 0
    sep = 5
    maroon = 9
    orange = 7
    student_colors = [1, 2, 3, 4, 6, 8]

    while True:
        room_rows = random.randint(2, 3)
        room_cols = random.randint(2, 3)
        room_h = random.randint(7, 9)
        room_w = random.randint(7, 9)
        H = room_rows * room_h + (room_rows - 1)
        W = room_cols * room_w + (room_cols - 1)
        if H <= 30 and W <= 30:
            break

    grid = np.full((H, W), bg, dtype=np.int64)
    for rr in range(room_rows - 1):
        sep_row = (rr + 1) * room_h + rr
        grid[sep_row, :] = sep
    for cc in range(room_cols - 1):
        sep_col = (cc + 1) * room_w + cc
        grid[:, sep_col] = sep

    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = rr * (room_h + 1)
            c0 = cc * (room_w + 1)
            room = grid[r0 : r0 + room_h, c0 : c0 + room_w]
            op = random.choice(OPS_2D)
            orient = random.choice(("h", "v"))
            side = random.choice((2, 3))

            while True:
                src_shape = _random_connected_in_box(side, side, min_cells=3)
                src_shape, (src_h, src_w) = _normalize_points(tuple(src_shape))
                dst_shape, _ = _transform_points(src_shape, src_h, src_w, op)
                if dst_shape != src_shape:
                    break

            if orient == "h":
                pair_w = side * 2 + 1
                top = random.randint(0, room_h - side)
                left = random.randint(0, room_w - pair_w)
                sep_pos = (top + random.randint(0, side - 1), left + side)
                for sr, sc in src_shape:
                    room[top + sr, left + sc] = maroon
                for dr, dc in dst_shape:
                    room[top + dr, left + side + 1 + dc] = maroon
                room[sep_pos] = orange
            else:
                pair_h = side * 2 + 1
                top = random.randint(0, room_h - pair_h)
                left = random.randint(0, room_w - side)
                sep_pos = (top + side, left + random.randint(0, side - 1))
                for sr, sc in src_shape:
                    room[top + sr, left + sc] = maroon
                for dr, dc in dst_shape:
                    room[top + side + 1 + dr, left + dc] = maroon
                room[sep_pos] = orange

            used_pre: Set[Tuple[int, int]] = set()
            used_post: Set[Tuple[int, int]] = set()
            for pr in range(room_h):
                for pc in range(room_w):
                    if room[pr, pc] != bg:
                        used_pre.add((pr, pc))
            used_pre_local = set(used_pre)

            students = random.randint(2, 4)
            for si in range(students):
                color = student_colors[(rr * room_cols + cc + si) % len(student_colors)]
                shape, shape_h, shape_w = None, 1, 1
                while True:
                    shape_tmp = _random_student_shape()
                    if len(shape_tmp[0]) < 2:
                        continue
                    shape, (shape_h, shape_w) = shape_tmp
                    break
                t_h, t_w = _transform_size_for_op(shape_h, shape_w, op)
                max_r = room_h - max(shape_h, t_h)
                max_c = room_w - max(shape_w, t_w)
                if max_r < 0 or max_c < 0:
                    continue
                placed = False
                for _ in range(120):
                    sr = random.randint(0, max_r)
                    sc = random.randint(0, max_c)
                    pre = {(sr + r, sc + c) for r, c in shape}
                    if pre & used_pre_local:
                        continue
                    post, _ = _transform_points(shape, shape_h, shape_w, op)
                    post_shift = {(sr + r, sc + c) for r, c in post}
                    if post_shift & used_post:
                        continue
                    # Keep clear output and avoid stacking transformed students.
                    if any(room[rp, cp] != bg for rp, cp in post_shift):
                        continue
                    for pr, pc in pre:
                        room[pr, pc] = color
                    for tr, tc in post_shift:
                        used_post.add((tr, tc))
                    used_pre_local |= pre
                    placed = True
                    break
                if not placed:
                    continue

    return grid

