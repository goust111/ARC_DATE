from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 6
BORDER = 0
CONCAVE = 2
CONVEX = 4
GREY = 5
ORANGE = 7
MARKER = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

SMALL_BORDER_TEMPLATES = [
    [(1, 5), (1, 6), (2, 6), (3, 6), (4, 6), (4, 5), (4, 4), (4, 3), (3, 3), (2, 3), (2, 4), (1, 4)],
    [(5, 2), (5, 3), (4, 3), (3, 3), (3, 2), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (6, 1), (6, 2)],
    [(0, 4), (0, 3), (0, 2), (0, 1), (1, 1), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (2, 4), (2, 5), (1, 5), (0, 5)],
]

ENERGY8_BORDER_TEMPLATES = [
    [(1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (7, 4), (7, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (3, 6), (3, 5), (2, 5), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1)],
    [(2, 5), (3, 5), (3, 6), (4, 6), (5, 6), (5, 7), (6, 7), (7, 7), (7, 6), (7, 5), (7, 4), (7, 3), (7, 2), (7, 1), (6, 1), (5, 1), (4, 1), (4, 2), (3, 2), (3, 3), (2, 3), (2, 4)],
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(sorted(pts), dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        deg = sum((r + dr, c + dc) in pts for dr, dc in DIRS4)
        assert deg == 2, (r, c, deg)

    start = min(pts)
    neigh = sorted((start[0] + dr, start[1] + dc) for dr, dc in DIRS4 if (start[0] + dr, start[1] + dc) in pts)
    assert len(neigh) == 2

    path = [start]
    prev, cur = start, neigh[0]
    while cur != start:
        path.append(cur)
        nxt = [(cur[0] + dr, cur[1] + dc) for dr, dc in DIRS4 if (cur[0] + dr, cur[1] + dc) in pts and (cur[0] + dr, cur[1] + dc) != prev]
        assert len(nxt) == 1, (cur, nxt)
        prev, cur = cur, nxt[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _turn_data(path: list[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], int, int]:
    counts: Counter[int] = Counter()
    turns: dict[tuple[int, int], int] = {}
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            continue
        sign = 1 if cross > 0 else -1
        turns[(r, c)] = sign
        counts[sign] += 1
    assert turns
    assert abs(counts[1] - counts[-1]) == 4, counts
    convex_sign = 1 if counts[1] > counts[-1] else -1
    return turns, convex_sign, int(counts[convex_sign])


def _enclosed_region(border_coords: np.ndarray) -> np.ndarray | None:
    r0, c0, r1, c1 = _bbox(border_coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    wall = np.zeros((h, w), dtype=bool)
    local = border_coords - np.array([r0, c0], dtype=int)
    wall[local[:, 0], local[:, 1]] = True
    passable = ~wall

    outside = np.zeros((h, w), dtype=bool)
    q: list[tuple[int, int]] = []
    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.pop()
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    comps = _components(enclosed)
    if len(comps) != 1:
        return None
    return comps[0] + np.array([r0, c0], dtype=int)


def _charge_order(interior: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    pts = [tuple(map(int, p)) for p in interior]
    if direction == (0, 1):
        key = lambda p: (-p[1], p[0])
    elif direction == (0, -1):
        key = lambda p: (p[1], p[0])
    elif direction == (1, 0):
        key = lambda p: (-p[0], p[1])
    else:
        key = lambda p: (p[0], p[1])
    return sorted(pts, key=key)


def _find_head_and_dest(
    g: np.ndarray,
    border_coords: np.ndarray,
    interior_coords: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]:
    border = {tuple(map(int, p)) for p in border_coords}
    interior = {tuple(map(int, p)) for p in interior_coords}
    r0, c0, r1, c1 = _bbox(border_coords)

    candidates: dict[tuple[int, int], tuple[int, int]] = {}
    for hr, hc in np.argwhere(g == MARKER):
        hr = int(hr)
        hc = int(hc)
        if (hr, hc) in border or (hr, hc) in interior:
            continue
        adj = [(hr - dr, hc - dc, (dr, dc)) for dr, dc in DIRS4 if (hr - dr, hc - dc) in border]
        if len(adj) != 1:
            continue
        _br, _bc, direction = adj[0]
        if not (hr < r0 or hr > r1 or hc < c0 or hc > c1):
            continue
        candidates[(hr, hc)] = direction

    assert len(candidates) == 1, candidates
    head = next(iter(candidates))
    direction = candidates[head]

    dr, dc = direction
    dist = 0
    r, c = head[0], head[1]
    while True:
        r += dr
        c += dc
        dist += 1
        assert _in_bounds(g.shape, r, c), "marker ray must hit a destination before leaving the grid"
        val = int(g[r, c])
        if val == MARKER:
            return head, (r, c), direction, dist
        assert val == BG, (head, direction, r, c, val)


def _detect_containers(g: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for comp in _components(g == BORDER):
        interior = _enclosed_region(comp)
        if interior is None or interior.size == 0:
            continue
        vals = {int(g[r, c]) for r, c in interior}
        if not vals <= {GREY, ORANGE}:
            continue
        path = _trace_cycle(comp)
        turns, convex_sign, convex_count = _turn_data(path)
        head, dest, direction, head_dist = _find_head_and_dest(g, comp, interior)
        bend_colors = {
            cell: CONVEX if sign == convex_sign else CONCAVE
            for cell, sign in turns.items()
        }
        containers.append(
            {
                "border": comp,
                "border_set": {tuple(map(int, p)) for p in comp},
                "interior": interior,
                "head": head,
                "dest": dest,
                "dir": direction,
                "head_dist": head_dist,
                "convex_count": convex_count,
                "bend_colors": bend_colors,
                "turns": turns,
            }
        )

    assert containers, "expected at least one bordered container"
    return containers


def _solve_with_details(g: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    containers = _detect_containers(g)
    out = np.full_like(g, BG)

    for info in containers:
        out[info["dest"]] = MARKER

    details: list[dict] = []
    for info in containers:
        steps = min(int(info["convex_count"]), int(len(info["interior"])))
        dr, dc = info["dir"]
        tr, tc = dr * steps, dc * steps
        ordered = _charge_order(info["interior"], info["dir"])
        charged = set(ordered[:steps])

        for r, c in map(tuple, info["interior"]):
            rr, cc = int(r + tr), int(c + tc)
            out[rr, cc] = ORANGE if (int(r), int(c)) in charged else GREY

        for r, c in map(tuple, info["border"]):
            rr, cc = int(r + tr), int(c + tc)
            out[rr, cc] = int(info["bend_colors"].get((int(r), int(c)), BORDER))

        out[int(info["head"][0] + tr), int(info["head"][1] + tc)] = MARKER
        details.append(
            {
                **info,
                "steps": steps,
                "translation": (tr, tc),
                "charged_count": steps,
            }
        )

    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each irregular black container toward its maroon target by as many steps as its convex border bends allow, charging front grey cells as it goes.

    Input structure:
    - The background is magenta.
    - Two or three 1-pixel-thick black closed orthogonal border loops enclose grey interiors; the loops can have concave notches, so they are not plain rectangles.
    - Each container has one adjacent maroon head pixel on an outer side and one farther maroon destination pixel on the same row or column.

    Rule:
    - Trace every black border cycle and classify each bend by its turn sign; bends with the majority sign are convex and the minority bends are concave.
    - Let the number of convex bends be the container energy.
    - Move the whole container together with its head toward the matching destination by that many steps, but stop early if the container runs out of grey interior pixels.
    - After each step, charge one interior pixel from the motion-facing front inward, turning it from grey to orange.
    - In the output, straight border pixels stay black, convex bends become yellow, concave bends become red, and destination markers remain maroon.

    Key insight:
    - The move distance is not encoded by the head-marker gap or by the raw interior size alone; it comes from tracing the border geometry first, because the convex-corner count defines the energy budget that is then capped by the available grey cells.
    """
    g = np.asarray(grid, dtype=int)
    out, _details = _solve_with_details(g)
    return out


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[0] != cells[0] and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _path_from_profiles(top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    width = len(top)
    path: list[tuple[int, int]] = []
    _append_segment(path, top[0], 0, top[0], 0)
    for c in range(width - 1):
        _append_segment(path, top[c], c, top[c], c + 1)
        _append_segment(path, top[c], c + 1, top[c + 1], c + 1)
    _append_segment(path, top[-1], width - 1, bottom[-1], width - 1)
    for c in range(width - 1, 0, -1):
        _append_segment(path, bottom[c], c, bottom[c], c - 1)
        _append_segment(path, bottom[c], c - 1, bottom[c - 1], c - 1)
    _append_segment(path, bottom[0], 0, top[0], 0)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _build_shape_from_profiles(top: list[int], bottom: list[int]) -> dict | None:
    h = max(bottom) + 1
    w = len(top)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in _path_from_profiles(top, bottom):
        mask[r, c] = True
    comps = _components(mask)
    if len(comps) != 1:
        return None
    border = comps[0]
    interior = _enclosed_region(border)
    if interior is None or interior.size == 0:
        return None
    try:
        path = _trace_cycle(border)
        turns, convex_sign, convex_count = _turn_data(path)
    except AssertionError:
        return None
    bend_colors = {
        cell: CONVEX if sign == convex_sign else CONCAVE
        for cell, sign in turns.items()
    }
    border_map = np.zeros((h, w), dtype=int)
    border_map[border[:, 0], border[:, 1]] = 1
    fill = border_map.copy()
    fill[interior[:, 0], interior[:, 1]] = 1
    return {
        "border": border,
        "interior": interior,
        "path": path,
        "turns": turns,
        "bend_colors": bend_colors,
        "convex_count": convex_count,
        "fill": fill,
    }


def _shape_from_border_points(points: list[tuple[int, int]]) -> dict:
    coords = np.array(points, dtype=int)
    r0, c0, _r1, _c1 = _bbox(coords)
    coords = coords - np.array([r0, c0], dtype=int)
    border = coords.astype(int)
    interior = _enclosed_region(border)
    assert interior is not None and interior.size > 0
    path = _trace_cycle(border)
    turns, convex_sign, convex_count = _turn_data(path)
    bend_colors = {
        cell: CONVEX if sign == convex_sign else CONCAVE
        for cell, sign in turns.items()
    }
    h = int(border[:, 0].max()) + 1
    w = int(border[:, 1].max()) + 1
    fill = np.zeros((h, w), dtype=int)
    fill[border[:, 0], border[:, 1]] = 1
    fill[interior[:, 0], interior[:, 1]] = 1
    return {
        "border": border,
        "interior": interior,
        "path": path,
        "turns": turns,
        "bend_colors": bend_colors,
        "convex_count": convex_count,
        "fill": fill,
    }


def _transform_shape(shape: dict, k: int, flip: bool) -> dict:
    patch = np.full(shape["fill"].shape, BG, dtype=int)
    patch[shape["interior"][:, 0], shape["interior"][:, 1]] = GREY
    patch[shape["border"][:, 0], shape["border"][:, 1]] = BORDER
    if flip:
        patch = np.fliplr(patch)
    rot = np.rot90(patch, k)
    border = np.argwhere(rot == BORDER)
    interior = np.argwhere(rot == GREY)
    path = _trace_cycle(border)
    turns, convex_sign, convex_count = _turn_data(path)
    bend_colors = {
        cell: CONVEX if sign == convex_sign else CONCAVE
        for cell, sign in turns.items()
    }
    fill = (rot != BG).astype(int)
    return {
        "border": border,
        "interior": interior,
        "path": path,
        "turns": turns,
        "bend_colors": bend_colors,
        "convex_count": convex_count,
        "fill": fill,
    }


def _outer_head_candidates(shape: dict) -> dict[tuple[int, int], list[tuple[int, int]]]:
    border = {tuple(map(int, p)) for p in shape["border"]}
    interior = {tuple(map(int, p)) for p in shape["interior"]}
    filled = border | interior
    coords = np.array(sorted(border), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    straight = set(border) - set(shape["turns"])
    out: dict[tuple[int, int], list[tuple[int, int]]] = {(dr, dc): [] for dr, dc in DIRS4}
    for r, c in straight:
        if r == r0 and (r - 1, c) not in filled:
            out[(-1, 0)].append((r - 1, c))
        if r == r1 and (r + 1, c) not in filled:
            out[(1, 0)].append((r + 1, c))
        if c == c0 and (r, c - 1) not in filled:
            out[(0, -1)].append((r, c - 1))
        if c == c1 and (r, c + 1) not in filled:
            out[(0, 1)].append((r, c + 1))
    return {k: sorted(v) for k, v in out.items() if v}


def _sample_intervals(width: int, count: int) -> list[tuple[int, int]] | None:
    intervals = []
    for start in range(2, width - 3):
        for length in (2, 3):
            end = start + length - 1
            if end <= width - 3:
                intervals.append((start, end))
    random.shuffle(intervals)
    chosen: list[tuple[int, int]] = []
    for start, end in intervals:
        if all(end + 2 < s or e + 2 < start for s, e in chosen):
            chosen.append((start, end))
            if len(chosen) == count:
                return chosen
    return None


def _sample_shape(branch: str, target_convex: int) -> dict:
    if branch == "grey":
        for _ in range(200):
            points = random.choice(SMALL_BORDER_TEMPLATES)
            shape = _shape_from_border_points(points)
            shape = _transform_shape(shape, random.randint(0, 3), random.choice([False, True]))
            area = int(len(shape["interior"]))
            if int(shape["convex_count"]) != target_convex:
                continue
            if not (1 <= area < target_convex):
                continue
            candidates = _outer_head_candidates(shape)
            if not candidates:
                continue
            shape["head_candidates"] = candidates
            return shape
        raise RuntimeError(f"failed to sample {branch} shape with convex={target_convex}")

    if branch == "energy" and target_convex == 8:
        for _ in range(200):
            points = random.choice(ENERGY8_BORDER_TEMPLATES)
            shape = _shape_from_border_points(points)
            shape = _transform_shape(shape, random.randint(0, 3), random.choice([False, True]))
            area = int(len(shape["interior"]))
            if area < target_convex + 1:
                continue
            candidates = _outer_head_candidates(shape)
            if not candidates:
                continue
            shape["head_candidates"] = candidates
            return shape
        raise RuntimeError(f"failed to sample {branch} shape with convex={target_convex}")

    notch_count = 1 if target_convex == 6 else 2
    for _ in range(300):
        h = random.randint(6, 9)
        w = random.randint(7, 10)
        if w < 7 and notch_count == 2:
            continue
        intervals = _sample_intervals(w, notch_count)
        if intervals is None:
            continue
        top = [0] * w
        bottom = [h - 1] * w
        sides = random.sample(["top", "bottom"] * notch_count, notch_count)
        max_depth = max(1, (h - 3) // 2)
        for (start, end), side in zip(intervals, sides):
            depth = random.randint(1, max_depth)
            if side == "top":
                for c in range(start, end + 1):
                    top[c] = depth
            else:
                for c in range(start, end + 1):
                    bottom[c] = h - 1 - depth
        if any(top[c] + 2 > bottom[c] for c in range(w)):
            continue

        shape = _build_shape_from_profiles(top, bottom)
        if shape is None:
            continue
        shape = _transform_shape(shape, random.randint(0, 3), random.choice([False, True]))
        area = int(len(shape["interior"]))
        if int(shape["convex_count"]) != target_convex:
            continue
        if area < target_convex + 1:
            continue
        candidates = _outer_head_candidates(shape)
        if not candidates:
            continue
        shape["head_candidates"] = candidates
        return shape
    raise RuntimeError(f"failed to sample {branch} shape with convex={target_convex}")


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr, dc in DIRS8 + [(0, 0)]:
            rr, cc = r + dr, c + dc
            if _in_bounds(shape, rr, cc):
                out.add((rr, cc))
    return out


def _shift_points(points: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in points}


def _segment_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _place_containers(h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    occupied_in: set[tuple[int, int]] = set()
    occupied_out: set[tuple[int, int]] = set()
    blocked_in: set[tuple[int, int]] = set()
    blocked_out: set[tuple[int, int]] = set()
    placed: list[dict] = []
    used_dirs: set[tuple[int, int]] = set()

    for idx, spec in enumerate(specs):
        dirs = list(spec["head_candidates"].keys())
        random.shuffle(dirs)
        dirs.sort(key=lambda d: (d in used_dirs, random.random()))
        ok = False
        for direction in dirs:
            heads = spec["head_candidates"][direction][:]
            random.shuffle(heads)
            for head_rel in heads:
                border_rel = [tuple(map(int, p)) for p in spec["border"]]
                interior_rel = [tuple(map(int, p)) for p in spec["interior"]]
                local_all = set(border_rel) | set(interior_rel) | {head_rel}
                steps = min(int(spec["convex_count"]), len(interior_rel))
                move_dist = int(spec["convex_count"]) + random.randint(1, 4)
                dest_rel = (head_rel[0] + direction[0] * (move_dist - steps), head_rel[1] + direction[1] * (move_dist - steps))
                input_shift = (-direction[0] * steps, -direction[1] * steps)

                rel_all = list(local_all | {dest_rel} | _shift_points(local_all, *input_shift))
                min_r = min(r for r, _c in rel_all)
                max_r = max(r for r, _c in rel_all)
                min_c = min(c for _r, c in rel_all)
                max_c = max(c for _r, c in rel_all)
                if max_r - min_r + 1 > h or max_c - min_c + 1 > w:
                    continue

                row_choices = list(range(-min_r, h - max_r))
                col_choices = list(range(-min_c, w - max_c))
                random.shuffle(row_choices)
                random.shuffle(col_choices)
                for fr in row_choices[:50]:
                    for fc in col_choices[:50]:
                        out_border = _shift_points(border_rel, fr, fc)
                        out_interior = _shift_points(interior_rel, fr, fc)
                        out_head = (head_rel[0] + fr, head_rel[1] + fc)
                        dest = (dest_rel[0] + fr, dest_rel[1] + fc)

                        in_border = _shift_points(out_border, *input_shift)
                        in_interior = _shift_points(out_interior, *input_shift)
                        in_head = (out_head[0] + input_shift[0], out_head[1] + input_shift[1])

                        input_cells = set(in_border) | set(in_interior) | {in_head, dest}
                        output_cells = set(out_border) | set(out_interior) | {out_head, dest}
                        if any(not _in_bounds((h, w), r, c) for r, c in input_cells | output_cells):
                            continue

                        path_in = _segment_points(in_head, dest)[1:-1]
                        path_out = _segment_points(out_head, dest)[1:-1]
                        if any(p in occupied_in for p in path_in) or any(p in occupied_out for p in path_out):
                            continue
                        if input_cells & blocked_in or output_cells & blocked_out:
                            continue
                        if any(p in blocked_in for p in path_in) or any(p in blocked_out for p in path_out):
                            continue

                        occupied_in |= input_cells
                        occupied_out |= output_cells
                        blocked_in |= _expand(input_cells, (h, w))
                        blocked_out |= _expand(output_cells, (h, w))
                        blocked_in |= set(path_in)
                        blocked_out |= set(path_out)
                        used_dirs.add(direction)
                        placed.append(
                            {
                                **spec,
                                "dir": direction,
                                "steps": steps,
                                "move_dist": move_dist,
                                "origin": (fr, fc),
                                "head_rel": head_rel,
                                "in_border": in_border,
                                "in_interior": in_interior,
                                "in_head": in_head,
                                "out_border": out_border,
                                "out_interior": out_interior,
                                "out_head": out_head,
                                "dest": dest,
                            }
                        )
                        ok = True
                        break
                    if ok:
                        break
                if ok:
                    break
            if ok:
                break
        if not ok:
            return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for item in placed:
        for r, c in item["in_interior"]:
            inp[r, c] = GREY
        for r, c in item["in_border"]:
            inp[r, c] = BORDER
        inp[item["in_head"]] = MARKER
        inp[item["dest"]] = MARKER

        ordered = _charge_order(np.array(sorted(item["out_interior"]), dtype=int), item["dir"])
        charged = set(ordered[: item["steps"]])
        for r, c in item["out_interior"]:
            out[r, c] = ORANGE if (r, c) in charged else GREY
        fr, fc = item["origin"]
        for r, c in item["out_border"]:
            local = (r - fr, c - fc)
            out[r, c] = int(item["bend_colors"].get(local, BORDER))
        out[item["out_head"]] = MARKER
        out[item["dest"]] = MARKER

    return inp, out, placed


def _make_specs(n: int) -> list[dict]:
    base = [
        _sample_shape("energy", 8),
        _sample_shape("grey", random.choice([5, 6])),
    ]
    if n == 3:
        branch, convex = random.choice(
            [("energy", 6), ("energy", 8), ("grey", 5), ("grey", 6)]
        )
        base.append(_sample_shape(branch, convex))
    random.shuffle(base)
    return base


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(16, 22)
        w = random.randint(19, 27)
        n = random.choice([2, 3])
        specs = _make_specs(n)
        placed = _place_containers(h, w, specs)
        if placed is None:
            continue
        inp, expected, meta = placed
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        if not (Counter(map(int, inp.flat))[BG] > max(Counter(map(int, inp.flat))[c] for c in (BORDER, GREY, MARKER))):
            continue

        out, details = _solve_with_details(inp)
        if not np.array_equal(out, expected):
            continue
        if len(details) != n:
            continue
        if len({d["convex_count"] for d in details}) < 2:
            continue
        if not any(d["steps"] < d["convex_count"] for d in details):
            continue
        if not any(d["steps"] == d["convex_count"] for d in details):
            continue
        if np.count_nonzero(out == CONVEX) == 0 or np.count_nonzero(out == CONCAVE) == 0:
            continue
        if np.array_equal(inp, out):
            continue
        if any(d["head_dist"] <= d["convex_count"] for d in details):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
