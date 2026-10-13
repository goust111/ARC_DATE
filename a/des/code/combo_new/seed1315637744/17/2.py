from __future__ import annotations

import random

import numpy as np


ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PALETTE = list(range(1, 10))

GEN_RETRIES = 400
GROUP_RETRIES = 220
OBJECT_RETRIES = 180
SHIFT_RETRIES = 80


def _bg_color(grid: np.ndarray) -> int:
    if bool(np.any(grid == 0)):
        return 0
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _same_color_neighbors(cell: tuple[int, int], cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ORTHO4:
        nxt = (r + dr, c + dc)
        if nxt in cells:
            out.append(nxt)
    for dr, dc in DIAG4:
        nxt = (r + dr, c + dc)
        if nxt not in cells:
            continue
        if (r, c + dc) in cells or (r + dr, c) in cells:
            continue
        out.append(nxt)
    return out


def _collect_components(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    comps: list[dict] = []
    pixel_to_comp: dict[tuple[int, int], int] = {}

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        all_cells = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
        unseen = set(all_cells)
        while unseen:
            start = unseen.pop()
            stack = [start]
            comp_cells = {start}
            while stack:
                cur = stack.pop()
                for nxt in _same_color_neighbors(cur, all_cells):
                    if nxt not in unseen:
                        continue
                    unseen.remove(nxt)
                    stack.append(nxt)
                    comp_cells.add(nxt)

            idx = len(comps)
            cells_sorted = sorted(comp_cells)
            comps.append(
                {
                    "id": idx,
                    "color": color,
                    "cells": cells_sorted,
                    "set": comp_cells,
                    "size": len(cells_sorted),
                }
            )
            for cell in cells_sorted:
                pixel_to_comp[cell] = idx

    return comps, pixel_to_comp


def _degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {cell: len(_same_color_neighbors(cell, cells)) for cell in cells}


def _neighbor_components4(
    cell: tuple[int, int], pixel_to_comp: dict[tuple[int, int], int], self_id: int
) -> set[int]:
    r, c = cell
    out: set[int] = set()
    for dr, dc in ORTHO4:
        nxt = (r + dr, c + dc)
        comp_id = pixel_to_comp.get(nxt)
        if comp_id is not None and comp_id != self_id:
            out.add(comp_id)
    return out


def _touch_points4(cells: set[tuple[int, int]], other: set[tuple[int, int]]) -> set[tuple[int, int]]:
    touched: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in ORTHO4:
            nbr = (r + dr, c + dc)
            if nbr in other:
                touched.add(nbr)
    return touched


def _straight_line(a: tuple[int, int], b: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    if dr == 0 and dc == 0:
        return [a], (0, 0)
    sr = _sign(dr)
    sc = _sign(dc)
    if dr == 0 and dc != 0:
        steps = abs(dc)
    elif dc == 0 and dr != 0:
        steps = abs(dr)
    elif abs(dr) == abs(dc) and dr != 0:
        steps = abs(dr)
    else:
        return None
    pts = [(a[0] + sr * t, a[1] + sc * t) for t in range(steps + 1)]
    return pts, (sr, sc)


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _identify_groups(grid: np.ndarray) -> list[dict]:
    bg = _bg_color(grid)
    comps, pixel_to_comp = _collect_components(grid, bg)
    groups: list[dict] = []
    used_objects: set[int] = set()

    for comp in comps:
        cells = comp["set"]
        if comp["size"] < 2:
            continue
        deg = _degrees(cells)
        endpoints = sorted(cell for cell, d in deg.items() if d == 1)
        if len(endpoints) != 2:
            continue
        if any(d not in (1, 2) for d in deg.values()):
            continue

        line_info = _straight_line(endpoints[0], endpoints[1])
        if line_info is None:
            continue
        line_pts, direction = line_info
        diff = comp["size"] - len(line_pts)
        if diff < 0 or diff % 2 != 0:
            continue

        valid = True
        for cell in cells:
            others = _neighbor_components4(cell, pixel_to_comp, comp["id"])
            if cell not in endpoints and others:
                valid = False
                break
        if not valid:
            continue

        obj_ids: list[int] = []
        for endpoint in endpoints:
            others = _neighbor_components4(endpoint, pixel_to_comp, comp["id"])
            if len(others) != 1:
                valid = False
                break
            obj_ids.append(next(iter(others)))
        if not valid or obj_ids[0] == obj_ids[1]:
            continue
        if any(obj_id in used_objects for obj_id in obj_ids):
            continue

        objects: list[dict] = []
        for obj_id in obj_ids:
            obj = comps[obj_id]
            if not (2 <= obj["size"] <= 4):
                valid = False
                break
            if obj["color"] == comp["color"]:
                valid = False
                break
            touched_by = set()
            for cell in obj["cells"]:
                touched_by |= _neighbor_components4(cell, pixel_to_comp, obj_id)
            if touched_by != {comp["id"]}:
                valid = False
                break
            objects.append(obj)
        if not valid:
            continue

        used_objects.update(obj_ids)
        groups.append(
            {
                "rope_color": comp["color"],
                "rope_cells": cells,
                "endpoints": endpoints,
                "direction": direction,
                "line_cells": set(line_pts),
                "move": diff // 2,
                "objects": [
                    {"color": objects[0]["color"], "cells": objects[0]["set"]},
                    {"color": objects[1]["color"], "cells": objects[1]["set"]},
                ],
            }
        )

    return groups


def _actual_shift(
    obj0: set[tuple[int, int]],
    obj1: set[tuple[int, int]],
    direction: tuple[int, int],
    wanted: int,
) -> int:
    dr, dc = direction
    for shift in range(wanted, -1, -1):
        moved0 = _translate(obj0, shift * dr, shift * dc)
        moved1 = _translate(obj1, -shift * dr, -shift * dc)
        if moved0.isdisjoint(moved1):
            return shift
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten each colored rope and slide its endpoint objects inward by half of the saved length.

    Input structure:
    - The background is black.
    - Several different colors form one-pixel-wide rope paths, each a simple polyline.
    - Each rope endpoint is 4-neighbor-adjacent to a small monochrome object of 2-4 pixels,
      and the two endpoint objects use colors different from the rope and from each other.
    - For every rope, its two endpoints lie on one row, one column, or one 45-degree diagonal.

    Rule:
    - Replace each rope by the straight shortest segment joining its endpoints, keeping the rope color.
    - Let D be the original rope pixel count minus the new straight-segment pixel count.
    - Translate the two endpoint objects rigidly toward each other along the rope direction by D/2 cells,
      clearing their old positions.
    - If those translated objects would overlap, reduce the symmetric motion until they are just short of overlap.

    Key insight:
    - The rope's bends encode hidden travel distance: the amount of length removed by straightening is exactly
      what pulls the attached objects inward.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _bg_color(g)
    groups = _identify_groups(g)
    if not groups:
        return out

    for group in groups:
        for cell in group["rope_cells"]:
            out[cell] = bg
        for obj in group["objects"]:
            for cell in obj["cells"]:
                out[cell] = bg

    for group in groups:
        direction = group["direction"]
        wanted = group["move"]
        obj0 = group["objects"][0]["cells"]
        obj1 = group["objects"][1]["cells"]
        shift = _actual_shift(obj0, obj1, direction, wanted)
        moved0 = _translate(obj0, shift * direction[0], shift * direction[1])
        moved1 = _translate(obj1, -shift * direction[0], -shift * direction[1])

        for r, c in group["line_cells"]:
            out[r, c] = group["rope_color"]
        for r, c in moved0:
            out[r, c] = group["objects"][0]["color"]
        for r, c in moved1:
            out[r, c] = group["objects"][1]["color"]

    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    line_info = _straight_line(a, b)
    assert line_info is not None
    return line_info[0]


def _join_segments(*segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for seg in segments:
        if not seg:
            continue
        if not out:
            out.extend(seg)
        else:
            out.extend(seg[1:])
    return out


def _shape_for_offset(offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    side = random.choice([-1, 1])
    if dr != 0:
        options = [
            {(0, 0), (dr, 0)},
            {(0, 0), (dr, 0), (2 * dr, 0)},
            {(0, 0), (dr, 0), (2 * dr, 0), (3 * dr, 0)},
            {(0, 0), (dr, 0), (2 * dr, 0), (2 * dr, side)},
        ]
    else:
        options = [
            {(0, 0), (0, dc)},
            {(0, 0), (0, dc), (0, 2 * dc)},
            {(0, 0), (0, dc), (0, 2 * dc), (0, 3 * dc)},
            {(0, 0), (0, dc), (0, 2 * dc), (side, 2 * dc)},
        ]
    return random.choice(options)


def _offset_options(step: tuple[int, int]) -> list[tuple[int, int]]:
    opts = list(ORTHO4)
    if step in opts:
        opts.remove(step)
    return opts


def _path_horizontal(need_overlap: bool) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], int]:
    n = random.randint(4, 7) if need_overlap else random.randint(6, 11)
    k = random.randint(2, 4) if need_overlap else random.randint(1, 3)
    side = random.choice([-1, 1])
    pivot = random.randint(0, n)
    a = (0, 0)
    b = (0, pivot)
    c = (side * k, pivot)
    d = (side * k, n)
    e = (0, n)
    path = _join_segments(_segment(a, b), _segment(b, c), _segment(c, d), _segment(d, e))
    line = _segment(a, e)
    return path, line, (0, 1), k


def _path_vertical(need_overlap: bool) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], int]:
    n = random.randint(4, 7) if need_overlap else random.randint(6, 11)
    k = random.randint(2, 4) if need_overlap else random.randint(1, 3)
    side = random.choice([-1, 1])
    pivot = random.randint(0, n)
    a = (0, 0)
    b = (pivot, 0)
    c = (pivot, side * k)
    d = (n, side * k)
    e = (n, 0)
    path = _join_segments(_segment(a, b), _segment(b, c), _segment(c, d), _segment(d, e))
    line = _segment(a, e)
    return path, line, (1, 0), k


def _path_diagonal(need_overlap: bool) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], int]:
    n = random.choice([2, 4]) if need_overlap else random.choice([4, 6, 8])
    extra = random.randint(1, 3) if need_overlap else random.randint(0, 2)
    m = extra
    sr = random.choice([-1, 1])
    sc = random.choice([-1, 1])
    a = (0, 0)
    e = (sr * n, sc * n)
    if random.choice([False, True]):
        b = (0, sc * (n + m))
        d = (sr * n, sc * (n + m))
    else:
        b = (sr * (n + m), 0)
        d = (sr * (n + m), sc * n)
    path = _join_segments(_segment(a, b), _segment(b, d), _segment(d, e))
    line = _segment(a, e)
    return path, line, (sr, sc), n // 2 + m


def _touches_exactly(
    obj: set[tuple[int, int]], target: set[tuple[int, int]], want: tuple[int, int]
) -> bool:
    return _touch_points4(obj, target) == {want}


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _choose_offsets(
    start_step: tuple[int, int], end_step: tuple[int, int], need_overlap: bool
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    opts0 = _offset_options(start_step)
    opts1 = _offset_options(end_step)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if need_overlap:
        common = [off for off in opts0 if off in opts1]
        random.shuffle(common)
        pairs.extend((off, off) for off in common)
    else:
        diff = [(a, b) for a in opts0 for b in opts1 if a != b]
        same = [(a, b) for a in opts0 for b in opts1 if a == b]
        random.shuffle(diff)
        random.shuffle(same)
        pairs.extend(diff + same)
    return pairs


def _render_group_candidate(
    h: int,
    w: int,
    rope_color: int,
    obj0_color: int,
    obj1_color: int,
    orientation: str,
    need_overlap: bool,
    blocked_in: set[tuple[int, int]],
    blocked_out: set[tuple[int, int]],
) -> dict | None:
    for _ in range(GROUP_RETRIES):
        if orientation == "h":
            path_rel, line_rel, direction, wanted = _path_horizontal(need_overlap)
        elif orientation == "v":
            path_rel, line_rel, direction, wanted = _path_vertical(need_overlap)
        else:
            path_rel, line_rel, direction, wanted = _path_diagonal(need_overlap)

        if len(set(path_rel)) != len(path_rel):
            continue

        path_set = set(path_rel)
        line_set = set(line_rel)
        end0 = path_rel[0]
        end1 = path_rel[-1]
        start_step = (path_rel[1][0] - path_rel[0][0], path_rel[1][1] - path_rel[0][1])
        end_step = (path_rel[-2][0] - path_rel[-1][0], path_rel[-2][1] - path_rel[-1][1])
        offset_pairs = _choose_offsets(start_step, end_step, need_overlap)
        if not offset_pairs:
            continue

        for _ in range(OBJECT_RETRIES):
            off0, off1 = random.choice(offset_pairs)
            shape0 = _shape_for_offset(off0)
            shape1 = _shape_for_offset(off1)
            obj0_rel = {(end0[0] + off0[0] + r, end0[1] + off0[1] + c) for r, c in shape0}
            obj1_rel = {(end1[0] + off1[0] + r, end1[1] + off1[1] + c) for r, c in shape1}

            if path_set & obj0_rel or path_set & obj1_rel or obj0_rel & obj1_rel:
                continue
            if not _touches_exactly(obj0_rel, path_set, end0):
                continue
            if not _touches_exactly(obj1_rel, path_set, end1):
                continue

            actual = _actual_shift(obj0_rel, obj1_rel, direction, wanted)
            if actual <= 0:
                continue
            if need_overlap != (actual < wanted):
                continue

            moved0_rel = _translate(obj0_rel, actual * direction[0], actual * direction[1])
            moved1_rel = _translate(obj1_rel, -actual * direction[0], -actual * direction[1])
            if line_set & moved0_rel or line_set & moved1_rel or moved0_rel & moved1_rel:
                continue

            contact0 = (end0[0] + actual * direction[0], end0[1] + actual * direction[1])
            contact1 = (end1[0] - actual * direction[0], end1[1] - actual * direction[1])
            if not _touches_exactly(moved0_rel, line_set, contact0):
                continue
            if not _touches_exactly(moved1_rel, line_set, contact1):
                continue

            all_rel = path_set | obj0_rel | obj1_rel | line_set | moved0_rel | moved1_rel
            rows = [r for r, _ in all_rel]
            cols = [c for _, c in all_rel]
            rmin, rmax = min(rows), max(rows)
            cmin, cmax = min(cols), max(cols)
            row_lo = -rmin
            row_hi = h - 1 - rmax
            col_lo = -cmin
            col_hi = w - 1 - cmax
            if row_lo > row_hi or col_lo > col_hi:
                continue

            for _ in range(SHIFT_RETRIES):
                dr = random.randint(row_lo, row_hi)
                dc = random.randint(col_lo, col_hi)
                path_abs = _translate(path_set, dr, dc)
                line_abs = _translate(line_set, dr, dc)
                obj0_abs = _translate(obj0_rel, dr, dc)
                obj1_abs = _translate(obj1_rel, dr, dc)
                moved0_abs = _translate(moved0_rel, dr, dc)
                moved1_abs = _translate(moved1_rel, dr, dc)

                input_cells = path_abs | obj0_abs | obj1_abs
                output_cells = line_abs | moved0_abs | moved1_abs
                if not _in_bounds(input_cells | output_cells, h, w):
                    continue
                if input_cells & blocked_in or output_cells & blocked_out:
                    continue

                return {
                    "rope_color": rope_color,
                    "obj0_color": obj0_color,
                    "obj1_color": obj1_color,
                    "path": path_abs,
                    "line": line_abs,
                    "obj0": obj0_abs,
                    "obj1": obj1_abs,
                    "moved0": moved0_abs,
                    "moved1": moved1_abs,
                    "blocked_in": _halo(input_cells, h, w),
                    "blocked_out": _halo(output_cells, h, w),
                }
    return None


def _render_scene(h: int, w: int, groups: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for group in groups:
        for r, c in group["path"]:
            inp[r, c] = group["rope_color"]
        for r, c in group["obj0"]:
            inp[r, c] = group["obj0_color"]
        for r, c in group["obj1"]:
            inp[r, c] = group["obj1_color"]

        for r, c in group["line"]:
            out[r, c] = group["rope_color"]
        for r, c in group["moved0"]:
            out[r, c] = group["obj0_color"]
        for r, c in group["moved1"]:
            out[r, c] = group["obj1_color"]
    return inp, out


def _grid_size(rope_count: int) -> tuple[int, int]:
    if rope_count == 1:
        return random.randint(12, 22), random.randint(12, 22)
    if rope_count == 2:
        return random.randint(16, 26), random.randint(16, 26)
    return random.randint(22, 30), random.randint(22, 30)


def _rope_count() -> int:
    return random.choices([1, 2, 3], weights=[2, 5, 3], k=1)[0]


def _orientations(rope_count: int) -> list[str]:
    base = ["h", "v", "d"]
    if rope_count == 1:
        return [random.choice(base)]
    out = random.sample(base, k=min(rope_count, 3))
    while len(out) < rope_count:
        out.append(random.choice(base))
    random.shuffle(out)
    return out


def _overlap_targets(rope_count: int) -> list[bool]:
    if rope_count == 1:
        return [random.choice([False, True])]
    if rope_count == 2:
        out = [False, True]
        random.shuffle(out)
        return out
    out = [False, True, random.choice([False, True])]
    random.shuffle(out)
    return out


def _gen_attempt() -> np.ndarray | None:
    rope_count = _rope_count()
    h, w = _grid_size(rope_count)
    colors = random.sample(PALETTE, 3 * rope_count)
    orientations = _orientations(rope_count)
    overlap_targets = _overlap_targets(rope_count)

    groups: list[dict] = []
    blocked_in: set[tuple[int, int]] = set()
    blocked_out: set[tuple[int, int]] = set()

    for i in range(rope_count):
        group = _render_group_candidate(
            h=h,
            w=w,
            rope_color=colors[3 * i],
            obj0_color=colors[3 * i + 1],
            obj1_color=colors[3 * i + 2],
            orientation=orientations[i],
            need_overlap=overlap_targets[i],
            blocked_in=blocked_in,
            blocked_out=blocked_out,
        )
        if group is None:
            return None
        groups.append(group)
        blocked_in |= group["blocked_in"]
        blocked_out |= group["blocked_out"]

    inp, expected = _render_scene(h, w, groups)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def _fast_horizontal_input() -> np.ndarray | None:
    h = random.randint(12, 26)
    w = random.randint(16, 30)
    rope_color, obj0_color, obj1_color = random.sample(PALETTE, 3)

    r = random.randint(3, h - 5)
    c0 = random.randint(2, max(2, w // 4))
    c1 = random.randint(max(c0 + 8, w // 2), w - 3)
    pivot = random.randint(c0 + 2, c1 - 2)
    k = random.randint(1, min(3, h - r - 3))

    path = set(_join_segments(
        _segment((r, c0), (r, pivot)),
        _segment((r, pivot), (r + k, pivot)),
        _segment((r + k, pivot), (r + k, c1)),
        _segment((r + k, c1), (r, c1)),
    ))
    obj0 = {(r - 1, c0), (r - 2, c0)}
    obj1 = {(r - 1, c1), (r - 2, c1)}

    grid = np.zeros((h, w), dtype=int)
    for rr, cc in path:
        grid[rr, cc] = rope_color
    for rr, cc in obj0:
        grid[rr, cc] = obj0_color
    for rr, cc in obj1:
        grid[rr, cc] = obj1_color

    if not _identify_groups(grid):
        return None
    out = main(grid)
    if not isinstance(out, np.ndarray) or out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(80):
        grid = _fast_horizontal_input()
        if grid is not None:
            return grid
    for _ in range(60):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
