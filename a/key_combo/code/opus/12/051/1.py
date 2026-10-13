from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    next_id = 0
    for color in PALETTE:
        for coords in _components_of_color(grid, color):
            objects.append(
                {
                    "id": next_id,
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )
            next_id += 1
    return objects


def _extract_guides(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _components_of_color(grid, GREY)


def _ordered_segment(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        ordered = sorted(coords, key=lambda rc: rc[1])
        assert [c for _, c in ordered] == list(range(ordered[0][1], ordered[0][1] + len(ordered)))
        return ordered, (0, 1)
    assert len(cols) == 1
    ordered = sorted(coords, key=lambda rc: rc[0])
    assert [r for r, _ in ordered] == list(range(ordered[0][0], ordered[0][0] + len(ordered)))
    return ordered, (1, 0)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    guides = _extract_guides(g)
    n = len(objects)
    assert 3 <= n <= 5
    assert len(guides) == n

    colors = [obj["color"] for obj in objects]
    assert len(colors) == len(set(colors))
    areas = [obj["area"] for obj in objects]
    assert len(areas) == len(set(areas))
    assert all(3 <= area <= 12 for area in areas)

    cell_to_object: dict[tuple[int, int], int] = {}
    for obj in objects:
        for rc in obj["coords"]:
            cell_to_object[rc] = obj["id"]

    guide_info: list[dict] = []
    for coords in guides:
        assert 2 <= len(coords) <= 3
        ordered, axis = _ordered_segment(coords)
        endpoints = [ordered[0], ordered[-1]]
        attached: list[tuple[int, int]] = []
        for endpoint_index, endpoint in enumerate(endpoints):
            adjacent_objects = {
                cell_to_object[(nr, nc)]
                for nr, nc in _neighbors4(*endpoint)
                if (nr, nc) in cell_to_object
            }
            if adjacent_objects:
                assert len(adjacent_objects) == 1
                attached.append((endpoint_index, next(iter(adjacent_objects))))
        assert len(attached) == 1

        endpoint_index, source_id = attached[0]
        if endpoint_index == 0:
            direction = axis
            near = ordered[0]
            far = ordered[-1]
        else:
            direction = (-axis[0], -axis[1])
            near = ordered[-1]
            far = ordered[0]

        for nr, nc in _neighbors4(*near):
            if (nr, nc) in cell_to_object:
                assert cell_to_object[(nr, nc)] == source_id

        rr = far[0] + direction[0]
        cc = far[1] + direction[1]
        target_id = None
        while _in_bounds(g.shape, rr, cc):
            value = int(g[rr, cc])
            assert value != GREY, "guide rays must not cross another guide"
            if value != BG:
                target_id = cell_to_object[(rr, cc)]
                break
            rr += direction[0]
            cc += direction[1]
        assert target_id is not None
        assert target_id != source_id

        guide_info.append(
            {
                "source": source_id,
                "target": target_id,
                "direction": direction,
                "near": near,
                "far": far,
                "length": len(coords),
            }
        )

    mapping = {info["source"]: info["target"] for info in guide_info}
    assert len(mapping) == n
    assert len(set(mapping.values())) == n

    ordered_objects = sorted(objects, key=lambda obj: obj["area"], reverse=True)
    ring_colors = [next(obj["color"] for obj in objects if obj["id"] == mapping[obj_src["id"]]) for obj_src in ordered_objects]

    return {
        "objects": objects,
        "guides": guide_info,
        "mapping": mapping,
        "ordered_objects": ordered_objects,
        "ring_colors": ring_colors,
    }


def _rings_from_colors(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each object's guide ray to another object, then repaint size-ranked concentric rings with the routed colors.

    Input structure:
    - The grid has black background, 3 to 5 uniquely colored 4-connected objects, and one gray straight guide segment attached to each object.
    - Every guide segment is horizontal or vertical, has length 2 or 3, and points away from its source object.
    - Extending a guide past its far endpoint hits exactly one other colored object first, and the guide network is a permutation on the objects.

    Rule:
    - Sort the colored objects by decreasing area.
    - For each object, follow its gray guide direction as a ray and record the color of the first other object it hits.
    - Build a square output of side 2N-1 with N concentric square rings; the largest object's ring is outermost, the smallest object's ring is the center, and each ring uses the routed target color rather than the source object's own color.

    Key insight:
    - Object size decides which ring position an object controls, while the gray guide decides which other object's color is painted there, so both analyses are required.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    return _rings_from_colors(analysis["ring_colors"])


def _random_shortest_path(
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    q = deque([start])
    parents = {start: None}
    while q:
        cur = q.popleft()
        if cur in goals:
            path = [cur]
            while parents[path[-1]] is not None:
                path.append(parents[path[-1]])
            path.reverse()
            return path
        nbrs = list(_neighbors4(*cur))
        rng.shuffle(nbrs)
        for nxt in nbrs:
            if nxt in allowed and nxt not in parents:
                parents[nxt] = cur
                q.append(nxt)
    return None


def _base_connected_shape(
    required: list[tuple[int, int]],
    box: tuple[int, int, int, int],
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    allowed = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    req = list(dict.fromkeys(required))
    rng.shuffle(req)
    shape = {req[0]}
    for cell in req[1:]:
        if cell in shape:
            continue
        path = _random_shortest_path(cell, shape, allowed, rng)
        if path is None:
            return None
        shape.update(path)
    return shape


def _grow_shape(
    required: list[tuple[int, int]],
    box: tuple[int, int, int, int],
    area: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    shape = _base_connected_shape(required, box, rng)
    if shape is None or len(shape) > area:
        return None

    r0, c0, r1, c1 = box
    allowed = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    frontier = {
        nxt
        for cell in shape
        for nxt in _neighbors4(*cell)
        if nxt in allowed and nxt not in shape
    }
    while len(shape) < area:
        if not frontier:
            return None
        choice = rng.choice(tuple(frontier))
        frontier.remove(choice)
        shape.add(choice)
        for nxt in _neighbors4(*choice):
            if nxt in allowed and nxt not in shape:
                frontier.add(nxt)
    return shape


def _choose_distinct_areas(lower_bounds: list[int], rng: random.Random) -> list[int] | None:
    values = list(range(3, 13))
    order = list(range(len(lower_bounds)))
    rng.shuffle(order)
    order.sort(key=lambda idx: lower_bounds[idx], reverse=True)
    assigned = [0] * len(lower_bounds)
    used: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        candidates = [v for v in values if v not in used and v >= lower_bounds[idx]]
        rng.shuffle(candidates)
        for value in candidates:
            assigned[idx] = value
            used.add(value)
            if backtrack(pos + 1):
                return True
            used.remove(value)
        return False

    return assigned if backtrack(0) else None


def _maybe_transform(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, k=rng.randrange(4))
    return np.asarray(out, dtype=int)


def _build_cycle4(rng: random.Random) -> dict:
    s = rng.randint(4, 5)
    gap_h = rng.randint(5, 7)
    gap_v = rng.randint(5, 7)
    top = rng.randint(1, 2)
    left = rng.randint(1, 2)
    bottom = top + s + gap_v
    right = left + s + gap_h
    h = bottom + s + rng.randint(1, 2)
    w = right + s + rng.randint(1, 2)

    row_ab = rng.randint(top, top + s - 1)
    col_bc = rng.randint(right, right + s - 1)
    row_cd = rng.randint(bottom, bottom + s - 1)
    col_da = rng.randint(left, left + s - 1)

    boxes = [
        (top, left, top + s - 1, left + s - 1),
        (top, right, top + s - 1, right + s - 1),
        (bottom, right, bottom + s - 1, right + s - 1),
        (bottom, left, bottom + s - 1, left + s - 1),
    ]
    edges = [
        {"source": 0, "target": 1, "out": (row_ab, left + s - 1), "hit": (row_ab, right), "dir": (0, 1)},
        {"source": 1, "target": 2, "out": (top + s - 1, col_bc), "hit": (bottom, col_bc), "dir": (1, 0)},
        {"source": 2, "target": 3, "out": (row_cd, right), "hit": (row_cd, left + s - 1), "dir": (0, -1)},
        {"source": 3, "target": 0, "out": (bottom, col_da), "hit": (top + s - 1, col_da), "dir": (-1, 0)},
    ]
    return {"shape": (h, w), "boxes": boxes, "edges": edges}


def _build_pair4(rng: random.Random) -> dict:
    s = rng.randint(4, 5)
    gap_h = rng.randint(5, 7)
    gap_v = rng.randint(5, 7)
    top = rng.randint(1, 2)
    left = rng.randint(1, 2)
    bottom = top + s + gap_v
    right = left + s + gap_h
    h = bottom + s + rng.randint(1, 2)
    w = right + s + rng.randint(1, 2)

    top_rows = rng.sample(list(range(top, top + s)), 2)
    bottom_rows = rng.sample(list(range(bottom, bottom + s)), 2)
    row_ab, row_ba = top_rows[0], top_rows[1]
    row_cd, row_dc = bottom_rows[0], bottom_rows[1]

    boxes = [
        (top, left, top + s - 1, left + s - 1),
        (top, right, top + s - 1, right + s - 1),
        (bottom, left, bottom + s - 1, left + s - 1),
        (bottom, right, bottom + s - 1, right + s - 1),
    ]
    edges = [
        {"source": 0, "target": 1, "out": (row_ab, left + s - 1), "hit": (row_ab, right), "dir": (0, 1)},
        {"source": 1, "target": 0, "out": (row_ba, right), "hit": (row_ba, left + s - 1), "dir": (0, -1)},
        {"source": 2, "target": 3, "out": (row_cd, left + s - 1), "hit": (row_cd, right), "dir": (0, 1)},
        {"source": 3, "target": 2, "out": (row_dc, right), "hit": (row_dc, left + s - 1), "dir": (0, -1)},
    ]
    return {"shape": (h, w), "boxes": boxes, "edges": edges}


def _build_cycle5(rng: random.Random) -> dict:
    s = rng.randint(4, 5)
    mid_h = rng.randint(3, 4)
    gap_h = rng.randint(5, 7)
    gap_top = rng.randint(2, 4)
    gap_bottom = rng.randint(2, 4)
    top = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = left + s + gap_h
    mid = top + s + gap_top
    bottom = mid + mid_h + gap_bottom
    h = bottom + s + rng.randint(1, 2)
    w = right + s + rng.randint(1, 2)

    row_ab = rng.randint(top, top + s - 1)
    col_be = rng.randint(right, right + s - 1)
    col_ec = rng.randint(right, right + s - 1)
    col_da = rng.randint(left, left + s - 1)
    row_cd = rng.randint(bottom, bottom + s - 1)

    boxes = [
        (top, left, top + s - 1, left + s - 1),
        (top, right, top + s - 1, right + s - 1),
        (mid, right, mid + mid_h - 1, right + s - 1),
        (bottom, right, bottom + s - 1, right + s - 1),
        (bottom, left, bottom + s - 1, left + s - 1),
    ]
    edges = [
        {"source": 0, "target": 1, "out": (row_ab, left + s - 1), "hit": (row_ab, right), "dir": (0, 1)},
        {"source": 1, "target": 2, "out": (top + s - 1, col_be), "hit": (mid, col_be), "dir": (1, 0)},
        {"source": 2, "target": 3, "out": (mid + mid_h - 1, col_ec), "hit": (bottom, col_ec), "dir": (1, 0)},
        {"source": 3, "target": 4, "out": (row_cd, right), "hit": (row_cd, left + s - 1), "dir": (0, -1)},
        {"source": 4, "target": 0, "out": (bottom, col_da), "hit": (top + s - 1, col_da), "dir": (-1, 0)},
    ]
    return {"shape": (h, w), "boxes": boxes, "edges": edges}


def _build_cycle3(rng: random.Random) -> dict:
    right_s = rng.randint(4, 5)
    left_w = rng.randint(4, 5)
    gap_h = rng.randint(5, 7)
    gap_v = rng.randint(2, 4)
    top = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = left + left_w + gap_h
    bottom = top + right_s + gap_v
    tall_h = bottom + right_s - top
    h = top + tall_h + rng.randint(1, 2)
    w = right + right_s + rng.randint(1, 2)

    row_ab = rng.randint(top, top + right_s - 1)
    row_ca = rng.randint(bottom, bottom + right_s - 1)
    col_bc = rng.randint(right, right + right_s - 1)

    boxes = [
        (top, left, top + tall_h - 1, left + left_w - 1),
        (top, right, top + right_s - 1, right + right_s - 1),
        (bottom, right, bottom + right_s - 1, right + right_s - 1),
    ]
    edges = [
        {"source": 0, "target": 1, "out": (row_ab, left + left_w - 1), "hit": (row_ab, right), "dir": (0, 1)},
        {"source": 1, "target": 2, "out": (top + right_s - 1, col_bc), "hit": (bottom, col_bc), "dir": (1, 0)},
        {"source": 2, "target": 0, "out": (row_ca, right), "hit": (row_ca, left + left_w - 1), "dir": (0, -1)},
    ]
    return {"shape": (h, w), "boxes": boxes, "edges": edges}


def _build_layout(rng: random.Random) -> dict:
    n = rng.choice([3, 4, 5])
    if n == 3:
        return _build_cycle3(rng)
    if n == 5:
        return _build_cycle5(rng)
    return rng.choice([_build_cycle4, _build_pair4])(rng)


def _render_layout(layout: dict, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = layout["shape"]
    n = len(layout["boxes"])
    colors = rng.sample(PALETTE, n)
    guide_lengths = [rng.randint(2, 3) for _ in range(n)]

    required: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for edge in layout["edges"]:
        required[edge["source"]].append(edge["out"])
        required[edge["target"]].append(edge["hit"])

    lower_bounds = []
    for box, req in zip(layout["boxes"], required):
        base_rng = random.Random(rng.randrange(10**9))
        base = _base_connected_shape(req, box, base_rng)
        if base is None:
            return None
        lower_bounds.append(max(3, len(base)))

    areas = _choose_distinct_areas(lower_bounds, rng)
    if areas is None:
        return None

    object_cells: list[set[tuple[int, int]]] = []
    for box, req, area in zip(layout["boxes"], required, areas):
        shape = None
        for _ in range(80):
            shape = _grow_shape(req, box, area, rng)
            if shape is not None:
                break
        if shape is None:
            return None
        object_cells.append(shape)

    grid = np.full((h, w), BG, dtype=int)
    for color, cells in zip(colors, object_cells):
        for r, c in cells:
            grid[r, c] = color

    for obj_index, edge in enumerate(layout["edges"]):
        dr, dc = edge["dir"]
        guide_len = guide_lengths[obj_index]
        out_r, out_c = edge["out"]
        for step in range(1, guide_len + 1):
            rr = out_r + dr * step
            cc = out_c + dc * step
            if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != BG:
                return None
            grid[rr, cc] = GREY

    mapping = {edge["source"]: edge["target"] for edge in layout["edges"]}
    order = sorted(range(n), key=lambda idx: areas[idx], reverse=True)
    expected = _rings_from_colors([colors[mapping[idx]] for idx in order])
    return grid, expected


def _validate_non_degenerate(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        analysis = _analyze_grid(grid)
    except AssertionError:
        return False

    if not np.array_equal(main(grid), expected):
        return False

    ordered = analysis["ordered_objects"]
    own_colors = [obj["color"] for obj in ordered]
    if analysis["ring_colors"] == own_colors:
        return False

    row_major_ids = [obj["id"] for obj in sorted(analysis["objects"], key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))]
    area_ids = [obj["id"] for obj in ordered]
    if area_ids == row_major_ids or area_ids == row_major_ids[::-1]:
        return False

    row_major_colors = [next(obj["color"] for obj in analysis["objects"] if obj["id"] == idx) for idx in row_major_ids]
    if analysis["ring_colors"] == row_major_colors:
        return False

    directions = {tuple(info["direction"]) for info in analysis["guides"]}
    if len(directions) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        layout = _build_layout(rng)
        rendered = _render_layout(layout, rng)
        if rendered is None:
            continue
        grid, expected = rendered
        grid = _maybe_transform(grid, rng)
        if _validate_non_degenerate(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
