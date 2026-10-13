from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
OBSTACLE = 5
OBJECT_COLORS = (1, 2, 3, 4)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (14, 18)
GRID_WIDTH_RANGE = (14, 18)
OBSTACLE_HEIGHT_RANGE = (5, 7)
OBSTACLE_WIDTH_RANGE = (5, 7)
OBJECT_COUNT_CHOICES = (3, 4)
OBJECT_AREA_RANGE = (2, 5)
PLACEMENT_TRIALS = 180
SCENE_MARGIN = 3
LOCAL_MAX_DEPTH = 3
LOCAL_MAX_SPAN = 2


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _collect_components(g: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(g.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords.sort()
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "area": len(coords),
                    "centroid": (
                        sum(rows) / len(rows),
                        sum(cols) / len(cols),
                    ),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _extract_boundary(obstacle_mask: np.ndarray) -> dict:
    h, w = obstacle_mask.shape
    raw_edges: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not obstacle_mask[r, c]:
                continue
            if r == 0 or not obstacle_mask[r - 1, c]:
                raw_edges.append(
                    {
                        "start": (r, c),
                        "end": (r, c + 1),
                        "cell": (r, c),
                        "side": (-1, 0),
                    }
                )
            if c == w - 1 or not obstacle_mask[r, c + 1]:
                raw_edges.append(
                    {
                        "start": (r, c + 1),
                        "end": (r + 1, c + 1),
                        "cell": (r, c),
                        "side": (0, 1),
                    }
                )
            if r == h - 1 or not obstacle_mask[r + 1, c]:
                raw_edges.append(
                    {
                        "start": (r + 1, c + 1),
                        "end": (r + 1, c),
                        "cell": (r, c),
                        "side": (1, 0),
                    }
                )
            if c == 0 or not obstacle_mask[r, c - 1]:
                raw_edges.append(
                    {
                        "start": (r + 1, c),
                        "end": (r, c),
                        "cell": (r, c),
                        "side": (0, -1),
                    }
                )

    assert raw_edges, "expected a non-empty obstacle"
    start_to_idx: dict[Coord, int] = {}
    for i, edge in enumerate(raw_edges):
        key = edge["start"]
        assert key not in start_to_idx, "boundary must be a simple directed cycle"
        start_to_idx[key] = i

    ordered_idx: list[int] = []
    cur = start_to_idx[min(start_to_idx)]
    seen: set[int] = set()
    while cur not in seen:
        seen.add(cur)
        ordered_idx.append(cur)
        nxt = raw_edges[cur]["end"]
        assert nxt in start_to_idx, "boundary chain must close"
        cur = start_to_idx[nxt]
    assert cur == ordered_idx[0], "boundary must return to its start"
    assert len(ordered_idx) == len(raw_edges), "expected exactly one outer boundary cycle"

    edges = []
    for raw_i in ordered_idx:
        edge = dict(raw_edges[raw_i])
        cell_r, cell_c = edge["cell"]
        side_r, side_c = edge["side"]
        edge["outside"] = (cell_r + side_r, cell_c + side_c)
        edges.append(edge)

    groups: list[list[int]] = []
    current = [0]
    for i in range(1, len(edges)):
        if edges[i]["cell"] == edges[i - 1]["cell"]:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    if len(groups) > 1 and edges[groups[0][0]]["cell"] == edges[groups[-1][0]]["cell"]:
        groups[0] = groups[-1] + groups[0]
        groups.pop()

    edge_to_group: dict[int, int] = {}
    edge_lookup: dict[tuple[int, int, int, int], int] = {}
    for group_i, group in enumerate(groups):
        for edge_i in group:
            edge_to_group[edge_i] = group_i
            cell_r, cell_c = edges[edge_i]["cell"]
            side_r, side_c = edges[edge_i]["side"]
            edge_lookup[(cell_r, cell_c, side_r, side_c)] = edge_i

    return {
        "edges": edges,
        "groups": groups,
        "edge_to_group": edge_to_group,
        "edge_lookup": edge_lookup,
        "group_count": len(groups),
    }


def _advance_edge(boundary: dict, start_edge: int, steps: int) -> int:
    if steps == 0:
        return start_edge
    group_i = boundary["edge_to_group"][start_edge]
    dest_group = (group_i + steps) % boundary["group_count"]
    return boundary["groups"][dest_group][0]


def _path_cells(boundary: dict, start_edge: int, steps: int) -> list[Coord]:
    cells = [boundary["edges"][start_edge]["cell"]]
    group_i = boundary["edge_to_group"][start_edge]
    for delta in range(1, steps + 1):
        edge_i = boundary["groups"][(group_i + delta) % boundary["group_count"]][0]
        cells.append(boundary["edges"][edge_i]["cell"])
    return cells


def _path_turns(boundary: dict, start_edge: int, steps: int) -> bool:
    cells = _path_cells(boundary, start_edge, steps)
    dirs: list[Coord] = []
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
        dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
        dirs.append((dr, dc))
    cleaned = [d for d in dirs if d != (0, 0)]
    return any(a != b for a, b in zip(cleaned, cleaned[1:]))


def _translated_coords(coords: list[Coord], anchor: Coord, dest_anchor: Coord) -> list[Coord]:
    dr = dest_anchor[0] - anchor[0]
    dc = dest_anchor[1] - anchor[1]
    return [(r + dr, c + dc) for r, c in coords]


def _identify_objects(g: np.ndarray, boundary: dict) -> list[dict]:
    obstacle_mask = g == OBSTACLE
    objects = _collect_components(g, set(OBJECT_COLORS))
    assert 3 <= len(objects) <= 4, "expected three or four colored objects"

    enriched: list[dict] = []
    for obj in objects:
        centroid_r, centroid_c = obj["centroid"]
        adjacency_pairs: list[tuple[Coord, Coord]] = []
        by_obstacle: dict[Coord, list[Coord]] = {}
        for pixel in obj["coords"]:
            r, c = pixel
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(g.shape, nr, nc) and obstacle_mask[nr, nc]:
                    obstacle_cell = (nr, nc)
                    adjacency_pairs.append((pixel, obstacle_cell))
                    by_obstacle.setdefault(obstacle_cell, []).append(pixel)

        assert adjacency_pairs, "every object must touch the obstacle"

        obstacle_candidates = sorted(
            by_obstacle,
            key=lambda pos: (
                (pos[0] - centroid_r) ** 2 + (pos[1] - centroid_c) ** 2,
                pos[0],
                pos[1],
            ),
        )
        entry_cell = obstacle_candidates[0]
        anchor_candidates = by_obstacle[entry_cell]
        anchor = min(
            anchor_candidates,
            key=lambda pos: (
                (pos[0] - centroid_r) ** 2 + (pos[1] - centroid_c) ** 2,
                pos[0],
                pos[1],
            ),
        )
        side = (anchor[0] - entry_cell[0], anchor[1] - entry_cell[1])
        assert side in DIR4, "anchor must be 4-neighbor-adjacent to the obstacle"
        edge_key = (entry_cell[0], entry_cell[1], side[0], side[1])
        assert edge_key in boundary["edge_lookup"], "entry side must lie on the obstacle exterior"

        meta = dict(obj)
        meta["entry_cell"] = entry_cell
        meta["anchor"] = anchor
        meta["start_edge"] = boundary["edge_lookup"][edge_key]
        meta["side"] = side
        enriched.append(meta)

    enriched.sort(key=lambda obj: (obj["area"], obj["centroid"][0], obj["centroid"][1]))
    return enriched


def _fits(coords: list[Coord], occupied: np.ndarray) -> bool:
    for r, c in coords:
        if not _in_bounds(occupied.shape, r, c) or occupied[r, c]:
            return False
    return True


def _apply_rule(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, OBSTACLE, *OBJECT_COLORS}

    obstacle_components = _collect_components(g, {OBSTACLE})
    assert len(obstacle_components) == 1, "expected one connected gray obstacle"
    obstacle_mask = g == OBSTACLE
    boundary = _extract_boundary(obstacle_mask)
    assert boundary["group_count"] > OBJECT_AREA_RANGE[1], "contour is too short for the required walk"

    objects = _identify_objects(g, boundary)

    out = np.full_like(g, BG)
    out[obstacle_mask] = OBSTACLE
    occupied = obstacle_mask.copy()
    placements: list[dict] = []

    for obj in objects:
        chosen: dict | None = None
        for steps in range(obj["area"], -1, -1):
            dest_edge = _advance_edge(boundary, obj["start_edge"], steps)
            dest_anchor = boundary["edges"][dest_edge]["outside"]
            candidate = _translated_coords(obj["coords"], obj["anchor"], dest_anchor)
            if _fits(candidate, occupied):
                chosen = {
                    "color": obj["color"],
                    "coords": candidate,
                    "coord_set": set(candidate),
                    "used_steps": steps,
                    "requested_steps": obj["area"],
                    "start_edge": obj["start_edge"],
                    "dest_edge": dest_edge,
                    "anchor": obj["anchor"],
                    "dest_anchor": dest_anchor,
                    "entry_cell": obj["entry_cell"],
                    "centroid": obj["centroid"],
                    "area": obj["area"],
                    "path_turns": _path_turns(boundary, obj["start_edge"], obj["area"]),
                }
                break
        assert chosen is not None, "generator assumptions require a valid fallback placement"
        for r, c in chosen["coords"]:
            occupied[r, c] = True
            out[r, c] = obj["color"]
        placements.append(chosen)

    return out, placements, boundary


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored object clockwise along the gray obstacle contour by a distance equal to its area.

    Input structure:
    - A black background contains one connected gray obstacle near the center and three or four small 4-connected colored objects around it.
    - Each colored object is monochrome, uses a distinct color from blue/red/green/yellow, and touches the obstacle from one clear exterior side.
    - The obstacle is holeless but irregular, so its exterior contour can bend around corners and repeated contour passes of the same gray cell are possible.

    Rule:
    - For each colored object, count its pixels to get area A, choose the adjacent gray entry cell closest to the object's centroid, and use the touching object pixel on that side as the anchor.
    - Treat every non-gray cell as exterior when tracing the obstacle boundary; starting from the anchor side, move clockwise to the next contour cell A times.
    - Translate the whole object without rotation so that its anchor lands on the exterior side of the destination contour cell, processing objects in increasing area order and backing up one contour step at a time if a move would overlap an already-placed object.

    Key insight:
    - The answer depends on combining two independent signals: the object's area gives the travel distance, while the obstacle's curved exterior contour determines where that distance lands and which exterior side the object must occupy.
    """
    out, _placements, _boundary = _apply_rule(grid)
    return out


def _generate_local_shape(area: int) -> list[tuple[int, int]]:
    assert 2 <= area <= 5
    for _ in range(100):
        cells = {(0, 0), (1, 0)}
        while len(cells) < area:
            frontier = []
            for u, v in sorted(cells):
                for du, dv in DIR4:
                    nu = u + du
                    nv = v + dv
                    if (nu, nv) in cells:
                        continue
                    if not (0 <= nu <= LOCAL_MAX_DEPTH and abs(nv) <= LOCAL_MAX_SPAN):
                        continue
                    if nu == 0 and nv != 0:
                        continue
                    if nu == 0 and (nu, nv) != (0, 0):
                        continue
                    frontier.append((nu, nv))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        if random.random() < 0.5:
            cells = {(u, -v) for u, v in cells}
        return sorted(cells)
    raise AssertionError("failed to generate a small outward-growing shape")


def _side_basis(side: Coord) -> tuple[Coord, Coord]:
    outward = side
    tangent = (0, 1) if side in {(-1, 0), (1, 0)} else (1, 0)
    return outward, tangent


def _placement_from_local(local: list[tuple[int, int]], edge: dict) -> list[Coord]:
    anchor_r, anchor_c = edge["outside"]
    outward, tangent = _side_basis(edge["side"])
    coords = []
    for u, v in local:
        coords.append(
            (
                anchor_r + u * outward[0] + v * tangent[0],
                anchor_c + u * outward[1] + v * tangent[1],
            )
        )
    return coords


def _adjacency_signature(coords: list[Coord], obstacle_mask: np.ndarray) -> list[tuple[Coord, Coord]]:
    pairs: list[tuple[Coord, Coord]] = []
    for pixel in coords:
        r, c = pixel
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(obstacle_mask.shape, nr, nc) and obstacle_mask[nr, nc]:
                pairs.append((pixel, (nr, nc)))
    return pairs


def _feasible_placements(local: list[tuple[int, int]], boundary: dict, obstacle_mask: np.ndarray) -> list[dict]:
    placements = []
    area = len(local)
    for start_edge, edge in enumerate(boundary["edges"]):
        coords = _placement_from_local(local, edge)
        coord_set = set(coords)
        if len(coord_set) != area:
            continue
        if not _fits(coords, obstacle_mask):
            continue
        pairs = _adjacency_signature(coords, obstacle_mask)
        if pairs != [(edge["outside"], edge["cell"])]:
            continue
        centroid_r = sum(r for r, _ in coords) / area
        centroid_c = sum(c for _, c in coords) / area
        dest_edge = _advance_edge(boundary, start_edge, area)
        dest_anchor = boundary["edges"][dest_edge]["outside"]
        naive_coords = _translated_coords(coords, edge["outside"], dest_anchor)
        placements.append(
            {
                "start_edge": start_edge,
                "coords": coords,
                "coord_set": coord_set,
                "centroid": (centroid_r, centroid_c),
                "naive_coords": naive_coords,
                "naive_set": set(naive_coords),
                "path_turns": _path_turns(boundary, start_edge, area),
                "side": edge["side"],
            }
        )
    return placements


def _objects_are_clear(a: dict, b: dict) -> bool:
    if a["coord_set"] & b["coord_set"]:
        return False
    for r1, c1 in a["coord_set"]:
        for r2, c2 in b["coord_set"]:
            if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                return False
    return True


def _generate_obstacle_mask(h: int, w: int) -> np.ndarray:
    for _ in range(200):
        intervals: list[tuple[int, int]] = []
        left = random.randint(0, max(0, w - 4))
        right = random.randint(left + 2, w - 1)
        changed = False
        for _row in range(h):
            if intervals:
                candidates = []
                prev_left, prev_right = intervals[-1]
                for dl in (-1, 0, 1):
                    for dr in (-1, 0, 1):
                        cur_left = prev_left + dl
                        cur_right = prev_right + dr
                        if not (0 <= cur_left < cur_right < w):
                            continue
                        if cur_right - cur_left + 1 < 3:
                            continue
                        if cur_left > prev_right or cur_right < prev_left:
                            continue
                        candidates.append((cur_left, cur_right))
                if not candidates:
                    break
                left, right = random.choice(candidates)
                if (left, right) != intervals[-1]:
                    changed = True
            intervals.append((left, right))
        if len(intervals) != h:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, (c0, c1) in enumerate(intervals):
            mask[r, c0 : c1 + 1] = True
        area = int(mask.sum())
        if not (18 <= area <= 32):
            continue
        if not changed:
            continue
        return mask
    raise AssertionError("failed to build an irregular obstacle")


def _semantic_ok(grid: np.ndarray, placements: list[dict], boundary: dict, want_collision: bool, want_equal_area: bool) -> bool:
    raw_objects = _identify_objects(grid, boundary)
    areas = [obj["area"] for obj in raw_objects]
    if len(set(obj["side"] for obj in raw_objects)) < 2:
        return False
    if not any(_path_turns(boundary, obj["start_edge"], obj["area"]) for obj in raw_objects):
        return False
    if max(areas) >= boundary["group_count"]:
        return False
    if want_collision and not any(p["used_steps"] < p["requested_steps"] for p in placements):
        return False
    if want_equal_area:
        found = False
        for left, right in zip(raw_objects, raw_objects[1:]):
            if left["area"] == right["area"]:
                found = True
                break
        if not found:
            return False
    return True


def _build_scene() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    obstacle_h = random.randint(*OBSTACLE_HEIGHT_RANGE)
    obstacle_w = random.randint(*OBSTACLE_WIDTH_RANGE)
    if h < obstacle_h + 2 * SCENE_MARGIN or w < obstacle_w + 2 * SCENE_MARGIN:
        return None

    obstacle_patch = _generate_obstacle_mask(obstacle_h, obstacle_w)
    grid = np.full((h, w), BG, dtype=int)
    center_r = (h - obstacle_h) // 2
    center_c = (w - obstacle_w) // 2
    r0_lo = max(SCENE_MARGIN, center_r - 1)
    r0_hi = min(h - obstacle_h - SCENE_MARGIN, center_r + 1)
    c0_lo = max(SCENE_MARGIN, center_c - 1)
    c0_hi = min(w - obstacle_w - SCENE_MARGIN, center_c + 1)
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    top = random.randint(r0_lo, r0_hi)
    left = random.randint(c0_lo, c0_hi)
    grid[top : top + obstacle_h, left : left + obstacle_w][obstacle_patch] = OBSTACLE

    obstacle_mask = grid == OBSTACLE
    boundary = _extract_boundary(obstacle_mask)
    if boundary["group_count"] <= OBJECT_AREA_RANGE[1]:
        return None

    want_collision = random.random() < 0.55
    want_equal_area = random.random() < 0.55
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(list(OBJECT_COLORS), object_count)

    areas = [random.randint(*OBJECT_AREA_RANGE) for _ in range(object_count)]
    if want_equal_area:
        shared = random.randint(*OBJECT_AREA_RANGE)
        areas[0] = shared
        areas[1] = shared
    if want_collision:
        areas[0] = random.randint(3, 5)
        if want_equal_area:
            areas[1] = areas[0]
        else:
            areas[1] = random.randint(2, 5)

    locals_by_object = [_generate_local_shape(area) for area in areas]
    feasible = [_feasible_placements(local, boundary, obstacle_mask) for local in locals_by_object]
    if any(not options for options in feasible):
        return None

    collision_pair: list[tuple[dict, dict]] | None = None
    if want_collision:
        collision_pair = []
        for left_place in feasible[0]:
            for right_place in feasible[1]:
                if not _objects_are_clear(left_place, right_place):
                    continue
                if left_place["naive_set"] & right_place["naive_set"]:
                    collision_pair.append((left_place, right_place))
        if not collision_pair:
            return None

    for _ in range(PLACEMENT_TRIALS):
        selected: list[dict | None] = [None] * object_count
        chosen: list[dict] = []
        used = 0

        if collision_pair:
            pair_left, pair_right = random.choice(collision_pair)
            selected[0] = pair_left
            selected[1] = pair_right
            chosen.extend([pair_left, pair_right])
            used = 2

        order = list(range(used, object_count))
        order.sort(key=lambda idx: len(feasible[idx]))
        success = True
        for idx in order:
            options = feasible[idx][:]
            random.shuffle(options)
            pick = None
            for option in options:
                if all(_objects_are_clear(option, other) for other in chosen):
                    pick = option
                    break
            if pick is None:
                success = False
                break
            selected[idx] = pick
            chosen.append(pick)
        if not success:
            continue

        scene = grid.copy()
        for idx, place in enumerate(selected):
            assert place is not None
            for r, c in place["coords"]:
                scene[r, c] = colors[idx]

        try:
            out, placements, derived_boundary = _apply_rule(scene)
        except AssertionError:
            continue
        if not _semantic_ok(scene, placements, derived_boundary, want_collision, want_equal_area):
            continue
        return scene

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid ARC input")
