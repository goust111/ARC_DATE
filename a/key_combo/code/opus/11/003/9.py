from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
DIRECTIONS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

PATH_COLORS = (1, 3, 7)
MARKER_COLORS = (4, 6, 8)
FRAME_COLORS = (2, 5, 9)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: Iterable[Coord]) -> BBox:
    pts = list(coords)
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_perimeter_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_rectangular_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    return set(comp["coords"]) == _rect_perimeter_cells(comp["bbox"])


def _neighbors_in(cell: Coord, cells: set[Coord]) -> list[Coord]:
    return [nbr for nbr in _neighbors4(*cell) if nbr in cells]


def _path_endpoints(cells: set[Coord]) -> tuple[Coord, Coord]:
    degrees = {cell: len(_neighbors_in(cell, cells)) for cell in cells}
    if not all(deg in (1, 2) for deg in degrees.values()):
        raise AssertionError("path must be an unbranched chain")
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        raise AssertionError("path must have exactly two endpoints")
    return endpoints[0], endpoints[1]


def _ordered_path(cells: set[Coord], start: Coord) -> list[Coord]:
    path = [start]
    prev: Coord | None = None
    cur = start
    while True:
        nxt = [nbr for nbr in _neighbors_in(cur, cells) if nbr != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            raise AssertionError("path branches while ordering")
        prev, cur = cur, nxt[0]
        path.append(cur)
    if len(path) != len(cells):
        raise AssertionError("ordered path did not cover the chain")
    return path


def _classify_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict], list[dict]]:
    bg = _get_bg(grid)
    comps = _find_components(grid, bg)
    frames: list[dict] = []
    paths: list[dict] = []
    markers: list[dict] = []
    for comp in comps:
        if comp["area"] == 1:
            markers.append(comp)
            continue
        if _is_rectangular_frame(comp):
            frames.append(comp)
            continue
        cells = set(comp["coords"])
        _path_endpoints(cells)
        paths.append(comp)
    if len(paths) != len(markers):
        raise AssertionError("expected one marker per path")
    return bg, frames, paths, markers


def _visible_endpoint_candidates(
    grid: np.ndarray,
    bg: int,
    marker: Coord,
    coord_to_path: dict[Coord, int],
    endpoints: list[tuple[Coord, Coord]],
) -> dict[int, tuple[int, Coord, tuple[Coord, ...]]]:
    h, w = grid.shape
    mr, mc = marker
    candidates: dict[int, tuple[int, Coord, tuple[Coord, ...]]] = {}
    for dr, dc in DIRECTIONS:
        gap: list[Coord] = []
        r = mr + dr
        c = mc + dc
        while 0 <= r < h and 0 <= c < w:
            if int(grid[r, c]) != bg:
                hit = (r, c)
                path_index = coord_to_path.get(hit)
                if path_index is not None and hit in endpoints[path_index]:
                    if path_index in candidates:
                        raise AssertionError("marker sees the same path twice")
                    candidates[path_index] = (path_index, hit, tuple(gap))
                break
            gap.append((r, c))
            r += dr
            c += dc
    if not candidates:
        raise AssertionError("marker sees no path endpoint")
    return candidates


def _unique_matching(
    candidate_maps: list[dict[int, tuple[int, Coord, tuple[Coord, ...]]]],
    path_count: int,
) -> list[tuple[int, Coord, tuple[Coord, ...]]]:
    solutions: list[tuple[tuple[int, Coord, tuple[Coord, ...]], ...]] = []

    def backtrack(i: int, used: set[int], cur: list[tuple[int, Coord, tuple[Coord, ...]]]) -> None:
        if i == len(candidate_maps):
            if len(used) == path_count:
                solutions.append(tuple(cur))
            return
        for path_index, cand in candidate_maps[i].items():
            if path_index in used:
                continue
            used.add(path_index)
            cur.append(cand)
            backtrack(i + 1, used, cur)
            cur.pop()
            used.remove(path_index)

    backtrack(0, set(), [])
    if len(solutions) != 1:
        raise AssertionError("marker-to-path assignment must be unique")
    return list(solutions[0])


def _innermost_frame_color(marker: Coord, frames: list[dict]) -> int | None:
    mr, mc = marker
    containing = [
        frame
        for frame in frames
        if frame["bbox"][0] < mr < frame["bbox"][2] and frame["bbox"][1] < mc < frame["bbox"][3]
    ]
    if not containing:
        return None
    best = min(containing, key=lambda frame: _bbox_area(frame["bbox"]))
    return int(best["color"])


def _analyze(grid: np.ndarray) -> dict:
    bg, frames, paths, markers = _classify_objects(np.asarray(grid, dtype=int))
    path_cells = [set(path["coords"]) for path in paths]
    endpoints = [_path_endpoints(cells) for cells in path_cells]
    coord_to_path: dict[Coord, int] = {}
    for i, cells in enumerate(path_cells):
        for cell in cells:
            coord_to_path[cell] = i
    marker_coords = [marker["coords"][0] for marker in markers]
    candidate_maps = [
        _visible_endpoint_candidates(grid, bg, marker, coord_to_path, endpoints)
        for marker in marker_coords
    ]
    assignment = _unique_matching(candidate_maps, len(paths))
    return {
        "bg": bg,
        "frames": frames,
        "paths": paths,
        "markers": markers,
        "path_cells": path_cells,
        "endpoints": endpoints,
        "marker_coords": marker_coords,
        "assignment": assignment,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match markers to visible path endpoints, extend the paths to the markers, then recolor each far tail by the marker's containing frame color.

    Input structure
    - The grid has black background, 2-3 single-pixel markers, 2-3 monochrome one-cell-wide orthogonal path objects, and 1-2 groups of rectangular frame perimeters.
    - Each path is a 4-connected unbranched chain with exactly two endpoints, and each marker has a clear horizontal or vertical background-only sightline to one path endpoint.
    - Some markers lie strictly inside one or more nested rectangular frames, while at least one marker lies outside all frames.

    Rule
    - Identify the unique one-to-one marker-to-path matching induced by visible endpoint sightlines.
    - For each matched pair, fill the background gap from the path endpoint toward the marker with the path color, leaving the marker pixel unchanged; let the gap length be G.
    - Recolor the first G cells of the same path starting from its opposite endpoint, using the color of the innermost frame that strictly contains the marker, or the marker's own color if no frame contains it.
    - Keep all rectangular frames unchanged.

    Key insight
    - The marker supplies distance through its sightline gap, but its transmitted recolor is indirect: nested frame containment overrides the marker's visible color.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    info = _analyze(g)
    out = g.copy()

    for marker_comp, (path_index, hit_endpoint, gap) in zip(info["markers"], info["assignment"]):
        path = info["paths"][path_index]
        path_color = int(path["color"])
        marker_color = int(marker_comp["color"])
        marker = marker_comp["coords"][0]
        recolor = _innermost_frame_color(marker, info["frames"])
        if recolor is None:
            recolor = marker_color

        for cell in gap:
            if int(g[cell]) != info["bg"]:
                raise AssertionError("sightline gap must be background")
            out[cell] = path_color

        a, b = info["endpoints"][path_index]
        other = b if hit_endpoint == a else a
        ordered = _ordered_path(info["path_cells"][path_index], other)
        for cell in ordered[: min(len(gap), len(ordered))]:
            out[cell] = recolor

    return out


def _paint_cells(grid: np.ndarray, cells: Iterable[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _zone_boxes(h: int, w: int) -> list[BBox]:
    rm = h // 2
    cm = w // 2
    return [
        (1, 1, rm - 2, cm - 2),
        (1, cm + 1, rm - 2, w - 2),
        (rm + 1, 1, h - 2, cm - 2),
        (rm + 1, cm + 1, h - 2, w - 2),
    ]


def _random_frame_group(zone: BBox, nested: bool, colors: tuple[int, ...]) -> dict | None:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    min_outer_h = 7
    min_outer_w = 7
    if zh < min_outer_h or zw < min_outer_w:
        return None
    outer_h = random.randint(min_outer_h, zh)
    outer_w = random.randint(min_outer_w, zw)
    outer_r0 = random.randint(zr0, zr1 - outer_h + 1)
    outer_c0 = random.randint(zc0, zc1 - outer_w + 1)
    outer = {
        "bbox": (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1),
        "color": int(colors[0]),
    }

    layers = [outer]
    usable = outer["bbox"]
    if nested:
        if outer_h < 8 or outer_w < 8:
            return None
        max_mt = min(2, outer_h - 6)
        max_mb = min(2, outer_h - 6)
        max_ml = min(2, outer_w - 6)
        max_mr = min(2, outer_w - 6)
        mt = random.randint(1, max_mt)
        mb = random.randint(1, max_mb)
        ml = random.randint(1, max_ml)
        mr = random.randint(1, max_mr)
        inner_h = outer_h - mt - mb
        inner_w = outer_w - ml - mr
        if inner_h < 6 or inner_w < 6:
            return None
        inner = {
            "bbox": (outer_r0 + mt, outer_c0 + ml, outer_r0 + mt + inner_h - 1, outer_c0 + ml + inner_w - 1),
            "color": int(colors[1]),
        }
        layers.append(inner)
        usable = inner["bbox"]

    ir0, ic0, ir1, ic1 = usable
    if ir1 - ir0 + 1 < 4 or ic1 - ic0 + 1 < 4:
        return None
    return {"layers": layers, "inside_region": (ir0 + 1, ic0 + 1, ir1 - 1, ic1 - 1)}


def _random_chain_local(height: int, width: int, length: int) -> list[Coord] | None:
    starts = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(starts)
    for start in starts[: min(len(starts), 20)]:
        path = [start]
        visited = {start}

        def dfs() -> list[Coord] | None:
            if len(path) == length:
                return list(path)
            cur = path[-1]
            options = []
            for nr, nc in _neighbors4(*cur):
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if (nr, nc) in visited:
                    continue
                adj = [cell for cell in _neighbors4(nr, nc) if cell in visited]
                if set(adj) <= {cur}:
                    options.append((nr, nc))
            random.shuffle(options)
            for nxt in options:
                path.append(nxt)
                visited.add(nxt)
                done = dfs()
                if done is not None:
                    return done
                visited.remove(nxt)
                path.pop()
            return None

        result = dfs()
        if result is not None:
            return result
    return None


def _has_nonframe_neighbor(cell: Coord, occupied_pairs: set[Coord]) -> bool:
    return any(nbr in occupied_pairs for nbr in _neighbors4(*cell))


def _marker_is_isolated(marker: Coord, path_cells: set[Coord], occupied_pairs: set[Coord], frame_cells: set[Coord]) -> bool:
    for nbr in _neighbors4(*marker):
        if nbr in path_cells or nbr in occupied_pairs or nbr in frame_cells:
            return False
    return True


def _gap_options(
    region: BBox,
    path: list[Coord],
    visible_index: int,
    occupied_pairs: set[Coord],
    frame_cells: set[Coord],
) -> list[tuple[tuple[Coord, ...], Coord]]:
    r0, c0, r1, c1 = region
    path_cells = set(path)
    endpoint = path[visible_index]
    options: list[tuple[tuple[Coord, ...], Coord]] = []
    for dr, dc in DIRECTIONS:
        limit = 0
        rr, cc = endpoint
        while True:
            rr += dr
            cc += dc
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                break
            limit += 1
        path_len = len(path)
        for gap_len in range(1, min(limit - 1, path_len) + 1):
            gap = tuple((endpoint[0] + dr * step, endpoint[1] + dc * step) for step in range(1, gap_len + 1))
            marker = (endpoint[0] + dr * (gap_len + 1), endpoint[1] + dc * (gap_len + 1))
            blocked = False
            for i, cell in enumerate(gap):
                if cell in path_cells or cell in occupied_pairs or cell in frame_cells:
                    blocked = True
                    break
                path_neighbors = [nbr for nbr in _neighbors4(*cell) if nbr in path_cells]
                if i == 0:
                    if set(path_neighbors) != {endpoint}:
                        blocked = True
                        break
                elif path_neighbors:
                    blocked = True
                    break
            if blocked or marker in path_cells or marker in occupied_pairs or marker in frame_cells:
                continue
            if not _marker_is_isolated(marker, path_cells, occupied_pairs, frame_cells):
                continue
            options.append((gap, marker))
    return options


def _generate_pair(
    region: BBox,
    path_color: int,
    marker_color: int,
    occupied_pairs: set[Coord],
    frame_cells: set[Coord],
    length_bounds: tuple[int, int],
) -> dict | None:
    r0, c0, r1, c1 = region
    if r1 < r0 or c1 < c0:
        return None
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    min_len, max_len = length_bounds
    max_len = min(max_len, height * width - 2)
    if max_len < min_len:
        return None

    for _ in range(250):
        length = random.randint(min_len, max_len)
        local = _random_chain_local(height, width, length)
        if local is None:
            continue
        path = [(r0 + rr, c0 + cc) for rr, cc in local]
        path_cells = set(path)
        if any(cell in frame_cells or cell in occupied_pairs for cell in path_cells):
            continue
        if any(_has_nonframe_neighbor(cell, occupied_pairs) for cell in path_cells):
            continue

        endpoints = (0, len(path) - 1)
        visible_index = random.choice(endpoints)
        options = _gap_options(region, path, visible_index, occupied_pairs, frame_cells)
        if not options:
            continue
        gap, marker = random.choice(options)
        return {
            "path": path,
            "path_color": int(path_color),
            "marker": marker,
            "marker_color": int(marker_color),
            "gap": gap,
            "visible_index": int(visible_index),
        }
    return None


def _spec_output(input_grid: np.ndarray, pair_specs: list[dict], frame_layers: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    for spec in pair_specs:
        path_color = int(spec["path_color"])
        marker = spec["marker"]
        marker_color = int(spec["marker_color"])
        recolor = None
        for frame in frame_layers:
            r0, c0, r1, c1 = frame["bbox"]
            if r0 < marker[0] < r1 and c0 < marker[1] < c1:
                if recolor is None or _bbox_area(frame["bbox"]) < _bbox_area(recolor["bbox"]):
                    recolor = frame
        tail_color = int(recolor["color"]) if recolor is not None else marker_color
        for cell in spec["gap"]:
            out[cell] = path_color
        other_to_visible = list(reversed(spec["path"])) if spec["visible_index"] == 0 else list(spec["path"])
        far_tail = other_to_visible[: len(spec["gap"])]
        for cell in far_tail:
            out[cell] = tail_color
    return out


def _build_input(h: int, w: int, frame_layers: list[dict], pair_specs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for frame in frame_layers:
        _paint_cells(grid, _rect_perimeter_cells(frame["bbox"]), int(frame["color"]))
    for spec in pair_specs:
        _paint_cells(grid, spec["path"], int(spec["path_color"]))
        mr, mc = spec["marker"]
        grid[mr, mc] = int(spec["marker_color"])
    return grid


def _validate_instance(grid: np.ndarray, pair_specs: list[dict], frame_layers: list[dict]) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    info = _analyze(grid)
    assert len(info["paths"]) == len(pair_specs)
    assert len(info["markers"]) == len(pair_specs)
    assert 2 <= len(pair_specs) <= 3
    assert 1 <= len(frame_layers) <= 4

    gaps = [len(spec["gap"]) for spec in pair_specs]
    assert min(gaps) >= 1
    assert len(set(gaps)) >= 2, "gap lengths must vary across pairs"
    assert any(
        _innermost_frame_color(spec["marker"], info["frames"]) is not None
        for spec in pair_specs
    ), "need at least one marker inside a frame"
    assert any(
        _innermost_frame_color(spec["marker"], info["frames"]) is None
        for spec in pair_specs
    ), "need at least one marker outside all frames"
    assert any(len({r for r, _c in spec["path"]}) > 1 and len({c for _r, c in spec["path"]}) > 1 for spec in pair_specs), (
        "need at least one bent path"
    )

    expected = _spec_output(grid, pair_specs, frame_layers)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise AssertionError("main(grid) disagrees with construction metadata")


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(20, 24)
    w = random.randint(20, 24)
    pair_count = random.choice((2, 3))
    zones = _zone_boxes(h, w)
    random.shuffle(zones)

    group_count = random.choice((1, 2))
    group_count = min(group_count, pair_count)
    frame_groups: list[dict] = []
    group_zones = zones[:group_count]
    free_zones = zones[group_count:]

    nested_first = random.random() < 0.65
    for i, zone in enumerate(group_zones):
        nested = nested_first if i == 0 else random.random() < 0.4
        if nested:
            color_choices = random.sample(FRAME_COLORS, 2)
            group = _random_frame_group(zone, True, (color_choices[0], color_choices[1]))
            if group is None:
                return None
        else:
            color = random.choice(FRAME_COLORS)
            group = _random_frame_group(zone, False, (color,))
            if group is None:
                return None
        frame_groups.append(group)

    inside_count = 1
    if pair_count == 3:
        inside_count = 2 if group_count == 2 and random.random() < 0.55 else 1
    inside_count = min(inside_count, group_count, pair_count - 1)
    outside_count = pair_count - inside_count
    if outside_count < 1:
        return None

    pair_regions: list[tuple[BBox, bool]] = []
    nested_groups = [group for group in frame_groups if len(group["layers"]) == 2]
    single_groups = [group for group in frame_groups if len(group["layers"]) == 1]
    ordered_groups = nested_groups + single_groups
    for group in ordered_groups[:inside_count]:
        pair_regions.append((group["inside_region"], True))
    for zone in free_zones[:outside_count]:
        pair_regions.append((zone, False))
    if len(pair_regions) != pair_count:
        return None
    random.shuffle(pair_regions)

    path_colors = random.sample(PATH_COLORS, pair_count)
    marker_colors = random.sample(MARKER_COLORS, pair_count)

    frame_layers = [layer for group in frame_groups for layer in group["layers"]]
    frame_cells = set().union(*(_rect_perimeter_cells(layer["bbox"]) for layer in frame_layers))
    occupied_pairs: set[Coord] = set()
    pair_specs: list[dict] = []

    for idx, (region, inside_flag) in enumerate(pair_regions):
        if inside_flag:
            length_bounds = (4, 6)
        else:
            length_bounds = (4, 8)
        spec = _generate_pair(region, path_colors[idx], marker_colors[idx], occupied_pairs, frame_cells, length_bounds)
        if spec is None:
            return None
        pair_specs.append(spec)
        occupied_pairs.update(spec["path"])
        occupied_pairs.add(spec["marker"])

    grid = _build_input(h, w, frame_layers, pair_specs)
    try:
        _validate_instance(grid, pair_specs, frame_layers)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
