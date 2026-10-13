import random
from collections import Counter, deque

import numpy as np


MARKER = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 600
GRID_RANGE = (18, 24)
COMPONENT_RANGE = (3, 4)
PATH_LENGTH_CHOICES = tuple(range(2, 9))
MIN_TOTAL_EMITTED = 3


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def add(pos: tuple[int, int], direction: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return pos[0] + direction[0] * k, pos[1] + direction[1] * k


def negate(direction: tuple[int, int]) -> tuple[int, int]:
    return -direction[0], -direction[1]


def turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def nonbg_components(grid: np.ndarray, bg: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _path_graph_info(
    comp: set[tuple[int, int]],
    center: tuple[int, int],
    marker: tuple[int, int],
    path_color: int,
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], int]:
    path_nodes = {marker}
    for cell in comp:
        if int(grid[cell]) == path_color:
            path_nodes.add(cell)
    if center not in path_nodes:
        raise AssertionError("gate center must be a path-color node")

    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in path_nodes:
        graph[cell] = [nb for nb in neighbors4(*cell) if nb in path_nodes]

    if len(graph[marker]) != 1:
        raise AssertionError("marker must touch exactly one path node")
    if len(graph[center]) != 1:
        raise AssertionError("gate center must have exactly one path continuation")

    for cell, nbrs in graph.items():
        if cell in (marker, center):
            continue
        if len(nbrs) != 2:
            raise AssertionError("path nodes must form a single simple chain")

    order = [marker]
    prev = None
    cur = marker
    while cur != center:
        nxts = [nb for nb in graph[cur] if nb != prev]
        if len(nxts) != 1:
            raise AssertionError("path ordering must be unique")
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt

    if len(order) != len(path_nodes):
        raise AssertionError("path graph must be connected")

    n = len(order) - 2
    if n < 1:
        raise AssertionError("path length must be positive")
    return order, n


def _parse_component(
    grid: np.ndarray,
    comp: set[tuple[int, int]],
    bg: int,
) -> dict[str, object]:
    markers = [cell for cell in comp if int(grid[cell]) == MARKER]
    if len(markers) != 1:
        raise AssertionError("each component must contain exactly one gray marker")
    marker = markers[0]

    candidates: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int, int]] = []
    for center in comp:
        if center == marker:
            continue
        path_color = int(grid[center])
        if path_color in (bg, MARKER):
            continue

        for axis in ((0, 1), (1, 0)):
            end1 = add(center, axis)
            end2 = add(center, negate(axis))
            if end1 not in comp or end2 not in comp:
                continue
            end_color = int(grid[end1])
            if end_color != int(grid[end2]) or end_color in (bg, MARKER, path_color):
                continue

            orth_dirs = [turn_left(axis), turn_right(axis)]
            path_dirs = [d for d in orth_dirs if add(center, d) in comp and int(grid[add(center, d)]) == path_color]
            if len(path_dirs) != 1:
                continue
            path_dir = path_dirs[0]
            out_dir = negate(path_dir)
            outward = add(center, out_dir)
            if outward in comp:
                continue
            if in_bounds(grid.shape, *outward) and int(grid[outward]) != bg:
                continue
            candidates.append((center, axis, out_dir, path_color, end_color))

    if len(candidates) != 1:
        raise AssertionError("each component must contain exactly one valid gate")

    center, axis, out_dir, path_color, gate_color = candidates[0]
    ordered_path, n = _path_graph_info(comp, center, marker, path_color, grid)
    path_dirs = [
        (ordered_path[i + 1][0] - ordered_path[i][0], ordered_path[i + 1][1] - ordered_path[i][1])
        for i in range(len(ordered_path) - 1)
    ]
    turn_count = sum(path_dirs[i] != path_dirs[i - 1] for i in range(1, len(path_dirs)))

    gate_ends = {add(center, axis), add(center, negate(axis))}
    path_cells = {cell for cell in comp if int(grid[cell]) == path_color and cell != center}
    if len(path_cells) != n:
        raise AssertionError("path-color count must match parsed path length")

    return {
        "cells": set(comp),
        "center": center,
        "marker": marker,
        "path_color": path_color,
        "gate_color": gate_color,
        "gate_ends": gate_ends,
        "out_dir": out_dir,
        "n": n,
        "ordered_path": ordered_path,
        "path_cells": path_cells,
        "turn_count": turn_count,
    }


def _analyze_grid(grid: np.ndarray) -> tuple[int, list[dict[str, object]], np.ndarray, list[list[tuple[int, int]]], list[int | None]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    bg = get_bg(g)
    comps = nonbg_components(g, bg)
    infos = [_parse_component(g, comp, bg) for comp in comps]

    base = g.copy()
    for info in infos:
        mr, mc = info["marker"]
        base[mr, mc] = bg
        for r, c in info["path_cells"]:
            base[r, c] = bg

    emissions: list[list[tuple[int, int]]] = []
    blockers: list[int | None] = []
    for info in infos:
        rr, cc = info["center"]
        dr, dc = info["out_dir"]
        n = int(info["n"])
        dist = 0
        points: list[tuple[int, int]] = []
        blocker_dist = None
        while True:
            dist += 1
            rr += dr
            cc += dc
            if not in_bounds(base.shape, rr, cc):
                break
            if int(base[rr, cc]) != bg:
                blocker_dist = dist
                break
            if dist % n == 0:
                points.append((rr, cc))
        emissions.append(points)
        blockers.append(blocker_dist)

    return bg, infos, base, emissions, blockers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray marker to its gate, erase the consumed path, then emit gray dots outward at the erased path's period.

    Input structure:
    - The dominant color is background.
    - Each 4-connected foreground component contains one gray marker, one single-color path, and one gate motif.
    - A gate is an end-center-end triplet: the center has the path color, the two gate ends share another color, and the marker-path chain approaches the center from exactly one perpendicular side.

    Rule:
    - For every component, count the number n of path-color cells excluding the gate center.
    - Erase the marker and all those path-color cells, but keep the gate center and the two gate ends.
    - From the gate center, move step by step along the gate's outward side; every time the walked distance reaches n, 2n, 3n, ... place a gray pixel.
    - Stop the outward scan as soon as it would leave the grid or touch any non-background cell.

    Key insight:
    - The path is both a track and a ruler: after the marker consumes it, the vanished path length becomes the spacing of the gate's emitted dot sequence.
    """
    g = np.asarray(grid, dtype=int)
    bg, infos, base, emissions, _blockers = _analyze_grid(g)
    out = base.copy()
    for points in emissions:
        for r, c in points:
            out[r, c] = MARKER
    return out


def _component_dirs_for_kind(
    n: int,
    out_dir: tuple[int, int],
    kind: str,
) -> list[tuple[int, int]] | None:
    in_dir = negate(out_dir)
    if kind == "straight":
        return [in_dir] * n
    if kind == "l":
        a = random.randint(1, n - 1)
        b = n - a
        bend = random.choice([turn_left(in_dir), turn_right(in_dir)])
        return [in_dir] * a + [bend] * b
    if kind == "z":
        if n < 4:
            return None
        a = random.randint(1, n - 3)
        c = random.randint(1, n - a - 2)
        b = n - a - c
        if b < 2:
            return None
        bend = random.choice([turn_left(in_dir), turn_right(in_dir)])
        return [in_dir] * a + [bend] * b + [in_dir] * c
    return None


def _build_component_geometry(
    shape: tuple[int, int],
    center: tuple[int, int],
    out_dir: tuple[int, int],
    n: int,
    *,
    preferred_kind: str | None = None,
) -> dict[str, object] | None:
    h, w = shape
    kind_options = [preferred_kind] if preferred_kind is not None else ["straight", "l", "z"]
    kind_options = [kind for kind in kind_options if kind is not None]
    random.shuffle(kind_options)
    for kind in kind_options:
        for _ in range(20):
            dirs = _component_dirs_for_kind(n, out_dir, kind)
            if dirs is None:
                continue
            gate_ends = {add(center, turn_left(out_dir)), add(center, turn_right(out_dir))}
            if any(not in_bounds((h, w), r, c) for r, c in gate_ends):
                continue

            path = []
            seen = {center}
            cur = center
            ok = True
            for direction in dirs:
                cur = add(cur, direction)
                if not in_bounds((h, w), *cur) or cur in seen:
                    ok = False
                    break
                path.append(cur)
                seen.add(cur)
            if not ok:
                continue

            marker = add(path[-1], dirs[-1])
            if not in_bounds((h, w), *marker) or marker in seen or marker in gate_ends:
                continue

            cells = set(path) | {center, marker} | gate_ends
            if len(cells) != len(path) + 4:
                continue

            outward = add(center, out_dir)
            if outward in cells:
                continue

            marker_neighbors = [nb for nb in neighbors4(*marker) if nb in cells]
            if len(marker_neighbors) != 1 or marker_neighbors[0] != path[-1]:
                continue

            return {
                "cells": cells,
                "center": center,
                "gate_ends": gate_ends,
                "path_cells": set(path),
                "marker": marker,
                "n": n,
                "out_dir": out_dir,
                "kind": kind,
                "turn_count": sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs))),
            }
    return None


def _padded_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    padded = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if in_bounds((h, w), rr, cc):
                    padded.add((rr, cc))
    return padded


def _fit_component(
    shape: tuple[int, int],
    geom: dict[str, object] | None,
    blocked_cells: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if geom is None:
        return False
    cells = geom["cells"]
    return not any(cell in blocked_cells or cell in reserved_bg for cell in cells)


def _border_distance(shape: tuple[int, int], center: tuple[int, int], out_dir: tuple[int, int]) -> int:
    rr, cc = center
    dist = 0
    while True:
        rr, cc = add((rr, cc), out_dir)
        if not in_bounds(shape, rr, cc):
            return dist
        dist += 1


def _corridor_cells(shape: tuple[int, int], center: tuple[int, int], out_dir: tuple[int, int], start_dist: int, end_dist: int | None) -> set[tuple[int, int]]:
    rr, cc = center
    cells = set()
    dist = 0
    while True:
        rr, cc = add((rr, cc), out_dir)
        if not in_bounds(shape, rr, cc):
            break
        dist += 1
        if dist >= start_dist and (end_dist is None or dist <= end_dist):
            cells.add((rr, cc))
    return cells


def _paint_component(grid: np.ndarray, geom: dict[str, object], path_color: int, gate_color: int) -> None:
    for r, c in geom["path_cells"] | {geom["center"]}:
        grid[r, c] = path_color
    for r, c in geom["gate_ends"]:
        grid[r, c] = gate_color
    mr, mc = geom["marker"]
    grid[mr, mc] = MARKER


def _try_place_random_component(
    shape: tuple[int, int],
    n: int,
    blocked_cells: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    *,
    must_have_open_ray: bool = False,
) -> dict[str, object] | None:
    h, w = shape
    centers = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(centers)
    for center in centers:
        out_dir = random.choice(DIRS4)
        if must_have_open_ray and _border_distance(shape, center, out_dir) < n:
            continue
        geom = _build_component_geometry(shape, center, out_dir, n)
        if _fit_component(shape, geom, blocked_cells, reserved_bg):
            return geom
    return None


def _validate_generated(grid: np.ndarray, expected_components: int, blocked_anchor_n: int) -> bool:
    try:
        bg, infos, base, emissions, blockers = _analyze_grid(grid)
    except AssertionError:
        return False

    if len(infos) != expected_components:
        return False
    if len({int(info["n"]) for info in infos}) != expected_components:
        return False
    if sum(len(points) for points in emissions) < MIN_TOTAL_EMITTED:
        return False
    if not any(info["turn_count"] >= 1 for info in infos):
        return False
    if len({tuple(info["out_dir"]) for info in infos}) < 2:
        return False

    emitted_once = set()
    for points in emissions:
        for cell in points:
            if int(base[cell]) != bg or cell in emitted_once:
                return False
            emitted_once.add(cell)

    anchor_ok = False
    free_ok = False
    for info, points, blocker in zip(infos, emissions, blockers):
        n = int(info["n"])
        if blocker is None and points:
            free_ok = True
        if n == blocked_anchor_n and blocker is not None and blocker > n and blocker % n != 0 and points:
            anchor_ok = True

    return anchor_ok and free_ok


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    shape = (h, w)
    n_components = random.randint(*COMPONENT_RANGE)

    usable_colors = [color for color in range(10) if color != MARKER]
    bg = random.choice(usable_colors)
    other_colors = [color for color in usable_colors if color != bg]
    random.shuffle(other_colors)
    if len(other_colors) < 2 * n_components:
        return None

    lengths = random.sample(PATH_LENGTH_CHOICES, n_components)
    n_a, n_b, n_c = lengths[:3]

    occupied = set()
    blocked_cells = set()
    reserved_bg = set()
    geoms: list[dict[str, object]] = []

    # A: the anchor component whose ray will be blocked at a non-multiple distance.
    placed_pair = False
    centers = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
    random.shuffle(centers)
    for center_a in centers:
        out_dir_a = random.choice(DIRS4)
        border_dist = _border_distance(shape, center_a, out_dir_a)
        if border_dist < n_a + n_b + 2:
            continue

        geom_a = _build_component_geometry(shape, center_a, out_dir_a, n_a)
        if not _fit_component(shape, geom_a, blocked_cells, reserved_bg):
            continue

        d_choices = [d for d in range(n_a + 1, min(2 * n_a, border_dist - n_b)) if d % n_a != 0]
        if not d_choices:
            continue
        random.shuffle(d_choices)
        for blocker_dist in d_choices:
            center_b = add(center_a, out_dir_a, blocker_dist)
            geom_b = _build_component_geometry(shape, center_b, negate(out_dir_a), n_b)
            corridor = _corridor_cells(shape, center_a, out_dir_a, 1, blocker_dist - 1)
            if not _fit_component(shape, geom_b, blocked_cells, reserved_bg | corridor):
                continue

            geoms.extend([geom_a, geom_b])
            pair_cells = geom_a["cells"] | geom_b["cells"]
            occupied |= pair_cells
            blocked_cells |= _padded_cells(shape, pair_cells)
            reserved_bg |= corridor
            placed_pair = True
            break
        if placed_pair:
            break
    if not placed_pair:
        return None

    # C: a guaranteed free emitter whose outward corridor stays empty.
    geom_c = None
    for _ in range(200):
        candidate = _try_place_random_component(shape, n_c, blocked_cells, reserved_bg, must_have_open_ray=True)
        if candidate is None:
            continue
        corridor = _corridor_cells(shape, candidate["center"], candidate["out_dir"], 1, None)
        if candidate["cells"] & corridor:
            continue
        geom_c = candidate
        reserved_bg |= corridor
        occupied |= candidate["cells"]
        blocked_cells |= _padded_cells(shape, candidate["cells"])
        geoms.append(candidate)
        break
    if geom_c is None:
        return None

    # Remaining components are placed randomly but cannot invade reserved emission corridors.
    for extra_n in lengths[3:]:
        geom = _try_place_random_component(shape, extra_n, blocked_cells, reserved_bg)
        if geom is None:
            return None
        geoms.append(geom)
        occupied |= geom["cells"]
        blocked_cells |= _padded_cells(shape, geom["cells"])

    color_pairs = [(other_colors[2 * i], other_colors[2 * i + 1]) for i in range(n_components)]
    grid = np.full(shape, bg, dtype=int)
    for geom, (path_color, gate_color) in zip(geoms, color_pairs):
        _paint_component(grid, geom, path_color, gate_color)

    if get_bg(grid) != bg:
        return None
    if not _validate_generated(grid, n_components, n_a):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-emission input")
