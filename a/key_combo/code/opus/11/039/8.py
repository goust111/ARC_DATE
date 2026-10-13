import random
from collections import Counter

import numpy as np


BG = 0
BODY = 8
COUPLER = 9
MARKER = 3
OPS = tuple((flip, rot) for flip in (False, True) for rot in range(4))
DIR_VECTORS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_op(patch: np.ndarray, op: tuple[bool, int]) -> np.ndarray:
    flip, rot = op
    out = np.fliplr(patch) if flip else patch.copy()
    if rot:
        out = np.rot90(out, rot)
    return out


def _coords_of_color(patch: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch == color)]


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shift(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _translation_matches(
    src: list[tuple[int, int]], dst: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    src_set = set(src)
    dst_set = set(dst)
    good: set[tuple[int, int]] = set()
    for sr, sc in src:
        for dr, dc in dst:
            off = (dr - sr, dc - sc)
            if {(r + off[0], c + off[1]) for r, c in src_set} == dst_set:
                good.add(off)
    return sorted(good)


def _marker_direction(patch: np.ndarray) -> str:
    marker_cells = _coords_of_color(patch, MARKER)
    assert len(marker_cells) == 1
    mr, mc = marker_cells[0]
    body_neighbors = []
    for nr, nc in _neighbors4(mr, mc):
        if _in_bounds(patch.shape, nr, nc) and patch[nr, nc] == BODY:
            body_neighbors.append((nr, nc))
    assert len(body_neighbors) == 1
    br, bc = body_neighbors[0]
    delta = (mr - br, mc - bc)
    for name, vec in DIR_VECTORS.items():
        if delta == vec:
            return name
    raise AssertionError("invalid marker direction")


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _nearest_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[str, int]:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    dists = {
        "top": r0,
        "bottom": h - 1 - r1,
        "left": c0,
        "right": w - 1 - c1,
    }
    best = min(dists.values())
    names = [name for name, dist in dists.items() if dist == best]
    assert len(names) == 1
    return names[0], best


def _station_info(coords: list[tuple[int, int]], shape: tuple[int, int]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    vertical = (r1 - r0) >= (c1 - c0)
    axis_values = sorted(c for _, c in coords) if vertical else sorted(r for r, _ in coords)
    axis = axis_values[len(axis_values) // 2]
    edge, best = _nearest_edge(coords, shape)
    dists = {
        "top": r0,
        "bottom": shape[0] - 1 - r1,
        "left": c0,
        "right": shape[1] - 1 - c1,
    }
    others = [dist for name, dist in dists.items() if name != edge]
    assert best + 3 <= min(others)
    return {
        "coords": sorted(coords),
        "size": len(coords),
        "bbox": (r0, c0, r1, c1),
        "vertical": vertical,
        "axis": axis,
        "edge": edge,
    }


def _parse_grid(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    stations: list[dict] = []
    for comp in _components(g == COUPLER):
        attached = False
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(g.shape, nr, nc) and g[nr, nc] == BODY:
                    attached = True
                    break
            if attached:
                break
        if not attached:
            stations.append(_station_info(comp, g.shape))

    sprites: list[dict] = []
    fg_mask = np.isin(g, [BODY, COUPLER, MARKER])
    for comp in _components(fg_mask):
        colors = {int(g[r, c]) for r, c in comp}
        if BODY not in colors:
            continue
        assert colors <= {BODY, COUPLER, MARKER}
        box = _bbox(comp)
        patch = _crop(g, box)
        couplers = _coords_of_color(patch, COUPLER)
        marker = _coords_of_color(patch, MARKER)
        assert len(marker) == 1
        sprites.append(
            {
                "coords": comp,
                "bbox": box,
                "patch": patch,
                "coupler_count": len(couplers),
            }
        )
    return stations, sprites


def _dock_options(
    patch: np.ndarray, station: dict, shape: tuple[int, int]
) -> list[dict]:
    opts: dict[tuple[tuple[tuple[int, int], ...], tuple[int, int]], dict] = {}
    target_station = station["coords"]
    target_edge = station["edge"]
    for op in OPS:
        transformed = _apply_op(patch, op)
        if _marker_direction(transformed) != target_edge:
            continue
        couplers = _coords_of_color(transformed, COUPLER)
        body = _coords_of_color(transformed, BODY)
        marker = _coords_of_color(transformed, MARKER)
        for dr, dc in _translation_matches(couplers, target_station):
            abs_body = _shift(body, dr, dc)
            abs_marker = _shift(marker, dr, dc)
            if not all(_in_bounds(shape, r, c) for r, c in abs_body + abs_marker):
                continue
            key = (tuple(sorted(abs_body)), abs_marker[0])
            opts[key] = {
                "op": op,
                "body": abs_body,
                "marker": abs_marker[0],
            }
    return list(opts.values())


def _solve_layout(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    stations, sprites = _parse_grid(g)
    assert len(stations) == len(sprites) == 2
    assert len({station["size"] for station in stations}) == len(stations)
    assert len({sprite["coupler_count"] for sprite in sprites}) == len(sprites)
    by_size = {station["size"]: station for station in stations}

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = BG

    option_groups: list[list[dict]] = []
    for sprite in sprites:
        station = by_size[sprite["coupler_count"]]
        raw_options = _dock_options(sprite["patch"], station, g.shape)
        assert raw_options
        group = []
        station_cells = set(station["coords"])
        for placed in raw_options:
            mirrored = _reflected_body(station, set(placed["body"]))
            final_cells = set(placed["body"]) | {placed["marker"]} | (mirrored - station_cells)
            if not all(_in_bounds(g.shape, r, c) for r, c in final_cells):
                continue
            group.append({"station": station, **placed, "mirrored": mirrored, "final_cells": final_cells})
        assert group
        option_groups.append(group)

    order = sorted(range(len(option_groups)), key=lambda idx: len(option_groups[idx]))
    station_occupied = {tuple(map(int, rc)) for rc in np.argwhere(out != BG)}
    solutions: dict[bytes, tuple[np.ndarray, list[dict]]] = {}

    def backtrack(pos: int, occupied: set[tuple[int, int]], chosen: list[dict]) -> None:
        if pos == len(order):
            candidate = out.copy()
            for placed in chosen:
                _paint(candidate, placed["body"], BODY)
                candidate[placed["marker"]] = MARKER
            for placed in chosen:
                for r, c in placed["mirrored"]:
                    if candidate[r, c] == BG:
                        candidate[r, c] = BODY
            solutions[candidate.tobytes()] = (candidate, chosen.copy())
            return

        idx = order[pos]
        for placed in option_groups[idx]:
            if placed["final_cells"] & occupied:
                continue
            backtrack(pos + 1, occupied | placed["final_cells"], chosen + [placed])

    backtrack(0, station_occupied.copy(), [])
    assert len(solutions) == 1
    solved, placements = next(iter(solutions.values()))
    return solved, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each coupler-bearing sprite to the same-sized station, dock it facing the station's nearest edge, then mirror-complete the docked body across the station axis.

    Input structure:
    - Black background with isolated maroon station clusters and separate multicolor sprites.
    - A station is a 4-connected maroon cluster not touching any cyan body pixel.
    - A sprite is one 4-connected component containing cyan body pixels, attached maroon couplers, and exactly one green marker pixel.

    Rule:
    - Count each sprite's maroon couplers and pair it with the unique station of the same size.
    - Read the target station's nearest grid edge and apply the unique rotation/flip that makes the green marker face that edge.
    - Translate the transformed sprite so its couplers land exactly on the station cells, erase the original sprite, and keep only the docked cyan body plus the green marker while the station maroon cells remain.
    - If the station is vertically arranged (row span >= column span), mirror every docked cyan body pixel across the station's center column; otherwise mirror across the station's center row. Only paint mirrored cyan pixels onto black cells.

    Key insight:
    - The station controls three separate decisions: coupler count chooses the partner, nearest edge fixes the final orientation, and station alignment chooses the mirror axis.
    """
    solved, _ = _solve_layout(np.asarray(grid, dtype=int))
    return solved


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for nr, nc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if _in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _reflected_body(station: dict, body: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    if station["vertical"]:
        axis_c = station["axis"]
        for r, c in body:
            out.add((r, 2 * axis_c - c))
    else:
        axis_r = station["axis"]
        for r, c in body:
            out.add((2 * axis_r - r, c))
    return out


def _mismatch(station_orientation: str, edge: str) -> bool:
    if station_orientation == "horizontal":
        return edge in {"left", "right"}
    return edge in {"top", "bottom"}


def _grow_body(
    station: dict,
    shape: tuple[int, int],
    occupied_forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    line_like = (
        station["size"] in {1, 2}
        and (
            len({r for r, _ in station["coords"]}) == 1
            or len({c for _, c in station["coords"]}) == 1
        )
    )
    if line_like:
        templated = _line_template_body(station)
        if templated is not None:
            body, marker = templated
            reflected = _reflected_body(station, body)
            footprint = set(station["coords"]) | body | reflected | {marker}
            if all(_in_bounds(shape, r, c) for r, c in footprint) and not (footprint & occupied_forbidden):
                patch = _build_docked_patch(body, station, marker)
                if _dock_options(patch, station, shape):
                    return body, marker

    station_cells = set(station["coords"])
    r0, c0, r1, c1 = station["bbox"]
    axis = station["axis"]
    if station["vertical"]:
        side_options = ["left", "right"]
    else:
        side_options = ["top", "bottom"]
    rng.shuffle(side_options)

    allowed = {
        (r, c)
        for r in range(r0 - 3, r1 + 4)
        for c in range(c0 - 3, c1 + 4)
        if _in_bounds(shape, r, c) and (r, c) not in station_cells
    }
    start_candidates = [
        cell
        for cell in allowed
        if any((nr, nc) in station_cells for nr, nc in _neighbors4(*cell))
        and cell not in occupied_forbidden
    ]
    if not start_candidates:
        return None

    for preferred_side in side_options:
        for _ in range(320):
            target_size = rng.randint(4, 7)
            body = {rng.choice(start_candidates)}
            while len(body) < target_size:
                frontier = []
                for br, bc in body:
                    for nr, nc in _neighbors4(br, bc):
                        if (nr, nc) in allowed and (nr, nc) not in body and (nr, nc) not in occupied_forbidden:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                body.add(rng.choice(frontier))
            if len(body) < target_size:
                continue
            if station["vertical"]:
                dominant = sum(c < axis for _, c in body) if preferred_side == "left" else sum(c > axis for _, c in body)
            else:
                dominant = sum(r < axis for r, _ in body) if preferred_side == "top" else sum(r > axis for r, _ in body)
            if dominant < len(body) - 2:
                continue
            if any(not any((nr, nc) in body for nr, nc in _neighbors4(r, c)) for r, c in station_cells):
                continue
            reflected = _reflected_body(station, body)
            footprint = body | reflected | station_cells
            if not all(_in_bounds(shape, r, c) for r, c in footprint):
                continue
            if footprint & occupied_forbidden:
                continue
            if not (reflected - body - station_cells):
                continue
            marker = _choose_marker(tuple(body), station["edge"], shape, station_cells, occupied_forbidden | reflected)
            if marker is None:
                continue
            patch = _build_docked_patch(body, station, marker)
            if not _dock_options(patch, station, shape):
                continue
            return body, marker
    return None


def _choose_marker(
    body: tuple[tuple[int, int], ...],
    edge: str,
    shape: tuple[int, int],
    station_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    dr, dc = DIR_VECTORS[edge]
    candidates = []
    body_set = set(body)
    for br, bc in body:
        mr, mc = br + dr, bc + dc
        if not _in_bounds(shape, mr, mc):
            continue
        if (mr, mc) in body_set or (mr, mc) in station_cells or (mr, mc) in forbidden:
            continue
        neighbor_count = 0
        for nr, nc in _neighbors4(mr, mc):
            if (nr, nc) in body_set:
                neighbor_count += 1
        if neighbor_count == 1:
            candidates.append(((br, bc), (mr, mc)))
    if not candidates:
        return None
    if edge == "top":
        candidates.sort(key=lambda item: (item[0][0], item[0][1]))
    elif edge == "bottom":
        candidates.sort(key=lambda item: (-item[0][0], item[0][1]))
    elif edge == "left":
        candidates.sort(key=lambda item: (item[0][1], item[0][0]))
    else:
        candidates.sort(key=lambda item: (-item[0][1], item[0][0]))
    return candidates[0][1]


def _line_template_body(station: dict) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    coords = sorted(station["coords"])
    if station["size"] == 1:
        r, c = coords[0]
        if station["edge"] == "left":
            body = {(r, c - 1), (r + 1, c - 1), (r + 1, c - 2), (r + 2, c - 2), (r + 2, c - 1)}
            marker = (r + 1, c - 3)
        elif station["edge"] == "right":
            body = {(r, c + 1), (r + 1, c + 1), (r + 1, c + 2), (r + 2, c + 2), (r + 2, c + 1)}
            marker = (r + 1, c + 3)
        else:
            return None
        return body, marker
    if station["vertical"]:
        c = station["axis"]
        rows = sorted(r for r, _ in coords)
        r0, r1 = rows[0], rows[-1]
        if station["edge"] == "left":
            body = {(r0, c - 1), (r1, c - 1), (r1, c - 2), (r1 + 1, c - 2), (r1 + 1, c - 1)}
            marker = (r1, c - 3)
        elif station["edge"] == "right":
            body = {(r0, c + 1), (r1, c + 1), (r1, c + 2), (r1 + 1, c + 2), (r1 + 1, c + 1)}
            marker = (r1, c + 3)
        else:
            return None
    else:
        r = station["axis"]
        cols = sorted(c for _, c in coords)
        c0, c1 = cols[0], cols[-1]
        if station["edge"] == "top":
            body = {(r - 1, c0), (r - 1, c1), (r - 2, c1), (r - 2, c1 + 1), (r - 1, c1 + 1)}
            marker = (r - 3, c1)
        elif station["edge"] == "bottom":
            body = {(r + 1, c0), (r + 1, c1), (r + 2, c1), (r + 2, c1 + 1), (r + 1, c1 + 1)}
            marker = (r + 3, c1)
        else:
            return None
    return body, marker


def _make_line_station(
    size: int, vertical: bool, edge: str, shape: tuple[int, int], rng: random.Random
) -> dict | None:
    h, w = shape
    if vertical:
        if edge == "top":
            r0 = rng.randint(4, 6)
            c = rng.randint(6, w - 7)
        elif edge == "bottom":
            r0 = h - size - rng.randint(4, 6)
            c = rng.randint(6, w - 7)
        elif edge == "left":
            c = rng.randint(4, 6)
            r0 = rng.randint(6, h - size - 7)
        else:
            c = w - 1 - rng.randint(4, 6)
            r0 = rng.randint(6, h - size - 7)
        coords = [(r0 + i, c) for i in range(size)]
    else:
        if edge == "top":
            r = rng.randint(4, 6)
            c0 = rng.randint(6, w - size - 7)
        elif edge == "bottom":
            r = h - 1 - rng.randint(4, 6)
            c0 = rng.randint(6, w - size - 7)
        elif edge == "left":
            c0 = rng.randint(4, 6)
            r = rng.randint(6, h - 7)
        else:
            c0 = w - size - rng.randint(4, 6)
            r = rng.randint(6, h - 7)
        coords = [(r, c0 + i) for i in range(size)]
    if not all(_in_bounds(shape, r, c) for r, c in coords):
        return None
    try:
        info = _station_info(coords, shape)
    except AssertionError:
        return None
    if info["edge"] != edge:
        return None
    return info


def _make_l_station(edge: str, shape: tuple[int, int], rng: random.Random) -> dict | None:
    h, w = shape
    if edge == "top":
        r0 = rng.randint(4, 6)
        c0 = rng.randint(7, w - 10)
    else:
        r0 = h - 2 - rng.randint(4, 6)
        c0 = rng.randint(7, w - 10)
    patterns = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
    ]
    rng.shuffle(patterns)
    for pattern in patterns:
        coords = [(r0 + dr, c0 + dc) for dr, dc in pattern]
        if not all(_in_bounds(shape, r, c) for r, c in coords):
            continue
        try:
            info = _station_info(coords, shape)
        except AssertionError:
            continue
        if info["edge"] == edge and info["vertical"]:
            return info
    return None


def _build_docked_patch(body: set[tuple[int, int]], station: dict, marker: tuple[int, int]) -> np.ndarray:
    cells = set(body) | set(station["coords"]) | {marker}
    r0, c0, r1, c1 = _bbox(list(cells))
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = BODY
    for r, c in station["coords"]:
        patch[r - r0, c - c0] = COUPLER
    patch[marker[0] - r0, marker[1] - c0] = MARKER
    return patch


def _scatter_patch(
    patch: np.ndarray,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden_boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    ops = list(OPS)
    rng.shuffle(ops)
    for op in ops:
        transformed = _apply_op(patch, op)
        cells = [tuple(map(int, rc)) for rc in np.argwhere(transformed != BG)]
        h, w = transformed.shape
        for _ in range(200):
            r0 = rng.randint(0, shape[0] - h)
            c0 = rng.randint(0, shape[1] - w)
            abs_cells = {(r0 + r, c0 + c) for r, c in cells}
            if _halo(abs_cells, shape) & occupied:
                continue
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(not (box[2] < b[0] - 1 or b[2] + 1 < box[0] or box[3] < b[1] - 1 or b[3] + 1 < box[1]) for b in forbidden_boxes):
                continue
            placed = np.zeros(shape, dtype=int)
            for r, c in cells:
                placed[r0 + r, c0 + c] = transformed[r, c]
            return placed
    return None


def _render_output(layout: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for item in layout:
        _paint(out, item["station"]["coords"], COUPLER)
        _paint(out, sorted(item["body"]), BODY)
        out[item["marker"]] = MARKER
    for item in layout:
        reflected = _reflected_body(item["station"], item["body"])
        for r, c in reflected:
            if _in_bounds(shape, r, c) and out[r, c] == BG:
                out[r, c] = BODY
    return out


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    try:
        stations, sprites = _parse_grid(inp)
    except AssertionError:
        return False
    if len(stations) != 2 or len(sprites) != 2:
        return False
    if {station["size"] for station in stations} != {1, 3}:
        return False
    if any(not _dock_options(sprite["patch"], {**station}, inp.shape) for sprite in sprites for station in stations if sprite["coupler_count"] == station["size"]):
        return False
    if np.array_equal(inp, expected):
        return False
    if np.count_nonzero(expected == BODY) <= np.count_nonzero(inp == BODY):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 24)
    w = rng.randint(18, 24)
    shape = (h, w)

    station_a = _make_line_station(1, True, rng.choice(["left", "right"]), shape, rng)
    station_b = _make_l_station(rng.choice(["top", "bottom"]), shape, rng)
    if station_a is None or station_b is None:
        return None
    stations = [station_a, station_b]
    if len(set(cell for station in stations for cell in station["coords"])) != 4:
        return None
    if _halo(set(stations[0]["coords"]), shape) & set(stations[1]["coords"]):
        return None

    layout: list[dict] = []
    occupied_output: set[tuple[int, int]] = set()
    for station in stations:
        built = _grow_body(station, shape, occupied_output, rng)
        if built is None:
            return None
        body, marker = built
        reflected = _reflected_body(station, body)
        footprint = set(station["coords"]) | body | reflected | {marker}
        if _halo(footprint, shape) & occupied_output:
            return None
        layout.append({"station": station, "body": body, "marker": marker})
        occupied_output |= footprint

    expected = _render_output(layout, shape)

    inp = np.zeros(shape, dtype=int)
    for station in stations:
        _paint(inp, station["coords"], COUPLER)

    occupied_input = set(station_cell for station in stations for station_cell in station["coords"])
    forbidden_boxes = [station["bbox"] for station in stations]
    for item in sorted(layout, key=lambda x: -len(x["body"])):
        docked_patch = _build_docked_patch(item["body"], item["station"], item["marker"])
        scattered = _scatter_patch(docked_patch, shape, occupied_input, forbidden_boxes, rng)
        if scattered is None:
            return None
        for r, c in np.argwhere(scattered != BG):
            occupied_input.add((int(r), int(c)))
        inp[scattered != BG] = scattered[scattered != BG]

    if not _validate_generated(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(3000):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
