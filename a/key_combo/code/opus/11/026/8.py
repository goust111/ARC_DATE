from __future__ import annotations

from collections import defaultdict

import numpy as np

Coord = tuple[int, int]

BG = 0
WALL = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _row_major_colors(grid: np.ndarray) -> list[int]:
    seen = {BG, WALL}
    colors: list[int] = []
    for value in map(int, grid.ravel()):
        if value not in seen:
            seen.add(value)
            colors.append(value)
    return colors


def _segment(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment endpoints must share a row or column")


def _join_segments(parts: list[list[Coord]]) -> list[Coord]:
    out: list[Coord] = []
    for idx, part in enumerate(parts):
        out.extend(part if idx == 0 else part[1:])
    return out


def _l_path(start: Coord, end: Coord, first_axis: str) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if first_axis == "h":
        corner = (r0, c1)
        return _join_segments([_segment(start, corner), _segment(corner, end)])
    if first_axis == "v":
        corner = (r1, c0)
        return _join_segments([_segment(start, corner), _segment(corner, end)])
    raise AssertionError("invalid L-path axis")


def _z_path(start: Coord, end: Coord, orientation: str, mid: int) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if orientation == "h":
        p1 = (r0, mid)
        p2 = (r1, mid)
        return _join_segments([_segment(start, p1), _segment(p1, p2), _segment(p2, end)])
    if orientation == "v":
        p1 = (mid, c0)
        p2 = (mid, c1)
        return _join_segments([_segment(start, p1), _segment(p1, p2), _segment(p2, end)])
    raise AssertionError("invalid Z-path orientation")


def _path_clear(path: list[Coord], blocked: set[Coord], shape: tuple[int, int], endpoints: set[Coord]) -> bool:
    for cell in path:
        if cell in endpoints:
            continue
        if cell in blocked or not _in_bounds(shape, cell[0], cell[1]):
            return False
    return True


def _path_kind(path: list[Coord]) -> str:
    dirs: list[Coord] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((np.sign(r1 - r0), np.sign(c1 - c0)))
    turns = 0
    compressed: list[Coord] = []
    for direction in dirs:
        if not compressed or compressed[-1] != direction:
            compressed.append(direction)
            if len(compressed) > 1:
                turns += 1
    if turns == 1:
        return "Lh" if compressed[0][0] == 0 else "Lv"
    if turns == 2:
        return "Zh" if compressed[0][0] == 0 else "Zv"
    raise AssertionError("path must have one or two bends")


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    pairs: list[dict] = []
    for color in _row_major_colors(grid):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 2, "each foreground color must mark exactly two endpoints"
        endpoints = sorted((int(r), int(c)) for r, c in coords)
        assert endpoints[0][0] != endpoints[1][0] and endpoints[0][1] != endpoints[1][1]
        pairs.append({"color": int(color), "endpoints": endpoints, "anchor": endpoints[0]})
    assert 2 <= len(pairs) <= 3, "expected 2 or 3 endpoint colors"
    return sorted(pairs, key=lambda pair: pair["anchor"])


def _plan_pair(pair: dict, shape: tuple[int, int], walls: set[Coord], all_endpoints: dict[Coord, int]) -> dict:
    start, end = pair["endpoints"]
    blocked = set(walls)
    blocked.update(cell for cell, color in all_endpoints.items() if color != pair["color"])
    endpoints = {start, end}

    path = _l_path(start, end, "h")
    if _path_clear(path, blocked, shape, endpoints):
        return {
            "color": pair["color"],
            "endpoints": pair["endpoints"],
            "anchor": pair["anchor"],
            "path": path,
            "length": len(path) - 1,
            "kind": "Lh",
        }

    path = _l_path(start, end, "v")
    if _path_clear(path, blocked, shape, endpoints):
        return {
            "color": pair["color"],
            "endpoints": pair["endpoints"],
            "anchor": pair["anchor"],
            "path": path,
            "length": len(path) - 1,
            "kind": "Lv",
        }

    candidates: list[dict] = []
    rows, cols = shape
    for mid_c in range(cols):
        if mid_c in (start[1], end[1]):
            continue
        candidate = _z_path(start, end, "h", mid_c)
        if _path_clear(candidate, blocked, shape, endpoints):
            candidates.append(
                {
                    "color": pair["color"],
                    "endpoints": pair["endpoints"],
                    "anchor": pair["anchor"],
                    "path": candidate,
                    "length": len(candidate) - 1,
                    "kind": "Zh",
                    "mid": mid_c,
                }
            )
    for mid_r in range(rows):
        if mid_r in (start[0], end[0]):
            continue
        candidate = _z_path(start, end, "v", mid_r)
        if _path_clear(candidate, blocked, shape, endpoints):
            candidates.append(
                {
                    "color": pair["color"],
                    "endpoints": pair["endpoints"],
                    "anchor": pair["anchor"],
                    "path": candidate,
                    "length": len(candidate) - 1,
                    "kind": "Zv",
                    "mid": mid_r,
                }
            )

    assert candidates, "expected a clear Z path when both L paths are blocked"
    best_length = min(candidate["length"] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate["length"] == best_length]
    assert len(best) == 1, "expected a unique shortest Z path"
    return best[0]


def _solve_routes(
    grid: np.ndarray, *, require_wall_count: bool = True
) -> tuple[list[dict], list[dict], set[Coord], dict[Coord, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must lie in 0..9"
    walls = {tuple(map(int, cell)) for cell in np.argwhere(g == WALL)}
    if require_wall_count:
        assert 5 <= len(walls) <= 10, "expected 5-10 teal wall pixels"

    pairs = _extract_pairs(g)
    endpoints = {cell: pair["color"] for pair in pairs for cell in pair["endpoints"]}
    for cell in endpoints:
        for nr, nc in _neighbors4(*cell):
            assert (nr, nc) not in walls, "endpoints must not be 4-adjacent to walls"

    routes = [_plan_pair(pair, g.shape, walls, endpoints) for pair in pairs]
    for route in routes:
        assert _path_kind(route["path"]) == route["kind"]
    return pairs, routes, walls, endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair by its shortest orthogonal detour path and resolve crossings by path priority.

    Input structure:
    - A black background contains 2 or 3 non-teal foreground colors, each appearing exactly twice as an endpoint pair.
    - Teal cells form one-pixel-wide obstacle wall segments that paths may not enter.
    - Endpoints of different colors are separate objects and are never 4-adjacent to teal wall cells.

    Rule:
    - For each color, find its two endpoints and route a shortest orthogonal path that avoids teal walls and every other color's endpoints.
    - Prefer the horizontal-then-vertical L path; if that is blocked, try the vertical-then-horizontal L path; if both are blocked, use the unique shortest Z path with two bends.
    - Draw all routed paths in their endpoint color, keep teal walls unchanged, and keep all endpoint cells unchanged.
    - When multiple routed paths pass through the same non-endpoint cell, show the color of the route with smaller total path length; if lengths tie, use the route whose top-left endpoint is lexicographically smaller.

    Key insight:
    - The output depends on three coupled decisions: obstacle-aware path planning chooses each route, crossings compare whole-route lengths rather than local geometry, and equal-length conflicts break by endpoint position instead of color identity.
    """
    g = np.asarray(grid, dtype=int)
    _pairs, routes, _walls, endpoints = _solve_routes(g)

    out = g.copy()
    claims: dict[Coord, list[dict]] = defaultdict(list)
    for route in routes:
        for cell in route["path"][1:-1]:
            claims[cell].append(route)

    for (r, c), owners in claims.items():
        if (r, c) in endpoints or int(g[r, c]) == WALL:
            continue
        winner = min(owners, key=lambda route: (route["length"], route["anchor"][0], route["anchor"][1]))
        out[r, c] = int(winner["color"])
    return out


def _paint_cells(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        assert int(grid[r, c]) == BG
        grid[r, c] = int(color)


def _base_case_ll(equal_lengths: bool, add_third: bool) -> np.ndarray:
    height = int(np.random.randint(14 if add_third else 12, 17))
    width = int(np.random.randint(14 if add_third else 12, 17))
    grid = np.zeros((height, width), dtype=int)
    colors = np.random.choice(np.array(PALETTE, dtype=int), size=3 if add_third else 2, replace=False)

    rp = int(np.random.randint(4, height - (7 if add_third else 5)))
    cp = int(np.random.randint(4, width - (7 if add_third else 5)))

    if equal_lengths:
        a_end = (rp + 3, cp)
        b_end = (rp, cp + 3)
    else:
        if bool(np.random.randint(2)):
            a_end = (rp + 3, cp)
            b_end = (rp, cp + 4)
        else:
            a_end = (rp + 4, cp)
            b_end = (rp, cp + 3)

    pair_a = [(rp, cp - 3), a_end]
    pair_b = [(rp - 3, cp), b_end]
    walls = {(rp + 2, cp - 3), (rp - 3, cp + 2), (height - 3, 2), (height - 2, 2), (height - 2, 3)}

    _paint_cells(grid, set(pair_a), int(colors[0]))
    _paint_cells(grid, set(pair_b), int(colors[1]))

    if add_third:
        pair_c = [(height - 5, width - 5), (height - 2, width - 2)]
        walls.add((height - 3, width - 5))
        _paint_cells(grid, set(pair_c), int(colors[2]))

    _paint_cells(grid, walls, WALL)
    return grid


def _base_case_lz(add_third: bool) -> np.ndarray:
    height = int(np.random.randint(14 if add_third else 12, 17))
    width = int(np.random.randint(14 if add_third else 13, 17))
    grid = np.zeros((height, width), dtype=int)
    colors = np.random.choice(np.array(PALETTE, dtype=int), size=3 if add_third else 2, replace=False)

    ra = int(np.random.randint(4, height - (8 if add_third else 6)))
    qc = int(np.random.randint(5, width - (8 if add_third else 7)))

    pair_a = [(ra, qc - 1), (ra + 4, qc + 1)]
    pair_b = [(ra - 2, qc - 3), (ra + 2, qc + 3)]
    walls = {
        (ra + 2, qc - 1),
        (ra + 2, qc - 2),
        (ra - 2, qc + 1),
        (ra - 2, qc + 2),
        (ra - 1, qc + 2),
        (ra, qc + 2),
        (ra + 1, qc + 2),
    }

    _paint_cells(grid, set(pair_a), int(colors[0]))
    _paint_cells(grid, set(pair_b), int(colors[1]))

    if add_third:
        pair_c = [(height - 5, width - 5), (height - 2, width - 2)]
        walls.add((height - 3, width - 5))
        _paint_cells(grid, set(pair_c), int(colors[2]))

    _paint_cells(grid, walls, WALL)
    return grid


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    transformed = np.rot90(grid, int(np.random.randint(4)))
    if bool(np.random.randint(2)):
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _route_signature(routes: list[dict]) -> dict[Coord, tuple[Coord, ...]]:
    return {route["anchor"]: tuple(route["path"]) for route in routes}


def _crossings(routes: list[dict]) -> dict[Coord, list[dict]]:
    claims: dict[Coord, list[dict]] = defaultdict(list)
    for route in routes:
        for cell in route["path"][1:-1]:
            claims[cell].append(route)
    return {cell: owners for cell, owners in claims.items() if len(owners) > 1}


def _validate_generated(grid: np.ndarray, mode: str, add_third: bool) -> bool:
    try:
        pairs, routes, walls, endpoints = _solve_routes(grid)
    except Exception:
        return False

    if len(pairs) != (3 if add_third else 2):
        return False

    crossings = _crossings(routes)
    if mode in {"ll_diff", "ll_equal"}:
        if len(crossings) != 1:
            return False
    elif mode == "lz_diff":
        if not (1 <= len(crossings) <= 3):
            return False
    else:
        return False

    cell, owners = next(iter(crossings.items()))
    lengths = sorted(route["length"] for route in owners)
    kinds = sorted(route["kind"] for route in routes)

    if mode == "ll_equal":
        if kinds.count("Lh") < 1 or kinds.count("Lv") < 1:
            return False
        if lengths[0] != lengths[-1]:
            return False
        winner = min(owners, key=lambda route: (route["length"], route["anchor"][0], route["anchor"][1]))
        output = main(grid)
        if int(output[cell]) != int(winner["color"]):
            return False
    elif mode == "ll_diff":
        if kinds.count("Lh") < 1 or kinds.count("Lv") < 1:
            return False
        if lengths[0] == lengths[-1]:
            return False
    elif mode == "lz_diff":
        if not any(kind.startswith("Z") for kind in kinds):
            return False
        if lengths[0] == lengths[-1]:
            return False

    output = main(grid)
    if np.array_equal(output, grid):
        return False
    for endpoint, color in endpoints.items():
        if int(output[endpoint]) != int(color):
            return False
    for wall in walls:
        if int(output[wall]) != WALL:
            return False

    without_walls = grid.copy()
    without_walls[without_walls == WALL] = BG
    try:
        _, routes_without, _, _ = _solve_routes(without_walls, require_wall_count=False)
    except Exception:
        return False
    if _route_signature(routes) == _route_signature(routes_without):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        add_third = bool(np.random.randint(2))
        mode = str(np.random.choice(np.array(["ll_diff", "ll_equal", "lz_diff"], dtype=object)))
        if mode == "ll_diff":
            grid = _base_case_ll(equal_lengths=False, add_third=add_third)
        elif mode == "ll_equal":
            grid = _base_case_ll(equal_lengths=True, add_third=add_third)
        else:
            grid = _base_case_lz(add_third=add_third)

        grid = _transform_grid(grid)
        if _validate_generated(grid, mode, add_third):
            return grid
    raise RuntimeError("failed to generate a valid new puzzle input")
