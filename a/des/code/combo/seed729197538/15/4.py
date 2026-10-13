from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


BG = 0
TOP_MARKER = 2
BOTTOM_MARKER = 1
LEFT_MARKER = 4
RIGHT_MARKER = 7
OBJECT_COLORS = (3, 5, 6, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_MAX_ATTEMPTS = 500
PAIR_COUNT_RANGE = (2, 4)
MIN_SIDE_BY_PAIR_COUNT = {2: 17, 3: 21, 4: 25}
MAX_SIDE = 31
PAIR_PLACEMENT_TRIES = 200
SHAPE_TRIES = 200


def _majority_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.reshape(-1))
    return int(counts.most_common(1)[0][0])


def _edge_positions(shape: tuple[int, int]) -> dict[str, Coord]:
    h, w = map(int, shape)
    return {
        "top": (0, w // 2),
        "bottom": (h - 1, w // 2),
        "left": (h // 2, 0),
        "right": (h // 2, w - 1),
    }


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int, banned: set[Coord]) -> list[list[Coord]]:
    h, w = grid.shape
    seen: set[Coord] = set()
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            pos = (int(r), int(c))
            if pos in banned or pos in seen or int(grid[r, c]) != int(color):
                continue
            q = deque([pos])
            seen.add(pos)
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    nxt = (nr, nc)
                    if nxt in banned or nxt in seen or int(grid[nr, nc]) != int(color):
                        continue
                    seen.add(nxt)
                    q.append(nxt)
            comps.append(sorted(comp))
    return comps


def _shape_key(coords: list[Coord]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _bbox_shape(coords: list[Coord]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _anchor_pixel(coords: list[Coord]) -> Coord:
    area = len(coords)
    sum_r = sum(r for r, _ in coords)
    sum_c = sum(c for _, c in coords)
    return min(
        coords,
        key=lambda p: (
            (p[0] * area - sum_r) ** 2 + (p[1] * area - sum_c) ** 2,
            p[0],
            p[1],
        ),
    )


def _line_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _route_cells(start: Coord, end: Coord, vertical_first: bool) -> list[Coord]:
    if vertical_first:
        corner = (end[0], start[1])
    else:
        corner = (start[0], end[1])
    first = _line_cells(start, corner)
    second = _line_cells(corner, end)
    return first + second[1:]


def _direction_from_objects(first: dict, second: dict) -> tuple[str, bool]:
    dr = int(second["sum_r"] - first["sum_r"])
    dc = int(second["sum_c"] - first["sum_c"])
    if abs(dr) > abs(dc):
        return ("bottom" if dr > 0 else "top"), True
    return ("right" if dc > 0 else "left"), False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shape objects and bridge each pair with an L-shaped line whose
    color is read from the edge marker matching the pair's dominant direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _majority_color(g)

    edge_positions = _edge_positions(g.shape)
    edge_colors = {edge: int(g[pos]) for edge, pos in edge_positions.items()}
    banned = set(edge_positions.values())

    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        comps = _components_of_color(g, color, banned)
        for comp in comps:
            area = len(comp)
            assert 3 <= area <= 6, "objects must have 3-6 pixels"
            objects.append(
                {
                    "color": int(color),
                    "coords": comp,
                    "shape_key": _shape_key(comp),
                    "sum_r": sum(r for r, _ in comp),
                    "sum_c": sum(c for _, c in comp),
                    "anchor": _anchor_pixel(comp),
                }
            )

    assert objects, "expected at least one object"

    groups: dict[tuple[int, tuple[Coord, ...]], list[dict]] = {}
    for obj in objects:
        key = (int(obj["color"]), obj["shape_key"])
        groups.setdefault(key, []).append(obj)

    assert all(len(group) == 2 for group in groups.values()), "each shape must appear exactly twice"

    out = g.copy()
    for key in sorted(groups):
        first, second = sorted(
            groups[key],
            key=lambda obj: (int(obj["sum_r"]), int(obj["sum_c"]), obj["anchor"]),
        )
        edge, vertical_first = _direction_from_objects(first, second)
        route = _route_cells(first["anchor"], second["anchor"], vertical_first)
        line_color = edge_colors[edge]
        for r, c in route[1:-1]:
            if int(g[r, c]) == int(bg):
                out[r, c] = int(line_color)
    return out


def _normalize_cells(cells: set[Coord]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_polyomino(area: int) -> tuple[Coord, ...]:
    cells = {(0, 0)}
    while len(cells) < area:
        r, c = random.choice(tuple(cells))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for nbr in nbrs:
            if nbr not in cells:
                cells.add(nbr)
                break
    return _normalize_cells(cells)


def _shape_info(shape_key: tuple[Coord, ...]) -> dict:
    coords = [(int(r), int(c)) for r, c in shape_key]
    height, width = _bbox_shape(coords)
    anchor = _anchor_pixel(coords)
    return {
        "shape_key": shape_key,
        "coords": coords,
        "height": int(height),
        "width": int(width),
        "anchor": (int(anchor[0]), int(anchor[1])),
    }


def _random_odd(lo: int, hi: int) -> int:
    choices = [v for v in range(lo, hi + 1) if v % 2 == 1]
    return int(random.choice(choices))


def _sample_shapes(n_pairs: int) -> list[dict]:
    shapes: list[dict] = []
    used: set[tuple[Coord, ...]] = set()
    for _ in range(n_pairs):
        for _attempt in range(SHAPE_TRIES):
            area = random.randint(3, 6)
            shape_key = _random_polyomino(area)
            if shape_key in used:
                continue
            used.add(shape_key)
            shapes.append(_shape_info(shape_key))
            break
        else:
            raise RuntimeError("failed to sample unique shapes")
    return shapes


def _sample_directions(n_pairs: int) -> list[str]:
    if n_pairs == 2:
        dirs = [random.choice(["up", "down"]), random.choice(["left", "right"])]
    elif n_pairs == 3:
        dirs = [
            random.choice(["up", "down"]),
            random.choice(["left", "right"]),
            random.choice(["up", "down", "left", "right"]),
        ]
    else:
        dirs = ["up", "down", "left", "right"]
    random.shuffle(dirs)
    return dirs


def _sample_delta(direction: str, height: int, width: int, h: int, w: int) -> tuple[int, int] | None:
    max_abs_dr = h - height - 3
    max_abs_dc = w - width - 3
    if direction in ("left", "right"):
        if max_abs_dc < 4:
            return None
        abs_dc = random.randint(4, max_abs_dc)
        use_tie = random.random() < 0.35 and max_abs_dr >= abs_dc
        if use_tie:
            abs_dr = abs_dc
        else:
            max_secondary = min(max_abs_dr, abs_dc - 1)
            abs_dr = 0 if max_secondary <= 0 else random.randint(0, max_secondary)
        dr = 0 if abs_dr == 0 else random.choice([-abs_dr, abs_dr])
        dc = abs_dc if direction == "right" else -abs_dc
        return int(dr), int(dc)

    if max_abs_dr < 4:
        return None
    abs_dr = random.randint(4, max_abs_dr)
    max_secondary = min(max_abs_dc, abs_dr - 1)
    abs_dc = 0 if max_secondary <= 0 else random.randint(0, max_secondary)
    dc = 0 if abs_dc == 0 else random.choice([-abs_dc, abs_dc])
    dr = abs_dr if direction == "down" else -abs_dr
    return int(dr), int(dc)


def _translate(coords: list[Coord], dr: int, dc: int) -> set[Coord]:
    return {(int(r + dr), int(c + dc)) for r, c in coords}


def _placement_bounds(height: int, width: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int, int, int]:
    row_lo = 1 + max(0, -dr)
    row_hi = h - height - 2 - max(0, dr)
    col_lo = 1 + max(0, -dc)
    col_hi = w - width - 2 - max(0, dc)
    return int(row_lo), int(row_hi), int(col_lo), int(col_hi)


def _well_separated(a: set[Coord], b: set[Coord], pad: int = 1) -> bool:
    for r, c in a:
        for nr in range(r - pad, r + pad + 1):
            for nc in range(c - pad, c + pad + 1):
                if (nr, nc) in b:
                    return False
    return True


def _trim_visible_route(route: list[Coord], first_cells: set[Coord], second_cells: set[Coord]) -> list[Coord] | None:
    if len(route) <= 2:
        return None
    start = 1
    end = len(route) - 2
    while start <= end and route[start] in first_cells:
        start += 1
    while end >= start and route[end] in second_cells:
        end -= 1
    if start > end:
        return None
    visible = route[start : end + 1]
    if any(cell in first_cells or cell in second_cells for cell in visible):
        return None
    return visible


def _direction_to_edge(direction: str) -> str:
    return {
        "up": "top",
        "down": "bottom",
        "left": "left",
        "right": "right",
    }[direction]


def _build_expected(inp: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = inp.copy()
    edge_positions = _edge_positions(inp.shape)
    edge_colors = {edge: int(inp[pos]) for edge, pos in edge_positions.items()}
    for pair in pairs:
        color = edge_colors[_direction_to_edge(pair["direction"])]
        for r, c in pair["visible_route"]:
            out[r, c] = int(color)
    return out


def _place_pair(shape: dict, direction: str, blocked: set[Coord], h: int, w: int) -> dict | None:
    delta = _sample_delta(direction, shape["height"], shape["width"], h, w)
    if delta is None:
        return None
    dr, dc = delta

    row_lo, row_hi, col_lo, col_hi = _placement_bounds(shape["height"], shape["width"], h, w, dr, dc)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    rel_coords = shape["coords"]
    anchor_rel = shape["anchor"]

    for _ in range(PAIR_PLACEMENT_TRIES):
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        first_cells = _translate(rel_coords, r0, c0)
        second_cells = _translate(rel_coords, r0 + dr, c0 + dc)
        if not _well_separated(first_cells, second_cells, pad=1):
            continue
        if not _well_separated(first_cells, blocked, pad=1):
            continue
        if not _well_separated(second_cells, blocked, pad=1):
            continue

        start = (r0 + anchor_rel[0], c0 + anchor_rel[1])
        end = (start[0] + dr, start[1] + dc)
        vertical_first = direction in ("up", "down")
        route = _route_cells(start, end, vertical_first)
        visible = _trim_visible_route(route, first_cells, second_cells)
        if visible is None or len(visible) < 3:
            continue
        if any(cell in blocked or cell in first_cells or cell in second_cells for cell in visible):
            continue

        return {
            "direction": direction,
            "first_cells": first_cells,
            "second_cells": second_cells,
            "visible_route": visible,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    side_min = MIN_SIDE_BY_PAIR_COUNT[n_pairs]
    h = _random_odd(side_min, MAX_SIDE)
    w = _random_odd(side_min, MAX_SIDE)

    inp = np.full((h, w), BG, dtype=int)
    edge_positions = _edge_positions(inp.shape)
    inp[edge_positions["top"]] = TOP_MARKER
    inp[edge_positions["bottom"]] = BOTTOM_MARKER
    inp[edge_positions["left"]] = LEFT_MARKER
    inp[edge_positions["right"]] = RIGHT_MARKER

    object_color = random.choice(OBJECT_COLORS)
    blocked = set(edge_positions.values())

    shapes = _sample_shapes(n_pairs)
    directions = _sample_directions(n_pairs)
    pairs: list[dict] = []

    for shape, direction in zip(shapes, directions):
        pair = _place_pair(shape, direction, blocked, h, w)
        if pair is None:
            return None
        pairs.append(pair)
        blocked |= pair["first_cells"]
        blocked |= pair["second_cells"]
        blocked |= set(pair["visible_route"])

    for pair in pairs:
        for r, c in pair["first_cells"] | pair["second_cells"]:
            inp[r, c] = int(object_color)

    expected = _build_expected(inp, pairs)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    sig_groups: dict[tuple[Coord, ...], int] = {}
    marker_cells = set(edge_positions.values())
    for comp in _components_of_color(inp, object_color, marker_cells):
        sig_groups[_shape_key(comp)] = sig_groups.get(_shape_key(comp), 0) + 1
    if sorted(sig_groups.values()) != [2] * n_pairs:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
