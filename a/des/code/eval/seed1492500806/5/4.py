from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

START = GREEN
END = RED
PATH = GREY
ARROW = AZURE
CARGO_COLORS = (BLUE, YELLOW, MAGENTA, ORANGE)
PATHLIKE = frozenset({START, END, PATH, *CARGO_COLORS})

GRID_SIDE_RANGE = (18, 31)
CONVEYOR_COUNT_SMALL = (1, 2)
CONVEYOR_COUNT_LARGE = (1, 3)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 120
INNER_MARGIN = 1
LENGTH_RANGE = (3, 8)
THIRD_SEGMENT_RANGE = (3, 7)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _trace_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [points[0]]
    for (r0, c0), (r1, c1) in zip(points[:-1], points[1:], strict=True):
        dr = _sign(r1 - r0)
        dc = _sign(c1 - c0)
        steps = abs(r1 - r0) + abs(c1 - c0)
        assert (dr == 0) != (dc == 0)
        for step in range(1, steps + 1):
            cells.append((r0 + dr * step, c0 + dc * step))
    return cells


def _segment_arrows(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arrows: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points[:-2], points[1:-1], strict=True):
        dr = _sign(r1 - r0)
        dc = _sign(c1 - c0)
        arrows.append((r1 + dr, c1 + dc))
    return arrows


def _ordered_path(component: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells = set(component)
    starts = [pos for pos in component if int(grid[pos]) == START]
    ends = [pos for pos in component if int(grid[pos]) == END]
    assert len(starts) == 1, "each conveyor needs exactly one green start"
    assert len(ends) == 1, "each conveyor needs exactly one red end"
    start = starts[0]
    end = ends[0]

    degree = {}
    for r, c in component:
        degree[(r, c)] = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, h, w))

    assert degree[start] == 1 and degree[end] == 1
    for pos, deg in degree.items():
        if pos not in {start, end}:
            assert deg == 2, "conveyor path must be a single non-branching chain"

    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        nxts = [pos for pos in _neighbors4(cur[0], cur[1], h, w) if pos in cells and pos != prev]
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(component)
    return order


def _move_component(order: list[tuple[int, int]], grid: np.ndarray, out: np.ndarray) -> None:
    cargos: list[int] = []
    cargo_positions: list[tuple[int, int]] = []
    for pos in order[1:-1]:
        value = int(grid[pos])
        if value in CARGO_COLORS:
            cargos.append(value)
            cargo_positions.append(pos)

    for r, c in cargo_positions:
        out[r, c] = PATH

    if not cargos:
        return

    final_slots = order[len(order) - 1 - len(cargos) : -1]
    for color, (r, c) in zip(cargos, final_slots, strict=True):
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each cargo along its non-branching conveyor toward the red endpoint,
    processing nearer cargos first so they compact into a queue just before the end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    mask = np.isin(g, tuple(PATHLIKE))
    comps = _components(mask)
    assert comps, "expected at least one conveyor"

    out = g.copy()
    for comp in comps:
        order = _ordered_path(comp, g)
        _move_component(order, g, out)
    return out


def _transform_point(point: tuple[int, int], rot: int, flip: bool) -> tuple[int, int]:
    r, c = point
    if flip:
        c = -c
    rot %= 4
    if rot == 0:
        return r, c
    if rot == 1:
        return c, -r
    if rot == 2:
        return -r, -c
    return -c, r


def _normalize_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    min_r = min(rows)
    min_c = min(cols)
    return [(r - min_r, c - min_c) for r, c in points]


def _sample_polyline() -> list[tuple[int, int]]:
    shape = random.choice(("elbow", "zigzag"))
    a = random.randint(*LENGTH_RANGE)
    b = random.randint(*LENGTH_RANGE)
    if shape == "elbow":
        return [(0, 0), (0, a), (b, a)]
    c = random.randint(*THIRD_SEGMENT_RANGE)
    return [(0, 0), (0, a), (b, a), (b, a + c)]


def _make_local_conveyor() -> dict | None:
    points = _sample_polyline()
    rot = random.randrange(4)
    flip = bool(random.randrange(2))
    points = [_transform_point(point, rot, flip) for point in points]

    raw_arrows = _segment_arrows(points)
    all_points = points + raw_arrows
    min_r = min(r for r, _ in all_points)
    min_c = min(c for _, c in all_points)
    points = [(r - min_r, c - min_c) for r, c in points]
    arrows = [(r - min_r, c - min_c) for r, c in raw_arrows]

    path_order = _trace_points(points)
    bends = set(points[1:-1])
    if len(set(path_order)) != len(path_order):
        return None
    if len(set(arrows)) != len(arrows):
        return None
    if set(arrows) & set(path_order):
        return None

    cargo_candidates = [i for i, pos in enumerate(path_order[1:-1], start=1) if pos not in bends]
    if len(cargo_candidates) < 2:
        return None

    max_cargo = min(4, len(cargo_candidates))
    if max_cargo < 1:
        return None
    if max_cargo >= 2 and random.random() < 0.5:
        cargo_count = random.randint(2, max_cargo)
    else:
        cargo_count = 1

    final_start = len(path_order) - 1 - cargo_count
    choices = cargo_candidates[:]
    random.shuffle(choices)
    cargo_indices = sorted(choices[:cargo_count])
    if not any(index < final_start for index in cargo_indices):
        return None

    cargo_colors = [random.choice(CARGO_COLORS) for _ in range(cargo_count)]
    height = 1 + max(r for r, _ in path_order + arrows)
    width = 1 + max(c for _, c in path_order + arrows)
    return {
        "points": points,
        "path_order": path_order,
        "arrows": arrows,
        "cargo_indices": cargo_indices,
        "cargo_colors": cargo_colors,
        "height": height,
        "width": width,
    }


def _dilate(coords: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked.add((rr, cc))
    return blocked


def _paint_conveyor(
    grid: np.ndarray,
    spec: dict,
    origin_r: int,
    origin_c: int,
) -> None:
    path_order = [(origin_r + r, origin_c + c) for r, c in spec["path_order"]]
    arrows = [(origin_r + r, origin_c + c) for r, c in spec["arrows"]]

    for r, c in path_order[1:-1]:
        grid[r, c] = PATH
    sr, sc = path_order[0]
    er, ec = path_order[-1]
    grid[sr, sc] = START
    grid[er, ec] = END

    for r, c in arrows:
        grid[r, c] = ARROW

    for index, color in zip(spec["cargo_indices"], spec["cargo_colors"], strict=True):
        r, c = path_order[index]
        grid[r, c] = int(color)


def _try_place_conveyor(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    spec: dict,
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    max_r0 = h - spec["height"] - INNER_MARGIN
    max_c0 = w - spec["width"] - INNER_MARGIN
    if max_r0 < INNER_MARGIN or max_c0 < INNER_MARGIN:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(INNER_MARGIN, max_r0)
        c0 = random.randint(INNER_MARGIN, max_c0)
        all_cells = {
            (r0 + r, c0 + c) for r, c in spec["path_order"]
        } | {
            (r0 + r, c0 + c) for r, c in spec["arrows"]
        }
        if any(cell in blocked for cell in all_cells):
            continue
        _paint_conveyor(grid, spec, r0, c0)
        return _dilate(all_cells, h, w)
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    count_range = CONVEYOR_COUNT_LARGE if min(h, w) >= 24 else CONVEYOR_COUNT_SMALL
    conveyor_count = random.randint(*count_range)
    blocked: set[tuple[int, int]] = set()

    for _ in range(conveyor_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            spec = _make_local_conveyor()
            if spec is None:
                continue
            new_blocked = _try_place_conveyor(grid, blocked, spec)
            if new_blocked is None:
                continue
            blocked |= new_blocked
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_RETRIES} attempts")
