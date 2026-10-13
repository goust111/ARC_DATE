import random
from collections import Counter, defaultdict

import numpy as np


BLUE = 1
RED = 2
GREEN = 3

SEED_COLORS = (BLUE, RED, GREEN)
PRIORITY = {GREEN: 0, BLUE: 1, RED: 2}
BG_CHOICES = (0, 4, 5, 6, 7, 8, 9)

RNG = random.Random()

GRID_H_RANGE = (12, 23)
GRID_W_RANGE = (12, 23)
PAIR_COUNT_RANGE = (2, 5)
MAX_PAIRS_PER_COLOR = 2
PAIR_ATTEMPTS = 200
GEN_ATTEMPTS = 400

SCENARIOS = ("normal", "cross", "edge", "diag")
UNITS = ((0, 1), (1, 0), (1, 1), (1, -1))
DIAGONAL_UNITS = ((1, 1), (1, -1))


Point = tuple[int, int]
Pair = tuple[int, Point, Point]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _chebyshev(a: Point, b: Point) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _segment_info(a: Point, b: Point) -> tuple[int, Point, str]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    if dr == 0 and dc != 0:
        return abs(dc), (0, _sign(dc)), "horizontal"
    if dc == 0 and dr != 0:
        return abs(dr), (_sign(dr), 0), "vertical"
    if dr != 0 and abs(dr) == abs(dc):
        return abs(dr), (_sign(dr), _sign(dc)), "diagonal"
    raise AssertionError("paired seeds must align horizontally, vertically, or diagonally")


def _cw90(vec: Point) -> Point:
    dr, dc = vec
    return (dc, -dr)


def _pair_branches(a: Point, b: Point, shape: tuple[int, int]) -> tuple[list[Point], list[Point], bool, str]:
    h, w = shape
    dist, unit, orientation = _segment_info(a, b)
    grow_a = _cw90(unit)
    grow_b = (-grow_a[0], -grow_a[1])

    branches: list[list[Point]] = []
    truncated = False
    for start, grow in ((a, grow_a), (b, grow_b)):
        sr, sc = start
        dr, dc = grow
        branch: list[Point] = []
        for step in range(1, dist + 1):
            nr = sr + dr * step
            nc = sc + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                truncated = True
                break
            branch.append((int(nr), int(nc)))
        branches.append(branch)

    return branches[0], branches[1], truncated, orientation


def _pair_points(points: list[Point]) -> list[tuple[Point, Point]]:
    pts = sorted((int(r), int(c)) for r, c in points)
    assert len(pts) % 2 == 0 and pts, "same-color seeds must appear in non-empty even counts"

    remaining = list(range(len(pts)))
    pairs: list[tuple[Point, Point]] = []
    while remaining:
        nearest: dict[int, list[int]] = {}
        nearest_dist: dict[int, int] = {}
        for i in remaining:
            options = sorted(
                (_chebyshev(pts[i], pts[j]), pts[j][0], pts[j][1], j)
                for j in remaining
                if j != i
            )
            assert options, "cannot pair a singleton seed"
            best_dist = int(options[0][0])
            nearest[i] = [int(j) for dist, _r, _c, j in options if int(dist) == best_dist]
            nearest_dist[i] = best_dist

        mutual: list[tuple[int, int, int, int, int]] = []
        for i in remaining:
            if len(nearest[i]) != 1:
                continue
            j = nearest[i][0]
            if i < j and len(nearest[j]) == 1 and nearest[j][0] == i:
                mutual.append((nearest_dist[i], pts[i][0], pts[i][1], i, j))

        if mutual:
            _dist, _r, _c, i_sel, j_sel = min(mutual)
        else:
            global_edges = sorted(
                (_chebyshev(pts[i], pts[j]), pts[i][0], pts[i][1], pts[j][0], pts[j][1], i, j)
                for idx, i in enumerate(remaining)
                for j in remaining[idx + 1 :]
            )
            _dist, _r0, _c0, _r1, _c1, i_sel, j_sel = global_edges[0]

        pairs.append((pts[i_sel], pts[j_sel]))
        remaining.remove(i_sel)
        remaining.remove(j_sel)

    return pairs


def _extract_pairs(grid: np.ndarray) -> list[Pair]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    seed_positions = np.argwhere(g != bg)
    assert seed_positions.size > 0, "expected at least one seed"
    colors = {int(g[r, c]) for r, c in seed_positions}
    assert colors <= set(SEED_COLORS), "input must only use red, blue, and green seeds"

    for r, c in seed_positions:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
                    raise AssertionError("seeds must stay isolated single pixels")

    by_color: dict[int, list[Point]] = defaultdict(list)
    for r, c in seed_positions:
        by_color[int(g[r, c])].append((int(r), int(c)))

    out: list[Pair] = []
    for color in sorted(by_color):
        for a, b in _pair_points(by_color[color]):
            _segment_info(a, b)
            out.append((int(color), a, b))
    out.sort(key=lambda item: (item[0], item[1], item[2]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    pairs = _extract_pairs(g)

    out = g.copy()
    seed_mask = g != bg
    paint_priority = np.full((h, w), -1, dtype=int)
    paint_color = np.full((h, w), bg, dtype=int)

    for color, a, b in pairs:
        branch_a, branch_b, _truncated, _orientation = _pair_branches(a, b, g.shape)
        for branch in (branch_a, branch_b):
            for r, c in branch:
                if seed_mask[r, c]:
                    continue
                prio = PRIORITY[color]
                if prio > int(paint_priority[r, c]):
                    paint_priority[r, c] = prio
                    paint_color[r, c] = color

    mask = (~seed_mask) & (paint_priority >= 0)
    out[mask] = paint_color[mask]
    return out


def _canonical_pairs(pairs: list[Pair]) -> list[tuple[int, Point, Point]]:
    canon = []
    for color, a, b in pairs:
        p0, p1 = sorted((a, b))
        canon.append((int(color), p0, p1))
    canon.sort()
    return canon


def _all_seed_cells(pairs: list[Pair]) -> set[Point]:
    cells: set[Point] = set()
    for _color, a, b in pairs:
        cells.add(a)
        cells.add(b)
    return cells


def _line_hits(pairs: list[Pair], shape: tuple[int, int]) -> dict[Point, list[int]]:
    hits: dict[Point, list[int]] = defaultdict(list)
    for color, a, b in pairs:
        branch_a, branch_b, _truncated, _orientation = _pair_branches(a, b, shape)
        for cell in branch_a + branch_b:
            hits[cell].append(int(color))
    return hits


def _pair_stats(pairs: list[Pair], shape: tuple[int, int]) -> dict[str, object]:
    hits = _line_hits(pairs, shape)
    orientations = Counter()
    has_truncation = False
    for _color, a, b in pairs:
        _branch_a, _branch_b, truncated, orientation = _pair_branches(a, b, shape)
        orientations[orientation] += 1
        has_truncation = has_truncation or truncated

    cross_cells = [cell for cell, colors in hits.items() if len(colors) >= 2]
    color_cross_cells = [cell for cell, colors in hits.items() if len(set(colors)) >= 2]
    return {
        "has_truncation": has_truncation,
        "has_cross": bool(cross_cells),
        "has_color_cross": bool(color_cross_cells),
        "orientations": orientations,
        "cross_cells": cross_cells,
        "color_cross_cells": color_cross_cells,
    }


def _isolated_seed(cell: Point, occupied: set[Point]) -> bool:
    for r, c in occupied:
        if max(abs(cell[0] - r), abs(cell[1] - c)) <= 1:
            return False
    return True


def _same_color_safe(color: int, a: Point, b: Point, pairs: list[Pair]) -> bool:
    dist = _chebyshev(a, b)
    for other_color, x, y in pairs:
        if other_color != color:
            continue
        other_dist = _chebyshev(x, y)
        for p in (a, b):
            for q in (x, y):
                if _chebyshev(p, q) <= max(dist, other_dist):
                    return False
    return True


def _pair_ok(
    pair: Pair,
    pairs: list[Pair],
    shape: tuple[int, int],
    *,
    allow_overlap: bool,
    require_overlap: bool,
    require_truncation: bool | None,
) -> bool:
    color, a, b = pair
    h, w = shape
    if not (0 <= a[0] < h and 0 <= a[1] < w and 0 <= b[0] < h and 0 <= b[1] < w):
        return False
    if a == b:
        return False
    try:
        _dist, _unit, _orientation = _segment_info(a, b)
    except AssertionError:
        return False

    occupied = _all_seed_cells(pairs)
    if a in occupied or b in occupied:
        return False
    if not _isolated_seed(a, occupied) or not _isolated_seed(b, occupied):
        return False
    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
        return False
    if not _same_color_safe(color, a, b, pairs):
        return False

    branch_a, branch_b, truncated, _orientation = _pair_branches(a, b, shape)
    line_cells = set(branch_a + branch_b)
    if not line_cells:
        return False
    if line_cells & occupied:
        return False
    if require_truncation is not None and truncated != require_truncation:
        return False

    existing_hits = _line_hits(pairs, shape)
    overlap_cells = line_cells & set(existing_hits)
    if require_overlap and not overlap_cells:
        return False
    if not allow_overlap and overlap_cells:
        return False
    if allow_overlap and len(overlap_cells) > 2:
        return False

    return True


def _max_dist_for_unit(shape: tuple[int, int], unit: Point) -> int:
    h, w = shape
    if unit == (0, 1):
        return w - 1
    if unit == (1, 0):
        return h - 1
    return min(h, w) - 1


def _random_endpoints(shape: tuple[int, int], unit: Point, dist: int, rng: random.Random) -> tuple[Point, Point]:
    h, w = shape
    if unit == (0, 1):
        r = rng.randrange(h)
        c = rng.randrange(w - dist)
        return (r, c), (r, c + dist)
    if unit == (1, 0):
        r = rng.randrange(h - dist)
        c = rng.randrange(w)
        return (r, c), (r + dist, c)
    if unit == (1, 1):
        r = rng.randrange(h - dist)
        c = rng.randrange(w - dist)
        return (r, c), (r + dist, c + dist)
    if unit == (1, -1):
        r = rng.randrange(h - dist)
        c = rng.randrange(dist, w)
        return (r, c), (r + dist, c - dist)
    raise AssertionError("unknown unit")


def _sample_color(counts: Counter, rng: random.Random) -> int:
    available = [color for color in SEED_COLORS if int(counts[color]) < MAX_PAIRS_PER_COLOR]
    assert available, "no colors left for another pair"
    repeated = [color for color in available if int(counts[color]) > 0]
    if repeated and rng.random() < 0.45:
        return int(rng.choice(repeated))
    return int(rng.choice(available))


def _sample_pair(
    shape: tuple[int, int],
    pairs: list[Pair],
    counts: Counter,
    rng: random.Random,
    *,
    color: int | None = None,
    units: tuple[Point, ...] = UNITS,
    allow_overlap: bool = False,
    require_overlap: bool = False,
    require_truncation: bool | None = None,
) -> Pair | None:
    for _ in range(PAIR_ATTEMPTS):
        use_color = int(color if color is not None else _sample_color(counts, rng))
        unit = rng.choice(units)
        max_dist = min(_max_dist_for_unit(shape, unit), 6)
        if max_dist < 2:
            continue
        dist = rng.randint(2, max_dist)
        a, b = _random_endpoints(shape, unit, dist, rng)
        pair = (use_color, a, b)
        if _pair_ok(
            pair,
            pairs,
            shape,
            allow_overlap=allow_overlap,
            require_overlap=require_overlap,
            require_truncation=require_truncation,
        ):
            return pair
    return None


def _make_cross_block(shape: tuple[int, int], rng: random.Random) -> list[Pair] | None:
    h, w = shape
    max_d = min(4, (h - 4) // 2, (w - 1) // 2)
    if max_d < 2:
        return None

    colors = rng.sample(SEED_COLORS, 2)
    for _ in range(PAIR_ATTEMPTS):
        d = rng.randint(2, max_d)
        min_r, max_r = d, h - d - 3
        min_c, max_c = max(0, d - 2), w - d - 3
        if min_r > max_r or min_c > max_c:
            continue
        r = rng.randint(min_r, max_r)
        c = rng.randint(min_c, max_c)

        pair1 = (int(colors[0]), (r, c), (r, c + d))
        pair2 = (int(colors[1]), (r + 2, c + 2), (r + 2 + d, c + 2))
        if not _pair_ok(
            pair1,
            [],
            shape,
            allow_overlap=False,
            require_overlap=False,
            require_truncation=False,
        ):
            continue
        if not _pair_ok(
            pair2,
            [pair1],
            shape,
            allow_overlap=True,
            require_overlap=True,
            require_truncation=False,
        ):
            continue
        stats = _pair_stats([pair1, pair2], shape)
        if stats["has_color_cross"] and not stats["has_truncation"]:
            return [pair1, pair2]
    return None


def _render_grid(shape: tuple[int, int], bg: int, pairs: list[Pair]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for color, a, b in pairs:
        grid[a] = int(color)
        grid[b] = int(color)
    return grid


def _scenario_ok(scenario: str, pairs: list[Pair], shape: tuple[int, int]) -> bool:
    stats = _pair_stats(pairs, shape)
    if scenario == "normal":
        return (
            not stats["has_truncation"]
            and not stats["has_cross"]
            and int(stats["orientations"]["diagonal"]) == 0
        )
    if scenario == "cross":
        return bool(stats["has_color_cross"]) and not stats["has_truncation"]
    if scenario == "edge":
        return stats["has_truncation"] and not stats["has_cross"]
    if scenario == "diag":
        return (
            int(stats["orientations"]["diagonal"]) >= 1
            and not stats["has_truncation"]
            and not stats["has_cross"]
        )
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    shape = (h, w)
    bg = int(rng.choice(BG_CHOICES))
    scenario = rng.choice(SCENARIOS)
    target_pairs = rng.randint(*PAIR_COUNT_RANGE)

    pairs: list[Pair] = []
    counts: Counter = Counter()
    filler_units = UNITS

    if scenario == "cross":
        block = _make_cross_block(shape, rng)
        if block is None:
            return None
        for pair in block:
            pairs.append(pair)
            counts[pair[0]] += 1
        filler_units = ((0, 1), (1, 0))
    elif scenario == "edge":
        pair = _sample_pair(
            shape,
            pairs,
            counts,
            rng,
            require_truncation=True,
            allow_overlap=False,
            require_overlap=False,
        )
        if pair is None:
            return None
        pairs.append(pair)
        counts[pair[0]] += 1
        filler_units = ((0, 1), (1, 0))
    elif scenario == "diag":
        pair = _sample_pair(
            shape,
            pairs,
            counts,
            rng,
            units=DIAGONAL_UNITS,
            require_truncation=False,
            allow_overlap=False,
            require_overlap=False,
        )
        if pair is None:
            return None
        pairs.append(pair)
        counts[pair[0]] += 1
        filler_units = ((0, 1), (1, 0))
    else:
        filler_units = ((0, 1), (1, 0))

    target_pairs = max(target_pairs, len(pairs))
    while len(pairs) < target_pairs:
        pair = _sample_pair(
            shape,
            pairs,
            counts,
            rng,
            units=filler_units,
            require_truncation=False,
            allow_overlap=False,
            require_overlap=False,
        )
        if pair is None:
            return None
        pairs.append(pair)
        counts[pair[0]] += 1

    if not _scenario_ok(scenario, pairs, shape):
        return None

    grid = _render_grid(shape, bg, pairs)
    try:
        detected = _extract_pairs(grid)
        if _canonical_pairs(detected) != _canonical_pairs(pairs):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
