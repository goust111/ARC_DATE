import random
from typing import Iterable

import numpy as np


BG = 0
OBSTACLE = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
PAIR_COUNT_RANGE = (3, 5)
PAIR_PLACEMENT_TRIES = 200
EXTRA_OBSTACLE_COUNT = (0, 2)
EXTRA_OBSTACLE_SIZE = (1, 4)
ANCHOR_OBSTACLE_SIZE = (1, 4)
GEN_RETRIES = 400


Coord = tuple[int, int]


def _priority(cell: Coord) -> tuple[int, int, int]:
    return (cell[0] + cell[1], cell[0], cell[1])


def _corners(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    return (a[0], b[1]), (b[0], a[1])


def _segment(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segment endpoints must be aligned")


def _path_cells(a: Coord, corner: Coord, b: Coord) -> list[Coord]:
    first = _segment(a, corner)
    second = _segment(corner, b)
    return first + second[1:]


def _chosen_corner(a: Coord, b: Coord, obstacle_mask: np.ndarray) -> Coord:
    c0, c1 = _corners(a, b)
    blocked0 = bool(obstacle_mask[c0])
    blocked1 = bool(obstacle_mask[c1])
    assert not (blocked0 and blocked1), "both candidate corners cannot be blocked"
    if blocked0:
        return c1
    if blocked1:
        return c0
    return c0 if _priority(c0) < _priority(c1) else c1


def _pair_order(grid: np.ndarray) -> list[tuple[Coord, int, Coord, Coord]]:
    pairs: list[tuple[Coord, int, Coord, Coord]] = []
    for color in map(int, np.unique(grid)):
        if color in (BG, OBSTACLE):
            continue
        pts = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        assert len(pts) == 2, f"color {color} must appear exactly twice"
        pts.sort()
        a, b = pts
        assert a[0] != b[0] and a[1] != b[1], "paired markers must require a turn"
        pairs.append((a, color, a, b))
    pairs.sort(key=lambda item: item[0])
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair with the only valid L-shaped turn.

    Input structure:
    - The background is black.
    - Several non-gray colors appear as isolated single-pixel markers, with each such color appearing exactly twice.
    - Gray pixels form obstacle components; markers of one pair are never in the same row or the same column.

    Rule:
    - For each same-color pair there are exactly two possible 1-pixel-wide L paths: row-first via (rA, cB) or column-first via (rB, cA).
    - If exactly one candidate corner is gray, use the other corner. If neither candidate corner is gray, use the corner with the smaller row-plus-column value.
    - Draw the full path from one marker to the corner and from the corner to the other marker, including the endpoints.
    - Draw pairs in the order of their first marker's row-major appearance; later paths overwrite earlier crossings.

    Key insight:
    Gray cells are not walls to route around; they only disqualify one of the two possible turning pixels for a same-color pair.
    """
    g = np.asarray(grid, dtype=int)
    obstacle_mask = g == OBSTACLE
    out = g.copy()

    for _first, color, a, b in _pair_order(g):
        corner = _chosen_corner(a, b, obstacle_mask)
        for r, c in _path_cells(a, corner, b):
            out[r, c] = color

    return out


def _sample_branches(pair_count: int, rng: random.Random) -> list[int]:
    base = [0, 1, 2]
    rng.shuffle(base)
    branches = base[: min(pair_count, 3)]
    while len(branches) < pair_count:
        branches.append(rng.randrange(3))
    rng.shuffle(branches)
    return branches


def _planned_choice(a: Coord, b: Coord, branch: int) -> tuple[Coord, Coord | None]:
    c0, c1 = _corners(a, b)
    if branch == 0:
        return c0, c1
    if branch == 1:
        return c1, c0
    return (c0, None) if _priority(c0) < _priority(c1) else (c1, None)


def _close_to_any(cell: Coord, others: Iterable[Coord]) -> bool:
    return any(max(abs(cell[0] - r), abs(cell[1] - c)) <= 1 for r, c in others)


def _grow_component(
    anchor: Coord,
    h: int,
    w: int,
    forbidden: set[Coord],
    occupied: set[Coord],
    rng: random.Random,
    size_range: tuple[int, int],
) -> set[Coord]:
    target = rng.randint(size_range[0], size_range[1])
    shape = {anchor}
    while len(shape) < target:
        frontier: list[Coord] = []
        for r, c in shape:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                cell = (nr, nc)
                if cell in shape or cell in occupied or cell in forbidden:
                    continue
                frontier.append(cell)
        if not frontier:
            break
        shape.add(rng.choice(frontier))
    return shape


def _expected_output(grid: np.ndarray, pair_specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in sorted(pair_specs, key=lambda item: min(item["a"], item["b"])):
        for r, c in spec["path"]:
            out[r, c] = spec["color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    max_pairs = min(PAIR_COUNT_RANGE[1], len(MARKER_COLORS), 4 if min(h, w) < 15 else 5)
    pair_count = rng.randint(PAIR_COUNT_RANGE[0], max_pairs)
    colors = rng.sample(MARKER_COLORS, pair_count)
    branches = _sample_branches(pair_count, rng)

    pair_specs: list[dict] = []
    marker_cells: set[Coord] = set()
    path_cells: set[Coord] = set()
    blocked_corners: set[Coord] = set()
    all_candidate_corners: set[Coord] = set()

    for color, branch in zip(colors, branches):
        placed = False
        for _ in range(PAIR_PLACEMENT_TRIES):
            a = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
            b = (rng.randrange(1, h - 1), rng.randrange(1, w - 1))
            if a == b or a[0] == b[0] or a[1] == b[1]:
                continue
            if abs(a[0] - b[0]) < 2 or abs(a[1] - b[1]) < 2:
                continue

            a, b = sorted((a, b))
            c0, c1 = _corners(a, b)
            if c0 in all_candidate_corners or c1 in all_candidate_corners:
                continue
            if c0 in marker_cells or c1 in marker_cells or c0 in blocked_corners or c1 in blocked_corners:
                continue
            if c0 in path_cells or c1 in path_cells:
                continue
            if a in path_cells or b in path_cells or a in blocked_corners or b in blocked_corners:
                continue
            if a in all_candidate_corners or b in all_candidate_corners:
                continue
            if _close_to_any(a, marker_cells) or _close_to_any(b, marker_cells):
                continue
            if branch == 2 and c0[0] + c0[1] == c1[0] + c1[1]:
                continue

            chosen, blocked = _planned_choice(a, b, branch)
            candidate_path = set(_path_cells(a, chosen, b))
            if candidate_path & marker_cells:
                continue
            if candidate_path & blocked_corners:
                continue
            if candidate_path & all_candidate_corners:
                continue
            if blocked is not None and blocked in path_cells:
                continue

            pair_specs.append(
                {
                    "color": color,
                    "a": a,
                    "b": b,
                    "c0": c0,
                    "c1": c1,
                    "chosen": chosen,
                    "blocked": blocked,
                    "path": _path_cells(a, chosen, b),
                }
            )
            marker_cells.update({a, b})
            path_cells.update(candidate_path)
            all_candidate_corners.update({c0, c1})
            if blocked is not None:
                blocked_corners.add(blocked)
            placed = True
            break
        if not placed:
            return None

    obstacles: set[Coord] = set()
    static_forbidden = marker_cells | path_cells | all_candidate_corners
    blocked_list = [spec["blocked"] for spec in pair_specs if spec["blocked"] is not None]
    for anchor in blocked_list:
        assert anchor is not None
        shape = _grow_component(
            anchor,
            h,
            w,
            forbidden=static_forbidden - {anchor},
            occupied=obstacles,
            rng=rng,
            size_range=ANCHOR_OBSTACLE_SIZE,
        )
        obstacles.update(shape)

    extra_components = rng.randint(*EXTRA_OBSTACLE_COUNT)
    for _ in range(extra_components):
        free = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in obstacles and (r, c) not in static_forbidden
        ]
        if not free:
            break
        anchor = rng.choice(free)
        shape = _grow_component(
            anchor,
            h,
            w,
            forbidden=static_forbidden,
            occupied=obstacles,
            rng=rng,
            size_range=EXTRA_OBSTACLE_SIZE,
        )
        obstacles.update(shape)

    if not obstacles:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    for spec in pair_specs:
        ar, ac = spec["a"]
        br, bc = spec["b"]
        grid[ar, ac] = spec["color"]
        grid[br, bc] = spec["color"]

    for spec in pair_specs:
        c0 = spec["c0"]
        c1 = spec["c1"]
        blocked0 = grid[c0] == OBSTACLE
        blocked1 = grid[c1] == OBSTACLE
        if spec["blocked"] is None:
            if blocked0 or blocked1:
                return None
        else:
            if spec["blocked"] == c0 and (not blocked0 or blocked1):
                return None
            if spec["blocked"] == c1 and (not blocked1 or blocked0):
                return None

    expected = _expected_output(grid, pair_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if int(np.sum(actual != grid)) < pair_count * 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
