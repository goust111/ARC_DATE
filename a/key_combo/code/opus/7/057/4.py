from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4, 7)
GEN_ATTEMPTS = 500
GRID_MIN_SIZE = 14
GRID_MAX_SIZE = 18
ANCHOR_COUNT_OPTIONS = (2, 3)
SINGLETON_COUNT_OPTIONS = (4, 5)
DISTANCE_OPTIONS = (2, 3, 4, 5)
MIN_ANCHOR_SEPARATION = 5
MIN_SOURCE_SEPARATION = 2


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _find_anchors(grid: np.ndarray) -> list[BBox]:
    comps = _components4(grid == GREY)
    assert comps, "expected grey 2x2 anchors"
    anchors: list[BBox] = []
    for comp in comps:
        assert len(comp) == 4, "each grey anchor must contain exactly four cells"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 == r0 + 1 and c1 == c0 + 1, "grey anchors must be solid 2x2 blocks"
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                assert int(grid[r, c]) == GREY, "grey anchors must fill their 2x2 bbox"
        anchors.append((r0, c0, r1, c1))
    anchors.sort()
    return anchors


def _anchor_cells(anchor: BBox) -> set[Coord]:
    r0, c0, r1, c1 = anchor
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _anchor_distance(point: Coord, anchor: BBox) -> int:
    r, c = point
    r0, c0, r1, c1 = anchor
    dr = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    dc = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return max(dr, dc)


def _frame_direction(anchor: BBox, cell: Coord) -> Coord:
    r, c = cell
    r0, c0, r1, c1 = anchor
    if r < r0:
        if c < c0:
            return (-1, -1)
        if c > c1:
            return (-1, 1)
        return (-1, 0)
    if r > r1:
        if c < c0:
            return (1, -1)
        if c > c1:
            return (1, 1)
        return (1, 0)
    return (0, -1) if c < c0 else (0, 1)


def _frame_cells(anchor: BBox) -> list[Coord]:
    r0, c0, r1, c1 = anchor
    cells: list[Coord] = []
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            cells.append((r, c))
    return cells


def _target_sort_key(source: Coord, anchor: BBox, cell: Coord) -> tuple[int, int, int, int, int]:
    direction = _frame_direction(anchor, cell)
    cardinal_priority = {
        (-1, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (0, -1): 3,
    }.get(direction, 4)
    return (
        _chebyshev(source, cell),
        cardinal_priority,
        _manhattan(source, cell),
        cell[0],
        cell[1],
    )


def _select_target_frame(source: Coord, anchor: BBox) -> Coord:
    return min(_frame_cells(anchor), key=lambda cell: _target_sort_key(source, anchor, cell))


def _nearest_anchor(source: Coord, anchors: list[BBox]) -> tuple[int, int]:
    distances = [_anchor_distance(source, anchor) for anchor in anchors]
    best = min(distances)
    winners = [idx for idx, dist in enumerate(distances) if dist == best]
    assert len(winners) == 1, "nearest grey anchor must be unique"
    return winners[0], best


def _nearest_anchor_if_unique(source: Coord, anchors: list[BBox]) -> tuple[int, int] | None:
    distances = [_anchor_distance(source, anchor) for anchor in anchors]
    best = min(distances)
    winners = [idx for idx, dist in enumerate(distances) if dist == best]
    if len(winners) != 1:
        return None
    return winners[0], best


def _ray_cells(start: Coord, direction: Coord, length: int) -> list[Coord]:
    return [(start[0] + step * direction[0], start[1] + step * direction[1]) for step in range(length)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Snap each colored singleton onto the nearest grey anchor frame and re-encode its original distance as an outward ray.

    Input structure:
    - The grid has a black background, 2-3 grey 2x2 anchor blocks, and several non-grey singleton pixels.
    - Every colored singleton is outside all anchors and has one unique nearest grey anchor under Chebyshev distance.
    - The anchor frame is the 12-cell ring immediately surrounding each 2x2 grey block.

    Rule:
    - For each colored singleton, find its unique nearest grey anchor and record the Chebyshev distance D from the singleton to that anchor.
    - Move the singleton onto the nearest cell of that anchor's outer frame; if several frame cells are equally near, prefer the cardinal directions in the order up, right, down, left.
    - Clear the original singleton positions, then process singletons in increasing D order and break ties by original row then column.
    - From each snapped frame cell, extend an outward ray in the direction indicated by that frame position, including the frame cell itself, for exactly D cells unless the path leaves the grid or reaches a non-black cell first.

    Key insight:
    - Snapping destroys the singleton's original position, so the solver must preserve that lost distance information by turning it into the length of the emitted ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    anchors = _find_anchors(g)
    sources: list[tuple[int, int, int]] = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere((g != BACKGROUND) & (g != GREY))
    ]
    assert sources, "expected colored singleton pixels"

    jobs: list[tuple[int, int, int, int, Coord, Coord]] = []
    for r, c, color in sources:
        anchor_idx, distance = _nearest_anchor((r, c), anchors)
        anchor = anchors[anchor_idx]
        target = _select_target_frame((r, c), anchor)
        direction = _frame_direction(anchor, target)
        jobs.append((distance, r, c, color, target, direction))

    out = g.copy()
    for r, c, _color in sources:
        out[r, c] = BACKGROUND

    for distance, r, c, color, target, direction in sorted(jobs, key=lambda item: (item[0], item[1], item[2])):
        tr, tc = target
        assert out[tr, tc] == BACKGROUND, "frame destinations must remain free before their own ray is drawn"
        out[tr, tc] = color
        rr, cc = tr, tc
        for _ in range(1, distance):
            rr += direction[0]
            cc += direction[1]
            if not _in_bounds(out.shape, rr, cc):
                break
            if out[rr, cc] != BACKGROUND:
                break
            out[rr, cc] = color

    return out


def _bbox_chebyshev_gap(a: BBox, b: BBox) -> int:
    if a[2] < b[0]:
        dr = b[0] - a[2]
    elif b[2] < a[0]:
        dr = a[0] - b[2]
    else:
        dr = 0
    if a[3] < b[1]:
        dc = b[1] - a[3]
    elif b[3] < a[1]:
        dc = a[1] - b[3]
    else:
        dc = 0
    return max(dr, dc)


def _sample_anchors(rng: random.Random, shape: tuple[int, int], count: int) -> list[BBox] | None:
    h, w = shape
    rows = list(range(3, h - 4))
    cols = list(range(3, w - 4))
    candidates = [(r, c, r + 1, c + 1) for r in rows for c in cols]
    rng.shuffle(candidates)

    chosen: list[BBox] = []
    for anchor in candidates:
        if all(_bbox_chebyshev_gap(anchor, other) >= MIN_ANCHOR_SEPARATION for other in chosen):
            chosen.append(anchor)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _assignment_counts(rng: random.Random, anchor_count: int, total: int) -> list[int]:
    counts = [1] * anchor_count
    for _ in range(total - anchor_count):
        counts[rng.randrange(anchor_count)] += 1
    rng.shuffle(counts)
    return counts


def _source_isolated(source: Coord, others: list[Coord]) -> bool:
    return all(_chebyshev(source, other) >= MIN_SOURCE_SEPARATION for other in others)


def _candidate_assignment_ok(
    shape: tuple[int, int],
    anchors: list[BBox],
    anchor_idx: int,
    target: Coord,
    direction: Coord,
    distance: int,
    reserved_output: set[Coord],
    reserved_sources: list[Coord],
) -> tuple[bool, Coord, list[Coord]]:
    path = _ray_cells(target, direction, distance)
    if any(not _in_bounds(shape, r, c) for r, c in path):
        return False, (0, 0), []

    anchor_cells = set().union(*(_anchor_cells(anchor) for anchor in anchors))
    if any(cell in anchor_cells for cell in path):
        return False, (0, 0), []
    if any(cell in reserved_output for cell in path):
        return False, (0, 0), []

    source = path[-1]
    if not _source_isolated(source, reserved_sources):
        return False, (0, 0), []

    nearest = _nearest_anchor_if_unique(source, anchors)
    if nearest is None:
        return False, (0, 0), []
    nearest_idx, actual_distance = nearest
    if nearest_idx != anchor_idx or actual_distance != distance:
        return False, (0, 0), []
    if _select_target_frame(source, anchors[anchor_idx]) != target:
        return False, (0, 0), []

    return True, source, path


def _build_input(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    anchor_count = rng.choice(ANCHOR_COUNT_OPTIONS)
    anchors = _sample_anchors(rng, (height, width), anchor_count)
    if anchors is None:
        return None

    singleton_count = rng.choice(SINGLETON_COUNT_OPTIONS)
    counts = _assignment_counts(rng, anchor_count, singleton_count)
    colors = list(SOURCE_COLORS)
    rng.shuffle(colors)
    colors = colors[:singleton_count]

    reserved_output: set[Coord] = set()
    reserved_sources: list[Coord] = []
    placements: list[tuple[Coord, int]] = []

    color_idx = 0
    for anchor_idx, anchor in enumerate(anchors):
        frame_specs = [(cell, _frame_direction(anchor, cell)) for cell in _frame_cells(anchor)]
        rng.shuffle(frame_specs)
        chosen_for_anchor = 0
        used_cells: set[Coord] = set()
        attempts = 0
        while chosen_for_anchor < counts[anchor_idx] and attempts < 200:
            attempts += 1
            target, direction = rng.choice(frame_specs)
            if target in used_cells:
                continue
            distance = rng.choice(DISTANCE_OPTIONS)
            ok, source, path = _candidate_assignment_ok(
                (height, width),
                anchors,
                anchor_idx,
                target,
                direction,
                distance,
                reserved_output,
                reserved_sources,
            )
            if not ok:
                continue
            used_cells.add(target)
            reserved_sources.append(source)
            reserved_output.update(path)
            placements.append((source, colors[color_idx]))
            color_idx += 1
            chosen_for_anchor += 1
        if chosen_for_anchor != counts[anchor_idx]:
            return None

    grid = np.zeros((height, width), dtype=int)
    for anchor in anchors:
        for r, c in _anchor_cells(anchor):
            grid[r, c] = GREY
    for (r, c), color in placements:
        if grid[r, c] != BACKGROUND:
            return None
        grid[r, c] = color

    try:
        output = main(grid)
    except AssertionError:
        return None

    source_coords = [coord for coord, _color in placements]
    if len(set(source_coords)) != len(source_coords):
        return None
    if max(grid.shape) > 30 or max(output.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
