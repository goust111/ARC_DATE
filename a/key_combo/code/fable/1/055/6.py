from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_SIDE_RANGE = (11, 19)
SOURCE_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400
MAX_HITS_PER_SOURCE = 3


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _extract_sources(grid: np.ndarray) -> list[tuple[int, int]]:
    h, _w = grid.shape
    sources = [(r, int(grid[r, 0])) for r in range(h) if int(grid[r, 0]) not in (BLACK, GREY)]
    colors = [color for _r, color in sources]
    assert len(colors) == len(set(colors)), "left-edge source colors must be distinct"

    colored = np.argwhere((grid != BLACK) & (grid != GREY))
    assert all(int(c) == 0 for _r, c in colored), "non-grey foreground must live on the left edge"
    return sources


def _extract_blockers(grid: np.ndarray) -> set[Coord]:
    blockers = {tuple(map(int, cell)) for cell in np.argwhere(grid == GREY)}
    assert all(c > 0 for _r, c in blockers), "grey blockers must be inside the grid, not on the left edge"
    return blockers


def _segment_cells(shape: tuple[int, int], pos: Coord, dr: int) -> list[Coord]:
    r, c = pos
    cells: list[Coord] = []
    while True:
        r += dr
        c += 1
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _trace_source(shape: tuple[int, int], start_r: int, blockers: set[Coord]) -> dict:
    path: list[Coord] = []
    hits: list[Coord] = []
    segments: list[tuple[int, tuple[Coord, ...], bool]] = []

    r, c = int(start_r), 0
    dr = 1
    while True:
        segment: list[Coord] = []
        while True:
            r += dr
            c += 1
            if not _in_bounds(shape, r, c):
                if segment:
                    segments.append((dr, tuple(segment), False))
                return {"path": tuple(path), "hits": tuple(hits), "segments": tuple(segments)}
            cell = (r, c)
            segment.append(cell)
            path.append(cell)
            if cell in blockers:
                hits.append(cell)
                segments.append((dr, tuple(segment), True))
                dr = -dr
                break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey blockers hit by left-edge diagonal rays that bounce vertically on every hit.

    Input structure:
    - The background is black.
    - Several distinct colored single pixels sit on the left edge and act as ray sources.
    - Inside the grid there are isolated grey blocker pixels; there are no visible trajectory marks.

    Rule:
    - Each left-edge source emits an invisible ray that starts by moving one cell at a time down-right.
    - Whenever the ray enters a grey blocker, that blocker is recolored to the source color and the ray flips only its vertical direction, continuing rightward from that hit cell.
    - Rays disappear when they leave the grid; black cells stay black, untouched blockers stay grey, and the source pixels remain unchanged.

    Key insight:
    - The output is determined by replaying the full chained bounce path of each source; a nearer blocker can divert the ray away so farther blockers on the same diagonal remain grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blockers = _extract_blockers(g)
    sources = _extract_sources(g)
    assert len(sources) >= 1

    out = g.copy()
    for start_r, color in sources:
        trace = _trace_source(g.shape, start_r, blockers)
        for r, c in trace["hits"]:
            out[r, c] = int(color)
    return out


def _too_close(cell: Coord, occupied: set[Coord]) -> bool:
    r, c = cell
    return any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in occupied)


def _pick_source_rows(h: int, count: int) -> list[int] | None:
    for min_gap in (2, 1):
        for _ in range(80):
            rows = sorted(random.sample(range(h - 1), count))
            if all(b - a >= min_gap for a, b in zip(rows, rows[1:])):
                return rows
    return None


def _sample_hit_targets(count: int) -> list[int]:
    assert count >= 2
    hero = random.randrange(count)
    active = random.choice([i for i in range(count) if i != hero])
    hit_counts = [0] * count
    hit_counts[hero] = random.randint(2, MAX_HITS_PER_SOURCE)
    hit_counts[active] = random.randint(1, min(2, MAX_HITS_PER_SOURCE))
    for i in range(count):
        if i in (hero, active):
            continue
        hit_counts[i] = random.choice((0, 1, 1, 2))
    if count >= 3 and random.random() < 0.55:
        zero_candidates = [i for i in range(count) if i != hero and i != active]
        if zero_candidates:
            hit_counts[random.choice(zero_candidates)] = 0
    return hit_counts


def _search_chain(
    shape: tuple[int, int],
    pos: Coord,
    dr: int,
    remaining_hits: int,
    fixed_blockers: set[Coord],
    forbidden_blocker_cells: set[Coord],
    occupied_blockers: set[Coord],
) -> list[Coord] | None:
    segment = _segment_cells(shape, pos, dr)
    if any(cell in fixed_blockers for cell in segment):
        return None
    if remaining_hits == 0:
        return []

    candidates = [
        (i, cell)
        for i, cell in enumerate(segment)
        if cell not in forbidden_blocker_cells and not _too_close(cell, occupied_blockers)
    ]
    if not candidates:
        return None

    preferred = [(i, cell) for i, cell in candidates if 1 <= i < len(segment) - 1]
    trial_cells = preferred if preferred else candidates
    random.shuffle(trial_cells)
    for _i, hit in trial_cells[:14]:
        tail = _search_chain(
            shape,
            hit,
            -dr,
            remaining_hits - 1,
            fixed_blockers,
            forbidden_blocker_cells,
            occupied_blockers | {hit},
        )
        if tail is not None:
            return [hit] + tail
    return None


def _plan_sources(shape: tuple[int, int], rows: list[int], hit_counts: list[int]) -> list[dict] | None:
    n_sources = len(rows)
    plans: list[dict | None] = [None] * n_sources
    reserved_blockers: set[Coord] = set()
    reserved_paths: set[Coord] = set()

    order = sorted(range(n_sources), key=lambda i: (-hit_counts[i], random.random()))
    for i in order:
        chain = _search_chain(
            shape,
            (rows[i], 0),
            1,
            hit_counts[i],
            reserved_blockers,
            reserved_paths,
            reserved_blockers,
        )
        if chain is None:
            return None

        own_blockers = set(chain)
        trace = _trace_source(shape, rows[i], reserved_blockers | own_blockers)
        if tuple(chain) != trace["hits"]:
            return None

        reserved_blockers |= own_blockers
        reserved_paths |= set(trace["path"])
        plans[i] = {
            "row": rows[i],
            "hits": tuple(chain),
            "trace": trace,
        }

    assert all(plan is not None for plan in plans)
    return [plan for plan in plans if plan is not None]


def _shadow_candidates(shape: tuple[int, int], plans: list[dict], reserved_paths: set[Coord], blockers: set[Coord]) -> list[Coord]:
    candidates: list[Coord] = []
    for plan in plans:
        for dr, cells, ended_with_hit in plan["trace"]["segments"]:
            if not ended_with_hit:
                continue
            hit = cells[-1]
            continuation = _segment_cells(shape, hit, dr)
            valid = [
                cell
                for i, cell in enumerate(continuation)
                if cell not in reserved_paths
                and cell not in blockers
                and not _too_close(cell, blockers)
                and (i >= 1 or len(continuation) == 1)
            ]
            random.shuffle(valid)
            candidates.extend(valid[:2])
    return candidates


def _extra_decoy_candidates(shape: tuple[int, int], reserved_paths: set[Coord], blockers: set[Coord]) -> list[Coord]:
    h, w = shape
    cells = [
        (r, c)
        for r in range(h)
        for c in range(1, w)
        if (r, c) not in reserved_paths and (r, c) not in blockers and not _too_close((r, c), blockers)
    ]
    random.shuffle(cells)
    return cells


def _build_grid(shape: tuple[int, int], rows: list[int], colors: list[int], blockers: set[Coord]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    for row, color in zip(rows, colors):
        grid[row, 0] = int(color)
    for r, c in blockers:
        grid[r, c] = GREY
    return grid


def _has_shadow_blocker(shape: tuple[int, int], traces: list[dict], blockers: set[Coord], hit_cells: set[Coord]) -> bool:
    for trace in traces:
        for dr, cells, ended_with_hit in trace["segments"]:
            if not ended_with_hit:
                continue
            hit = cells[-1]
            for cell in _segment_cells(shape, hit, dr):
                if cell in blockers and cell not in hit_cells:
                    return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        sources = _extract_sources(grid)
        blockers = _extract_blockers(grid)
    except AssertionError:
        return False

    if not (SOURCE_COUNT_RANGE[0] <= len(sources) <= SOURCE_COUNT_RANGE[1]):
        return False

    traces = [_trace_source(grid.shape, row, blockers) for row, _color in sources]
    hit_cells: set[Coord] = set()
    hit_counts: list[int] = []
    for trace in traces:
        hit_counts.append(len(trace["hits"]))
        for cell in trace["hits"]:
            if cell in hit_cells:
                return False
            hit_cells.add(cell)

    if max(hit_counts, default=0) < 2:
        return False
    if sum(count > 0 for count in hit_counts) < 2:
        return False
    if not (blockers - hit_cells):
        return False
    if not _has_shadow_blocker(grid.shape, traces, blockers, hit_cells):
        return False

    expected = grid.copy()
    for (_row, color), trace in zip(sources, traces):
        for r, c in trace["hits"]:
            expected[r, c] = int(color)
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_sources = random.randint(*SOURCE_COUNT_RANGE)
    rows = _pick_source_rows(h, n_sources)
    if rows is None:
        return None

    hit_counts = _sample_hit_targets(n_sources)
    plans = _plan_sources((h, w), rows, hit_counts)
    if plans is None:
        return None

    reserved_paths = {cell for plan in plans for cell in plan["trace"]["path"]}
    blockers = {cell for plan in plans for cell in plan["hits"]}

    shadow_cells = _shadow_candidates((h, w), plans, reserved_paths, blockers)
    if not shadow_cells:
        return None
    random.shuffle(shadow_cells)
    blockers.add(shadow_cells[0])

    for cell in shadow_cells[1:]:
        if random.random() < 0.35 and cell not in blockers and not _too_close(cell, blockers):
            blockers.add(cell)

    for cell in _extra_decoy_candidates((h, w), reserved_paths, blockers):
        if random.random() < 0.18 and not _too_close(cell, blockers):
            blockers.add(cell)
        if len(blockers) >= sum(hit_counts) + 3:
            break

    colors = random.sample(SOURCE_COLORS, n_sources)
    grid = _build_grid((h, w), rows, colors, blockers)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marble-darkroom input")
