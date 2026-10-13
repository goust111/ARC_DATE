from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PINK = 6

Coord = tuple[int, int]
Candidate = tuple[Coord, Coord, Coord | None, str, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _coords_of(grid: np.ndarray, color: int) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]


def _unique_row_col_pairs(sources: list[Coord], pivots: list[Coord]) -> list[tuple[Coord, Coord]]:
    assert len(sources) == len(pivots) > 0, "expected equal nonzero source and pivot counts"
    candidates = {
        source: [pivot for pivot in pivots if pivot[0] == source[0] or pivot[1] == source[1]]
        for source in sources
    }
    assert all(candidates[source] for source in sources), "each source must align with a pivot"

    ordered_sources = sorted(sources, key=lambda source: (len(candidates[source]), source[0], source[1]))
    solutions: list[dict[Coord, Coord]] = []

    def search(i: int, used: set[Coord], current: dict[Coord, Coord]) -> None:
        if len(solutions) > 1:
            return
        if i == len(ordered_sources):
            solutions.append(current.copy())
            return
        source = ordered_sources[i]
        for pivot in candidates[source]:
            if pivot in used:
                continue
            used.add(pivot)
            current[source] = pivot
            search(i + 1, used, current)
            del current[source]
            used.remove(pivot)

    search(0, set(), {})
    assert len(solutions) == 1, "expected a unique one-to-one row/column matching"
    return [(source, solutions[0][source]) for source in sources]


def _matching_count(sources: list[Coord], pivots: list[Coord]) -> int:
    candidates = {
        source: [pivot for pivot in pivots if pivot[0] == source[0] or pivot[1] == source[1]]
        for source in sources
    }
    if any(not options for options in candidates.values()):
        return 0

    ordered_sources = sorted(sources, key=lambda source: (len(candidates[source]), source[0], source[1]))
    count = 0

    def search(i: int, used: set[Coord]) -> None:
        nonlocal count
        if count > 1:
            return
        if i == len(ordered_sources):
            count += 1
            return
        for pivot in candidates[ordered_sources[i]]:
            if pivot in used:
                continue
            used.add(pivot)
            search(i + 1, used)
            used.remove(pivot)

    search(0, set())
    return count


def _non_red_exterior(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[Coord] = deque()

    def add(r: int, c: int) -> None:
        if not _in_bounds(grid.shape, r, c):
            return
        if int(grid[r, c]) == RED or exterior[r, c]:
            return
        exterior[r, c] = True
        queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            add(rr, cc)

    return exterior


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label += 1
            labels[r, c] = label
            queue: deque[Coord] = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
    return labels, label


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each magenta source pixel 90 degrees counterclockwise around its matched blue pivot, then fill the enclosed red-walled cavity containing the landing point.

    Input structure:
    - The grid has a black background with red one-cell-thick wall segments.
    - Some red walls form closed cavities while others are open distractors.
    - Blue pivot pixels and magenta source pixels lie outside the walls in open space, with each source uniquely matched to one pivot by sharing a row or column.

    Rule:
    - Remove all magenta source pixels.
    - For each matched source/pivot pair, rotate the source offset 90 degrees counterclockwise around the pivot to get a landing point.
    - If that landing point is an in-bounds background cell inside a region fully enclosed by red walls, flood-fill that entire enclosed background region magenta; otherwise do nothing there.
    - Keep the blue pivots and all red walls unchanged.

    Key insight:
    - The visible magenta pixels are only pointers: the actual filled target is determined by an unseen rotated landing point, and enclosure depends only on the red-wall topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    colors = set(map(int, np.unique(g)))
    assert bg == BG, "expected black background"
    assert colors <= {BG, BLUE, RED, PINK}, "unexpected colors in grid"

    sources = _coords_of(g, PINK)
    pivots = _coords_of(g, BLUE)
    pairs = _unique_row_col_pairs(sources, pivots)

    exterior = _non_red_exterior(g)
    enclosed_bg = (g == BG) & ~exterior
    labels, _ = _label_regions(enclosed_bg)

    out = g.copy()
    out[g == PINK] = BG

    fill_labels: set[int] = set()
    for (sr, sc), (pr, pc) in pairs:
        dr = sr - pr
        dc = sc - pc
        tr = pr - dc
        tc = pc + dr
        if _in_bounds(g.shape, tr, tc) and int(labels[tr, tc]) > 0:
            fill_labels.add(int(labels[tr, tc]))

    for label in fill_labels:
        out[labels == label] = PINK

    return out


GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
CLOSED_FRAME_COUNT_RANGE = (2, 4)
FRAME_HEIGHT_RANGE = (5, 9)
FRAME_WIDTH_RANGE = (5, 9)
OPEN_DISTRACTOR_COUNT_RANGE = (1, 3)
PAIR_COUNT_RANGE = (2, 4)
DISTANCE_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 300
MAX_FRAME_ATTEMPTS = 60
MAX_DISTRACTOR_ATTEMPTS = 40
MAX_POOL_SIZE = 300


def _draw_cells(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _frame_cells(r0: int, c0: int, h: int, w: int) -> list[Coord]:
    cells: set[Coord] = set()
    for c in range(c0, c0 + w):
        cells.add((r0, c))
        cells.add((r0 + h - 1, c))
    for r in range(r0, r0 + h):
        cells.add((r, c0))
        cells.add((r, c0 + w - 1))
    return sorted(cells)


def _is_red_clear(grid: np.ndarray, cells: list[Coord], pad: int) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if not _in_bounds(grid.shape, rr, cc) or (rr, cc) in cell_set:
                    continue
                if int(grid[rr, cc]) == RED:
                    return False
    return True


def _place_closed_frames(grid: np.ndarray) -> list[int] | None:
    frame_count = random.randint(*CLOSED_FRAME_COUNT_RANGE)
    regions: list[int] = []
    for _ in range(frame_count):
        placed = False
        for _ in range(MAX_FRAME_ATTEMPTS):
            h = random.randint(*FRAME_HEIGHT_RANGE)
            w = random.randint(*FRAME_WIDTH_RANGE)
            if h >= grid.shape[0] - 2 or w >= grid.shape[1] - 2:
                continue
            r0 = random.randint(1, grid.shape[0] - h - 1)
            c0 = random.randint(1, grid.shape[1] - w - 1)
            cells = _frame_cells(r0, c0, h, w)
            if not _is_red_clear(grid, cells, pad=2):
                continue
            _draw_cells(grid, cells, RED)
            regions.append(len(regions) + 1)
            placed = True
            break
        if not placed:
            return None
    return regions


def _distractor_cells(shape: tuple[int, int], kind: str) -> list[Coord] | None:
    h, w = shape
    if kind == "hline":
        length = random.randint(4, 8)
        r = random.randint(1, h - 2)
        c0 = random.randint(1, w - length - 1)
        return [(r, c) for c in range(c0, c0 + length)]
    if kind == "vline":
        length = random.randint(4, 8)
        c = random.randint(1, w - 2)
        r0 = random.randint(1, h - length - 1)
        return [(r, c) for r in range(r0, r0 + length)]
    arm_a = random.randint(3, 6)
    arm_b = random.randint(3, 6)
    r0 = random.randint(1, h - arm_a - 1)
    c0 = random.randint(1, w - arm_b - 1)
    cells = [(r0 + i, c0) for i in range(arm_a)]
    cells.extend((r0, c0 + j) for j in range(arm_b))
    return list(dict.fromkeys(cells))


def _place_open_distractors(grid: np.ndarray) -> bool:
    count = random.randint(*OPEN_DISTRACTOR_COUNT_RANGE)
    for _ in range(count):
        placed = False
        for _ in range(MAX_DISTRACTOR_ATTEMPTS):
            kind = random.choice(["hline", "vline", "elbow"])
            cells = _distractor_cells(grid.shape, kind)
            if cells is None:
                continue
            if not _is_red_clear(grid, cells, pad=1):
                continue
            _draw_cells(grid, cells, RED)
            placed = True
            break
        if not placed:
            return False
    return True


def _enumerate_candidates(grid: np.ndarray, exterior: np.ndarray, labels: np.ndarray) -> dict[tuple[str, int], list[Candidate]]:
    h, w = grid.shape
    pools: dict[tuple[str, int], list[Candidate]] = {}
    for key in [("open", 0), ("wall", 0), ("oob", 0)]:
        pools[key] = []

    for pr, pc in map(tuple, np.argwhere(exterior & (grid == BG))):
        pr, pc = int(pr), int(pc)
        for distance in range(DISTANCE_RANGE[0], DISTANCE_RANGE[1] + 1):
            for dr, dc in ((0, distance), (0, -distance), (distance, 0), (-distance, 0)):
                sr, sc = pr + dr, pc + dc
                if not _in_bounds(grid.shape, sr, sc) or not bool(exterior[sr, sc]):
                    continue
                tr = pr - dc
                tc = pc + dr
                source = (sr, sc)
                pivot = (pr, pc)
                if not _in_bounds(grid.shape, tr, tc):
                    pools[("oob", 0)].append((pivot, source, None, "oob", 0))
                    continue
                target = (tr, tc)
                if int(grid[tr, tc]) == RED:
                    pools[("wall", 0)].append((pivot, source, target, "wall", 0))
                elif int(labels[tr, tc]) > 0:
                    region = int(labels[tr, tc])
                    pools.setdefault(("hit", region), []).append((pivot, source, target, "hit", region))
                elif bool(exterior[tr, tc]):
                    pools[("open", 0)].append((pivot, source, target, "open", 0))

    for key, candidates in list(pools.items()):
        random.shuffle(candidates)
        pools[key] = candidates[:MAX_POOL_SIZE]
    return pools


def _candidate_compatible(
    candidate: Candidate,
    pivots: list[Coord],
    sources: list[Coord],
    used_targets: set[Coord],
) -> bool:
    pivot, source, target, _kind, _region = candidate
    used_objects = set(pivots) | set(sources)
    if pivot == source or pivot in used_objects or source in used_objects:
        return False

    for old_source in sources:
        if old_source[0] == pivot[0] or old_source[1] == pivot[1]:
            return False
    for old_pivot in pivots:
        if source[0] == old_pivot[0] or source[1] == old_pivot[1]:
            return False

    if target is not None and (target in used_objects or target in used_targets):
        return False
    return True


def _choose_pairs(requirements: list[tuple[str, int]], pools: dict[tuple[str, int], list[Candidate]]) -> list[Candidate] | None:
    ordered = sorted(requirements, key=lambda requirement: len(pools.get(requirement, [])))
    if any(not pools.get(requirement) for requirement in ordered):
        return None

    chosen: list[Candidate] = []
    pivots: list[Coord] = []
    sources: list[Coord] = []
    used_targets: set[Coord] = set()

    def search(i: int) -> bool:
        if i == len(ordered):
            return _matching_count(sources, pivots) == 1

        requirement = ordered[i]
        for candidate in pools[requirement]:
            if not _candidate_compatible(candidate, pivots, sources, used_targets):
                continue
            pivot, source, target, _kind, _region = candidate
            chosen.append(candidate)
            pivots.append(pivot)
            sources.append(source)
            added_target = False
            if target is not None:
                used_targets.add(target)
                added_target = True
            if search(i + 1):
                return True
            if added_target:
                used_targets.remove(target)
            sources.pop()
            pivots.pop()
            chosen.pop()
        return False

    return chosen if search(0) else None


def _build_requirements(region_ids: list[int], pools: dict[tuple[str, int], list[Candidate]]) -> list[tuple[str, int]] | None:
    hit_regions = [region for region in region_ids if pools.get(("hit", region))]
    miss_kinds = [kind for kind in ("open", "wall", "oob") if pools.get((kind, 0))]
    if len(hit_regions) < 1 or not miss_kinds:
        return None

    pair_count = random.randint(*PAIR_COUNT_RANGE)
    max_hits = min(pair_count - 1, len(hit_regions))
    if len(hit_regions) > 1:
        max_hits = min(max_hits, len(hit_regions) - 1)
    max_hits = max(1, max_hits)
    num_hits = random.randint(1, max_hits)
    num_misses = pair_count - num_hits

    chosen_regions = random.sample(hit_regions, num_hits)
    requirements = [("hit", region) for region in chosen_regions]
    miss_weights = {"open": 4, "wall": 3, "oob": 2}
    for _ in range(num_misses):
        requirements.append((random.choices(miss_kinds, weights=[miss_weights[k] for k in miss_kinds])[0], 0))
    random.shuffle(requirements)
    return requirements


def _expected_output(inp: np.ndarray, labels: np.ndarray, chosen: list[Candidate]) -> np.ndarray:
    out = inp.copy()
    out[out == PINK] = BG
    for _pivot, _source, _target, kind, region in chosen:
        if kind == "hit":
            out[labels == region] = PINK
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    if _place_closed_frames(grid) is None:
        return None
    if not _place_open_distractors(grid):
        return None

    exterior = _non_red_exterior(grid)
    enclosed_bg = (grid == BG) & ~exterior
    labels, region_count = _label_regions(enclosed_bg)
    if region_count < 2:
        return None

    region_ids = list(range(1, region_count + 1))
    pools = _enumerate_candidates(grid, exterior, labels)
    requirements = _build_requirements(region_ids, pools)
    if requirements is None:
        return None

    chosen = _choose_pairs(requirements, pools)
    if chosen is None:
        return None

    inp = grid.copy()
    for pivot, source, _target, _kind, _region in chosen:
        inp[pivot] = BLUE
        inp[source] = PINK

    expected = _expected_output(inp, labels, chosen)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
