import random
from collections import deque

import numpy as np


BLACK = 0
WALL = 8
SEED_COLORS = (2, 3, 4, 7, 6)  # red > green > yellow > orange > magenta
PRIORITY_ORDER = (2, 3, 4, 7, 6, 1, 5, 9)
PRIORITY_RANK = {color: i for i, color in enumerate(PRIORITY_ORDER)}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ORDER_PAIRS = tuple(zip(SEED_COLORS[:-1], SEED_COLORS[1:]))

GEN_ATTEMPTS = 400
COARSE_H_RANGE = (4, 7)
COARSE_W_RANGE = (4, 7)
CELL_SPAN_CHOICES = (2, 3)
REGION_COUNT_RANGE = (4, 6)
MIN_REGION_CELLS = 2


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    comps.sort(key=lambda pts: min(pts))
    return comps


def _bfs_distances(allowed: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = allowed.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c, h, w):
            if allowed[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _solve_with_priority(grid: np.ndarray, priority_order: tuple[int, ...]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    free = g != WALL
    rank = {color: i for i, color in enumerate(priority_order)}

    for coords in _components(free):
        seeds = sorted(
            (r, c, int(g[r, c]))
            for r, c in coords
            if int(g[r, c]) not in (BLACK, WALL)
        )
        assert seeds, "each walled region must contain at least one seed"

        allowed = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            allowed[r, c] = True

        seed_dists = [
            (color, _bfs_distances(allowed, (sr, sc)))
            for sr, sc, color in seeds
        ]

        for r, c in coords:
            best_color = None
            best_key = None
            for seed_order, (color, dist) in enumerate(seed_dists):
                d = int(dist[r, c])
                assert d >= 0, "component cells must stay reachable from every seed search"
                key = (d, rank.get(color, len(rank)), seed_order)
                if best_key is None or key < best_key:
                    best_key = key
                    best_color = color
            out[r, c] = int(best_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood each walled room from its seed pixels, breaking equal-distance ties by a fixed color hierarchy.

    Input structure:
    - The background is black.
    - Thin azure walls split the canvas into several enclosed orthogonal regions.
    - Every region contains 1-4 single-pixel seeds drawn from a small fixed color set, with no repeated seed color inside one region.

    Rule:
    - Treat each non-wall region independently.
    - Within a region, all seeds expand simultaneously by one 4-neighbor step at a time and never cross azure wall cells.
    - A cell takes the color of the seed with the smallest within-region path distance to it; if two seeds reach the cell at the same time, the fixed priority order red > green > yellow > orange > magenta decides the winner.
    - Continue until every non-wall cell in the region is recolored to some seed color, while azure walls stay unchanged.

    Key insight:
    - The fill is geodesic multi-source growth inside each room, not a per-room solid repaint; the only hidden part is the global tie-break order among the seed colors.
    """
    return _solve_with_priority(np.asarray(grid, dtype=int), PRIORITY_ORDER)


def _grow_regions(ch: int, cw: int, n_regions: int, rng: random.Random) -> np.ndarray | None:
    region = np.full((ch, cw), -1, dtype=int)
    cells = [(r, c) for r in range(ch) for c in range(cw)]
    rng.shuffle(cells)
    seeds = cells[:n_regions]
    if len(seeds) < n_regions:
        return None

    frontier: list[tuple[int, int, int]] = []
    for rid, (r, c) in enumerate(seeds):
        region[r, c] = rid
        for nr, nc in _neighbors4(r, c, ch, cw):
            if region[nr, nc] < 0:
                frontier.append((nr, nc, rid))
    rng.shuffle(frontier)

    while frontier:
        idx = rng.randrange(len(frontier))
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if region[r, c] >= 0:
            continue
        region[r, c] = rid
        for nr, nc in _neighbors4(r, c, ch, cw):
            if region[nr, nc] < 0:
                frontier.append((nr, nc, rid))

    counts = [int(np.count_nonzero(region == rid)) for rid in range(n_regions)]
    if min(counts) < MIN_REGION_CELLS:
        return None
    return region


def _render_rooms(region: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    ch, cw = region.shape
    h = ch * cell_h + (ch + 1)
    w = cw * cell_w + (cw + 1)
    grid = np.full((h, w), WALL, dtype=int)

    def block_bounds(r: int, c: int) -> tuple[int, int, int, int]:
        r0 = 1 + r * (cell_h + 1)
        c0 = 1 + c * (cell_w + 1)
        return r0, c0, r0 + cell_h, c0 + cell_w

    for r in range(ch):
        for c in range(cw):
            r0, c0, r1, c1 = block_bounds(r, c)
            grid[r0:r1, c0:c1] = BLACK
            if c + 1 < cw and int(region[r, c]) == int(region[r, c + 1]):
                grid[r0:r1, c1] = BLACK
            if r + 1 < ch and int(region[r, c]) == int(region[r + 1, c]):
                grid[r1, c0:c1] = BLACK
    return grid


def _coord_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _line_segments(coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in coords:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    segments: list[list[tuple[int, int]]] = []
    for r, cols in by_row.items():
        cols.sort()
        run = [cols[0]]
        for c in cols[1:]:
            if c == run[-1] + 1:
                run.append(c)
            else:
                if len(run) >= 3:
                    segments.append([(r, x) for x in run])
                run = [c]
        if len(run) >= 3:
            segments.append([(r, x) for x in run])

    for c, rows in by_col.items():
        rows.sort()
        run = [rows[0]]
        for r in rows[1:]:
            if r == run[-1] + 1:
                run.append(r)
            else:
                if len(run) >= 3:
                    segments.append([(x, c) for x in run])
                run = [r]
        if len(run) >= 3:
            segments.append([(x, c) for x in run])

    return segments


def _find_priority_pair(coords: list[tuple[int, int]], rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]] | None:
    segments = _line_segments(coords)
    rng.shuffle(segments)
    for segment in segments:
        choices = []
        for i in range(len(segment)):
            for j in range(i + 2, len(segment)):
                if (j - i) % 2 == 0:
                    choices.append((j - i, segment[i], segment[j]))
        if not choices:
            continue
        choices.sort(reverse=True)
        span, a, b = choices[0]
        if span >= 2:
            return a, b

    allowed = _coord_mask(
        (
            max(r for r, _ in coords) + 1,
            max(c for _, c in coords) + 1,
        ),
        coords,
    )
    candidates = list(coords)
    rng.shuffle(candidates)
    candidates = candidates[: min(len(candidates), 24)]
    for i, a in enumerate(candidates):
        dist_a = _bfs_distances(allowed, a)
        for b in candidates[i + 1 :]:
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 3:
                continue
            dist_b = _bfs_distances(allowed, b)
            tie_count = int(np.count_nonzero((dist_a >= 0) & (dist_a == dist_b)))
            if tie_count >= 1:
                return a, b
    return None


def _pick_spaced_cells(
    coords: list[tuple[int, int]],
    count: int,
    used: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    pool = list(coords)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if cell in used:
            continue
        if any(abs(cell[0] - r) + abs(cell[1] - c) < 3 for r, c in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _region_seed_count(area: int, rng: random.Random, force_single: bool = False) -> int:
    if force_single:
        return 1
    if area < 18:
        return rng.choice([1, 2])
    if area < 32:
        return rng.choice([1, 2, 3])
    return rng.choice([1, 2, 3, 4])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.choice(CELL_SPAN_CHOICES)
    cell_w = rng.choice(CELL_SPAN_CHOICES)
    ch = rng.randint(*COARSE_H_RANGE)
    cw = rng.randint(*COARSE_W_RANGE)
    if ch * (cell_h + 1) + 1 > 30 or cw * (cell_w + 1) + 1 > 30:
        return None

    n_regions = rng.randint(*REGION_COUNT_RANGE)
    if n_regions > ch * cw:
        return None

    coarse = _grow_regions(ch, cw, n_regions, rng)
    if coarse is None:
        return None

    grid = _render_rooms(coarse, cell_h, cell_w)
    comps = _components(grid != WALL)
    if len(comps) != n_regions:
        return None

    infos = []
    for idx, coords in enumerate(comps):
        segments = _line_segments(coords)
        max_segment = max((len(seg) for seg in segments), default=0)
        infos.append(
            {
                "id": idx,
                "coords": coords,
                "area": len(coords),
                "max_segment": max_segment,
            }
        )
    infos.sort(key=lambda info: (-info["area"], min(info["coords"])))

    priority_candidates = [
        info for info in infos
        if info["area"] >= 14 and info["max_segment"] >= 3
    ]
    if len(priority_candidates) < 2:
        return None

    k_priority = min(len(priority_candidates), rng.randint(2, 4), len(ORDER_PAIRS))
    priority_infos = priority_candidates[:k_priority]
    pair_colors = rng.sample(list(ORDER_PAIRS), k_priority)

    placements: dict[int, list[tuple[int, int, int]]] = {}
    for info, (high, low) in zip(priority_infos, pair_colors):
        pair = _find_priority_pair(info["coords"], rng)
        if pair is None:
            return None
        a, b = pair
        if rng.random() < 0.5:
            placements[info["id"]] = [(a[0], a[1], high), (b[0], b[1], low)]
        else:
            placements[info["id"]] = [(a[0], a[1], low), (b[0], b[1], high)]

    remaining = [info for info in infos if info["id"] not in placements]
    if not remaining:
        return None

    complex_candidates = [info for info in remaining if info["area"] >= 24]
    complex_region = rng.choice(complex_candidates) if complex_candidates else None

    for i, info in enumerate(remaining):
        force_single = i == 0
        if complex_region is not None and info["id"] == complex_region["id"]:
            count = 3
        else:
            count = _region_seed_count(info["area"], rng, force_single=force_single)
        count = min(count, len(SEED_COLORS))
        used = set()
        cells = _pick_spaced_cells(info["coords"], count, used, rng)
        if cells is None:
            if count == 1:
                return None
            cells = _pick_spaced_cells(info["coords"], 1, used, rng)
            if cells is None:
                return None
            count = 1
        colors = rng.sample(SEED_COLORS, count)
        placements[info["id"]] = [(r, c, color) for (r, c), color in zip(cells, colors)]

    for info in infos:
        seeds = placements.get(info["id"], [])
        if not (1 <= len(seeds) <= 4):
            return None
        if len({color for _, _, color in seeds}) != len(seeds):
            return None
        for r, c, color in seeds:
            if grid[r, c] != BLACK:
                return None
            grid[r, c] = color

    by_comp = {info["id"]: placements[info["id"]] for info in infos}
    multi_regions = sum(len(seeds) > 1 for seeds in by_comp.values())
    single_regions = sum(len(seeds) == 1 for seeds in by_comp.values())
    if multi_regions < 2 or single_regions < 1:
        return None

    actual = _solve_with_priority(grid, PRIORITY_ORDER)
    reversed_out = _solve_with_priority(grid, tuple(reversed(SEED_COLORS)))
    if np.array_equal(actual, reversed_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
