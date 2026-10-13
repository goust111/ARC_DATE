from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7

SEED_COLORS = (BLUE, RED)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

LAYOUT_OPTIONS = ((2, 3), (3, 2), (2, 4), (4, 2))
SIZE_RANGE = (2, 6)
MARGIN_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 600


def _other_seed_color(color: int) -> int:
    return RED if color == BLUE else BLUE


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bfs_dist(graph: dict[int, set[int]], start: int) -> dict[int, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def _analyze_grid(grid: np.ndarray) -> dict:
    """Parse closed regions, black-pixel areas, seed locations, and orange-boundary adjacency."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert np.all(np.isin(g, [BLACK, BLUE, RED, ORANGE]))

    open_mask = g != ORANGE
    open_components = _components(open_mask)
    region_of = -np.ones((h, w), dtype=int)
    regions = []
    seed_regions: dict[int, int] = {}

    for coords in open_components:
        touches_edge = bool(
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        if touches_edge:
            continue

        region_id = len(regions)
        rows = coords[:, 0]
        cols = coords[:, 1]
        region_of[rows, cols] = region_id

        values = g[rows, cols]
        black_mask = values == BLACK
        black_coords = coords[black_mask]
        area = int(black_mask.sum())
        assert area > 0

        present_seeds = {int(v) for v in values if int(v) in SEED_COLORS}
        assert len(present_seeds) <= 1
        if present_seeds:
            seed_color = next(iter(present_seeds))
            assert seed_color not in seed_regions
            seed_regions[seed_color] = region_id

        regions.append(
            {
                "id": region_id,
                "coords": coords,
                "black_coords": black_coords,
                "area": area,
            }
        )

    assert len(seed_regions) == 2

    adjacency = {rid: set() for rid in range(len(regions))}
    for r, c in np.argwhere(g == ORANGE):
        neighbor_ids = {
            int(region_of[nr, nc])
            for dr, dc in DIR4
            for nr, nc in [(int(r) + dr, int(c) + dc)]
            if _in_bounds(g.shape, nr, nc) and region_of[nr, nc] >= 0
        }
        for a, b in combinations(sorted(neighbor_ids), 2):
            adjacency[a].add(b)
            adjacency[b].add(a)

    return {"regions": regions, "adjacency": adjacency, "seed_regions": seed_regions}


def _color_plan(areas: dict[int, int], adjacency: dict[int, set[int]], red_seed: int, blue_seed: int, barrier: int | None) -> dict:
    active = [rid for rid in adjacency if rid != barrier]
    graph = {rid: {nbr for nbr in adjacency[rid] if nbr != barrier} for rid in active}
    assert red_seed in graph and blue_seed in graph and red_seed != blue_seed

    dist_red = _bfs_dist(graph, red_seed)
    dist_blue = _bfs_dist(graph, blue_seed)
    assert set(dist_red) == set(graph) == set(dist_blue)

    fills: dict[int, int] = {}
    for rid in graph:
        dr = dist_red[rid]
        db = dist_blue[rid]
        if dr < db:
            base = RED
            d = dr
        elif db < dr:
            base = BLUE
            d = db
        else:
            assert areas[red_seed] != areas[blue_seed]
            base = RED if areas[red_seed] > areas[blue_seed] else BLUE
            d = dr
        fills[rid] = base if d % 2 == 0 else _other_seed_color(base)

    return {"graph": graph, "dist_red": dist_red, "dist_blue": dist_blue, "fills": fills}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood closed regions by region-graph parity after removing the smallest region as a barrier.

    Input structure:
    - Black background and orange 1-pixel walls form several closed non-edge regions plus one exterior background.
    - Exactly one red seed pixel and one blue seed pixel lie inside two different closed regions.
    - Closed regions are defined by 4-connected non-orange cells, while orange wall pixels stay fixed.

    Rule:
    - Identify all closed non-edge regions and rank them by black-pixel area; the unique smallest region is a barrier.
    - Build a region adjacency graph where two non-barrier regions are adjacent iff they each touch the same orange pixel.
    - Compute BFS distance from the red-seed region and from the blue-seed region on that region graph.
    - For each non-barrier region, choose the nearer seed color; if the distances tie, prefer the seed whose own region has larger area. Fill that region's black pixels with the chosen color on even distance and the opposite seed color on odd distance. Keep orange walls, seeds, and the barrier region unchanged.

    Key insight:
    The alternating flood advances by crossing region boundaries, not pixel steps, and deleting the smallest region changes the region graph itself, which can flip both nearest-seed choice and parity.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    regions = info["regions"]
    adjacency = info["adjacency"]
    seed_regions = info["seed_regions"]

    areas = {region["id"]: region["area"] for region in regions}
    barrier = min(areas, key=areas.get)
    assert list(areas.values()).count(areas[barrier]) == 1

    red_seed = seed_regions[RED]
    blue_seed = seed_regions[BLUE]
    assert red_seed != barrier and blue_seed != barrier

    plan = _color_plan(areas, adjacency, red_seed, blue_seed, barrier)

    out = g.copy()
    for region in regions:
        rid = region["id"]
        if rid == barrier:
            continue
        fill = plan["fills"][rid]
        black_coords = region["black_coords"]
        out[black_coords[:, 0], black_coords[:, 1]] = fill
    return out


def _sample_band_sizes(count: int, low: int, high: int) -> list[int]:
    return [random.randint(low, high) for _ in range(count)]


def _frame_span(sizes: list[int]) -> int:
    return sum(sizes) + len(sizes) + 1


def _cell_graph(rows: int, cols: int) -> dict[int, set[int]]:
    graph = {rid: set() for rid in range(rows * cols)}
    for r in range(rows):
        for c in range(cols):
            rid = r * cols + c
            if r > 0:
                graph[rid].add((r - 1) * cols + c)
            if r + 1 < rows:
                graph[rid].add((r + 1) * cols + c)
            if c > 0:
                graph[rid].add(r * cols + (c - 1))
            if c + 1 < cols:
                graph[rid].add(r * cols + (c + 1))
    return graph


def _abstract_plan(raw_areas: dict[int, int], graph: dict[int, set[int]], red_seed: int, blue_seed: int) -> dict | None:
    if red_seed == blue_seed:
        return None

    eff_areas = dict(raw_areas)
    eff_areas[red_seed] -= 1
    eff_areas[blue_seed] -= 1
    if min(eff_areas.values()) <= 0:
        return None
    if len(set(eff_areas.values())) != len(eff_areas):
        return None

    barrier = min(eff_areas, key=eff_areas.get)
    if barrier in {red_seed, blue_seed}:
        return None
    if list(eff_areas.values()).count(eff_areas[barrier]) != 1:
        return None

    filtered = {rid: {nbr for nbr in graph[rid] if nbr != barrier} for rid in graph if rid != barrier}
    if not filtered:
        return None
    seen = _bfs_dist(filtered, red_seed)
    if set(seen) != set(filtered):
        return None
    if blue_seed not in seen:
        return None

    plan_removed = _color_plan(eff_areas, graph, red_seed, blue_seed, barrier)
    plan_full = _color_plan(eff_areas, graph, red_seed, blue_seed, None)

    active = sorted(filtered)
    tie_regions = [rid for rid in active if rid not in {red_seed, blue_seed} and plan_removed["dist_red"][rid] == plan_removed["dist_blue"][rid]]
    odd_regions = [rid for rid in active if rid not in {red_seed, blue_seed} and min(plan_removed["dist_red"][rid], plan_removed["dist_blue"][rid]) % 2 == 1]
    even_regions = [rid for rid in active if rid not in {red_seed, blue_seed} and min(plan_removed["dist_red"][rid], plan_removed["dist_blue"][rid]) % 2 == 0]
    barrier_effect = any(plan_removed["fills"][rid] != plan_full["fills"][rid] for rid in active if rid not in {red_seed, blue_seed})
    color_mix = len({plan_removed["fills"][rid] for rid in active if rid not in {red_seed, blue_seed}}) == 2
    tie_prefers_larger = any(
        plan_removed["dist_red"][rid] == plan_removed["dist_blue"][rid]
        and plan_removed["fills"][rid] == (RED if eff_areas[red_seed] > eff_areas[blue_seed] else BLUE)
        for rid in tie_regions
    )

    score = 0
    score += 4 if barrier_effect else 0
    score += 3 if tie_regions else 0
    score += 2 if odd_regions else 0
    score += 1 if even_regions else 0
    score += 1 if color_mix else 0

    if not odd_regions or not color_mix:
        return None

    return {
        "effective_areas": eff_areas,
        "barrier": barrier,
        "plan_removed": plan_removed,
        "plan_full": plan_full,
        "tie_regions": tie_regions,
        "odd_regions": odd_regions,
        "even_regions": even_regions,
        "barrier_effect": barrier_effect,
        "tie_prefers_larger": tie_prefers_larger,
        "score": score,
    }


def _choose_layout() -> tuple[int, int]:
    return random.choice(LAYOUT_OPTIONS)


def _sample_partition() -> tuple[int, int, list[int], list[int]] | None:
    rows, cols = _choose_layout()
    heights = _sample_band_sizes(rows, *SIZE_RANGE)
    widths = _sample_band_sizes(cols, *SIZE_RANGE)

    frame_h = _frame_span(heights)
    frame_w = _frame_span(widths)
    margins = [random.randint(*MARGIN_RANGE) for _ in range(4)]
    total_h = margins[0] + frame_h + margins[1]
    total_w = margins[2] + frame_w + margins[3]
    if max(total_h, total_w) > 30:
        return None

    areas = [h * w for h in heights for w in widths]
    if len(set(areas)) != rows * cols:
        return None
    return rows, cols, heights, widths


def _cell_rectangles(top: int, left: int, heights: list[int], widths: list[int]) -> tuple[list[int], list[int], dict[int, tuple[int, int, int, int]]]:
    wall_rows = [top]
    cur = top
    for size in heights:
        cur += size + 1
        wall_rows.append(cur)

    wall_cols = [left]
    cur = left
    for size in widths:
        cur += size + 1
        wall_cols.append(cur)

    rects = {}
    cols = len(widths)
    for r_idx in range(len(heights)):
        for c_idx in range(len(widths)):
            rid = r_idx * cols + c_idx
            r0 = wall_rows[r_idx] + 1
            r1 = wall_rows[r_idx + 1] - 1
            c0 = wall_cols[c_idx] + 1
            c1 = wall_cols[c_idx + 1] - 1
            rects[rid] = (r0, c0, r1, c1)
    return wall_rows, wall_cols, rects


def _build_grid(rows: int, cols: int, heights: list[int], widths: list[int], red_seed: int, blue_seed: int) -> tuple[np.ndarray, dict]:
    frame_h = _frame_span(heights)
    frame_w = _frame_span(widths)
    top = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    h = top + frame_h + bottom_margin
    w = left + frame_w + right_margin
    assert max(h, w) <= 30

    grid = np.full((h, w), BLACK, dtype=int)
    wall_rows, wall_cols, rects = _cell_rectangles(top, left, heights, widths)
    for wr in wall_rows:
        grid[wr, wall_cols[0] : wall_cols[-1] + 1] = ORANGE
    for wc in wall_cols:
        grid[wall_rows[0] : wall_rows[-1] + 1, wc] = ORANGE

    seed_positions = {}
    for rid, (r0, c0, r1, c1) in rects.items():
        if rid == red_seed:
            sr = random.randint(r0, r1)
            sc = random.randint(c0, c1)
            grid[sr, sc] = RED
            seed_positions[RED] = (sr, sc)
        elif rid == blue_seed:
            sr = random.randint(r0, r1)
            sc = random.randint(c0, c1)
            grid[sr, sc] = BLUE
            seed_positions[BLUE] = (sr, sc)

    meta = {
        "rows": rows,
        "cols": cols,
        "heights": heights,
        "widths": widths,
        "rects": rects,
        "wall_rows": wall_rows,
        "wall_cols": wall_cols,
        "seed_positions": seed_positions,
    }
    return grid, meta


def _choose_seed_pair(raw_areas: dict[int, int], graph: dict[int, set[int]]) -> tuple[int, int, dict] | None:
    ids = sorted(graph)
    candidates = []
    for red_seed in ids:
        for blue_seed in ids:
            if red_seed == blue_seed:
                continue
            plan = _abstract_plan(raw_areas, graph, red_seed, blue_seed)
            if plan is not None:
                candidates.append((red_seed, blue_seed, plan))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[2]["score"], item[2]["barrier_effect"], item[2]["tie_prefers_larger"]), reverse=True)
    best_score = candidates[0][2]["score"]
    best = [item for item in candidates if item[2]["score"] == best_score]
    return random.choice(best)


def _expected_from_abstract(grid: np.ndarray, rects: dict[int, tuple[int, int, int, int]], plan: dict, barrier: int) -> np.ndarray:
    out = grid.copy()
    for rid, (r0, c0, r1, c1) in rects.items():
        if rid == barrier:
            continue
        fill = plan["fills"][rid]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r, c] == BLACK:
                    out[r, c] = fill
    return out


def _validate_generated(grid: np.ndarray, meta: dict, raw_areas: dict[int, int], abstract: dict, red_seed: int, blue_seed: int) -> None:
    info = _analyze_grid(grid)
    regions = info["regions"]
    adjacency = info["adjacency"]
    mapped: dict[int, int] = {}
    for rid, (r0, c0, _r1, _c1) in meta["rects"].items():
        seed_color = RED if rid == red_seed else BLUE if rid == blue_seed else None
        if seed_color is not None:
            rr, cc = meta["seed_positions"][seed_color]
        else:
            rr, cc = r0, c0
        match = [region["id"] for region in regions if any((int(r) == rr and int(c) == cc) for r, c in region["coords"])]
        assert len(match) == 1
        mapped[rid] = match[0]

    assert len(set(mapped.values())) == len(mapped) == len(regions)

    analyzed_areas = {region["id"]: region["area"] for region in regions}
    for abstract_id, region_id in mapped.items():
        assert analyzed_areas[region_id] == abstract["effective_areas"][abstract_id]

    graph = _cell_graph(meta["rows"], meta["cols"])
    for abstract_id, region_id in mapped.items():
        expected_neighbors = {mapped[nbr] for nbr in graph[abstract_id]}
        assert adjacency[region_id] == expected_neighbors

    seed_regions = info["seed_regions"]
    assert mapped[red_seed] == seed_regions[RED]
    assert mapped[blue_seed] == seed_regions[BLUE]
    assert mapped[abstract["barrier"]] not in {seed_regions[RED], seed_regions[BLUE]}

    expected = _expected_from_abstract(grid, meta["rects"], abstract["plan_removed"], abstract["barrier"])
    actual = main(grid)
    assert np.array_equal(actual, expected)

    non_seed_regions = [rid for rid in graph if rid not in {red_seed, blue_seed, abstract["barrier"]}]
    assert non_seed_regions
    assert abstract["odd_regions"]
    assert len({abstract["plan_removed"]["fills"][rid] for rid in non_seed_regions}) == 2


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_partition()
    if sampled is None:
        return None
    rows, cols, heights, widths = sampled
    graph = _cell_graph(rows, cols)
    raw_areas = {rid: heights[rid // cols] * widths[rid % cols] for rid in range(rows * cols)}

    chosen = _choose_seed_pair(raw_areas, graph)
    if chosen is None:
        return None
    red_seed, blue_seed, abstract = chosen

    grid, meta = _build_grid(rows, cols, heights, widths, red_seed, blue_seed)
    try:
        _validate_generated(grid, meta, raw_areas, abstract, red_seed, blue_seed)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for ARC puzzle 7")
