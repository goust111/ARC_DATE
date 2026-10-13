from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    labels = np.zeros(mask.shape, dtype=int)
    components: list[list[tuple[int, int]]] = []
    next_id = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components.append(coords)
    return labels, components


def _unique_majority(counts: Counter[int]) -> int | None:
    ranked = counts.most_common()
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return int(ranked[0][0])


def _analyze_wall_map(grid: np.ndarray, wall_color: int) -> tuple[np.ndarray, dict[int, int]] | None:
    g = np.asarray(grid, dtype=int)
    region_labels, regions = _label_components(g != int(wall_color))
    n_regions = len(regions)
    if n_regions < 2:
        return None

    wall_labels, walls = _label_components(g == int(wall_color))
    if not walls:
        return None

    graph: dict[int, set[int]] = {region_id: set() for region_id in range(1, n_regions + 1)}
    for wall_id, coords in enumerate(walls, start=1):
        adjacent = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(g.shape, nr, nc):
                    region_id = int(region_labels[nr, nc])
                    if region_id > 0:
                        adjacent.add(region_id)
        if len(adjacent) != 2:
            return None
        a, b = sorted(adjacent)
        graph[a].add(b)
        graph[b].add(a)

    start = 1
    parity = {start: 0}
    q = deque([start])
    while q:
        region_id = q.popleft()
        for nxt in graph[region_id]:
            want = 1 - parity[region_id]
            if nxt in parity:
                if parity[nxt] != want:
                    return None
            else:
                parity[nxt] = want
                q.append(nxt)

    if len(parity) != n_regions:
        return None
    return region_labels, parity


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-separated regions by the majority marker color of their bipartite camp.

    Input structure:
    - The grid contains one dominant background color, one wall color, and scattered
      single-pixel colored markers inside the non-wall regions.
    - Walls are one-cell-thick 4-connected segments, and each wall segment separates
      exactly two background regions.
    - The region adjacency graph induced by crossing a wall is globally bipartite.

    Rule:
    - Treat every non-wall cell as belonging to a wall-separated region, so marker
      pixels do not break a region apart.
    - Two regions belong to opposite camps iff they touch opposite sides of the same wall.
    - For each camp, collect all marker pixels across all of its regions, choose the
      unique majority marker color, and repaint every region in that camp with that color.
      Remove the original markers and keep every wall unchanged.

    Key insight:
    - The decisive vote is not local to one region: walls define a two-camp region graph,
      and the correct fill color comes from aggregating markers over the whole camp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    wall_candidates = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        analysis = _analyze_wall_map(g, color)
        if analysis is not None:
            wall_candidates.append((int(color), analysis))
    assert len(wall_candidates) == 1, "expected a unique wall color"

    wall_color, (region_labels, parity) = wall_candidates[0]
    party_counts = {0: Counter(), 1: Counter()}
    for region_id, party in parity.items():
        region_mask = region_labels == int(region_id)
        markers = g[region_mask & (g != bg) & (g != wall_color)]
        party_counts[party].update(map(int, markers))

    fill_colors: dict[int, int] = {}
    for party in (0, 1):
        fill = _unique_majority(party_counts[party])
        assert fill is not None, "each camp must have a unique majority marker color"
        fill_colors[party] = fill

    out = g.copy()
    for region_id, party in parity.items():
        out[region_labels == int(region_id)] = int(fill_colors[party])
    out[g == int(wall_color)] = int(wall_color)
    return out


GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
WALL_COUNT_RANGE = (3, 5)
ENDPOINT_GAP = 3
MAX_STAIR_PARTS = 4
MAX_ATTEMPTS = 400


def _split_positive(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(rng.sample(range(1, int(total)), parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [int(total)])]


def _spaced_combos(values: list[int], n: int, gap: int) -> list[tuple[int, ...]]:
    return [
        combo
        for combo in itertools.combinations(values, n)
        if all(combo[i + 1] - combo[i] >= gap for i in range(len(combo) - 1))
    ]


def _choose_endpoints(h: int, w: int, n_walls: int, rng: random.Random) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    rows = _spaced_combos(list(range(2, h - 2)), n_walls, ENDPOINT_GAP)
    cols = _spaced_combos(list(range(2, w - 2)), n_walls, ENDPOINT_GAP)
    if not rows or not cols:
        return None

    valid = []
    for start_rows in rows:
        for col_combo in cols:
            end_cols = tuple(reversed(col_combo))
            if all(h - 1 - r >= 2 and c >= 2 for r, c in zip(start_rows, end_cols)):
                valid.append((start_rows, end_cols))
    if not valid:
        return None
    return rng.choice(valid)


def _stair_path_lower_left(
    h: int, start_r: int, end_c: int, min_first_col: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    down = h - 1 - int(start_r)
    right = int(end_c)
    max_parts = min(MAX_STAIR_PARTS, down, right)
    feasible = [k for k in range(1, max_parts + 1) if right >= min_first_col + k - 1]
    if not feasible:
        return None

    parts = rng.choice(feasible)
    if parts == 1:
        horiz_runs = [right]
    else:
        first = rng.randint(int(min_first_col), right - (parts - 1))
        horiz_runs = [first] + _split_positive(right - first, parts - 1, rng)
    vert_runs = _split_positive(down, parts, rng)

    r, c = int(start_r), 0
    cells = [(r, c)]
    for h_run, v_run in zip(horiz_runs, vert_runs):
        for _ in range(h_run):
            c += 1
            cells.append((r, c))
        for _ in range(v_run):
            r += 1
            cells.append((r, c))
    if (r, c) != (h - 1, int(end_c)):
        return None
    return cells


def _build_wall_grid(h: int, w: int, bg: int, wall: int, n_walls: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(24):
        endpoints = _choose_endpoints(h, w, n_walls, rng)
        if endpoints is None:
            return None

        start_rows, end_cols = endpoints
        grid = np.full((h, w), int(bg), dtype=int)
        ok = True
        for i, (start_r, end_c) in enumerate(zip(start_rows, end_cols)):
            min_first_col = end_cols[i + 1] + ENDPOINT_GAP if i + 1 < n_walls else 1
            path = _stair_path_lower_left(h, int(start_r), int(end_c), int(min_first_col), rng)
            if path is None:
                ok = False
                break
            for r, c in path:
                if grid[r, c] != int(bg):
                    ok = False
                    break
            if not ok:
                break
            for r, c in path:
                grid[r, c] = int(wall)
        if ok:
            return grid
    return None


def _apply_transform(grid: np.ndarray, op: int) -> np.ndarray:
    if op == 0:
        return grid.copy()
    if op == 1:
        return np.rot90(grid, 1)
    if op == 2:
        return np.rot90(grid, 2)
    if op == 3:
        return np.rot90(grid, 3)
    if op == 4:
        return np.fliplr(grid)
    if op == 5:
        return np.flipud(grid)
    if op == 6:
        return np.transpose(grid)
    if op == 7:
        return np.fliplr(np.transpose(grid))
    raise ValueError(f"unknown dihedral op: {op}")


def _region_candidate_cells(labels: np.ndarray, region_id: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(labels == int(region_id))]
    scored: list[tuple[int, float, tuple[int, int]]] = []
    h, w = labels.shape
    for r, c in coords:
        same4 = 0
        for nr, nc in _neighbors4(r, c):
            if _in_bounds((h, w), nr, nc) and labels[nr, nc] == int(region_id):
                same4 += 1
        center_bias = -abs(r - (h - 1) / 2.0) - abs(c - (w - 1) / 2.0)
        scored.append((same4, center_bias, (r, c)))
    scored.sort(reverse=True)
    return [coord for _score, _bias, coord in scored]


def _is_far_from_markers(pos: tuple[int, int], placed: set[tuple[int, int]]) -> bool:
    r, c = pos
    for rr, cc in placed:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _choose_positions_for_region(
    labels: np.ndarray,
    region_id: int,
    count: int,
    placed: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    ordered = _region_candidate_cells(labels, region_id)
    if len(ordered) < count:
        return None

    chosen: list[tuple[int, int]] = []
    blocked = set(placed)
    for pos in ordered:
        if _is_far_from_markers(pos, blocked):
            chosen.append(pos)
            blocked.add(pos)
            if len(chosen) == count:
                return chosen
    return None


def _plan_party_markers(
    region_ids: list[int],
    majority_color: int,
    noise_colors: list[int],
    rng: random.Random,
) -> dict[int, Counter[int]]:
    assert len(region_ids) >= 2
    contrarian = rng.choice(region_ids)
    plan: dict[int, Counter[int]] = {}
    for region_id in region_ids:
        counts: Counter[int] = Counter()
        if region_id == contrarian:
            alt = rng.choice(noise_colors)
            counts[majority_color] = rng.choice([1, 2])
            counts[alt] = rng.choice([2, 3])
            if rng.random() < 0.25:
                extra = rng.choice([c for c in noise_colors if c != alt] or [alt])
                counts[extra] += 1
        else:
            base_majority = 3 if len(region_ids) == 2 else rng.choice([2, 3])
            counts[majority_color] = base_majority
            if rng.random() < 0.5:
                counts[rng.choice(noise_colors)] += 1
        plan[int(region_id)] = counts

    totals = Counter()
    for counts in plan.values():
        totals.update(counts)
    max_other = max((n for color, n in totals.items() if color != majority_color), default=0)
    if totals[majority_color] <= max_other:
        needed = max_other - totals[majority_color] + 1
        for region_id in region_ids:
            if region_id != contrarian:
                plan[int(region_id)][majority_color] += needed
                break
    return plan


def _render_markers(
    wall_grid: np.ndarray,
    labels: np.ndarray,
    parity: dict[int, int],
    bg: int,
    wall: int,
    fill0: int,
    fill1: int,
    noise0: int,
    noise1: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    regions_by_party = {
        0: sorted(region_id for region_id, party in parity.items() if party == 0),
        1: sorted(region_id for region_id, party in parity.items() if party == 1),
    }
    if min(len(regions_by_party[0]), len(regions_by_party[1])) < 2:
        return None

    plans = {}
    plans.update(_plan_party_markers(regions_by_party[0], fill0, [fill1, noise0, noise1], rng))
    plans.update(_plan_party_markers(regions_by_party[1], fill1, [fill0, noise0, noise1], rng))

    inp = wall_grid.copy()
    placed: set[tuple[int, int]] = set()
    for region_id in sorted(plans):
        counts = plans[region_id]
        total = sum(counts.values())
        positions = _choose_positions_for_region(labels, region_id, total, placed, rng)
        if positions is None:
            return None
        idx = 0
        color_items = list(counts.items())
        rng.shuffle(color_items)
        for color, amount in color_items:
            for _ in range(amount):
                r, c = positions[idx]
                inp[r, c] = int(color)
                placed.add((r, c))
                idx += 1

    expected = wall_grid.copy()
    for region_id, party in parity.items():
        expected[labels == int(region_id)] = int(fill0 if party == 0 else fill1)
    expected[wall_grid == int(wall)] = int(wall)

    if not np.any(inp == int(bg)):
        return None
    return inp, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_walls = rng.randint(*WALL_COUNT_RANGE)

    colors = rng.sample(list(range(10)), 6)
    bg, wall, fill0, fill1, noise0, noise1 = colors
    wall_grid = _build_wall_grid(h, w, bg, wall, n_walls, rng)
    if wall_grid is None:
        return None

    wall_grid = _apply_transform(wall_grid, rng.randrange(8))
    analysis = _analyze_wall_map(wall_grid, wall)
    if analysis is None:
        return None
    labels, parity = analysis

    rendered = _render_markers(wall_grid, labels, parity, bg, wall, fill0, fill1, noise0, noise1, rng)
    if rendered is None:
        return None
    inp, expected = rendered

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        seed = random.randrange(1 << 30)
        grid = _gen_attempt(random.Random(seed))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 3")
