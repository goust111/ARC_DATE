from __future__ import annotations

import random

import numpy as np


BG = 8
WALL = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 9]
GENERATION_ATTEMPTS = 500
ROOM_ROWS_RANGE = (3, 5)
ROOM_COLS_RANGE = (3, 5)
CELL_SIZE_CHOICES = (3, 4)
REGIONS_RANGE = (3, 6)
MAX_DOORS = 5


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _simulate_fill(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    tie_mask = np.zeros((h, w), dtype=bool)
    processed = np.zeros((h, w), dtype=bool)
    arrival = np.full((h, w), -1, dtype=int)

    seeds: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for r0, c0 in np.argwhere((g != BG) & (g != WALL)):
        r = int(r0)
        c = int(c0)
        color = int(g[r, c])
        assert color not in seen_colors, "each seed color must be unique"
        seen_colors.add(color)
        seeds.append((r, c, color))
        processed[r, c] = True
        arrival[r, c] = 0

    frontier = seeds
    step = 0
    while frontier:
        proposals: dict[tuple[int, int], set[int]] = {}
        for r, c, color in frontier:
            for nr, nc in _neighbors4(r, c, h, w):
                if processed[nr, nc]:
                    continue
                if out[nr, nc] != BG:
                    continue
                proposals.setdefault((nr, nc), set()).add(color)

        next_frontier: list[tuple[int, int, int]] = []
        for (r, c), colors in proposals.items():
            processed[r, c] = True
            arrival[r, c] = step + 1
            if len(colors) == 1:
                color = next(iter(colors))
                out[r, c] = color
                next_frontier.append((r, c, color))
            else:
                tie_mask[r, c] = True

        frontier = next_frontier
        step += 1

    return out, tie_mask, arrival


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each seed through the door-connected chambers while simultaneous collisions stay azure.

    Input structure:
    - Azure is the background and Grey cells are fixed wall segments.
    - The walls partition the grid into enclosed chambers, and each chamber contains exactly one single-pixel seed in a distinct non-Azure, non-Grey color.
    - Some wall segments contain a one-pixel door gap that lets adjacent chambers connect.

    Rule:
    - Start a synchronous 4-neighbor expansion from every seed at once.
    - Grey wall cells are never recolored and cannot be crossed; only door gaps can connect two chambers.
    - When exactly one color reaches an unprocessed Azure cell in a round, paint that cell with the arriving color and continue expanding from it in later rounds.
    - When two or more colors reach the same Azure cell in the same round, leave that cell Azure forever and stop further spreading through it.

    Key insight:
    - The fill is a simultaneous wavefront process over the door-defined connectivity graph, so solving the task requires reasoning about both wall openings and equal-time collisions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit inside 30x30"
    assert np.any(g == WALL), "expected grey wall segments"
    assert np.any(g == BG), "expected azure background cells"
    assert np.any((g != BG) & (g != WALL)), "expected at least one seed"
    out, _, _ = _simulate_fill(g)
    return out


def _cell_start(index: int, span: int) -> int:
    return 1 + index * (span + 1)


def _grow_regions(room_h: int, room_w: int, n_regions: int, rng: random.Random) -> np.ndarray:
    coords = [(r, c) for r in range(room_h) for c in range(room_w)]
    seeds = rng.sample(coords, n_regions)
    labels = np.full((room_h, room_w), -1, dtype=int)
    frontiers: dict[int, list[tuple[int, int]]] = {}
    for region_id, (r, c) in enumerate(seeds):
        labels[r, c] = region_id
        frontiers[region_id] = [(r, c)]

    while np.any(labels == -1):
        proposals: dict[tuple[int, int], set[int]] = {}
        for region_id, frontier in frontiers.items():
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c, room_h, room_w):
                    if labels[nr, nc] == -1:
                        proposals.setdefault((nr, nc), set()).add(region_id)
        if not proposals:
            return labels

        items = list(proposals.items())
        rng.shuffle(items)
        next_frontiers = {region_id: [] for region_id in range(n_regions)}
        for (r, c), region_ids in items:
            if labels[r, c] != -1:
                continue
            region_id = rng.choice(tuple(region_ids))
            labels[r, c] = region_id
            next_frontiers[region_id].append((r, c))
        frontiers = next_frontiers

    return labels


def _has_irregular_region(labels: np.ndarray) -> bool:
    for region_id in np.unique(labels):
        coords = np.argwhere(labels == region_id)
        if len(coords) < 3:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if bbox_area != int(len(coords)):
            return True
    return False


def _same_region_components(owner: np.ndarray, region_id: int) -> int:
    mask = owner == region_id
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _door_candidates(
    labels: np.ndarray,
    cell_h: int,
    cell_w: int,
) -> dict[tuple[int, int], list[dict[str, object]]]:
    room_h, room_w = labels.shape
    pair_to_edges: dict[tuple[int, int], list[dict[str, object]]] = {}

    for r in range(room_h):
        row0 = _cell_start(r, cell_h)
        for c in range(room_w - 1):
            left = int(labels[r, c])
            right = int(labels[r, c + 1])
            if left == right:
                continue
            col = (c + 1) * (cell_w + 1)
            rows = list(range(row0 + 1, row0 + cell_h - 1))
            if not rows:
                continue
            pair = tuple(sorted((left, right)))
            pair_to_edges.setdefault(pair, []).append(
                {
                    "kind": "vertical",
                    "line": col,
                    "positions": [(rr, col) for rr in rows],
                }
            )

    for r in range(room_h - 1):
        row = (r + 1) * (cell_h + 1)
        for c in range(room_w):
            top = int(labels[r, c])
            bottom = int(labels[r + 1, c])
            if top == bottom:
                continue
            col0 = _cell_start(c, cell_w)
            cols = list(range(col0 + 1, col0 + cell_w - 1))
            if not cols:
                continue
            pair = tuple(sorted((top, bottom)))
            pair_to_edges.setdefault(pair, []).append(
                {
                    "kind": "horizontal",
                    "line": row,
                    "positions": [(row, cc) for cc in cols],
                }
            )

    return pair_to_edges


def _choose_doors(
    pair_to_edges: dict[tuple[int, int], list[dict[str, object]]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    region_pairs = list(pair_to_edges)
    if len(region_pairs) < 2:
        return []

    door_count = rng.randint(2, min(MAX_DOORS, len(region_pairs)))
    chosen_pairs = rng.sample(region_pairs, door_count)
    doors: list[tuple[int, int]] = []
    for pair in chosen_pairs:
        edge = rng.choice(pair_to_edges[pair])
        doors.append(rng.choice(edge["positions"]))  # type: ignore[arg-type]
    return doors


def _render_layout(
    labels: np.ndarray,
    cell_h: int,
    cell_w: int,
    doors: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, dict[int, list[tuple[int, int]]]]:
    room_h, room_w = labels.shape
    grid_h = room_h * cell_h + room_h + 1
    grid_w = room_w * cell_w + room_w + 1
    grid = np.full((grid_h, grid_w), WALL, dtype=int)
    owner = np.full((grid_h, grid_w), -1, dtype=int)
    seed_pools: dict[int, list[tuple[int, int]]] = {int(rid): [] for rid in np.unique(labels)}

    for r in range(room_h):
        row0 = _cell_start(r, cell_h)
        for c in range(room_w):
            col0 = _cell_start(c, cell_w)
            region_id = int(labels[r, c])
            grid[row0 : row0 + cell_h, col0 : col0 + cell_w] = BG
            owner[row0 : row0 + cell_h, col0 : col0 + cell_w] = region_id
            seed_pools[region_id].extend(
                (rr, cc)
                for rr in range(row0, row0 + cell_h)
                for cc in range(col0, col0 + cell_w)
            )

    for r in range(room_h):
        row0 = _cell_start(r, cell_h)
        for c in range(room_w - 1):
            left = int(labels[r, c])
            right = int(labels[r, c + 1])
            if left != right:
                continue
            col = (c + 1) * (cell_w + 1)
            grid[row0 : row0 + cell_h, col] = BG
            owner[row0 : row0 + cell_h, col] = left

    for r in range(room_h - 1):
        row = (r + 1) * (cell_h + 1)
        for c in range(room_w):
            top = int(labels[r, c])
            bottom = int(labels[r + 1, c])
            if top != bottom:
                continue
            col0 = _cell_start(c, cell_w)
            grid[row, col0 : col0 + cell_w] = BG
            owner[row, col0 : col0 + cell_w] = top

    for r in range(1, room_h):
        row = r * (cell_h + 1)
        for c in range(1, room_w):
            col = c * (cell_w + 1)
            ids = {
                int(labels[r - 1, c - 1]),
                int(labels[r - 1, c]),
                int(labels[r, c - 1]),
                int(labels[r, c]),
            }
            if len(ids) == 1:
                region_id = next(iter(ids))
                grid[row, col] = BG
                owner[row, col] = region_id

    for r, c in doors:
        grid[r, c] = BG

    return grid, owner, seed_pools


def _place_seeds(
    grid: np.ndarray,
    labels: np.ndarray,
    seed_pools: dict[int, list[tuple[int, int]]],
    rng: random.Random,
) -> dict[int, int]:
    region_ids = sorted(int(x) for x in np.unique(labels))
    colors = rng.sample(SEED_COLORS, len(region_ids))
    region_colors: dict[int, int] = {}
    used: set[tuple[int, int]] = set()
    for region_id, color in zip(region_ids, colors):
        pool = [cell for cell in seed_pools[region_id] if cell not in used]
        if not pool:
            raise RuntimeError("seed placement pool unexpectedly empty")
        r, c = rng.choice(pool)
        grid[r, c] = color
        used.add((r, c))
        region_colors[region_id] = color
    return region_colors


def _cross_region_capture_count(
    out: np.ndarray,
    owner: np.ndarray,
    region_colors: dict[int, int],
) -> int:
    captured = 0
    for region_id, own_color in region_colors.items():
        vals = {int(x) for x in np.unique(out[owner == region_id])}
        foreign = vals - {BG, own_color}
        if foreign:
            captured += 1
    return captured


def _door_is_used(out: np.ndarray, tie_mask: np.ndarray, doors: list[tuple[int, int]]) -> bool:
    return any(int(out[r, c]) != BG or bool(tie_mask[r, c]) for r, c in doors)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    room_h = rng.randint(*ROOM_ROWS_RANGE)
    room_w = rng.randint(*ROOM_COLS_RANGE)
    cell_h = rng.choice(CELL_SIZE_CHOICES)
    cell_w = rng.choice(CELL_SIZE_CHOICES)
    max_regions = min(REGIONS_RANGE[1], len(SEED_COLORS), room_h * room_w - 1)
    if max_regions < REGIONS_RANGE[0]:
        return None
    n_regions = rng.randint(REGIONS_RANGE[0], max_regions)

    labels = _grow_regions(room_h, room_w, n_regions, rng)
    if np.any(labels == -1):
        return None
    if not _has_irregular_region(labels):
        return None

    pair_to_edges = _door_candidates(labels, cell_h, cell_w)
    doors = _choose_doors(pair_to_edges, rng)
    if len(doors) < 2:
        return None

    grid, owner, seed_pools = _render_layout(labels, cell_h, cell_w, doors)
    if any(_same_region_components(owner, rid) != 1 for rid in np.unique(labels)):
        return None

    try:
        region_colors = _place_seeds(grid, labels, seed_pools, rng)
    except RuntimeError:
        return None

    out, tie_mask, _arrival = _simulate_fill(grid)
    changed = int(np.sum((grid == BG) & (out != BG)))
    if changed == 0:
        return None
    if int(tie_mask.sum()) == 0:
        return None
    if _cross_region_capture_count(out, owner, region_colors) == 0:
        return None
    if not _door_is_used(out, tie_mask, doors):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
