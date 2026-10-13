import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
WALL = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

CELL = 3
GEN_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread red/blue seeds through black channels bounded by grey walls.

    Each colored frontier pixel extends in the four cardinal directions.
    The maximum straight-line distance for one extension equals the local
    channel width measured perpendicular to that direction. Expansion stops
    at walls or already colored pixels. After diffusion stabilizes, every
    4-neighbor red/blue contact pair is recolored yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colored = _diffuse(g, use_local_width=True)
    return _mark_contacts(colored)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted generation attempts")


def _diffuse(grid: np.ndarray, use_local_width: bool) -> np.ndarray:
    state = np.asarray(grid, dtype=int).copy()
    h, w = state.shape
    passable = state != WALL
    frontier = [
        (int(r), int(c))
        for r, c in np.argwhere((state == RED) | (state == BLUE))
    ]

    while frontier:
        proposals: dict[tuple[int, int], set[int]] = {}
        for r, c in frontier:
            color = int(state[r, c])
            if color not in (RED, BLUE):
                continue
            for dr, dc in DIRS4:
                limit = _local_width(passable, r, c, dr, dc) if use_local_width else 1
                rr, cc = r, c
                for _ in range(limit):
                    rr += dr
                    cc += dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    if not bool(passable[rr, cc]):
                        break
                    if int(state[rr, cc]) != BG:
                        break
                    proposals.setdefault((rr, cc), set()).add(color)

        next_frontier = []
        for (r, c), colors in proposals.items():
            if int(state[r, c]) != BG:
                continue
            if len(colors) == 1:
                color = next(iter(colors))
                state[r, c] = color
                next_frontier.append((r, c))
        frontier = next_frontier

    return state


def _mark_contacts(state: np.ndarray) -> np.ndarray:
    out = np.asarray(state, dtype=int).copy()
    h, w = out.shape
    touch = set()
    for r, c in np.argwhere(out == RED):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) == BLUE:
                touch.add((r, c))
                touch.add((nr, nc))
    for r, c in touch:
        out[r, c] = YELLOW
    return out


def _local_width(passable: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    if dr != 0:
        return 1 + _run_length(passable, r, c, 0, -1) + _run_length(passable, r, c, 0, 1)
    return 1 + _run_length(passable, r, c, -1, 0) + _run_length(passable, r, c, 1, 0)


def _run_length(passable: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = passable.shape
    count = 0
    rr = r + dr
    cc = c + dc
    while 0 <= rr < h and 0 <= cc < w and bool(passable[rr, cc]):
        count += 1
        rr += dr
        cc += dc
    return count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    coarse_rows = rng.randint(3, 5)
    coarse_cols = rng.randint(4, 6)
    row_gap = rng.choice([2, 3])
    col_gap = rng.choice([2, 3])
    margin_r = rng.choice([1, 2])
    margin_c = rng.choice([1, 2])

    base_rows = [margin_r + i * (CELL + row_gap) for i in range(coarse_rows)]
    base_cols = [margin_c + j * (CELL + col_gap) for j in range(coarse_cols)]
    height = base_rows[-1] + CELL + margin_r
    width = base_cols[-1] + CELL + margin_c

    collision_case = rng.random() < 0.5
    coarse = _build_coarse_components(coarse_rows, coarse_cols, collision_case, rng)
    if coarse is None:
        return None

    coarse_components, edge_specs = coarse
    if collision_case:
        widths = {edge_specs[edge]["width"] for edge in coarse_components[0]["edges"]}
        if len(widths) < 2:
            return None

    grid = np.full((height, width), WALL, dtype=int)
    for comp in coarse_components:
        for r, c in comp["nodes"]:
            rr0 = base_rows[r]
            cc0 = base_cols[c]
            grid[rr0 : rr0 + CELL, cc0 : cc0 + CELL] = BG

    for edge, spec in edge_specs.items():
        (r1, c1), (r2, c2) = edge
        width_band = int(spec["width"])
        offset = int(spec["offset"])
        if r1 == r2:
            rr0 = base_rows[r1] + offset
            cc0 = base_cols[min(c1, c2)]
            cc1 = base_cols[max(c1, c2)] + CELL
            grid[rr0 : rr0 + width_band, cc0:cc1] = BG
        else:
            rr0 = base_rows[min(r1, r2)]
            rr1 = base_rows[max(r1, r2)] + CELL
            cc0 = base_cols[c1] + offset
            grid[rr0:rr1, cc0 : cc0 + width_band] = BG

    black_components = _components_mask(grid == BG)
    if collision_case:
        if len(black_components) != 1:
            return None
    else:
        if len(black_components) != 2:
            return None

    seeded = grid.copy()
    if collision_case:
        comp = black_components[0]
        if len(comp) < 24:
            return None
        a, b, dist_ab = _farthest_pair(comp)
        if dist_ab < 8:
            return None
        seeded[a] = RED
        seeded[b] = BLUE

        extra_count = rng.choice([0, 1])
        for _ in range(extra_count):
            extra = _pick_extra_seed(comp, [a, b], rng)
            if extra is None:
                break
            seeded[extra] = rng.choice([RED, BLUE])
    else:
        black_components.sort(key=len, reverse=True)
        comp_r, comp_b = black_components[:2]
        if min(len(comp_r), len(comp_b)) < 12:
            return None
        red_seed, _, _ = _farthest_pair(comp_r)
        blue_seed, _, _ = _farthest_pair(comp_b)
        seeded[red_seed] = RED
        seeded[blue_seed] = BLUE

        if rng.random() < 0.4:
            if rng.random() < 0.5:
                extra = _pick_extra_seed(comp_r, [red_seed], rng)
                if extra is not None:
                    seeded[extra] = RED
            else:
                extra = _pick_extra_seed(comp_b, [blue_seed], rng)
                if extra is not None:
                    seeded[extra] = BLUE

    diffused = _diffuse(seeded, use_local_width=True)
    out = _mark_contacts(diffused)
    if collision_case:
        if not bool(np.any(out == YELLOW)):
            return None
        uniform_out = _mark_contacts(_diffuse(seeded, use_local_width=False))
        if np.array_equal(out, uniform_out):
            return None
    else:
        if bool(np.any(out == YELLOW)):
            return None

    colored_cells = int(np.count_nonzero((out == RED) | (out == BLUE) | (out == YELLOW)))
    if colored_cells < 18:
        return None

    return seeded


def _build_coarse_components(
    rows: int, cols: int, collision_case: bool, rng: random.Random
) -> tuple[list[dict], dict] | None:
    occupied: set[tuple[int, int]] = set()
    components = []

    if collision_case:
        sizes = [rng.randint(6, min(rows * cols, 11))]
    else:
        if rows * cols < 8:
            return None
        first = rng.randint(3, min(6, rows * cols - 3))
        second = rng.randint(3, min(6, rows * cols - first))
        sizes = [first, second]

    for size in sizes:
        comp = _grow_component(rows, cols, occupied, size, rng)
        if comp is None:
            return None
        occupied |= comp["nodes"]
        components.append(comp)

    edge_specs = {}
    for comp in components:
        for edge in comp["edges"]:
            if edge in edge_specs:
                continue
            width_band = rng.choice([1, 2, 3])
            edge_specs[edge] = {
                "width": width_band,
                "offset": rng.randint(0, CELL - width_band),
            }

    return components, edge_specs


def _grow_component(
    rows: int, cols: int, blocked: set[tuple[int, int]], target_size: int, rng: random.Random
) -> dict | None:
    available = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked]
    if len(available) < target_size:
        return None

    start = rng.choice(available)
    nodes = {start}
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while len(nodes) < target_size:
        candidates = []
        for node in nodes:
            for nbr in _coarse_neighbors(node, rows, cols):
                if nbr in nodes or nbr in blocked:
                    continue
                candidates.append((node, nbr))
        if not candidates:
            return None
        a, b = rng.choice(candidates)
        nodes.add(b)
        edges.add(_norm_edge(a, b))

    extra_candidates = []
    for node in nodes:
        for nbr in _coarse_neighbors(node, rows, cols):
            if nbr in nodes:
                edge = _norm_edge(node, nbr)
                if edge not in edges:
                    extra_candidates.append(edge)
    rng.shuffle(extra_candidates)
    extra_budget = rng.randint(0, min(2, len(extra_candidates)))
    for edge in extra_candidates[:extra_budget]:
        edges.add(edge)

    return {"nodes": nodes, "edges": edges}


def _coarse_neighbors(node: tuple[int, int], rows: int, cols: int):
    r, c = node
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)


def _norm_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _farthest_pair(component: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], int]:
    start = component[0]
    a, _ = _farthest_from(component, start)
    b, dist_map = _farthest_from(component, a)
    return a, b, int(dist_map[b])


def _farthest_from(
    component: list[tuple[int, int]], start: tuple[int, int]
) -> tuple[tuple[int, int], dict[tuple[int, int], int]]:
    cell_set = set(component)
    dist = {start: 0}
    q = deque([start])
    farthest = start
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nbr = (r + dr, c + dc)
            if nbr in cell_set and nbr not in dist:
                dist[nbr] = dist[(r, c)] + 1
                q.append(nbr)
                if dist[nbr] > dist[farthest]:
                    farthest = nbr
    return farthest, dist


def _pick_extra_seed(
    component: list[tuple[int, int]], existing: list[tuple[int, int]], rng: random.Random
) -> tuple[int, int] | None:
    cell_set = set(component)
    dist_maps = []
    for seed in existing:
        if seed not in cell_set:
            return None
        _, dist_map = _farthest_from(component, seed)
        dist_maps.append(dist_map)

    candidates = []
    for cell in component:
        if cell in existing:
            continue
        min_dist = min(dist_map.get(cell, -1) for dist_map in dist_maps)
        if min_dist >= 4:
            candidates.append((min_dist, cell))
    if not candidates:
        return None
    best_gap = max(score for score, _ in candidates)
    pool = [cell for score, cell in candidates if score >= best_gap - 1]
    return rng.choice(pool)

