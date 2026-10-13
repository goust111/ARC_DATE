from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, in_bounds, neighbors4


BG = 0
OBSTACLE = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GRID_H_RANGE = (19, 26)
GRID_W_RANGE = (19, 26)
OBSTACLE_COUNT_RANGE = (3, 5)
OBSTACLE_AREA_RANGE = (5, 12)
PAIR_COUNT_WEIGHTS = [3, 3, 4, 4, 5]
PAIR_DISTANCE_RANGE = (10, 32)
MAX_GENERATION_ATTEMPTS = 400
MAX_COMPONENT_ATTEMPTS = 80
MAX_CANDIDATE_ATTEMPTS = 500


Coord = tuple[int, int]
Pair = tuple[int, Coord, Coord]


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _sorted_pair_cells(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    return tuple(sorted((tuple(map(int, a)), tuple(map(int, b)))))  # type: ignore[return-value]


def _pair_sort_key(color: int, a: Coord, b: Coord) -> tuple[int, int, int, int, int]:
    p0, p1 = _sorted_pair_cells(a, b)
    return (_manhattan(a, b) - 1, p0[0], p0[1], p1[0], p1[1])


def _connector_pairs(grid: np.ndarray) -> list[Pair]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = [int(c) for c in np.unique(g) if int(c) not in {bg, OBSTACLE}]
    pairs: list[Pair] = []
    for color in colors:
        pts = np.argwhere(g == color)
        assert pts.shape[0] == 2, f"color {color} must appear exactly twice as connectors"
        a = (int(pts[0, 0]), int(pts[0, 1]))
        b = (int(pts[1, 0]), int(pts[1, 1]))
        pairs.append((color, a, b))
    pairs.sort(key=lambda item: _pair_sort_key(*item))
    return pairs


def _distance_map(start: Coord, walkable: np.ndarray) -> np.ndarray:
    h, w = walkable.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= h or nc >= w or not bool(walkable[nr, nc]):
                continue
            if dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _dir_priority(cur: Coord, end: Coord, prev_dir: int) -> list[int]:
    r, c = cur
    er, ec = end
    row_dir = 0 if er < r else 2
    col_dir = 1 if ec > c else 3
    wanted = []
    if r != er:
        wanted.append(row_dir)
    if c != ec:
        wanted.append(col_dir)

    order: list[int] = []
    if prev_dir in wanted:
        order.append(prev_dir)
    row_gap = abs(er - r)
    col_gap = abs(ec - c)
    first = row_dir if row_gap >= col_gap else col_dir
    second = col_dir if first == row_dir else row_dir
    for d in (first, second):
        if d in wanted and d not in order:
            order.append(d)
    for d in wanted:
        if d not in order:
            order.append(d)
    return order


def _shortest_hugging_path(start: Coord, end: Coord, walkable: np.ndarray) -> list[Coord]:
    walk = walkable.copy()
    walk[start] = True
    walk[end] = True

    dist_s = _distance_map(start, walk)
    shortest = int(dist_s[end])
    if shortest < 0:
        raise ValueError("endpoints are disconnected")
    if shortest != _manhattan(start, end):
        raise ValueError("no conserved-length orthogonal path exists")

    dist_t = _distance_map(end, walk)
    blocked = ~walk
    blocked[start] = False
    blocked[end] = False

    hug_score = np.zeros_like(dist_s, dtype=int)
    h, w = walk.shape
    for r in range(h):
        for c in range(w):
            if not bool(walk[r, c]) or (r, c) in {start, end}:
                continue
            score = 0
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and bool(blocked[nr, nc]):
                    score += 1
            hug_score[r, c] = score

    @lru_cache(None)
    def solve(r: int, c: int, prev_dir: int) -> tuple[tuple[int, int], tuple[Coord, ...]]:
        cur = (int(r), int(c))
        if cur == end:
            return (0, 0), ()

        best_key: tuple[int, int] | None = None
        best_tail: tuple[Coord, ...] | None = None
        for dir_idx in _dir_priority(cur, end, int(prev_dir)):
            dr, dc = DIRS[dir_idx]
            nr, nc = r + dr, c + dc
            if not in_bounds(walk.shape, nr, nc) or not bool(walk[nr, nc]):
                continue
            if dist_s[nr, nc] != dist_s[r, c] + 1:
                continue
            if dist_s[nr, nc] + dist_t[nr, nc] != shortest:
                continue

            next_key, next_tail = solve(nr, nc, dir_idx)
            score = next_key[0] + (0 if (nr, nc) == end else int(hug_score[nr, nc]))
            turns = next_key[1] + (0 if prev_dir == -1 or prev_dir == dir_idx else 1)
            key = (score, -turns)
            if best_key is None or key > best_key:
                best_key = key
                best_tail = ((nr, nc),) + next_tail

        if best_key is None or best_tail is None:
            raise ValueError("no shortest continuation found")
        return best_key, best_tail

    _key, tail = solve(start[0], start[1], -1)
    path = [start, *tail]
    assert path[-1] == end
    return path


def _render_with_pairs(grid: np.ndarray, ordered_pairs: list[Pair]) -> tuple[np.ndarray, list[list[Coord]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "generator and solver assume black background"

    out = g.copy()
    paths: list[list[Coord]] = []
    for color, a, b in ordered_pairs:
        walkable = out == bg
        walkable[a] = True
        walkable[b] = True
        path = _shortest_hugging_path(a, b, walkable)
        assert len(path) - 2 == _manhattan(a, b) - 1
        for cell in path[1:-1]:
            assert out[cell] == bg, "new path pixels may not overwrite existing structure"
            out[cell] = int(color)
        paths.append(path)
    return out, paths


def _monotone_l_paths(a: Coord, b: Coord) -> list[list[Coord]]:
    if a[0] == b[0] or a[1] == b[1]:
        return []

    def walk(first_vertical: bool) -> list[Coord]:
        r, c = a
        cells = [a]
        while r != b[0]:
            r += 1 if b[0] > r else -1
            cells.append((r, c))
        while c != b[1]:
            c += 1 if b[1] > c else -1
            cells.append((r, c))
        return cells

    row_first = walk(True)
    r, c = a
    cells = [a]
    while c != b[1]:
        c += 1 if b[1] > c else -1
        cells.append((r, c))
    while r != b[0]:
        r += 1 if b[0] > r else -1
        cells.append((r, c))
    col_first = cells
    return [row_first, col_first]


def _pair_needs_reroute(occupied: np.ndarray, a: Coord, b: Coord) -> bool:
    paths = _monotone_l_paths(a, b)
    if not paths:
        return False
    blocked = occupied.copy()
    blocked[a] = False
    blocked[b] = False
    return any(any(bool(blocked[cell]) for cell in path[1:-1]) for path in paths)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored border connectors and draw conserved-length obstacle-hugging paths between them.

    Input structure:
    - Black background surrounds 3-5 irregular grey obstacle sprites in the central area.
    - Each non-grey foreground color appears exactly twice as singleton connector pixels near the border band.
    - Connector colors define endpoint pairs; grey cells and previously drawn paths act as blocking terrain.

    Rule:
    - Pair the two pixels of each connector color and compute their target path length as the Manhattan distance minus the two endpoints.
    - Process pairs from smaller target length to larger, breaking ties by the lexicographically smaller endpoint of the pair.
    - For each pair, draw a 4-connected path of that exact length without crossing obstacles or earlier paths; among conserved-length shortest routes, prefer the one that hugs blocking boundaries most closely.
    - Keep the connector endpoints and grey obstacle cells unchanged, and treat each completed path as a new obstacle for later pairs.

    Key insight:
    - The path order matters because every earlier connector path reshapes the obstacle field for later shortest boundary-hugging reroutes.
    """
    ordered_pairs = _connector_pairs(np.asarray(grid, dtype=int))
    out, _paths = _render_with_pairs(np.asarray(grid, dtype=int), ordered_pairs)
    return out


def _random_connected_mask(area: int) -> np.ndarray:
    for _ in range(MAX_COMPONENT_ATTEMPTS):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        if h * w < area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        cells = [seed]
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.append((nr, nc))
            if not frontier:
                break
            nxt = random.choice(frontier)
            mask[nxt] = True
            cells.append(nxt)

        if int(mask.sum()) != area:
            continue
        if not np.any(~mask):
            continue
        if mask.shape[0] * mask.shape[1] == area:
            continue
        return mask
    raise RuntimeError("failed to generate an irregular obstacle mask")


def _can_place_obstacle(mask: np.ndarray, grid: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    coords = np.argwhere(mask)
    for rr, cc in coords:
        r = int(r0 + rr)
        c = int(c0 + cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BG:
                    return False
    return True


def _place_obstacles(shape: tuple[int, int], count: int) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    if count == 0:
        return grid
    attempts = 0
    placed = 0
    while placed < count and attempts < MAX_GENERATION_ATTEMPTS * 20:
        attempts += 1
        area = random.randint(*OBSTACLE_AREA_RANGE)
        mask = _random_connected_mask(area)
        mh, mw = mask.shape
        r_lo = 3
        r_hi = h - mh - 4
        c_lo = 3
        c_hi = w - mw - 4
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        center_r = r0 + (mh - 1) / 2.0
        center_c = c0 + (mw - 1) / 2.0
        if not (h * 0.25 <= center_r <= h * 0.75 and w * 0.25 <= center_c <= w * 0.75):
            continue
        if not _can_place_obstacle(mask, grid, r0, c0):
            continue
        coords = np.argwhere(mask)
        for rr, cc in coords:
            grid[r0 + int(rr), c0 + int(cc)] = OBSTACLE
        placed += 1

    if placed != count:
        raise RuntimeError("failed to place all obstacles")
    return grid


def _boundary_band_cells(shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    cells = []
    for r in range(h):
        for c in range(w):
            d = min(r, c, h - 1 - r, w - 1 - c)
            if 1 <= d <= 3:
                cells.append((r, c))
    return cells


def _well_spaced(cell: Coord, chosen: list[Coord], min_l1: int = 4) -> bool:
    return all(_manhattan(cell, other) >= min_l1 for other in chosen)


def _count_obstacles(grid: np.ndarray) -> int:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[OBSTACLE], return_format="dict")
    return len(comps)


def _obstacle_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[OBSTACLE], return_format="dict")
    comps.sort(key=lambda comp: (-int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))
    return comps


def _anchor_candidates(comp: dict, limit: int = 4) -> list[Coord]:
    coords = comp["coords"]
    mean_r = float(coords[:, 0].mean())
    mean_c = float(coords[:, 1].mean())
    ranked = sorted(
        [(int(r), int(c)) for r, c in coords],
        key=lambda cell: (abs(cell[0] - mean_r) + abs(cell[1] - mean_c), cell[0], cell[1]),
    )
    return ranked[:limit]


def _path_adjacent_to_gray(path: list[Coord], obstacle_mask: np.ndarray) -> int:
    total = 0
    h, w = obstacle_mask.shape
    for cell in path[1:-1]:
        r, c = cell
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(obstacle_mask[nr, nc]):
                total += 1
    return total


def _template_endpoints(shape: tuple[int, int], anchor: Coord, template: str) -> tuple[Coord, Coord]:
    h, w = shape
    rr, cc = anchor
    band_a = random.randint(1, 3)
    band_b = random.randint(1, 3)
    if template == "TR":
        return (band_a, cc), (rr, w - 1 - band_b)
    if template == "LB":
        return (rr, band_a), (h - 1 - band_b, cc)
    if template == "TL":
        return (band_a, cc), (rr, band_b)
    if template == "RB":
        return (rr, w - 1 - band_a), (h - 1 - band_b, cc)
    raise ValueError(f"unknown template: {template}")


def _validate_candidate_pair(
    input_grid: np.ndarray,
    sim_grid: np.ndarray,
    compare_grid: np.ndarray,
    used_endpoints: list[Coord],
    min_key: tuple[int, int, int, int, int] | None,
    require_interaction: bool,
    a: Coord,
    b: Coord,
) -> tuple[Coord, Coord, list[Coord]] | None:
    if a[0] == b[0] or a[1] == b[1]:
        return None
    if input_grid[a] != BG or input_grid[b] != BG or sim_grid[a] != BG or sim_grid[b] != BG:
        return None
    if not _well_spaced(a, used_endpoints) or not _well_spaced(b, used_endpoints):
        return None

    dummy_color = 9
    key = _pair_sort_key(dummy_color, a, b)
    if min_key is not None and key <= min_key:
        return None
    if not (PAIR_DISTANCE_RANGE[0] <= key[0] <= PAIR_DISTANCE_RANGE[1]):
        return None

    obstacle_mask = input_grid == OBSTACLE
    probe_sim = sim_grid.copy()
    probe_sim[a] = dummy_color
    probe_sim[b] = dummy_color
    if not _pair_needs_reroute(probe_sim != BG, a, b):
        return None

    try:
        walkable = probe_sim == BG
        walkable[a] = True
        walkable[b] = True
        path = _shortest_hugging_path(a, b, walkable)
    except ValueError:
        return None

    if len(path) - 2 != key[0]:
        return None
    if _path_adjacent_to_gray(path, obstacle_mask) == 0:
        return None
    if set(path[1:-1]) & set(used_endpoints):
        return None

    if require_interaction:
        compare_probe = compare_grid.copy()
        compare_probe[a] = dummy_color
        compare_probe[b] = dummy_color
        try:
            base_walkable = compare_probe == BG
            base_walkable[a] = True
            base_walkable[b] = True
            base_path = _shortest_hugging_path(a, b, base_walkable)
        except ValueError:
            return None
        if base_path[1:-1] == path[1:-1]:
            return None

    return a, b, path


def _pick_component_pair(
    input_grid: np.ndarray,
    sim_grid: np.ndarray,
    compare_grid: np.ndarray,
    comp: dict,
    templates: list[str],
    used_endpoints: list[Coord],
    min_key: tuple[int, int, int, int, int] | None,
    require_interaction: bool,
) -> tuple[Coord, Coord, list[Coord]] | None:
    for anchor in _anchor_candidates(comp):
        for template in templates:
            for _ in range(6):
                a, b = _template_endpoints(input_grid.shape, anchor, template)
                picked = _validate_candidate_pair(
                    input_grid,
                    sim_grid,
                    compare_grid,
                    used_endpoints,
                    min_key,
                    require_interaction,
                    a,
                    b,
                )
                if picked is not None:
                    return picked
    return None


def _pick_next_pair(
    input_grid: np.ndarray,
    sim_grid: np.ndarray,
    compare_grid: np.ndarray,
    components: list[dict],
    used_endpoints: list[Coord],
    min_key: tuple[int, int, int, int, int] | None,
) -> tuple[Coord, Coord, list[Coord]] | None:
    comp_order = list(components)
    random.shuffle(comp_order)
    for comp in comp_order:
        templates = ["TR", "LB", "TL", "RB"]
        random.shuffle(templates)
        picked = _pick_component_pair(
            input_grid,
            sim_grid,
            compare_grid,
            comp,
            templates,
            used_endpoints,
            min_key,
            False,
        )
        if picked is not None:
            return picked
    return None


def _reverse_order_output(grid: np.ndarray) -> np.ndarray | None:
    ordered_pairs = _connector_pairs(grid)
    reversed_pairs = list(reversed(ordered_pairs))
    try:
        out, _paths = _render_with_pairs(grid, reversed_pairs)
    except ValueError:
        return None
    return out


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        out, paths = _render_with_pairs(inp, _connector_pairs(inp))
    except Exception:
        return False

    if np.array_equal(inp, out):
        return False

    obstacle_count = _count_obstacles(inp)
    if not (OBSTACLE_COUNT_RANGE[0] <= obstacle_count <= OBSTACLE_COUNT_RANGE[1]):
        return False

    pairs = _connector_pairs(inp)
    if not (3 <= len(pairs) <= 5):
        return False

    obstacle_mask = inp == OBSTACLE
    for (color, a, b), path in zip(pairs, paths):
        if len(path) - 2 != _manhattan(a, b) - 1:
            return False
        if out[a] != color or out[b] != color:
            return False
        if _path_adjacent_to_gray(path, obstacle_mask) == 0:
            return False
        if not _pair_needs_reroute(inp != BG, a, b):
            return False

    reversed_out = _reverse_order_output(inp)
    if reversed_out is None or np.array_equal(out, reversed_out):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    obstacle_count = random.randint(*OBSTACLE_COUNT_RANGE)
    pair_count = random.choice(PAIR_COUNT_WEIGHTS)
    if pair_count > len(PATH_COLORS):
        return None

    try:
        input_grid = _place_obstacles((h, w), obstacle_count)
    except RuntimeError:
        return None
    sim_grid = input_grid.copy()
    compare_grid = input_grid.copy()
    chosen_pairs: list[tuple[Coord, Coord, list[Coord]]] = []
    used_endpoints: list[Coord] = []
    last_key: tuple[int, int, int, int, int] | None = None
    components = _obstacle_components(input_grid)
    if len(components) < 3:
        return None

    paired_templates = random.choice([["TR", "LB"], ["TL", "RB"]])
    plan: list[tuple[dict, list[str], bool]] = [
        (components[0], [paired_templates[0]], False),
        (components[0], [paired_templates[1]], False),
    ]
    for comp, templates, need_interaction in plan:
        picked = _pick_component_pair(input_grid, sim_grid, compare_grid, comp, templates, used_endpoints, last_key, need_interaction)
        if picked is None:
            return None

        a, b, path = picked
        used_endpoints.extend([a, b])
        last_key = _pair_sort_key(0, a, b)
        chosen_pairs.append((a, b, path))

        input_grid[a] = 1
        input_grid[b] = 1
        sim_grid[a] = 1
        sim_grid[b] = 1
        for cell in path[1:-1]:
            sim_grid[cell] = 1

    while len(chosen_pairs) < pair_count:
        picked = _pick_next_pair(input_grid, sim_grid, compare_grid, components, used_endpoints, last_key)
        if picked is None:
            return None
        a, b, path = picked
        used_endpoints.extend([a, b])
        last_key = _pair_sort_key(0, a, b)
        chosen_pairs.append((a, b, path))
        input_grid[a] = 1
        input_grid[b] = 1
        sim_grid[a] = 1
        sim_grid[b] = 1
        for cell in path[1:-1]:
            sim_grid[cell] = 1

    colors = random.sample(PATH_COLORS, len(chosen_pairs))
    final_input = input_grid.copy()
    final_input[final_input == 1] = BG
    final_input[input_grid == OBSTACLE] = OBSTACLE

    for color, (a, b, _path) in zip(colors, chosen_pairs):
        final_input[a] = color
        final_input[b] = color

    if Counter(map(int, final_input.flatten()))[OBSTACLE] < 15:
        return None
    if not _semantic_check(final_input):
        return None
    return final_input


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connector-routing puzzle")
