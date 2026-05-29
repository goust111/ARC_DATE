# Combo variant of 7b5033c1 (index 14)
# combo_spec: D6/0+D7/0
# Name: 八连通遍历+噪声过滤

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    Original task: every non-background cell belongs to one 4-connected snake, so
#    `pts = all non-bg`, degree is counted in 4-neighbors, and traversal simply
#    walks from one endpoint to the other.
#    Fused task: the true object is an 8-connected snake, and extra singleton noise
#    pixels are scattered in the grid. Some noise pixels are only diagonally
#    adjacent to the snake, so under 8-connectivity they merge into the same raw
#    component even though they are still visually isolated one-cell distractors.
#    Example:
#      path = [(1,1),(2,2),(3,3),(4,4)], noise = [(1,3),(0,0)]
#    Old logic:
#      - 4-neighbor degree sees the diagonal path as disconnected;
#      - collecting all non-bg cells also includes both noise pixels;
#      - endpoint logic and traversal no longer reflect the intended strip.
#    New logic:
#      - build the 8-neighbor graph of all non-bg cells;
#      - in each component, recover the longest leaf-to-leaf path (tree diameter);
#      - this diameter is the real snake, while singleton diagonal leaves are
#        filtered out as off-diameter branches;
#      - start from the top-left endpoint of that recovered snake and output colors
#        as a single column.
#    The synergy is deliberate: 8-connectivity is what allows diagonal snake steps,
#    and the same 8-connectivity is what makes diagonal noise harder, so filtering
#    must reason about graph structure rather than component membership alone.
# b) Complexity calibration:
#    The generator forces several diagonal snake steps and at least two diagonal
#    leaf noises attached to interior snake cells. This prevents degeneration into
#    either the original clean 4-connected task or the simpler "largest component"
#    D7-only behavior. Noise count stays small (3-5 total) so the image is still
#    readable and the main cue remains a single dominant snake.
# c) Implicit-assumption audit of the original main():
#    - `pts = all non-bg cells`: broken, because noise exists.
#    - 4-neighbor degree counting: broken, because real path links can be diagonal.
#    - `ends = degree==1` over all non-bg cells: broken, because noise also creates
#      degree-1 or degree-0 cells.
#    - traversal over 4-neighbors only: broken, misses diagonal snake edges.
#    - iterating for `len(pts)`: broken, because noise cells should not be emitted.
#    The fused solver therefore extracts the longest 8-neighbor leaf-to-leaf path
#    and traverses only that path.
# d) Placement legality:
#    The snake is generated as a self-avoiding 8-connected walk where each new cell
#    may 8-touch only its predecessor, so the snake itself is a simple chain.
#    Attached noise is placed on background cells that are not 4-neighbor to any
#    occupied cell and 8-touch exactly one interior snake cell; isolated noise
#    8-touches nothing. This prevents overlap, out-of-bounds placement, and branch
#    ambiguity beyond the intended one-pixel leaf distractors.
# e) Visual observability:
#    Across samples, humans can see diagonal snake segments and also see small
#    singleton pixels that do not fit the main left-to-right / top-to-bottom snake
#    flow. Because some distractors are earlier than the snake start, "read every
#    non-bg pixel" is visibly wrong.
# f) Visual cleanliness:
#    Noise remains one-cell wide and sparse. No cell is shared, no object stacks on
#    top of another, and noise is kept away from 4-neighbor contact so the picture
#    still looks like one dominant snake with a few stray pixels nearby.
# g) Color inferability:
#    No color remapping is introduced. Noise reuses the same palette as the snake,
#    so filtering must be structural rather than color-based, which is inferable
#    from examples because output preserves only the ordered snake colors.
# h) Quantitative non-degeneracy target:
#    Generated samples are required to have both `diag_steps >= 6` and
#    `attached_noise >= 2`. Those two counts are later measured directly, because
#    `main(input) == output` alone would not prove the combo behavior actually
#    occurred.
# i) New solver assumptions and generator guarantees:
#    - Assumption: each relevant 8-component is a tree whose diameter is the snake.
#      Guarantee: the snake is a simple chain and every attached noise pixel is a
#      single leaf off an interior snake cell.
#    - Assumption: the intended snake is the unique longest recovered diameter.
#      Guarantee: snake length is 20-28, while noise branches have length 1 only.
#    - Assumption: choosing the lexicographically smallest diameter endpoint fixes
#      the output orientation.
#      Guarantee: the snake has exactly two true endpoints.


DIRS8: list[tuple[int, int]] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
DIAG_DIRS: list[tuple[int, int]] = [(-1, -1), (-1, 1), (1, 1), (1, -1)]


def _neighbors8(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [(r + dr, c + dc) for dr, dc in DIRS8]


def _neighbors4(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components8(points: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(points)
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        remaining.remove(start)
        stack = [start]
        comp = [start]
        while stack:
            cur = stack.pop()
            for nxt in _neighbors8(cur):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
        comps.append(comp)
    return comps


def _adjacency(component: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    pset = set(component)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for p in component:
        adj[p] = sorted([q for q in _neighbors8(p) if q in pset])
    return adj


def _bfs_tree(
    start: tuple[int, int],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    dist = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nxt in adj[cur]:
            if nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            parent[nxt] = cur
            queue.append(nxt)
    return dist, parent


def _reconstruct_path(
    end: tuple[int, int],
    parent: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _diameter_path(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not component:
        return []
    if len(component) == 1:
        return [component[0]]

    adj = _adjacency(component)
    leaves = sorted([p for p in component if len(adj[p]) <= 1])
    best: list[tuple[int, int]] = []

    for start in leaves:
        dist, parent = _bfs_tree(start, adj)
        for end in leaves:
            if end not in dist:
                continue
            path = _reconstruct_path(end, parent)
            if path[0] > path[-1]:
                path = list(reversed(path))
            if len(path) > len(best):
                best = path
            elif len(path) == len(best) and tuple(path) < tuple(best):
                best = path
    return best


def _extract_main_path(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pts = {(int(r), int(c)) for r, c in np.argwhere(grid != bg)}
    if not pts:
        return []

    best_path: list[tuple[int, int]] = []
    best_comp_size = -1
    for comp in _components8(pts):
        path = _diameter_path(comp)
        if len(path) > len(best_path):
            best_path = path
            best_comp_size = len(comp)
        elif len(path) == len(best_path):
            if len(comp) > best_comp_size:
                best_path = path
                best_comp_size = len(comp)
            elif len(comp) == best_comp_size and tuple(path) < tuple(best_path):
                best_path = path
    return best_path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - non-background cells contain one multicolor 8-connected snake plus sparse
      singleton noise pixels;
    - some noise pixels may diagonally 8-touch the snake, so raw component
      extraction is insufficient;
    - keep only the longest leaf-to-leaf path in the 8-neighbor graph, start from
      its top-left endpoint, and output its colors as an N x 1 column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    path = _extract_main_path(g, bg)
    if not path:
        return np.zeros((0, 1), dtype=int)

    cols = [int(g[r, c]) for r, c in path]
    return np.array(cols, dtype=int).reshape((-1, 1))


def _legal_next(
    path_set: set[tuple[int, int]],
    cur: tuple[int, int],
    nxt: tuple[int, int],
    h: int,
    w: int,
) -> bool:
    r, c = nxt
    if r < 1 or r >= h - 1 or c < 1 or c >= w - 1:
        return False
    if nxt in path_set:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            cell = (rr, cc)
            if cell == nxt or cell == cur:
                continue
            if cell in path_set:
                return False
    return True


def _build_path(target_len: int, h: int, w: int) -> tuple[list[tuple[int, int]], int, int] | None:
    start = (random.randint(3, h - 4), random.randint(3, w - 4))
    path = [start]
    path_set = {start}
    turns = 0
    diag_steps = 0
    prev_dir: tuple[int, int] | None = None

    while len(path) < target_len:
        dirs = list(DIAG_DIRS)
        random.shuffle(dirs)
        chosen: tuple[int, int] | None = None
        for dr, dc in dirs:
            nxt = (path[-1][0] + dr, path[-1][1] + dc)
            if _legal_next(path_set, path[-1], nxt, h, w):
                chosen = (dr, dc)
                break
        if chosen is None:
            return None

        if prev_dir is not None and chosen != prev_dir:
            turns += 1
        prev_dir = chosen
        diag_steps += 1
        nxt = (path[-1][0] + chosen[0], path[-1][1] + chosen[1])
        path.append(nxt)
        path_set.add(nxt)

    if turns < 6 or diag_steps < 6:
        return None

    deg = {p: sum(1 for q in _neighbors8(p) if q in path_set) for p in path}
    if sum(1 for d in deg.values() if d == 1) != 2:
        return None
    if any(d > 2 for d in deg.values()):
        return None
    return path, turns, diag_steps


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i in range(num_parts) if parts[i] < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def _candidate_noise(
    h: int,
    w: int,
    path_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    path_deg: dict[tuple[int, int], int],
    require_attach: bool,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in occupied:
                continue
            if any(q in occupied for q in _neighbors4(cell)):
                continue
            n8 = [q for q in _neighbors8(cell) if q in occupied]
            if require_attach:
                if len(n8) != 1:
                    continue
                if n8[0] not in path_set:
                    continue
                if path_deg[n8[0]] != 2:
                    continue
            else:
                if n8:
                    continue
            out.append(cell)
    return out


def _place_noise(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    palette: list[int],
) -> tuple[int, int] | None:
    h, w = map(int, grid.shape)
    path_set = set(path)
    path_deg = {p: sum(1 for q in _neighbors8(p) if q in path_set) for p in path}
    occupied = set(path_set)
    chosen: list[tuple[int, int]] = []
    start = min(path[0], path[-1])

    attach_target = random.randint(2, 4)
    iso_target = random.randint(1, 2)
    early_pool = _candidate_noise(h, w, path_set, occupied, path_deg, True) + _candidate_noise(
        h, w, path_set, occupied, path_deg, False
    )
    early_pool = [cell for cell in early_pool if cell < start]
    if early_pool:
        first = random.choice(early_pool)
        chosen.append(first)
        occupied.add(first)

    attached = sum(1 for cell in chosen if any(q in path_set for q in _neighbors8(cell)))
    isolated = len(chosen) - attached

    while attached < attach_target:
        pool = _candidate_noise(h, w, path_set, occupied, path_deg, True)
        if not pool:
            return None
        pick = random.choice(pool)
        chosen.append(pick)
        occupied.add(pick)
        attached += 1

    while isolated < iso_target:
        pool = _candidate_noise(h, w, path_set, occupied, path_deg, False)
        if not pool:
            break
        pick = random.choice(pool)
        chosen.append(pick)
        occupied.add(pick)
        isolated += 1

    if attached < 2 or len(chosen) < 3:
        return None

    for cell in chosen:
        grid[cell] = random.choice(palette)
    return attached, len(chosen)


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    bg = random.randint(0, 9)
    palette_size = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], palette_size)
    target_len = random.randint(20, 28)

    built = _build_path(target_len, h, w)
    if built is None:
        return None
    path, _, diag_steps = built
    total = len(path)

    num_segments = random.randint(max(4, palette_size), min(7, total // 2))
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=8)
    if seg_lengths is None:
        return None

    shuffled_palette = list(palette)
    random.shuffle(shuffled_palette)
    seg_colors = list(shuffled_palette[: min(num_segments, len(shuffled_palette))])
    while len(seg_colors) < num_segments:
        prev = seg_colors[-1] if seg_colors else None
        choices = [c for c in palette if c != prev]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    noise_stats = _place_noise(grid, path, palette)
    if noise_stats is None:
        return None
    attached_noise, total_noise = noise_stats
    if attached_noise < 2 or total_noise < 3 or diag_steps < 6:
        return None

    expected_colors = colors if min(path[0], path[-1]) == path[0] else list(reversed(colors))
    expected_out = np.array(expected_colors, dtype=int).reshape((-1, 1))
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate fused 8-connected noisy input after 400 attempts")
