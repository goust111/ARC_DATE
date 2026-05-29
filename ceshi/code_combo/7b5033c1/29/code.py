# Combo variant of 7b5033c1 (index 29)
# combo_spec: D6/0+D8/0
# Name: 八连通+去噪再线性化
#
# Design analysis
# a) Concrete behavior and trace:
#    Original 7b5033c1 treats every non-background pixel as one 4-connected chain, finds a
#    top-left endpoint, and linearizes the path into an N x 1 column. That breaks twice here.
#    Example cleaned path:
#      (1,1)->(1,2)->(2,3)->(3,3)->(4,4)
#    with colors [2,2,7,7,4]. Add diagonal leaf noise at (2,2) touching only (1,1)? That would
#    be ambiguous, so the generator does not do that. Instead it adds noise to interior cells,
#    e.g. a noise pixel at (2,4) touching only path cell (3,3) diagonally. The raw input is no
#    longer a simple chain: the host cell gets 8-degree 3 and the noise pixel is a 1-degree leaf.
#    The fused solver first deletes all such diagonal leaves attached to branch nodes, then traces
#    the remaining 8-connected chain from the top-left endpoint, outputting [2,2,7,7,4]^T.
#    Differences from the original:
#    - 4-neighbor traversal would split at the diagonal steps even after denoising.
#    - Skipping the denoising step leaves a branched graph, so "trace all non-bg pixels" fails.
# b) Complexity calibration:
#    The generator forces both dimensions to matter in the same sample set:
#    - several genuine diagonal path edges, so 8-connectivity is visually necessary;
#    - 2-4 diagonal noise leaves, so the raw non-bg set is not a simple chain.
#    Noise is always length-1 and only hangs from interior path nodes. This keeps the rule
#    legible: remove obvious diagonal twigs, then read the snake. More branch types or longer
#    side paths would add a separate graph-reasoning dimension and make samples too noisy.
# c) Hidden-assumption audit of the original main():
#    - `mask = g != bg` still finds all foreground cells, but those cells now include noise.
#    - counting degree in only 4 directions no longer matches the true chain connectivity.
#    - even 8-degree endpoint detection on the raw foreground is invalid because noise creates
#      extra leaves and branch nodes.
#    - the old traversal assumes each current cell has at most one valid successor after removing
#      the previous cell; this is false before denoising.
#    The new main() therefore does:
#      foreground extraction -> iterative diagonal-leaf pruning -> simple 8-chain trace.
# d) Placement legality:
#    The generator first creates a self-avoiding 8-connected path where every new path cell may
#    8-touch only its predecessor. Then it places noise only on background cells that:
#    - are diagonal neighbors of one chosen interior host cell,
#    - 8-touch no other path cell,
#    - 8-touch no previously placed noise.
#    This guarantees no overlap, no out-of-bounds cells, and no accidental second attachments.
# e) Visual observability:
#    A human sees a multicolor snake with diagonal segments plus a few one-pixel diagonal burrs.
#    Across examples, those burrs disappear in the output while the remaining snake is straightened
#    into a color strip. Because the output is still a single column, the denoising stage is not a
#    separate visual object; it is a necessary precursor to make the linearization well-defined.
# f) Visual cleanliness:
#    There is only one meaningful object family on the input canvas: one snake and a handful of
#    isolated diagonal leaf pixels. No overlapping frames, masks, or recolor legends are added.
#    Noise count is capped and candidate cells must stay isolated from one another, so the grid
#    remains readable instead of collapsing into a dense 8-connected blob.
# g) Color-change inferability:
#    There is no recoloring rule in this combo. Path colors are preserved exactly in traversal
#    order; noise colors are irrelevant because those pixels vanish during the denoising step.
#    The generator uses repeated path colors in contiguous runs so the output looks like a strip of
#    colored bands rather than a random list, which matches the original task's visual rhythm.
# 9) New assumptions introduced by the fused main():
#    - Real path endpoints are not diagonal leaves attached to branch nodes; both endpoint edges are
#      generated as orthogonal steps, so the pruning rule never removes a true endpoint.
#    - Every removable noise pixel has 8-degree 1, is diagonally attached, and its host has degree
#      at least 3 before pruning.
#    - After pruning stabilizes, the remaining foreground is one simple 8-connected chain with two
#      endpoints and no extra self-touching contacts.
#    generate_input() enforces these assumptions and validates every sample with main().

from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS8: list[tuple[int, int]] = [
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
]
ORTH_DIRS: list[tuple[int, int]] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIAG_DIRS: list[tuple[int, int]] = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
ORTH_SET = set(ORTH_DIRS)
DIAG_SET = set(DIAG_DIRS)
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS8)}
TURN_ORDER = [0, 1, 7, 2, 6, 3, 5, 4]
INITIAL_DIR_ORDER = [2, 3, 4, 5, 6, 7, 0, 1]

HEIGHT = 16
WIDTH = 16
MIN_LEN = 22
MAX_LEN = 28
MIN_TURNS = 7
MIN_DIAG_STEPS = 6
MIN_NOISE = 2
MAX_NOISE = 4


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors8(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [(r + dr, c + dc) for dr, dc in DIRS8]


def _dir_idx(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    dr = 0 if dr == 0 else dr // abs(dr)
    dc = 0 if dc == 0 else dc // abs(dc)
    return DIR_TO_IDX[(dr, dc)]


def _turn_rank(prev_idx: int | None, next_idx: int) -> int:
    if prev_idx is None:
        return INITIAL_DIR_ORDER.index(next_idx)
    delta = (next_idx - prev_idx) % 8
    return TURN_ORDER.index(delta)


def _degree_map(points: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {p: sum(1 for q in _neighbors8(p) if q in points) for p in points}


def _remove_diagonal_noise(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    current = set(points)
    while True:
        deg = _degree_map(current)
        removable: list[tuple[int, int]] = []
        for p in current:
            if deg[p] != 1:
                continue
            q = next(q for q in _neighbors8(p) if q in current)
            dr = abs(p[0] - q[0])
            dc = abs(p[1] - q[1])
            if (dr, dc) != (1, 1):
                continue
            if deg[q] < 3:
                continue
            removable.append(p)
        if not removable:
            return current
        current.difference_update(removable)


def _is_simple_chain(points: set[tuple[int, int]]) -> bool:
    if not points:
        return True
    deg = _degree_map(points)
    if any(d > 2 or d == 0 for d in deg.values()):
        return False
    ends = [p for p, d in deg.items() if d == 1]
    if len(points) == 1:
        return True
    if len(ends) != 2:
        return False

    seen = set()
    stack = [ends[0]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for q in _neighbors8(cur):
            if q in points and q not in seen:
                stack.append(q)
    return len(seen) == len(points)


def _trace_chain(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points:
        return []
    if len(points) == 1:
        return [next(iter(points))]

    deg = _degree_map(points)
    ends = [p for p, d in deg.items() if d == 1]
    assert len(ends) == 2, "expected a simple 8-connected chain after denoising"
    start = min(ends)

    order = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start
    prev_dir: int | None = None

    for _ in range(len(points) - 1):
        candidates: list[tuple[int, tuple[int, int]]] = []
        for q in _neighbors8(cur):
            if q not in points or q == prev or q in visited:
                continue
            candidates.append((_turn_rank(prev_dir, _dir_idx(cur, q)), q))
        if not candidates:
            break
        _, nxt = min(candidates, key=lambda item: (item[0], item[1]))
        order.append(nxt)
        visited.add(nxt)
        prev_dir = _dir_idx(cur, nxt)
        prev, cur = cur, nxt

    assert len(order) == len(points), "8-connected trace did not consume the whole cleaned path"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Treat all non-background cells as a noisy 8-neighborhood graph.
    2. Repeatedly remove any degree-1 pixel whose only attachment is diagonal and whose host is
       a branch node (degree >= 3). These are the injected diagonal leaf noises.
    3. Trace the remaining simple 8-connected snake from the top-left endpoint.
    4. Output the encountered colors as an N x 1 column.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)

    pts = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}
    if not pts:
        return np.zeros((0, 1), dtype=int)

    cleaned = _remove_diagonal_noise(pts)
    assert _is_simple_chain(cleaned), "denoising did not produce a simple 8-connected chain"
    order = _trace_chain(cleaned)
    colors = [int(g[r, c]) for r, c in order]
    return np.array(colors, dtype=int).reshape((-1, 1))


def _legal_next(
    path_set: set[tuple[int, int]],
    cur: tuple[int, int],
    nxt: tuple[int, int],
    height: int,
    width: int,
) -> bool:
    r, c = nxt
    if r < 1 or r >= height - 1 or c < 1 or c >= width - 1:
        return False
    if nxt in path_set:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            cell = (rr, cc)
            if cell in (cur, nxt):
                continue
            if cell in path_set:
                return False
    return True


def _candidate_dirs(
    cur: tuple[int, int],
    prev_dir: tuple[int, int],
    turns: int,
    diag_steps: int,
) -> list[tuple[int, int]]:
    dirs = list(DIRS8)
    random.shuffle(dirs)

    def score(d: tuple[int, int]) -> tuple[int, int, int, int]:
        nr, nc = cur[0] + d[0], cur[1] + d[1]
        need_turn = 0 if turns < MIN_TURNS and d != prev_dir else 1
        need_diag = 0 if diag_steps < MIN_DIAG_STEPS and d in DIAG_SET else 1
        same_dir = 0 if d == prev_dir else 1
        center = abs(nr - height_mid) + abs(nc - width_mid)
        return (need_turn, need_diag, center, same_dir)

    height_mid = HEIGHT // 2
    width_mid = WIDTH // 2
    dirs.sort(key=score)
    return dirs


def _search_path(
    path: list[tuple[int, int]],
    path_set: set[tuple[int, int]],
    prev_dir: tuple[int, int],
    target_len: int,
    turns: int,
    diag_steps: int,
    budget: list[int],
) -> list[tuple[int, int]] | None:
    budget[0] += 1
    if budget[0] > 50000:
        return None

    if len(path) == target_len:
        if prev_dir not in ORTH_SET:
            return None
        if turns < MIN_TURNS or diag_steps < MIN_DIAG_STEPS:
            return None
        return list(path)

    remaining = target_len - len(path)
    if turns + remaining < MIN_TURNS:
        return None
    if diag_steps + remaining < MIN_DIAG_STEPS:
        return None

    cur = path[-1]
    dirs = _candidate_dirs(cur, prev_dir, turns, diag_steps)
    for d in dirs:
        if len(path) == target_len - 1 and d not in ORTH_SET:
            continue
        nxt = (cur[0] + d[0], cur[1] + d[1])
        if not _legal_next(path_set, cur, nxt, HEIGHT, WIDTH):
            continue
        path.append(nxt)
        path_set.add(nxt)
        new_turns = turns + int(d != prev_dir)
        new_diag = diag_steps + int(d in DIAG_SET)
        result = _search_path(path, path_set, d, target_len, new_turns, new_diag, budget)
        if result is not None:
            return result
        path.pop()
        path_set.remove(nxt)
    return None


def _build_path(target_len: int, height: int, width: int) -> list[tuple[int, int]] | None:
    del height, width
    for _ in range(200):
        start = (random.randint(3, HEIGHT - 4), random.randint(3, WIDTH - 4))
        first_dirs = list(ORTH_DIRS)
        random.shuffle(first_dirs)
        for d in first_dirs:
            nxt = (start[0] + d[0], start[1] + d[1])
            if not _legal_next({start}, start, nxt, HEIGHT, WIDTH):
                continue
            path = [start, nxt]
            path_set = {start, nxt}
            budget = [0]
            result = _search_path(
                path,
                path_set,
                d,
                target_len,
                turns=0,
                diag_steps=0,
                budget=budget,
            )
            if result is None:
                continue
            if sum(int(_dir_idx(result[i], result[i + 1]) in {1, 3, 5, 7}) for i in range(len(result) - 1)) < MIN_DIAG_STEPS:
                continue
            if any(result[i + 1][0] - result[i][0] != 0 and result[i + 1][1] - result[i][1] != 0 for i in range(len(result) - 1)):
                return result
    return None


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


def _build_colors(total: int, bg: int) -> list[int] | None:
    palette_size = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], palette_size)
    num_segments = random.randint(max(4, palette_size), 7)
    lengths = _random_partition(total, num_segments, min_len=2, max_len=8)
    if lengths is None:
        return None

    seg_colors = list(random.sample(palette, min(len(palette), num_segments)))
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(lengths, seg_colors):
        colors.extend([col] * seg_len)
    return colors


def _noise_candidates(
    path_set: set[tuple[int, int]],
    host: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIAG_DIRS:
        cell = (host[0] + dr, host[1] + dc)
        r, c = cell
        if r < 1 or r >= HEIGHT - 1 or c < 1 or c >= WIDTH - 1:
            continue
        if cell in occupied:
            continue
        touch_path = [q for q in _neighbors8(cell) if q in path_set]
        if touch_path != [host] and set(touch_path) != {host}:
            continue
        if any(q in occupied and q != host for q in _neighbors8(cell)):
            continue
        out.append(cell)
    return out


def _place_noise(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: list[int],
) -> int | None:
    path_set = set(path)
    occupied = set(path)
    interior_hosts = list(path[2:-2])
    random.shuffle(interior_hosts)

    want = random.randint(MIN_NOISE, MAX_NOISE)
    placed: list[tuple[int, int]] = []
    for host in interior_hosts:
        if len(placed) >= want:
            break
        candidates = _noise_candidates(path_set, host, occupied)
        if not candidates:
            continue
        choice = random.choice(candidates)
        grid[choice] = random.choice(colors)
        placed.append(choice)
        occupied.add(choice)

    if len(placed) < MIN_NOISE:
        return None
    return len(placed)


def _expected_output(path: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    expected_colors = colors if min(path[0], path[-1]) == path[0] else list(reversed(colors))
    return np.array(expected_colors, dtype=int).reshape((-1, 1))


def _components4(points: set[tuple[int, int]]) -> int:
    remaining = set(points)
    count = 0
    while remaining:
        count += 1
        start = next(iter(remaining))
        remaining.remove(start)
        stack = [start]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
    return count


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    raw_pts = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}
    clean_pts = _remove_diagonal_noise(raw_pts)
    raw_deg = _degree_map(raw_pts)
    clean_order = _trace_chain(clean_pts)
    diag_edges = sum(
        1
        for a, b in zip(clean_order, clean_order[1:])
        if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1
    )
    return {
        "raw_points": len(raw_pts),
        "clean_points": len(clean_pts),
        "noise_points": len(raw_pts) - len(clean_pts),
        "raw_branch_nodes": sum(d >= 3 for d in raw_deg.values()),
        "diag_edges": diag_edges,
        "clean_components4": _components4(clean_pts),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    target_len = random.randint(MIN_LEN, MAX_LEN)

    path = _build_path(target_len, HEIGHT, WIDTH)
    if path is None:
        return None

    colors = _build_colors(target_len, bg)
    if colors is None:
        return None

    grid = np.full((HEIGHT, WIDTH), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    noise_count = _place_noise(grid, path, sorted(set(colors)))
    if noise_count is None:
        return None

    expected = _expected_output(path, colors)
    metrics = _behavior_metrics(grid)

    if metrics["noise_points"] < MIN_NOISE:
        return None
    if metrics["raw_branch_nodes"] < 1:
        return None
    if metrics["diag_edges"] < MIN_DIAG_STEPS:
        return None
    if metrics["clean_components4"] <= 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate fused 8-connected noisy input after 300 attempts")
