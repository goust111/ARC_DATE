# Variant of 7b5033c1
# Dimension: D6 — 路径遍历连通性 (transform)
# Change: 严格4-连通（上下左右），路径中无2x2色块... → 8-连通遍历

from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Design analysis for the D6 variant
# a) Concrete behavior:
#    The input is still a single colored "snake", but consecutive pixels may now
#    meet diagonally. Example path:
#    (1,1)->(2,2)->(3,3)->(3,4)
#    Old 4-neighbor logic would see the first three cells as disconnected, so
#    endpoint detection and traversal fail. The new logic counts 8-neighbors,
#    starts from the top-left endpoint (1,1), and outputs the colors in that
#    diagonal-then-horizontal order as a single N x 1 strip.
# b) Complexity calibration:
#    The visual change must be obvious, so the generator forces several diagonal
#    steps and multiple turns. At the same time, it avoids extra self-touching
#    contacts because those would introduce a new branch/overlap ambiguity
#    dimension unrelated to D6. The only intended change is "what counts as the
#    next path pixel", not "how to resolve graph branches".
# c) Old implicit assumptions that break:
#    - degree counting over only 4 directions no longer works;
#    - endpoints must be found with 8-neighbors;
#    - traversing by checking only up/down/left/right misses diagonal steps;
#    - choosing the next cell by raw lexicographic order was only harmless
#      because the old generator guaranteed a unique 4-neighbor successor.
#    The new solver therefore traces over 8 directions and keeps a direction-
#    priority tie-breaker for robustness, even though the generator still builds
#    a simple chain with one valid unvisited successor.
# d) Placement legality:
#    The generator grows a self-avoiding walk inside a 1-cell border. Every new
#    cell may 8-touch only its predecessor, so the final occupied pixels form a
#    single valid 8-connected chain with exactly two endpoints and no accidental
#    shortcuts. This guarantees that main() sees the intended object.
# e) Visual observability:
#    Each generated sample contains multiple diagonal edges, so more than 10% of
#    samples cannot collapse back to the old 4-connected behavior. A solver that
#    ignores diagonal adjacency would visibly break on these examples.
# f) New solver assumptions and generator guarantees:
#    - main() assumes the non-background pixels form one simple 8-connected path;
#    - exactly two cells have 8-degree 1;
#    - after removing the previous/visited cell, traversal always has a single
#      intended successor.
#    generate_input() explicitly enforces these properties.


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
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS8)}
TURN_ORDER = [0, 1, 7, 2, 6, 3, 5, 4]
INITIAL_DIR_ORDER = [2, 3, 4, 5, 6, 7, 0, 1]
ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTH_SET = set(ORTH_DIRS)
DIAG_ONLY_DIRS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a colored 8-connected snake into a single-column strip.

    The object is still a simple chain with two endpoints, but consecutive cells
    may connect diagonally. Traversal starts from the top-left endpoint and then
    moves through 8-neighbors, using a turn-priority tie-breaker if needed.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)

    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return np.zeros((0, 1), dtype=int)

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for p in pts:
        deg[p] = sum(1 for q in _neighbors8(p) if q in pset)

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2, "expected a simple 8-connected path with two endpoints"
    start = min(ends)

    order = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start
    prev_dir: int | None = None

    for _ in range(len(pts) - 1):
        candidates: list[tuple[int, tuple[int, int]]] = []
        for q in _neighbors8(cur):
            if q not in pset or q == prev or q in visited:
                continue
            candidates.append((_turn_rank(prev_dir, _dir_idx(cur, q)), q))

        if not candidates:
            break

        _, nxt = min(candidates, key=lambda item: (item[0], item[1]))
        order.append(nxt)
        visited.add(nxt)
        prev_dir = _dir_idx(cur, nxt)
        prev, cur = cur, nxt

    assert len(order) == len(pts), "8-connected trace did not consume the whole path"
    cols = [int(g[r, c]) for r, c in order]
    return np.array(cols, dtype=int).reshape((-1, 1))


def _legal_next(
    path_set: set[tuple[int, int]],
    cur: tuple[int, int],
    nxt: tuple[int, int],
    H: int,
    W: int,
) -> bool:
    r, c = nxt
    if r < 1 or r >= H - 1 or c < 1 or c >= W - 1:
        return False
    if nxt in path_set:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in (cur, nxt):
                continue
            if (rr, cc) in path_set:
                return False
    return True


def _build_path(target_len: int, H: int, W: int) -> list[tuple[int, int]] | None:
    start = (random.randint(2, H - 3), random.randint(2, W - 3))
    path = [start]
    path_set = {start}
    turns = 0
    prev_dir: tuple[int, int] | None = None

    while len(path) < target_len:
        dirs = list(DIAG_ONLY_DIRS)
        random.shuffle(dirs)
        chosen: tuple[int, int] | None = None
        for dr, dc in dirs:
            nxt = (path[-1][0] + dr, path[-1][1] + dc)
            if _legal_next(path_set, path[-1], nxt, H, W):
                chosen = (dr, dc)
                break

        if chosen is None:
            return None

        if prev_dir is not None and chosen != prev_dir:
            turns += 1
        prev_dir = chosen
        nxt = (path[-1][0] + chosen[0], path[-1][1] + chosen[1])
        path.append(nxt)
        path_set.add(nxt)

    if turns < 6:
        return None

    deg = {p: sum(1 for q in _neighbors8(p) if q in path_set) for p in path}
    if sum(1 for d in deg.values() if d == 1) != 2:
        return None
    if any(d > 2 for d in deg.values()):
        return None
    return path


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


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], non_bg_count)
    target_len = random.randint(22, 28)

    path = _build_path(target_len, H, W)
    if path is None:
        return None
    total = len(path)

    num_segments = random.randint(max(4, non_bg_count), 7)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=8)
    if seg_lengths is None:
        return None

    shuffled_palette = list(palette)
    random.shuffle(shuffled_palette)
    seg_colors = list(shuffled_palette[: min(num_segments, len(shuffled_palette))])
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    expected_colors = colors if min(path[0], path[-1]) == path[0] else list(reversed(colors))
    expected_out = np.array(expected_colors, dtype=int).reshape(-1, 1)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate 8-connected input after 300 attempts")
