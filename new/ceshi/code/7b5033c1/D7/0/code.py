# Variant of 7b5033c1
# Dimension: D7 — 背景与干扰 (src)
# Change: 纯净背景，无干扰元素，所有非背景像素均属于蛇形路径... → 添加孤立干扰像素
#
# Design analysis
# a) Concrete behavior:
#    The true object is still one multicolor 4-connected snake. The new inputs also
#    contain several isolated non-background singletons scattered in the grid. Those
#    distractors may reuse the snake's colors and can appear above / left of the
#    snake, so "collect every non-bg cell" is no longer the right mental model.
#    Example: if the snake is cells [(2,4),(3,4),(3,5)] and we add noise at (0,1)
#    and (5,0), the original task would treat all 5 non-bg cells as belonging to
#    one object; the variant must first keep only the snake's 4-connected component
#    and then trace [(2,4),(3,4),(3,5)] into a 3x1 output strip.
# b) Complexity calibration:
#    The distractors are not color-coded "easy rejects"; they can share colors with
#    the snake and at least one is usually lexicographically earlier than the snake
#    start, making the D7 change visible in the raw grid.
# c) Hidden assumptions in the original main():
#    - `pts = all non-bg cells` assumed every non-bg cell belongs to the snake.
#    - degree computation on `pts` assumed there is a single relevant component.
#    - traversal length used `len(pts)`, which becomes semantically wrong once noise
#      exists, even though isolated degree-0 cells happened to be harmless.
#    The variant fixes this by explicitly extracting the unique largest 4-connected
#    chain component before any endpoint logic.
# d) Placement legality:
#    Generator keeps the snake self-avoiding and 4-connected as before. Noise cells
#    are placed on background cells only, never 4-neighboring the snake or each
#    other, so they remain isolated singleton components.
# e) Visual observability:
#    Every generated sample contains multiple isolated non-bg pixels; at least one
#    distractor is biased to appear earlier than the snake start when possible, so a
#    human sees that some non-bg cells must be ignored before extracting the column.

from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(points: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(points)
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        remaining.remove(start)
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for q in _neighbors4(r, c):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    comp.append(q)
        comps.append(comp)
    return comps


def _component_score(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    pset = set(comp)
    degs = [
        sum((q in pset) for q in _neighbors4(r, c))
        for r, c in comp
    ]
    end_count = sum(d == 1 for d in degs)
    chain_like = int(end_count == 2 and all(d in (1, 2) for d in degs))
    r0, c0 = min(comp)
    return chain_like, len(comp), -r0, -c0


def _extract_path_cells(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pts = {(int(r), int(c)) for r, c in np.argwhere(grid != bg)}
    if not pts:
        return []
    comps = _components4(pts)
    return max(comps, key=_component_score)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Non-bg pixels contain one 4-connected multicolor snake plus isolated distractors.
    - Ignore the distractors by selecting the snake's main 4-connected component.
    - Trace the snake from the top-left-most endpoint and emit its colors as an Nx1 column.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)

    path_pts = _extract_path_cells(g, bg)
    if not path_pts:
        return np.zeros((0, 1), dtype=int)

    pset = set(path_pts)
    deg = {
        (r, c): sum((q in pset) for q in _neighbors4(r, c))
        for r, c in path_pts
    }
    ends = [p for p in path_pts if deg[p] == 1]

    if len(path_pts) == 1:
        start = path_pts[0]
    else:
        assert len(ends) == 2
        start = min(ends)

    order: list[tuple[int, int]] = []
    prev = None
    cur = start
    for _ in range(len(path_pts)):
        order.append(cur)
        nbrs = [q for q in _neighbors4(*cur) if q in pset and q != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    cols = [int(g[r, c]) for r, c in order]
    return np.array(cols, dtype=int).reshape((-1, 1))


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= h - 1 or nc < 1 or nc >= w - 1:
            break
        if (nr, nc) in path_set:
            break
        pred = (r + dr * (i - 1), c + dc * (i - 1))
        bad = False
        for q in _neighbors4(nr, nc):
            if q != pred and q in path_set:
                bad = True
                break
        if bad:
            break
        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


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


def _path_endpoints(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pset = set(path)
    return [
        p
        for p in path
        if sum((q in pset) for q in _neighbors4(*p)) == 1
    ]


def _noise_candidates(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in blocked:
                continue
            if any((q in blocked) for q in _neighbors4(r, c)):
                continue
            out.append(cell)
    return out


def _place_noise(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    bg: int,
    palette: list[int],
) -> int | None:
    h, w = map(int, grid.shape)
    endpoints = _path_endpoints(path)
    start = min(endpoints) if endpoints else min(path)

    noise_set: set[tuple[int, int]] = set()
    blocked = set(path)
    candidates = _noise_candidates(h, w, blocked)
    if len(candidates) < 3:
        return None

    early_candidates = [p for p in candidates if p < start]
    count = random.randint(3, 6)
    if len(candidates) < count:
        return None

    chosen: list[tuple[int, int]] = []
    if early_candidates:
        first = random.choice(early_candidates)
        chosen.append(first)
        noise_set.add(first)

    while len(chosen) < count:
        blocked_now = blocked | noise_set
        pool = _noise_candidates(h, w, blocked_now)
        remaining = count - len(chosen)
        if len(pool) < remaining:
            return None
        pick = random.choice(pool)
        chosen.append(pick)
        noise_set.add(pick)

    for cell in chosen:
        grid[cell] = random.choice(palette)
    return len(chosen)


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    bg = random.randint(0, 9)
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], non_bg_count)

    target_len = random.randint(20, 30)
    start_r = random.randint(1, 4)
    start_c = random.randint(1, w - 2)

    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

    while len(path) < target_len:
        run_len = min(random.randint(1, 5), target_len - len(path))
        if run_len <= 0:
            break

        last = path[-1]
        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, run_len, h, w)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, h, w)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, h, w)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, h, w)
                if segment:
                    h_dir = -h_dir
            h_dir = -h_dir

        if not segment:
            break

        path.extend(segment)
        turns += 1
        is_vertical = not is_vertical

    total = len(path)
    if total < 20 or total > 30:
        return None
    if turns < 7 or turns > 11:
        return None

    num_segments = random.randint(max(3, non_bg_count), 6)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=9)
    if seg_lengths is None:
        return None

    shuffled_palette = list(palette)
    random.shuffle(shuffled_palette)
    seg_colors = list(shuffled_palette[:num_segments])
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

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    noise_count = _place_noise(grid, path, bg, palette)
    if noise_count is None or noise_count < 3:
        return None

    expected_out = np.array(colors, dtype=int).reshape(-1, 1)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 300 attempts")
