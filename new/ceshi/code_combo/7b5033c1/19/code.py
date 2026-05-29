# Combo variant of 7b5033c1 (index 19)
# combo_spec: D5/0+D7/0
# Name: 原尺寸输出+噪声移除
#
# Design analysis
# a) Concrete behavior:
#    Original 7b5033c1 treats every non-background pixel as part of one snake,
#    traverses it from the top-left endpoint, and outputs an Nx1 color strip.
#    This combo changes both the goal and the visible clutter:
#    - D7 adds isolated non-background singleton noise pixels.
#    - The combo-level D5 reinterpretation keeps the output at the original size
#      and leaves the snake in place instead of straightening it.
#    Small example:
#      input: bg=0, snake at [(1,1)=2,(2,1)=3,(2,2)=4], noise at [(0,4)=2,(4,0)=4]
#      original main(): collects all 5 non-bg cells and tries to form one path,
#      which is semantically wrong once isolated singletons exist.
#      combo main(): first keeps only the unique chain-like 4-connected component
#      [(1,1),(2,1),(2,2)], then returns a 5x5 grid where those 3 cells keep
#      their colors and both singleton noise cells become 0.
#    The key difference is that "extract path order" becomes "identify the real
#    object, preserve it, erase unrelated foreground."
#
# b) Complexity calibration:
#    The noise is not color-coded; it reuses snake colors and is scattered in
#    multiple regions, often including at least one lexicographically earlier cell
#    than the snake start. That makes the D7 change visible. At the same time,
#    noise is restricted to isolated singletons and kept to 3-6 cells so the grid
#    stays readable and the intended object remains unambiguous.
#
# c) Hidden assumptions broken from the original main():
#    - `pts = all non-bg cells` no longer means "the snake cells".
#    - Degree counts on all non-bg cells no longer identify valid endpoints.
#    - Traversing for `len(pts)` steps is no longer meaningful.
#    - Returning an Nx1 strip no longer matches the combo output spec.
#    The revised solver therefore:
#    1. finds 4-connected components among non-bg cells,
#    2. selects the best chain-like component as the real snake,
#    3. emits a same-shape grid containing only that component.
#
# d) Placement legality:
#    Generator builds one self-avoiding 4-connected snake inside a 16x16 board.
#    Noise is added only on background cells that are 4-neighbor-disjoint from
#    both the snake and previously placed noise, so every distractor remains an
#    isolated singleton and nothing overlaps or leaves the grid.
#
# e) Visual observability:
#    Across examples, the output looks like the input with only the detached dots
#    removed; the long bent multicolor snake remains untouched. Because the snake
#    keeps its shape and colors, humans can directly compare input/output and infer
#    that connectivity, not color, decides what survives.
#
# f) Visual cleanliness:
#    Noise never touches the snake and never clusters with other noise. This avoids
#    ambiguous blobs and keeps one dominant object plus a few clearly detached dots.
#
# g) Color inference:
#    There is no new color mapping. Surviving snake pixels preserve their original
#    colors, and removed pixels become the existing background color. Since noise
#    shares the snake palette, the rule is inferable from geometry alone.
#
# h) Quantitative anti-degeneration target:
#    Valid samples must contain at least 3 isolated noise cells, so input and
#    output differ on at least 3 positions. This ensures the combo does not reduce
#    back to the clean original task.
#
# i) New solver assumptions and generator guarantees:
#    - Solver assumes one dominant chain-like component exists among non-bg cells.
#      Generator guarantees exactly one long path and only singleton distractors.
#    - Solver assumes background is the modal color. Generator uses a mostly empty
#      board, so the background remains the most frequent value.
#    - Solver assumes surviving object should be copied in place. Generator's
#      expected output is built exactly that way.

from __future__ import annotations

import random
from collections import Counter

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
    degs = [sum((q in pset) for q in _neighbors4(r, c)) for r, c in comp]
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
    Combo rule:
    - Input contains one multicolor 4-connected snake plus isolated singleton noise.
    - Output keeps the original grid size.
    - The snake stays in place with its original colors.
    - All isolated distractor pixels are removed to background.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    path_pts = _extract_path_cells(g, bg)
    for r, c in path_pts:
        out[r, c] = g[r, c]
    return out


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
        if any((q != pred and q in path_set) for q in _neighbors4(nr, nc)):
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
    return [p for p in path if sum((q in pset) for q in _neighbors4(*p)) == 1]


def _noise_candidates(h: int, w: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
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
    palette: list[int],
) -> int | None:
    h, w = map(int, grid.shape)
    endpoints = _path_endpoints(path)
    start = min(endpoints) if endpoints else min(path)

    noise_set: set[tuple[int, int]] = set()
    blocked = set(path)
    initial_pool = _noise_candidates(h, w, blocked)
    count = random.randint(3, 6)
    if len(initial_pool) < count:
        return None

    early_pool = [p for p in initial_pool if p < start]
    chosen: list[tuple[int, int]] = []
    if early_pool:
        first = random.choice(early_pool)
        chosen.append(first)
        noise_set.add(first)

    while len(chosen) < count:
        blocked_now = blocked | noise_set
        pool = _noise_candidates(h, w, blocked_now)
        remaining = count - len(chosen)
        if len(pool) < remaining:
            return None
        cell = random.choice(pool)
        chosen.append(cell)
        noise_set.add(cell)

    for r, c in chosen:
        grid[r, c] = random.choice(palette)
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

    noise_count = _place_noise(grid, path, palette)
    if noise_count is None or noise_count < 3:
        return None

    expected_out = np.full_like(grid, bg)
    for (r, c), col in zip(path, colors):
        expected_out[r, c] = col

    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    changed_cells = int(np.sum(grid != actual_out))
    if changed_cells != noise_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 300 attempts")
