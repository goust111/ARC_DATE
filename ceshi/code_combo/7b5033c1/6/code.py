# Combo variant of 7b5033c1 (index 6)
# combo_spec: D1/3+D7/0
# Name: 原地重着色+去噪
#
# Design analysis
# a) Concrete behavior and trace:
#    The original task treats every non-background cell as part of one snake, traces it from the
#    top-left endpoint, and outputs the encountered colors as an N x 1 column. This combo changes
#    two stages of that pipeline at once.
#    Example trace:
#    - input path traversal colors: [8, 8, 1, 1, 1, 3, 3, 3, 3, 3, 3]
#    - isolated noise pixels: one 8 at (0, 0), one 1 at (15, 15)
#    - original logic would either be confused by "all non-bg cells" or would still aim for a
#      column output. The combo logic first removes the two singleton components, keeps only the
#      unique long 4-connected chain, then splits the traced colors into runs [8x2, 1x3, 3x6].
#      Those runs are recolored in place by length bucket: 2 -> 2, 3 -> 2, 6 -> 6. The output is
#      the full 16x16 grid with the snake still in the same cells, noise turned back into bg.
#    The key difference is that traversal still matters, but it now drives structural recoloring
#    on the original canvas after a denoising step.
# b) Complexity calibration:
#    The combo must not collapse into either "ignore a few dots" or "global color remap". The
#    generator therefore enforces three run-length buckets (short / medium / long) and multiple
#    isolated distractors. Input segment colors are reused across different buckets, and each
#    bucket can be realized by multiple input colors, so the answer cannot be a simple per-color
#    lookup. At the same time the output palette is only {2,4,6}, which is small enough for a
#    human to infer from 2-4 examples.
# c) Hidden-assumption audit of the original main():
#    - `mask = g != bg` no longer isolates exactly the snake because D7 adds non-path pixels.
#    - `pts = all non-bg cells` is invalid after noise injection; we must extract components first.
#    - endpoint detection still works, but only on the chosen snake component.
#    - `return Nx1 strip` is invalid after D1/3; output must preserve the whole canvas size.
#    - reading raw colors is no longer the terminal action; colors are now only evidence for
#      maximal same-color runs along traversal order.
#    The fused solver repairs these assumptions by: component extraction -> snake traversal ->
#    run bucketing -> repaint snake on a clean background canvas.
# d) Placement legality:
#    The generator builds a single self-avoiding 4-connected snake inside the inner 14x14 area,
#    paints it by contiguous runs, and places noise only on background cells that are not
#    4-neighbors of the snake or of previously placed noise. This guarantees no overlap, no
#    accidental merging, and no ambiguity about which component is the snake.
# e) Visual observability:
#    Across samples, the same output colors always represent short / medium / long runs, while
#    the input snake colors vary independently. Noise often appears above or left of the true
#    start, so a human can see that some non-bg pixels disappear entirely. The path geometry stays
#    fixed between input and output, which makes the recoloring rule visually legible.
# f) Visual cleanliness:
#    Only two visual element types exist on the input canvas: one snake and a handful of isolated
#    singletons. There are no overlapping markings, frames, or secondary objects. Noise is sparse
#    and separated by at least one background cell, so it reads as clutter to be removed rather
#    than a second structured object.
# g) Color-change inferability:
#    The output palette is deliberately restricted to three colors tied to three length buckets:
#    2-3 cells -> 2, 4-5 cells -> 4, 6+ cells -> 6. The generator guarantees all three buckets
#    appear, with repeated support, and avoids using {2,4,6} in the input snake or noise. This
#    makes every output color change attributable to run length rather than to literal input color.
# 9) New assumptions introduced by the fused main():
#    - The relevant object is the unique largest chain-like 4-connected component.
#    - Adjacent traversal runs must differ in input color so run boundaries are recoverable.
#    - Noise components must stay isolated singletons and must not touch the snake by 4-neighbors.
#    - The generator enforces all of the above and validates every sample with `main(input)`.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


SHORT_COLOR = 2
MEDIUM_COLOR = 4
LONG_COLOR = 6
OUTPUT_COLORS = {SHORT_COLOR, MEDIUM_COLOR, LONG_COLOR}
RUN_LENGTH_BUCKETS = {
    "short": [2, 3],
    "medium": [4, 5],
    "long": [6, 7, 8],
}
BG_CHOICES = [0, 1, 3, 5, 7, 8, 9]


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


def _trace_path(path_pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not path_pts:
        return []
    if len(path_pts) == 1:
        return [path_pts[0]]

    pset = set(path_pts)
    deg = {
        p: sum((q in pset) for q in _neighbors4(*p))
        for p in path_pts
    }
    assert all(d <= 2 for d in deg.values())
    ends = [p for p in path_pts if deg[p] == 1]
    assert len(ends) == 2
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(len(path_pts)):
        order.append(cur)
        nbrs = [q for q in _neighbors4(*cur) if q in pset and q != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(path_pts)
    return order


def _bucket_color(run_len: int) -> int:
    if run_len <= 3:
        return SHORT_COLOR
    if run_len <= 5:
        return MEDIUM_COLOR
    return LONG_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Non-background pixels contain one multicolor snake plus isolated distractors.
    - Ignore all distractors by keeping only the main 4-connected chain component.
    - Trace the snake from the top-left endpoint.
    - Split the traversal into maximal same-color runs.
    - Recolor each run in place by run length bucket:
      * 2-3 cells -> 2
      * 4-5 cells -> 4
      * 6+ cells  -> 6
    - Remove all distractors by painting everything outside the snake back to background.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)

    path_pts = _extract_path_cells(g, bg)
    if not path_pts:
        return g.copy()

    order = _trace_path(path_pts)
    out = np.full(g.shape, bg, dtype=int)
    path_colors = [int(g[r, c]) for r, c in order]

    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and path_colors[j] == path_colors[i]:
            j += 1
        new_color = _bucket_color(j - i)
        for r, c in order[i:j]:
            out[r, c] = new_color
        i = j

    return out


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= height - 1 or nc < 1 or nc >= width - 1:
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


def _sample_run_lengths() -> list[int] | None:
    labels = list(RUN_LENGTH_BUCKETS)
    for _ in range(200):
        num_segments = random.randint(5, 6)
        categories = labels + [random.choice(labels) for _ in range(num_segments - len(labels))]
        random.shuffle(categories)
        lengths = [random.choice(RUN_LENGTH_BUCKETS[label]) for label in categories]
        counts = Counter(categories)
        if 20 <= sum(lengths) <= 30 and max(counts.values()) >= 2:
            return lengths
    return None


def _choose_input_colors(lengths: list[int], bg: int) -> list[int] | None:
    buckets = [_bucket_color(length) for length in lengths]
    available = [c for c in range(10) if c != bg and c not in OUTPUT_COLORS]
    for _ in range(200):
        palette = random.sample(available, 3)
        colors: list[int] = []
        prev = None
        for _ in lengths:
            choices = [c for c in palette if c != prev]
            col = random.choice(choices)
            colors.append(col)
            prev = col

        color_to_buckets: dict[int, set[int]] = defaultdict(set)
        bucket_to_colors: dict[int, set[int]] = defaultdict(set)
        for col, bucket in zip(colors, buckets):
            color_to_buckets[col].add(bucket)
            bucket_to_colors[bucket].add(col)

        reused_across_buckets = any(len(v) >= 2 for v in color_to_buckets.values())
        mixed_bucket = any(len(v) >= 2 for v in bucket_to_colors.values())
        if reused_across_buckets and mixed_bucket:
            return colors
    return None


def _build_snake(target_len: int, height: int, width: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        start_r = random.randint(1, 4)
        start_c = random.randint(1, width - 2)
        path = [(start_r, start_c)]
        path_set = {(start_r, start_c)}
        turns = 0
        is_vertical = True
        h_dir = random.choice([-1, 1])

        while len(path) < target_len:
            run_len = min(random.randint(1, 5), target_len - len(path))
            last = path[-1]

            if is_vertical:
                v_dir = 1 if random.random() < 0.85 else -1
                segment = _try_extend(path_set, last, v_dir, 0, run_len, height, width)
                if not segment:
                    segment = _try_extend(path_set, last, -v_dir, 0, run_len, height, width)
            else:
                segment = _try_extend(path_set, last, 0, h_dir, run_len, height, width)
                if not segment:
                    segment = _try_extend(path_set, last, 0, -h_dir, run_len, height, width)
                    if segment:
                        h_dir = -h_dir
                h_dir = -h_dir

            if not segment:
                break

            path.extend(segment)
            turns += 1
            is_vertical = not is_vertical

        if len(path) != target_len or not (7 <= turns <= 11):
            continue

        order = _trace_path(path)
        if len(order) == target_len:
            return order

    return None


def _paint_runs(
    shape: tuple[int, int],
    order: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
    bg: int,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    idx = 0
    for seg_len, col in zip(lengths, colors):
        for r, c in order[idx : idx + seg_len]:
            grid[r, c] = col
        idx += seg_len
    assert idx == len(order)
    return grid


def _path_endpoints(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pset = set(path)
    return [p for p in path if sum((q in pset) for q in _neighbors4(*p)) == 1]


def _noise_candidates(
    height: int,
    width: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
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
    height, width = map(int, grid.shape)
    endpoints = _path_endpoints(path)
    start = min(endpoints) if endpoints else min(path)

    path_set = set(path)
    candidates = _noise_candidates(height, width, path_set)
    count = random.randint(3, 6)
    if len(candidates) < count:
        return None

    chosen: list[tuple[int, int]] = []
    noise_set: set[tuple[int, int]] = set()

    early_candidates = [p for p in candidates if p < start]
    if early_candidates:
        first = random.choice(early_candidates)
        chosen.append(first)
        noise_set.add(first)

    while len(chosen) < count:
        blocked_now = path_set | noise_set
        pool = _noise_candidates(height, width, blocked_now)
        remaining = count - len(chosen)
        if len(pool) < remaining:
            return None
        pick = random.choice(pool)
        chosen.append(pick)
        noise_set.add(pick)

    for cell in chosen:
        grid[cell] = random.choice(palette)
    return len(chosen)


def _expected_output(
    shape: tuple[int, int],
    order: list[tuple[int, int]],
    lengths: list[int],
    bg: int,
) -> np.ndarray:
    return _paint_runs(
        shape,
        order,
        lengths,
        [_bucket_color(length) for length in lengths],
        bg,
    )


def _gen_attempt() -> np.ndarray | None:
    height, width = 16, 16
    bg = random.choice(BG_CHOICES)

    lengths = _sample_run_lengths()
    if lengths is None:
        return None

    order = _build_snake(sum(lengths), height, width)
    if order is None:
        return None

    segment_colors = _choose_input_colors(lengths, bg)
    if segment_colors is None:
        return None

    palette = sorted(set(segment_colors))
    inp = _paint_runs((height, width), order, lengths, segment_colors, bg)
    noise_count = _place_noise(inp, order, palette)
    if noise_count is None or noise_count < 3:
        return None

    expected = _expected_output((height, width), order, lengths, bg)

    # Non-degeneracy checks:
    # - path cells must actually recolor;
    # - output must remove at least the placed noise cells;
    # - the input must contain >1 connected component.
    path_mask = expected != bg
    if np.any(inp[path_mask] == expected[path_mask]):
        return None
    non_bg_pts = {(int(r), int(c)) for r, c in np.argwhere(inp != bg)}
    if len(_components4(non_bg_pts)) <= 1:
        return None
    if np.count_nonzero(inp != bg) - np.count_nonzero(expected != bg) < noise_count:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input after 300 attempts")
