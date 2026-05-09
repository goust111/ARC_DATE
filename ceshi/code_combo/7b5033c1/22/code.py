# Combo variant of 7b5033c1 (index 22)
# combo_spec: D1/3+D4/1+D6/0
# Name: 八连通+渐变+原地重着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The original task traced a 4-connected snake from the top-left endpoint and exported the
#    visited colors as an N x 1 column. Here three changes must cooperate:
#    - D6/0 changes the object itself: consecutive snake cells may touch diagonally, so the path
#      must be traced in 8-neighborhood rather than 4-neighborhood.
#    - D4/1 changes the color source cue: along the intended reading direction, the snake colors
#      form an ascending 3-color cycle a,b,c,a,b,c,... . Reading from the opposite endpoint yields
#      the reverse rhythm c,b,a,c,b,a,..., so direction is now determined by color pattern rather
#      than by "top-left endpoint".
#    - D1/3 changes the output form: instead of extracting colors into a column, the solver keeps
#      the whole grid and recolors the path in place.
#    A minimal trace example is an 8-connected path whose intended order is:
#      cells:  p0  p1  p2  p3  p4  p5
#      input:  a   b   c   a   b   c   a   b   c
#      output: 2   4   6   4   6   2   6   2   4
#    Consecutive triples are recolored by a rotating canonical gradient:
#      block 0 -> [2,4,6]
#      block 1 -> [4,6,2]
#      block 2 -> [6,2,4]
#    and then the 3-block cycle repeats. This is not a global color substitution because the same
#    input color can receive different outputs in different traversal blocks.
# b) Complexity calibration:
#    The recolor rule uses only three output colors {2,4,6} and a fixed rotating triple rhythm
#    [2,4,6] -> [4,6,2] -> [6,2,4]. That keeps the rule visually learnable while still requiring ordered
#    traversal. The generator forces path lengths that contain many repeated 3-cell blocks and
#    also forces the intended start to be the non-top-left endpoint, so both the direction cue
#    and the recolor cue are active in every sample.
# c) Hidden assumptions audit against the original main():
#    - `bg = most common color` still holds and is reused.
#    - `mask = g != bg` still isolates a single object, but the old assumption "4-neighbor path"
#      no longer holds; all degree counting and traversal must use 8-neighbors.
#    - The old `start = min(endpoints)` is no longer reliable because the intended direction is
#      encoded by the ascending cycle. The solver now traces from both endpoints and keeps the
#      traversal with fewer mismatches against the sorted palette cycle.
#    - The old `return Nx1 column` is invalid because D1/3 requires a full-grid output.
#    - The old logic could read each visited cell color as the final answer. Now colors are only a
#      cue for orientation; the final colors depend on traversal index grouped into 3-cell blocks.
# d) Placement legality:
#    The generator builds one self-avoiding 8-connected chain inside a 1-cell border. A candidate
#    new cell is rejected if it would 8-touch any previous path cell other than its predecessor,
#    preventing overlaps, branches, or shortcut contacts. This guarantees exactly two endpoints and
#    a unique trace for `main()`.
# e) Visual observability:
#    A human sees one diagonal-heavy snake whose colors cycle repeatedly through three hues. In the
#    output, the same snake remains in place but its colors become repeated 3-step gradients that
#    flip direction every block. Across 2-4 examples, the constant output palette and alternating
#    block structure make the recolor rule discoverable.
# f) Visual cleanliness:
#    There is only one object, no overlaid markers, and only three output colors. The path is kept
#    away from the border and from itself, so diagonal links are readable rather than tangled.
# g) Color-change inferability:
#    The solver uses a small fixed output palette and a deterministic relation to traversal order:
#    every consecutive triple is recolored by the next rotation of [2,4,6]. Because the input
#    repeats the same 3-color cycle many times, humans get repeated evidence for the same rule
#    instead of one-off arbitrary substitutions.
# 9) New assumptions introduced by the new main() and generator guarantees:
#    - The path must be a single simple 8-connected chain with exactly two endpoints; `_build_path`
#      enforces this and `main()` checks it through 8-degree counts.
#    - Exactly one endpoint traversal should match the ascending color cycle better; the generator
#      paints the cycle from the non-top-left endpoint and rejects ties.
#    - The cycle period is visually stable and small; the generator always uses exactly 3 distinct
#      input colors and path lengths that are multiples of 3.
#    - The recolor rule assumes repeated blocks exist; generated lengths are at least 21, so both
#      block parities appear multiple times.


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
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS8)}
TURN_ORDER = [0, 1, 7, 2, 6, 3, 5, 4]
INITIAL_DIR_ORDER = [2, 3, 4, 5, 6, 7, 0, 1]
DIAG_ONLY_DIRS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]

ASC_OUT = [2, 4, 6]
OUTPUT_COLORS = set(ASC_OUT)
BG_CHOICES = [0, 1, 3, 5, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
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


def _trace_path(start: tuple[int, int], pset: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(pset) == 1:
        return [start]

    order = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start
    prev_dir: int | None = None

    for _ in range(len(pset) - 1):
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

    if len(order) != len(pset):
        raise ValueError("8-connected trace did not consume the whole path")
    return order


def _cycle_mismatch(colors: list[int], palette: list[int]) -> int:
    period = len(palette)
    return sum(int(col != palette[i % period]) for i, col in enumerate(colors))


def _choose_oriented_order(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(grid != bg)]
    if not pts:
        return []
    if len(pts) == 1:
        return [pts[0]]

    pset = set(pts)
    deg = {p: sum(1 for q in _neighbors8(p) if q in pset) for p in pts}
    ends = [p for p in pts if deg[p] == 1]
    if len(ends) != 2 or any(d > 2 for d in deg.values()):
        raise ValueError("Expected a simple 8-connected chain")

    palette = sorted({int(grid[r, c]) for r, c in pts})
    traced_orders = [_trace_path(end, pset) for end in ends]
    traced_colors = [[int(grid[r, c]) for r, c in order] for order in traced_orders]
    mismatch_scores = [_cycle_mismatch(colors, palette) for colors in traced_colors]

    if mismatch_scores[0] < mismatch_scores[1]:
        return traced_orders[0]
    if mismatch_scores[1] < mismatch_scores[0]:
        return traced_orders[1]

    return traced_orders[ends.index(min(ends))]


def _output_color(index: int) -> int:
    block = index // 3
    phase = index % 3
    return ASC_OUT[(phase + block) % 3]


def _paint_order(
    shape: tuple[int, int],
    order: list[tuple[int, int]],
    colors: list[int],
    bg: int,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for (r, c), col in zip(order, colors):
        grid[r, c] = col
    return grid


def _expected_output(
    shape: tuple[int, int],
    order: list[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for idx, (r, c) in enumerate(order):
        out[r, c] = _output_color(idx)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a single 8-connected color-cycle snake and recolor it in place.

    Input structure:
    - Background is the most common color.
    - Non-background cells form one simple 8-connected path with two endpoints.
    - From the intended endpoint, path colors follow a repeated ascending 3-color cycle.

    Rule:
    - Trace the path from the endpoint whose traversal best matches the ascending cycle.
    - Keep the full canvas.
    - Recolor the path by traversal order in rotating 3-cell gradient blocks:
      [2,4,6], then [4,6,2], then [6,2,4], then repeat.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    order = _choose_oriented_order(g, bg)
    if not order:
        return g.copy()

    out = g.copy()
    for idx, (r, c) in enumerate(order):
        out[r, c] = _output_color(idx)
    return out


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
            if (rr, cc) in (cur, nxt):
                continue
            if (rr, cc) in path_set:
                return False
    return True


def _build_path(target_len: int, height: int, width: int) -> list[tuple[int, int]] | None:
    start = (random.randint(2, height - 3), random.randint(2, width - 3))
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
            if _legal_next(path_set, path[-1], nxt, height, width):
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


def _build_input_colors(length: int, palette: list[int]) -> list[int]:
    return [palette[i % len(palette)] for i in range(length)]


def _gen_attempt() -> np.ndarray | None:
    height, width = 16, 16
    bg = random.choice(BG_CHOICES)
    available = [c for c in BG_CHOICES if c != bg]
    palette = sorted(random.sample(available, 3))
    target_len = random.choice([21, 24, 27, 30])

    path = _build_path(target_len, height, width)
    if path is None:
        return None

    endpoints = [path[0], path[-1]]
    top_left = min(endpoints)
    true_start = endpoints[0] if endpoints[1] == top_left else endpoints[1]
    oriented_path = path if path[0] == true_start else list(reversed(path))

    colors = _build_input_colors(target_len, palette)
    inp = _paint_order((height, width), oriented_path, colors, bg)
    expected = _expected_output((height, width), oriented_path, bg)

    pset = set(path)
    top_left_order = _trace_path(top_left, pset)
    top_left_colors = [int(inp[r, c]) for r, c in top_left_order]
    true_colors = [int(inp[r, c]) for r, c in oriented_path]
    if _cycle_mismatch(true_colors, palette) != 0:
        return None
    if _cycle_mismatch(top_left_colors, palette) <= 0:
        return None

    mapping: dict[int, set[int]] = {}
    for idx, (r, c) in enumerate(oriented_path):
        mapping.setdefault(int(inp[r, c]), set()).add(_output_color(idx))
    if sum(1 for outs in mapping.values() if len(outs) >= 2) < 2:
        return None

    if np.any((inp != bg) & (inp == expected)):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate combo input after 300 attempts")
