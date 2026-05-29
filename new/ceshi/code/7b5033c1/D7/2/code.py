from __future__ import annotations

# Variant of 7b5033c1
# Dimension: D7 — 背景与干扰 (src)
# Change: 纯净背景，无干扰元素，所有非背景像素均属于蛇形路径... → 多条路径选择最长
#
# Design analysis for this variant:
# 1. Concrete behavior:
#    The input no longer contains one snake. It contains 2-3 disjoint 4-connected
#    colored snakes on the same background. Only the longest snake is relevant;
#    the output is the same single-column color strip as before, but traced from
#    the longest snake only, starting at that snake's top-left endpoint.
#    Small example:
#    - Old task: one 22-cell snake -> collect all non-bg cells, trace from the
#      global top-left endpoint, output 22 colors.
#    - New task: a 7-cell snake sits in the upper-left, a 21-cell snake sits
#      lower/right. Old code would still collect all non-bg cells together,
#      choose the 7-cell snake's endpoint because it is globally top-left, and
#      stop after tracing only that short component. The variant must instead
#      split the non-bg cells into connected components, compare lengths [7, 21],
#      choose 21, and only then trace/order colors.
# 2. Complexity calibration:
#    The distractors are not isolated single pixels; they are genuine snakes with
#    multiple turns and multi-color segments. This keeps the visual change obvious
#    and prevents the task from collapsing into "ignore tiny junk pixels".
# 3. Hidden assumption audit of the original solver:
#    - `pts = argwhere(mask)` assumed all non-bg cells belonged to one component.
#    - `deg` was computed over the union of all non-bg cells; harmless before,
#      insufficient now because endpoints from multiple snakes are mixed.
#    - `start = min(ends)` implicitly meant "top-left endpoint of the only snake";
#      now it would incorrectly pick the top-left distractor.
#    - The tracing loop walked at most one simple chain and then stopped; with
#      multiple components it would never reach the longest snake unless that
#      snake also happened to contain the globally top-left endpoint.
#    The fix is to identify components first, choose the longest component, and
#    apply the original endpoint/traversal logic only inside that component.
# 4. Placement legality:
#    The generator keeps every snake inside the interior of the grid, forbids
#    overlap, and reserves an 8-neighbor moat around completed snakes so that
#    components remain visually separate. Each generated component is a simple
#    chain with no branches, so the solver's local degree logic remains valid.
# 5. Visual observability:
#    The generator intentionally places one shorter distractor in the upper-left
#    region, while the longest snake is generated elsewhere. Across examples this
#    makes it clear that the rule is not "take the top-left snake" but "find the
#    longest snake, then linearize it".

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/arc2_solver_select")
    from arc_utils import get_bg


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    pts = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    components: list[list[Coord]] = []
    while pts:
        start = pts.pop()
        stack = [start]
        comp = [start]
        while stack:
            cur = stack.pop()
            for nb in _neighbors4(*cur):
                if nb in pts:
                    pts.remove(nb)
                    stack.append(nb)
                    comp.append(nb)
        components.append(comp)
    return components


def _trace_component(component: list[Coord]) -> list[Coord]:
    pset = set(component)
    deg: dict[Coord, int] = {}
    for r, c in component:
        deg[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    ends = [p for p in component if deg[p] <= 1]
    start = min(ends) if ends else min(component)

    order: list[Coord] = []
    visited: set[Coord] = set()
    prev: Coord | None = None
    cur = start
    while cur not in visited:
        order.append(cur)
        visited.add(cur)
        nbrs = [q for q in _neighbors4(*cur) if q in pset and q != prev and q not in visited]
        if not nbrs:
            break
        prev, cur = cur, min(nbrs)

    # Fallback for malformed inputs: append any remainder deterministically.
    if len(order) != len(component):
        order.extend(sorted(pset - visited))
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - bg = most common color.
    - Non-bg pixels form 2-3 separate snake paths.
    - Select the longest 4-connected component.
    - Trace that component from its top-left endpoint and output its colors as an
      (N x 1) column.

    New solver assumptions:
    - Each component is still a simple path (or a trivial singleton fallback).
    - The intended target is the unique longest component.
    The generator enforces both conditions.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    mask = g != bg
    if not np.any(mask):
        return np.zeros((0, 1), dtype=int)

    components = _find_components(mask)
    best = max(components, key=lambda comp: (len(comp), -min(comp)[0], -min(comp)[1]))
    order = _trace_component(best)
    cols = [int(g[r, c]) for r, c in order]
    return np.array(cols, dtype=int).reshape((-1, 1))


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


def _buffer8(cells: list[Coord], H: int, W: int) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out.add((nr, nc))
    return out


def _try_extend(
    path_set: set[Coord],
    blocked: set[Coord],
    last: Coord,
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[Coord]:
    segment: list[Coord] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in blocked or (nr, nc) in path_set:
            break
        pred = (r + dr * (i - 1), c + dc * (i - 1))
        bad = False
        for nb in _neighbors4(nr, nc):
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break
        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _grow_snake(
    H: int,
    W: int,
    target_len: int,
    blocked: set[Coord],
    min_turns: int,
    preferred_cells: list[Coord] | None = None,
) -> list[Coord] | None:
    interior = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    for _ in range(250):
        free = [p for p in interior if p not in blocked]
        if preferred_cells is not None:
            preferred_free = [p for p in preferred_cells if p not in blocked]
            start_pool = preferred_free if preferred_free else free
        else:
            start_pool = free
        if not start_pool:
            return None

        start = random.choice(start_pool)
        path = [start]
        path_set = {start}
        turns = 0
        is_vertical = random.choice([True, False])
        h_dir = random.choice([-1, 1])

        for _step in range(target_len * 4):
            if len(path) >= target_len:
                break

            run_len = min(random.randint(1, 5), target_len - len(path))
            last = path[-1]
            segment: list[Coord] = []

            if is_vertical:
                dirs = [1, -1]
                random.shuffle(dirs)
                for v_dir in dirs:
                    segment = _try_extend(path_set, blocked, last, v_dir, 0, run_len, H, W)
                    if segment:
                        break
            else:
                used_dir: int | None = None
                for cand in (h_dir, -h_dir):
                    segment = _try_extend(path_set, blocked, last, 0, cand, run_len, H, W)
                    if segment:
                        used_dir = cand
                        break
                h_dir = -used_dir if used_dir is not None else -h_dir

            if not segment:
                break

            path.extend(segment)
            turns += 1
            is_vertical = not is_vertical

        if len(path) == target_len and turns >= min_turns:
            return path
    return None


def _make_colors(total: int, bg: int) -> list[int] | None:
    usable = [c for c in range(10) if c != bg]
    min_seg_len = 2 if total >= 10 else 1
    min_segments = 3 if total >= 10 else 2
    max_segments = min(6, total)
    if min_segments > max_segments:
        return None

    for _ in range(60):
        num_segments = random.randint(min_segments, max_segments)
        palette_size = random.randint(2, min(5, num_segments, len(usable)))
        palette = random.sample(usable, palette_size)
        seg_lengths = _random_partition(total, num_segments, min_seg_len, max(6, total))
        if seg_lengths is None:
            continue

        seg_colors = list(palette)
        random.shuffle(seg_colors)
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
        if len(colors) == total:
            return colors
    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    num_paths = random.choice([2, 3])

    longest_len = random.randint(19, 24)
    short1_len = random.randint(6, 10)
    lengths = [short1_len, longest_len]
    if num_paths == 3:
        short2_upper = min(11, longest_len - 7)
        short2_len = random.randint(5, max(5, short2_upper))
        lengths.append(short2_len)

    upper_left_cells = [(r, c) for r in range(1, 6) for c in range(1, 6)]
    away_from_corner = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if r >= 4 or c >= 4]
    lower_right_cells = [(r, c) for r in range(6, H - 1) for c in range(6, W - 1)]

    blocked: set[Coord] = set()

    short1 = _grow_snake(H, W, lengths[0], blocked, min_turns=2, preferred_cells=upper_left_cells)
    if short1 is None:
        return None
    blocked |= _buffer8(short1, H, W)

    longest = _grow_snake(H, W, lengths[1], blocked, min_turns=6, preferred_cells=away_from_corner)
    if longest is None:
        return None
    blocked |= _buffer8(longest, H, W)

    paths = [short1, longest]
    if num_paths == 3:
        short2 = _grow_snake(H, W, lengths[2], blocked, min_turns=2, preferred_cells=lower_right_cells)
        if short2 is None:
            return None
        paths.append(short2)

    # Make the D7 change visually obvious: a shorter component must exist above/left of the target.
    if min(min(path) for path in paths) == min(longest):
        return None

    colors_per_path: list[list[int]] = []
    for path in paths:
        colors = _make_colors(len(path), bg)
        if colors is None:
            return None
        colors_per_path.append(colors)

    grid = np.full((H, W), bg, dtype=int)
    for path, colors in zip(paths, colors_per_path):
        for (r, c), col in zip(path, colors):
            grid[r, c] = col

    coord_to_color = {coord: col for coord, col in zip(longest, colors_per_path[1])}
    expected_order = _trace_component(longest)
    expected_out = np.array([coord_to_color[p] for p in expected_order], dtype=int).reshape((-1, 1))

    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    # Extra guard against degeneration: keep a clear gap to the second-longest path.
    comp_lengths = sorted((len(path) for path in paths), reverse=True)
    if len(comp_lengths) < 2 or comp_lengths[0] - comp_lengths[1] < 6:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")


if __name__ == "__main__":
    out_dir = Path("/home/lixinyu/snap/new/ceshi/code/7b5033c1/D7/2")
    out_dir.mkdir(parents=True, exist_ok=True)
