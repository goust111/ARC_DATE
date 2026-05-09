# Combo variant of 7b5033c1 (index 9)
# combo_spec: D2/2+D7/0
# Name: 标记起点+噪声干扰
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Base task: one multicolor 4-connected snake is the whole foreground. The solver
#      can take all non-bg pixels, find the two degree-1 endpoints, start from the
#      lexicographically smaller endpoint, and list colors along the chain.
#    - Fused variant: the input now contains three foreground roles:
#        1) one large snake component,
#        2) one isolated marker pixel in a special color that never appears on the snake,
#        3) several isolated noise pixels that may reuse snake colors.
#    - Small trace example:
#      snake endpoints = (1, 5) and (5, 3), default start = min(...) = (1, 5).
#      marker color = red, marker cell = (4, 4), which is closer to (5, 3).
#      noise cells = {(0, 0), (2, 0), (6, 6)} in ordinary snake colors.
#      Original logic would break at two places:
#        * "all non-bg is snake" is false because the noise and marker are detached;
#        * "start = min(endpoints)" is false because the marker points to the other end.
#      Correct fused logic is:
#        * keep only the unique large chain-like component as the snake,
#        * ignore singleton noise components,
#        * locate the special-color marker outside the snake,
#        * choose the endpoint nearest to that marker,
#        * trace the snake from that endpoint and emit its colors as an Nx1 column.
# b) Complexity calibration:
#    - The marker uses one fixed special color, so it is visually distinguishable from
#      the noise. This keeps the symbol-reading step learnable from few examples.
#    - The noise reuses ordinary snake colors, so filtering cannot be done by color
#      alone; the solver must use connectivity. This makes D7 materially active.
#    - The generator sometimes marks the non-default endpoint, so D2 is not a cosmetic
#      addition. Across samples, the output can stay in the original direction or flip.
# c) Hidden-assumption audit of the original main():
#    - `mask = g != bg`: originally meant "foreground == snake". Broken now because
#      foreground also contains marker and noise.
#    - `pts = argwhere(mask)`: originally safe for degree counting. Broken now because
#      singleton noise contributes degree 0 and the marker is not path geometry.
#    - `ends = [p for p in pts if deg[p] == 1]`: originally produced exactly two
#      endpoints. Broken now because detached cells create extra components.
#    - `start = min(ends)`: broken because the marker, not top-left priority, decides
#      direction.
#    - traversal loop length `len(pts)`: broken because not every foreground cell
#      belongs to the snake.
#    The fused solver therefore extracts the main snake component first, then reads
#    the special marker, then traces only the snake cells.
# d) Placement legality:
#    - The snake remains self-avoiding and 4-connected.
#    - The marker is placed on a background cell, in-bounds, not on the snake, and not
#      4-neighbor to the snake; it must also be strictly closer to the chosen endpoint
#      than to the other endpoint under the solver's distance rule.
#    - Noise cells are placed on background cells and are never 4-neighbor to the
#      snake, the marker, or each other, so every distractor stays a singleton.
# e) Visual observability:
#    - A human sees one obvious long snake, one lone special-color symbol near an end,
#      and several extra isolated pixels elsewhere. Across 2-4 examples, it is visible
#      that isolated ordinary-colored pixels do not affect the output, while the end
#      near the special marker determines which snake color appears first.
# f) Visual cleanliness:
#    - The marker is kept near an endpoint so its role is spatially legible.
#    - Noise is limited to a few isolated singleton pixels and kept off the snake and
#      off the marker's 4-neighborhood, so the scene remains readable rather than busy.
# g) Color inferability:
#    - There is no arbitrary color remapping. The only color convention is structural:
#      one reserved marker color means "start here", while all other non-bg colors are
#      ordinary snake/noise colors. This is easy to infer because the reserved color
#      never appears on the snake strip output.
# 9) New solver assumptions and generator guarantees:
#    - There is exactly one large 4-connected chain component; that is the snake.
#    - There is exactly one isolated marker cell in MARK color outside the snake.
#    - Noise never uses MARK color and is always split into singleton components.
#    - The marker is uniquely closer to the intended start endpoint than to the other.
#    - `generate_input()` verifies every accepted grid by running `main()` on it.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import get_bg


MARK = 2
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARK_OFFSETS = [
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-2, -1),
    (-2, 1),
    (2, -1),
    (2, 1),
    (-1, -2),
    (1, -2),
    (-1, 2),
    (1, 2),
    (-2, -2),
    (-2, 2),
    (2, -2),
    (2, 2),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
]


def _dist_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    return max(dr, dc), dr + dc


def _components4_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((int(rr), int(cc)))
            for dr, dc in DIR4:
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(comp)
    return comps


def _component_score(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    pset = set(comp)
    degs = [sum(((r + dr, c + dc) in pset) for dr, dc in DIR4) for r, c in comp]
    end_count = sum(d == 1 for d in degs)
    chain_like = int(end_count == 2 and all(d in (1, 2) for d in degs))
    r0, c0 = min(comp)
    return chain_like, len(comp), -r0, -c0


def _endpoint_degrees(path_pts: list[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    pset = set(path_pts)
    deg: dict[tuple[int, int], int] = {}
    ends: list[tuple[int, int]] = []
    for r, c in path_pts:
        d = sum(((int(r + dr), int(c + dc)) in pset) for dr, dc in DIR4)
        deg[(int(r), int(c))] = int(d)
        if d == 1:
            ends.append((int(r), int(c)))
    return deg, ends


def _trace_path(path_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    pset = set(path_pts)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = (int(start[0]), int(start[1]))
    for _ in range(len(path_pts)):
        order.append(cur)
        r, c = cur
        nbrs = [(int(r + dr), int(c + dc)) for dr, dc in DIR4 if (int(r + dr), int(c + dc)) in pset and (int(r + dr), int(c + dc)) != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]
    return order


def _extract_snake(mask: np.ndarray) -> list[tuple[int, int]]:
    comps = _components4_mask(mask)
    if not comps:
        return []
    return max(comps, key=_component_score)


def _marker_start(ends: list[tuple[int, int]], marker: tuple[int, int] | None) -> tuple[int, int]:
    if marker is None:
        return min(ends)
    scored = [(_dist_key(marker, end), end) for end in ends]
    scored.sort()
    return scored[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Foreground contains one long multicolor snake, one special-color marker, and
      several isolated ordinary-colored noise pixels.
    - Ignore every detached singleton except the special marker, keep the large snake
      component, and start from the endpoint nearest to the marker.
    - Output the snake colors in traversal order as an (N x 1) column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    mask = g != bg
    if not np.any(mask):
        return np.zeros((0, 1), dtype=int)

    path_pts = _extract_snake(mask)
    if not path_pts:
        return np.zeros((0, 1), dtype=int)

    path_set = set(path_pts)
    _deg, ends = _endpoint_degrees(path_pts)
    if len(path_pts) == 1:
        start = path_pts[0]
    else:
        assert len(ends) == 2
        marker_candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(g == MARK)
            if (int(r), int(c)) not in path_set
        ]
        marker = None
        if marker_candidates:
            marker = min(marker_candidates, key=lambda pt: min(_dist_key(pt, end) for end in ends))
        start = _marker_start(ends, marker)

    order = _trace_path(path_pts, start)
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
        nr, nc = int(r + dr * i), int(c + dc * i)
        if nr < 1 or nr >= h - 1 or nc < 1 or nc >= w - 1:
            break
        if (nr, nc) in path_set:
            break
        pred = (int(r + dr * (i - 1)), int(c + dc * (i - 1)))
        bad = False
        for ddr, ddc in DIR4:
            nb = (int(nr + ddr), int(nc + ddc))
            if nb != pred and nb in path_set:
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


def _choose_marker_cell(
    path_set: set[tuple[int, int]],
    chosen_end: tuple[int, int],
    other_end: tuple[int, int],
    h: int,
    w: int,
) -> tuple[int, int] | None:
    offsets = list(MARK_OFFSETS)
    random.shuffle(offsets)
    offsets.sort(key=lambda d: (max(abs(d[0]), abs(d[1])), abs(d[0]) + abs(d[1])))
    for dr, dc in offsets:
        nr, nc = int(chosen_end[0] + dr), int(chosen_end[1] + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) in path_set:
            continue
        if any((int(nr + ddr), int(nc + ddc)) in path_set for ddr, ddc in DIR4):
            continue
        if _dist_key((nr, nc), chosen_end) >= _dist_key((nr, nc), other_end):
            continue
        return nr, nc
    return None


def _noise_candidates(h: int, w: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (int(r), int(c))
            if cell in blocked:
                continue
            if any((int(r + dr), int(c + dc)) in blocked for dr, dc in DIR4):
                continue
            out.append(cell)
    return out


def _place_noise(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    marker: tuple[int, int],
    default_start: tuple[int, int],
    palette: list[int],
) -> list[tuple[int, int]] | None:
    h, w = map(int, grid.shape)
    noise_set: set[tuple[int, int]] = set()
    blocked = set(path) | {marker}
    candidates = _noise_candidates(h, w, blocked)
    count = random.randint(3, 6)
    if len(candidates) < count:
        return None

    early_candidates = [p for p in candidates if p < default_start]
    chosen: list[tuple[int, int]] = []
    if early_candidates:
        first = random.choice(early_candidates)
        chosen.append(first)
        noise_set.add(first)

    while len(chosen) < count:
        pool = _noise_candidates(h, w, blocked | noise_set)
        remaining = count - len(chosen)
        if len(pool) < remaining:
            return None
        pick = random.choice(pool)
        chosen.append(pick)
        noise_set.add(pick)

    for cell in chosen:
        grid[cell] = int(random.choice(palette))
    return chosen


def _sample_path_and_colors() -> tuple[list[tuple[int, int]], list[int], int, list[int]] | None:
    h, w = 16, 16
    bg = random.choice([c for c in range(10) if c != MARK])
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c not in (bg, MARK)], non_bg_count)

    target_len = random.randint(20, 30)
    start_r = random.randint(1, 4)
    start_c = random.randint(1, w - 2)

    path = [(int(start_r), int(start_c))]
    path_set = {path[0]}
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
        colors.extend([int(col)] * int(seg_len))

    return path, colors, bg, palette


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_path_and_colors()
    if sampled is None:
        return None
    path, colors, bg, palette = sampled
    h, w = 16, 16

    _deg, ends = _endpoint_degrees(path)
    if len(ends) != 2:
        return None
    default_start = min(ends)
    other_start = max(ends)

    # Prefer marking the non-default endpoint often enough that D2 is visibly active.
    if random.random() < 0.6:
        end_order = [other_start, default_start]
    else:
        end_order = [default_start, other_start]

    marked_start = None
    marker = None
    path_set = set(path)
    for candidate in end_order:
        other_end = ends[0] if candidate == ends[1] else ends[1]
        marker_cell = _choose_marker_cell(path_set, candidate, other_end, h, w)
        if marker_cell is not None:
            marked_start = candidate
            marker = marker_cell
            break
    if marked_start is None or marker is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = int(col)
    grid[marker] = int(MARK)

    noise_cells = _place_noise(grid, path, marker, default_start, palette)
    if noise_cells is None or len(noise_cells) < 3:
        return None

    expected_order = _trace_path(path, marked_start)
    expected_out = np.array([int(grid[r, c]) for r, c in expected_order], dtype=int).reshape((-1, 1))
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
