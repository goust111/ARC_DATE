# Variant of 7b5033c1
# Dimension: D2 — 遍历起点选择规则 (transform)
# Change: 从两个端点中选择左上角优先的端点（min(row, col)）开始遍历... → 由标记像素指定起点

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "arc2_solver_select"))
    from arc_utils import get_bg


# Design analysis for this D2 variant
# a) Concrete behavior and trace:
#    - Input still contains one multicolor 4-connected snake path. A single red marker
#      pixel is added near exactly one endpoint, but it is kept off the path by making
#      sure it is not 4-neighbor to any snake cell.
#    - Original logic: endpoints = [(1, 11), (12, 8)] -> start = min(endpoints) = (1, 11).
#      Variant logic: marker at (11, 9) sits closest to endpoint (12, 8), so traversal
#      starts there instead and the output strip is the reversed snake-color order.
#    - The core change therefore hits the "choose traversal start" stage only; path
#      tracing and output shaping stay unchanged.
# b) Complexity calibration:
#    - A disconnected marker component is visually obvious and does not blend into the
#      snake itself. The generator lets the marker point to either endpoint, so the
#      task is really "read the marker" rather than "always reverse the path".
# c) Old hidden assumptions that break now:
#    - `mask = g != bg` used to mean "all foreground belongs to the snake". Now one
#      foreground pixel is not part of the snake, so the solver must split foreground
#      into 4-connected components and keep only the large chain component for degree
#      counting and tracing.
#    - `start = min(ends)` is no longer valid; start must be chosen from marker geometry.
# d) Placement legality:
#    - The generator only uses marker cells that are in-bounds, empty, not 4-neighbor to
#      any snake cell, and strictly closer to the designated endpoint than to the other
#      endpoint under the solver's distance rule.
# e) Visual observability:
#    - In examples, the extra red pixel is the only isolated symbol and it sits beside the
#      endpoint whose color appears first in the output strip. Some samples therefore
#      match the original left-top endpoint choice and some reverse it; the intended
#      rule is to follow the marker, not to infer a fixed directional bias.
# 9) New solver assumptions and generator guarantees:
#    - The largest 4-connected non-background component is the snake.
#    - There is at least one extra non-background cell outside that component; the nearest
#      such cell is the start marker.
#    - The marked endpoint is uniquely closest to that marker by (Chebyshev, Manhattan)
#      distance. `generate_input()` enforces all three conditions before accepting a grid.


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
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))), abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
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
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(comp)
    return comps


def _endpoint_degrees(path_pts: list[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    pset = set(path_pts)
    deg: dict[tuple[int, int], int] = {}
    ends: list[tuple[int, int]] = []
    for r, c in path_pts:
        d = 0
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in pset:
                d += 1
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
        nbrs = []
        for dr, dc in DIR4:
            q = (int(r + dr), int(c + dc))
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break
        prev, cur = cur, min(nbrs)
    return order


def _marker_start(ends: list[tuple[int, int]], marker: tuple[int, int] | None) -> tuple[int, int]:
    if marker is None:
        return min(ends)
    scored = [(_dist_key(marker, end), end) for end in ends]
    scored.sort()
    return scored[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a colored snake path into a single-column strip preserving pixel order.

    Variant rule:
    - Non-bg cells contain one large snake component plus one isolated marker pixel.
    - The endpoint nearest to the marker is the traversal start.
    - Output is the snake's colors in that traversal order, as an (N x 1) column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    mask = g != bg
    if not np.any(mask):
        return np.zeros((0, 1), dtype=int)

    comps = _components4(mask)
    path_pts = max(comps, key=len)
    extra_pts = [pt for comp in comps if comp is not path_pts for pt in comp]

    _deg, ends = _endpoint_degrees(path_pts)
    assert len(ends) == 2

    marker = None
    if extra_pts:
        marker = min(extra_pts, key=lambda pt: min(_dist_key(pt, end) for end in ends))

    start = _marker_start(ends, marker)
    order = _trace_path(path_pts, start)
    cols = [int(g[r, c]) for r, c in order]
    return np.array(cols, dtype=int).reshape((-1, 1))


def _try_extend(path_set: set[tuple[int, int]], last: tuple[int, int], dr: int, dc: int, count: int, H: int, W: int) -> list[tuple[int, int]]:
    """Try to extend the path by `count` steps in direction (dr, dc)."""
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = int(r + dr * i), int(c + dc * i)
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
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
    """Partition total into num_parts values each in [min_len, max_len]."""
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


def _choose_marker_cell(path_set: set[tuple[int, int]], chosen_end: tuple[int, int], other_end: tuple[int, int], H: int, W: int) -> tuple[int, int] | None:
    offsets = list(MARK_OFFSETS)
    random.shuffle(offsets)
    offsets.sort(key=lambda d: (max(abs(d[0]), abs(d[1])), abs(d[0]) + abs(d[1])))
    for dr, dc in offsets:
        nr, nc = int(chosen_end[0] + dr), int(chosen_end[1] + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if (nr, nc) in path_set:
            continue
        if any((int(nr + ddr), int(nc + ddc)) in path_set for ddr, ddc in DIR4):
            continue
        if _dist_key((nr, nc), chosen_end) >= _dist_key((nr, nc), other_end):
            continue
        return nr, nc
    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.choice([c for c in range(10) if c != MARK])
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c not in (bg, MARK)], non_bg_count)

    target_len = random.randint(20, 30)

    start_r = random.randint(1, 4)
    start_c = random.randint(1, W - 2)

    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

    while len(path) < target_len:
        run_len = random.randint(1, 5)
        run_len = min(run_len, target_len - len(path))
        if run_len <= 0:
            break

        last = path[-1]
        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, H, W)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, H, W)
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

    _deg, ends = _endpoint_degrees(path)
    if len(ends) != 2:
        return None
    default_start = min(ends)
    marked_start = random.choice(ends)
    other_end = ends[0] if marked_start == ends[1] else ends[1]

    marker = _choose_marker_cell(path_set, marked_start, other_end, H, W)
    if marker is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = int(col)
    grid[marker] = int(MARK)

    expected_order = _trace_path(path, marked_start)
    expected_out = np.array([int(grid[r, c]) for r, c in expected_order], dtype=int).reshape((-1, 1))
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 400 attempts")
