# Combo variant of 7b5033c1 (index 1)
# combo_spec: D1/0+D2/2
# Name: 标记起点+折叠矩阵
#
# Design analysis
# a) Concrete behavior and trace:
#    - Original task: every non-background cell belongs to one 4-connected snake.
#      The solver picks the top-left endpoint, traces the snake, and outputs an N x 1
#      color column.
#    - Combo task: the input now has two foreground components:
#        1. one large multicolor snake component
#        2. one isolated marker pixel near exactly one snake endpoint
#      The solver must first split components, keep the large component as the snake,
#      then choose the endpoint nearest to the marker as the traversal start.
#      After tracing that color stream, it folds the stream into a boustrophedon
#      matrix whose width K is the number of distinct snake colors.
#    - Example sketch:
#      Suppose the snake colors from endpoint A are
#      [1,1,3,3,8,8,4,4,1,1,3,3,8,8,4,4], with K = 4.
#      Original output from endpoint A:
#      [[1],[1],[3],[3],...,[4],[4]]
#      Combo output if the marker points to A:
#      [[1,1,3,3],
#       [4,4,8,8],
#       [1,1,3,3],
#       [4,4,8,8]]
#      Combo output if the marker points to the opposite endpoint B:
#      trace order reverses first, then the same K = 4 fold is applied to the reversed
#      stream, so both the first seen color and several row contents change.
#    - The fused difference is therefore not "marker only" or "fold only":
#      marker geometry decides which endpoint seeds the 1-D sequence, and the fold
#      width rule decides how that sequence becomes a 2-D output.
# b) Complexity calibration:
#    - Snake color count K is restricted to 3..5 so width is visible but not too wide.
#    - Snake length is chosen as a multiple of K, with 4..7 rows after folding, so
#      the rectangle is neither degenerate nor ambiguous.
#    - Colors appear in contiguous segments and every color repeats, giving humans
#      enough support to notice both the start-direction effect and the fold pattern.
#    - The marker uses one fixed special color and sits within a short list of offsets,
#      so the signal is visually crisp instead of noisy.
# c) Implicit-assumption audit of the original solver:
#    - `mask = g != bg` no longer means "all foreground is the snake"; the marker is
#      an extra non-background component. The combo solver must extract components and
#      keep only the largest one for degree counting and tracing.
#    - `start = min(ends)` is invalid because traversal direction is now specified by
#      the marker location. The combo solver replaces it with nearest-endpoint logic.
#    - `reshape((-1, 1))` is invalid because the output width is no longer fixed at 1.
#      It is replaced by `width = number of distinct snake colors` plus row-wise
#      boustrophedon folding.
#    - New interaction check: the marker must not be counted as a snake color, so fold
#      width must be computed from traced snake cells only. Splitting components first
#      resolves that conflict cleanly.
# d) Placement legality:
#    - The generator grows the snake inside the border and rejects any extension that
#      would touch an older snake cell except through its predecessor, so the path is a
#      simple chain with exactly two endpoints.
#    - The marker is placed in-bounds, off the snake, not 4-neighbor to the snake, and
#      strictly closer to the chosen endpoint than to the other endpoint under the same
#      distance rule used by the solver.
#    - The snake palette excludes the marker color, and the length is divisible by K,
#      so solver assumptions about width and folding are satisfied.
# e) Visual observability:
#    - Across examples, the only isolated symbol is the marker pixel, so the start cue
#      is easy to spot by eye.
#    - Output width changes with the number of distinct snake colors, which is visible
#      in the input because those colors recur along the snake.
#    - When the marker picks the non-default endpoint, the first output row changes in
#      a way that cannot be explained by the old top-left rule alone.
# f) Visual cleanliness:
#    - The scene contains only one snake and one isolated marker, with no overlapping
#      object layers.
#    - The marker is kept at diagonal or distance-2 offsets, avoiding accidental 4-link
#      fusion with the snake while still reading as "near this endpoint".
#    - The fold happens only in the output, so the input remains a single clean object
#      plus one cue pixel.
# g) Color-change inferability:
#    - There is no arbitrary recoloring. Colors are preserved exactly; only traversal
#      order and 2-D arrangement change.
#    - The palette size is capped at 3..5 and every chosen color appears multiple times
#      in the snake, so the width rule is inferable from a few examples.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: the largest 4-connected non-background component is the snake.
#      Guarantee: the marker is a single isolated component.
#    - Assumption: the snake is a simple chain with exactly two endpoints.
#      Guarantee: `_try_extend` prevents branches and self-touching placements.
#    - Assumption: the marked endpoint is uniquely nearest to the marker by the solver's
#      `(Chebyshev, Manhattan)` tie-break key.
#      Guarantee: marker placement enforces strict ordering under that key.
#    - Assumption: traced snake length is divisible by the distinct snake-color count.
#      Guarantee: generator chooses target lengths from multiples of K only.
#    - Assumption: when the generator claims to use the non-default endpoint, that
#      direction actually changes the combo output.
#      Guarantee: `_gen_attempt` rejects samples where marker-chosen traversal and
#      top-left traversal would fold to the same matrix.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import get_bg


MARK = 2
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MARK_OFFSETS = [
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
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
]


def _dist_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (
        max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))),
        abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])),
    )


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
                    stack.append((nr, nc))
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
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]
    assert len(order) == len(path_pts)
    return order


def _marker_start(ends: list[tuple[int, int]], marker: tuple[int, int] | None) -> tuple[int, int]:
    if marker is None:
        return min(ends)
    scored = [(_dist_key(marker, end), end) for end in ends]
    scored.sort()
    return scored[0][1]


def _fold_colors(colors: list[int], width: int) -> np.ndarray:
    assert width > 0
    assert len(colors) % width == 0
    height = len(colors) // width
    out = np.zeros((height, width), dtype=int)
    idx = 0
    for r in range(height):
        chunk = colors[idx : idx + width]
        idx += width
        out[r] = chunk if r % 2 == 0 else chunk[::-1]
    return out


def _extract_snake_and_marker(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    mask = g != bg
    if not np.any(mask):
        return [], None

    comps = _components4(mask)
    path_idx = max(range(len(comps)), key=lambda i: len(comps[i]))
    path_pts = comps[path_idx]
    extra_pts = [pt for i, comp in enumerate(comps) if i != path_idx for pt in comp]

    marker = None
    if extra_pts:
        deg, ends = _endpoint_degrees(path_pts)
        assert len(ends) == 2
        assert all(d <= 2 for d in deg.values())
        marker = min(extra_pts, key=lambda pt: min(_dist_key(pt, end) for end in ends))
    return path_pts, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the marker-designated snake direction, then fold the traced colors into a
    boustrophedon matrix whose width equals the number of distinct snake colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    path_pts, marker = _extract_snake_and_marker(g, bg)
    if not path_pts:
        return np.zeros((0, 0), dtype=int)

    deg, ends = _endpoint_degrees(path_pts)
    assert len(ends) == 2
    assert all(d <= 2 for d in deg.values())

    start = _marker_start(ends, marker)
    order = _trace_path(path_pts, start)
    colors = [int(g[r, c]) for r, c in order]
    fold_width = len(set(colors))
    return _fold_colors(colors, fold_width)


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Try to extend the path by `count` steps in direction (dr,dc)."""
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


def _build_color_stream(total: int, palette: list[int]) -> list[int] | None:
    color_count = len(palette)
    min_segments = max(color_count + 1, 6)
    max_segments = min(9, total // 2)
    if min_segments > max_segments:
        return None

    num_segments = random.randint(min_segments, max_segments)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=7)
    if seg_lengths is None:
        return None

    seg_colors = palette[:]
    random.shuffle(seg_colors)
    while len(seg_colors) < num_segments:
        prev = seg_colors[-1]
        choices = [c for c in palette if c != prev]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([int(col)] * int(seg_len))

    if len(set(colors)) != color_count:
        return None
    return colors


def _choose_marker_cell(
    path_set: set[tuple[int, int]],
    chosen_end: tuple[int, int],
    other_end: tuple[int, int],
    H: int,
    W: int,
) -> tuple[int, int] | None:
    candidates = []
    for dr, dc in MARK_OFFSETS:
        nr, nc = int(chosen_end[0] + dr), int(chosen_end[1] + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if (nr, nc) in path_set:
            continue
        if any((int(nr + ddr), int(nc + ddc)) in path_set for ddr, ddc in DIR4):
            continue
        if _dist_key((nr, nc), chosen_end) >= _dist_key((nr, nc), other_end):
            continue
        candidates.append((nr, nc))
    if not candidates:
        return None
    return random.choice(candidates)


def _path_and_colors_to_grid(
    path: list[tuple[int, int]],
    colors: list[int],
    bg: int,
    H: int,
    W: int,
    marker: tuple[int, int] | None = None,
) -> np.ndarray:
    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = int(col)
    if marker is not None:
        grid[marker] = int(MARK)
    return grid


def _ordered_colors_for_start(path: list[tuple[int, int]], colors: list[int], start: tuple[int, int]) -> list[int]:
    if start == path[0]:
        return list(colors)
    if start == path[-1]:
        return list(reversed(colors))
    index = {pt: i for i, pt in enumerate(path)}
    order = _trace_path(path, start)
    return [int(colors[index[pt]]) for pt in order]


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.choice([c for c in range(10) if c != MARK])
    color_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c not in (bg, MARK)], color_count)

    lengths = [n for n in range(16, 31) if n % color_count == 0 and 4 <= n // color_count <= 7]
    if not lengths:
        return None
    target_len = random.choice(lengths)

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
    if total != target_len:
        return None
    if turns < 6 or turns > 12:
        return None

    deg, ends = _endpoint_degrees(path)
    if len(ends) != 2 or not all(d <= 2 for d in deg.values()):
        return None

    colors = _build_color_stream(total, palette)
    if colors is None:
        return None

    default_start = min(ends)
    other_start = ends[0] if ends[1] == default_start else ends[1]
    marked_start = other_start if random.random() < 0.65 else default_start
    marker = _choose_marker_cell(path_set, marked_start, default_start if marked_start == other_start else other_start, H, W)
    if marker is None:
        return None

    chosen_colors = _ordered_colors_for_start(path, colors, marked_start)
    default_colors = _ordered_colors_for_start(path, colors, default_start)
    expected_out = _fold_colors(chosen_colors, color_count)
    baseline_out = _fold_colors(default_colors, color_count)

    if marked_start != default_start and np.array_equal(expected_out, baseline_out):
        return None

    grid = _path_and_colors_to_grid(path, colors, bg, H, W, marker)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 600 attempts")
