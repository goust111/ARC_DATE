# Variant of 7b5033c1
# Dimension: D8 — 多步处理与序列化 (meta)
# Change: 单步处理：直接找到路径并线性化输出... → 线性化后再重排序
#
# Design analysis for this D8 variant:
# a) Concrete behavior:
#    The input is still one multicolor 4-connected snake. Step 1 is unchanged:
#    trace the snake from the top-left endpoint and read its color sequence.
#    Step 2 is new: compress that sequence into contiguous color segments and
#    sort whole segments by (-segment_length, color). Example:
#    traversal colors = [2,2,2, 5,5, 7,7,7,7, 2,2,2]
#    segments         = [(3,2), (2,5), (4,7), (3,2)]
#    sorted segments  = [(4,7), (3,2), (3,2), (2,5)]
#    output column    = [7,7,7,7, 2,2,2, 2,2,2, 5,5]^T
#    The important difference is that the output is no longer the raw traversal
#    order: the solver must first serialize, then reorder segment units.
# b) Complexity calibration:
#    If segment lengths were nearly identical or already sorted, the D8 change
#    would be visually weak. The generator therefore rejects samples unless
#    there are at least 4 segments, at least 3 distinct segment lengths, a
#    non-adjacent repeated color, and the sorted output differs from the raw
#    traversal sequence.
# c) Implicit-assumption audit:
#    - Path extraction code still relies on the old geometric invariant:
#      non-background cells form a single simple chain with two endpoints.
#      The generator preserves that.
#    - The original final line "return traversal colors directly" depended on
#      traversal order being the full rule. That assumption is exactly what D8
#      changes; now traversal colors are only an intermediate representation.
#    - Segment sorting assumes equal-color runs stay contiguous in the traced
#      sequence. The generator guarantees segment colors are assigned run-wise
#      and never merges adjacent runs with the same color.
# d) Placement legality:
#    The snake generator keeps a one-cell border, forbids revisits, and forbids
#    side-adjacencies except to the predecessor, so the placed object remains a
#    legal non-self-touching chain. Colors are assigned only after computing
#    the canonical endpoint-to-endpoint order used by the solver.
# e) Visual observability:
#    Across examples, the input still shows mixed run lengths along the snake,
#    while the output starts with the longest bands first. Repeated colors in
#    separated runs stop the task from collapsing into simple color counting.
# 9) New assumptions introduced by main():
#    - Sorting units are contiguous traversal segments, not whole colors.
#    - Ties are resolved by color id ascending.
#    - The generator enforces these assumptions by creating run-based coloring,
#      keeping adjacent segment colors distinct, and rejecting degenerate cases.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


def _trace_path_order(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points:
        return []
    if len(points) == 1:
        return [(int(points[0][0]), int(points[0][1]))]

    pts = [(int(r), int(c)) for r, c in points]
    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        d = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in pset:
                d += 1
        deg[(r, c)] = d

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)
        r, c = cur
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts)
    return order


def _trace_path_colors(grid: np.ndarray) -> list[int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    points = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    order = _trace_path_order(points)
    return [int(g[r, c]) for r, c in order]


def _segmentize(colors: list[int]) -> list[tuple[int, int]]:
    if not colors:
        return []
    segments: list[tuple[int, int]] = []
    cur_color = int(colors[0])
    cur_len = 1
    for color in colors[1:]:
        color = int(color)
        if color == cur_color:
            cur_len += 1
        else:
            segments.append((cur_len, cur_color))
            cur_color = color
            cur_len = 1
    segments.append((cur_len, cur_color))
    return segments


def _reorder_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(segments, key=lambda seg: (-seg[0], seg[1]))


def _flatten_segments(segments: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    for seg_len, color in segments:
        out.extend([int(color)] * int(seg_len))
    return out


def _has_nonadjacent_repeat(seg_colors: list[int]) -> bool:
    for i, color in enumerate(seg_colors):
        for j in range(i + 2, len(seg_colors)):
            if seg_colors[j] == color:
                return True
    return False


def _variant_metrics(colors: list[int]) -> dict[str, int]:
    segments = _segmentize(colors)
    seg_lengths = [seg_len for seg_len, _color in segments]
    seg_colors = [color for _seg_len, color in segments]
    reordered = _flatten_segments(_reorder_segments(segments))
    return {
        "segment_count": len(segments),
        "distinct_lengths": len(set(seg_lengths)),
        "length_span": max(seg_lengths) - min(seg_lengths),
        "nonadjacent_repeat": int(_has_nonadjacent_repeat(seg_colors)),
        "changed_after_sort": int(reordered != colors),
    }


def _is_visible_variant(colors: list[int]) -> bool:
    metrics = _variant_metrics(colors)
    return (
        metrics["segment_count"] >= 4
        and metrics["distinct_lengths"] >= 3
        and metrics["length_span"] >= 2
        and metrics["nonadjacent_repeat"] == 1
        and metrics["changed_after_sort"] == 1
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Step 1: trace the snake from the top-left endpoint into a 1D color sequence.
    Step 2: split that sequence into contiguous color segments and reorder whole
    segments by decreasing length, breaking ties by increasing color id.
    """
    colors = _trace_path_colors(np.array(grid, dtype=int))
    if not colors:
        return np.zeros((0, 1), dtype=int)
    segments = _segmentize(colors)
    reordered = _flatten_segments(_reorder_segments(segments))
    return np.array(reordered, dtype=int).reshape((-1, 1))


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in path_set:
            break
        pred = (r + dr * (i - 1), c + dc * (i - 1))
        bad = False
        for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (nr + ddr, nc + ddc)
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


def _build_segment_colors(palette: list[int], num_segments: int) -> list[int] | None:
    for _ in range(200):
        seg_colors = list(palette)
        random.shuffle(seg_colors)
        while len(seg_colors) < num_segments:
            choices = [c for c in palette if c != seg_colors[-1]]
            seg_colors.append(random.choice(choices))
        if any(seg_colors[i] == seg_colors[i - 1] for i in range(1, len(seg_colors))):
            continue
        if not _has_nonadjacent_repeat(seg_colors):
            continue
        return seg_colors
    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], non_bg_count)

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

    num_segments = random.randint(max(4, non_bg_count + 1), 6)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=9)
    if seg_lengths is None:
        return None

    seg_colors = _build_segment_colors(palette, num_segments)
    if seg_colors is None:
        return None

    canonical_path = _trace_path_order(path)
    colors = _flatten_segments(list(zip(seg_lengths, seg_colors)))
    if len(colors) != len(canonical_path):
        return None
    if not _is_visible_variant(colors):
        return None

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), color in zip(canonical_path, colors):
        grid[r, c] = color

    expected_out = np.array(
        _flatten_segments(_reorder_segments(_segmentize(colors))),
        dtype=int,
    ).reshape((-1, 1))
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
