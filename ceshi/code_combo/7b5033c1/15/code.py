# Combo variant of 7b5033c1 (index 15)
# combo_spec: D8/1+D9/2
# Name: 条件过滤+段重排
#
# Design analysis for the fused variant:
# a) Concrete behavior and trace:
#    The input is still one multicolor 4-connected snake with exactly two endpoints.
#    As in the source task, the solver first traces the snake from the top-left endpoint
#    and linearizes its colors. The fusion changes what happens after serialization:
#    1. Split the 1D color sequence into maximal equal-color runs.
#    2. Delete every run whose length is 1 or 2.
#    3. Reorder the surviving whole runs by color value ascending.
#    Example:
#      traced colors   = [8,8,8, 3, 6,6,6,6, 1,1, 4,4,4]
#      runs            = [(3,8), (1,3), (4,6), (2,1), (3,4)]
#      keep len>=3     = [(3,8), (4,6), (3,4)]
#      sort by color   = [(3,4), (4,6), (3,8)]
#      output column   = [4,4,4, 6,6,6,6, 8,8,8]^T
#    Compared with the original task, traversal order is no longer the final answer.
#    Compared with the single D9 variant, surviving runs no longer stay in traversal order.
# b) Complexity calibration:
#    The generator explicitly avoids three degeneracies:
#    - no filtering: rejected unless at least 2 short runs are deleted;
#    - no reordering: rejected unless the kept-run color order is not already ascending;
#    - over-fragmented ambiguity: kept runs use 3 or 4 distinct colors so the ascending-color
#      reorder is visible, but the palette remains small enough to infer from 2-4 examples.
# c) Implicit-assumption audit against main():
#    - get_bg() still assumes the background is globally most common; generator preserves that.
#    - non-bg cells are still assumed to form a single simple 4-connected chain; generator
#      keeps one-cell margins, forbids revisits, and forbids side-adjacency except to predecessor.
#    - endpoint tracing still assumes either one cell or exactly two degree-1 endpoints.
#    - run processing introduces a new semantic assumption: "segment" means a contiguous run
#      along traced order, not a spatial blob. Generator colors cells only after canonical path
#      order is known, so this assumption is stable.
#    - reordering assumes whole surviving runs are the sortable units. Generator keeps adjacent
#      segment colors distinct, so no accidental merge happens before filtering.
# d) Placement legality:
#    Path construction checks bounds, overlap, and forbidden 4-neighbor contact on every added
#    cell, so the final object is always a legal non-self-touching snake.
# e) Visual observability:
#    Across samples, the output is both shorter than the full snake and visibly regrouped into
#    ascending-color bands. This gives the solver two independent clues: short inserts vanish,
#    and the remaining long bands are sorted by color.
# f) Visualization cleanliness:
#    There is still only one object in the grid. No overlays or markers are added. Short and
#    long runs are expressed only through band lengths along the snake, so the scene remains
#    visually sparse and readable.
# g) Color-change inferability:
#    The fusion does not remap colors; it only deletes some runs and reorders survivors.
#    The generator restricts kept runs to 3-4 distinct colors, which is enough to observe
#    ascending-color sorting without overwhelming the sample.
# 9) New assumptions introduced by the fused main():
#    - There is at least one surviving run for informative training cases; generator enforces it.
#    - The final reorder key is color id ascending. Generator guarantees at least 3 surviving
#      colors and rejects already-ascending kept order, so this key has visible effect.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trace_path_order(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points:
        return []
    if len(points) == 1:
        r, c = points[0]
        return [(int(r), int(c))]

    pts = [(int(r), int(c)) for r, c in points]
    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2, "Expected a simple snake with exactly two endpoints"
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)
        nbrs = [q for q in _neighbors4(*cur) if q in pset and q != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1, "Path must be a simple chain"
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts), "All path cells must be visited"
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


def _filter_segments(segments: list[tuple[int, int]], min_len: int = 3) -> list[tuple[int, int]]:
    return [(seg_len, color) for seg_len, color in segments if int(seg_len) >= int(min_len)]


def _reorder_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(segments, key=lambda seg: seg[1])


def _flatten_segments(segments: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    for seg_len, color in segments:
        out.extend([int(color)] * int(seg_len))
    return out


def _apply_combo(colors: list[int]) -> list[int]:
    segments = _segmentize(colors)
    kept = _filter_segments(segments, min_len=3)
    reordered = _reorder_segments(kept)
    return _flatten_segments(reordered)


def _variant_metrics(colors: list[int]) -> dict[str, int]:
    segments = _segmentize(colors)
    kept = _filter_segments(segments, min_len=3)
    filtered = _flatten_segments(kept)
    final = _flatten_segments(_reorder_segments(kept))
    kept_colors = [color for _seg_len, color in kept]
    deleted_pixels = len(colors) - len(filtered)
    return {
        "segment_count": len(segments),
        "deleted_segment_count": len(segments) - len(kept),
        "kept_segment_count": len(kept),
        "kept_distinct_colors": len(set(kept_colors)),
        "deleted_pixels": deleted_pixels,
        "output_len": len(final),
        "filter_changed": int(filtered != colors),
        "reorder_changed": int(final != filtered),
        "combo_changed": int(final != colors),
    }


def _is_visible_variant(colors: list[int]) -> bool:
    metrics = _variant_metrics(colors)
    return (
        metrics["segment_count"] >= 6
        and metrics["deleted_segment_count"] >= 2
        and metrics["kept_segment_count"] >= 3
        and metrics["kept_distinct_colors"] >= 3
        and metrics["deleted_pixels"] >= 3
        and metrics["output_len"] >= 9
        and metrics["filter_changed"] == 1
        and metrics["reorder_changed"] == 1
        and metrics["combo_changed"] == 1
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the snake, delete short runs, then sort surviving runs by color ascending.
    """
    colors = _trace_path_colors(np.array(grid, dtype=int))
    if not colors:
        return np.zeros((0, 1), dtype=int)
    out = _apply_combo(colors)
    return np.array(out, dtype=int).reshape((-1, 1))


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
        for nb in _neighbors4(nr, nc):
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _random_typed_partition(total: int) -> tuple[list[int], set[int]] | None:
    """
    Return segment lengths and the indices of the long segments (length >= 3).

    Guarantees for accepted outputs:
    - at least 2 short segments (length 1 or 2),
    - at least 3 long segments (length >= 3),
    - enough kept pixels for a readable output strip.
    """
    for _ in range(300):
        num_segments = random.randint(6, 8)
        long_count = random.randint(3, min(4, num_segments - 2))
        long_idx = set(random.sample(range(num_segments), long_count))

        parts = [3 if i in long_idx else 1 for i in range(num_segments)]
        caps = [7 if i in long_idx else 2 for i in range(num_segments)]
        remaining = total - sum(parts)
        if remaining < 0:
            continue

        while remaining > 0:
            eligible = [i for i in range(num_segments) if parts[i] < caps[i]]
            if not eligible:
                break
            pick = random.choice(eligible)
            parts[pick] += 1
            remaining -= 1
        if remaining != 0:
            continue

        if sum(1 for x in parts if x < 3) < 2:
            continue
        if sum(1 for x in parts if x >= 3) < 3:
            continue
        if sum(x for x in parts if x < 3) < 3:
            continue
        if sum(x for x in parts if x >= 3) < 10:
            continue
        return parts, {i for i, seg_len in enumerate(parts) if seg_len >= 3}

    return None


def _build_segment_colors(palette: list[int], seg_lengths: list[int], long_idx: set[int]) -> list[int] | None:
    num_segments = len(seg_lengths)
    long_positions = sorted(long_idx)
    if len(long_positions) < 3:
        return None

    for _ in range(300):
        seg_colors = [-1] * num_segments

        long_colors = random.sample(palette, len(long_positions))
        random.shuffle(long_colors)
        if long_colors == sorted(long_colors):
            continue
        for pos, color in zip(long_positions, long_colors):
            seg_colors[pos] = color

        ok = True
        for i in range(num_segments):
            if seg_colors[i] != -1:
                continue
            choices = list(palette)
            random.shuffle(choices)
            prev_color = seg_colors[i - 1] if i > 0 else None
            next_fixed = None
            for j in range(i + 1, num_segments):
                if seg_colors[j] != -1:
                    next_fixed = seg_colors[j]
                    break
            placed = False
            for color in choices:
                if prev_color is not None and color == prev_color:
                    continue
                if next_fixed is not None and color == next_fixed:
                    continue
                seg_colors[i] = color
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        if any(seg_colors[i] == seg_colors[i - 1] for i in range(1, num_segments)):
            continue

        kept_colors = [seg_colors[i] for i in range(num_segments) if seg_lengths[i] >= 3]
        if len(set(kept_colors)) < 3:
            continue
        if kept_colors == sorted(kept_colors):
            continue
        return [int(color) for color in seg_colors]

    return None


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    non_bg_count = random.randint(4, 6)
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
        run_len = min(random.randint(1, 5), target_len - len(path))
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
    if not (20 <= total <= 30):
        return None
    if not (7 <= turns <= 11):
        return None

    partition = _random_typed_partition(total)
    if partition is None:
        return None
    seg_lengths, long_idx = partition

    seg_colors = _build_segment_colors(palette, seg_lengths, long_idx)
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

    expected_out = np.array(_apply_combo(colors), dtype=int).reshape((-1, 1))
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
