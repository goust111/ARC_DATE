# Combo variant of 7b5033c1 (index 11)
# combo_spec: D4/1+D8/1
# Name: 渐变颜色+线性化后重排

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import get_bg


# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    The object is still one simple 4-connected snake, but the intended reading direction is no
#    longer geometric. Colors are assigned as contiguous monochrome segments whose segment-colors
#    follow a repeated ascending palette cycle, e.g. palette [2, 5, 7] and segments
#    [(4,2), (2,5), (5,7), (3,2), (2,5), (4,7)].
#    Original task: choose the top-left endpoint, trace, and output raw colors.
#    D4 cue here: trace from both endpoints and keep the traversal whose segment-colors best match
#    2,5,7,2,5,7,... . The reverse traversal gives 7,5,2,7,5,2,... and therefore scores worse.
#    D8 step here: after the correct direction is found, reorder whole segments by segment length
#    descending. The sort is intentionally stable: ties keep their traversal order. For the example
#    above, the output segments become [(5,7), (4,2), (4,7), (3,2), (2,5), (2,5)].
#    This stable-tie choice is what lets the D4 direction still matter after the D8 reorder:
#    reversing the snake reverses same-length tie groups, changing the final strip.
# b) Complexity calibration:
#    The generator uses 3-4 colors, 6-8 segments, and segment lengths in [2,7]. It rejects cases
#    unless there are at least 3 distinct lengths, at least one repeated length shared by different
#    colors, the stable length-sort changes the raw traversal, and sorting after reversing the path
#    would produce a different output. That keeps both dimensions visually active without creating a
#    combinatorial mess.
# c) Implicit-assumption audit of main():
#    - Path extraction still assumes exactly one simple chain with two endpoints. The generator keeps
#      one-cell margins, forbids revisits, and forbids side-adjacencies except to the predecessor.
#    - Direction inference no longer relies on `min(endpoints)`. It assumes one endpoint traversal
#      matches the repeated ascending segment-color cycle better than the other; the generator rejects
#      ambiguous samples.
#    - Segment sorting assumes the meaningful units are contiguous same-color runs. The generator
#      colors the snake run-wise and cycles colors so adjacent runs never merge.
#    - Sorting by length alone would collapse to a direction-insensitive multiset if no informative
#      ties existed. The generator therefore guarantees at least one cross-color tie length so stable
#      tie order carries real information from the chosen traversal.
# d) Placement legality:
#    `_try_extend` keeps the snake inside a 16x16 board with a one-cell border, forbids overlaps,
#    and rejects any step that would touch an older path cell other than its predecessor. This
#    preserves a legal simple snake throughout generation.
# e) Visual observability:
#    Across examples, humans can see two regularities: the snake colors repeat cyclically by segment
#    (A,B,C,A,B,C,...) from one endpoint only, and the output groups the longest bands first. Equal
#    band lengths reappear in multiple colors, so choosing the wrong endpoint visibly changes the
#    order inside those tied groups.
# f) Visual cleanliness:
#    There is only one object class: the snake. No auxiliary markers or overlays are added. Limiting
#    the palette to 3-4 colors and segment count to 6-8 keeps the grid readable.
# g) Color-change inferability:
#    Colors do not undergo an arbitrary mapping. The structural cue is the repeated ascending cycle
#    over the same palette, which is reinforced several times in each example. The color behavior is
#    therefore inferable from a small number of samples.
# h) Quantitative anti-degeneracy metric:
#    `variant_metrics()` reports `direction_margin`, `sort_changes_sequence`, and
#    `direction_changes_sorted_output`. A sample is accepted only if all three are active.
# i) New assumptions introduced by main():
#    - The repeated pattern is defined over segment-colors, not over individual pixels.
#    - Length sorting is stable and keyed only by length.
#    - Enough segments exist for at least two cycle repetitions or near-repetitions.
#    `generate_input()` explicitly enforces these assumptions.

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _path_neighbors(cell: tuple[int, int], pset: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pset]


def _trace_path(start: tuple[int, int], pset: set[tuple[int, int]]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start

    for _ in range(len(pset)):
        order.append(cur)
        nxts = [q for q in _path_neighbors(cur, pset) if q != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise ValueError("Path is not a simple chain")
        prev, cur = cur, nxts[0]

    if len(order) != len(pset):
        raise ValueError("Failed to trace full path")
    return order


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


def _flatten_segments(segments: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    for seg_len, color in segments:
        out.extend([int(color)] * int(seg_len))
    return out


def _reorder_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # Python's sort is stable, so equal-length segments preserve traversal order.
    return sorted(segments, key=lambda seg: -int(seg[0]))


def _cycle_mismatch_count(segments: list[tuple[int, int]], palette: list[int]) -> int:
    period = len(palette)
    return sum(int(color != palette[i % period]) for i, (_seg_len, color) in enumerate(segments))


def _extract_path_info(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    pset = set(pts)
    deg = {p: len(_path_neighbors(p, pset)) for p in pts}
    ends = [p for p in pts if deg[p] == 1]
    return g, pts, pset, ends


def _choose_direction_and_segments(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int]], dict[str, int]]:
    g, pts, pset, ends = _extract_path_info(grid)
    if not pts:
        return [], [], {"chosen_index": 0, "direction_margin": 0}
    if len(ends) != 2:
        raise ValueError("Expected exactly one snake path with two endpoints")

    palette = sorted({int(g[r, c]) for r, c in pts})
    traced_orders = [_trace_path(end, pset) for end in ends]
    traced_colors = [[int(g[r, c]) for r, c in order] for order in traced_orders]
    traced_segments = [_segmentize(colors) for colors in traced_colors]
    mismatch_scores = [_cycle_mismatch_count(segments, palette) for segments in traced_segments]

    if mismatch_scores[0] < mismatch_scores[1]:
        chosen_index = 0
    elif mismatch_scores[1] < mismatch_scores[0]:
        chosen_index = 1
    else:
        chosen_index = ends.index(min(ends))

    meta = {
        "chosen_index": int(chosen_index),
        "direction_margin": int(abs(mismatch_scores[0] - mismatch_scores[1])),
    }
    return traced_colors[chosen_index], traced_segments[chosen_index], meta


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g, pts, pset, ends = _extract_path_info(grid)
    if not pts:
        return {
            "segment_count": 0,
            "distinct_lengths": 0,
            "length_span": 0,
            "cross_color_tie_lengths": 0,
            "sort_changes_sequence": 0,
            "direction_margin": 0,
            "direction_changes_sorted_output": 0,
            "top_left_is_wrong_start": 0,
        }

    palette = sorted({int(g[r, c]) for r, c in pts})
    traced_orders = [_trace_path(end, pset) for end in ends]
    traced_colors = [[int(g[r, c]) for r, c in order] for order in traced_orders]
    traced_segments = [_segmentize(colors) for colors in traced_colors]
    mismatch_scores = [_cycle_mismatch_count(segments, palette) for segments in traced_segments]

    if mismatch_scores[0] < mismatch_scores[1]:
        good_idx = 0
    elif mismatch_scores[1] < mismatch_scores[0]:
        good_idx = 1
    else:
        good_idx = ends.index(min(ends))

    good_segments = traced_segments[good_idx]
    bad_segments = traced_segments[1 - good_idx]
    seg_lengths = [seg_len for seg_len, _color in good_segments]
    grouped_colors: dict[int, set[int]] = {}
    for seg_len, color in good_segments:
        grouped_colors.setdefault(int(seg_len), set()).add(int(color))

    cross_color_tie_lengths = sum(
        1
        for seg_len, colors in grouped_colors.items()
        if len(colors) >= 2 and seg_lengths.count(seg_len) >= 2
    )

    good_sorted = _flatten_segments(_reorder_segments(good_segments))
    bad_sorted = _flatten_segments(_reorder_segments(bad_segments))
    top_left_idx = ends.index(min(ends))

    return {
        "segment_count": len(good_segments),
        "distinct_lengths": len(set(seg_lengths)),
        "length_span": max(seg_lengths) - min(seg_lengths),
        "cross_color_tie_lengths": cross_color_tie_lengths,
        "sort_changes_sequence": int(good_sorted != traced_colors[good_idx]),
        "direction_margin": abs(mismatch_scores[0] - mismatch_scores[1]),
        "direction_changes_sorted_output": int(good_sorted != bad_sorted),
        "top_left_is_wrong_start": int(good_idx != top_left_idx),
    }


def main(grid: np.ndarray) -> np.ndarray:
    colors, segments, _meta = _choose_direction_and_segments(np.array(grid, dtype=int))
    if not colors:
        return np.zeros((0, 1), dtype=int)
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
        for ddr, ddc in DIRS4:
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


def _build_cycle_segment_colors(palette: list[int], num_segments: int) -> list[int]:
    return [int(palette[i % len(palette)]) for i in range(num_segments)]


def _is_visible_variant(segments: list[tuple[int, int]], palette: list[int]) -> bool:
    colors = _flatten_segments(segments)
    reversed_segments = list(reversed(segments))
    seg_lengths = [seg_len for seg_len, _color in segments]
    distinct_lengths = len(set(seg_lengths))
    length_span = max(seg_lengths) - min(seg_lengths)
    grouped_colors: dict[int, set[int]] = {}
    for seg_len, color in segments:
        grouped_colors.setdefault(int(seg_len), set()).add(int(color))

    cross_color_tie_lengths = sum(
        1
        for seg_len, colors_at_len in grouped_colors.items()
        if len(colors_at_len) >= 2 and seg_lengths.count(seg_len) >= 2
    )

    forward_mismatch = _cycle_mismatch_count(segments, palette)
    reverse_mismatch = _cycle_mismatch_count(reversed_segments, palette)
    forward_sorted = _flatten_segments(_reorder_segments(segments))
    reverse_sorted = _flatten_segments(_reorder_segments(reversed_segments))

    return (
        len(segments) >= 6
        and len(segments) >= len(palette) + 2
        and distinct_lengths >= 3
        and length_span >= 2
        and cross_color_tie_lengths >= 1
        and forward_mismatch < reverse_mismatch
        and forward_sorted != colors
        and forward_sorted != reverse_sorted
    )


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    palette_size = random.choice([3, 4])
    palette = sorted(random.sample([c for c in range(10) if c != bg], palette_size))

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
    if total < 20 or total > 30:
        return None
    if turns < 7 or turns > 11:
        return None

    if palette_size == 3:
        num_segments = random.choice([6, 7])
    else:
        num_segments = 8

    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=7)
    if seg_lengths is None:
        return None

    seg_colors = _build_cycle_segment_colors(palette, num_segments)
    segments = list(zip(seg_lengths, seg_colors))
    if not _is_visible_variant(segments, palette):
        return None

    endpoints = [path[0], path[-1]]
    top_left = min(endpoints)
    true_start = endpoints[0] if endpoints[1] == top_left else endpoints[1]
    oriented_path = path if path[0] == true_start else list(reversed(path))

    colors = _flatten_segments(segments)
    if len(colors) != len(oriented_path):
        return None

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), color in zip(oriented_path, colors):
        grid[r, c] = color

    expected_out = np.array(_flatten_segments(_reorder_segments(segments)), dtype=int).reshape((-1, 1))
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    metrics = variant_metrics(grid)
    if (
        metrics["top_left_is_wrong_start"] != 1
        or metrics["direction_margin"] <= 0
        or metrics["sort_changes_sequence"] != 1
        or metrics["direction_changes_sorted_output"] != 1
        or metrics["cross_color_tie_lengths"] < 1
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
