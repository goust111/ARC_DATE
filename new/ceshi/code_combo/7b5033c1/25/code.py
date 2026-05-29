# Combo variant of 7b5033c1 (index 25)
# combo_spec: D1/2+D2/1+D4/2
# Name: RLE+段长度起点+回文色段
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    The input is still one simple 4-connected snake with exactly two endpoints. The change is
#    now joint rather than additive:
#    - D4 makes the path-color segments palindromic, e.g. [1x4, 2x3, 3x2, 5x4, 3x2, 2x3, 1x2].
#      The color column of the RLE output is therefore a palindrome: [1,2,3,5,3,2,1].
#    - D2 chooses the traversal start from the endpoint whose terminal color segment is shorter.
#      In the example above, the left end has length 4 and the right end has length 2, so
#      traversal starts from the right even if the left endpoint is more top-left.
#    - D1 replaces pixelwise output with RLE rows [segment_color, length_code], using the fixed
#      map 2->4, 3->6, 4->8. The example above outputs
#      [[1,4],[2,6],[3,4],[5,8],[3,4],[2,6],[1,8]] when read from the shorter endpoint.
#    Relative to the original task, geometry detection is unchanged, but direction selection and
#    the output representation both change. Because the colors are palindromic, reversing the path
#    preserves the first column and only reverses the length-code pattern, which is exactly the
#    intended interaction between D2 and D4.
# b) Complexity calibration:
#    The generator keeps the visible degrees of freedom small:
#    - run lengths are restricted to {2,3,4}, so the length-code column only uses {4,6,8};
#    - every sample contains all three length values, so the right column has repeated evidence;
#    - the segment count is fixed to 7. This is deliberate: an even-length palindrome with
#      adjacent-distinct segment colors would force the two middle colors to be equal and adjacent,
#      so odd length is the clean visible choice;
#    - the shorter endpoint is forced to be opposite the old top-left endpoint, and the endpoint
#      lengths differ, so the fused task does not silently collapse to the original direction rule.
# c) Implicit assumption audit for main():
#    - Non-background cells must still be a single simple chain; endpoint detection depends on it.
#    - Tracing from either endpoint must visit all path cells exactly once; the generator enforces
#      no branches, no overlaps, and no extra 4-neighbors.
#    - Endpoint run lengths are read from the first and last same-color streaks of a traced path.
#      This assumes colors are assigned in contiguous segments along path order; the generator does
#      exactly that.
#    - RLE encoding assumes every run length is in {2,3,4}; this is new compared with the source
#      task, so the generator must never emit length 1 or >4 segments.
#    - Tie handling for equal endpoint lengths is not part of the intended samples, but main()
#      keeps the original top-left tie-break so the rule is deterministic on edge cases.
# d) Placement legality:
#    The path is grown one straight mini-segment at a time inside a 16x16 board. Each new cell is
#    checked for bounds, overlap, and forbidden side contact with older path cells other than its
#    predecessor. That preserves a single non-self-touching chain that the solver can trace.
# e) Visual observability:
#    Across a few examples, a human sees a snake whose colors mirror around the center, while the
#    output is a much shorter 7x2 table. The first output column is visibly palindromic, and the
#    second column uses only three stable code colors. Comparing multiple samples makes it possible
#    to infer both "compress by color runs" and "start from the shorter terminal segment".
# f) Visual cleanliness:
#    No extra markers are added to the input grid, so there is only one object class: the snake.
#    The output has two columns with a clear role split: left for segment color, right for length
#    code. Excluding the code colors from the input palette keeps those two roles visually distinct.
# g) Color-change inferability:
#    The transformation does not invent arbitrary recolorings. Input path colors come from only
#    four non-background colors arranged as a palindrome, while the output appends one of three
#    fixed code colors determined solely by segment length. That limited palette and repeated
#    structure make the color behavior inferable from 2-4 examples.

from __future__ import annotations

from itertools import product
from pathlib import Path
import random
import sys

import numpy as np

for helper_dir in (
    Path("/home/lixinyu/snap/new/arc2_solver_select"),
    Path("/home/lixinyu/snap/new/solvers"),
):
    if helper_dir.exists():
        sys.path.insert(0, str(helper_dir))

from arc_utils import get_bg


LENGTH_TO_CODE = {2: 4, 3: 6, 4: 8}
ALLOWED_LENGTHS = tuple(LENGTH_TO_CODE)
LENGTH_CODE_SET = set(LENGTH_TO_CODE.values())
NUM_SEGMENTS = 7


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trace_path(pset: set[tuple[int, int]], start: tuple[int, int], total: int) -> list[tuple[int, int]]:
    """Trace the unique chain from one endpoint to the other."""
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start

    for _ in range(total):
        order.append(cur)
        nbrs = [q for q in _neighbors4(*cur) if q in pset and q != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1, "Expected a simple chain with no branches"
        prev, cur = cur, nbrs[0]

    assert len(order) == total, "Trace must visit every path cell exactly once"
    return order


def _get_path_order_and_ends(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return [], []

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2, "Input must contain exactly two endpoints"
    order = _trace_path(pset, ends[0], len(pts))
    return order, ends


def _endpoint_run_lengths(colors: list[int]) -> tuple[int, int]:
    """Return the first and last same-color run lengths along a traced path."""
    prefix = 1
    while prefix < len(colors) and colors[prefix] == colors[0]:
        prefix += 1

    suffix = 1
    while suffix < len(colors) and colors[-1 - suffix] == colors[-1]:
        suffix += 1

    return prefix, suffix


def _orient_colors_from_shorter_endpoint(
    order: list[tuple[int, int]],
    ends: list[tuple[int, int]],
    g: np.ndarray,
) -> list[int]:
    """Orient the traced colors so traversal starts from the shorter endpoint segment."""
    colors = [int(g[r, c]) for r, c in order]
    first_len, last_len = _endpoint_run_lengths(colors)

    # New direction rule:
    # - shorter terminal segment wins;
    # - if tied, keep the old top-left endpoint rule for determinism.
    if last_len < first_len or (last_len == first_len and order[0] != min(ends)):
        colors.reverse()
    return colors


def _encode_runs(colors: list[int]) -> np.ndarray:
    """Encode path colors as rows [segment_color, length_code_color]."""
    if not colors:
        return np.zeros((0, 2), dtype=int)

    rows: list[list[int]] = []
    cur_color = int(colors[0])
    cur_len = 1
    for color in colors[1:]:
        color = int(color)
        if color == cur_color:
            cur_len += 1
        else:
            assert cur_len in LENGTH_TO_CODE
            rows.append([cur_color, LENGTH_TO_CODE[cur_len]])
            cur_color = color
            cur_len = 1

    assert cur_len in LENGTH_TO_CODE
    rows.append([cur_color, LENGTH_TO_CODE[cur_len]])
    return np.array(rows, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the snake, choose the endpoint with the shorter terminal color segment, then emit RLE.

    Output format:
    - one row per color segment along the chosen traversal direction;
    - column 0: segment color;
    - column 1: length code color using 2->4, 3->6, 4->8.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    order, ends = _get_path_order_and_ends(g, bg)
    if not order:
        return np.zeros((0, 2), dtype=int)

    colors = _orient_colors_from_shorter_endpoint(order, ends, g)
    return _encode_runs(colors)


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """Try to extend the path by count steps; stop before any illegal placement."""
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= h - 1 or nc < 1 or nc >= w - 1:
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


def _generate_path(target_len: int, h: int, w: int) -> list[tuple[int, int]] | None:
    """Build a legal snake of the requested length."""
    for _ in range(150):
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

        if len(path) == target_len and 7 <= turns <= 11:
            return path
    return None


def _build_palindrome_colors(palette: list[int]) -> list[int]:
    """Use four visible colors arranged as a 7-segment palindrome."""
    a, b, c, d = palette
    return [a, b, c, d, c, b, a]


def _sample_segment_lengths(total: int, first_is_top_left: bool) -> list[int] | None:
    """
    Pick 7 segment lengths from {2,3,4}.

    Requirements:
    - all three lengths appear;
    - the shorter endpoint is opposite the old top-left endpoint;
    - endpoint lengths differ, so the new direction rule changes the output order.
    """
    candidates: list[tuple[int, ...]] = []
    for lengths in product(ALLOWED_LENGTHS, repeat=NUM_SEGMENTS):
        if sum(lengths) != total:
            continue
        if set(lengths) != set(ALLOWED_LENGTHS):
            continue
        if first_is_top_left:
            if not (lengths[0] > lengths[-1]):
                continue
        else:
            if not (lengths[0] < lengths[-1]):
                continue
        candidates.append(lengths)

    if not candidates:
        return None
    return list(random.choice(candidates))


def _expand_colors(seg_colors: list[int], seg_lengths: list[int]) -> list[int]:
    colors: list[int] = []
    for color, seg_len in zip(seg_colors, seg_lengths):
        colors.extend([color] * seg_len)
    return colors


def _expected_rows(seg_colors: list[int], seg_lengths: list[int]) -> np.ndarray:
    rows = np.array(
        [[color, LENGTH_TO_CODE[seg_len]] for color, seg_len in zip(seg_colors, seg_lengths)],
        dtype=int,
    )
    if seg_lengths[-1] < seg_lengths[0]:
        rows = rows[::-1]
    return rows


def _old_top_left_rows(seg_colors: list[int], seg_lengths: list[int], first_is_top_left: bool) -> np.ndarray:
    rows = np.array(
        [[color, LENGTH_TO_CODE[seg_len]] for color, seg_len in zip(seg_colors, seg_lengths)],
        dtype=int,
    )
    if not first_is_top_left:
        rows = rows[::-1]
    return rows


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    available_for_input = [c for c in range(10) if c not in LENGTH_CODE_SET]
    bg = random.choice(available_for_input)
    palette = random.sample([c for c in available_for_input if c != bg], 4)

    target_len = random.randint(20, 24)
    path = _generate_path(target_len, h, w)
    if path is None:
        return None

    first_is_top_left = path[0] == min(path[0], path[-1])
    seg_lengths = _sample_segment_lengths(target_len, first_is_top_left)
    if seg_lengths is None:
        return None

    seg_colors = _build_palindrome_colors(palette)
    colors = _expand_colors(seg_colors, seg_lengths)
    assert len(colors) == len(path)

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color

    expected_out = _expected_rows(seg_colors, seg_lengths)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    # Generator-side fusion checks:
    # - color rows must be palindromic;
    # - the new direction rule must differ from the old top-left rule on this sample.
    if actual_out[:, 0].tolist() != actual_out[::-1, 0].tolist():
        return None
    if np.array_equal(actual_out, _old_top_left_rows(seg_colors, seg_lengths, first_is_top_left)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
