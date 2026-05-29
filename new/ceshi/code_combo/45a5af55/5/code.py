# Combo variant of 45a5af55 (index 5)
# combo_spec: D2/2+D11/0
# Name: 多组矩形框架并排

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

BG = 0
GAP = 1
MAX_SIDE = 30
ASPECT_OPTIONS: tuple[tuple[int, int], ...] = ((2, 1), (3, 2), (4, 3))
GROUP_SHAPES: tuple[tuple[tuple[int, int], int], ...] = (
    ((2, 1), 2),
    ((2, 1), 3),
    ((2, 1), 4),
    ((2, 1), 5),
    ((3, 2), 1),
    ((3, 2), 2),
    ((4, 3), 1),
)

# Design analysis for the fused D2/2 + D11/0 variant
# a) Concrete behavior and trace:
#    The original task reads one uninterrupted stack of full-width color stripes,
#    bottom-to-top, and turns them into one square nesting. This fusion first cuts
#    the input into several row groups separated by all-background rows. Inside each
#    group, the colored rows occupy only one constant-width horizontal segment with
#    background margins on both sides; that segment's bounding-box width and the
#    group's height define a reduced aspect ratio. Example:
#    - Group A segment is 8x4 with bottom-to-top stripe heights [2, 1, 1], so its
#      reduced aspect is 2:1. Original logic would make a 2x2 center, then square
#      rings of thickness 1 and 1. Fused logic makes a 2x4 center, then wraps with
#      rectangular rings of pads (1 row, 2 cols) twice, yielding a 6x12 rectangle.
#    - Group B segment is 6x4 with heights [3, 1], so its aspect is 3:2. The center
#      becomes 6x9 and the outer ring adds (2 rows, 3 cols) on each side.
#    The final output is not one shape but `[frame(A)] 0 [frame(B)] ...`, laid out
#    left-to-right in the same order as the groups appear from top to bottom.
# b) Complexity calibration:
#    The group count is limited to 2-3 so many:many behavior is always visible
#    without exhausting the 30x30 canvas. Aspect ratios are drawn from a small set
#    {(2:1), (3:2), (4:3)} so the rectangle rule is learnable rather than noisy.
#    Using centered width segments makes the ratio cue visually obvious in each
#    group, and requiring at least two distinct aspects per sample prevents the task
#    from collapsing into "just repeat the same subproblem twice".
# c) Implicit-assumption audit:
#    - The original `while r >= 0` assumed every row belongs to one stack; false
#      after introducing separator rows, so `main()` must split into groups first.
#    - The original `col = g[r, 0]` assumed the active stripe touches column 0;
#      false after adding horizontal background margins, so each group must be
#      cropped to its non-background bounding box before parsing runs.
#    - The original `np.full((h0, h0), ...)` assumed a square center; false after
#      D2/2, so the center must use `(h0 * rh, h0 * rw)`.
#    - The original `new_n = cur.shape[0] + 2 * h` assumed equal expansion on both
#      axes; false after D2/2, so vertical and horizontal padding differ.
#    - The original single `cur` assumed one final object; false after D11/0, so
#      `main()` must accumulate multiple frames and then pack them safely.
# d) Placement legality:
#    The generator renders each group as non-overlapping centered stripe segments
#    inside its own row block, with separator rows of pure background between
#    groups. The output canvas is allocated from the exact frame sizes, with one
#    background column between frames and bottom alignment, so no overlap occurs.
# e) Visual observability:
#    A human sees multiple stripe stacks separated by blank rows and can also see
#    that different groups have different segment widths. In the outputs, multiple
#    independently nested rectangles appear side by side. Because at least two
#    groups use different reduced aspects, the "rectangular not square" rule is
#    supported by repeated evidence rather than a one-off coincidence.
# f) Visual cleanliness:
#    The input uses only solid stripes plus background margins and blank separator
#    rows. The output uses only clean rectangular frames and fixed one-column gaps,
#    so no region has competing annotations or partial overlaps.
# g) Color inference:
#    Colors keep their original meaning: each stripe color maps directly to the
#    corresponding center/fame layer of its own group. No extra color remapping is
#    introduced, so the only new reasoning burden is grouping plus aspect handling.
# 9) New solver assumptions checked against generator:
#    - Each row group must contain exactly one contiguous non-background segment of
#      constant width across the group. The generator centers one such segment per
#      group and never varies the left/right bounds inside the group.
#    - The group's bounding-box width and height must reduce to one of the intended
#      non-square aspect ratios. The generator samples only from `GROUP_SHAPES`.
#    - Output dimensions for every group and for the packed whole must stay within
#      30x30. The generator rejects attempts that violate those bounds.
#    - Adjacent stripe colors inside a group must differ so bottom-up run parsing is
#      unambiguous. The generator samples colors with no immediate repetition.


def _parse_groups(grid: np.ndarray) -> list[np.ndarray]:
    """Split the input into row groups separated by all-background rows."""
    groups: list[np.ndarray] = []
    h = int(grid.shape[0])
    r = 0
    while r < h:
        while r < h and not bool((grid[r] != BG).any()):
            r += 1
        if r >= h:
            break
        start = r
        while r < h and bool((grid[r] != BG).any()):
            r += 1
        groups.append(grid[start:r])
    return groups


def _extract_segment(block: np.ndarray) -> np.ndarray:
    """Crop one group to its unique non-background width segment."""
    mask = block != BG
    assert bool(mask.any())
    cols = np.flatnonzero(mask.any(axis=0))
    left = int(cols[0])
    right = int(cols[-1])

    if left > 0:
        assert not bool((block[:, :left] != BG).any())
    if right + 1 < block.shape[1]:
        assert not bool((block[:, right + 1 :] != BG).any())

    segment = block[:, left : right + 1]
    assert bool((segment != BG).all())
    return segment


def _parse_group(block: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Parse one group into bottom-up runs plus its reduced width:height aspect."""
    segment = _extract_segment(block)
    h_total, w_total = map(int, segment.shape)

    for row in segment:
        col = int(row[0])
        assert col != BG
        assert bool((row == col).all())

    runs: list[tuple[int, int]] = []
    r = h_total - 1
    while r >= 0:
        col = int(segment[r, 0])
        run_h = 1
        rr = r - 1
        while rr >= 0 and int(segment[rr, 0]) == col:
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr

    scale = math.gcd(w_total, h_total)
    aspect = (w_total // scale, h_total // scale)
    return runs, aspect


def _build_rect_frame(runs: list[tuple[int, int]], aspect: tuple[int, int]) -> np.ndarray:
    """Build one nested rectangular frame stack from bottom-up stripe runs."""
    assert runs
    rw, rh = aspect
    c0, h0 = runs[0]
    cur = np.full((h0 * rh, h0 * rw), c0, dtype=int)

    for col, h in runs[1:]:
        pad_y = h * rh
        pad_x = h * rw
        nxt = np.full((cur.shape[0] + 2 * pad_y, cur.shape[1] + 2 * pad_x), col, dtype=int)
        nxt[pad_y : pad_y + cur.shape[0], pad_x : pad_x + cur.shape[1]] = cur
        cur = nxt

    return cur


def _pack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Place independent frames left-to-right with a background gap."""
    assert frames
    out_h = max(int(frame.shape[0]) for frame in frames)
    out_w = sum(int(frame.shape[1]) for frame in frames) + GAP * (len(frames) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for frame in frames:
        fh, fw = map(int, frame.shape)
        out[out_h - fh :, c : c + fw] = frame
        c += fw + GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Split the input into row groups separated by background rows.
    - Each group contains one constant-width stack of horizontal color stripes.
    - The group's reduced segment width:height aspect controls rectangular nesting.
    - Pack all resulting rectangular frames side by side, bottom aligned.
    """
    g = np.array(grid, dtype=int)
    groups = _parse_groups(g)
    assert groups

    frames = []
    for block in groups:
        runs, aspect = _parse_group(block)
        frames.append(_build_rect_frame(runs, aspect))
    return _pack_frames(frames)


@lru_cache(maxsize=None)
def _bounded_compositions(total: int, parts: int, max_part: int) -> tuple[tuple[int, ...], ...]:
    """All positive `parts`-tuples summing to `total`, with each part <= max_part."""
    if parts == 0:
        return ((),) if total == 0 else ()
    if total < parts or total > parts * max_part:
        return ()

    out: list[tuple[int, ...]] = []
    max_first = min(max_part, total - (parts - 1))
    for first in range(1, max_first + 1):
        for rest in _bounded_compositions(total - first, parts - 1, max_part):
            out.append((first,) + rest)
    return tuple(out)


def _sample_colors(n: int) -> list[int]:
    colors: list[int] = []
    prev = None
    for _ in range(n):
        c = int(np.random.randint(1, 10))
        while c == prev:
            c = int(np.random.randint(1, 10))
        colors.append(c)
        prev = c
    return colors


def _sample_runs_for_group(total_h: int, aspect: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Sample bottom-up stripe runs whose output still fits the canvas."""
    rw, rh = aspect
    max_scale = min(MAX_SIDE // rw, MAX_SIDE // rh)
    candidates: list[tuple[int, tuple[int, ...]]] = []

    for n_runs in range(2, min(4, total_h) + 1):
        outer_parts = n_runs - 1
        max_outer_sum = min(total_h - 1, 3 * outer_parts)
        for outer_sum in range(outer_parts, max_outer_sum + 1):
            bottom_h = total_h - outer_sum
            output_scale = bottom_h + 2 * outer_sum
            if output_scale > max_scale:
                continue
            for outer in _bounded_compositions(outer_sum, outer_parts, 3):
                candidates.append((bottom_h, outer))

    if not candidates:
        return None

    bottom_h, outer = candidates[int(np.random.randint(len(candidates)))]
    heights = (bottom_h,) + outer
    colors = _sample_colors(len(heights))
    return list(zip(colors, heights))


def _render_group_input(
    runs: list[tuple[int, int]],
    segment_w: int,
    canvas_w: int,
) -> np.ndarray:
    """Render one stripe group centered inside a wider background canvas."""
    total_h = sum(height for _color, height in runs)
    block = np.full((total_h, canvas_w), BG, dtype=int)
    left = (canvas_w - segment_w) // 2
    right = left + segment_w

    r = 0
    for color, height in reversed(runs):
        block[r : r + height, left:right] = color
        r += height
    return block


def _gen_attempt() -> np.ndarray | None:
    n_groups = int(np.random.randint(2, 4))
    chosen_aspects = [
        ASPECT_OPTIONS[int(np.random.randint(len(ASPECT_OPTIONS)))]
        for _ in range(n_groups)
    ]
    if len(set(chosen_aspects)) < 2:
        return None

    group_specs: list[tuple[list[tuple[int, int]], tuple[int, int], int, int]] = []
    input_heights: list[int] = []
    input_widths: list[int] = []
    frames: list[np.ndarray] = []

    for aspect in chosen_aspects:
        shape_candidates = [shape for shape in GROUP_SHAPES if shape[0] == aspect]
        aspect_choice, scale = shape_candidates[int(np.random.randint(len(shape_candidates)))]
        rw, rh = aspect_choice
        total_h = rh * scale
        segment_w = rw * scale
        runs = _sample_runs_for_group(total_h, aspect_choice)
        if runs is None:
            return None

        frame = _build_rect_frame(runs, aspect_choice)
        group_specs.append((runs, aspect_choice, total_h, segment_w))
        input_heights.append(total_h)
        input_widths.append(segment_w)
        frames.append(frame)

    sep_heights = [int(np.random.randint(1, 3)) for _ in range(n_groups - 1)]
    canvas_w = max(input_widths) + 2 * int(np.random.randint(1, 4))
    total_input_h = sum(input_heights) + sum(sep_heights)
    total_output_h = max(int(frame.shape[0]) for frame in frames)
    total_output_w = sum(int(frame.shape[1]) for frame in frames) + GAP * (n_groups - 1)

    if canvas_w > MAX_SIDE or total_input_h > MAX_SIDE:
        return None
    if total_output_h > MAX_SIDE or total_output_w > MAX_SIDE:
        return None

    rendered_groups = [
        _render_group_input(runs, segment_w, canvas_w)
        for runs, _aspect, _total_h, segment_w in group_specs
    ]

    parts: list[np.ndarray] = []
    for idx, block in enumerate(rendered_groups):
        parts.append(block)
        if idx != len(rendered_groups) - 1:
            parts.append(np.full((sep_heights[idx], canvas_w), BG, dtype=int))
    grid = np.vstack(parts)

    # Validate that the intended fused behavior is present and not collapsed.
    parsed = [_parse_group(block) for block in _parse_groups(grid)]
    aspects_seen = [aspect for _runs, aspect in parsed]
    if len(set(aspects_seen)) < 2:
        return None
    if any(rw == rh for rw, rh in aspects_seen):
        return None

    expected = _pack_frames([_build_rect_frame(runs, aspect) for runs, aspect in parsed])
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise RuntimeError("generator produced an invalid sample")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
