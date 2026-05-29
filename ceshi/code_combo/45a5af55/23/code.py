# Combo variant of 45a5af55 (index 23)
# combo_spec: D2/2+D7/2+D11/0
# Name: 多组矩形定宽嵌套

from __future__ import annotations

import math

import numpy as np

BG_GAP = 1
MAX_SIDE = 30
ASPECT_OPTIONS: tuple[tuple[int, int], ...] = ((2, 1), (3, 2))


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    Original task: one full-width stripe stack fills the whole input. Reading
#    bottom->top heights [1(red), 2(blue), 1(green)] yields a 1x1 red center,
#    then a 5x5 blue-wrapped square, then a 7x7 green-wrapped square.
#    Combo behavior: the input now contains multiple stripe groups separated by
#    pure background rows and surrounded by background margins. Suppose the full
#    input is 8x16, so its reduced aspect is 1:2 in height:width, i.e. output
#    rectangles use width:height = 2:1. If the top group has heights [1, 1] and
#    the lower group has heights [2, 1], then:
#    - original single-stack logic would fail conceptually because background rows
#      are separators, not real stripes;
#    - fused logic first splits the input into two groups;
#    - group A becomes a 3x6 rectangular frame, group B becomes a 4x8 one;
#    - the two rectangles are packed left-to-right with a 1-cell gap, bottom
#      aligned within their shared strip;
#    - that packed strip is centered inside a 16x16 background canvas because the
#      final output side is the input width.
#    Differences from the original are therefore simultaneous and coupled:
#    grouping changes the object count, global input aspect changes each group's
#    geometry, and input width changes the final canvas instead of stripe heights.
#
# b) Complexity calibration
#    - Only 2-3 groups are used so many:many behavior is obvious without making
#      the output too crowded.
#    - Only two reduced aspect ratios are used: 2:1 and 3:2. Both are visibly
#      non-square and easy to compare across examples.
#    - Stripe heights stay in 1-3 and group run counts stay in 2-4, preserving
#      the original nesting idea while keeping output sizes ARC-friendly.
#    - The generator enforces visible outer background margins in the output, so
#      the width-determined canvas rule does not collapse back to tight packing.
#
# c) Implicit-assumption audit
#    The original solver relied on several assumptions that no longer hold:
#    - `while r >= 0` over the whole grid assumed one uninterrupted stripe stack.
#      Here background rows split the input into multiple subproblems.
#    - `g[r, 0]` assumed every row belongs to a stripe. Here corners and separator
#      rows are background, so each group must first be cropped to its non-bg box.
#    - `np.full((h0, h0), ...)` assumed a square center. The combo must instead use
#      `(h0 * rh, h0 * rw)` from the reduced whole-grid aspect `(rw, rh)`.
#    - `new_n = cur.shape[0] + 2 * h` assumed equal expansion on both axes. The
#      combo expands by `(2*h*rh, 2*h*rw)`.
#    - `return cur` assumed output size comes only from nesting thicknesses. The
#      combo instead packs all group results into a `W x W` background canvas.
#    New assumptions introduced by the combo:
#    - the four corners carry the background color;
#    - each group occupies one solid rectangular stripe block after cropping;
#    - all groups share one global aspect ratio from the whole input size;
#    - the horizontally packed frames fit inside the width-based output canvas.
#    The generator enforces each of these assumptions directly.
#
# d) Placement legality
#    Input placement:
#    - each group is rendered as a single monochrome-row stripe block;
#    - groups are separated by full background rows, so they never touch;
#    - every block stays inside shared side margins, leaving the corners background.
#    Output placement:
#    - each group frame is generated independently;
#    - frames are laid out left-to-right with a fixed 1-cell background gap;
#    - all frames are bottom-aligned inside the packed band and the packed band is
#      centered inside the square output canvas, so nothing overlaps or overflows.
#
# e) Visual observability
#    A human can infer the rule from a few examples because three cues are visible:
#    - the input visibly contains multiple stripe stacks, not one;
#    - output objects are visibly rectangular rather than square, and all share the
#      same width:height ratio tied to the full input shape;
#    - the output sits inside a larger square background canvas whose side matches
#      the input width, leaving an obvious border around the packed frames.
#
# f) Visual cleanliness
#    The input uses only clean stripe rectangles plus background margins. The output
#    uses only clean nested rectangles plus one background gap between objects and
#    a uniform outer border. There are no overlapping annotations or mixed signals.
#
# g) Change inferability
#    No color remapping is introduced. Colors keep their original role: each stripe
#    color becomes the corresponding rectangular layer color in the matching group.
#    Because geometry and grouping carry the new information, the user does not need
#    to infer any hidden color-dependent rule.


def _infer_bg(g: np.ndarray) -> int:
    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    return max(set(corners), key=corners.count)


def _parse_groups(g: np.ndarray, bg: int) -> list[np.ndarray]:
    """Split the input into contiguous non-background row bands."""
    groups: list[np.ndarray] = []
    h = int(g.shape[0])
    r = 0
    while r < h:
        while r < h and bool((g[r] == int(bg)).all()):
            r += 1
        if r >= h:
            break
        start = r
        while r < h and not bool((g[r] == int(bg)).all()):
            r += 1
        groups.append(g[start:r])
    return groups


def _parse_runs(block: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Crop one group to its solid stripe rectangle and read runs bottom->top."""
    non_bg = block != int(bg)
    assert bool(non_bg.any())

    rows, cols = np.where(non_bg)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    crop = block[r0 : r1 + 1, c0 : c1 + 1]
    assert not bool((crop == int(bg)).any())

    runs: list[tuple[int, int]] = []
    r = int(crop.shape[0] - 1)
    while r >= 0:
        row = crop[r]
        col = int(row[0])
        assert col != int(bg)
        assert bool((row == col).all())
        run_h = 1
        rr = r - 1
        while rr >= 0 and bool((crop[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr

    assert runs
    return runs


def _build_rect_frame(runs: list[tuple[int, int]], aspect: tuple[int, int]) -> np.ndarray:
    """Build one group's nested rectangular frame using the global input aspect."""
    rw, rh = aspect
    c0, h0 = runs[0]
    cur = np.full((int(h0 * rh), int(h0 * rw)), int(c0), dtype=int)

    for col, h in runs[1:]:
        pad_y = int(h * rh)
        pad_x = int(h * rw)
        nxt = np.full((cur.shape[0] + 2 * pad_y, cur.shape[1] + 2 * pad_x), int(col), dtype=int)
        nxt[pad_y : pad_y + cur.shape[0], pad_x : pad_x + cur.shape[1]] = cur
        cur = nxt

    return cur


def _pack_on_fixed_canvas(frames: list[np.ndarray], canvas_side: int, bg: int) -> np.ndarray:
    """Pack frames left-to-right, bottom-aligned within a centered band on a square canvas."""
    assert frames
    pack_h = max(int(frame.shape[0]) for frame in frames)
    pack_w = sum(int(frame.shape[1]) for frame in frames) + BG_GAP * (len(frames) - 1)
    assert pack_h <= int(canvas_side)
    assert pack_w <= int(canvas_side)

    out = np.full((int(canvas_side), int(canvas_side)), int(bg), dtype=int)
    top = int((canvas_side - pack_h) // 2)
    left = int((canvas_side - pack_w) // 2)
    c = left
    for frame in frames:
        fh, fw = frame.shape
        r0 = top + (pack_h - int(fh))
        out[r0 : r0 + fh, c : c + fw] = frame
        c += int(fw) + BG_GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Split the background-separated stripe groups.
    - Use the reduced whole-input aspect ratio to turn each group into a nested
      rectangular frame.
    - Pack the resulting frames left-to-right and center the packed strip inside
      a square background canvas whose side equals the input width.
    """
    g = np.array(grid, dtype=int)
    h_total, w_total = map(int, g.shape)
    bg = _infer_bg(g)
    groups = _parse_groups(g, bg)
    assert groups

    scale = math.gcd(h_total, w_total)
    aspect = (w_total // scale, h_total // scale)
    frames = [_build_rect_frame(_parse_runs(block, bg), aspect) for block in groups]
    return _pack_on_fixed_canvas(frames, w_total, bg)


def _sample_bounded_partition(total: int, n_parts: int, min_part: int, max_part: int) -> list[int] | None:
    if total < n_parts * min_part or total > n_parts * max_part:
        return None

    parts = [int(min_part) for _ in range(n_parts)]
    capacity = [int(max_part - min_part) for _ in range(n_parts)]
    rem = int(total - n_parts * min_part)
    while rem > 0:
        choices = [idx for idx, cap in enumerate(capacity) if cap > 0]
        if not choices:
            return None
        idx = int(choices[int(np.random.randint(len(choices)))])
        parts[idx] += 1
        capacity[idx] -= 1
        rem -= 1

    order = np.random.permutation(n_parts)
    return [int(parts[idx]) for idx in order]


def _sample_runs_for_block(block_h: int, bg: int) -> list[tuple[int, int]] | None:
    feasible = [n for n in range(2, min(4, int(block_h)) + 1) if n <= block_h <= 3 * n]
    if not feasible:
        return None

    n_runs = int(feasible[int(np.random.randint(len(feasible)))])
    heights = _sample_bounded_partition(int(block_h), n_runs, 1, 3)
    if heights is None:
        return None

    colors: list[int] = []
    for _ in range(n_runs):
        c = int(np.random.randint(0, 10))
        while c == int(bg) or (colors and c == colors[-1]):
            c = int(np.random.randint(0, 10))
        colors.append(c)

    return list(zip(colors, heights))


def _render_group_input(runs: list[tuple[int, int]], width: int) -> np.ndarray:
    stripes = [np.full((int(h), int(width)), int(col), dtype=int) for col, h in reversed(runs)]
    return np.vstack(stripes)


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.randint(0, 10))
    rw, rh = ASPECT_OPTIONS[int(np.random.randint(len(ASPECT_OPTIONS)))]

    feasible_scales = [s for s in range(4, (MAX_SIDE // rw) + 1) if s * rh <= MAX_SIDE]
    if not feasible_scales:
        return None
    scale = int(feasible_scales[int(np.random.randint(len(feasible_scales)))])

    h_total = int(scale * rh)
    w_total = int(scale * rw)

    feasible_group_counts = [n for n in (2, 3) if 2 * n + (n - 1) + 2 <= h_total]
    if not feasible_group_counts:
        return None
    n_groups = int(feasible_group_counts[int(np.random.randint(len(feasible_group_counts)))])

    side_margin_max = max(1, min(3, (w_total - 2) // 2))
    side_margin = int(np.random.randint(1, side_margin_max + 1))
    stripe_w = int(w_total - 2 * side_margin)
    if stripe_w <= 0:
        return None

    group_heights = [int(np.random.randint(2, 5)) for _ in range(n_groups)]
    remaining_bg = int(h_total - sum(group_heights))
    n_bg_slots = int(n_groups + 1)  # top, separators, bottom
    bg_slots = _sample_bounded_partition(remaining_bg, n_bg_slots, 1, remaining_bg)
    if bg_slots is None:
        return None
    top_pad = int(bg_slots[0])
    bottom_pad = int(bg_slots[-1])
    sep_heights = [int(x) for x in bg_slots[1:-1]]

    groups: list[list[tuple[int, int]]] = []
    frames: list[np.ndarray] = []
    for block_h in group_heights:
        runs = _sample_runs_for_block(int(block_h), bg)
        if runs is None:
            return None
        frame = _build_rect_frame(runs, (rw, rh))
        groups.append(runs)
        frames.append(frame)

    pack_h = max(int(frame.shape[0]) for frame in frames)
    pack_w = sum(int(frame.shape[1]) for frame in frames) + BG_GAP * (len(frames) - 1)

    # Keep every fused dimension visibly active.
    if any(int(frame.shape[0]) == int(frame.shape[1]) for frame in frames):
        return None
    if pack_w > w_total - 2 or pack_h > w_total - 2:
        return None
    if len(set(tuple(map(int, frame.shape)) for frame in frames)) == 1 and np.random.rand() < 0.5:
        return None

    grid = np.full((h_total, w_total), int(bg), dtype=int)
    r = int(top_pad)
    for idx, runs in enumerate(groups):
        block = _render_group_input(runs, stripe_w)
        bh = int(block.shape[0])
        grid[r : r + bh, side_margin : side_margin + stripe_w] = block
        r += bh
        if idx != n_groups - 1:
            r += int(sep_heights[idx])

    if r + bottom_pad != h_total:
        return None

    expected = _pack_on_fixed_canvas(frames, w_total, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise RuntimeError("generator produced an invalid sample")

    # Reject cases that collapse to a simpler variant on the same input.
    tight = np.full((pack_h, pack_w), int(bg), dtype=int)
    c = 0
    for frame in frames:
        fh, fw = frame.shape
        tight[pack_h - fh :, c : c + fw] = frame
        c += int(fw) + BG_GAP
    if np.array_equal(expected, tight):
        return None

    square_frames = []
    for runs in groups:
        c0, h0 = runs[0]
        cur = np.full((int(h0), int(h0)), int(c0), dtype=int)
        for col, h in runs[1:]:
            nxt = np.full((cur.shape[0] + 2 * int(h), cur.shape[1] + 2 * int(h)), int(col), dtype=int)
            nxt[int(h) : int(h) + cur.shape[0], int(h) : int(h) + cur.shape[1]] = cur
            cur = nxt
        square_frames.append(cur)
    square_canvas = _pack_on_fixed_canvas(square_frames, w_total, bg)
    if np.array_equal(expected, square_canvas):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")
