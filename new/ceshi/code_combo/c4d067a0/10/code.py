# Combo variant of c4d067a0 (index 10)
# combo_spec: D6/2+D8/3
# Name: 分帧输出+从示例学习规则

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task completed missing enlarged blocks inside one grid by using
#    two anchor blocks. Here the input is reframed as three panels:
#    [example legend | example enlarged result | query legend]. The output is two
#    panels: [example enlarged result | query enlarged result]. A solver must first
#    learn the enlargement recipe from the example pair, then apply it to the new
#    legend, and finally emit a multi-frame answer rather than painting in place.
# b) Complexity calibration:
#    The learned recipe varies along three visible axes: block size, inter-block
#    spacing, and frame offset of the enlarged pattern. If these were fixed, the
#    example panel would be redundant and the D8/3 layer would collapse. Their
#    domains are kept small (2-4, 1-3, 0-3) so the rule is still learnable.
# c) Implicit assumptions changed:
#    The original solver assumed one mixed canvas containing both legend pixels and
#    existing blocks. That assumption is invalid now, so `main()` first splits
#    panels, then treats legend-only and block-only panels separately. It also no
#    longer depends on partial anchors; instead it infers the shared rendering spec
#    from the complete example pair.
# d) Placement legality:
#    All frames have equal height, divider columns are isolated colors, legend dots
#    are spaced by 2 so they remain singleton components, and enlarged blocks use
#    gaps >= 1 so they remain distinct square components. The generator chooses a
#    frame size that fits both example and query enlargements under the sampled
#    block size / step / offset.
# e) Visual observability:
#    Humans can directly compare the first two input panels to see how a sparse
#    legend becomes a block mosaic, then apply that learned geometry to the third
#    panel. The second output panel differs visibly from copying because the query
#    legend shape and colors are newly sampled.
# f) Visual cleanliness:
#    Each panel contains only one object family: singleton legend dots or enlarged
#    blocks. No legend overlays sit on top of the block canvas, which keeps the
#    frame-separated variant legible.
# g) Learnability / non-degeneration:
#    Colors are preserved rather than remapped, so the example teaches only the
#    spatial rule. The generator later validates that sampled cases exercise
#    multiple block sizes, gaps, and offsets; otherwise the fusion would regress to
#    a fixed-template renderer.


def _gcd_of_diffs(vals: list[int]) -> int:
    vals = sorted(set(int(v) for v in vals))
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1) if vals[i + 1] != vals[i]]
    if not diffs:
        return 0
    return reduce(gcd, diffs)


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    h, w = g.shape
    bg = get_bg(g)
    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [c for c in solid_cols if int(g[0, c]) != bg]
    assert div_cols, "expected full-height divider columns"
    divider = int(g[0, div_cols[0]])

    cuts = [-1] + div_cols + [w]
    frames: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        assert a < b, "empty frame slice is not allowed"
        frames.append(g[:, a:b])
    return frames, bg, divider


def _join_frames(frames: list[np.ndarray], divider: int) -> np.ndarray:
    assert frames, "must join at least one frame"
    heights = {int(fr.shape[0]) for fr in frames}
    assert len(heights) == 1, "all frames must have the same height"
    h = heights.pop()
    total_w = sum(int(fr.shape[1]) for fr in frames) + len(frames) - 1
    out = np.full((h, total_w), divider, dtype=int)
    c0 = 0
    for i, fr in enumerate(frames):
        w = int(fr.shape[1])
        out[:, c0 : c0 + w] = fr
        c0 += w
        if i < len(frames) - 1:
            c0 += 1
    return out


def _extract_squares(frame: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    bg = get_bg(frame)
    comps = components_mask(frame != bg, connectivity=4)
    squares: list[tuple[int, int, int, int]] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = frame[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            squares.append((int(r0), int(c0), int(hh), col))
    if not squares:
        return 0, []
    size_votes: dict[int, int] = {}
    for _, _, size, _ in squares:
        size_votes[size] = size_votes.get(size, 0) + 1
    block_size = max(size_votes.items(), key=lambda kv: kv[1])[0]
    filtered = [sq for sq in squares if sq[2] == block_size]
    return int(block_size), filtered


def _classify_frame(frame: np.ndarray) -> str:
    bg = get_bg(frame)
    comps = components_mask(frame != bg, connectivity=4)
    if not comps:
        return "empty"

    single_count = 0
    square_count = 0
    bad = 0
    for pts in comps:
        if len(pts) == 1:
            single_count += 1
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        sub = frame[r0 : r1 + 1, c0 : c1 + 1]
        if hh == ww and hh >= 2 and np.all(sub == sub[0, 0]) and sub.size == len(pts):
            square_count += 1
        else:
            bad += 1

    if bad == 0 and single_count > 0 and square_count == 0:
        return "legend"
    if bad == 0 and square_count > 0 and single_count == 0:
        return "blocks"
    return "mixed"


def _parse_legend_frame(frame: np.ndarray) -> list[tuple[int, int, int]]:
    bg = get_bg(frame)
    comps = components_mask(frame != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    for pts in comps:
        assert len(pts) == 1, "legend frame must contain isolated pixels only"
        r, c = pts[0]
        singles.append((int(r), int(c), int(frame[r, c])))
    assert singles, "legend frame must not be empty"

    rows = sorted({r for r, _, _ in singles})
    cols = sorted({c for _, c, _ in singles})
    step_r = _gcd_of_diffs(rows)
    step_c = _gcd_of_diffs(cols)
    step = step_r or step_c
    assert step > 0, "legend lattice step must be recoverable"

    rmin = min(r for r, _, _ in singles)
    cmin = min(c for _, c, _ in singles)
    pts = [((r - rmin) // step, (c - cmin) // step, col) for r, c, col in singles]
    return sorted((int(rr), int(cc), int(col)) for rr, cc, col in pts)


def _parse_block_frame(frame: np.ndarray) -> tuple[int, int, int, int]:
    block_size, squares = _extract_squares(frame)
    assert block_size > 0 and squares, "block frame must contain uniform square blocks"
    tops = [r0 for r0, _, _, _ in squares]
    lefts = [c0 for _, c0, _, _ in squares]
    row_step = _gcd_of_diffs(tops)
    col_step = _gcd_of_diffs(lefts)
    step = reduce(gcd, [v for v in (row_step, col_step) if v > 0]) if (row_step or col_step) else 0
    assert step > 0, "block spacing must be recoverable"
    base_r = min(tops)
    base_c = min(lefts)
    return block_size, step, base_r, base_c


def _render_legend_frame(shape: tuple[int, int], legend_pts: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for rr, cc, col in legend_pts:
        out[LEGEND_ORIGIN[0] + rr * LEGEND_STEP, LEGEND_ORIGIN[1] + cc * LEGEND_STEP] = col
    return out


def _render_block_frame(
    shape: tuple[int, int],
    legend_pts: list[tuple[int, int, int]],
    bg: int,
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    for rr, cc, col in legend_pts:
        r0 = base_r + rr * step
        c0 = base_c + cc * step
        assert 0 <= r0 <= h - block_size and 0 <= c0 <= w - block_size, "rendered block must fit"
        out[r0 : r0 + block_size, c0 : c0 + block_size] = col
    return out


def _infer_mapping(example_legend: list[tuple[int, int, int]], example_blocks: np.ndarray) -> tuple[int, int, int, int]:
    block_size, step, base_r, base_c = _parse_block_frame(example_blocks)
    rr_vals = {rr for rr, _, _ in example_legend}
    cc_vals = {cc for _, cc, _ in example_legend}
    assert 0 in rr_vals and 0 in cc_vals, "example legend must expose the origin cell"
    return block_size, step, base_r, base_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn how one legend panel is enlarged into a block panel, then render a second legend the same way.

    Input structure:
    - The grid is split into three horizontal frames by full-height divider columns.
    - Frame 1 is a sparse legend made of isolated colored pixels on a regular lattice.
    - Frame 2 is the completed enlarged version of Frame 1, using separated uniform square blocks.
    - Frame 3 is a new legend in the same legend format.

    Rule:
    - Use the example pair (Frame 1 -> Frame 2) to infer the enlargement recipe:
      block size, spacing between block origins, and the offset of the block mosaic inside its frame.
    - Apply that same recipe to the new legend in Frame 3.
    - Output two frames side by side: the example enlarged frame, then the newly enlarged frame, with the same divider color between them.

    Key insight:
    - The example pair converts the original anchor-based completion puzzle into an analogy task:
      the first legend teaches the geometry of the enlargement, and the second legend reuses that learned geometry.
    """
    g = np.asarray(grid, dtype=int)
    frames, _bg, divider = _split_frames(g)
    assert len(frames) == 3, "expected exactly three input frames"

    kinds = [_classify_frame(fr) for fr in frames]
    assert kinds == ["legend", "blocks", "legend"], "input must be legend | blocks | legend"

    example_legend = _parse_legend_frame(frames[0])
    query_legend = _parse_legend_frame(frames[2])
    block_size, step, base_r, base_c = _infer_mapping(example_legend, frames[1])
    query_out = _render_block_frame(frames[1].shape, query_legend, get_bg(frames[1]), block_size, step, base_r, base_c)
    return _join_frames([frames[1].copy(), query_out], divider)


# --------------- generator ---------------

# The legend keeps the original staircase family so the fused task still looks
# recognizably related to c4d067a0, but the solver must now learn rendering
# parameters from a full example pair rather than partial anchors.

NUM_COLORS = 10
MAX_RETRIES = 400

LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
BASE_CHOICES = (0, 1, 2, 3)
FRAME_SIZE_MIN, FRAME_SIZE_MAX = 21, 24


def _enumerate_staircase_shapes():
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix, remaining, r):
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            shapes.append(tuple(prefix))
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    return [(r, c) for r, w in enumerate(widths) for c in range(w)]


def _random_color_counts(total: int, k: int) -> list[int]:
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        i = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[i]))
        counts[i] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_colors(pts: list[tuple[int, int]], palette: list[int]) -> list[tuple[int, int, int]]:
    counts = dict(zip(palette, _random_color_counts(len(pts), len(palette))))
    order = pts[:]
    random.shuffle(order)
    out: list[tuple[int, int, int]] = []
    for rr, cc in order:
        options = [col for col in palette if counts[col] > 0]
        col = random.choice(options)
        counts[col] -= 1
        out.append((rr, cc, col))
    return sorted(out)


def _min_frame_size(legend_pts: list[tuple[int, int, int]], block_size: int, step: int, base_r: int, base_c: int) -> int:
    max_r = max(rr for rr, _, _ in legend_pts)
    max_c = max(cc for _, cc, _ in legend_pts)
    need_h = base_r + max_r * step + block_size
    need_w = base_c + max_c * step + block_size
    return max(int(need_h), int(need_w), 1 + max_r * LEGEND_STEP + 1, 1 + max_c * LEGEND_STEP + 1)


def _build_case():
    bg = random.randrange(NUM_COLORS)
    divider_candidates = [c for c in range(NUM_COLORS) if c != bg]
    divider = random.choice(divider_candidates)
    color_pool = [c for c in range(NUM_COLORS) if c not in {bg, divider}]
    assert len(color_pool) >= PALETTE_SIZE_MAX, "not enough colors to sample"

    example_shape = _pick_staircase()
    query_shape = _pick_staircase()

    block_size = random.choice(tuple(BLOCK_SIZE_RANGE))
    gap = random.choice(tuple(GAP_RANGE))
    step = block_size + gap
    base_r = random.choice(BASE_CHOICES)
    base_c = random.choice(BASE_CHOICES)

    ex_k = random.randint(PALETTE_SIZE_MIN, min(PALETTE_SIZE_MAX, len(example_shape)))
    q_k = random.randint(PALETTE_SIZE_MIN, min(PALETTE_SIZE_MAX, len(query_shape)))
    example_palette = random.sample(color_pool, ex_k)
    query_palette = random.sample(color_pool, q_k)

    example_legend = _assign_colors(example_shape, example_palette)
    query_legend = _assign_colors(query_shape, query_palette)

    need = max(
        _min_frame_size(example_legend, block_size, step, base_r, base_c),
        _min_frame_size(query_legend, block_size, step, base_r, base_c),
    )
    if need > FRAME_SIZE_MAX:
        return None
    frame_size = random.randint(max(FRAME_SIZE_MIN, need), FRAME_SIZE_MAX)
    shape = (frame_size, frame_size)

    example_legend_frame = _render_legend_frame(shape, example_legend, bg)
    example_block_frame = _render_block_frame(shape, example_legend, bg, block_size, step, base_r, base_c)
    query_legend_frame = _render_legend_frame(shape, query_legend, bg)
    query_block_frame = _render_block_frame(shape, query_legend, bg, block_size, step, base_r, base_c)

    inp = _join_frames([example_legend_frame, example_block_frame, query_legend_frame], divider)
    out = _join_frames([example_block_frame, query_block_frame], divider)

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    return {
        "input": inp,
        "output": out,
        "meta": {
            "block_size": block_size,
            "gap": gap,
            "step": step,
            "base_r": base_r,
            "base_c": base_c,
            "frame_size": frame_size,
            "example_cells": len(example_legend),
            "query_cells": len(query_legend),
        },
    }


def generate_input():
    for _ in range(MAX_RETRIES):
        case = _build_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid framed-learning c4d067a0 variant")
