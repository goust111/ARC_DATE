# Variant of 8f215267
# Dimension: D5 — 变换规则：点的布局方式 (transform)
# Change: 在框架中心行从右侧起每隔2列放置N个同色点（列坐标: c1-2, c1-4, c1-6, ...）... → 从左侧开始放置

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D5 variant
#
# source_task:
#   None. No external reference task is required for this inversion variant.
#
# variant_spec:
#   grid:
#     23 columns fixed
#     rows determined by 3-4 stacked frames plus optional bottom padding
#   objects:
#     - 3-4 hollow 5x11 monochrome rectangular frames on the left
#     - many small monochrome sprites on the right
#     - 1-2 distractor sprite colors that do not match any frame
#   preserved_rule:
#     - remove every right-side sprite from the output
#     - for each frame color C, count how many sprites have color C
#     - show the count as evenly spaced dots on the frame center row
#   changed_rule:
#     old placement:
#       columns c1-2, c1-4, c1-6, ... measured from the frame's right border
#     new placement:
#       columns c0+2, c0+4, c0+6, ... measured from the frame's left border
#
# a) concretization and behavior trace:
#   A frame with bbox (r0, c0, r1, c1) = (1, 1, 5, 11) has center row r=3.
#   If it matches 3 sprites:
#     - old task writes dots at columns 9, 7, 5
#     - new variant writes dots at columns 3, 5, 7
#   So the output is a left-right mirror of the tally pattern inside each frame.
#   The input construction changes only when removing the future tally dots from
#   the output-first template: the generator must now erase left-starting dots
#   instead of right-starting dots.
#
# b) complexity calibration:
#   The core rule stays simple: count by color, then display a tally. That keeps
#   the task aligned with the original concept. To make the D5 change visible,
#   the generator guarantees at least one frame receives >=2 dots. With only a
#   single dot, the direction change is still visible, but repeated spacing from
#   the left wall gives stronger evidence across a small number of examples.
#   No extra object types, ordering rules, or color remappings are introduced.
#
# c) hidden-assumption audit for main():
#   - bg = get_bg(g):
#       assumes the background is the most common color. Still valid because
#       frames are hollow and sprites are sparse.
#   - find_components(... monochromatic=True, connectivity=4):
#       assumes each frame/sprite is a single-color 4-connected component.
#       Generator preserves that.
#   - is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7):
#       assumes frames are large sparse rectangles and sprites are smaller.
#       Still valid because frames remain 5x11 hollow rectangles and sprites
#       stay within bbox <= 4x3.
#   - removing every non-frame component before drawing dots:
#       assumes sprites never overlap frames. Generator keeps all sprites in the
#       right-side region only, so erasing patterns cannot damage a frame.
#   - counting patterns by exact color equality:
#       assumes frame colors are distinct and distractor colors do not match any
#       frame color. Generator preserves that palette separation.
#   - new dot placement loop:
#       assumes at most four dots are needed so the columns c0+2, c0+4, c0+6,
#       c0+8 stay strictly inside a width-11 frame. Generator enforces the same
#       MAX_DOTS_PER_FRAME=4 cap as the original task.
#
# d) placement legality:
#   Frames occupy columns 1..11 and sprites are pasted only into a separate
#   right canvas of width 9, then overlaid starting at global column 13. This
#   preserves a blank separator column and prevents frame/sprite overlap. Sprite
#   placement uses random_free_location(..., padding=1), so sprites neither
#   overlap nor touch each other diagonally.
#
# e) visual observability:
#   The difference is human-visible because the output dots visibly hug the left
#   inner wall instead of the right inner wall. Ensuring at least one tally with
#   >=2 dots makes the left-to-right accumulation easy to infer from examples.
#
# f) visual cleanliness:
#   The left region contains only frames and tally dots; the right region
#   contains only sprites. There is no mixed overlay in the same area, and the
#   center-row dots remain aligned and evenly spaced.
#
# g) color-change inferability:
#   No color transformation is added. Colors still indicate the mapping between
#   each frame and its matching sprites, which keeps the original reasoning path.
#
# h) quantitative non-degeneration target:
#   A generated sample is considered expressive for this variant if:
#     1. at least one frame has a positive tally,
#     2. at least one frame has tally >= 2, and
#     3. at least one non-zero tally is partial: 1, 2, or 3 rather than 4
#   Condition (3) matters because a full 4-dot row occupies columns 2,4,6,8,
#   which is identical under left-fill and right-fill. The generator therefore
#   avoids samples where every non-zero tally is exactly 4.
#
# i) new assumptions introduced by the modified main():
#   main() now assumes the intended tally anchor is the left inner wall rather
#   than the right inner wall. generate_input() guarantees this by building the
#   target output with the same left-starting column sequence and erasing those
#   exact positions when deriving the input.
# ---------------------------------------------------------------------------


# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _dot_cols_from_left(n_dots: int) -> list[int]:
    """Interior patch columns for left-anchored tally dots."""
    return [2 + 2 * i for i in range(int(n_dots))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and represent the tally as dots inside.

    Variant change:
    - Dots are placed on the center row from the LEFT interior wall, not from
      the right interior wall used in the original task.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(int(num_patterns)):
            cc = int(c0 + 2 + 2 * i)
            if cc < int(c1):
                out[int(r_mid), int(cc)] = col

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with left-anchored tally dots."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    patch[mid_r, _dot_cols_from_left(n_dots)] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """
    Distribute total_match sprites across n_frames, each capped at MAX_DOTS_PER_FRAME.

    Adaptation for D5 observability:
    guarantee at least one frame receives >=2 dots, and guarantee at least one
    non-zero tally is partial (1..3) so the sample cannot collapse back to the
    original right-anchored output.
    """
    counts = [0] * n_frames
    order = rng.permutation(n_frames)
    remaining = total_match
    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        lo = max(0, remaining - MAX_DOTS_PER_FRAME * frames_left)
        hi = min(MAX_DOTS_PER_FRAME, remaining)
        val = int(rng.integers(lo, hi + 1))
        counts[int(idx)] = val
        remaining -= val

    if max(counts) >= 2:
        pass
    else:
        positive = [i for i, v in enumerate(counts) if v > 0]
        if len(positive) >= 2:
            receiver = int(positive[0])
            donor = int(positive[-1])
            counts[receiver] += 1
            counts[donor] -= 1

    if not any(0 < v < MAX_DOTS_PER_FRAME for v in counts):
        donor = next(i for i, v in enumerate(counts) if v == MAX_DOTS_PER_FRAME)
        receiver = next(i for i, v in enumerate(counts) if v == 0)
        counts[donor] -= 1
        counts[receiver] += 1
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino of the given area inside an h x w bounding box."""
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.append(nb)
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    """Generate a small random connected sprite, tight-cropped to its bounding box."""
    area = int(rng.integers(*SPRITE_AREA_RANGE))
    dims = [
        (h, w)
        for h in range(1, SPRITE_MAX_H + 1)
        for w in range(1, SPRITE_MAX_W + 1)
        if h * w >= area
    ]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, n_frames, total_match)
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + col for col in _dot_cols_from_left(count)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
