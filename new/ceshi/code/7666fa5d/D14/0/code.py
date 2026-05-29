# Variant of 7666fa5d
# Dimension: D14 — 符号编码：控制信息 (symbol_instructions)
# Change: 无符号编码——任务规则完全由空间几何关系隐含定义，无图例、标记像素或编码控制参数... → 图例指示填充颜色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the D14 legend variant
# =========================================
# spec_variant:
#   grid: 16x16
#   background: 8
#   new_symbol:
#     location: fixed top-left corner
#     footprint: 3x5
#     visual_form:
#       - whole 3x5 patch is black
#       - exactly one row contains two non-black swatches
#       - left swatch = line color, right swatch = fill color
#     concrete pattern:
#       row 0: 0 0 0 0 0
#       row 1: 0 line 0 fill 0
#       row 2: 0 0 0 0 0
#   task_body:
#     unchanged_geometry: anti-diagonal segments on r+c=k, filled between adjacent active segments along r-c=b
#     changed_parameter: fill color is no longer constant red; it is read from the legend
#
# a) Concrete behavior trace
#   Original flow:
#   - detect the dominant non-bg line color
#   - collect its anti-diagonal segments
#   - whenever two consecutive active segments bracket empty cells, paint those cells with fixed color 2
#   Variant flow:
#   - first read the legend row [line -> fill]
#   - ignore the legend patch when extracting segments
#   - run the same geometric sweep, but paint with the legend's fill color instead of hard-coded 2
#   Small example:
#   - legend row is [0, 4, 0, 6, 0]
#   - body contains yellow (4) anti-diagonal segments
#   - original solver would paint red (2)
#   - variant paints pink (6)
#   - the only rule change is "which color gets written"; the region shape stays identical
#
# b) Complexity calibration
#   - Keep one line color per puzzle. That preserves the original geometric reasoning.
#   - Make fill color vary over several non-bg colors so the legend is visually necessary.
#   - Use a black framed patch rather than free-floating marker pixels, so humans can reliably notice it as a legend.
#   - Avoid adding extra distractor mappings; one clear mapping is enough and keeps the variant readable.
#
# c) Old implicit assumptions that break
#   - "fill = 2" is no longer valid.
#   - "largest non-bg color is the only relevant body color" is unsafe because the legend adds extra non-bg pixels.
#   - "every component of the line color belongs to the task body" is unsafe because the legend also contains one pixel
#     of the line color.
#   Therefore main() must parse the legend first, blank out that patch for body analysis, and only then extract segments.
#
# d) Placement legality
#   - The legend occupies rows 0..2, cols 0..4.
#   - The generator constrains all task geometry to k=r+c >= 7, so neither line cells nor filled cells can ever enter the
#     legend patch (its maximum k is 2+4=6).
#   - This removes overlap ambiguity and guarantees the legend remains untouched in input and output.
#
# e) Visual observability
#   - Across examples, the body shape changes as before, but the filled color is no longer always red.
#   - The top-left legend directly shows the line color and the produced fill color in the same sample.
#   - With 2-3 samples, a human can align "left swatch matches segment color" and "right swatch matches new output color".
#
# f) Visual cleanliness
#   - The legend sits in an otherwise reserved corner.
#   - The black patch is compact and visually separate from the diagonal structure.
#   - No task fill is allowed underneath it, so there is no layered clutter.
#
# g) Color-change inferability
#   - Fill colors are chosen from a small ARC palette subset excluding bg and black.
#   - The line color is reused both in the legend and in the body, giving a concrete lookup key.
#   - Because the geometry is unchanged, the only moving part is the color lookup, which is exactly what D14 wants.
#
# New main() assumptions review
#   - Assumption 1: the legend is always in the fixed top-left 3x5 patch.
#     Generator guarantee: every sample writes the legend there.
#   - Assumption 2: that patch contains at least one row with exactly two non-black swatches.
#     Generator guarantee: it always writes exactly one such row.
#   - Assumption 3: body line colors appear in the legend mapping.
#     Generator guarantee: every body segment uses the legend's left swatch color.
#   - Assumption 4: body geometry never overlaps the legend patch.
#     Generator guarantee: all body cells satisfy k>=7, while legend cells satisfy k<=6.


BLACK = 0
H = W = 16
BG = 8

LEGEND_R0 = 0
LEGEND_C0 = 0
LEGEND_H = 3
LEGEND_W = 5

LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]
FILL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 7, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_B = 2

SPLIT_PROB = 0.4
SPLIT_MIN_LEN = 4
SPLIT_B_GAP = 4
SPLIT_PIECE2_LEN_MIN = 2
SPLIT_PIECE2_LEN_MAX = 6

SEG_RETRY = 40
SPLIT_RETRY = 10
GEN_RETRY = 300
MIN_FILL_CELLS = 3


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[LEGEND_R0 : LEGEND_R0 + LEGEND_H, LEGEND_C0 : LEGEND_C0 + LEGEND_W] = True
    return mask


def _read_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    box = grid[LEGEND_R0 : LEGEND_R0 + LEGEND_H, LEGEND_C0 : LEGEND_C0 + LEGEND_W]
    mapping: dict[int, int] = {}
    for rr in range(box.shape[0]):
        row_entries = [
            (int(cc), int(box[rr, cc]))
            for cc in range(box.shape[1])
            if int(box[rr, cc]) not in (int(bg), BLACK)
        ]
        if len(row_entries) != 2:
            continue
        row_entries.sort(key=lambda t: t[0])
        line = row_entries[0][1]
        fill = row_entries[1][1]
        mapping[line] = fill
    assert mapping, "legend mapping missing"
    return mapping


def _draw_legend(grid: np.ndarray, line: int, fill: int) -> None:
    grid[LEGEND_R0 : LEGEND_R0 + LEGEND_H, LEGEND_C0 : LEGEND_C0 + LEGEND_W] = BLACK
    grid[LEGEND_R0 + 1, LEGEND_C0 + 1] = int(line)
    grid[LEGEND_R0 + 1, LEGEND_C0 + 3] = int(fill)


def _extract_segments(grid: np.ndarray, legend_colors: dict[int, int]) -> list[tuple[int, int, int, int]]:
    segs: list[tuple[int, int, int, int]] = []
    for line in sorted(legend_colors):
        for pts in _cc8(grid == int(line)):
            ks = {r + c for r, c in pts}
            if len(ks) != 1:
                continue
            k = int(next(iter(ks)))
            bs = [r - c for r, c in pts]
            segs.append((k, int(min(bs)), int(max(bs)), int(line)))
    return segs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same geometric fill as 7666fa5d, but the fill color comes from a corner legend.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    legend = _read_legend(g, bg)

    body = g.copy()
    body[_legend_mask(g.shape)] = bg
    segs = _extract_segments(body, legend)
    assert segs, "grid must contain legend-referenced line segments"

    out = g.copy()

    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left, _b0, _b1, line_left = left
            k_right, _c0, _c1, line_right = right
            if line_left != line_right:
                continue
            fill = int(legend[line_left])
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill

    return out


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _apply_fill(
    grid: np.ndarray, segs: list[tuple[int, int, int, int]], legend: dict[int, int]
) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        r_min, r_max = max(0, b), min(H - 1, (W - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left, _b0, _b1, line_left = left
            k_right, _c0, _c1, line_right = right
            if line_left != line_right:
                continue
            fill = int(legend[line_left])
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = fill
    return out


def _choose_k_values(parity: int) -> list[int] | None:
    num_ks = random.choices(NUM_KS_CHOICES, weights=NUM_KS_WEIGHTS)[0]
    if num_ks == 4:
        valid_combos = [
            (a, b, c)
            for a in GAP_VALUES
            for b in GAP_VALUES
            for c in GAP_VALUES
            if a + b + c in GAP_TARGET_SPANS
        ]
        gaps = list(random.choice(valid_combos))
    else:
        gaps = [4, 4, 4, 4]

    total_span = sum(gaps)
    k0_candidates = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not k0_candidates:
        return None
    k0 = random.choice(k0_candidates)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    if ks[-1] > K_HI or len(set(ks)) != len(ks):
        return None
    return ks


def _random_seg(
    k: int, constrain_lo: int | None = None, constrain_hi: int | None = None
) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k(k)
    if b_lo % 2 != k % 2:
        b_lo += 1
    if b_hi % 2 != k % 2:
        b_hi -= 1
    if b_lo > b_hi:
        return None

    for _ in range(SEG_RETRY):
        length = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
        step = 2 * (length - 1)
        if b_lo + step > b_hi:
            continue
        candidates = list(range(b_lo, b_hi - step + 1, 2))
        if constrain_lo is not None and constrain_hi is not None:
            candidates = [
                b_min
                for b_min in candidates
                if b_min + step >= constrain_lo + OVERLAP_MIN_B and b_min <= constrain_hi - OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _try_split_segment(segs: list[tuple[int, int, int, int]], idx: int) -> None:
    k, bmin, bmax, line = segs[idx]
    length = (bmax - bmin) // 2 + 1
    if length < SPLIT_MIN_LEN:
        return

    _, b_hi = _b_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        l1 = random.randint(2, length - 2)
        piece1_max = bmin + 2 * (l1 - 1)
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if b % 2 == k % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        l2 = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (l2 - 1)
        if p2_max > b_hi:
            continue
        segs[idx] = (k, bmin, piece1_max, line)
        segs.append((k, p2_min, p2_max, line))
        return


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    fill_choices = [c for c in FILL_COLORS if c not in (BLACK, BG, line)]
    fill = random.choice(fill_choices)
    legend = {line: fill}
    parity = random.randint(0, 1)

    ks = _choose_k_values(parity)
    if ks is None:
        return None

    segs: list[tuple[int, int, int, int]] = []
    first = _random_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1], line))

    for k in ks[1:]:
        prev_bmin, prev_bmax = segs[-1][1], segs[-1][2]
        res = _random_seg(k, prev_bmin, prev_bmax)
        if res is None:
            return None
        segs.append((k, res[0], res[1], line))

    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax, seg_line in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            grid[r, c] = seg_line

    legend_mask = _legend_mask(grid.shape)
    if np.any(grid[legend_mask] != BG):
        return None

    output = _apply_fill(grid, segs, legend)
    if int(np.sum(output == fill)) < MIN_FILL_CELLS:
        return None
    if np.any(output[legend_mask] != BG):
        return None

    inp = output.copy()
    body_fill = (inp == fill) & (~legend_mask)
    inp[body_fill] = BG

    _draw_legend(inp, line, fill)
    _draw_legend(output, line, fill)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
