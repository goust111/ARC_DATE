# Variant of 7666fa5d
# Dimension: D1 — 匹配条件：投影配对机制 (match_cond)
# Change: s.left_projection_point == d — 右侧对角线段的端点沿主对角线方向投影到左侧平行线段上，配对... → 颜色编码配对替代投影配对

"""
Design analysis for the D1 variant
==================================

a) Concrete behavior and trace
------------------------------
Original task:
- The input contains several anti-diagonal segments on distinct k=r+c lines.
- Along each main diagonal b=r-c, the solver collects all segments whose b-span
  covers that b, sorts them by k, and fills every gap between consecutive
  segments with the same color 2.

Variant:
- Keep the same 16x16 geometry and the same sweep along b=r-c.
- Change the pairing rule from "every consecutive pair fills" to
  "only consecutive pairs whose endpoint colors match a fixed color table fill".
- Each segment is monochrome, and the output fill color depends on the two
  boundary colors:
    {1,4} -> 2
    {3,6} -> 7
  Any other adjacent color pair produces no fill.

Small trace on one b-line:
- Suppose active segments sorted by k have colors [1,4,3,6].
- Original logic fills the 1|4 gap, the 4|3 gap, and the 3|6 gap, all in red.
- Variant logic fills only:
  - between 1 and 4 with color 2
  - between 3 and 6 with color 7
  - leaves the middle 4|3 gap as background
- The geometric sweep is unchanged; only the match condition and fill color are
  color-coded.

b) Complexity calibration
-------------------------
- Use exactly 4 segments on 4 distinct anti-diagonals. This keeps the visual
  layout close to the source task while making the color rule visible.
- Use 4 distinct line colors from a fixed small set {1,4,3,6}.
- Restrict the allowed filled color pairs to two repeated partner families:
  {1,4} and {3,6}. This is small enough to infer from few examples.
- Force the middle pair of consecutive segments to overlap geometrically but not
  match by color, so the variant differs visibly from the original task.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- `line = max(non_bg, key=...)`
  Old assumption: all line segments share one color. Invalid now.
  Fix: collect components for every non-bg color.
- `fill = 2`
  Old assumption: every filled region is red. Invalid now.
  Fix: choose fill color from a pair-to-color lookup.
- `comps = _cc8(g == line)`
  Old assumption: one mask is enough to recover every segment. Invalid now.
  Fix: run component extraction independently per color.
- `active.sort(key=lambda t: t[0])` and then fill every adjacent pair
  Old assumption: geometry alone decides pairing. Invalid now.
  Fix: after sorting by k, test the adjacent colors against the color table.
- `if k_left < k < k_right and out[r, c] == bg: out[r, c] = fill`
  Old assumption: no ambiguity from a single global fill color. Invalid now.
  Fix: use the mapped fill color for the specific left/right segment colors.

d) Placement legality
---------------------
- All segments still lie on one anti-diagonal each and stay in-bounds by
  sampling valid b-ranges from the original formula.
- Consecutive segments are forced to overlap in b-space so the sweep sees them
  on the same main diagonals.
- No two segments share a cell because every segment has a distinct k value.
- The generator validates `main(input) == output` before returning.

e) Visual observability
-----------------------
- The input already exposes the control signal: four differently colored
  anti-diagonal segments.
- The output shows two filled parallelograms in two different colors and one
  skipped geometric gap in the middle.
- Because the skipped middle gap is geometrically valid, a human can see that
  color matching, not geometry alone, is controlling the fill.

f) Visual cleanliness
---------------------
- There are only 4 segments and at most 2 fill colors, so the grid stays clean.
- Fill regions occupy disjoint k-intervals, so different fill colors do not
  overwrite each other.
- No extra legend objects are added; the task remains a direct diagonal-fill
  puzzle rather than becoming a multi-object scene.

g) Color-rule inferability
--------------------------
- The line-color alphabet is fixed and small: {1,4,3,6}.
- The fill-color alphabet is fixed and small: {2,7}.
- The generator always activates both mapping entries and also includes one
  non-matching adjacent pair. This gives examples positive and negative evidence
  for the same color rule.

New assumptions introduced by this variant
------------------------------------------
- `main()` assumes every meaningful segment is a monochrome 8-connected
  component lying on exactly one anti-diagonal.
- `main()` assumes filled regions themselves are not single-k components, so
  they are ignored if `main()` is run on an already-solved output.
- The generator guarantees the segment colors come from the fixed palette and
  that the k-order template is one of the supported partner-pair templates.
- The generator also guarantees the middle non-matching pair has positive
  geometric overlap, preventing accidental degeneration back to the original
  "fill every gap" behavior.
"""

import random
import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


H = W = 16
BG = 8
PAIR_TO_FILL = {
    frozenset((1, 4)): 2,
    frozenset((3, 6)): 7,
}
LINE_COLORS = (1, 4, 3, 6)
COLOR_ORDER_TEMPLATES = (
    (1, 4, 3, 6),
    (4, 1, 3, 6),
    (1, 4, 6, 3),
    (4, 1, 6, 3),
    (3, 6, 1, 4),
    (6, 3, 1, 4),
    (3, 6, 4, 1),
    (6, 3, 4, 1),
)

K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
SEG_LEN_MIN, SEG_LEN_MAX = 3, 9
OVERLAP_MIN_B = 4
SEG_RETRY = 60
GEN_RETRY = 400
MIN_PAIR_FILL_CELLS = 3
MIN_BLOCKED_CELLS = 4


def _segment_info(pts: list[tuple[int, int]], color: int) -> tuple[int, int, int, int] | None:
    ks = {r + c for r, c in pts}
    if len(ks) != 1:
        return None
    k = int(next(iter(ks)))
    bs = sorted(r - c for r, c in pts)
    if not bs:
        return None
    if any(bs[i] + 2 != bs[i + 1] for i in range(len(bs) - 1)):
        return None
    return k, int(min(bs)), int(max(bs)), int(color)


def _collect_segments(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    bg = int(_bg_color(g))
    segs: list[tuple[int, int, int, int]] = []
    for color in sorted(set(map(int, g.ravel()))):
        if color == bg:
            continue
        for pts in _cc8(g == color):
            info = _segment_info(pts, color)
            if info is not None:
                segs.append(info)
    segs.sort(key=lambda t: (t[0], t[1], t[3]))
    return segs


def _fill_mask_and_output(
    grid: np.ndarray,
    segs: list[tuple[int, int, int, int]],
    *,
    pair_to_fill: dict[frozenset[int], int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(grid, dtype=int).copy()
    mask = np.zeros_like(out, dtype=bool)
    h, w = out.shape
    bg = int(_bg_color(out))

    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for left, right in zip(active, active[1:]):
            if pair_to_fill is None:
                fill = 1
            else:
                fill = pair_to_fill.get(frozenset((left[3], right[3])))
                if fill is None:
                    continue
            k_left, k_right = int(left[0]), int(right[0])
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = int(fill)
                    mask[r, c] = True
    return out, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only those inter-segment gaps whose adjacent segment colors match one
    of the allowed color pairs. The fill color depends on the pair.
    """
    g = np.asarray(grid, dtype=int)
    segs = _collect_segments(g)
    assert segs, "grid must contain anti-diagonal line segments"
    out, _mask = _fill_mask_and_output(g, segs, pair_to_fill=PAIR_TO_FILL)
    return out


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _choose_k_values(parity: int) -> list[int] | None:
    valid_gaps = [(a, b, c) for a in GAP_VALUES for b in GAP_VALUES for c in GAP_VALUES if a + b + c in GAP_TARGET_SPANS]
    gaps = random.choice(valid_gaps)
    total_span = sum(gaps)
    starts = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not starts:
        return None
    k0 = random.choice(starts)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    return ks if ks[-1] <= K_HI else None


def _random_seg(
    k: int,
    constrain_lo: int | None = None,
    constrain_hi: int | None = None,
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
                start
                for start in candidates
                if start + step >= constrain_lo + OVERLAP_MIN_B and start <= constrain_hi - OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _blocked_gap_cells(segs: list[tuple[int, int, int, int]]) -> int:
    segs = sorted(segs, key=lambda t: t[0])
    total = 0
    for left, right in zip(segs, segs[1:]):
        if frozenset((left[3], right[3])) in PAIR_TO_FILL:
            continue
        overlap_lo = max(left[1], right[1])
        overlap_hi = min(left[2], right[2])
        if overlap_lo > overlap_hi:
            continue
        for b in range(overlap_lo, overlap_hi + 1, 2):
            total += max(0, (right[0] - left[0]) // 2 - 1)
    return total


def _gen_attempt() -> np.ndarray | None:
    parity = random.randint(0, 1)
    ks = _choose_k_values(parity)
    if ks is None:
        return None

    order = random.choice(COLOR_ORDER_TEMPLATES)
    segs: list[tuple[int, int, int, int]] = []

    first = _random_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1], order[0]))

    for k, color in zip(ks[1:], order[1:]):
        prev = segs[-1]
        res = _random_seg(k, prev[1], prev[2])
        if res is None:
            return None
        segs.append((k, res[0], res[1], color))

    blocked_cells = _blocked_gap_cells(segs)
    if blocked_cells < MIN_BLOCKED_CELLS:
        return None

    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max, color in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = color

    output, variant_mask = _fill_mask_and_output(grid, segs, pair_to_fill=PAIR_TO_FILL)
    _old_output, old_mask = _fill_mask_and_output(grid, segs, pair_to_fill=None)

    if int(np.sum(old_mask & (~variant_mask))) < MIN_BLOCKED_CELLS:
        return None

    if not all(int(np.sum(output == fill)) >= MIN_PAIR_FILL_CELLS for fill in sorted(set(PAIR_TO_FILL.values()))):
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
