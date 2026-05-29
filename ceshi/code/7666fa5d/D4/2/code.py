# Variant of 7666fa5d
# Dimension: D4 — 目标位置：填充区域定义 (dst)
# Change: 沿主对角线方向(b=r-c固定)，在相邻反对角线段之间严格内部的背景色像素位置填充红色... → 填充扩展到最外侧线段之外

"""
Design analysis for the D4 variant.

variant_spec:
  core_change:
    original: |
      On each main diagonal b=r-c, gather active anti-diagonal segments (constant k=r+c)
      and fill only bg cells with k strictly between consecutive active k-values.
    variant: |
      On each main diagonal b=r-c, if there is at least one active segment, fill every bg
      cell on that diagonal except the line cells themselves. Equivalently, keep the original
      "between consecutive segments" fill and extend it beyond the outermost active segments
      to both ends of that diagonal inside the grid.
  pixel_behavior:
    objects:
      - one line color forms short connected anti-diagonal segments
      - background is 8
      - output adds red 2
    visual_effect:
      - original outputs only bounded parallelogram strips between neighboring segments
      - variant outputs the same interior strips plus triangular / trapezoidal outer tails
        that reach the grid border along the same b-lines
      - on some b-lines with only one active segment, the whole rest of that diagonal turns red
  concrete_trace:
    setup:
      - fixed diagonal b has cells with k values [3,5,7,9,11]
      - active segment cells on that b occur at k=5 and k=9
    original:
      - fill only k=7
    variant:
      - fill k=3,7,11 as well, because they are outside the outermost segments and still bg
    difference:
      - the solver must reason about diagonal endpoints, not only adjacent segment pairs
  complexity_calibration:
    - keep the original 16x16 size, one line color, and anti-diagonal segment layout
    - keep 4-5 k-levels and optional splitting so examples still show repeated structure
    - require generated examples to contain a noticeable number of outer-extension cells,
      otherwise the sample could look too similar to the original task
  implicit_assumption_audit:
    - component extraction still assumes each useful object is an 8-connected segment on one k
    - choosing the dominant non-bg color as the line color still holds because the input contains
      only bg plus that single line color
    - the old solver's `if len(active) < 2: continue` is no longer valid; one active segment is
      sufficient in the variant because outside fill is defined relative to the diagonal boundary
    - the old pairwise fill loop only reasons about gaps between segments; the variant additionally
      depends on the first and last reachable k on each diagonal
    - writing fill only onto bg remains safe, so line cells are preserved even when the variant
      paints most of a diagonal
  placement_legality:
    - all segment cells remain inside the 16x16 grid via exact b-bounds checks
    - consecutive k-levels still overlap in b-space, so multi-segment diagonals exist
    - optional split pieces on the same k stay 8-disconnected and do not overlap
    - generator rejects cases where the variant adds too few cells or fails to reach the border
  observability:
    - humans can see that red does not stop at the outer line segments anymore; it runs to the edge
    - border-touching red is a strong cue, so the generator requires it in the variant-only region
  cleanliness:
    - only three colors remain visible in any sample: bg, one line color, red fill
    - no extra markers or color remappings are introduced
  new_solver_assumptions_and_generator_support:
    - solver assumes active segments are still recoverable as single-k components
    - solver assumes bg is the only paintable color
    - solver assumes at least one non-bg component exists
    - generator enforces all of the above and validates every sample with `main(inp) == output`
"""

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _diag_row_bounds(b: int, h: int, w: int) -> tuple[int, int]:
    """Inclusive row range for the main diagonal r-c=b."""
    return max(0, b), min(h - 1, (w - 1) + b)


def _fill_variant(grid: np.ndarray, segs: list[tuple[int, int, int]], bg: int, fill: int) -> np.ndarray:
    """
    Variant fill rule:
    on each fixed b=r-c diagonal, if at least one segment is active on that b,
    paint every background cell on that diagonal except the segment cells themselves.
    """
    h, w = grid.shape
    out = grid.copy()

    for b in range(-(w - 1), h):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if not active:
            continue
        ks = [t[0] for t in active]
        r_min, r_max = _diag_row_bounds(b, h, w)
        for r in range(r_min, r_max + 1):
            c = r - b
            if out[r, c] != bg:
                continue
            k = r + c
            if k < ks[0] or k > ks[-1]:
                out[r, c] = fill
                continue
            for k_left, k_right in zip(ks, ks[1:]):
                if k_left < k < k_right:
                    out[r, c] = fill
                    break

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver for 7666fa5d.

    Recover anti-diagonal line segments, then sweep each main diagonal b=r-c.
    Instead of filling only the strict interior between consecutive active segments,
    fill every background cell on that b-line outside the line cells as well, all the
    way to the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2

    comps = _cc8(g == line)

    segs: list[tuple[int, int, int]] = []
    for pts in comps:
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))

    return _fill_variant(g, segs, bg, fill)


# --------------- generator ---------------

H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
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
GEN_RETRY = 400

MIN_FILL_CELLS = 12
MIN_OUTER_ONLY_CELLS = 8
MIN_OUTER_EDGE_CELLS = 4
MIN_LEFT_EXT_DIAGS = 2
MIN_RIGHT_EXT_DIAGS = 2


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive b-range such that (r,c)=((k+b)/2,(k-b)/2) lies within the grid."""
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    """Cells of a segment on anti-diagonal k spanning b_min..b_max, step 2 in b."""
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _apply_fill_inner(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    """Original task fill, kept for variant-vs-original rejection checks."""
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        ks = [t[0] for t in active]
        r_min, r_max = _diag_row_bounds(b, H, W)
        for k_left, k_right in zip(ks, ks[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def _apply_fill_outer(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    return _fill_variant(grid, segs, BG, FILL)


def _choose_k_values(parity: int) -> list[int] | None:
    """Pick 4-5 evenly-spaced anti-diagonal indices with the given parity."""
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
    k: int,
    constrain_lo: int | None = None,
    constrain_hi: int | None = None,
) -> tuple[int, int] | None:
    """
    Pick a random segment b-range on anti-diagonal k.

    If constrain_lo/hi are given, the new segment must overlap that b-range
    by at least OVERLAP_MIN_B in b-space.
    """
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
                if b_min + step >= constrain_lo + OVERLAP_MIN_B
                and b_min <= constrain_hi - OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _try_split_segment(segs: list[tuple[int, int, int]], idx: int) -> None:
    """Try to split one segment into two 8-disconnected pieces on the same k."""
    k, b_min, b_max = segs[idx]
    length = (b_max - b_min) // 2 + 1
    if length < SPLIT_MIN_LEN:
        return

    _, b_hi = _b_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        first_len = random.randint(2, length - 2)
        piece1_max = b_min + 2 * (first_len - 1)
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if b % 2 == k % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        second_len = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (second_len - 1)
        if p2_max > b_hi:
            continue
        segs[idx] = (k, b_min, piece1_max)
        segs.append((k, p2_min, p2_max))
        return


def _boundary_mask() -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


BOUNDARY_MASK = _boundary_mask()


def _variant_metrics(segs: list[tuple[int, int, int]], outer: np.ndarray, inner: np.ndarray) -> dict[str, int]:
    outer_only = (outer == FILL) & (inner != FILL)
    left_ext_diags = 0
    right_ext_diags = 0

    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if not active:
            continue
        r_min, r_max = _diag_row_bounds(b, H, W)
        k_start = 2 * r_min - b
        k_end = 2 * r_max - b
        if k_start < active[0][0]:
            left_ext_diags += 1
        if k_end > active[-1][0]:
            right_ext_diags += 1

    return {
        "fill_cells": int(np.sum(outer == FILL)),
        "outer_only_cells": int(np.sum(outer_only)),
        "outer_edge_cells": int(np.sum(outer_only & BOUNDARY_MASK)),
        "left_ext_diags": left_ext_diags,
        "right_ext_diags": right_ext_diags,
    }


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt."""
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)

    ks = _choose_k_values(parity)
    if ks is None:
        return None

    segs: list[tuple[int, int, int]] = []
    first = _random_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1]))

    for k in ks[1:]:
        prev_b_min, prev_b_max = segs[-1][1], segs[-1][2]
        nxt = _random_seg(k, prev_b_min, prev_b_max)
        if nxt is None:
            return None
        segs.append((k, nxt[0], nxt[1]))

    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    outer = _apply_fill_outer(grid, segs)
    inner = _apply_fill_inner(grid, segs)
    metrics = _variant_metrics(segs, outer, inner)

    if metrics["fill_cells"] < MIN_FILL_CELLS:
        return None
    if metrics["outer_only_cells"] < MIN_OUTER_ONLY_CELLS:
        return None
    if metrics["outer_edge_cells"] < MIN_OUTER_EDGE_CELLS:
        return None
    if metrics["left_ext_diags"] < MIN_LEFT_EXT_DIAGS:
        return None
    if metrics["right_ext_diags"] < MIN_RIGHT_EXT_DIAGS:
        return None

    inp = outer.copy()
    inp[inp == FILL] = BG

    if not np.array_equal(main(inp), outer):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input")
