# Variant of 7666fa5d
# Dimension: D5 — 变换操作：填充颜色规则 (transform)
# Change: 所有填充区域统一使用红色(color 2)，不受线段颜色影响... → 交替颜色填充

"""
Design analysis for the D5 variant.

spec_variant:
  unchanged_structure:
    grid: 16x16
    background: 8
    line_object: one non-bg color forming anti-diagonal segments on r+c=k
    geometry_rule: fill still happens between consecutive active anti-diagonal segments while scanning b=r-c
  changed_transform:
    old: every filled cell becomes 2
    new: fill colors alternate by the left-to-right order of adjacent gaps on each scanned b=r-c diagonal
    palette: [2, 1] meaning red, blue, red, blue, ...
    ordering_basis: for one fixed b-line, sort active segments by k; the first visible gap is region 0, the next visible gap is region 1, etc.
  concrete_trace:
    example_segments_k: [6, 10, 14, 18]
    on one eligible b-line:
      old_output: cells between 6-10, 10-14, 14-18 all become 2
      new_output: cells between 6-10 -> 2, 10-14 -> 1, 14-18 -> 2
    difference:
      geometry is identical; only the mapping from "which visible adjacent gap is this on the current sweep line?" to fill color changes
  complexity_calibration:
    why_not_more_colors:
      - two fill colors are enough to make the change visible
      - more colors would add arbitrary memorization without stronger structure
    why_fixed_start_color:
      - always starting from red on the leftmost visible gap gives a stable, learnable pattern across 2-4 examples
    supporting_repetition:
      - generator keeps 4-5 distinct k-values, so there are 3-4 gap regions and both fill colors repeat
  implicit_assumption_review:
    old_main_dependencies:
      - choosing one line color from the input still holds because inputs keep only one non-bg line color
      - extracting components on constant k still holds unchanged
      - "fill = 2" no longer holds and must be replaced by gap-dependent color lookup
      - output writes only on bg cells; this remains important so line cells are preserved
    generator_dependencies_changed:
      - line color can no longer be 1, otherwise blue fill would be visually ambiguous
      - removing fill from output to recover input must erase both fill colors, not just 2
      - success validation must reject samples where only one fill color appears, otherwise the variant can visually degenerate to the old task
  placement_legality:
    - all segments are still generated from valid (k, b_min, b_max) ranges inside the 16x16 board
    - consecutive k-values still need overlapping b-ranges so every adjacent gap can produce fill
    - optional split pieces on the same k remain 8-disconnected and do not create illegal overlaps
  observability:
    - the input looks like the original task: only diagonal segments are visible
    - the output now shows alternating red/blue parallelograms aligned between neighboring segment bands
    - because every generated sample contains at least one red gap and one blue gap, the alternating rule is visible without relying on hidden numeric ordering
  cleanliness:
    - only one line color is present, so the two fill colors are easy to distinguish from the boundaries
    - fills are still disjoint parallelogram regions rather than noisy per-cell recoloring
  new_main_assumptions:
    - each scanned b-line defines a stable left-to-right gap order among currently active segments
    - generator ensures there exists at least one b-line with two or more visible gaps, so blue appears and the alternation is observable
    - line color is chosen from colors excluding 1 and 2 so output colors are unambiguous
"""

import numpy as np
import random

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return segments as (k, b_min, b_max) for 8-connected components on one line color."""
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(mask):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the parallelogram-shaped regions between adjacent anti-diagonal segments.

    Variant rule:
    - Geometry is unchanged from the original task.
    - On each scanned main diagonal b=r-c, the visible adjacent gaps alternate colors:
      leftmost visible gap red (2), next blue (1), next red, ...
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g == line)

    out = g.copy()

    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])
        ks = [t[0] for t in active]

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for idx, (k_left, k_right) in enumerate(zip(ks, ks[1:])):
            fill = 2 if idx % 2 == 0 else 1
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill

    return out


# --------------- generator ---------------

"""Generator for the D5 variant of ARC-AGI-2 task 7666fa5d."""

# -- Grid & color constants --
H = W = 16
BG = 8
FILL_COLORS = (2, 1)
LINE_COLORS = [3, 4, 5, 6, 7, 9]  # exclude 1/2 so alternating fills stay visually distinct

# -- Anti-diagonal (k = r+c) layout --
K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

# -- Segment geometry --
SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_B = 2

# -- Segment splitting --
SPLIT_PROB = 0.4
SPLIT_MIN_LEN = 4
SPLIT_B_GAP = 4
SPLIT_PIECE2_LEN_MIN = 2
SPLIT_PIECE2_LEN_MAX = 6

# -- Retry budgets --
SEG_RETRY = 40
SPLIT_RETRY = 10
GEN_RETRY = 300

# -- Validation --
MIN_FILL_CELLS = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive b-range such that (r,c)=((k+b)/2,(k-b)/2) lies within the grid."""
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    """Return (r,c) cells for a segment on anti-diagonal k spanning b_min..b_max (step 2)."""
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _apply_fill(
    grid: np.ndarray, segs: list[tuple[int, int, int]]
) -> tuple[np.ndarray, dict[int, int], int]:
    """Apply the variant fill logic and count produced cells by fill color."""
    out = grid.copy()
    color_counts = {color: 0 for color in FILL_COLORS}
    max_visible_gaps = 0
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        ks = [t[0] for t in active]
        max_visible_gaps = max(max_visible_gaps, len(ks) - 1)
        r_min, r_max = max(0, b), min(H - 1, (W - 1) + b)
        for idx, (k_left, k_right) in enumerate(zip(ks, ks[1:])):
            fill = 2 if idx % 2 == 0 else 1
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = fill
                    color_counts[fill] += 1
    return out, color_counts, max_visible_gaps


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
    k: int, constrain_lo: int | None = None, constrain_hi: int | None = None
) -> tuple[int, int] | None:
    """Pick a random segment b-range on anti-diagonal k."""
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
                bm
                for bm in candidates
                if bm + step >= constrain_lo + OVERLAP_MIN_B and bm <= constrain_hi - OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _try_split_segment(segs: list[tuple[int, int, int]], idx: int) -> None:
    """Try to split segs[idx] into two 8-disconnected pieces on the same k (in-place)."""
    k, bmin, bmax = segs[idx]
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
        segs[idx] = (k, bmin, piece1_max)
        segs.append((k, p2_min, p2_max))
        return


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
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
        prev_bmin, prev_bmax = segs[-1][1], segs[-1][2]
        res = _random_seg(k, prev_bmin, prev_bmax)
        if res is None:
            return None
        segs.append((k, res[0], res[1]))

    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    output, color_counts, max_visible_gaps = _apply_fill(grid, segs)

    if sum(color_counts.values()) < MIN_FILL_CELLS:
        return None
    if max_visible_gaps < 2:
        return None
    if any(color_counts[color] == 0 for color in FILL_COLORS):
        return None

    inp = output.copy()
    for color in FILL_COLORS:
        inp[inp == color] = BG

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
