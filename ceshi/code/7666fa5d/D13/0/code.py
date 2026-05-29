# Variant of 7666fa5d
# Dimension: D13 — 线段重叠范围约束 (meta)
# Change: 相邻反对角线上的线段必须有至少2个b值的重叠范围(OVERLAP_MIN_B=2)，确保它们之间存在可填充区域... → 无重叠约束（允许无填充线段对）
#
# Design analysis
# - a) Concrete behavior and trace:
#   Objects are still short contiguous segments on anti-diagonals k=r+c, and output still fills red
#   cells by sweeping each main diagonal b=r-c. The variant is visible when two neighboring
#   anti-diagonals exist but their segment b-ranges do not overlap at all, so there is no b on which
#   they are simultaneously active. Example:
#   k=8,  b in [-2, 4]
#   k=12, b in [ 6, 12]
#   On b=4 only the first segment is active; on b=6 only the second is active; therefore the active
#   list never contains both segments together, so no red strip appears between them. Under the
#   original generator this pair was illegal because it violated the forced overlap rule. In the
#   variant generator this pair is legal, but other neighboring anti-diagonals still overlap so the
#   puzzle retains fill regions and the solver must decide which neighboring pairs actually produce
#   red.
# - b) Complexity calibration:
#   The generator always creates two contiguous anti-diagonal groups. Inside each group, neighboring
#   segments overlap enough to yield fill. Across the single group boundary, the b-bands are fully
#   disjoint. This keeps the new condition observable without introducing many unrelated degrees of
#   freedom.
# - c) Implicit assumption audit for main():
#   `line = max(non_bg, key=count)` assumes one foreground line color; generator still guarantees it.
#   Component extraction assumes each component lies on one anti-diagonal; generator still builds only
#   contiguous anti-diagonal segments.
#   `active = [s for s in segs if s[1] <= b <= s[2]]` is the key line: it does not require global
#   overlap, only per-b simultaneous activity. That remains correct in the variant; disjoint neighbors
#   simply never co-activate.
#   `if len(active) < 2: continue` is still valid because some b lines now legitimately have too few
#   active segments to fill.
#   Filling only background cells avoids overwriting line cells; generator still ensures line cells are
#   the only non-background input marks.
# - d) Placement legality:
#   Every segment is sampled from valid in-bounds b ranges for its k. The second anti-diagonal group is
#   forced entirely above or below the first group's total b-span, so there is no accidental
#   cross-group fill. Output is still validated by running main(input) == output.
# - e) Visual observability:
#   Each generated puzzle contains at least one neighboring distinct-k pair with zero shared b values
#   and at least one overlapping pair that does fill. From 2-4 examples, a human can see that some
#   neighboring rails generate red parallelograms while another neighboring gap does not.
# - f) Visual cleanliness:
#   The optional same-k split from the original generator is disabled here. That split can place an
#   extra component on the same anti-diagonal in a distant b band, which obscures whether D13 is the
#   reason a neighboring pair does or does not fill. Removing it is a compatibility adjustment to make
#   the new behavior legible.
# - g) Color inference:
#   Colors stay unchanged from the base task: one line color, red fill, azure background. No hidden
#   color-based ordering is introduced.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Extract (k, b_min, b_max) for each 8-connected anti-diagonal segment."""
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
    Fill red cells between anti-diagonal segments by scanning each main diagonal b=r-c.

    Variant note:
    Adjacent anti-diagonal segments are no longer assumed to overlap globally in b-space.
    The correct rule is local: on each b, only the segments active on that b participate in
    filling. If two neighboring segments have disjoint b-ranges, they simply never appear
    together in the active list, so no fill is produced between them.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2
    segs = _extract_segments(g == line)

    out = g.copy()
    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])
        for (k_left, _, _), (k_right, _, _) in zip(active, active[1:]):
            r_min = max(0, b)
            r_max = min(h - 1, (w - 1) + b)
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill

    return out


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
DISJOINT_GAP_MIN_B = 2

SPLIT_PROB = 0.0

SEG_RETRY = 40
GEN_RETRY = 400

MIN_FILL_CELLS = 5


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _overlap_width(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> int:
    return max(0, min(a_hi, b_hi) - max(a_lo, b_lo))


def _shared_b_count(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> int:
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if lo > hi:
        return 0
    return (hi - lo) // 2 + 1


def _apply_fill(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        r_min, r_max = max(0, b), min(H - 1, (W - 1) + b)
        for (k_left, _, _), (k_right, _, _) in zip(active, active[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def _choose_k_values(parity: int) -> list[int] | None:
    num_ks = random.choices(NUM_KS_CHOICES, weights=NUM_KS_WEIGHTS)[0]
    if num_ks == 4:
        valid_combos = [(a, b, c) for a in GAP_VALUES for b in GAP_VALUES for c in GAP_VALUES
                        if a + b + c in GAP_TARGET_SPANS]
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
    overlap_ref: tuple[int, int] | None = None,
    outside_span: tuple[int, int] | None = None,
    outside_side: str | None = None,
) -> tuple[int, int] | None:
    """
    Sample a contiguous b-range on anti-diagonal k.

    overlap_ref: if set, require old-task overlap width >= OVERLAP_MIN_B with that segment.
    outside_span + outside_side: if set, keep the whole segment strictly above or below a band,
    creating a deliberate no-overlap boundary.
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

        if overlap_ref is not None:
            ref_lo, ref_hi = overlap_ref
            candidates = [
                bm for bm in candidates
                if _overlap_width(bm, bm + step, ref_lo, ref_hi) >= OVERLAP_MIN_B
            ]

        if outside_span is not None and outside_side is not None:
            span_lo, span_hi = outside_span
            if outside_side == "above":
                candidates = [bm for bm in candidates if bm >= span_hi + DISJOINT_GAP_MIN_B]
            else:
                candidates = [bm for bm in candidates if bm + step <= span_lo - DISJOINT_GAP_MIN_B]

        if not candidates:
            continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _build_overlapping_group(
    ks: list[int],
    outside_span: tuple[int, int] | None = None,
    outside_side: str | None = None,
) -> list[tuple[int, int, int]] | None:
    segs: list[tuple[int, int, int]] = []
    first = _random_seg(ks[0], outside_span=outside_span, outside_side=outside_side)
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1]))

    for k in ks[1:]:
        prev = segs[-1][1], segs[-1][2]
        nxt = _random_seg(k, overlap_ref=prev, outside_span=outside_span, outside_side=outside_side)
        if nxt is None:
            return None
        segs.append((k, nxt[0], nxt[1]))
    return segs


def _group_span(segs: list[tuple[int, int, int]]) -> tuple[int, int]:
    return min(s[1] for s in segs), max(s[2] for s in segs)


def _build_base_segments(ks: list[int]) -> tuple[list[tuple[int, int, int]], dict[str, object]] | None:
    split_idx = 2 if len(ks) == 4 else random.choice([2, 3])
    group1_ks = ks[:split_idx]
    group2_ks = ks[split_idx:]

    group1 = _build_overlapping_group(group1_ks)
    if group1 is None:
        return None
    span1 = _group_span(group1)

    for side in random.sample(["above", "below"], k=2):
        group2 = _build_overlapping_group(group2_ks, outside_span=span1, outside_side=side)
        if group2 is None:
            continue

        base = group1 + group2
        shared_counts = [
            _shared_b_count(base[i][1], base[i][2], base[i + 1][1], base[i + 1][2])
            for i in range(len(base) - 1)
        ]
        zero_overlap_pairs = [i for i, count in enumerate(shared_counts) if count == 0]
        if zero_overlap_pairs != [split_idx - 1]:
            continue
        if not any(count > 0 for count in shared_counts):
            continue

        meta = {
            "split_idx": split_idx,
            "group_boundary_pair": split_idx - 1,
            "group2_side": side,
            "shared_b_counts": shared_counts,
            "zero_overlap_pairs": zero_overlap_pairs,
        }
        return base, meta

    return None


def _gen_attempt(return_meta: bool = False):
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)
    ks = _choose_k_values(parity)
    if ks is None:
        return None

    built = _build_base_segments(ks)
    if built is None:
        return None
    segs, meta = built

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            grid[r, c] = line

    output = _apply_fill(grid, segs)
    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        return None

    inp = output.copy()
    inp[inp == FILL] = BG
    if not np.array_equal(main(inp), output):
        return None

    if return_meta:
        meta = dict(meta)
        meta["ks"] = ks
        meta["segments"] = segs
        meta["fill_cells"] = int(np.sum(output == FILL))
        return inp, meta
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
