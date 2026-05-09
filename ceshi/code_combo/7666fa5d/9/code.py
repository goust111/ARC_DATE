# Combo variant of 7666fa5d (index 9)
# combo_spec: D9/2+D10/0+D13/0
# Name: 最小间距阈值+单线段+无重叠约束

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for combo D9/2 + D10/0 + D13/0
# a) Concrete behavior and trace:
#    The visible objects remain short line segments on anti-diagonals k=r+c, and red fill still grows
#    along main diagonals b=r-c. The fused change adds two independent gates:
#    1) every k is used by at most one contiguous segment, so each rail is visually clean;
#    2) a neighboring pair can fill only if it is simultaneously active on some b and its k-gap is
#       strictly larger than 4.
#    Small example with four segments:
#      k=6,  b in [-2, 4]
#      k=10, b in [ 0, 6]
#      k=16, b in [10, 14]
#      k=22, b in [10, 16]
#    Original task: [6,10] would fill on shared b values, because any overlapping neighboring pair
#    was fillable. Fused task:
#      - [6,10] shares b values but gap=4, so it stays blank.
#      - [10,16] never shares any b, so it also stays blank.
#      - [16,22] shares b values and gap=6, so that strip fills red.
#    The user therefore sees both kinds of "no fill": too close, or not co-active.
# b) Complexity calibration:
#    The threshold is fixed at one simple cutoff (>4), and k-gaps are sampled from the small set
#    {4,6,8}. Each puzzle is forced to contain one close overlapping pair, one far overlapping pair,
#    and one neighboring zero-overlap boundary. That gives repeated support for the rule without
#    introducing many free parameters.
# c) Hidden-assumption audit of main():
#    - `line = max(non_bg, key=count)` still assumes one foreground line color; generator keeps that.
#    - connected components are assumed to be single anti-diagonal segments; D10/0 enforces exactly one
#      contiguous segment per used k, so extraction remains valid.
#    - sorting active segments by k and zipping neighboring pairs is still the right geometric notion,
#      but the original "always fill neighbors" assumption is now false. The solver must additionally
#      test `k_right-k_left > 4`.
#    - no global overlap promise remains. `active = [s for s in segs if s[1] <= b <= s[2]]` becomes the
#      only correct overlap test: disjoint neighboring segments simply never appear together.
# d) Placement legality:
#    Generator samples every segment from in-bounds (k,b) coordinates, uses one segment per k, and
#    splits the four k-values into two groups. The second group is placed entirely above or below the
#    first group's total b-span, guaranteeing exactly one zero-overlap neighboring pair and preventing
#    accidental cross-group fill.
# e) Visual observability:
#    In 2-4 examples a human can compare three local situations with the same shape vocabulary:
#    close+overlap -> blank, far+overlap -> red band, no-overlap -> blank. No hidden ordering by color
#    value is used; only visible spacing and visible shared span matter.
# f) Visual cleanliness:
#    Same-k splitting is removed, so every anti-diagonal shows a single clean line segment. Keeping only
#    four rails avoids clutter while still making the two suppression mechanisms visible.
# g) New solver assumptions and generator guarantees:
#    The new solver assumes useful tasks contain at least one far overlapping pair and at least one
#    close overlapping pair. Generator validates this with two quantitative checks:
#    `fill_cells > 0` for the far pair, and `suppressed_cells > 0` compared with the no-threshold
#    legacy fill, so the minimum-gap rule cannot silently disappear.


MIN_FILL_GAP = 4


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return each connected anti-diagonal segment as (k, b_min, b_max)."""
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(mask):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    segs.sort(key=lambda t: (t[0], t[1], t[2]))
    return segs


def _shared_b_count(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> int:
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if lo > hi:
        return 0
    return (hi - lo) // 2 + 1


def _apply_fill(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    *,
    bg: int,
    fill: int,
    min_gap: int,
) -> np.ndarray:
    """Fill between neighboring active anti-diagonals whose k-gap is strictly larger than min_gap."""
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape

    for b in range(-(w - 1), h):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: (t[0], t[1], t[2]))
        if len(active) < 2:
            continue

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        if r_min > r_max:
            continue

        for (k_left, _a0, _a1), (k_right, _b0, _b1) in zip(active, active[1:]):
            if k_right - k_left <= min_gap:
                continue
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill red strips between anti-diagonal segments, but only when a neighboring pair is both far
    enough apart and simultaneously active on the same main diagonal.

    Input structure:
    - The grid is 16x16 with one dominant background color and one foreground line color.
    - Each foreground component is a single contiguous segment on one anti-diagonal k=r+c.
    - There is at most one segment on any given k, and neighboring k-values may either overlap in
      b=r-c or be fully disjoint.

    Rule:
    - Convert each connected line segment into its anti-diagonal index k and its covered b-range.
    - For every main diagonal b, keep only the segments whose b-range contains that b and sort them
      by k.
    - Fill background cells red only between consecutive active segments whose k-gap is greater than
      4; close pairs or disjoint pairs produce no red strip.

    Key insight:
    - Missing fill now has two visible causes. Some neighboring rails never co-occur on the same
      b-line, while other neighboring rails do overlap but are still too close together. Only pairs
      that pass both tests generate a parallelogram band.
    """
    g = np.asarray(grid, dtype=int)

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g == line)
    return _apply_fill(g, segs, bg=bg, fill=2, min_gap=MIN_FILL_GAP)


# --------------- generator ---------------

H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
K_GAP_PATTERNS = [
    (4, 6, 6),
    (6, 6, 4),
    (4, 8, 6),
    (6, 8, 4),
    (4, 6, 8),
    (8, 6, 4),
]
BOUNDARY_PAIR_IDX = 1

SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
MIN_SHARED_B = 3
DISJOINT_GAP_MIN_B = 2

SEG_RETRY = 80
GEN_RETRY = 600

MIN_FILL_CELLS = 4
MIN_SUPPRESSED_CELLS = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive b-range such that ((k+b)/2, (k-b)/2) stays inside the grid."""
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _choose_k_layout(parity: int) -> tuple[list[int], tuple[int, int, int]] | None:
    """Pick four anti-diagonals with one close pair, one far boundary, and one far/close pair."""
    gaps = random.choice(K_GAP_PATTERNS)
    total_span = sum(gaps)
    starts = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not starts:
        return None

    k0 = random.choice(starts)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    if ks[-1] > K_HI:
        return None
    return ks, gaps


def _random_seg(
    k: int,
    *,
    overlap_ref: tuple[int, int] | None = None,
    outside_span: tuple[int, int] | None = None,
    outside_side: str | None = None,
) -> tuple[int, int] | None:
    """
    Sample one contiguous segment on anti-diagonal k.

    overlap_ref forces a visible shared b-span with a previous segment.
    outside_span/outside_side places the segment wholly above or below a prior group's b-span.
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
                bm
                for bm in candidates
                if _shared_b_count(bm, bm + step, ref_lo, ref_hi) >= MIN_SHARED_B
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
    *,
    outside_span: tuple[int, int] | None = None,
    outside_side: str | None = None,
) -> list[tuple[int, int, int]] | None:
    """Build one contiguous two-segment group whose neighboring pair visibly overlaps in b."""
    first = _random_seg(ks[0], outside_span=outside_span, outside_side=outside_side)
    if first is None:
        return None

    second = _random_seg(
        ks[1],
        overlap_ref=first,
        outside_span=outside_span,
        outside_side=outside_side,
    )
    if second is None:
        return None

    return [(ks[0], first[0], first[1]), (ks[1], second[0], second[1])]


def _group_span(segs: list[tuple[int, int, int]]) -> tuple[int, int]:
    return min(s[1] for s in segs), max(s[2] for s in segs)


def _neighbor_meta(segs: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    pairs: list[dict[str, int]] = []
    for i in range(len(segs) - 1):
        k0, a0, a1 = segs[i]
        k1, b0, b1 = segs[i + 1]
        pairs.append(
            {
                "idx": i,
                "gap": k1 - k0,
                "shared_b": _shared_b_count(a0, a1, b0, b1),
            }
        )
    return pairs


def _gen_attempt(return_meta: bool = False):
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)
    layout = _choose_k_layout(parity)
    if layout is None:
        return None
    ks, gaps = layout

    group1 = _build_overlapping_group(ks[:2])
    if group1 is None:
        return None
    span1 = _group_span(group1)

    segs: list[tuple[int, int, int]] | None = None
    side_used = None
    for side in random.sample(["above", "below"], k=2):
        group2 = _build_overlapping_group(ks[2:], outside_span=span1, outside_side=side)
        if group2 is None:
            continue
        candidate = group1 + group2
        pair_meta = _neighbor_meta(candidate)
        zero_overlap_pairs = [p["idx"] for p in pair_meta if p["shared_b"] == 0]
        if zero_overlap_pairs != [BOUNDARY_PAIR_IDX]:
            continue

        near_overlap = [p for p in pair_meta if p["gap"] <= MIN_FILL_GAP and p["shared_b"] >= MIN_SHARED_B]
        far_overlap = [p for p in pair_meta if p["gap"] > MIN_FILL_GAP and p["shared_b"] >= MIN_SHARED_B]
        if not near_overlap or not far_overlap:
            continue

        segs = candidate
        side_used = side
        break

    if segs is None or side_used is None:
        return None

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    output = _apply_fill(grid, segs, bg=BG, fill=FILL, min_gap=MIN_FILL_GAP)
    legacy_output = _apply_fill(grid, segs, bg=BG, fill=FILL, min_gap=-1)

    fill_cells = int(np.sum(output == FILL))
    suppressed_cells = int(np.sum((legacy_output == FILL) & (output != FILL)))
    if fill_cells < MIN_FILL_CELLS or suppressed_cells < MIN_SUPPRESSED_CELLS:
        return None

    inp = output.copy()
    inp[inp == FILL] = BG
    if not np.array_equal(main(inp), output):
        return None

    if return_meta:
        pair_meta = _neighbor_meta(segs)
        return inp, {
            "ks": ks,
            "gaps": gaps,
            "segments": segs,
            "group2_side": side_used,
            "pair_meta": pair_meta,
            "fill_cells": fill_cells,
            "suppressed_cells": suppressed_cells,
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
