# Variant of 7666fa5d
# Dimension: D1 — 匹配条件：投影配对机制 (match_cond)
# Change: s.left_projection_point == d — 右侧对角线段的端点沿主对角线方向投影到左侧平行线段上，配对... → 线段长度排序匹配

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the D1 variant
# ==================================
# a) Concrete behavior and trace
#    Original task:
#    - On each main diagonal b=r-c, collect every anti-diagonal segment covering that b.
#    - Sort those active segments by k=r+c, then fill between neighboring k-values.
#    Variant:
#    - Ignore local k-order during pairing.
#    - First extract all segments globally, sort them by segment length descending
#      (ties by smaller k), and only consider adjacent pairs in this global ranking.
#    - For each ranked pair, fill along the overlap of their b-ranges and between their
#      two k-values.
#    Example with 4 segments in k-order S0,S1,S2,S3 and lengths 8,3,6,5:
#    - Original active-pair logic on a low b-band fills S0-S1 because they are adjacent in k.
#    - Variant global length order is S0,S2,S3,S1, so the low band instead fills S3-S1,
#      while S0-S1 may remain empty even when both are visible.
#    The visible cue is therefore "which gap turns red" rather than "whether overlap exists".
#
# b) Complexity calibration
#    - Keep 4 segments only: enough for three ranked adjacencies, but still readable.
#    - Use distinct lengths most of the time so the length order is visually inferable.
#    - Constrain k-gaps to 4 so components stay separated yet the red regions remain compact.
#    - Use two mirrored b-range templates; this avoids all samples looking identical while
#      keeping the pairing logic stable and observable.
#
# c) Old implicit assumptions that break
#    - Original solver loop:
#        active.sort(key=lambda t: t[0])
#        for k_left, k_right in zip(ks, ks[1:]):
#      assumes "the correct pair on each b-line is the local k-neighbor pair".
#      That is false in this variant.
#    - Original code also scans every b independently, which is harmless only because the
#      pairing criterion itself is local-in-b. The new criterion is global, so we must sort
#      segments once globally and iterate ranked pairs instead.
#    - We still rely on segments being single constant-k connected components; generator
#      preserves that by never splitting components and by spacing distinct k-values by 4.
#
# d) Placement legality
#    - All segments share one parity, so any overlapping b-band corresponds to valid cells on
#      both segments.
#    - k-values differ by 4, preventing 8-connectivity mergers between different segments.
#    - Each template is translated only by parity-preserving deltas that keep every b-range
#      inside the legal bounds for its k.
#
# e) Visual observability
#    - Degenerate scenes where all segments overlap on the same b-lines are rejected because
#      any pairing chain would then cover nearly the same union.
#    - Generator explicitly requires the new fill to differ from the original k-neighbor fill
#      by many cells, so the D1 change is visible in the output.
#
# f) Visual cleanliness
#    - Only one line color plus red fill is used, matching the original task's palette.
#    - Four segments and three fill regions keep the scene structured; there are no same-k
#      duplicate components, so there is no ambiguity about what counts as one segment.
#
# g) Color inference
#    - No color-dependent sorting is used. Length is the only ranking key, with k only as a
#      geometric tie-breaker when two lengths match.
#
# 9) New solver assumptions audited
#    - Every relevant component is monochrome and lies on exactly one anti-diagonal.
#      Generator guarantees this.
#    - Ranked-adjacent pairs must overlap in b often enough to produce red cells.
#      Generator templates guarantee all three ranked pairs have non-empty overlap.
#    - The variant should not silently collapse back to the original behavior.
#      Generator compares the new fill against the original k-ordered fill and rejects equal
#      or near-equal cases.


H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]
K_LO, K_HI = 4, 26
GEN_RETRY = 200
MIN_FILL_CELLS = 12
MIN_DIFF_FROM_ORIGINAL = 20

# In k-order, the four segments have lengths 8,3,6,5. After length sort they become
# S0,S2,S3,S1, so the ranked-adjacent pairs differ from the original k-adjacent pairs.
BASE_TEMPLATES = [
    [(-6, 8), (-6, -2), (2, 12), (-4, 4)],
    [(-8, 6), (2, 6), (-12, -2), (-4, 4)],
]


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return int(b_lo), int(b_hi)


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _seg_len(seg: tuple[int, int, int]) -> int:
    _k, b_min, b_max = seg
    return int((b_max - b_min) // 2 + 1)


def _extract_segments(g: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(g == line):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _fill_by_length_order(grid: np.ndarray, segs: list[tuple[int, int, int]], bg: int, fill: int) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    ordered = sorted(segs, key=lambda s: (-_seg_len(s), s[0]))
    for s_a, s_b in zip(ordered, ordered[1:]):
        k_a, a_lo, a_hi = s_a
        k_b, b_lo, b_hi = s_b
        overlap_lo = max(a_lo, b_lo)
        overlap_hi = min(a_hi, b_hi)
        if overlap_lo > overlap_hi:
            continue

        parity = k_a % 2
        if overlap_lo % 2 != parity:
            overlap_lo += 1

        k_lo, k_hi = sorted((k_a, k_b))
        for b in range(overlap_lo, overlap_hi + 1, 2):
            r_min = max(0, b)
            r_max = min(h - 1, (w - 1) + b)
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_lo < k < k_hi and out[r, c] == bg:
                    out[r, c] = fill
    return out


def _fill_by_k_order(grid: np.ndarray, segs: list[tuple[int, int, int]], bg: int, fill: int) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda s: s[0])
        for s_a, s_b in zip(active, active[1:]):
            k_lo, k_hi = sorted((s_a[0], s_b[0]))
            r_min = max(0, b)
            r_max = min(h - 1, (w - 1) + b)
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_lo < k < k_hi and out[r, c] == bg:
                    out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g, line)
    assert segs, "expected at least one anti-diagonal segment"

    return _fill_by_length_order(g, segs, bg=bg, fill=FILL)


def _candidate_layouts() -> list[list[tuple[int, int, int]]]:
    layouts: list[list[tuple[int, int, int]]] = []
    for base in BASE_TEMPLATES:
        for parity in (0, 1):
            parity_ranges = [(a + parity, b + parity) for a, b in base]
            for k0 in range(K_LO, K_HI - 12 + 1):
                if k0 % 2 != parity:
                    continue
                ks = [k0, k0 + 4, k0 + 8, k0 + 12]
                delta_lo, delta_hi = -99, 99
                for k, (b_min, b_max) in zip(ks, parity_ranges):
                    legal_lo, legal_hi = _b_bounds_for_k(k)
                    delta_lo = max(delta_lo, legal_lo - b_min)
                    delta_hi = min(delta_hi, legal_hi - b_max)
                deltas = [d for d in range(delta_lo, delta_hi + 1) if d % 2 == 0]
                for delta in deltas:
                    segs = [(k, b_min + delta, b_max + delta) for k, (b_min, b_max) in zip(ks, parity_ranges)]
                    layouts.append(segs)
    return layouts


LAYOUTS = _candidate_layouts()


def _variant_metrics(inp: np.ndarray, segs: list[tuple[int, int, int]]) -> dict[str, int | tuple[int, ...]]:
    new_out = _fill_by_length_order(inp, segs, bg=BG, fill=FILL)
    old_out = _fill_by_k_order(inp, segs, bg=BG, fill=FILL)
    length_order = tuple(s[0] for s in sorted(segs, key=lambda s: (-_seg_len(s), s[0])))
    k_order = tuple(s[0] for s in sorted(segs, key=lambda s: s[0]))
    return {
        "fill_cells": int(np.sum(new_out == FILL)),
        "diff_cells": int(np.sum(new_out != old_out)),
        "length_order_ks": length_order,
        "k_order_ks": k_order,
    }


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    segs = [tuple(seg) for seg in random.choice(LAYOUTS)]

    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    out = _fill_by_length_order(grid, segs, bg=BG, fill=FILL)
    metrics = _variant_metrics(grid, segs)

    if metrics["fill_cells"] < MIN_FILL_CELLS:
        return None
    if metrics["diff_cells"] < MIN_DIFF_FROM_ORIGINAL:
        return None
    if metrics["length_order_ks"] == metrics["k_order_ks"]:
        return None
    if np.array_equal(out, _fill_by_k_order(grid, segs, bg=BG, fill=FILL)):
        return None
    if not np.array_equal(main(grid), out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
