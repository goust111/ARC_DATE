# Combo variant of 7666fa5d (index 7)
# combo_spec: D1/2+D3/2
# Name: 端点推断长度排序配对

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import get_bg as _bg_color


"""
design_analysis:
  variant_goal:
    preserve:
      - 16x16 canvas
      - one visible non-background line color plus red fill
      - hidden geometry still consists of anti-diagonal segments on constant k=r+c
      - fill still runs along main diagonals b=r-c between paired hidden segments
    change:
      - each segment is no longer drawn as a contiguous stroke
      - only the two endpoints of each segment remain visible as singleton pixels
      - segment pairing is no longer local k-neighbor order
      - instead, infer every hidden segment length from its two endpoints, sort lengths
        descending, tie-break by smaller k, then fill only between ranked-adjacent pairs

  concrete_visual_form:
    input:
      - background is 8
      - one line color marks exactly eight singleton pixels
      - the pixels come in four same-color pairs, one pair per k=r+c level
      - within each k-level the two pixels are the hidden segment endpoints
    output:
      - the singleton endpoints remain unchanged
      - red parallelogram bands appear only between length-ranked adjacent hidden segments
      - no hidden segment body is ever drawn; the solver must reason from absent structure

  behavior_trace:
    example:
      - four hidden segments appear on k-order [k0,k1,k2,k3]
      - inferred lengths are [8,3,6,5]
      - original task would use local k pairs:
        (k0,k1), (k1,k2), (k2,k3)
      - fused variant first sorts by length to get:
        [k0(len8), k2(len6), k3(len5), k1(len3)]
      - it therefore fills only:
        (k0,k2), (k2,k3), (k3,k1)
      - the red bands visibly shift away from the original left-to-right k adjacency
    difference:
      - step 1 changed from "connected component = segment" to "two points on one k imply
        an unseen segment spanning their b-extremes"
      - step 2 changed from per-b local pairing to one global length ranking
      - later reasoning depends on earlier inference, so this is a true chained fusion

  complexity_calibration:
    choices:
      - keep exactly four hidden segments
      - use two mirrored templates whose k-order lengths are always 8,3,6,5
      - keep all segments on one parity class and space k-values by 4
      - show only one visible line color, avoiding color-coded pairing shortcuts
    reason:
      - four segments are enough to make length order differ from k order in a stable way
      - fixed distinct lengths remove tie ambiguity in most samples while still allowing
        translation and mirroring for variation
      - grouping by shared k keeps endpoint pairing human-visible without extra markers

  implicit_assumption_audit_of_original_main:
    - comps = _cc8(g == line):
        old_assumption: each visible component is a full segment
        broken_now: every visible object is a singleton endpoint, so connectivity says nothing
        about the hidden segment extent
    - ks = {r + c for r, c in pts}; bs = [r - c ...]:
        old_assumption: one component already contains the full b-span
        broken_now: the b-span must be reconstructed by pairing the two pixels on the same k
    - active.sort(key=lambda t: t[0]):
        old_assumption: the correct pair for fill is the local k-neighbor pair on each b-line
        broken_now: pairing is global and based on inferred length, not local geometric order
    - scan every b and fill all adjacent k gaps:
        old_assumption: every neighboring active pair is eligible
        broken_now: only globally ranked-adjacent hidden segments may generate fill

  placement_legality:
    - each hidden segment is stored as (k,b_min,b_max) with exact grid-bound checks
    - all k-values share one parity and differ by 4, so endpoint pixels never overlap or merge
    - every k level has exactly two endpoints, preventing ambiguous within-k pairing
    - ranked-adjacent pairs are generated with non-empty b-overlap, so intended fill exists

  visual_observability:
    - humans can see that two pixels on one anti-diagonal imply a hidden diagonal span because
      red appears across the whole span between those endpoint rows, not just near the dots
    - across examples, the gap that turns red tracks endpoint distance, not k-order adjacency
    - generator rejects cases where length-order fill matches the original k-ordered fill, so
      the fusion cannot silently collapse to the source task

  cleanliness:
    - only three colors appear: bg, endpoint color, fill color
    - there are no extra legends or distractor markers
    - endpoint pixels are sparse and well-separated, so the four hidden segments remain legible

  new_solver_assumptions_and_generator_support:
    - solver assumes every useful k-level contains exactly two same-color endpoint pixels
    - solver assumes each inferred segment length is at least 2 cells and lies fully in-bounds
    - solver assumes ranked-adjacent segments can be compared by integer length; generator uses
      templates with distinct lengths to keep that inference visible
    - solver assumes the intended variant differs from local k-order; generator measures that
      difference directly and rejects degenerate scenes
"""


H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]
K_LO, K_HI = 4, 26
GEN_RETRY = 300
MIN_FILL_CELLS = 12
MIN_DIFF_FROM_ORIGINAL = 20

# In k-order, these four segments have lengths 8,3,6,5, so the global
# length ranking is guaranteed to differ from the original local k order.
BASE_TEMPLATES = [
    [(-6, 8), (-6, -2), (2, 12), (-4, 4)],
    [(-8, 6), (2, 6), (-12, -2), (-4, 4)],
]


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return int(b_lo), int(b_hi)


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _seg_len(seg: tuple[int, int, int]) -> int:
    _k, b_min, b_max = seg
    return int((b_max - b_min) // 2 + 1)


def _extract_hidden_segments(g: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    """
    Reconstruct hidden segments from endpoint pairs.

    Valid input for this fusion has exactly two line-colored pixels on each used
    anti-diagonal k=r+c; their b-extremes define the hidden segment span.
    """
    by_k: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == line):
        rr, cc = int(r), int(c)
        by_k.setdefault(rr + cc, []).append((rr, cc))

    segs: list[tuple[int, int, int]] = []
    for k, pts in by_k.items():
        if len(pts) != 2:
            continue
        bs = sorted(int(r - c) for r, c in pts)
        if bs[0] == bs[1]:
            continue
        segs.append((int(k), int(bs[0]), int(bs[1])))
    return segs


def _fill_by_length_order(
    grid: np.ndarray, segs: list[tuple[int, int, int]], bg: int, fill: int
) -> np.ndarray:
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

        parity = a_lo & 1
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


def _fill_by_k_order(
    grid: np.ndarray, segs: list[tuple[int, int, int]], bg: int, fill: int
) -> np.ndarray:
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
    """
    Fill red bands between hidden anti-diagonal segments inferred only from endpoint markers.

    Input structure:
    - bg is the most common color.
    - One non-bg color appears only as isolated endpoint pixels.
    - For each used anti-diagonal k=r+c, exactly two such pixels mark the endpoints of one
      hidden segment, whose span runs continuously in b=r-c between them.

    Rule:
    - Reconstruct every hidden segment from the two endpoint pixels on the same k value.
    - Infer each hidden segment length from its endpoint distance and sort segments by length
      descending, breaking ties by smaller k.
    - For every adjacent pair in that ranking, fill bg cells with red (2) on the overlap of
      their b-ranges and strictly between their two k values.

    Key insight: The visible dots are not the objects to connect directly; they encode unseen
    diagonal spans, and only after reconstructing and length-sorting those spans do the correct
    fill bands become determined.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_hidden_segments(g, line)
    assert segs, "expected hidden segments encoded by endpoint pairs"

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
                    segs = [
                        (k, b_min + delta, b_max + delta)
                        for k, (b_min, b_max) in zip(ks, parity_ranges)
                    ]
                    layouts.append(segs)
    return layouts


LAYOUTS = _candidate_layouts()


def _render_endpoint_input(segs: list[tuple[int, int, int]], line: int) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        cells = _seg_cells(k, b_min, b_max)
        if len(cells) < 2:
            raise ValueError("hidden segment must have two distinct endpoints")
        for r, c in (cells[0], cells[-1]):
            if grid[r, c] != BG:
                raise ValueError("endpoint overlap")
            grid[r, c] = line
    return grid


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
        "endpoint_pixels": int(np.sum(inp != BG)),
    }


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    segs = [tuple(seg) for seg in random.choice(LAYOUTS)]

    try:
        inp = _render_endpoint_input(segs, line)
    except ValueError:
        return None

    out = _fill_by_length_order(inp, segs, bg=BG, fill=FILL)
    metrics = _variant_metrics(inp, segs)

    if metrics["fill_cells"] < MIN_FILL_CELLS:
        return None
    if metrics["diff_cells"] < MIN_DIFF_FROM_ORIGINAL:
        return None
    if metrics["length_order_ks"] == metrics["k_order_ks"]:
        return None
    if metrics["endpoint_pixels"] != 8:
        return None
    if np.array_equal(out, _fill_by_k_order(inp, segs, bg=BG, fill=FILL)):
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
