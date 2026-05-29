# Variant of 7666fa5d
# Dimension: D8 — 多步处理：填充顺序与迭代 (multi_step)
# Change: 单步处理——对每条主对角线独立扫描，收集活跃线段后一次性填充所有区间
#         → 按线段距离顺序处理；同一条主对角线上考虑所有同时活跃的线段对，先填最近的，
#         再填更远的，并给不同距离层分配不同颜色。

"""
Design analysis for the D8 variant
==================================

variant_spec:
  preserved_structure:
    - grid remains 16x16
    - background remains 8
    - input still contains one non-background line color forming anti-diagonal segments (r+c=k)
    - filling still happens by scanning main diagonals (r-c=b)
  changed_rule:
    - original: for each b, sort active segments by k and fill only between consecutive active pairs using one fill color
    - variant: for each b, consider every simultaneously active segment pair
    - group those pairs by their k-distance; process smaller distances first
    - later stages only paint cells that are still background, so order matters
    - each distance stage gets its own fill color
  concrete_visual_behavior:
    - narrow gaps between nearby parallel segments become one color
    - medium and wide gaps appear in other colors when inner segments are absent on some b-ranges
    - the output therefore shows several colored parallelogram bands instead of a single red region
  traced_example:
    - suppose active k values on one b are [8, 12, 20, 24]
    - pair distances are 4, 8, 12, 16
    - stage 1 (distance 4): fills immediate bands (8,12) and (20,24)
    - stage 2 (distance 8): fills the central band (12,20)
    - stage 3 (distance 12): only appears on b where one inner segment is inactive, so (12,24) or (8,20) becomes adjacent in the active set and can still claim remaining background
    - stage 4 (distance 16): only appears on edge b where only the two outer segments remain active
    - difference from original: original logic would color every reachable interior cell red in one pass; the variant keeps the same geometry but exposes the width/order structure by staged colors
  complexity_calibration:
    - using exactly 4 anti-diagonals keeps the object class recognizable while still allowing 4/8/12/16 distance stages
    - fixed color order avoids an underconstrained color mapping
    - staggered segment lengths make farther-pair stages visible instead of being fully hidden by nearer pairs
  line_by_line_assumption_review:
    - extracting 8-connected components is still valid because each segment remains a contiguous anti-diagonal line
    - inferring each segment as one constant-k component is still required; generator preserves that
    - the old solver assumed "adjacent pairs are sufficient"; this is false now, so the fill loop must enumerate all active pairs
    - the old unconditional fill color "2" would destroy multi-stage information; the new solver must map distance stage -> color
    - the old logic was harmlessly single-pass because all fills had the same color; now fill order matters because later stages may only use cells left as background
  placement_legality:
    - generator chooses four k-values with total span 16 (gaps are a permutation of 4,4,8)
    - it then finds a common legal b-window shared by all four anti-diagonals
    - outer segments use the full window
    - two inner segments are shortened and shifted so that active sets change across b:
      * edge b: only outer pair active -> farthest stage visible
      * near-edge b: one inner + outers active -> 12-distance stage visible
      * center b: all four active -> 4- and 8-distance stages visible
    - this guarantees no overlap, no out-of-bounds cells, and compatibility with the solver
  visual_observability:
    - the change is visible directly: outputs contain multiple filled colors arranged by gap width
    - color assignment depends on geometric distance, not on hidden numeric color ids
    - each generated example is required to use all four stage colors, so the rule is learnable from a small sample set
  visual_cleanliness:
    - only one line color appears in the input
    - fill colors are reserved and never reused as line colors
    - bands are diagonal and non-noisy; there is no marker clutter
  color_inference:
    - at most four fill colors are used
    - they always correspond to increasing pair distance within the same puzzle
    - generator enforces appearance of every distance stage, preventing collapse back to the original single-color behavior
  new_solver_assumptions_and_generator_guarantees:
    - solver assumes the number of distance stages is at most 4; generator enforces pair distances [4, 8, 12, 16]
    - solver assumes segment pairs matter only when their b-ranges overlap; generator explicitly builds overlapping ranges for the intended stages
    - solver assumes line color is the only non-bg color in the input; generator preserves that
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

ROOT = "/home/lixinyu/snap/new"
for _path in (ROOT, os.path.join(ROOT, "solvers")):
    if _path not in sys.path:
        sys.path.append(_path)

from arc_utils import components_mask, get_bg as _bg_color


H = W = 16
BG = 8
LINE_COLORS = [1, 3, 5, 9]
FILL_COLORS = [2, 4, 6, 7]
K_LO, K_HI = 4, 26
K_GAP_PATTERNS = [(4, 4, 8), (4, 8, 4), (8, 4, 4)]
MASTER_SPANS = (14, 12)
GEN_RETRY = 300


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _extract_segments(grid: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(grid == line):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _distance_palette(segs: list[tuple[int, int, int]]) -> dict[int, int]:
    distances = sorted(
        {
            segs[j][0] - segs[i][0]
            for i in range(len(segs))
            for j in range(i + 1, len(segs))
            if max(segs[i][1], segs[j][1]) <= min(segs[i][2], segs[j][2])
        }
    )
    assert 1 <= len(distances) <= len(FILL_COLORS), "unsupported number of distance stages"
    return {dist: FILL_COLORS[idx] for idx, dist in enumerate(distances)}


def _apply_fill(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    dist_to_color = _distance_palette(segs)

    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue

        pairs_by_dist: dict[int, list[tuple[int, int]]] = {}
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                dist = active[j][0] - active[i][0]
                pairs_by_dist.setdefault(dist, []).append((active[i][0], active[j][0]))

        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for dist in sorted(pairs_by_dist):
            color = dist_to_color[dist]
            for k_left, k_right in pairs_by_dist[dist]:
                for r in range(r_min, r_max + 1):
                    c = r - b
                    k = r + c
                    if k_left < k < k_right and out[r, c] == BG:
                        out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g, line)
    assert segs, "expected at least one anti-diagonal segment"
    return _apply_fill(g, segs)


def _choose_layout() -> tuple[list[int], int, int] | None:
    candidates: list[tuple[list[int], int, int]] = []
    parity = random.randint(0, 1)

    for gaps in K_GAP_PATTERNS:
        total_span = sum(gaps)
        for k0 in range(K_LO, K_HI - total_span + 1):
            if k0 % 2 != parity:
                continue
            ks = [k0]
            for gap in gaps:
                ks.append(ks[-1] + gap)

            common_lo = max(_b_bounds_for_k(k)[0] for k in ks)
            common_hi = min(_b_bounds_for_k(k)[1] for k in ks)
            if common_lo % 2 != parity:
                common_lo += 1
            if common_hi % 2 != parity:
                common_hi -= 1
            if common_lo > common_hi:
                continue

            for master_span in MASTER_SPANS:
                start_max = common_hi - master_span
                if start_max < common_lo:
                    continue
                starts = list(range(common_lo, start_max + 1, 2))
                for master_lo in starts:
                    candidates.append((ks, master_lo, master_span))

    if not candidates:
        return None
    return random.choice(candidates)


def _build_segments(ks: list[int], master_lo: int, master_span: int) -> list[tuple[int, int, int]]:
    hi = master_lo + master_span
    if random.random() < 0.5:
        inner1 = (master_lo + 2, hi - 4)
        inner2 = (master_lo + 4, hi - 2)
    else:
        inner1 = (master_lo + 4, hi - 2)
        inner2 = (master_lo + 2, hi - 4)

    return [
        (ks[0], master_lo, hi),
        (ks[1], inner1[0], inner1[1]),
        (ks[2], inner2[0], inner2[1]),
        (ks[3], master_lo, hi),
    ]


def _valid_segments(segs: list[tuple[int, int, int]]) -> bool:
    for k, b_min, b_max in segs:
        if b_min > b_max:
            return False
        lo, hi = _b_bounds_for_k(k)
        if b_min < lo or b_max > hi:
            return False
        if (b_min - k) % 2 or (b_max - k) % 2:
            return False
        if len(_seg_cells(k, b_min, b_max)) < 2:
            return False
    return True


def _stage_color_set(grid: np.ndarray) -> set[int]:
    return {int(c) for c in np.unique(grid) if int(c) in FILL_COLORS}


def _gen_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None

    ks, master_lo, master_span = layout
    segs = _build_segments(ks, master_lo, master_span)
    if not _valid_segments(segs):
        return None

    line = random.choice(LINE_COLORS)
    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    output = _apply_fill(grid, segs)
    if _stage_color_set(output) != set(FILL_COLORS):
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate D8 variant input")
