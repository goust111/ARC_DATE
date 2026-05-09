# Variant of 7666fa5d
# Dimension: D1 — 匹配条件：投影配对机制 (match_cond)
# Change: s.left_projection_point == d — 右侧对角线段的端点沿主对角线方向投影到左侧平行线段上，配对... → 端点颜色匹配决定填充连接

import numpy as np
import random

from arc_utils import components_mask, get_bg as _bg_color


"""
design_analysis:
  variant_goal:
    preserve:
      - 16x16 grid
      - short anti-diagonal segments on constant k=r+c
      - red fill (2) drawn along main diagonals b=r-c
      - output-first generator with solver oracle check
    change:
      - segments are no longer all monochrome
      - each segment has a shared endpoint marker color on its two ends
      - fill is produced only between segments whose endpoint marker colors match

  concrete_visual_form:
    input:
      - background is 8
      - every segment is still a single 8-connected anti-diagonal component
      - interior cells of each segment use one shared body color
      - the two endpoint pixels of the segment are recolored to a marker color
      - marker colors appear in repeated pairs across the puzzle, e.g. A,A,B,B or A,A,B,B,C,C
    output:
      - same segments remain unchanged
      - red cells appear only inside the bands bounded by same-marker segment pairs
      - bands between adjacent but different-marker pairs stay background, making the D1 change visible

  behavior_trace:
    example:
      - active segments on some b-line are ordered by k as [A, A, B, B]
      - original logic would fill between all consecutive boundaries:
        (A1,A2), (A2,B1), (B1,B2)
      - new logic fills only the same-color pairs:
        (A1,A2) and (B1,B2)
      - the middle gap between A2 and B1 remains background
    effect:
      - pairing is no longer inferred from geometric adjacency alone
      - the solver must read endpoint colors and group segments by that visible attribute

  complexity_calibration:
    choices:
      - use 2 or 3 marker colors per puzzle; each marker color is repeated exactly twice
      - keep pairs contiguous in k-order as A,A,B,B,(C,C)
      - require overlap across every consecutive boundary, including the mismatched pair boundary
    reason:
      - repeated colors give enough support for humans to infer the rule
      - contiguous pairs keep the picture clean and avoid crossing/nested fills
      - forced overlap on mismatched boundaries guarantees that the new behavior differs from the original

  implicit_assumption_audit_of_original_main:
    - line = max(non_bg, ...):
        old_assumption: all segment pixels share one non-bg color
        broken_now: endpoints use extra colors, so "the line color" is no longer the whole object
    - comps = _cc8(g == line):
        old_assumption: thresholding by one color preserves whole segments
        broken_now: endpoint recoloring would chop endpoints off every segment
    - segs only store (k,b_min,b_max):
        old_assumption: adjacency in k is sufficient to decide all fills
        broken_now: pairing must carry marker-color identity
    - fill consecutive active pairs:
        old_assumption: every adjacent active pair should be connected
        broken_now: adjacent different-marker pairs must be skipped

  legality_and_generator_guarantees:
    - all segments use distinct k-values with even gaps >= 4, so separate components never touch
    - segment lengths are at least 4, so endpoint markers are visually distinct from the body
    - consecutive segments are generated with overlapping b-ranges, so intended fills exist
    - marker colors are distinct from bg, fill, and body color
    - output-first validation checks main(input) == expected_output
    - an extra non-degeneracy check enforces that the new output differs from the old adjacency-only fill

  visual_observability:
    - humans can spot that red appears only between same-colored endpoint pairs
    - the skipped middle band between different marker colors is the main cue
    - only a small marker palette is used, and each color repeats, so the color rule is learnable from 2-3 examples

  cleanliness:
    - no overlapping fills are generated
    - no split-on-same-k segments are used, because they add ambiguity without supporting D1
    - fills only overwrite background, so segments and markers remain legible

  new_solver_assumptions_to_check:
    - every meaningful segment is one 8-connected non-bg component on a single anti-diagonal
    - a valid segment has matching colors at its two b-extreme cells
    - marker colors are reused in pairs; the generator guarantees this
    - same-color matched segments have overlapping b-ranges on at least one b-line
"""


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(g: np.ndarray, bg: int, fill: int) -> list[tuple[int, int, int, int]]:
    """
    Return segments as (k, b_min, b_max, marker_color).

    Segment detection uses all non-bg/non-fill pixels as one object mask so endpoint recoloring
    does not split the segment into multiple pieces.
    """
    segs: list[tuple[int, int, int, int]] = []
    for pts in _cc8((g != bg) & (g != fill)):
        if len(pts) < 2:
            continue
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        ordered = sorted(((r - c, r, c) for r, c in pts), key=lambda t: t[0])
        b_min, r0, c0 = ordered[0]
        b_max, r1, c1 = ordered[-1]
        left_col = int(g[r0, c0])
        right_col = int(g[r1, c1])
        if left_col != right_col:
            continue
        segs.append((k, int(b_min), int(b_max), left_col))
    return segs


def _fill_between_pairs(
    grid: np.ndarray, segs: list[tuple[int, int, int, int]], bg: int, fill: int
) -> np.ndarray:
    """Fill only between active segment pairs that share the same endpoint marker color."""
    out = grid.copy()
    h, w = out.shape
    for b in range(-(w - 1), h):
        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        active_by_marker: dict[int, list[tuple[int, int, int, int]]] = {}
        for seg in segs:
            if seg[1] <= b <= seg[2]:
                active_by_marker.setdefault(seg[3], []).append(seg)
        for group in active_by_marker.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda t: t[0])
            for left, right in zip(group[0::2], group[1::2]):
                k_left, k_right = left[0], right[0]
                for r in range(r_min, r_max + 1):
                    c = r - b
                    k = r + c
                    if k_left < k < k_right and out[r, c] == bg:
                        out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the parallelogram-shaped region between anti-diagonal segments whose endpoint colors match.

    Input structure:
    - bg = most common color.
    - Each segment is an 8-connected anti-diagonal component on constant k=r+c.
    - Segment interiors share one body color; the two endpoints share a special marker color.

    Rule:
    - For each main diagonal b=r-c, collect the active segments whose b-range covers b.
    - Group active segments by their endpoint marker color.
    - For each same-color pair, fill bg cells strictly between the two segments with red (2).

    Key insight: Geometry still determines where a fill can lie, but endpoint colors determine
    which segment pair is allowed to connect.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    fill = 2
    segs = _extract_segments(g, bg, fill)
    return _fill_between_pairs(g, segs, bg, fill)


# --------------- generator ---------------

H = W = 16
BG = 8
FILL = 2
BODY_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
PAIR_COUNTS = [2, 3]
PAIR_WEIGHTS = [2, 1]
FOUR_SEG_GAP_VALUES = (4, 6)
FOUR_SEG_TOTAL_SPANS = (14, 16)

SEG_LEN_MIN, SEG_LEN_MAX = 4, 9
OVERLAP_MIN_B = 2

GEN_RETRY = 400
SEG_RETRY = 50
MIN_FILL_CELLS = 4
MIN_MARKER_EFFECT = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive b-range such that (r,c)=((k+b)/2,(k-b)/2) lies within the grid."""
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    cells = []
    for b in range(b_min, b_max + 1, 2):
        cells.append(((k + b) // 2, (k - b) // 2))
    return cells


def _apply_original_fill(
    grid: np.ndarray, segs: list[tuple[int, int, int, int]], bg: int, fill: int
) -> np.ndarray:
    """Old adjacency-only rule, kept only for non-degeneracy checks in the generator."""
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for left, right in zip(active, active[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if left[0] < k < right[0] and out[r, c] == bg:
                    out[r, c] = fill
    return out


def _choose_k_values(parity: int, n_seg: int) -> list[int] | None:
    """Pick 4 or 6 anti-diagonal indices with fixed parity."""
    if n_seg == 4:
        valid_combos = [
            (a, b, c)
            for a in FOUR_SEG_GAP_VALUES
            for b in FOUR_SEG_GAP_VALUES
            for c in FOUR_SEG_GAP_VALUES
            if a + b + c in FOUR_SEG_TOTAL_SPANS
        ]
        gaps = list(random.choice(valid_combos))
    elif n_seg == 6:
        gaps = [4, 4, 4, 4, 4]
    else:
        return None

    total_span = sum(gaps)
    candidates = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not candidates:
        return None
    k0 = random.choice(candidates)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    return ks


def _random_seg(
    k: int, constrain_lo: int | None = None, constrain_hi: int | None = None
) -> tuple[int, int] | None:
    """Pick a random segment range on anti-diagonal k, optionally overlapping another range."""
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


def _render_segments(
    segs: list[tuple[int, int, int, int]], body: int
) -> np.ndarray | None:
    """Render segments whose endpoints use their marker color and middle cells use the body color."""
    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max, marker in segs:
        cells = _seg_cells(k, b_min, b_max)
        if len(cells) < 2:
            return None
        for idx, (r, c) in enumerate(cells):
            if not (0 <= r < H and 0 <= c < W):
                return None
            color = marker if idx == 0 or idx == len(cells) - 1 else body
            if grid[r, c] != BG:
                return None
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices(PAIR_COUNTS, weights=PAIR_WEIGHTS)[0]
    n_seg = 2 * pair_count

    palette = BODY_COLORS.copy()
    body = random.choice(palette)
    marker_candidates = [c for c in palette if c != body]
    if len(marker_candidates) < pair_count:
        return None
    markers = random.sample(marker_candidates, pair_count)
    marker_seq = [m for m in markers for _ in range(2)]

    parity = random.randint(0, 1)
    ks = _choose_k_values(parity, n_seg)
    if ks is None:
        return None

    segs: list[tuple[int, int, int, int]] = []
    first = _random_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1], marker_seq[0]))

    for idx, k in enumerate(ks[1:], start=1):
        prev = segs[-1]
        nxt = _random_seg(k, prev[1], prev[2])
        if nxt is None:
            return None
        segs.append((k, nxt[0], nxt[1], marker_seq[idx]))

    grid = _render_segments(segs, body)
    if grid is None:
        return None

    output = _fill_between_pairs(grid, segs, BG, FILL)
    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        return None

    old_output = _apply_original_fill(grid, segs, BG, FILL)
    marker_effect = int(np.sum((old_output == FILL) & (output != FILL)))
    if marker_effect < MIN_MARKER_EFFECT:
        return None

    inp = output.copy()
    inp[inp == FILL] = BG
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
