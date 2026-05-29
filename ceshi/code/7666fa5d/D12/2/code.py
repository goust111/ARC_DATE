# Variant of 7666fa5d
# Dimension: D12 — 背景色与颜色体系 (meta)
# Change: 背景色为azure(8)，线段颜色为单一非背景非红色(可为3/4/9等)，填充色固定为红色(2)... → 填充色由线段颜色的高对比搭档推导

"""
Design analysis for the D12 variant.

spec_variant:
  preserved_geometry:
    grid: 16x16
    background: 8 in generated samples
    objects:
      - one non-bg line color forming anti-diagonal segments on r+c=k
      - 4-6 connected components total
      - 4-5 distinct k values, with optional split on one k
    action:
      - sweep each main diagonal b=r-c
      - collect active segments whose b-range covers that b
      - sort them by k
      - fill bg cells strictly between consecutive active k values
  D12_change:
    old_fill: always 2
    new_fill: a fixed high-contrast partner of the line color
    partner_rule:
      - 1 (blue) <-> 7 (orange)
      - 3 (green) <-> 6 (magenta)
    generator_policy:
      - generated line colors are restricted to {1,3,6,7}
      - each output therefore uses a strongly contrasting partner color instead of a subtle palette step

behavior_trace:
  small_example:
    input:
      - line color 7 on several anti-diagonal segments
      - bg color 8 elsewhere
    old_solver:
      - detects line=7
      - fills all eligible interior cells with 2
    new_solver:
      - detects line=7
      - looks up the paired partner of 7, which is 1
      - fills the same interior geometry with 1
    difference:
      - geometry is unchanged
      - only the output color of the filled parallelograms changes
      - the new color is now visibly "the paired opposite" of the line color, not a barely noticeable step

complexity_calibration:
  - The task still has a single geometric rule; only the output color changes.
  - The mapping is intentionally low-complexity: one fixed partner lookup repeated across all examples.
  - The paired colors are chosen to be visually far apart, so the meta-color rule is easy to notice.
  - The generator avoids the old fixed-red behavior entirely.

implicit_assumption_audit_main:
  - old: "line = most frequent non-bg color" remains valid because inputs still contain exactly one non-bg color.
  - old: "fill = 2" is no longer valid; this is the core line that must change.
  - old: writing fill unconditionally onto bg cells was harmless because fill was constant.
    new: still safe because fill is derived once from the unique line color and never equals bg.
  - old: connected components are extracted only from the line color.
    new: still valid because input grids contain no fill cells, only bg and line.
  - old: segment geometry ignores color identity after detection.
    new: still valid because D12 changes only the palette rule, not the geometry.

placement_legality:
  - Segments must stay on a single anti-diagonal and within bounds.
  - Consecutive k values must overlap in b-space so some fill exists.
  - Optional split segments on the same k must remain 8-disconnected.
  - Fill is only written into bg cells, so it never overwrites line cells.
  - Because fill is looked up from a small high-contrast partner map and skipped over bg, fill cannot visually merge with the background.

visual_observability:
  - Humans still see the same diagonal-line-to-parallelogram relation.
  - The difference is visible because the interior color switches to a strongly contrasting partner color.
  - Reusing the same two partner pairs across examples makes the color rule easier to infer.

cleanliness:
  - Only two foreground colors appear in outputs: the line color and its derived partner fill color.
  - No extra markers are added, so the images remain as clean as the source task.

new_solver_assumptions_to_match_in_generator:
  - inputs contain exactly one non-background color
  - that color forms the diagonal segments
  - background is the most frequent color
  - derived fill color differs from background
  - there is at least one fillable region
"""

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


LINE_TO_FILL = {
    1: 7,
    7: 1,
    3: 6,
    6: 3,
}


def _derived_fill_color(line: int, bg: int) -> int:
    """Map the line color to a fixed, high-contrast partner color."""
    fill = LINE_TO_FILL[int(line)]
    assert fill != int(bg), "partner fill color must differ from background"
    return int(fill)


def _extract_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return segments as (k, b_min, b_max) for 8-connected anti-diagonal components."""
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
    Fill the parallelogram-shaped region between parallel anti-diagonal line segments.

    This D12 variant preserves the original geometry and only changes the output color:
    the fill is derived from the unique line color via a small high-contrast partner map
    (blue<->orange, green<->magenta).
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = _derived_fill_color(line, bg)

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
        for k_left, k_right in zip(ks, ks[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill

    return out


# --------------- generator ---------------

H = W = 16
BG = 8
LINE_COLORS = [1, 3, 6, 7]  # use only high-contrast partner pairs

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
GEN_RETRY = 300

MIN_FILL_CELLS = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    cells = []
    for b in range(b_min, b_max + 1, 2):
        cells.append(((k + b) // 2, (k - b) // 2))
    return cells


def _apply_fill(grid: np.ndarray, segs: list[tuple[int, int, int]], fill: int) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        ks = [t[0] for t in active]
        r_min, r_max = max(0, b), min(H - 1, (W - 1) + b)
        for k_left, k_right in zip(ks, ks[1:]):
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
                b_min
                for b_min in candidates
                if b_min + step >= constrain_lo + OVERLAP_MIN_B and b_min <= constrain_hi - OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _try_split_segment(segs: list[tuple[int, int, int]], idx: int) -> None:
    k, bmin, bmax = segs[idx]
    length = (bmax - bmin) // 2 + 1
    if length < SPLIT_MIN_LEN:
        return

    _, b_hi = _b_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        piece1_len = random.randint(2, length - 2)
        piece1_max = bmin + 2 * (piece1_len - 1)
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if b % 2 == k % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        piece2_len = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (piece2_len - 1)
        if p2_max > b_hi:
            continue
        segs[idx] = (k, bmin, piece1_max)
        segs.append((k, p2_min, p2_max))
        return


def _variant_signature(inp: np.ndarray) -> dict[str, int | bool]:
    """Small metric bundle for post-generation audits."""
    bg = _bg_color(inp)
    non_bg = [c for c in sorted(set(map(int, inp.ravel()))) if c != bg]
    line = max(non_bg, key=lambda c: int(np.sum(inp == c)))
    fill = _derived_fill_color(line, bg)
    return {
        "line": int(line),
        "fill": int(fill),
        "matches_old_fixed_red": bool(fill == 2),
    }


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    fill = _derived_fill_color(line, BG)
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
            grid[r, c] = line

    output = _apply_fill(grid, segs, fill)
    fill_cells = int(np.sum(output == fill))
    if fill_cells < MIN_FILL_CELLS:
        return None

    inp = output.copy()
    inp[inp == fill] = BG

    # Variant enforcement: generated samples should actually demonstrate the new color rule.
    if fill == 2:
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
