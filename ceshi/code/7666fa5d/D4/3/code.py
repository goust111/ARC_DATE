# Variant of 7666fa5d
# Dimension: D4 — 目标位置：填充区域定义 (dst)
# Change: 沿主对角线方向(b=r-c固定)，在相邻反对角线段之间严格内部的背景色像素位置填充红色... → 间隔填充（棋盘式）

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the D4 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original: Between consecutive anti-diagonal segments, every background cell
#       in the projected interior is filled red.
#     variant: Keep the same interior region, but only fill the cells whose
#       checkerboard parity matches the segment endpoints. Because every cell of
#       a segment lies on one anti-diagonal r+c=k, all cells of that segment and
#       its endpoints share parity (r+c)%2 = k%2.
#   concrete_pixel_behavior:
#     - Input still contains one non-background line color on 4-5 anti-diagonals.
#     - Output still preserves those line cells.
#     - The filled parallelogram interiors become a checkerboard mask: among all
#       cells that the original task would fill, keep only those with
#       (r+c)%2 == line_parity.
#     - Since the original solver treats each segment span as continuous over all
#       integer b=r-c values between b_min and b_max, alternating b-bands inside
#       the overlap remain visible and produce the intended chessboard effect.
#   traced_example:
#     - Suppose two active segments have k=8 and k=12 on the same overlap band.
#     - Original logic on b=5 fills every background cell with 8 < r+c < 12.
#     - Variant logic first computes line_parity = 8%2 = 0. It still checks the
#       same cells, but only paints those where (r+c)%2 == 0.
#     - Difference: region membership is unchanged; the final destination mask is
#       parity-filtered, so roughly half of the original fill survives.
#   complexity_calibration:
#     - The generator keeps the original 16x16 size, one line color, 4-5 k-values,
#       and even gaps 4/6/8. That is enough structure for humans to see several
#       repeated diagonal bands, but it avoids introducing new object types.
#     - To keep the variant visually observable, generation rejects samples where
#       the checker fill differs from the solid fill by too few cells.
#   implicit_assumption_review_main:
#     - bg = most common color: unchanged and still guaranteed by generation.
#     - line = most frequent non-bg color: unchanged because inputs still contain
#       exactly one non-bg drawing color.
#     - components extracted with 8-connectivity: unchanged because segments are
#       still contiguous along anti-diagonals, and optional same-k splits remain
#       separated by a b-gap large enough to break 8-connectivity.
#     - seg span stored as (k, min_b, max_b): unchanged. The variant deliberately
#       keeps the original "continuous projected span" interpretation; only the
#       write condition changes.
#     - new assumption: all line segments share one checkerboard parity. The
#       generator guarantees this by sampling all k-values with the same parity.
#   placement_legality:
#     - All segment cells are still generated from exact ((k+b)/2,(k-b)/2)
#       coordinates and checked to stay in-bounds.
#     - Consecutive segments still overlap in projected b-range, so there are
#       interior regions to fill.
#     - Optional same-k split only duplicates one anti-diagonal with disconnected
#       pieces; it does not invalidate parity because same k means same parity.
#   observability_and_cleanliness:
#     - Humans can compare input/output and see the same regions as before, but
#       now only every other checkerboard class inside those regions turns red.
#     - The color vocabulary stays minimal: background 8, line one color, fill 2.
#     - Generator validation enforces both enough filled cells and enough skipped
#       interior cells, preventing accidental regression to the original solid fill.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(grid: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    """Return anti-diagonal segments as (k, b_min, b_max)."""
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(grid == line):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _apply_fill(grid: np.ndarray, segs: list[tuple[int, int, int]], bg: int, fill: int,
                line_parity: int | None = None, checker: bool = True) -> np.ndarray:
    """Fill between consecutive active segments; optionally keep only one parity class."""
    out = grid.copy()
    h, w = out.shape

    if line_parity is None:
        if not segs:
            return out
        line_parity = segs[0][0] & 1

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
                    if (not checker) or ((k & 1) == line_parity):
                        out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: fill only the checkerboard class that matches the segment endpoints.

    The original task fills every bg cell strictly between consecutive active
    anti-diagonal segments. This variant keeps the same geometric region but paints
    only those cells whose parity (r+c)%2 matches the line-segment parity.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2
    segs = _extract_segments(g, line)
    assert segs, "grid must contain at least one valid anti-diagonal segment"

    line_parity = segs[0][0] & 1
    return _apply_fill(g, segs, bg=bg, fill=fill, line_parity=line_parity, checker=True)


# --------------- generator ---------------

"""Generator for the checker-filled variant of ARC-AGI-2 task 7666fa5d."""

# -- Grid & color constants --
H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

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
GEN_RETRY = 400

# -- Validation --
MIN_FILL_CELLS = 4
MIN_SOLID_MINUS_CHECKER = 4


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    cells = []
    for b in range(b_min, b_max + 1, 2):
        cells.append(((k + b) // 2, (k - b) // 2))
    return cells


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
    k0_candidates = [k for k in range(K_LO, K_HI - total_span + 1) if (k & 1) == parity]
    if not k0_candidates:
        return None
    k0 = random.choice(k0_candidates)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    if ks[-1] > K_HI or len(set(ks)) != len(ks):
        return None
    return ks


def _random_seg(k: int, constrain_lo: int | None = None,
                constrain_hi: int | None = None) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k(k)
    if (b_lo & 1) != (k & 1):
        b_lo += 1
    if (b_hi & 1) != (k & 1):
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
                bm for bm in candidates
                if bm + step >= constrain_lo + OVERLAP_MIN_B and bm <= constrain_hi - OVERLAP_MIN_B
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
        length1 = random.randint(2, length - 2)
        piece1_max = bmin + 2 * (length1 - 1)
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if (b & 1) == (k & 1)]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        length2 = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (length2 - 1)
        if p2_max > b_hi:
            continue
        segs[idx] = (k, bmin, piece1_max)
        segs.append((k, p2_min, p2_max))
        return


def _variant_stats(grid: np.ndarray, segs: list[tuple[int, int, int]], parity: int) -> tuple[np.ndarray, np.ndarray, int]:
    solid = _apply_fill(grid, segs, bg=BG, fill=FILL, line_parity=parity, checker=False)
    checker = _apply_fill(grid, segs, bg=BG, fill=FILL, line_parity=parity, checker=True)
    skipped = int(np.sum((solid == FILL) & (checker != FILL)))
    return solid, checker, skipped


def _gen_attempt() -> np.ndarray | None:
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
            grid[r, c] = line

    solid, output, skipped = _variant_stats(grid, segs, parity)
    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        return None
    if skipped < MIN_SOLID_MINUS_CHECKER:
        return None

    inp = output.copy()
    inp[inp == FILL] = BG

    if not np.array_equal(main(inp), output):
        return None
    if np.array_equal(solid, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
