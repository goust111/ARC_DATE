# Variant of 7666fa5d
# Dimension: D9 — 线段间的空间关系：平行性约束 (meta)
# Change: 所有线段均位于反对角线上(r+c=常数)，因此所有线段天然平行... → 平行线段对之间要求最小间距

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the D9 variant
# a) Concrete behavior and trace:
#    The geometry stays the same as the original task: one color draws short line segments on
#    anti-diagonals k=r+c, and red fill grows along main diagonals b=r-c. The only rule change is
#    that a gap is fillable only when the two neighboring active anti-diagonals are far enough apart.
#    Here "far enough" is a visible geometric quantity: fill between consecutive active segments only
#    if k_right-k_left > 4. Example on one b-line:
#      active k values = [8, 12, 18]
#      original task: fill both gaps (8,12) and (12,18)
#      variant: 12-8=4 is too close, so leave that strip blank; 18-12=6 is wide enough, so fill it.
#    The changed behavior only touches the pair-selection stage; extraction of segments and actual cell
#    painting remain the original parallelogram sweep.
# b) Complexity calibration:
#    The threshold is fixed and low-cardinality rather than free-form. Generator gaps are chosen from
#    {4,6,8}, so each puzzle repeatedly shows "close" (4) and "far" (6/8) examples. This avoids
#    ambiguous one-off numeric coincidences and makes the new rule visually learnable from 2-3 samples.
# c) Hidden-assumption audit of the original solver:
#    - line = most frequent non-bg color: still safe because generator keeps exactly one line color.
#    - each connected component lies on one anti-diagonal: still true; generator only places such pieces.
#    - active.sort(...); zip(neighbors): still the correct adjacency notion, but the original code assumed
#      every adjacent pair should fill. That assumption is now false, so the solver must gate each pair by
#      its k-distance before painting.
#    - unconditional out[r,c]=fill on every admissible interior bg cell was harmless originally; in this
#      variant it would overfill narrow 4-gap strips. The new distance check prevents that regression.
# d) Placement legality:
#    Generator still builds in-bounds contiguous segments in (k,b)-space. Consecutive base segments are
#    forced to overlap in b so both close and far pairs are actually active somewhere. Optional splitting
#    stays on the same k and keeps 8-disconnected pieces, which the solver already handles as separate
#    components on one anti-diagonal.
# e) Visual observability:
#    The user can see the difference directly: some adjacent parallel line pairs create red bands while
#    equally oriented but tighter neighboring pairs do not. Because only the spacing changes, the D9
#    modification is visible without relying on hidden color order or symbolic labels.
# f) Visual cleanliness:
#    The puzzle still uses one line color plus red fill on azure background. No extra markers are added.
#    Generator enforces both visible fill and visible suppressed fill so the output does not collapse into
#    either "fill everything" or "fill nothing".
# g) New solver assumptions and generator guarantees:
#    The solver now assumes that useful examples contain at least one adjacent active pair with k-gap 4
#    and one with k-gap 6/8; otherwise the variant would be hard to infer or could degenerate to the
#    original behavior. Generator validates this indirectly by requiring both positive fill cells and a
#    positive number of cells that the legacy solver would have filled but this variant leaves blank.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(grid: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    """Return segments as (k, b_min, b_max) for connected line components."""
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(grid == line):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _apply_fill(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    *,
    bg: int,
    fill: int,
    min_gap: int,
) -> np.ndarray:
    """Fill between consecutive active anti-diagonal segments whose k-gap exceeds min_gap."""
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

        for (k_left, _b0, _b1), (k_right, _c0, _c1) in zip(active, active[1:]):
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
    Variant rule: keep the original anti-diagonal geometry, but only fill between consecutive active
    segments when their anti-diagonal indices differ by more than 4.
    """
    g = np.asarray(grid, dtype=int)

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2
    segs = _extract_segments(g, line)
    return _apply_fill(g, segs, bg=bg, fill=fill, min_gap=MIN_FILL_GAP)


# ---------------- generator ----------------

H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
MIN_FILL_GAP = 4
K_GAP_PATTERNS = [
    (4, 6, 4),
    (6, 4, 6),
    (4, 8, 4),
    (8, 4, 6),
    (6, 4, 8),
    (4, 6, 4, 6),
    (6, 4, 6, 4),
    (4, 8, 4, 6),
    (6, 4, 8, 4),
]

SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_B = 4

SPLIT_PROB = 0.25
SPLIT_MIN_LEN = 4
SPLIT_B_GAP = 4
SPLIT_PIECE2_LEN_MIN = 2
SPLIT_PIECE2_LEN_MAX = 6

SEG_RETRY = 50
SPLIT_RETRY = 12
GEN_RETRY = 500

MIN_FILL_CELLS = 3
MIN_SUPPRESSED_CELLS = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive b-range such that (r,c)=((k+b)/2,(k-b)/2) lies within the grid."""
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    """Return (r,c) cells for a segment on anti-diagonal k spanning b_min..b_max (step 2)."""
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _choose_k_values(parity: int) -> list[int] | None:
    """Choose 4-5 anti-diagonal indices with both close (4) and far (6/8) gaps present."""
    gaps = list(random.choice(K_GAP_PATTERNS))
    total_span = sum(gaps)
    k0_candidates = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not k0_candidates:
        return None

    k0 = random.choice(k0_candidates)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    if ks[-1] > K_HI:
        return None
    return ks


def _random_seg(
    k: int,
    constrain_lo: int | None = None,
    constrain_hi: int | None = None,
) -> tuple[int, int] | None:
    """Pick a random segment b-range on anti-diagonal k, optionally overlapping a prior range."""
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
    """Try to split segs[idx] into two 8-disconnected pieces on the same k."""
    k, bmin, bmax = segs[idx]
    length = (bmax - bmin) // 2 + 1
    if length < SPLIT_MIN_LEN:
        return

    _, b_hi = _b_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        len1 = random.randint(2, length - 2)
        piece1_max = bmin + 2 * (len1 - 1)
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if b % 2 == k % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        len2 = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (len2 - 1)
        if p2_max > b_hi:
            continue
        segs[idx] = (k, bmin, piece1_max)
        segs.append((k, p2_min, p2_max))
        return


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate an input whose output exhibits the minimum-gap rule."""
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

    if random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

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
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
