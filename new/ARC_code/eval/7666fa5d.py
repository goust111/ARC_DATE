import numpy as np
import random

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the parallelogram-shaped region between parallel anti-diagonal line segments with red.

    Input structure:
    - bg = most common color.
    - One non-bg color forms short diagonal segments, each lying on a single anti-diagonal (r+c = const).
    - Segments come in parallel pairs bounding a region to be filled.

    Rule:
    - For each main-diagonal axis (r−c = const), collect anti-diagonal segments whose span covers that axis.
    - Sort those segments by their anti-diagonal index (r+c).
    - Fill all bg cells lying strictly between consecutive segment pairs along that axis with red (color 2).

    Key insight: The two diagonal directions decouple — segments sit on anti-diagonals, and filling sweeps along main diagonals between them, forming parallelogram regions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    # This task uses one diagonal-line color and fills with 2.
    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2

    comps = _cc8(g == line)

    # Each diagonal segment is on a constant anti-diagonal sum k=r+c,
    # and spans a continuous range of main-diagonal indices b=r-c.
    segs: list[tuple[int, int, int]] = []
    for pts in comps:
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))

    out = g.copy()

    # Scan each main diagonal b=r-c. For every b, find the active diagonal segments
    # (those whose b-range covers b), sort them by k=r+c, then fill between adjacent
    # segments for cells on this b with k strictly between.
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

# --------------- generator (Claude attempt 1) ---------------

"""Generator for ARC-AGI-2 task 7666fa5d (parallelogram fill between anti-diagonal segments)."""

# -- Grid & color constants --
H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]  # not 2 (fill), not 8 (bg)

# -- Anti-diagonal (k = r+c) layout --
K_LO, K_HI = 4, 26                    # allowed k range
GAP_VALUES = (4, 6, 8)                 # even gaps between consecutive k-values
GAP_TARGET_SPANS = (14, 16)            # valid total k-span for 4-k layouts
NUM_KS_CHOICES = [4, 5]                # number of anti-diagonals to use
NUM_KS_WEIGHTS = [1, 1]                # sampling weights for num_ks

# -- Segment geometry --
SEG_LEN_MIN, SEG_LEN_MAX = 2, 9       # segment length in cells
OVERLAP_MIN_B = 2                      # min b-overlap between consecutive segments

# -- Segment splitting --
SPLIT_PROB = 0.4                       # probability of splitting one segment
SPLIT_MIN_LEN = 4                      # min segment length (cells) to be splittable
SPLIT_B_GAP = 4                        # min b-gap to break 8-connectivity
SPLIT_PIECE2_LEN_MIN = 2              # second piece length range
SPLIT_PIECE2_LEN_MAX = 6

# -- Retry budgets --
SEG_RETRY = 40                         # retries per segment placement
SPLIT_RETRY = 10                       # retries for segment splitting
GEN_RETRY = 300                        # outer generate_input retries

# -- Validation --
MIN_FILL_CELLS = 3                     # min fill cells in output


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    """Inclusive b-range such that (r,c)=((k+b)/2,(k-b)/2) lies within the grid."""
    # r = (k+b)/2 in [0, H-1]  =>  b in [-k, 2(H-1)-k]
    # c = (k-b)/2 in [0, W-1]  =>  b in [k-2(W-1), k]
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    """Return (r,c) cells for a segment on anti-diagonal k spanning b_min..b_max (step 2)."""
    cells = []
    for b in range(b_min, b_max + 1, 2):
        cells.append(((k + b) // 2, (k - b) // 2))
    return cells


def _apply_fill(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    """Apply the solver's fill logic: fill bg cells between consecutive active segments per b-line."""
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
                    out[r, c] = FILL
    return out


def _choose_k_values(parity: int) -> list[int] | None:
    """Pick 4-5 evenly-spaced anti-diagonal indices with the given parity."""
    num_ks = random.choices(NUM_KS_CHOICES, weights=NUM_KS_WEIGHTS)[0]
    if num_ks == 4:
        # 3 gaps from GAP_VALUES with total span in GAP_TARGET_SPANS
        valid_combos = [(a, b, c) for a in GAP_VALUES for b in GAP_VALUES for c in GAP_VALUES
                        if a + b + c in GAP_TARGET_SPANS]
        gaps = list(random.choice(valid_combos))
    else:
        # 5 ks: 4 gaps summing to 16 => only 4+4+4+4=16
        gaps = [4, 4, 4, 4]

    total_span = sum(gaps)
    k0_candidates = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not k0_candidates:
        return None
    k0 = random.choice(k0_candidates)
    ks = [k0]
    for g in gaps:
        ks.append(ks[-1] + g)
    if ks[-1] > K_HI or len(set(ks)) != len(ks):
        return None
    return ks


def _random_seg(k: int, constrain_lo: int | None = None,
                constrain_hi: int | None = None) -> tuple[int, int] | None:
    """Pick a random segment b-range on anti-diagonal k.

    If constrain_lo/hi are given, the new segment must overlap
    [constrain_lo, constrain_hi] by at least OVERLAP_MIN_B in b-space.
    """
    b_lo, b_hi = _b_bounds_for_k(k)
    # Align bounds to k-parity (b must have same parity as k)
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
            # b_max >= constrain_lo + OVERLAP_MIN_B AND b_min <= constrain_hi - OVERLAP_MIN_B
            candidates = [bm for bm in candidates
                          if bm + step >= constrain_lo + OVERLAP_MIN_B
                          and bm <= constrain_hi - OVERLAP_MIN_B]
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
        L1 = random.randint(2, length - 2)
        piece1_max = bmin + 2 * (L1 - 1)
        # Second piece starts beyond the 8-connectivity gap
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2)
                         if b % 2 == k % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        L2 = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (L2 - 1)
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

    # Build segments with consecutive-overlap constraint
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

    # Optionally split one segment into two non-overlapping pieces on the same k
    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    # Place line cells on grid
    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            grid[r, c] = line

    output = _apply_fill(grid, segs)

    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        return None

    # Input = output with fill replaced by bg
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
