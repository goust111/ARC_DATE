# Combo variant of 7666fa5d (index 2)
# combo_spec: D2/0+D11/0
# Name: 方向完全反转（主对角线段+反对角线扫描）

"""
Design analysis for the mirrored combo.

a) Concrete behavior:
   Original task: components lie on anti-diagonals (constant k=r+c), each segment spans
   a range of main-diagonal indices b=r-c. Filling scans one main diagonal at a time.
   Combo task: components lie on main diagonals (constant b=r-c), each segment spans
   a range of anti-diagonal indices k=r+c. Filling scans one anti-diagonal at a time.
   Example: if we place segments on b=-4, 0, 4 and all three cover k=10, then cells
   on anti-diagonal k=10 are ordered by b, and background cells with -4 < b < 0 and
   0 < b < 4 become red. This is the geometric mirror of the source task.

b) Complexity calibration:
   The generator still uses 4-5 distinct diagonal tracks plus an optional split, so the
   puzzle keeps repeated evidence instead of collapsing to a single trivial band. Gaps
   between b-values are restricted to a small even set, preserving visible regularity.

c) Hidden assumptions audit:
   - `main()` assumes the dominant non-background color forms 8-connected diagonal
     components; the generator uses only one line color.
   - Each kept component must have constant b and a contiguous k-range; the generator
     constructs segments exactly that way and only splits with a gap that breaks 8-connectivity.
   - Filling relies on at least two active segments on some anti-diagonal; consecutive
     generated segments are forced to overlap in k-space, so active sets exist.
   - Ordering by b must correspond to left-to-right traversal along a fixed anti-diagonal;
     for k fixed, b increases monotonically with row, so the solver's ordering is stable.

d) Placement legality:
   For a chosen b=r-c we derive the valid k=r+c interval from grid bounds, sample only
   parity-compatible endpoints, and convert back to integer (r,c) cells. Split pieces on
   the same b are separated by at least 4 in k, so they neither touch nor overlap.

e) Visual observability:
   Humans can see that the source strokes now slant top-left to bottom-right, while the
   red fill is assembled by anti-diagonal cross-sections. The transformation differs
   clearly from the original because the input components themselves have the opposite
   orientation.

f) Visual cleanliness:
   Only one line color and one fill color are used. Different components occupy distinct
   main diagonals, with at most one optional second piece on the same diagonal, so the
   scene stays readable.

g) Inference friendliness:
   No hidden ordering uses color ids or metadata. The only cue is geometric structure:
   parallel main-diagonal segments define lanes, and red fills appear between adjacent
   active lanes on each anti-diagonal.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the parallelogram-like gaps between parallel main-diagonal line segments with red.

    Input structure:
    - The background is the most frequent color.
    - One other color forms short segments, each confined to a single main diagonal (r-c = const).
    - Several such segments overlap in anti-diagonal reach, so they define adjacent lanes.

    Rule:
    - For each anti-diagonal slice (r+c = const), gather the main-diagonal segments whose k-range covers it.
    - Sort those active segments by their main-diagonal index b=r-c.
    - Recolor background cells lying strictly between consecutive active segments on that slice to red (2).

    Key insight:
    - The task is the exact directional mirror of the source puzzle: segment extraction happens on main
      diagonals, while the fill sweep happens on anti-diagonals. Swapping these two coordinate systems
      preserves the algorithmic pattern but rotates the geometric reasoning.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2

    comps = _cc8(g == line)

    # Each kept component is a segment on a constant main diagonal b=r-c and spans a
    # continuous anti-diagonal range k=r+c.
    segs: list[tuple[int, int, int]] = []
    for pts in comps:
        bs = {r - c for r, c in pts}
        if len(bs) != 1:
            continue
        b = int(next(iter(bs)))
        ks = [r + c for r, c in pts]
        segs.append((b, int(min(ks)), int(max(ks))))

    out = g.copy()

    # Scan each anti-diagonal k=r+c. For every k, find the active main-diagonal segments
    # (those whose k-range covers k), order them by b=r-c, then fill cells whose b lies
    # strictly between consecutive segment diagonals.
    for k in range(h + w - 1):
        active = [s for s in segs if s[1] <= k <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])
        bs = [t[0] for t in active]

        r_min = max(0, k - (w - 1))
        r_max = min(h - 1, k)
        for b_left, b_right in zip(bs, bs[1:]):
            for r in range(r_min, r_max + 1):
                c = k - r
                b = r - c
                if b_left < b < b_right and out[r, c] == bg:
                    out[r, c] = fill

    return out


# --------------- generator ---------------

H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

# -- Main-diagonal (b = r-c) layout --
B_LO, B_HI = -11, 11
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_BS_CHOICES = [4, 5]
NUM_BS_WEIGHTS = [1, 1]

# -- Segment geometry --
SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_K = 2

# -- Segment splitting --
SPLIT_PROB = 0.4
SPLIT_MIN_LEN = 4
SPLIT_K_GAP = 4
SPLIT_PIECE2_LEN_MIN = 2
SPLIT_PIECE2_LEN_MAX = 6

# -- Retry budgets --
SEG_RETRY = 40
SPLIT_RETRY = 10
GEN_RETRY = 400

# -- Validation --
MIN_FILL_CELLS = 3
MIN_COMPONENTS = 4
MAX_COMPONENTS = 6


def _k_bounds_for_b(b: int) -> tuple[int, int]:
    """Inclusive k-range for cells on main diagonal b=r-c that stay inside the grid."""
    r_lo = max(0, b)
    r_hi = min(H - 1, (W - 1) + b)
    k_lo = 2 * r_lo - b
    k_hi = 2 * r_hi - b
    return k_lo, k_hi


def _seg_cells(b: int, k_min: int, k_max: int) -> list[tuple[int, int]]:
    """Return (r,c) cells for a segment on main diagonal b spanning k_min..k_max."""
    cells = []
    for k in range(k_min, k_max + 1, 2):
        cells.append(((k + b) // 2, (k - b) // 2))
    return cells


def _apply_fill(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    """Apply the mirrored fill logic used by main()."""
    out = grid.copy()
    for k in range(H + W - 1):
        active = sorted([s for s in segs if s[1] <= k <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        bs = [t[0] for t in active]
        r_min = max(0, k - (W - 1))
        r_max = min(H - 1, k)
        for b_left, b_right in zip(bs, bs[1:]):
            for r in range(r_min, r_max + 1):
                c = k - r
                b = r - c
                if b_left < b < b_right and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def _choose_b_values(parity: int) -> list[int] | None:
    """Pick 4-5 evenly-spaced main-diagonal indices with shared parity."""
    num_bs = random.choices(NUM_BS_CHOICES, weights=NUM_BS_WEIGHTS)[0]
    if num_bs == 4:
        valid_combos = [(a, b, c) for a in GAP_VALUES for b in GAP_VALUES for c in GAP_VALUES
                        if a + b + c in GAP_TARGET_SPANS]
        gaps = list(random.choice(valid_combos))
    else:
        gaps = [4, 4, 4, 4]

    total_span = sum(gaps)
    b0_candidates = [b for b in range(B_LO, B_HI - total_span + 1) if b % 2 == parity]
    if not b0_candidates:
        return None
    b0 = random.choice(b0_candidates)
    bs = [b0]
    for gap in gaps:
        bs.append(bs[-1] + gap)
    if bs[-1] > B_HI or len(set(bs)) != len(bs):
        return None
    return bs


def _random_seg(
    b: int,
    constrain_lo: int | None = None,
    constrain_hi: int | None = None,
) -> tuple[int, int] | None:
    """Pick a random k-range on main diagonal b, optionally overlapping a prior k-range."""
    k_lo, k_hi = _k_bounds_for_b(b)
    if k_lo % 2 != b % 2:
        k_lo += 1
    if k_hi % 2 != b % 2:
        k_hi -= 1
    if k_lo > k_hi:
        return None

    for _ in range(SEG_RETRY):
        length = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
        step = 2 * (length - 1)
        if k_lo + step > k_hi:
            continue
        candidates = list(range(k_lo, k_hi - step + 1, 2))
        if constrain_lo is not None and constrain_hi is not None:
            candidates = [
                km for km in candidates
                if km + step >= constrain_lo + OVERLAP_MIN_K and km <= constrain_hi - OVERLAP_MIN_K
            ]
            if not candidates:
                continue
        k_min = random.choice(candidates)
        return k_min, k_min + step
    return None


def _try_split_segment(segs: list[tuple[int, int, int]], idx: int) -> None:
    """Try to split segs[idx] into two 8-disconnected pieces on the same main diagonal."""
    b, k_min, k_max = segs[idx]
    length = (k_max - k_min) // 2 + 1
    if length < SPLIT_MIN_LEN:
        return

    _, k_hi = _k_bounds_for_b(b)
    for _ in range(SPLIT_RETRY):
        first_len = random.randint(2, length - 2)
        piece1_max = k_min + 2 * (first_len - 1)
        p2_candidates = [k for k in range(piece1_max + SPLIT_K_GAP, k_hi - 1, 2) if k % 2 == b % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        second_len = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (second_len - 1)
        if p2_max > k_hi:
            continue
        segs[idx] = (b, k_min, piece1_max)
        segs.append((b, p2_min, p2_max))
        return


def _old_main_like_original(grid: np.ndarray) -> np.ndarray:
    """Reference the pre-combo behavior to detect degeneracy in generated samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    comps = _cc8(g == line)

    segs: list[tuple[int, int, int]] = []
    for pts in comps:
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))

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
                    out[r, c] = FILL
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid mirrored input grid."""
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)

    bs = _choose_b_values(parity)
    if bs is None:
        return None

    segs: list[tuple[int, int, int]] = []
    first = _random_seg(bs[0])
    if first is None:
        return None
    segs.append((bs[0], first[0], first[1]))

    for b in bs[1:]:
        prev_kmin, prev_kmax = segs[-1][1], segs[-1][2]
        res = _random_seg(b, prev_kmin, prev_kmax)
        if res is None:
            return None
        segs.append((b, res[0], res[1]))

    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for b, k_min, k_max in segs:
        for r, c in _seg_cells(b, k_min, k_max):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    if not (MIN_COMPONENTS <= len(_cc8(grid == line)) <= MAX_COMPONENTS):
        return None

    output = _apply_fill(grid, segs)
    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        return None

    inp = output.copy()
    inp[inp == FILL] = BG

    if not np.array_equal(main(inp), output):
        return None

    # Degeneracy guard: the mirrored inputs should not also trigger the original solver.
    if np.array_equal(_old_main_like_original(inp), output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
