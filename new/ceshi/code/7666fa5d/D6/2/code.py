# Variant of 7666fa5d
# Dimension: D6 — 轨迹处理：源像素保留策略 (trajectory)
# Change: 保留源线段像素不变（线段颜色不被红色覆盖），仅填充背景色位置... → 连续轨迹：线段向填充方向延伸涂色

"""
Design analysis for the D6 trajectory variant.

reference_task:
  source_task: 8b9c3697
  borrowed_mechanism:
    - The reference solver keeps the moving source object visible through all
      intermediate translated positions, painting a continuous trail before the
      final placement.
    - For this task there is no single final docking cell, so the analogous
      adaptation is: each boundary segment grows translated copies toward the
      interior, and the red fill survives only as the central band left between
      the two opposed trails.

variant_spec:
  grid: 16x16
  background: 8
  colors:
    line: one non-bg color, chosen from {1,3,4,5,6,7,9}
    fill: 2
  objects:
    - input contains 3-4 anti-diagonal line segments or split pieces
    - every connected component still lies on one anti-diagonal r+c=k
    - output preserves the original source segments
  core_change:
    original:
      - for each main diagonal b=r-c, cells strictly between adjacent active
        segments are all painted red
    variant:
      - for each b-line and each adjacent segment pair, interior cells are
        ordered from the left segment toward the right segment in k=r+c order
      - the outer half of that ordered list becomes line-colored trail copied
        from the two boundary segments
      - a central band stays red; width is 1 cell when the number of interior
        cells is odd, otherwise 2 cells
      - if the gap is too small (1-2 interior cells), no trail fits and the
        behavior degenerates locally to pure red fill; the generator therefore
        enforces at least one wide gap per sample so the D6 behavior is visible

behavior_trace:
  example_pair:
    segments_on_k: [8, 16]
    active_b_range: shared
    interior_k_values: [10, 12, 14]
    original_output: [2, 2, 2]
    variant_output: [line, 2, line]
    difference:
      - original has one solid red strip between boundaries
      - variant shows line-colored rays advancing inward from both sides,
        leaving only the centered red strip between those rays

complexity_calibration:
  - Keeping only one line color avoids introducing a second inference problem.
  - The generator increases anti-diagonal gaps to mostly 8/10 so trails are
    visible in 2-4 examples instead of silently collapsing back to the original.
  - The number of k-values is reduced to 3-4 because wider gaps must still fit
    into a 16x16 canvas. This is the minimum adaptation needed to expose D6.

implicit_assumption_review_main:
  - Component extraction still assumes each source component is 8-connected and
    lies on one k=r+c value. This remains valid because only the output gains
    trails; the input still contains only the original segments.
  - Sorting active segments by k still works because generator keeps all ks on
    one parity, so intermediate cells on a fixed b-line remain integral.
  - Writing line-colored trails would be unsafe if we overwrote source pixels.
    The implementation therefore paints only cells strictly between the two
    source segments.
  - Adjacent-pair processing assumes pair intervals on one b-line do not
    overlap. This still holds because sorted consecutive k-intervals partition
    that b-line in the same way as the original task.

placement_legality:
  - All segment cells are generated from valid (k,b) ranges derived from grid
    bounds, so every implied coordinate stays in-bounds.
  - Consecutive segments must overlap in b by at least 4, which guarantees
    visible parallelogram regions and lets the solver detect active pairs.
  - Optional same-k splitting uses a b-gap of 4, which breaks 8-connectivity so
    the solver still sees separate source components.

visual_observability:
  - The new rule is visible because wide gaps produce line-colored border strips
    around a thinner red center. Across examples this repeats enough times to be
    legible as “the lines continue inward”.
  - Samples where the new output equals the original all-red fill are rejected.

main_generator_contract:
  - main() assumes at least one non-bg color and at least one segment pair with
    overlapping b-range.
  - generate_input() guarantees both conditions, plus a minimum number of trail
    cells and a strict difference from the original fill-only baseline.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _extract_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (k, b_min, b_max) for each 8-connected source segment."""
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(mask):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _variant_paint(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    *,
    bg: int,
    line: int,
    fill: int,
) -> tuple[np.ndarray, int, int]:
    """
    Paint the D6 variant output.

    For each b=r-c line, sort active segments by k=r+c. Between consecutive
    active pairs, color symmetric outer runs with the line color (continuous
    trail) and keep only the central band red.
    """
    h, w = grid.shape
    out = grid.copy()
    trail_cells = 0
    fill_cells = 0

    for b in range(-(w - 1), h):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for (k_left, _, _), (k_right, _, _) in zip(active, active[1:]):
            between: list[tuple[int, int]] = []
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right:
                    between.append((int(r), int(c)))

            n = len(between)
            if n == 0:
                continue

            trail_each = max(0, (n - 1) // 2)
            left_trail = between[:trail_each]
            right_trail = between[n - trail_each :] if trail_each else []
            middle_fill = between[trail_each : n - trail_each]

            for r, c in left_trail:
                if out[r, c] == bg:
                    out[r, c] = line
                    trail_cells += 1
            for r, c in right_trail:
                if out[r, c] == bg:
                    out[r, c] = line
                    trail_cells += 1
            for r, c in middle_fill:
                if out[r, c] == bg:
                    out[r, c] = fill
                    fill_cells += 1

    return out, int(trail_cells), int(fill_cells)


def _baseline_fill(grid: np.ndarray, segs: list[tuple[int, int, int]], *, bg: int, fill: int) -> np.ndarray:
    """Original task behavior: all interior cells become red."""
    h, w = grid.shape
    out = grid.copy()
    for b in range(-(w - 1), h):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for (k_left, _, _), (k_right, _, _) in zip(active, active[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: parallel anti-diagonal segments spawn inward line-colored
    trails, while red occupies only the centered band between those trails.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    fill = 2
    segs = _extract_segments(g == line)
    assert segs, "expected at least one anti-diagonal source segment"

    out, _trail_cells, _fill_cells = _variant_paint(g, segs, bg=bg, line=int(line), fill=fill)
    return out


# --------------- generator ---------------

H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
NUM_KS_CHOICES = [3, 4]
NUM_KS_WEIGHTS = [2, 1]
GAP_PATTERNS_3 = (
    (8, 8),
    (8, 10),
    (10, 8),
)
GAP_PATTERNS_4 = (
    (8, 8, 6),
    (8, 6, 8),
    (6, 8, 8),
    (10, 6, 6),
    (6, 10, 6),
    (6, 6, 10),
)

SEG_LEN_MIN, SEG_LEN_MAX = 3, 9
OVERLAP_MIN_B = 4

SPLIT_PROB = 0.35
SPLIT_MIN_LEN = 5
SPLIT_B_GAP = 4
SPLIT_PIECE2_LEN_MIN = 2
SPLIT_PIECE2_LEN_MAX = 5

SEG_RETRY = 60
SPLIT_RETRY = 12
GEN_RETRY = 400

MIN_FILL_CELLS = 3
MIN_TRAIL_CELLS = 4


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return int(b_lo), int(b_hi)


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _choose_k_values(parity: int) -> list[int] | None:
    num_ks = random.choices(NUM_KS_CHOICES, weights=NUM_KS_WEIGHTS)[0]
    gaps = list(random.choice(GAP_PATTERNS_3 if num_ks == 3 else GAP_PATTERNS_4))
    total_span = sum(gaps)
    starts = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not starts:
        return None
    k0 = random.choice(starts)
    ks = [int(k0)]
    for gap in gaps:
        ks.append(int(ks[-1] + gap))
    if ks[-1] > K_HI:
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
                bm
                for bm in candidates
                if bm + step >= constrain_lo + OVERLAP_MIN_B and bm <= constrain_hi - OVERLAP_MIN_B
            ]
        if not candidates:
            continue
        b_min = random.choice(candidates)
        return int(b_min), int(b_min + step)
    return None


def _try_split_segment(segs: list[tuple[int, int, int]], idx: int) -> None:
    k, bmin, bmax = segs[idx]
    length = (bmax - bmin) // 2 + 1
    if length < SPLIT_MIN_LEN:
        return

    _, b_hi = _b_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        l1 = random.randint(2, length - 2)
        piece1_max = bmin + 2 * (l1 - 1)
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if b % 2 == k % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        l2 = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (l2 - 1)
        if p2_max > b_hi:
            continue
        segs[idx] = (int(k), int(bmin), int(piece1_max))
        segs.append((int(k), int(p2_min), int(p2_max)))
        return


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
    segs.append((int(ks[0]), int(first[0]), int(first[1])))

    for k in ks[1:]:
        prev_bmin, prev_bmax = segs[-1][1], segs[-1][2]
        res = _random_seg(k, prev_bmin, prev_bmax)
        if res is None:
            return None
        segs.append((int(k), int(res[0]), int(res[1])))

    if len(segs) <= 4 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            grid[r, c] = int(line)

    segs_for_solver = _extract_segments(grid == int(line))
    if len(segs_for_solver) < 2:
        return None

    output, trail_cells, fill_cells = _variant_paint(grid, segs_for_solver, bg=BG, line=int(line), fill=FILL)
    baseline = _baseline_fill(grid, segs_for_solver, bg=BG, fill=FILL)

    if fill_cells < MIN_FILL_CELLS or trail_cells < MIN_TRAIL_CELLS:
        return None
    if np.array_equal(output, baseline):
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
