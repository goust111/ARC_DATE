# Variant of 7666fa5d
# Dimension: D9 — 线段间的空间关系：平行性约束 (meta)
# Change: 所有线段均位于反对角线上(r+c=常数)，因此所有线段天然平行... → 线段方向由颜色编码

import numpy as np
import random

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the D9 variant
# ----------------------------------
# spec:
#   grid: 16x16
#   background: 8
#   fill: 2
#   line_families:
#     - one non-bg color is used only for anti-diagonal segments (constant r+c)
#     - a second non-bg color is used only for main-diagonal segments (constant r-c)
#   fill_rule:
#     - anti family: identical to the original task, fill between consecutive active k=r+c
#       segments while scanning each main diagonal b=r-c
#     - main family: symmetric rule, fill between consecutive active b=r-c segments while
#       scanning each anti-diagonal k=r+c
#     - filling is only across segments from the same color family; opposite-orientation
#       segments do not interact
#   generator_layout:
#     - to keep the new parallelism contrast legible, place the two families in two separated
#       7x7 windows, with the anti/main family assignment to the two windows randomized
#     - each window contains 3 short diagonal segments and must generate non-zero fill
#
# a) Concrete behavior / trace:
#   Original: every non-bg component lies on some k=r+c, so the solver can collect one line
#   color, assume every component has constant k, and only perform the "scan b, sort by k"
#   fill.
#   Variant example:
#     - blue components: (1,5)-(2,4)-(3,3) and (1,7)-(2,6)-(3,5), so both are anti-diagonal
#       segments with k=6 and k=8; they fill the bg cells strictly between them.
#     - green components: (10,1)-(11,2)-(12,3) and (8,1)-(9,2)-(10,3), so both are
#       main-diagonal segments with b=9 and b=7; they fill the bg cells strictly between
#       them along anti-diagonals.
#   Difference: the solver can no longer pick one dominant line color or assume "constant k"
#   for every component; it must decode each color's direction first.
#
# b) Complexity calibration:
#   - exactly two line colors keeps the color→direction rule observable
#   - 3 segments per family give repeated evidence for each family without making the grid
#     chaotic
#   - the families are separated spatially so red fill regions are readable and do not merge
#
# c) Implicit assumptions from the original main() that break here:
#   - "line = max(non_bg, key=count)" is invalid now because there are two line colors
#   - "_cc8(g == line)" is invalid because it ignores the second family entirely
#   - "len(ks) == 1" as the only accepted geometry is invalid for main-diagonal segments,
#     which instead satisfy len(bs) == 1
#   - only scanning b-lines and sorting by k is insufficient; the main-diagonal family needs
#     the symmetric "scan k-lines and sort by b" fill
#
# d) Placement legality:
#   - each local family is generated output-first inside a 7x7 window, using only contiguous
#     diagonal segments with in-bounds cells
#   - the two windows are disjoint, so line colors never overlap and the two fill systems do
#     not block each other except through the intentional bg-only fill rule
#
# e/f/g) Visual observability / cleanliness / color inference:
#   - every sample contains both orientations and both colors
#   - each color repeats across multiple segments, so the color family is easy to perceive
#   - window assignment is randomized, so location is not a fixed proxy for orientation
#   - the human-visible rule is: "same-colored parallel diagonal segments create red strips
#     between them; one color uses /-style diagonals and the other uses \\-style diagonals"


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _component_orientation(pts: list[tuple[int, int]]) -> str | None:
    if len(pts) < 2:
        return None
    ks = {r + c for r, c in pts}
    bs = {r - c for r, c in pts}
    if len(ks) == 1 and len(bs) != 1:
        return "anti"
    if len(bs) == 1 and len(ks) != 1:
        return "main"
    return None


def _extract_segments_for_color(comps: list[list[tuple[int, int]]]) -> tuple[str | None, list[tuple[int, int, int]]]:
    anti: list[tuple[int, int, int]] = []
    main: list[tuple[int, int, int]] = []

    for pts in comps:
        ori = _component_orientation(pts)
        if ori == "anti":
            k = pts[0][0] + pts[0][1]
            bs = [r - c for r, c in pts]
            anti.append((int(k), int(min(bs)), int(max(bs))))
        elif ori == "main":
            b = pts[0][0] - pts[0][1]
            ks = [r + c for r, c in pts]
            main.append((int(b), int(min(ks)), int(max(ks))))

    if anti and not main:
        return "anti", anti
    if main and not anti:
        return "main", main
    return None, []


def _fill_between_anti_segments(out: np.ndarray, bg: int, segs: list[tuple[int, int, int]], fill: int) -> None:
    h, w = out.shape
    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])
        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for k_left, k_right in zip(active, active[1:]):
            left = k_left[0]
            right = k_right[0]
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if left < k < right and out[r, c] == bg:
                    out[r, c] = fill


def _fill_between_main_segments(out: np.ndarray, bg: int, segs: list[tuple[int, int, int]], fill: int) -> None:
    h, w = out.shape
    for k in range(h + w - 1):
        active = [s for s in segs if s[1] <= k <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])
        r_min = max(0, k - (w - 1))
        r_max = min(h - 1, k)
        for b_low, b_high in zip(active, active[1:]):
            low = b_low[0]
            high = b_high[0]
            for r in range(r_min, r_max + 1):
                c = k - r
                b = r - c
                if low < b < high and out[r, c] == bg:
                    out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    D9 variant:
    - one line color encodes anti-diagonal segments (constant r+c)
    - another line color encodes main-diagonal segments (constant r-c)
    - each color family fills only between its own parallel segments
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    fill = 2

    line_colors = [c for c in sorted(set(map(int, g.ravel()))) if c not in (bg, fill)]
    assert line_colors, "grid must contain at least one non-background line color"

    anti_segs: list[tuple[int, int, int]] = []
    main_segs: list[tuple[int, int, int]] = []
    for color in line_colors:
        family, segs = _extract_segments_for_color(_cc8(g == color))
        if family == "anti":
            anti_segs.extend(segs)
        elif family == "main":
            main_segs.extend(segs)

    out = g.copy()
    _fill_between_anti_segments(out, bg, anti_segs, fill)
    _fill_between_main_segments(out, bg, main_segs, fill)
    return out


# --------------- generator ---------------

H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

SUB = 7
WINDOW_SLOTS = [(0, 9), (9, 0)]

LOCAL_K_LO, LOCAL_K_HI = 1, 11
LOCAL_GAP_VALUES = (4,)

SEG_LEN_MIN, SEG_LEN_MAX = 2, 5
OVERLAP_MIN_B = 2

SEG_RETRY = 50
LOCAL_GEN_RETRY = 120
GEN_RETRY = 400

MIN_LOCAL_FILL_CELLS = 2


def _b_bounds_for_k_local(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (SUB - 1))
    b_hi = min(2 * (SUB - 1) - k, k)
    return b_lo, b_hi


def _seg_cells_local(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    cells = []
    for b in range(b_min, b_max + 1, 2):
        cells.append(((k + b) // 2, (k - b) // 2))
    return cells


def _apply_fill_local(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    _fill_between_anti_segments(out, 0, segs, 2)
    return out


def _choose_local_k_values(parity: int) -> list[int] | None:
    gap_pairs = [(a, b) for a in LOCAL_GAP_VALUES for b in LOCAL_GAP_VALUES if a + b == 8]
    gaps = list(random.choice(gap_pairs))
    total_span = sum(gaps)
    k0_candidates = [k for k in range(LOCAL_K_LO, LOCAL_K_HI - total_span + 1) if k % 2 == parity]
    if not k0_candidates:
        return None

    k0 = random.choice(k0_candidates)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    if ks[-1] > LOCAL_K_HI or len(set(ks)) != len(ks):
        return None
    return ks


def _random_local_seg(k: int, constrain_lo: int | None = None, constrain_hi: int | None = None) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k_local(k)
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


def _gen_local_anti_pattern() -> np.ndarray | None:
    parity = random.randint(0, 1)
    ks = _choose_local_k_values(parity)
    if ks is None:
        return None

    segs: list[tuple[int, int, int]] = []
    first = _random_local_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1]))

    for k in ks[1:]:
        prev_bmin, prev_bmax = segs[-1][1], segs[-1][2]
        nxt = _random_local_seg(k, prev_bmin, prev_bmax)
        if nxt is None:
            return None
        segs.append((k, nxt[0], nxt[1]))

    grid = np.zeros((SUB, SUB), dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells_local(k, bmin, bmax):
            if not (0 <= r < SUB and 0 <= c < SUB):
                return None
            grid[r, c] = 1

    out = _apply_fill_local(grid, segs)
    if int(np.sum(out == 2)) < MIN_LOCAL_FILL_CELLS:
        return None
    return out


def _generate_local_family() -> np.ndarray:
    for _ in range(LOCAL_GEN_RETRY):
        out = _gen_local_anti_pattern()
        if out is not None:
            return out
    raise RuntimeError("Failed to generate local family")


def _paste_family(dest: np.ndarray, family: np.ndarray, top: int, left: int, line_color: int) -> None:
    h, w = family.shape
    for r in range(h):
        for c in range(w):
            val = int(family[r, c])
            if val == 1:
                dest[top + r, left + c] = line_color
            elif val == 2:
                dest[top + r, left + c] = FILL


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        anti_local = _generate_local_family()
        main_local = np.fliplr(_generate_local_family())

        anti_color, main_color = random.sample(LINE_COLORS, 2)
        anti_slot, main_slot = random.sample(WINDOW_SLOTS, 2)

        output = np.full((H, W), BG, dtype=int)
        _paste_family(output, anti_local, anti_slot[0], anti_slot[1], anti_color)
        _paste_family(output, main_local, main_slot[0], main_slot[1], main_color)

        inp = output.copy()
        inp[inp == FILL] = BG

        if int(np.sum(output == FILL)) < 2 * MIN_LOCAL_FILL_CELLS:
            continue
        if not np.any(output[anti_slot[0] : anti_slot[0] + SUB, anti_slot[1] : anti_slot[1] + SUB] == FILL):
            continue
        if not np.any(output[main_slot[0] : main_slot[0] + SUB, main_slot[1] : main_slot[1] + SUB] == FILL):
            continue
        if not np.array_equal(main(inp), output):
            continue
        return inp
    raise RuntimeError("Failed to generate input")
