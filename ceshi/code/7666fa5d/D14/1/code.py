# Variant of 7666fa5d
# Dimension: D14 — 符号编码：控制信息 (symbol_instructions)
# Change: 无符号编码——任务规则完全由空间几何关系隐含定义，无图例、标记像素或编码控制参数... → 标记像素指示填充方向

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


"""
design_analysis:
  variant_visual:
    grid: 16x16
    background: 8
    line_color: one non-bg color, never 2
    marker_color: 2 (red), always a singleton pixel in the input
    objects:
      - 4-5 anti-diagonal line segments on distinct k=r+c values
      - each segment may have:
          * one red pixel exactly one row above one segment cell
          * one red pixel exactly one row below one segment cell
          * or no marker
    marker_meaning:
      above: fill toward smaller k (the side where the marker sits)
      below: fill toward larger k
      none: that segment initiates no fill

  behavior_trace:
    original:
      - if segments at k=8,12,16 are active on a b-line, both gaps (8,12) and (12,16)
        are always filled, because the solver fills every adjacent active pair.
    variant:
      - if k=8 has a below-marker, k=12 has no marker, and k=16 has an above-marker:
        * gap (8,12) fills because the left segment points into it
        * gap (12,16) fills because the right segment points into it
      - if k=8 has no marker and k=12 has a below-marker:
        * gap (8,12) does not fill, because neither boundary points into that gap
    difference:
      - fillability is no longer determined by geometry alone; the singleton red markers
        select which side of each segment is "open" for filling.

  complexity_calibration:
    control_states_per_segment: [above, none, below]
    kept_simple:
      - only one marker pixel per segment at most
      - marker direction has only 3 states
      - output color remains the original fill red (2)
    anti_degenerate_requirements_in_generator:
      - at least one fillable geometric gap is suppressed by markers
      - at least one gap is still filled
      - at least one segment has no marker
      - both directions (above and below) appear

  hidden_assumptions_review:
    old_main_line_choice:
      code: "line = max(non_bg, key=count)"
      dependency: assumed the only non-bg color is the line color
      variant_status: broken, because red markers are also non-bg
      fix: choose the unique non-bg color other than 2 as the line color
    old_main_fill_loop:
      code: filled every adjacent active (k_left, k_right) pair
      dependency: assumed every adjacent pair should fill
      variant_status: broken
      fix: fill a gap only if the left segment points right or the right segment points left
    marker_detection:
      dependency: red in the input must mean marker, not already-filled region
      fix: only singleton red components count as markers, so rerunning on an output
        would not reinterpret large red regions as instructions

  placement_legality:
    segments:
      - stay fully in bounds
      - consecutive k-values still have overlapping b-ranges, preserving the original
        parallelogram geometry
    markers:
      - are placed only on empty cells
      - are exactly vertical neighbors of their owning segment
      - never share cells with lines or other markers
      - extreme segments only point inward or stay unmarked, avoiding useless outward markers

  visual_observability:
    rationale:
      - the input now visibly contains tiny red symbols near some segments
      - two otherwise similar gaps can behave differently depending on the marker side
      - examples will contain both "fills" and "doesn't fill despite similar geometry"
        cases, so humans can infer the symbolic instruction

  visual_cleanliness:
    decisions:
      - no same-k split segments in this variant generator, because marker ownership becomes
        visually noisier while not helping D14
      - only singleton red markers are added, so the scene stays close to the original task

  color_inference:
    colors:
      - line color varies as before
      - red has a stable semantic role: input marker + output fill
    inference_support:
      - red is not used for anything else
      - direction comes from position relative to a line, not from hidden numeric ordering
"""


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _singleton_red_markers(g: np.ndarray) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for comp in components_mask(g == FILL, connectivity=8):
        if len(comp) == 1:
            r, c = comp[0]
            markers.append((int(r), int(c)))
    return markers


def _segment_triples(pts_list: list[list[tuple[int, int]]]) -> list[tuple[int, int, int]]:
    segs: list[tuple[int, int, int]] = []
    for pts in pts_list:
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _segment_dirs(
    g: np.ndarray, pts_list: list[list[tuple[int, int]]]
) -> list[int]:
    cell_to_seg: dict[tuple[int, int], int] = {}
    for idx, pts in enumerate(pts_list):
        for cell in pts:
            cell_to_seg[cell] = idx

    dirs = [0] * len(pts_list)
    ambiguous: set[int] = set()
    for r, c in _singleton_red_markers(g):
        hits: list[tuple[int, int]] = []
        above_owner = cell_to_seg.get((r + 1, c))
        if above_owner is not None:
            hits.append((above_owner, -1))
        below_owner = cell_to_seg.get((r - 1, c))
        if below_owner is not None:
            hits.append((below_owner, 1))
        if len(hits) != 1:
            continue
        idx, direction = hits[0]
        if dirs[idx] == 0:
            dirs[idx] = direction
        elif dirs[idx] != direction:
            ambiguous.add(idx)
    for idx in ambiguous:
        dirs[idx] = 0
    return dirs


def _fill_directional(
    grid: np.ndarray, segs: list[tuple[int, int, int]], dirs: list[int]
) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = [
            (k, b_min, b_max, dirs[idx])
            for idx, (k, b_min, b_max) in enumerate(segs)
            if b_min <= b <= b_max
        ]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left, _bl0, _bl1, d_left = left
            k_right, _br0, _br1, d_right = right
            if k_left >= k_right:
                continue
            if d_left != 1 and d_right != -1:
                continue
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def _fill_baseline(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for (k_left, _a0, _a1), (k_right, _b0, _b1) in zip(active, active[1:]):
            if k_left >= k_right:
                continue
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Diagonal line segments still lie on anti-diagonals k=r+c.
    - A singleton red marker above a segment means: fill from that segment toward smaller k.
    - A singleton red marker below a segment means: fill toward larger k.
    - No marker means that segment does not trigger fill.
    - For each b=r-c, a gap between adjacent active segments fills iff one of its boundary
      segments points into that gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    colors = sorted(set(map(int, g.ravel())))
    line_candidates = [c for c in colors if c not in (bg, FILL)]
    assert line_candidates, "grid must contain a non-red line color"
    line = max(line_candidates, key=lambda c: int(np.sum(g == c)))

    comps = _cc8(g == line)
    segs = _segment_triples(comps)
    dirs = _segment_dirs(g, comps)
    return _fill_directional(g, segs, dirs)


# --------------- generator ---------------

H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_B = 2

SEG_RETRY = 40
DIR_RETRY = 80
GEN_RETRY = 400

MIN_NEW_FILL = 4
MIN_SUPPRESSED_FILL = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


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
    starts = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not starts:
        return None

    k0 = random.choice(starts)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    if ks[-1] > K_HI:
        return None
    return ks


def _random_seg(
    k: int, constrain_lo: int | None = None, constrain_hi: int | None = None
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
                if b_min + step >= constrain_lo + OVERLAP_MIN_B
                and b_min <= constrain_hi - OVERLAP_MIN_B
            ]
        if not candidates:
            continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _sample_dirs(num_segs: int) -> list[int] | None:
    for _ in range(DIR_RETRY):
        dirs: list[int] = []
        for idx in range(num_segs):
            if idx == 0:
                dirs.append(random.choice([0, 1]))
            elif idx == num_segs - 1:
                dirs.append(random.choice([-1, 0]))
            else:
                dirs.append(random.choice([-1, 0, 1]))

        gap_fills = [dirs[i] == 1 or dirs[i + 1] == -1 for i in range(num_segs - 1)]
        if not any(gap_fills):
            continue
        if all(gap_fills):
            continue
        if 0 not in dirs or 1 not in dirs or -1 not in dirs:
            continue
        return dirs
    return None


def _pick_marker_cell(
    seg: tuple[int, int, int], direction: int, occupied: set[tuple[int, int]]
) -> tuple[int, int] | None:
    k, b_min, b_max = seg
    cells = _seg_cells(k, b_min, b_max)
    mid = len(cells) // 2
    order = sorted(range(len(cells)), key=lambda idx: (abs(idx - mid), idx))
    for idx in order:
        r, c = cells[idx]
        mr = r - 1 if direction == -1 else r + 1
        if 0 <= mr < H and (mr, c) not in occupied:
            return mr, c
    return None


def _place_markers(
    segs: list[tuple[int, int, int]], dirs: list[int]
) -> list[tuple[int, int]] | None:
    occupied = {cell for seg in segs for cell in _seg_cells(*seg)}
    markers: list[tuple[int, int]] = []
    for seg, direction in zip(segs, dirs):
        if direction == 0:
            continue
        marker = _pick_marker_cell(seg, direction, occupied)
        if marker is None:
            return None
        occupied.add(marker)
        markers.append(marker)
    return markers


def _variant_counts(
    inp: np.ndarray, out: np.ndarray, baseline: np.ndarray
) -> tuple[int, int]:
    new_fill = int(np.sum((out == FILL) & (inp == BG)))
    suppressed = int(np.sum((baseline == FILL) & (out != FILL) & (inp == BG)))
    return new_fill, suppressed


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
        prev_b_min, prev_b_max = segs[-1][1], segs[-1][2]
        nxt = _random_seg(k, prev_b_min, prev_b_max)
        if nxt is None:
            return None
        segs.append((k, nxt[0], nxt[1]))

    dirs = _sample_dirs(len(segs))
    if dirs is None:
        return None

    markers = _place_markers(segs, dirs)
    if markers is None:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            inp[r, c] = line
    for r, c in markers:
        if inp[r, c] != BG:
            return None
        inp[r, c] = FILL

    out = _fill_directional(inp, segs, dirs)
    baseline = _fill_baseline(inp, segs)
    new_fill, suppressed = _variant_counts(inp, out, baseline)

    if new_fill < MIN_NEW_FILL:
        return None
    if suppressed < MIN_SUPPRESSED_FILL:
        return None
    if np.array_equal(out, baseline):
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
