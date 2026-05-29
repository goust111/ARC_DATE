# Variant of 7666fa5d
# Dimension: D14 — 符号编码：控制信息 (symbol_instructions)
# Change: 无符号编码——任务规则完全由空间几何关系隐含定义，无图例、标记像素或编码控制参数
#   → 控制像素指示线段连接顺序

import itertools
import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color


"""
design_analysis:
  variant_goal:
    keep:
      - 16x16 grid
      - one line color + red fill color 2 + bg color 8
      - interior objects are anti-diagonal segments on k=r+c
      - filling still runs along main diagonals b=r-c
    change:
      - original task paired segments implicitly by geometric k-order
      - new task decodes an explicit border order, then pairs consecutive controls two-by-two

  a_concretization_and_trace:
    input_objects:
      - 4-5 interior diagonal segments, all in one non-bg color
      - 4-5 border singleton control pixels in the same color
      - each control pixel lies on the same anti-diagonal k=r+c as exactly one segment
      - reading border controls clockwise gives a global segment order
    output_behavior:
      - read border controls clockwise
      - pair control 1 with 2, 3 with 4, ...
      - for each pair, fill bg cells between the two referenced segments on the b-lines where
        both segments are active
    concrete_trace:
      - suppose the interior segments have k-values [6, 10, 18, 22]
      - original logic fills across the whole chain 6↔10↔18↔22
      - if border controls decode to [10, 22, 6, 18], then the explicit pairs are (10,22)
        and (6,18)
      - therefore the red regions now depend on the border code; geometry alone is insufficient

  b_complexity_calibration:
    choices:
      - use exactly 4 segments so the border order cleanly defines 2 explicit pairs
      - keep unique k per segment; no same-k split components in this variant
      - force all segments to share at least one common b-value so any chosen pair can produce
        a visible fill region
      - use one control pixel per segment so the code is repeated across the sample, not a
        single fragile marker
    why_not_simpler:
      - if only one pair can overlap, border order becomes weak or partly decorative
      - if border pixels are not tied to segment identity, mapping becomes ambiguous
      - if segments may touch the border, controls become visually indistinguishable from lines

  c_implicit_assumption_audit_for_original_main:
    original_lines:
      - line = most common non-bg color
        still valid because controls and segments intentionally share one color
      - comps = connected components of all line pixels
        no longer valid if controls are included, because border singletons are not segments
        fix: exclude border line pixels before component extraction
      - seg identity = component's constant k
        still valid, but now each k must be unique because controls refer to k
      - active.sort(key=lambda t: t[0])
        invalid in variant: control pixels now define explicit pair membership
      - fill between adjacent active k-values
        invalid in variant: only the two segments assigned to the same control-pair may connect

  d_placement_legality:
    generator_guarantees:
      - all true segments stay strictly inside the border, so border-color cells are controls only
      - each segment uses a unique k and one contiguous b-interval
      - each chosen control pixel is one of the two border intersections of that segment's
        anti-diagonal, so the segment-control mapping is exact
      - all segments share a common b-anchor, so multiple segments are active together
      - the selected control layout must make variant output differ from geometric-order output

  e_visual_observability:
    visible_rule:
      - border dots clearly act as instructions because they are isolated, lie on the frame,
        and share color with the interior segments they refer to
      - the clockwise order around the frame is visually available without using color ids
      - the output differs in a way humans can inspect: two red regions bridge the two border-
        indicated pairs, even when those pairs are not neighbors by k

  f_visual_cleanliness:
    cleanup_decisions:
      - border controls are singletons, not bars or multi-color legends
      - segments never overlap controls or the border
      - only one instruction channel exists: same-color border points on empty frame cells
      - red fill never overwrites line/control pixels

  g_color_inference:
    colors:
      - line/control: one sampled color from the original palette excluding 2 and 8
      - fill: fixed red 2
      - bg: fixed azure 8
    inference_reason:
      - no hidden numeric color ordering is used; only visible geometry and border order matter

  h_new_main_assumptions_and_generator_support:
    main_assumptions:
      - every segment has exactly one matching border control with the same k
      - the number of segments / controls is even, so the border order can be chunked into pairs
      - border line pixels are controls, not segment continuations
      - k-values are unique among segments
      - paired segments overlap in b enough to matter
    generator_support:
      - unique-k generation enforces one control per segment
      - interior-only b-bounds keep segments off the border
      - shared b-anchor enforces simultaneous activity
      - control-layout search rejects cases where variant == original
"""


H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 6, 24
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS = 4

SEG_LEN_MIN, SEG_LEN_MAX = 2, 8
SEG_RETRY = 50
GEN_RETRY = 400
MIN_FILL_CELLS = 4
MIN_DIFF_FROM_NATURAL = 4


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[h - 1, :] = True
    mask[:, 0] = True
    mask[:, w - 1] = True
    return mask


def _perimeter_index(r: int, c: int, h: int, w: int) -> int:
    if r == 0:
        return c
    if c == w - 1:
        return (w - 1) + r
    if r == h - 1:
        return (w - 1) + (h - 1) + (w - 1 - c)
    return (w - 1) + (h - 1) + (w - 1) + (h - 1 - r)


def _edge_name(r: int, c: int, h: int, w: int) -> str:
    if r == 0:
        return "top"
    if c == w - 1:
        return "right"
    if r == h - 1:
        return "bottom"
    return "left"


def _border_points_for_k(k: int) -> list[tuple[int, int]]:
    if k <= W - 1:
        return [(0, k), (k, 0)]
    return [(k - (W - 1), W - 1), (H - 1, k - (H - 1))]


def _b_bounds_for_k_interior(k: int) -> tuple[int, int]:
    b_lo = max(2 - k, k - 2 * (W - 2))
    b_hi = min(2 * (H - 2) - k, k - 2)
    return int(b_lo), int(b_hi)


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _collect_segments(g: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    border = _border_mask(h, w)
    comps = _cc8((g == line) & (~border))
    segs: list[tuple[int, int, int]] = []
    for pts in comps:
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _control_order_ks(ctrl_pts: list[tuple[int, int]], seg_keys: set[int], h: int, w: int) -> list[int]:
    ordered = sorted(ctrl_pts, key=lambda rc: _perimeter_index(rc[0], rc[1], h, w))
    seen: set[int] = set()
    out: list[int] = []
    for r, c in ordered:
        k = int(r + c)
        if k in seg_keys and k not in seen:
            out.append(k)
            seen.add(k)
    return out


def _fill_between_two_segments(
    grid: np.ndarray,
    seg_a: tuple[int, int, int],
    seg_b: tuple[int, int, int],
    *,
    bg: int,
    fill: int,
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    k_left, b_min_left, b_max_left = map(int, seg_a)
    k_right, b_min_right, b_max_right = map(int, seg_b)
    k_lo, k_hi = sorted((k_left, k_right))
    b_lo = max(b_min_left, b_min_right)
    b_hi = min(b_max_left, b_max_right)
    if b_lo > b_hi:
        return out
    if b_lo % 2 != k_left % 2:
        b_lo += 1
    if b_hi % 2 != k_left % 2:
        b_hi -= 1
    if b_lo > b_hi:
        return out

    for b in range(b_lo, b_hi + 1, 2):
        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for r in range(r_min, r_max + 1):
            c = r - b
            k = r + c
            if k_lo < k < k_hi and out[r, c] == bg:
                out[r, c] = fill
    return out


def _apply_fill_control_pairs(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    order_ks: list[int],
    *,
    bg: int,
    fill: int,
) -> np.ndarray:
    out = grid.copy()
    seg_by_k = {int(k): (int(k), int(b_min), int(b_max)) for k, b_min, b_max in segs}
    assert len(order_ks) % 2 == 0
    for i in range(0, len(order_ks), 2):
        pair_out = _fill_between_two_segments(
            out,
            seg_by_k[order_ks[i]],
            seg_by_k[order_ks[i + 1]],
            bg=bg,
            fill=fill,
        )
        out = pair_out
    return out


def _apply_fill_original(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    *,
    bg: int,
    fill: int,
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for b in range(-(w - 1), h):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain one non-background line/control color"
    line = max(non_bg, key=lambda c: int(np.sum(g == c)))

    border = _border_mask(h, w)
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere((g == line) & border)]
    assert ctrl_pts, "variant requires border control pixels"

    segs = _collect_segments(g, int(line))
    assert segs, "variant requires interior anti-diagonal segments"

    seg_keys = [int(k) for k, _, _ in segs]
    assert len(seg_keys) == len(set(seg_keys)), "each segment must have a unique anti-diagonal k"

    order_ks = _control_order_ks(ctrl_pts, set(seg_keys), h, w)
    assert len(order_ks) == len(segs), "every segment must have exactly one matching control pixel"
    assert len(order_ks) % 2 == 0, "control order must define complete pairs"

    return _apply_fill_control_pairs(g, segs, order_ks, bg=int(bg), fill=FILL)


def _choose_k_values(parity: int) -> list[int] | None:
    for _ in range(60):
        valid_combos = [
            (a, b, c)
            for a in GAP_VALUES
            for b in GAP_VALUES
            for c in GAP_VALUES
            if a + b + c in GAP_TARGET_SPANS
        ]
        gaps = list(random.choice(valid_combos))

        total_span = sum(gaps)
        k0_candidates = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
        random.shuffle(k0_candidates)
        for k0 in k0_candidates:
            ks = [k0]
            for gap in gaps:
                ks.append(ks[-1] + gap)
            if ks[-1] > K_HI:
                continue
            if 15 in ks:
                continue
            return ks
    return None


def _choose_common_b_anchor(ks: list[int]) -> int | None:
    parity = ks[0] % 2
    lo = max(_b_bounds_for_k_interior(k)[0] for k in ks)
    hi = min(_b_bounds_for_k_interior(k)[1] for k in ks)
    if lo % 2 != parity:
        lo += 1
    if hi % 2 != parity:
        hi -= 1
    if lo > hi:
        return None
    return random.choice(list(range(lo, hi + 1, 2)))


def _random_seg_covering(k: int, b_anchor: int) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k_interior(k)
    max_len = (b_hi - b_lo) // 2 + 1
    if max_len < SEG_LEN_MIN:
        return None

    for _ in range(SEG_RETRY):
        length = random.randint(SEG_LEN_MIN, min(SEG_LEN_MAX, max_len))
        step = 2 * (length - 1)
        start_lo = max(b_lo, b_anchor - step)
        start_hi = min(b_anchor, b_hi - step)
        if start_lo % 2 != k % 2:
            start_lo += 1
        if start_hi % 2 != k % 2:
            start_hi -= 1
        if start_lo > start_hi:
            continue
        b_min = random.choice(list(range(start_lo, start_hi + 1, 2)))
        b_max = b_min + step
        if b_min <= b_anchor <= b_max:
            return int(b_min), int(b_max)
    return None


def _max_active_depth(segs: list[tuple[int, int, int]]) -> int:
    best = 0
    for b in range(-(W - 1), H):
        best = max(best, sum(1 for _, b_min, b_max in segs if b_min <= b <= b_max))
    return best


def _choose_control_layout(
    base_grid: np.ndarray,
    segs: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int]], list[int]] | None:
    ks = [int(k) for k, _, _ in segs]
    natural_order = sorted(ks)
    natural_out = _apply_fill_original(base_grid, segs, bg=BG, fill=FILL)
    sorted_rank = {k: i for i, k in enumerate(natural_order)}

    best: tuple[tuple[int, int, int], list[tuple[int, int]], list[int]] | None = None
    candidates = [_border_points_for_k(k) for k in ks]

    for picks in itertools.product([0, 1], repeat=len(ks)):
        ctrl_pts = [candidates[i][pick] for i, pick in enumerate(picks)]
        order_ks = _control_order_ks(ctrl_pts, set(ks), H, W)
        if len(order_ks) != len(ks):
            continue
        pair_set = {
            tuple(sorted((order_ks[i], order_ks[i + 1])))
            for i in range(0, len(order_ks), 2)
        }
        natural_pair_set = {
            tuple(sorted((natural_order[i], natural_order[i + 1])))
            for i in range(0, len(natural_order), 2)
        }
        if pair_set == natural_pair_set:
            continue

        variant_out = _apply_fill_control_pairs(base_grid, segs, order_ks, bg=BG, fill=FILL)
        diff = int(np.sum(variant_out != natural_out))
        if diff < MIN_DIFF_FROM_NATURAL:
            continue

        nonlocal_pairs = sum(
            1
            for i in range(0, len(order_ks), 2)
            for a, b in [(order_ks[i], order_ks[i + 1])]
            if abs(sorted_rank[a] - sorted_rank[b]) != 1
        )
        edge_count = len({_edge_name(r, c, H, W) for r, c in ctrl_pts})
        score = (diff, nonlocal_pairs, edge_count)
        if best is None or score > best[0]:
            best = (score, ctrl_pts, order_ks)

    if best is None:
        return None
    return best[1], best[2]


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)

    ks = _choose_k_values(parity)
    if ks is None:
        return None
    if len(ks) != NUM_KS:
        return None

    b_anchor = _choose_common_b_anchor(ks)
    if b_anchor is None:
        return None

    segs: list[tuple[int, int, int]] = []
    for k in ks:
        seg = _random_seg_covering(k, b_anchor)
        if seg is None:
            return None
        segs.append((int(k), int(seg[0]), int(seg[1])))

    if _max_active_depth(segs) < 4:
        return None

    base_grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if not (1 <= r < H - 1 and 1 <= c < W - 1):
                return None
            if base_grid[r, c] != BG:
                return None
            base_grid[r, c] = line

    control_layout = _choose_control_layout(base_grid, segs)
    if control_layout is None:
        return None
    ctrl_pts, order_ks = control_layout

    inp = base_grid.copy()
    for r, c in ctrl_pts:
        if inp[r, c] != BG:
            return None
        inp[r, c] = line

    output = main(inp)
    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        return None

    natural_out = _apply_fill_original(inp, segs, bg=BG, fill=FILL)
    if int(np.sum(output != natural_out)) < MIN_DIFF_FROM_NATURAL:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate D14 control-order variant input")
