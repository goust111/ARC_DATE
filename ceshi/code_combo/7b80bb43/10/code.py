from __future__ import annotations

# Combo variant of 7b80bb43 (index 10)
# combo_spec: D2/1+D11/2
# Name: 颜色双编码提示
#
# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    - Original task: a 3-cell gap on a line is hinted by same-color diagonal fg pixels,
#      so the solver must first detect "diagonal-only fg" by topology and then infer
#      whether to refill a row-gap or a column-gap from nearby line fragments.
#    - Fused task: the diagonal hint pixels are no longer fg. Red markers mean
#      "repair a horizontal gap"; blue markers mean "repair a vertical gap".
#      Example: if row 8 has fg at cols 2..4 and 8..11 with bg at 5..7, then a red
#      chain at (9,5),(10,6) is deleted and authorizes filling row 8 cols 5..7.
#      A blue chain next to a broken column behaves analogously for a vertical fill.
#    - Difference: hint discovery is now color-driven rather than topology-driven,
#      and the marker color, not the staircase shape, selects the fill axis.
# b) Complexity calibration:
#    - Keep only two marker colors with fixed semantics: red=horizontal, blue=vertical.
#    - Preserve the original gap size (3) and chain size (2) so the visual language
#      stays close to the seed task while the multicolor rule is clearly visible.
#    - Every generated sample contains at least one red chain and one blue chain, so
#      the color-direction mapping has repeated support instead of appearing once.
# c) Hidden assumptions broken by the fusion:
#    - `fg = next(iter(colors))` is invalid once markers add extra non-bg colors.
#    - `mask = (g == fg)` can no longer find hints, because hints are not fg anymore.
#    - "remove diagonal-only fg" would miss every real hint. The new solver instead
#      treats the dominant non-bg color as the line color and removes red/blue markers
#      by color, then fills only the axis associated with each marker color.
# d) Placement legality:
#    - The generator reserves ARC red(2) and blue(1) for markers and picks bg/fg from
#      the remaining palette, so color roles never collide.
#    - Gap chains are only placed on verified bg cells, cannot overlap each other, and
#      must still satisfy the old geometric legality: no orthogonal fg neighbor and at
#      least one diagonal fg-or-chain neighbor.
# e) Visual observability:
#    - The two marker colors are bright canonical ARC colors and appear beside both row
#      and column repairs. A human can compare 2-4 examples and see that red markers
#      always disappear while closing horizontal breaks, blue markers always disappear
#      while closing vertical breaks.
# f) Visual cleanliness:
#    - Free-floating distractor staircases are removed in this combo. Keeping them would
#      reintroduce topology-only noise and muddy the meaning of the new marker colors.
#      The grid therefore contains just the line network plus the color-coded hints.
# g) New solver assumptions and generator guarantees:
#    - Assumption: the true network color is the dominant non-bg color.
#      Guarantee: markers are only 2 pixels per gap, while the line network is much
#      larger, so fg dominates non-bg counts.
#    - Assumption: a valid repair gap has a matching marker within Chebyshev-1 of the
#      gap span and the wrong-color markers should be ignored.
#      Guarantee: each gap is generated from the original validated placements and
#      colored according to its intended repair axis.

import random
import numpy as np

from arc_utils import draw_line, find_components, get_bg


MARK_H = 2  # red = repair horizontal gap
MARK_V = 1  # blue = repair vertical gap
MARKER_COLORS = {MARK_H, MARK_V}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete color-coded diagonal hints and reconnect the broken line network.

    Input structure:
    - bg is the most common color. One dominant non-bg color fg forms the axis-aligned line network.
    - Red staircase markers indicate horizontal repairs; blue staircase markers indicate vertical repairs.
    - Marker pixels are separate from fg and sit diagonally near 3-cell gaps.

    Rule:
    - Remove every red/blue marker pixel from the grid.
    - Fill a short horizontal gap only when a red marker lies next to that gap; fill a short vertical gap only when a blue marker lies next to that gap.
    - After filling, prune any remaining diagonal leaf fg pixels in the former marker neighborhoods and output a single-color clean network.

    Key insight: Marker color now carries the missing orientation, so the solver must treat red and blue as semantic instructions rather than as ordinary foreground noise.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    colors, counts = np.unique(g, return_counts=True)
    non_bg = [(int(col), int(cnt)) for col, cnt in zip(colors, counts) if int(col) != int(bg)]
    assert non_bg, "grid must contain a non-background color"
    fg = max(non_bg, key=lambda item: item[1])[0]

    base = g == int(fg)
    h_hint = g == int(MARK_H)
    v_hint = g == int(MARK_V)
    any_hint = h_hint | v_hint

    def deg4(mask: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(mask[r - 1, c]):
            d += 1
        if r + 1 < H and bool(mask[r + 1, c]):
            d += 1
        if c > 0 and bool(mask[r, c - 1]):
            d += 1
        if c + 1 < W and bool(mask[r, c + 1]):
            d += 1
        return int(d)

    def has_diag(mask: np.ndarray, r: int, c: int) -> bool:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
                return True
        return False

    near = np.zeros((H, W), dtype=bool)
    for r, c in np.argwhere(any_hint):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    near[rr, cc] = True

    repaired = base.copy()

    # Red markers authorize only horizontal repairs.
    for r in range(H):
        cols = np.where(base[r, :])[0]
        if cols.size == 0:
            continue
        segs: list[tuple[int, int]] = []
        start = int(cols[0])
        prev = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c == prev + 1:
                prev = c
            else:
                segs.append((int(start), int(prev)))
                start = prev = c
        segs.append((int(start), int(prev)))

        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0 or int(gap1 - gap0 + 1) != 3:
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(H - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(W - 1), int(gap1 + 1))
            if bool(h_hint[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                repaired[r, gap0 : gap1 + 1] = True

    # Blue markers authorize only vertical repairs.
    for c in range(W):
        rows = np.where(base[:, c])[0]
        if rows.size == 0:
            continue
        segs: list[tuple[int, int]] = []
        start = int(rows[0])
        prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r == prev + 1:
                prev = r
            else:
                segs.append((int(start), int(prev)))
                start = prev = r
        segs.append((int(start), int(prev)))

        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0 or int(gap1 - gap0 + 1) != 3:
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(H - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(W - 1), int(c + 1))
            if bool(v_hint[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                repaired[gap0 : gap1 + 1, c] = True

    # Conservative local pruning near former marker sites.
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not bool(repaired[r, c]) or not bool(near[r, c]):
                    continue
                if deg4(repaired, int(r), int(c)) != 1:
                    continue
                if not has_diag(repaired, int(r), int(c)):
                    continue
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                repaired[r, c] = False

    out = np.full((H, W), int(bg), dtype=int)
    out[repaired] = int(fg)
    return out


# --------------- generator ---------------

# Grid dimensions
GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

# Segment network topology (h + v = segments; junctions = T-junction count)
VALID_CONFIGS = [
    {"segments": 6, "h": 2, "v": 4, "junctions": 2},
    {"segments": 6, "h": 3, "v": 3, "junctions": 2},
    {"segments": 6, "h": 3, "v": 3, "junctions": 3},
    {"segments": 7, "h": 2, "v": 5, "junctions": 2},
    {"segments": 7, "h": 3, "v": 4, "junctions": 2},
    {"segments": 7, "h": 4, "v": 3, "junctions": 2},
    {"segments": 7, "h": 3, "v": 4, "junctions": 3},
    {"segments": 7, "h": 4, "v": 3, "junctions": 3},
    {"segments": 8, "h": 3, "v": 5, "junctions": 3},
    {"segments": 8, "h": 4, "v": 4, "junctions": 3},
    {"segments": 8, "h": 5, "v": 3, "junctions": 3},
]
T_SLOTS = ("LT", "RT", "LB", "RB")

# Layout spacing
EDGE_MARGIN = 3
ROW_COL_GAP = 2
T_BRANCH_OFFSET_MIN = 3
T_BRANCH_OFFSET_MAX = 9
H_SEG_END_MARGIN = 4
H_SEG_MAX_END = 25

# Simple H-segment sizing
SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

# Gap / chain placement
GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

# Retry budget
GEN_RETRIES = 300


def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int = ROW_COL_GAP) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - p) >= gap for p in picked):
            picked.append(v)
            if len(picked) == k:
                return sorted(picked)
    return []


def _h_chain_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + i, gap_start - 1 + i) for i in range(1, CHAIN_LENGTH + 1)]
    return [(row + i, gap_start + GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]
    return [(gap_start + GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    fg: int,
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg):
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg)):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int, branch_col: int, end_col: int) -> tuple[dict, dict]:
    H, W = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else W - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else H - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}}
    v_seg = {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    W = out.shape[1]
    c0 = 0 if side == "left" else W - length
    c1 = length - 1 if side == "left" else W - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": row, "c0": c0, "c1": c1, "blocked": set()}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    H = out.shape[0]
    r0 = 0 if side == "top" else H - length
    r1 = length - 1 if side == "top" else H - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": col, "r0": r0, "r1": r1, "blocked": set()}


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _h_chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((g, anchor))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                options.append((g, anchor, dc))
    return options


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int, cfg: dict) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), bg, dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    h_simple = cfg["h"] - cfg["junctions"]
    v_simple = cfg["v"] - cfg["junctions"]
    t_slots = rng.sample(list(T_SLOTS), cfg["junctions"])

    top_slots = [s for s in t_slots if s[1] == "T"]
    bottom_slots = [s for s in t_slots if s[1] == "B"]
    top_pool = list(range(EDGE_MARGIN, max(EDGE_MARGIN + 1, h // 2 - 2)))
    bottom_pool = list(range(max(h // 2 + 1, EDGE_MARGIN + 2), h - EDGE_MARGIN))
    top_rows = _pick_distinct(rng, top_pool, len(top_slots))
    bottom_rows = _pick_distinct(rng, bottom_pool, len(bottom_slots))
    if len(top_rows) != len(top_slots) or len(bottom_rows) != len(bottom_slots):
        return None
    row_by_slot = dict(zip(top_slots, top_rows)) | dict(zip(bottom_slots, bottom_rows))

    branch_cols = _pick_distinct(rng, list(range(EDGE_MARGIN, w - EDGE_MARGIN)), len(t_slots))
    if len(branch_cols) != len(t_slots):
        return None
    rng.shuffle(branch_cols)
    branch_by_slot = dict(zip(t_slots, branch_cols))

    for slot in t_slots:
        branch_col = branch_by_slot[slot]
        row = row_by_slot[slot]
        if slot[0] == "L":
            min_end = max(SIMPLE_H_LEN_MIN + 1, branch_col + T_BRANCH_OFFSET_MIN)
            max_end = min(w - H_SEG_END_MARGIN, branch_col + T_BRANCH_OFFSET_MAX, H_SEG_MAX_END)
        else:
            min_end = max(2, branch_col - T_BRANCH_OFFSET_MAX, w - H_SEG_MAX_END - 1)
            max_end = min(w - SIMPLE_H_LEN_MIN - 1, branch_col - T_BRANCH_OFFSET_MIN)
        if min_end > max_end:
            return None
        end_col = rng.randint(min_end, max_end)
        h_seg, v_seg = _place_t_component(out, fg, slot, row, branch_col, end_col)
        h_segments.append(h_seg)
        v_segments.append(v_seg)

    center_start = max(top_rows + [1]) + 2
    center_end = min(bottom_rows + [h - 2]) - 2
    if h_simple > 0 and center_start > center_end:
        return None

    simple_h_rows = _pick_distinct(rng, list(range(center_start, center_end + 1)), h_simple) if h_simple else []
    if len(simple_h_rows) != h_simple:
        return None
    if h_simple == 0:
        h_sides: list[str] = []
    elif h_simple == 1:
        h_sides = [rng.choice(["left", "right"])]
    else:
        h_sides = ["left", "right"]
        rng.shuffle(h_sides)

    simple_v_cols: list[int] = []
    used_branch_cols = set(branch_by_slot.values())
    if v_simple:
        has_left = "left" in h_sides
        has_right = "right" in h_sides
        if has_left and has_right:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_NEAR))
        elif has_left:
            col_pool = list(range(SIMPLE_V_MARGIN_NEAR, w - SIMPLE_V_MARGIN_FAR))
        elif has_right:
            col_pool = list(range(2, w - SIMPLE_V_MARGIN_NEAR))
        else:
            col_pool = list(range(2, w - 2))
        col_pool = [c for c in col_pool if c not in used_branch_cols]
        simple_v_cols = _pick_distinct(rng, col_pool, v_simple)
        if len(simple_v_cols) != v_simple:
            return None

    for row, side in zip(simple_h_rows, h_sides):
        if side == "left":
            max_len = (min(simple_v_cols) if simple_v_cols else w - EDGE_MARGIN) - 1
        else:
            max_len = w - (max(simple_v_cols) if simple_v_cols else 2) - 2
        max_len = min(SIMPLE_H_LEN_MAX, max_len)
        if max_len < SIMPLE_H_LEN_MIN:
            return None
        h_segments.append(_place_simple_h(out, fg, side, row, rng.randint(SIMPLE_H_LEN_MIN, max_len)))

    top_v = min(2, v_simple)
    top_v_cols = simple_v_cols[:top_v]
    bottom_v_cols = simple_v_cols[top_v:]
    top_limit = min(min(top_rows) - 2 if top_rows else center_start - 2, center_start - 2)
    bottom_start = max(max(bottom_rows) + 2 if bottom_rows else center_end + 2, center_end + 2)
    for col in top_v_cols:
        if top_limit < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "top", col, rng.randint(2, top_limit)))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "bottom", col, rng.randint(2, max_len)))

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    return out, h_segments, v_segments


def _apply_h_gap(
    rng: random.Random,
    inp: np.ndarray,
    out: np.ndarray,
    seg: dict,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> bool:
    opts = _h_gap_options(seg, out, fg, bg, occupied)
    if not opts:
        return False
    gap_start, anchor = rng.choice(opts)
    row = int(seg["row"])
    chain = _h_chain_coords(row, int(gap_start), anchor)
    inp[row, gap_start : gap_start + GAP_SIZE] = int(bg)
    for r, c in chain:
        inp[int(r), int(c)] = int(MARK_H)
    occupied.update((int(r), int(c)) for r, c in chain)
    return True


def _apply_v_gap(
    rng: random.Random,
    inp: np.ndarray,
    out: np.ndarray,
    seg: dict,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> bool:
    opts = _v_gap_options(seg, out, fg, bg, occupied)
    if not opts:
        return False
    gap_start, anchor, dc = rng.choice(opts)
    col = int(seg["col"])
    chain = _v_chain_coords(col, int(gap_start), anchor, int(dc))
    inp[gap_start : gap_start + GAP_SIZE, col] = int(bg)
    for r, c in chain:
        inp[int(r), int(c)] = int(MARK_V)
    occupied.update((int(r), int(c)) for r, c in chain)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    color_pool = [c for c in range(N_COLORS) if c not in MARKER_COLORS]
    bg = rng.choice(color_pool)
    fg = rng.choice([c for c in color_pool if c != bg])
    cfg = rng.choice(VALID_CONFIGS)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()

    h_candidates = [idx for idx, seg in enumerate(h_segments) if _h_gap_options(seg, out, fg, bg, occupied)]
    v_candidates = [idx for idx, seg in enumerate(v_segments) if _v_gap_options(seg, out, fg, bg, occupied)]
    if not h_candidates or not v_candidates:
        return None

    used_h: set[int] = set()
    used_v: set[int] = set()

    h_idx = rng.choice(h_candidates)
    if not _apply_h_gap(rng, inp, out, h_segments[h_idx], fg, bg, occupied):
        return None
    used_h.add(int(h_idx))

    v_idx = rng.choice(v_candidates)
    if not _apply_v_gap(rng, inp, out, v_segments[v_idx], fg, bg, occupied):
        return None
    used_v.add(int(v_idx))

    if rng.random() < 0.5:
        extra_orients: list[str] = []
        if any(idx not in used_h and _h_gap_options(h_segments[idx], out, fg, bg, occupied) for idx in h_candidates):
            extra_orients.append("h")
        if any(idx not in used_v and _v_gap_options(v_segments[idx], out, fg, bg, occupied) for idx in v_candidates):
            extra_orients.append("v")
        if extra_orients:
            orient = rng.choice(extra_orients)
            if orient == "h":
                options = [idx for idx in h_candidates if idx not in used_h and _h_gap_options(h_segments[idx], out, fg, bg, occupied)]
                if not options:
                    return None
                if not _apply_h_gap(rng, inp, out, h_segments[rng.choice(options)], fg, bg, occupied):
                    return None
            else:
                options = [idx for idx in v_candidates if idx not in used_v and _v_gap_options(v_segments[idx], out, fg, bg, occupied)]
                if not options:
                    return None
                if not _apply_v_gap(rng, inp, out, v_segments[rng.choice(options)], fg, bg, occupied):
                    return None

    # Oracle check: require exact reconstruction and ensure both marker colors are present.
    if not (np.any(inp == int(MARK_H)) and np.any(inp == int(MARK_V))):
        return None
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for fused variant 7b80bb43")
