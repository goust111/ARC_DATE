# Variant of 7b80bb43
# Dimension: D13 — 多步处理序列 (multi_step)
# Change: 三步顺序处理：(1)识别并删除对角像素 → (2)填补相邻间隙 → (3)迭代修剪叶子像素... → 增加预处理步骤：先连接近距离线段

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


# Design analysis for the D13 variant
# ==================================
# spec_variant:
#   base_visual:
#     - One background color and one foreground color.
#     - Output is still a clean network of axis-aligned H/V segments with 2-3 T-junctions.
#   new_input_corruption:
#     - Keep the original corruption type: a 3-cell gap plus a 2-pixel diagonal hint chain.
#     - Add a second corruption type: a "bare" 1-2 cell gap with no diagonal hint at all.
#     - Bare gaps are placed only on straight interior portions of a segment; both remaining sides
#       must extend away from the gap so the solver can recognize two parallel collinear segments.
#   solver_sequence:
#     - Step 0 (new): pre-connect short bare gaps of length <= 2 between two collinear segments.
#     - Step 1: identify/remove diagonal-only pixels from the original input.
#     - Step 2: fill hint-marked gaps exactly as in the original task.
#     - Step 3: prune leaf pixels near former diagonal hint sites.
#   concrete_behavior_trace:
#     - Original logic on "... fg fg bg bg fg fg ..." with no diagonal hint would do nothing.
#     - Variant logic first bridges the 1-2 bg cells, turning that row into one continuous line.
#     - Separately, a 3-cell gap with a diagonal hint chain is still repaired only after deleting
#       the diagonal chain, so the puzzle now contains two visibly different repair mechanisms.
#   complexity_calibration:
#     - Every generated sample contains at least one bare short gap and at least two original
#       hint-marked gaps, so the new step is not optional noise.
#     - Bare gaps stay length 1 or 2 only; hint-marked gaps stay length 3 only. This keeps the
#       categories visually distinct and limits the solver's branching complexity.
#   implicit_assumption_audit:
#     - Diagonal-only detection is still computed from the raw input; generator keeps bare gaps
#       away from diagonal chains, so the new pre-connect step does not change which pixels count
#       as diagonal hints.
#     - Pre-connect scans only non-diagonal foreground pixels and requires outward continuation on
#       both sides, preventing distractor chains or isolated leaf pixels from being bridged.
#     - Pre-connect also rejects gap cells that would create a junction (off-axis fg adjacent to
#       the gap), so generator must cut bare gaps only on straight interior spans.
#     - Standard hint-based filling still relies on diagonal neighborhoods; generator reserves
#       space so bare gaps never accidentally satisfy the diagonal-based repair rule.
#   placement_legality:
#     - Corruptions never overlap, never go out of bounds, and reserve a Chebyshev-1 buffer so
#       diagonal hints, bare gaps, and distractors do not interfere with one another.
#     - At most one corruption is applied per base segment, which keeps solver assumptions simple.
#   observability:
#     - In 2-4 ARC examples, a human can see two classes of defects:
#       1) short naked breaks that are reconnected directly;
#       2) longer 3-cell breaks accompanied by diagonal stair hints.
#     - Because both classes repeat across the same foreground color, the difference is structural,
#       not dependent on hidden color ids.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant solver with a new preprocessing step:
    0) bridge short bare gaps (length 1-2) between collinear line segments
    1) delete diagonal-only hint pixels
    2) fill hint-marked short gaps
    3) prune diagonal-adjacent leaves near former hint sites
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

    mask = g == fg
    H, W = g.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < H and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < W and bool(m[r, c + 1]):
            d += 1
        return d

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[r - 1, c - 1]):
            return True
        if r > 0 and c + 1 < W and bool(m[r - 1, c + 1]):
            return True
        if r + 1 < H and c > 0 and bool(m[r + 1, c - 1]):
            return True
        if r + 1 < H and c + 1 < W and bool(m[r + 1, c + 1]):
            return True
        return False

    def runs(vals: np.ndarray) -> list[tuple[int, int]]:
        if vals.size == 0:
            return []
        seqs: list[tuple[int, int]] = []
        start = int(vals[0])
        prev = int(vals[0])
        for v in vals[1:]:
            v = int(v)
            if v == prev + 1:
                prev = v
            else:
                seqs.append((start, prev))
                start = prev = v
        seqs.append((start, prev))
        return seqs

    def diag_near_gap_horizontal(diag_mask: np.ndarray, r: int, gap0: int, gap1: int) -> bool:
        rr0 = max(0, r - 1)
        rr1 = min(H - 1, r + 1)
        cc0 = max(0, gap0 - 1)
        cc1 = min(W - 1, gap1 + 1)
        return bool(diag_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())

    def diag_near_gap_vertical(diag_mask: np.ndarray, c: int, gap0: int, gap1: int) -> bool:
        rr0 = max(0, gap0 - 1)
        rr1 = min(H - 1, gap1 + 1)
        cc0 = max(0, c - 1)
        cc1 = min(W - 1, c + 1)
        return bool(diag_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())

    def off_axis_clear_h(core_mask: np.ndarray, r: int, gap0: int, gap1: int) -> bool:
        for cc in range(gap0, gap1 + 1):
            if r > 0 and bool(core_mask[r - 1, cc]):
                return False
            if r + 1 < H and bool(core_mask[r + 1, cc]):
                return False
        return True

    def off_axis_clear_v(core_mask: np.ndarray, c: int, gap0: int, gap1: int) -> bool:
        for rr in range(gap0, gap1 + 1):
            if c > 0 and bool(core_mask[rr, c - 1]):
                return False
            if c + 1 < W and bool(core_mask[rr, c + 1]):
                return False
        return True

    # Step 1 in the original task, but we only *identify* here so Step 0 can ignore them.
    diag = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag[r, c] = True

    near = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not bool(diag[r, c]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    # Step 0: pre-connect short bare gaps using only axis-aligned line pixels.
    core = mask & (~diag)
    pre = core.copy()

    for r in range(H):
        segs = runs(np.where(core[r, :])[0])
        for (a1, b1), (a2, b2) in zip(segs, segs[1:]):
            gap0 = b1 + 1
            gap1 = a2 - 1
            gap_len = gap1 - gap0 + 1
            if gap_len < 1 or gap_len > 2:
                continue
            if (b1 - a1 + 1) < 2 or (b2 - a2 + 1) < 2:
                continue
            if diag_near_gap_horizontal(diag, r, gap0, gap1):
                continue
            if not off_axis_clear_h(core, r, gap0, gap1):
                continue
            pre[r, gap0 : gap1 + 1] = True

    for c in range(W):
        segs = runs(np.where(core[:, c])[0])
        for (a1, b1), (a2, b2) in zip(segs, segs[1:]):
            gap0 = b1 + 1
            gap1 = a2 - 1
            gap_len = gap1 - gap0 + 1
            if gap_len < 1 or gap_len > 2:
                continue
            if (b1 - a1 + 1) < 2 or (b2 - a2 + 1) < 2:
                continue
            if diag_near_gap_vertical(diag, c, gap0, gap1):
                continue
            if not off_axis_clear_v(core, c, gap0, gap1):
                continue
            pre[gap0 : gap1 + 1, c] = True

    # Standard diagonal deletion now happens after the pre-connect stage.
    work = pre.copy()
    work[diag] = False

    MAXG = 3
    if bool(diag.any()):
        for coords in components_mask(diag, connectivity=8):
            if coords.size == 0:
                continue
            roots = set()
            for r, c in coords:
                r = int(r)
                c = int(c)
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and not bool(diag[rr, cc]):
                        roots.add((rr, cc))
            assert len(coords) + len(roots) <= 3, (
                f"diagonal chain too large ({len(coords)} diag + {len(roots)} root)"
            )
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            MAXG = max(MAXG, int(r1 - r0 + 1), int(c1 - c0 + 1))

    # Step 2: hint-based gap fill, using the post-preconnect working mask.
    out = np.full((H, W), bg, dtype=int)
    out[work] = fg

    for r in range(H):
        segs = runs(np.where(work[r, :])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = b1 + 1
            gap1 = a2 - 1
            if gap1 < gap0:
                continue
            if (gap1 - gap0 + 1) > MAXG:
                continue
            if diag_near_gap_horizontal(diag, r, gap0, gap1):
                out[r, gap0 : gap1 + 1] = fg

    for c in range(W):
        segs = runs(np.where(work[:, c])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = b1 + 1
            gap1 = a2 - 1
            if gap1 < gap0:
                continue
            if (gap1 - gap0 + 1) > MAXG:
                continue
            if diag_near_gap_vertical(diag, c, gap0, gap1):
                out[gap0 : gap1 + 1, c] = fg

    # Step 3: local pruning near former diagonal hint sites.
    m = out == fg
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not bool(m[r, c]) or not bool(near[r, c]):
                    continue
                if deg4(m, r, c) != 1:
                    continue
                if not has_diag(m, r, c):
                    continue
                to_clear.append((r, c))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out2 = np.full((H, W), bg, dtype=int)
    out2[m] = fg
    return out2


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

# Simple H sizing
SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

# Original diagonal-gap corruption
GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

# New pre-connect corruption
PRE_GAP_SIZES = (1, 2)
PRE_BLOCKED_MARGIN = 2
PRE_SIDE_MIN = 2

# Distractors
DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1

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


def _seg_key(seg: dict) -> tuple[int, ...]:
    if "row" in seg:
        return (0, int(seg["row"]), int(seg["c0"]), int(seg["c1"]))
    return (1, int(seg["col"]), int(seg["r0"]), int(seg["r1"]))


def _chain_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
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
        has_d = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg)):
                has_d = True
                break
        if not has_d:
            return False
    return True


def _line_zone_h(row: int, gap_start: int, gap_len: int) -> list[tuple[int, int]]:
    return [(row, c) for c in range(gap_start - 1, gap_start + gap_len + 1)]


def _line_zone_v(col: int, gap_start: int, gap_len: int) -> list[tuple[int, int]]:
    return [(r, col) for r in range(gap_start - 1, gap_start + gap_len + 1)]


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _place_t_component(
    out: np.ndarray,
    fg: int,
    slot: str,
    row: int,
    branch_col: int,
    end_col: int,
) -> tuple[dict, dict]:
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


def _h_diag_gap_options(
    seg: dict,
    grid: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, str]]:
    row, c0, c1 = int(seg["row"]), int(seg["c0"]), int(seg["c1"])
    blocked = set(map(int, seg["blocked"]))
    options: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        zone = _line_zone_h(row, g, GAP_SIZE)
        if any(cell in reserved for cell in zone):
            continue
        if not np.all(grid[row, g : g + GAP_SIZE] == fg):
            continue
        temp = grid.copy()
        temp[row, g : g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if any(cell in reserved for cell in coords):
                continue
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((g, anchor))
    return options


def _v_diag_gap_options(
    seg: dict,
    grid: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
    col, r0, r1 = int(seg["col"]), int(seg["r0"]), int(seg["r1"])
    blocked = set(map(int, seg["blocked"]))
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        zone = _line_zone_v(col, g, GAP_SIZE)
        if any(cell in reserved for cell in zone):
            continue
        if not np.all(grid[g : g + GAP_SIZE, col] == fg):
            continue
        temp = grid.copy()
        temp[g : g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g, anchor, dc)
                if any(cell in reserved for cell in coords):
                    continue
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                options.append((g, anchor, dc))
    return options


def _h_pre_gap_options(seg: dict, grid: np.ndarray, fg: int, reserved: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, c0, c1 = int(seg["row"]), int(seg["c0"]), int(seg["c1"])
    blocked = set(map(int, seg["blocked"]))
    options: list[tuple[int, int]] = []
    for gap_len in PRE_GAP_SIZES:
        for g in range(c0 + 1, c1 + 1):
            left_len = g - c0
            right_len = c1 - (g + gap_len) + 1
            if left_len < PRE_SIDE_MIN or right_len < PRE_SIDE_MIN:
                continue
            if any(abs(b - g) <= PRE_BLOCKED_MARGIN or abs(b - (g + gap_len - 1)) <= PRE_BLOCKED_MARGIN for b in blocked):
                continue
            if not np.all(grid[row, g : g + gap_len] == fg):
                continue
            zone = _line_zone_h(row, g, gap_len)
            if any(cell in reserved for cell in zone):
                continue
            off_axis_ok = True
            for cc in range(g, g + gap_len):
                if row > 0 and int(grid[row - 1, cc]) == int(fg):
                    off_axis_ok = False
                    break
                if row + 1 < grid.shape[0] and int(grid[row + 1, cc]) == int(fg):
                    off_axis_ok = False
                    break
            if off_axis_ok:
                options.append((g, gap_len))
    return options


def _v_pre_gap_options(seg: dict, grid: np.ndarray, fg: int, reserved: set[tuple[int, int]]) -> list[tuple[int, int]]:
    col, r0, r1 = int(seg["col"]), int(seg["r0"]), int(seg["r1"])
    blocked = set(map(int, seg["blocked"]))
    options: list[tuple[int, int]] = []
    for gap_len in PRE_GAP_SIZES:
        for g in range(r0 + 1, r1 + 1):
            top_len = g - r0
            bottom_len = r1 - (g + gap_len) + 1
            if top_len < PRE_SIDE_MIN or bottom_len < PRE_SIDE_MIN:
                continue
            if any(abs(b - g) <= PRE_BLOCKED_MARGIN or abs(b - (g + gap_len - 1)) <= PRE_BLOCKED_MARGIN for b in blocked):
                continue
            if not np.all(grid[g : g + gap_len, col] == fg):
                continue
            zone = _line_zone_v(col, g, gap_len)
            if any(cell in reserved for cell in zone):
                continue
            off_axis_ok = True
            for rr in range(g, g + gap_len):
                if col > 0 and int(grid[rr, col - 1]) == int(fg):
                    off_axis_ok = False
                    break
                if col + 1 < grid.shape[1] and int(grid[rr, col + 1]) == int(fg):
                    off_axis_ok = False
                    break
            if off_axis_ok:
                options.append((g, gap_len))
    return options


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, h - length):
            for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
                for direction in (-1, 1):
                    coords = [(r + i, c + direction * i) for i in range(length)]
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    options.append(coords)
    return options


def _build_output(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    fg: int,
    cfg: dict,
) -> tuple[np.ndarray, list[dict], list[dict]] | None:
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


def _place_diag_gaps(
    rng: random.Random,
    out: np.ndarray,
    inp: np.ndarray,
    h_segments: list[dict],
    v_segments: list[dict],
    fg: int,
    bg: int,
    reserved: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    used_keys: set[tuple[int, ...]],
    gap_count: int,
) -> bool:
    h_pool = h_segments[:]
    v_pool = v_segments[:]
    rng.shuffle(h_pool)
    rng.shuffle(v_pool)

    picked_h = False
    for seg in h_pool:
        key = _seg_key(seg)
        if key in used_keys:
            continue
        opts = _h_diag_gap_options(seg, inp, fg, bg, occupied, reserved)
        if not opts:
            continue
        gap_start, anchor = rng.choice(opts)
        row = int(seg["row"])
        chain = _chain_coords(row, gap_start, anchor)
        inp[row, gap_start : gap_start + GAP_SIZE] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)
        _reserve_neighborhood(reserved, _line_zone_h(row, gap_start, GAP_SIZE) + chain)
        used_keys.add(key)
        picked_h = True
        break
    if not picked_h:
        return False

    v_needed = gap_count - 1
    picked_v = 0
    for seg in v_pool:
        if picked_v >= v_needed:
            break
        key = _seg_key(seg)
        if key in used_keys:
            continue
        opts = _v_diag_gap_options(seg, inp, fg, bg, occupied, reserved)
        if not opts:
            continue
        gap_start, anchor, dc = rng.choice(opts)
        col = int(seg["col"])
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)
        _reserve_neighborhood(reserved, _line_zone_v(col, gap_start, GAP_SIZE) + chain)
        used_keys.add(key)
        picked_v += 1
    return picked_v == v_needed


def _place_pre_gaps(
    rng: random.Random,
    inp: np.ndarray,
    h_segments: list[dict],
    v_segments: list[dict],
    fg: int,
    reserved: set[tuple[int, int]],
    used_keys: set[tuple[int, ...]],
) -> int:
    placed = 0
    all_segs: list[tuple[str, dict]] = [("h", seg) for seg in h_segments] + [("v", seg) for seg in v_segments]
    rng.shuffle(all_segs)

    eligible = 0
    for orient, seg in all_segs:
        key = _seg_key(seg)
        if key in used_keys:
            continue
        opts = _h_pre_gap_options(seg, inp, fg, reserved) if orient == "h" else _v_pre_gap_options(seg, inp, fg, reserved)
        if opts:
            eligible += 1
    if eligible == 0:
        return 0

    target = 2 if eligible >= 2 and rng.random() < 0.6 else 1
    for orient, seg in all_segs:
        if placed >= target:
            break
        key = _seg_key(seg)
        if key in used_keys:
            continue
        opts = _h_pre_gap_options(seg, inp, fg, reserved) if orient == "h" else _v_pre_gap_options(seg, inp, fg, reserved)
        if not opts:
            continue
        gap_start, gap_len = rng.choice(opts)
        if orient == "h":
            row = int(seg["row"])
            inp[row, gap_start : gap_start + gap_len] = int(get_bg(inp))
            _reserve_neighborhood(reserved, _line_zone_h(row, gap_start, gap_len))
        else:
            col = int(seg["col"])
            inp[gap_start : gap_start + gap_len, col] = int(get_bg(inp))
            _reserve_neighborhood(reserved, _line_zone_v(col, gap_start, gap_len))
        used_keys.add(key)
        placed += 1
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    used_keys: set[tuple[int, ...]] = set()

    if not _place_diag_gaps(rng, out, inp, h_segments, v_segments, fg, bg, reserved, occupied, used_keys, gap_count):
        return None

    pre_count = _place_pre_gaps(rng, inp, h_segments, v_segments, fg, reserved, used_keys)
    if pre_count < 1:
        return None

    forbidden: set[tuple[int, int]] = set(reserved)
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
    _reserve_neighborhood(forbidden, list(occupied))
    distractors = _distractor_options(inp, bg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any(cell in forbidden for cell in coords):
            continue
        for r, c in coords:
            inp[r, c] = fg
        _reserve_neighborhood(forbidden, coords)
        placed += 1

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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D13 variant")
