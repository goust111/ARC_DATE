# Combo variant of 7b80bb43 (index 6)
# combo_spec: D5/3+D16/0
# Name: 十字形中心提示

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, draw_line, find_components, get_bg

# Design analysis for combo D5/3 + D16/0
# ------------------------------------------------------------
# a) Concrete behavior trace
#    Original: a 3-cell straight gap is truly empty, while a 2-pixel diagonal stair
#    sits next to one gap endpoint. The solver deletes diagonal-only pixels, then uses
#    their location to refill the nearby 3-cell row/column gap.
#    Fused variant: the cue moves to the gap center and becomes a centered cross around
#    a 1-cell hole. Example for a horizontal bridge:
#      clean output span:    .....#####.....
#      fused input locally:
#             ..#..
#             .#.#.
#             ..#..
#    The missing center is the gap to refill. The four surrounding arms point in all
#    directions; the true axis is the opposite pair whose run continues farther through
#    the network. Output fills the center and removes the short perpendicular stubs.
#
# b) Complexity calibration
#    If the plus were floating in open background, the task would collapse to simple
#    shape deletion. If the whole line were already obvious without directional
#    ambiguity, D16/0 would be visually inert. Centering the plus on the hidden bridge
#    avoids both failures: every hint is locally ambiguous until the solver compares the
#    four arm runs, yet the rule remains low-arity and repeated across 2-3 hints.
#
# c) Broken assumptions in the original main()
#    - "Hint pixels have no orthogonal neighbors" is false now: the centered cue is
#      built from orthogonal arms around the missing pixel.
#    - "Gap evidence is adjacent to one endpoint" is false now: the cue is centered on
#      the gap midpoint.
#    - "Gap cells are completely empty until diagonal cleanup refills them" is only
#      partly true now: the midpoint is empty, but the four neighboring arm pixels are
#      already present. The solver must resolve centered crosses before the legacy
#      diagonal-distractor pass.
#
# d) Placement legality
#    The generator only places a centered cross on a valid internal 3-cell span of a
#    segment. The midpoint is cleared to background, and the two short perpendicular
#    stubs must land on background cells with no orthogonal fg neighbors except their
#    arm root, guaranteeing they become removable 1-pixel leaves. Hints are spaced by
#    the original blocked-zone logic, so no two centered crosses interfere.
#
# e) Visual observability
#    Each sample includes at least one horizontal and one vertical centered cross. A
#    human can compare examples and see that the output always keeps the longer opposite
#    pair through the center and deletes the shorter pair. No hidden color ordering or
#    index-based convention is used.
#
# f) Visual cleanliness
#    The short stubs are single pixels orthogonal to an otherwise clean line network,
#    so the cross reads as one compact mark rather than a pile of overlapping motifs.
#    Distractor diagonal chains remain isolated in open background and never touch the
#    centered crosses.
#
# g) New solver assumptions and generator guarantees
#    - Assumption: every centered cross is encoded by one background midpoint with all
#      four orthogonal neighbors present.
#      Guarantee: the output network uses only T-junctions, never true 4-way crossings,
#      so such a midpoint pattern is reserved for the hint.
#    - Assumption: the true axis has run length >= 2 on both sides, while the false axis
#      has run length exactly 1.
#      Guarantee: gap sites are internal 3-cell spans away from endpoints/junctions, and
#      added stubs are forced to be orthogonal leaves.
#    - Assumption: any remaining diagonal-only components are legacy distractors, not
#      part of the centered-cross encoding.
#      Guarantee: cross hints use orthogonal connectivity only; diagonal distractors are
#      generated separately in open background.

# --- Configuration constants ---

GRID_H_RANGE = (16, 29)
GRID_W_RANGE = (17, 30)
N_COLORS = 10

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

EDGE_MARGIN = 3
ROW_COL_GAP = 2
T_BRANCH_OFFSET_MIN = 3
T_BRANCH_OFFSET_MAX = 9
H_SEG_END_MARGIN = 4
H_SEG_MAX_END = 25

SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

GAP_SIZE = 3
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1

GEN_RETRIES = 300

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _deg4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    d = 0
    if r > 0 and bool(mask[r - 1, c]):
        d += 1
    if r + 1 < h and bool(mask[r + 1, c]):
        d += 1
    if c > 0 and bool(mask[r, c - 1]):
        d += 1
    if c + 1 < w and bool(mask[r, c + 1]):
        d += 1
    return d


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in DIAG_DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            return True
    return False


def _run_len(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = mask.shape
    rr, cc = r + dr, c + dc
    n = 0
    while 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
        n += 1
        rr += dr
        cc += dc
    return n


def _find_cross_hints(mask: np.ndarray) -> list[dict]:
    hints: list[dict] = []
    h, w = mask.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if bool(mask[r, c]):
                continue
            if not (bool(mask[r - 1, c]) and bool(mask[r + 1, c]) and bool(mask[r, c - 1]) and bool(mask[r, c + 1])):
                continue
            up = _run_len(mask, r, c, -1, 0)
            down = _run_len(mask, r, c, 1, 0)
            left = _run_len(mask, r, c, 0, -1)
            right = _run_len(mask, r, c, 0, 1)
            horiz = min(left, right)
            vert = min(up, down)
            if horiz >= 2 and vert == 1:
                axis = "h"
            elif vert >= 2 and horiz == 1:
                axis = "v"
            else:
                continue
            hints.append(
                {
                    "center": (r, c),
                    "axis": axis,
                    "runs": {"up": up, "down": down, "left": left, "right": right},
                }
            )
    return hints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve centered cross-gap hints and clean leftover diagonal distractors.

    Input structure:
    - One foreground color forms an axis-aligned line network on a background color.
    - Some internal 3-cell line spans are broken at their midpoint. Around that empty
      midpoint sits a compact cross hint: the four orthogonal neighbor cells are present,
      but only one opposite pair continues deeper into the network.
    - Separate diagonal-only foreground chains may still appear as open-background
      distractors.

    Rule:
    - For each centered cross-gap hint, fill the empty midpoint, keep the axis whose two
      opposite arms run farther from the center, and delete the other two 1-pixel side
      stubs.
    - Then remove diagonal-only distractor pixels, use their vicinity to heal any nearby
      short straight gaps, and prune diagonal-adjacent leaves near those former
      distractor sites.

    Key insight: The plus is not endpoint-attached anymore; it is centered on the hidden
    bridge midpoint. The correct orientation is revealed by local run length from that
    missing center: one axis extends through the network on both sides, while the
    perpendicular axis stops after one pixel in each direction.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

    mask = g == fg
    h, w = g.shape

    # Resolve centered cross-gap hints first; this is the D5/3 + D16/0 fusion core.
    resolved = mask.copy()
    for hint in _find_cross_hints(mask):
        r, c = hint["center"]
        resolved[r, c] = True
        if hint["axis"] == "h":
            resolved[r - 1, c] = False
            resolved[r + 1, c] = False
        else:
            resolved[r, c - 1] = False
            resolved[r, c + 1] = False

    # Legacy diagonal distractor cleanup from the source task.
    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(resolved[r, c]) and _deg4(resolved, r, c) == 0 and _has_diag(resolved, r, c):
                diag[r, c] = True

    near = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(diag[r, c]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        near[rr, cc] = True

    line_mask = resolved & (~diag)
    out = line_mask.copy()

    max_gap = 3
    if bool(diag.any()):
        for coords in components_mask(diag, connectivity=8):
            if coords.size == 0:
                continue
            roots = set()
            for r, c in coords:
                r = int(r)
                c = int(c)
                for dr, dc in DIAG_DIRS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and bool(resolved[rr, cc]) and not bool(diag[rr, cc]):
                        roots.add((rr, cc))
            assert len(coords) + len(roots) <= 3, "diagonal distractor unexpectedly large"
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            max_gap = max(max_gap, int(r1 - r0 + 1), int(c1 - c0 + 1))

    for r in range(h):
        cols = np.where(line_mask[r, :])[0]
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
                segs.append((start, prev))
                start = prev = c
        segs.append((start, prev))

        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0 or int(gap1 - gap0 + 1) > max_gap:
                continue
            rr0 = max(0, r - 1)
            rr1 = min(h - 1, r + 1)
            cc0 = max(0, gap0 - 1)
            cc1 = min(w - 1, gap1 + 1)
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[r, gap0 : gap1 + 1] = True

    for c in range(w):
        rows = np.where(line_mask[:, c])[0]
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
                segs.append((start, prev))
                start = prev = r
        segs.append((start, prev))

        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0 or int(gap1 - gap0 + 1) > max_gap:
                continue
            rr0 = max(0, gap0 - 1)
            rr1 = min(h - 1, gap1 + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w - 1, c + 1)
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[gap0 : gap1 + 1, c] = True

    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if not bool(out[r, c]) or not bool(near[r, c]):
                    continue
                if _deg4(out, r, c) != 1:
                    continue
                if not _has_diag(out, r, c):
                    continue
                to_clear.append((r, c))
        if to_clear:
            changed = True
            for r, c in to_clear:
                out[r, c] = False

    result = np.full((h, w), bg, dtype=int)
    result[out] = fg
    return result


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


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    hh, ww = grid.shape
    for r, c in coords:
        if not (0 <= r < hh and 0 <= c < ww) or grid[r, c] != color:
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int, occupied: set[tuple[int, int]]) -> bool:
    hh, ww = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < hh and 0 <= cc < ww and grid[rr, cc] == fg:
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < hh and 0 <= cc < ww and grid[rr, cc] == fg):
                has_diag = True
                break
        if not has_diag:
            return False
    return True


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    hh, ww = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, hh - length):
            for c in range(DISTRACTOR_MARGIN, ww - DISTRACTOR_MARGIN):
                for direction in (-1, 1):
                    coords = [(r + i, c + direction * i) for i in range(length)]
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    options.append(coords)
    return options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int, branch_col: int, end_col: int) -> tuple[dict, dict]:
    hh, ww = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else ww - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else hh - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}}
    v_seg = {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    ww = out.shape[1]
    c0 = 0 if side == "left" else ww - length
    c1 = length - 1 if side == "left" else ww - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": row, "c0": c0, "c1": c1, "blocked": set()}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    hh = out.shape[0]
    r0 = 0 if side == "top" else hh - length
    r1 = length - 1 if side == "top" else hh - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": col, "r0": r0, "r1": r1, "blocked": set()}


def _stub_slot_clear(grid: np.ndarray, fg: int, center: tuple[int, int], stub: tuple[int, int]) -> bool:
    hh, ww = grid.shape
    sr, sc = stub
    cr, cc = center
    if not (0 <= sr < hh and 0 <= sc < ww):
        return False
    if grid[sr, sc] == fg:
        return False
    for rr, cc2 in ((sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1)):
        if (rr, cc2) == (cr, cc):
            continue
        if 0 <= rr < hh and 0 <= cc2 < ww and grid[rr, cc2] == fg:
            return False
    return True


def _h_cross_stubs(row: int, gap_start: int) -> list[tuple[int, int]]:
    center_c = gap_start + 1
    return [(row - 1, center_c), (row + 1, center_c)]


def _v_cross_stubs(col: int, gap_start: int) -> list[tuple[int, int]]:
    center_r = gap_start + 1
    return [(center_r, col - 1), (center_r, col + 1)]


def _h_cross_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[int]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[int] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        stubs = _h_cross_stubs(row, g)
        center = (row, g + 1)
        if any(cell in occupied for cell in stubs):
            continue
        if not _cells_are_color(out, stubs, bg):
            continue
        if not all(_stub_slot_clear(out, fg, center, stub) for stub in stubs):
            continue
        options.append(g)
    return options


def _v_cross_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[int]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[int] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        stubs = _v_cross_stubs(col, g)
        center = (g + 1, col)
        if any(cell in occupied for cell in stubs):
            continue
        if not _cells_are_color(out, stubs, bg):
            continue
        if not all(_stub_slot_clear(out, fg, center, stub) for stub in stubs):
            continue
        options.append(g)
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
    centers: set[tuple[int, int]] = set()

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_cross_options(seg, out, fg, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_cross_options(seg, out, fg, bg, occupied)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_cross_options(h_seg, out, fg, bg, occupied) or h_initial
    gap_start = rng.choice(h_opts)
    center = (h_seg["row"], gap_start + 1)
    inp[center] = bg
    centers.add(center)
    for r, c in _h_cross_stubs(h_seg["row"], gap_start):
        inp[r, c] = fg
        occupied.add((r, c))

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_cross_options(v_seg, out, fg, bg, occupied) or v_initial
        gap_start = rng.choice(v_opts)
        center = (gap_start + 1, v_seg["col"])
        inp[center] = bg
        centers.add(center)
        for r, c in _v_cross_stubs(v_seg["col"], gap_start):
            inp[r, c] = fg
            occupied.add((r, c))

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
    _reserve_neighborhood(forbidden, list(occupied))
    _reserve_neighborhood(forbidden, list(centers))
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo D5/3+D16/0")
