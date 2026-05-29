# Variant of 7b80bb43
# Dimension: D5 — 对角链的几何形态 (src)
# Change: 2像素45度阶梯链，从间隙端点沿对角方向延伸（down-left或down-right）... → L形提示标记

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, draw_line, find_components, get_bg


# Design analysis for the D5 variant
# ==================================
# source_task:
#   None. No external reference solver is consulted.
#
# concretization_and_trace:
#   original_input:
#     Gaps are still exactly 3 background cells cut out of a straight H/V segment.
#     The original diagonal hint was a 2-cell 45-degree staircase detached from the line.
#   variant_input:
#     Each hint is now a 3-cell, 4-connected L-shaped component inside a 2x2 box.
#     The clean network/output is unchanged: one fg color, axis-aligned segments, 2-3 gaps.
#     Four L orientations are used, one per gap-endpoint type:
#       - {(0,1),(1,0),(1,1)}  -> horizontal gap, anchored at the left endpoint
#       - {(0,0),(1,0),(1,1)}  -> horizontal gap, anchored at the right endpoint
#       - {(0,0),(0,1),(1,0)}  -> vertical gap, anchored at the top endpoint
#       - {(0,0),(0,1),(1,1)}  -> vertical gap, anchored at the bottom endpoint
#     Example:
#       Suppose a horizontal segment on row r has a 3-cell gap at cols g..g+2.
#       Original code removed a diagonal chain near the gap and then scanned the row.
#       New code instead detects a 3-cell L below the gap endpoint:
#         left-anchor marker = {(r+1,g),(r+2,g-1),(r+2,g)}
#       Its orientation directly implies "fill row r, cols g..g+2".
#       The line-restoration stage therefore moves from a generic "nearby diagonal pixels"
#       heuristic to an orientation-aware "small L component -> exact gap template" rule.
#
# complexity_calibration:
#   The marker is not simplified into a 2-cell domino; it remains a distinctive shape with
#   four rotations, so humans can see a new visual grammar rather than a small perturbation.
#   The number of marker categories is fixed at four, matching the four endpoint cases,
#   which keeps inference finite and repeatable across 2-4 examples.
#
# implicit_assumption_audit:
#   Original `main()` assumptions that break under this variant:
#     - "hint pixels have no orthogonal fg neighbors" no longer holds because L pixels are
#       internally 4-connected.
#     - `components_mask(diag, connectivity=8)` computed MAXG from diagonal-only pixels;
#       that entire notion disappears, so gap filling must no longer depend on diagonal span.
#     - row/col gap filling previously used "any diagonal hint in a 3xN or Nx3 window".
#       With L markers this becomes too permissive and can miss the intended orientation.
#   New solver assumptions:
#     - Every hint/distractor is a detached 3-cell L component with bbox 2x2.
#     - No true output object forms such a detached 3-cell L.
#     - A gap hint's implied fill cells are currently bg and have fg immediately beyond both
#       ends of the 3-cell run. The generator guarantees these conditions.
#
# placement_legality:
#   The generator still builds the clean output first. It then cuts a 3-cell gap and places
#   an L marker in an empty 2x2 box diagonally offset from the chosen endpoint:
#     - horizontal markers are placed below the line
#     - vertical markers are placed to the right of the line
#   Validation requires:
#     - all marker cells stay in bounds and lie on bg
#     - marker cells never overlap other markers
#     - marker cells have no orthogonal fg neighbor outside the marker
#     - gap sites stay away from blocked junction coordinates
#   Distractor Ls are placed only in open background with a reserved neighborhood, so they
#   are visually separate and are safely removed by the solver without triggering fills.
#
# visual_observability:
#   In the input/output pairs, the only disappearing objects are 3-cell right-angle motifs,
#   not diagonal staircases. Some of those Ls sit near broken lines and cause restoration;
#   others are isolated distractors and simply vanish. That is visible in a few examples.
#
# visual_cleanliness:
#   Markers are detached from lines by at least diagonal spacing and distractors are kept away
#   from the network by a Chebyshev-1 reserve. This avoids stacked ambiguity around junctions.
#
# color_inference:
#   No new color rule is introduced. The task remains two-color: one bg and one fg.


GAP_SIZE = 3
GAP_SEG_INSET = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4

L_SHAPES: dict[frozenset[tuple[int, int]], tuple[str, str]] = {
    frozenset({(0, 1), (1, 0), (1, 1)}): ("h", "left"),
    frozenset({(0, 0), (1, 0), (1, 1)}): ("h", "right"),
    frozenset({(0, 0), (0, 1), (1, 0)}): ("v", "top"),
    frozenset({(0, 0), (0, 1), (1, 1)}): ("v", "bottom"),
}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

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
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and bool(m[rr, cc]):
                    return True
        return False

    mask = g == fg
    hint_cells = np.zeros((H, W), dtype=bool)
    near = np.zeros((H, W), dtype=bool)
    hints: list[tuple[str, str, int, int]] = []

    for coords in components_mask(mask, connectivity=4):
        if len(coords) != 3:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0) != 1 or int(c1 - c0) != 1:
            continue
        rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
        if rel not in L_SHAPES:
            continue
        axis, anchor = L_SHAPES[rel]
        hints.append((axis, anchor, int(r0), int(c0)))
        for r, c in coords:
            hint_cells[int(r), int(c)] = True
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    out = g.copy()
    out[hint_cells] = bg

    def endpoint_fill_cells(axis: str, anchor: str, r0: int, c0: int) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
        if axis == "h" and anchor == "left":
            cells = [(r0 - 1, c0 + dc) for dc in (1, 2, 3)]
            return cells, (r0 - 1, c0), (r0 - 1, c0 + 4)
        if axis == "h" and anchor == "right":
            cells = [(r0 - 1, c0 + dc) for dc in (-2, -1, 0)]
            return cells, (r0 - 1, c0 - 3), (r0 - 1, c0 + 1)
        if axis == "v" and anchor == "top":
            cells = [(r0 + dr, c0 - 1) for dr in (0, 1, 2)]
            return cells, (r0 - 1, c0 - 1), (r0 + 3, c0 - 1)
        cells = [(r0 + dr, c0 - 1) for dr in (-2, -1, 0)]
        return cells, (r0 - 3, c0 - 1), (r0 + 1, c0 - 1)

    def in_bounds_cells(cells: list[tuple[int, int]]) -> bool:
        return all(0 <= r < H and 0 <= c < W for r, c in cells)

    def endpoint_fg(m: np.ndarray, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < H and 0 <= c < W and bool(m[r, c])

    current = out == fg
    for axis, anchor, r0, c0 in hints:
        cells, end0, end1 = endpoint_fill_cells(axis, anchor, r0, c0)
        if not in_bounds_cells(cells):
            continue
        if not all(out[r, c] == bg for r, c in cells):
            continue
        if not (endpoint_fg(current, end0) and endpoint_fg(current, end1)):
            continue
        for r, c in cells:
            out[r, c] = fg
            current[r, c] = True

    # Preserve the original "cleanup near removed hints" idea, adapted to L neighborhoods.
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

# Segment network topology
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

# Distractor placement
DISTRACTOR_MARGIN = 1

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


def _h_marker_coords(row: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "left":
        return [(row + 1, gap_start), (row + 2, gap_start - 1), (row + 2, gap_start)]
    return [(row + 1, gap_start + 2), (row + 2, gap_start + 2), (row + 2, gap_start + 3)]


def _v_marker_coords(col: int, gap_start: int, anchor: str) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start, col + 1), (gap_start, col + 2), (gap_start + 1, col + 1)]
    return [(gap_start + 2, col + 1), (gap_start + 2, col + 2), (gap_start + 3, col + 2)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != color:
            return False
    return True


def _validate_marker(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    fg: int,
    occupied: set[tuple[int, int]],
    *,
    require_diag: bool,
) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    if not _cells_are_color(grid, coords, int(get_bg(grid))):
        return False
    found_diag = False
    for r, c in coords:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                continue
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg:
                return False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set:
                continue
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg:
                found_diag = True
    return found_diag or (not require_diag)


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
            coords = _h_marker_coords(row, g, anchor)
            if not _validate_marker(temp, coords, fg, occupied, require_diag=True):
                continue
            options.append((g, anchor))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            coords = _v_marker_coords(col, g, anchor)
            if not _validate_marker(temp, coords, fg, occupied, require_diag=True):
                continue
            options.append((g, anchor))
    return options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _distractor_options(inp: np.ndarray, fg: int, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    rel_shapes = [
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
    ]
    for r in range(DISTRACTOR_MARGIN, h - 1 - DISTRACTOR_MARGIN):
        for c in range(DISTRACTOR_MARGIN, w - 1 - DISTRACTOR_MARGIN):
            for rel in rel_shapes:
                coords = [(r + dr, c + dc) for dr, dc in rel]
                if any(cell in forbidden for cell in coords):
                    continue
                if not _cells_are_color(inp, coords, bg):
                    continue
                if not _validate_marker(inp, coords, fg, set(), require_diag=False):
                    continue
                options.append(coords)
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

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, out, fg, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, out, fg, bg, occupied)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, out, fg, bg, occupied) or h_initial
    gap_start, anchor = rng.choice(h_opts)
    row = h_seg["row"]
    marker = _h_marker_coords(row, gap_start, anchor)
    inp[row, gap_start : gap_start + GAP_SIZE] = bg
    for r, c in marker:
        inp[r, c] = fg
    occupied.update(marker)

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, bg, occupied) or v_initial
        gap_start, anchor = rng.choice(v_opts)
        col = v_seg["col"]
        marker = _v_marker_coords(col, gap_start, anchor)
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in marker:
            inp[r, c] = fg
        occupied.update(marker)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, [tuple(x) for x in np.argwhere(inp == fg)])
    _reserve_neighborhood(forbidden, list(occupied))
    distractors = _distractor_options(inp, fg, bg, forbidden)
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D5 variant")
