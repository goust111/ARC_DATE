# Combo variant of 7b80bb43 (index 16)
# combo_spec: D1/4+D5/1+D16/0
# Name: L形中心方向填充
#
# Design analysis
# ===============
# variant_spec:
#   input_structure:
#     - The clean target is still a one-color network of axis-aligned H/V segments.
#     - Real corruptions are no longer endpoint-anchored diagonal chains.
#       Each real corruption is a detached 3-cell L tromino occupying three cells of a 2x2 box.
#       That 2x2 box is centered off the gap midpoint by one step, so the missing corner is
#       not itself the midpoint, but its rotation plus its offset identifies the midpoint.
#     - The L has four rotations. Two rotations encode horizontal partial repair
#       (fill left or fill right), and two rotations encode vertical partial repair
#       (fill up or fill down).
#     - Optional distractors use the same detached L geometry but sit in open background.
#   concretized_example:
#     - Suppose a horizontal line has a 3-cell gap at columns c-1,c,c+1 on row r.
#     - A "fill-right" hint places fg at (r+1,c+1), (r+2,c), (r+2,c+1).
#       This is a detached L just below the gap center (r,c); its missing NW corner sits
#       one row below the center and is enough to recover both orientation and midpoint.
#     - Output removes the L and restores only (r,c) and (r,c+1), leaving (r,c-1) empty.
#     - Old task behavior would have restored all three gap cells and would not have
#       needed the center-aligned L rotation.
#   complexity_calibration:
#     - Real gaps are placed on both horizontal and vertical segments so all four
#       directional outcomes matter across samples.
#     - Each sample has 2-3 real gaps, with at least one horizontal and one vertical.
#       When a third gap is added, the generator prefers the opposite direction on an
#       already-used axis so the directional rule is visible inside one grid.
#     - Distractors keep the "remove detached noise without filling anything" branch alive.
#   hidden_assumptions_review:
#     - The original solver assumed hints were diagonal-only, endpoint-aligned, and that
#       any nearby 3-cell gap should be fully restored. All three assumptions break here.
#     - The new solver instead assumes:
#         1. Real hints are detached 4-connected size-3 components with bbox 2x2.
#         2. The missing bbox corner, together with the corner type, maps to one intended
#            gap center next to the L.
#         3. That center lies on exactly one valid 3-cell horizontal or vertical gap
#            pattern consistent with the hint rotation.
#     - The generator guarantees those assumptions by construction and rejects samples
#       failing the oracle check.
#   placement_legality:
#     - Every real gap is still exactly 3 cells long and stays away from endpoints and
#       junctions using the original blocked-window logic.
#     - Each L hint must fit in bounds, stay on background after the full 3-cell gap is cut,
#       and have no orthogonal contact with existing line pixels, so it remains detached.
#     - Hint neighborhoods and gap bands are reserved to avoid overlap or visual merging.
#   observability:
#     - The target output intentionally keeps one cell of each 3-cell break empty, so the
#       difference from the original task is visible.
#     - The L sits in a fixed relative position next to the gap center, so "where is the
#       center?" and "which direction gets restored?" are both visible from the same local cue.
#   cleanliness:
#     - The only small detached shapes in the input are L hints or L distractors.
#       The line network itself never creates detached 3-cell 2x2 L components.
#     - Spacing rules keep hints out of dense junction neighborhoods and away from one another.
#   new_solver_assumptions_to_guarantee:
#     - Missing-NW and missing-NE L rotations are used only for horizontal gaps below the line.
#     - Missing-SE and missing-SW L rotations are used only for vertical gaps beside the line.
#     - The line has foreground two cells beyond the gap center on both sides of the
#       intended axis, while the entire 3-cell gap is background in the input.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


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
TARGET_GAP_COUNTS = (2, 3)

DISTRACTOR_MARGIN = 1
GEN_RETRIES = 300

DIR_TO_HINT = {
    "left": [(1, -1), (2, -1), (2, 0)],
    "right": [(1, 1), (2, 0), (2, 1)],
    "up": [(-1, -2), (-1, -1), (0, -2)],
    "down": [(-1, 1), (-1, 2), (0, 2)],
}
DIR_TO_MISSING = {
    "right": "NW",
    "left": "NE",
    "down": "SW",
    "up": "SE",
}
MISSING_TO_DIR = {missing: direction for direction, missing in DIR_TO_MISSING.items()}
OPPOSITE_DIR = {"left": "right", "right": "left", "up": "down", "down": "up"}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove detached center-aligned L hints and partially restore the marked gap.

    Input structure:
    - One background color and one foreground color.
    - Large foreground structures are axis-aligned horizontal/vertical line segments.
    - Small detached foreground components may be 3-cell L trominoes sitting one step
      away from the center of a 3-cell break.

    Rule:
    - Delete every detached 3-cell L tromino.
    - Read its missing 2x2 corner plus the rotation-specific offset to recover the gap center.
    - Missing NW -> horizontal gap, refill the right two cells.
    - Missing NE -> horizontal gap, refill the left two cells.
    - Missing SW -> vertical gap, refill the lower two cells.
    - Missing SE -> vertical gap, refill the upper two cells.
    - If an L sits in open background without a compatible gap, it is just removed.

    Key insight:
    The same local 2x2 L geometry simultaneously encodes three things: which nearby
    background cell is the gap center, whether the broken line is horizontal or vertical,
    and which side of that 3-cell gap should be restored.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

    mask = g == fg
    H, W = g.shape

    def l_signature(coords: np.ndarray) -> tuple[tuple[int, int], str] | None:
        if int(coords.shape[0]) != 3:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0) != 1 or int(c1 - c0) != 1:
            return None
        box = {(int(r), int(c)) for r in (int(r0), int(r1)) for c in (int(c0), int(c1))}
        pts = {(int(r), int(c)) for r, c in coords}
        missing = box - pts
        if len(missing) != 1:
            return None
        missing_cell = next(iter(missing))
        if missing_cell == (int(r0), int(c0)):
            key = "NW"
        elif missing_cell == (int(r0), int(c1)):
            key = "NE"
        elif missing_cell == (int(r1), int(c0)):
            key = "SW"
        else:
            key = "SE"
        mr, mc = missing_cell
        if key == "NW":
            center = (mr - 1, mc)
        elif key == "NE":
            center = (mr - 1, mc)
        elif key == "SW":
            center = (mr, mc - 1)
        else:
            center = (mr, mc + 1)
        return center, key

    out = g.copy()
    hints: list[tuple[tuple[int, int], str]] = []
    for coords in components_mask(mask, connectivity=4):
        sig = l_signature(coords)
        if sig is None:
            continue
        hints.append(sig)
        for r, c in coords:
            out[int(r), int(c)] = bg

    base = out == fg
    for (r, c), key in hints:
        direction = MISSING_TO_DIR[key]
        if direction in ("left", "right"):
            gap_cols = [c - 1, c, c + 1]
            if not (0 <= r < H and 0 <= c - 2 and c + 2 < W):
                continue
            if not (bool(base[r, c - 2]) and bool(base[r, c + 2])):
                continue
            if any(bool(base[r, cc]) for cc in gap_cols):
                continue
            fill_cols = [c - 1, c] if direction == "left" else [c, c + 1]
            for cc in fill_cols:
                out[r, cc] = fg
        else:
            gap_rows = [r - 1, r, r + 1]
            if not (0 <= c < W and 0 <= r - 2 and r + 2 < H):
                continue
            if not (bool(base[r - 2, c]) and bool(base[r + 2, c])):
                continue
            if any(bool(base[rr, c]) for rr in gap_rows):
                continue
            fill_rows = [r - 1, r] if direction == "up" else [r, r + 1]
            for rr in fill_rows:
                out[rr, c] = fg

    return out


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


def _hint_coords(center: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in DIR_TO_HINT[direction]]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_hint(grid: np.ndarray, coords: list[tuple[int, int]], fg: int,
                   occupied: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                continue
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg):
                return False
    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int,
                       branch_col: int, end_col: int) -> tuple[dict, dict]:
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


def _gap_band(axis: str, fixed: int, gap_start: int) -> set[tuple[int, int]]:
    if axis == "h":
        return {(fixed, c) for c in range(gap_start - 1, gap_start + GAP_SIZE + 1)}
    return {(r, fixed) for r in range(gap_start - 1, gap_start + GAP_SIZE + 1)}


def _reserve_neighborhood(cells: set[tuple[int, int]], coords) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str, list[tuple[int, int]]]]:
    row, c0, c1 = int(seg["row"]), int(seg["c0"]), int(seg["c1"])
    blocked = set(int(x) for x in seg["blocked"])
    options: list[tuple[int, str, list[tuple[int, int]]]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[row, g:g + GAP_SIZE] = bg
        center = (row, g + 1)
        for direction in ("left", "right"):
            coords = _hint_coords(center, direction)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_hint(temp, coords, fg, occupied):
                continue
            options.append((g, direction, coords))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   occupied: set[tuple[int, int]]) -> list[tuple[int, str, list[tuple[int, int]]]]:
    col, r0, r1 = int(seg["col"]), int(seg["r0"]), int(seg["r1"])
    blocked = set(int(x) for x in seg["blocked"])
    options: list[tuple[int, str, list[tuple[int, int]]]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[g:g + GAP_SIZE, col] = bg
        center = (g + 1, col)
        for direction in ("up", "down"):
            coords = _hint_coords(center, direction)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_hint(temp, coords, fg, occupied):
                continue
            options.append((g, direction, coords))
    return options


def _output_hole_cell(axis: str, fixed: int, gap_start: int, direction: str) -> tuple[int, int]:
    if axis == "h":
        row = fixed
        if direction == "left":
            return row, gap_start + 2
        return row, gap_start
    col = fixed
    if direction == "up":
        return gap_start + 2, col
    return gap_start, col


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int,
                  cfg: dict) -> tuple[np.ndarray, list[dict], list[dict]] | None:
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
        branch_col = int(branch_by_slot[slot])
        row = int(row_by_slot[slot])
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
        h_segments.append(_place_simple_h(out, fg, side, int(row), rng.randint(SIMPLE_H_LEN_MIN, max_len)))

    top_v = min(2, v_simple)
    top_v_cols = simple_v_cols[:top_v]
    bottom_v_cols = simple_v_cols[top_v:]
    top_limit = min(min(top_rows) - 2 if top_rows else center_start - 2, center_start - 2)
    bottom_start = max(max(bottom_rows) + 2 if bottom_rows else center_end + 2, center_end + 2)
    for col in top_v_cols:
        if top_limit < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "top", int(col), rng.randint(2, top_limit)))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "bottom", int(col), rng.randint(2, max_len)))

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    return out, h_segments, v_segments


def _collect_h_candidates(clean: np.ndarray, h_segments: list[dict], fg: int, bg: int,
                          occupied: set[tuple[int, int]], forbidden: set[tuple[int, int]],
                          required_dir: str | None = None) -> list[tuple[str, int, int, str, list[tuple[int, int]]]]:
    candidates: list[tuple[str, int, int, str, list[tuple[int, int]]]] = []
    for seg in h_segments:
        row = int(seg["row"])
        for gap_start, direction, coords in _h_gap_options(seg, clean, fg, bg, occupied):
            if required_dir is not None and direction != required_dir:
                continue
            band = _gap_band("h", row, int(gap_start))
            if any(cell in forbidden for cell in coords):
                continue
            if any(cell in forbidden for cell in band):
                continue
            candidates.append(("h", row, int(gap_start), direction, coords))
    return candidates


def _collect_v_candidates(clean: np.ndarray, v_segments: list[dict], fg: int, bg: int,
                          occupied: set[tuple[int, int]], forbidden: set[tuple[int, int]],
                          required_dir: str | None = None) -> list[tuple[str, int, int, str, list[tuple[int, int]]]]:
    candidates: list[tuple[str, int, int, str, list[tuple[int, int]]]] = []
    for seg in v_segments:
        col = int(seg["col"])
        for gap_start, direction, coords in _v_gap_options(seg, clean, fg, bg, occupied):
            if required_dir is not None and direction != required_dir:
                continue
            band = _gap_band("v", col, int(gap_start))
            if any(cell in forbidden for cell in coords):
                continue
            if any(cell in forbidden for cell in band):
                continue
            candidates.append(("v", col, int(gap_start), direction, coords))
    return candidates


def _commit_choice(choice: tuple[str, int, int, str, list[tuple[int, int]]],
                   occupied: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> None:
    axis, fixed, gap_start, _direction, coords = choice
    occupied.update(coords)
    _reserve_neighborhood(forbidden, coords)
    band = _gap_band(axis, fixed, gap_start)
    forbidden.update(band)
    _reserve_neighborhood(forbidden, band)


def _choose_real_gaps(rng: random.Random, clean: np.ndarray, h_segments: list[dict], v_segments: list[dict],
                      fg: int, bg: int, gap_count: int) -> list[tuple[str, int, int, str, list[tuple[int, int]]]] | None:
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    chosen: list[tuple[str, int, int, str, list[tuple[int, int]]]] = []
    chosen_dir_by_axis: dict[str, str] = {}

    h_candidates = _collect_h_candidates(clean, h_segments, fg, bg, occupied, forbidden)
    v_candidates = _collect_v_candidates(clean, v_segments, fg, bg, occupied, forbidden)
    if not h_candidates or not v_candidates:
        return None

    first_h = rng.choice(h_candidates)
    chosen.append(first_h)
    chosen_dir_by_axis["h"] = first_h[3]
    _commit_choice(first_h, occupied, forbidden)

    v_candidates = _collect_v_candidates(clean, v_segments, fg, bg, occupied, forbidden)
    if not v_candidates:
        return None
    first_v = rng.choice(v_candidates)
    chosen.append(first_v)
    chosen_dir_by_axis["v"] = first_v[3]
    _commit_choice(first_v, occupied, forbidden)

    while len(chosen) < gap_count:
        preferred: list[tuple[str, int, int, str, list[tuple[int, int]]]] = []
        if "h" in chosen_dir_by_axis:
            preferred.extend(
                _collect_h_candidates(
                    clean, h_segments, fg, bg, occupied, forbidden, OPPOSITE_DIR[chosen_dir_by_axis["h"]]
                )
            )
        if "v" in chosen_dir_by_axis:
            preferred.extend(
                _collect_v_candidates(
                    clean, v_segments, fg, bg, occupied, forbidden, OPPOSITE_DIR[chosen_dir_by_axis["v"]]
                )
            )
        pool = preferred
        if not pool:
            pool = _collect_h_candidates(clean, h_segments, fg, bg, occupied, forbidden)
            pool += _collect_v_candidates(clean, v_segments, fg, bg, occupied, forbidden)
        if not pool:
            return None
        pick = rng.choice(pool)
        chosen.append(pick)
        chosen_dir_by_axis[pick[0]] = pick[3]
        _commit_choice(pick, occupied, forbidden)

    return chosen


def _distractor_options(inp: np.ndarray, bg: int, fg: int,
                        forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for r in range(DISTRACTOR_MARGIN, h - DISTRACTOR_MARGIN):
        for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
            center = (r, c)
            for direction in ("left", "right", "up", "down"):
                coords = _hint_coords(center, direction)
                if not _cells_are_color(inp, coords, bg):
                    continue
                if any(cell in forbidden for cell in coords):
                    continue
                if not _validate_hint(inp, coords, fg, set()):
                    continue
                options.append(coords)
    return options


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = rng.choice(TARGET_GAP_COUNTS)
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    clean, h_segments, v_segments = built

    chosen = _choose_real_gaps(rng, clean, h_segments, v_segments, fg, bg, gap_count)
    if chosen is None:
        return None

    out = clean.copy()
    inp = clean.copy()
    real_hint_cells: set[tuple[int, int]] = set()
    for axis, fixed, gap_start, direction, coords in chosen:
        hole_r, hole_c = _output_hole_cell(axis, fixed, gap_start, direction)
        out[hole_r, hole_c] = bg
        if axis == "h":
            inp[fixed, gap_start:gap_start + GAP_SIZE] = bg
        else:
            inp[gap_start:gap_start + GAP_SIZE, fixed] = bg
        for r, c in coords:
            inp[r, c] = fg
        real_hint_cells.update(coords)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
    _reserve_neighborhood(forbidden, real_hint_cells)
    distractors = _distractor_options(inp, bg, fg, forbidden)
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo variant 16")
