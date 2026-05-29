# Combo variant of 7b80bb43 (index 5)
# combo_spec: D5/1+D16/2
# Name: L形对称提示

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import draw_line, find_components, get_bg


# Design analysis for the fused variant:
# - Concrete geometry: every corrupted gap still has length 3, but each old
#   single hint is replaced by two mirrored diagonal-L components. Every prompt
#   has a long 45-degree arm plus one orthogonal foot at the arm tip, so D5/1's
#   "longer diagonal chain" remains visible inside the new L-shaped family.
#   For a horizontal gap on row r at cols g..g+2, the pair sits two rows below
#   the gap: left prompt starts at (r+2,g) and runs down-left for arm_len cells;
#   right prompt mirrors it from (r+2,g+2) down-right. For a vertical gap on
#   col c at rows g..g+2, the mirrored pair sits two columns to the right and
#   runs up-right / down-right from the top and bottom gap cells.
# - Small example: if a T-junction on a horizontal segment is 4 cells left of a
#   gap, the original task would place one short detached staircase near one
#   endpoint. Here both endpoints get mirrored diagonal-Ls of arm length 4. The
#   solver pairs them, infers "horizontal gap" from the below-row placement, and
#   uses the shared arm length to verify that a vertical crossing exists exactly
#   4 cells from one side before filling.
# - Complexity calibration: only two prompt placement families exist (below a
#   horizontal gap or right of a vertical gap), and arm lengths are restricted to
#   3..5 so the value is visible and repeated across samples without overcrowding.
# - Hidden-assumption audit: the original solver assumed every hint pixel was a
#   detached diagonal-only leaf. That breaks here because each prompt is an
#   8-connected mini-object with one orthogonal elbow. The new solver therefore
#   classifies whole prompt components by exact mirrored templates, then checks
#   that the encoded length matches a real T-crossing in the surviving network.
# - Placement legality: the generator only cuts T-junction segments, so every gap
#   has a real crossing to encode. It also checks that every prompt cell stays in
#   background, stays isolated from the line network under 4-connectivity, never
#   overlaps another prompt, and stays within bounds.
# - Visual observability and cleanliness: all samples contain at least one
#   horizontal and one vertical prompt pair, so a human sees both orientation
#   codes. Prompts are isolated mirrored diagonal corners of one consistent
#   family, with no competing cue types mixed into the same local region.


GAP_SIZE = 3
ARM_LEN_MIN = 3
ARM_LEN_MAX = 5

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 30)
N_COLORS = 10

GEN_RETRIES = 400


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(color):
            return False
    return True


def _h_prompt_coords(row: int, gap_start: int, arm_len: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    left = [(row + 2 + k, gap_start - k) for k in range(arm_len)]
    left.append((row + arm_len + 1, gap_start - arm_len + 2))
    right_col = gap_start + GAP_SIZE - 1
    right = [(row + 2 + k, right_col + k) for k in range(arm_len)]
    right.append((row + arm_len + 1, right_col + arm_len - 2))
    return left, right


def _v_prompt_coords(col: int, gap_start: int, arm_len: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    top = [(gap_start - k, col + 2 + k) for k in range(arm_len)]
    top.append((gap_start - arm_len + 2, col + arm_len + 1))
    bottom_row = gap_start + GAP_SIZE - 1
    bottom = [(bottom_row + k, col + 2 + k) for k in range(arm_len)]
    bottom.append((bottom_row + arm_len - 2, col + arm_len + 1))
    return top, bottom


def _validate_prompt_cells(
    temp: np.ndarray,
    coords: list[tuple[int, int]],
    fg: int,
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = temp.shape
    coord_set = set(coords)
    if len(coord_set) != len(coords):
        return False
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        if int(temp[r, c]) != int(get_bg(temp)) and int(temp[r, c]) != int(fg):
            return False
        if int(temp[r, c]) == int(fg):
            return False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                continue
            if 0 <= rr < h and 0 <= cc < w and int(temp[rr, cc]) == int(fg):
                return False
    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int, branch_col: int, end_col: int) -> tuple[dict, dict]:
    h, w = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else w - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else h - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}}
    v_seg = {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}}
    return h_seg, v_seg


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), bg, dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    r1_lo, r1_hi = 7, h // 2 - 2
    r2_lo, r2_hi = h // 2 + 2, h - 8
    b1_lo, b1_hi = 6, w // 2 - 3
    b2_lo, b2_hi = w // 2 + 2, w - 7
    if r1_lo > r1_hi or r2_lo > r2_hi or b1_lo > b1_hi or b2_lo > b2_hi:
        return None

    r1 = rng.randint(r1_lo, r1_hi)
    r2_start = max(r2_lo, r1 + 5)
    if r2_start > r2_hi:
        return None
    r2 = rng.randint(r2_start, r2_hi)
    b1 = rng.randint(b1_lo, b1_hi)
    b2_start = max(b2_lo, b1 + 6)
    if b2_start > b2_hi:
        return None
    b2 = rng.randint(b2_start, b2_hi)

    e1_lo = b1 + ARM_LEN_MIN + GAP_SIZE + 1
    e1_hi = w - 4
    e2_lo = 3
    e2_hi = b2 - (ARM_LEN_MIN + GAP_SIZE + 1)
    if e1_lo > e1_hi or e2_lo > e2_hi:
        return None
    e1 = rng.randint(e1_lo, e1_hi)
    e2 = rng.randint(e2_lo, e2_hi)

    h_seg, v_seg = _place_t_component(out, fg, "LT", r1, b1, e1)
    h_segments.append(h_seg)
    v_segments.append(v_seg)
    h_seg, v_seg = _place_t_component(out, fg, "RB", r2, b2, e2)
    h_segments.append(h_seg)
    v_segments.append(v_seg)

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != 2:
        return None
    return out, h_segments, v_segments


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(seg["blocked"]) != 1:
        return []
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    branch = next(iter(seg["blocked"]))
    options: list[tuple[int, int]] = []
    for arm_len in range(ARM_LEN_MIN, ARM_LEN_MAX + 1):
        candidates = [
            branch + arm_len + 1,
            branch - arm_len - GAP_SIZE,
        ]
        for gap_start in candidates:
            if gap_start < c0 + 2 or gap_start + GAP_SIZE - 1 > c1 - 2:
                continue
            temp = out.copy()
            temp[row, gap_start : gap_start + GAP_SIZE] = bg
            left, right = _h_prompt_coords(row, gap_start, arm_len)
            coords = left + right
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_prompt_cells(temp, coords, fg, occupied):
                continue
            options.append((gap_start, arm_len))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(seg["blocked"]) != 1:
        return []
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    branch = next(iter(seg["blocked"]))
    options: list[tuple[int, int]] = []
    for arm_len in range(ARM_LEN_MIN, ARM_LEN_MAX + 1):
        candidates = [
            branch + arm_len + 1,
            branch - arm_len - GAP_SIZE,
        ]
        for gap_start in candidates:
            if gap_start < r0 + 2 or gap_start + GAP_SIZE - 1 > r1 - 2:
                continue
            temp = out.copy()
            temp[gap_start : gap_start + GAP_SIZE, col] = bg
            top, bottom = _v_prompt_coords(col, gap_start, arm_len)
            coords = top + bottom
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_prompt_cells(temp, coords, fg, occupied):
                continue
            options.append((gap_start, arm_len))
    return options


def _classify_prompt_component(coords: np.ndarray) -> dict | None:
    pts = {(int(r), int(c)) for r, c in coords}
    rows = sorted({r for r, _c in pts})
    cols = sorted({c for _r, c in pts})
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rel = {(r - r0, c - c0) for r, c in pts}

    if h == w and ARM_LEN_MIN <= h <= ARM_LEN_MAX and len(pts) == h + 1:
        left_shape = {(k, h - 1 - k) for k in range(h)} | {(h - 1, 1)}
        right_shape = {(k, k) for k in range(h)} | {(h - 1, h - 2)}
        top_shape = {(h - 1 - k, k) for k in range(h)} | {(1, h - 1)}
        bottom_shape = {(k, k) for k in range(h)} | {(h - 2, h - 1)}
        if rel == left_shape:
            return {"mode": "h_left", "top": r0, "arm_len": h, "anchor": c1, "coords": pts}
        if rel == right_shape:
            return {"mode": "h_right", "top": r0, "arm_len": h, "anchor": c0, "coords": pts}
        if rel == top_shape:
            return {"mode": "v_top", "left": c0, "arm_len": h, "anchor": r1, "coords": pts}
        if rel == bottom_shape:
            return {"mode": "v_bottom", "left": c0, "arm_len": h, "anchor": r0, "coords": pts}
    return None


def _has_vertical_crossing(mask: np.ndarray, row: int, col: int) -> bool:
    h, w = mask.shape
    if not (0 <= row < h and 0 <= col < w) or not bool(mask[row, col]):
        return False
    up = row > 0 and bool(mask[row - 1, col])
    down = row + 1 < h and bool(mask[row + 1, col])
    left = col > 0 and bool(mask[row, col - 1])
    right = col + 1 < w and bool(mask[row, col + 1])
    return (up or down) and (left or right)


def _has_horizontal_crossing(mask: np.ndarray, row: int, col: int) -> bool:
    h, w = mask.shape
    if not (0 <= row < h and 0 <= col < w) or not bool(mask[row, col]):
        return False
    up = row > 0 and bool(mask[row - 1, col])
    down = row + 1 < h and bool(mask[row + 1, col])
    left = col > 0 and bool(mask[row, col - 1])
    right = col + 1 < w and bool(mask[row, col + 1])
    return (left or right) and (up or down)


def _extract_gap_repairs(mask: np.ndarray) -> tuple[set[tuple[int, int]], list[tuple[str, int, int, int]]]:
    comps = find_components(mask.astype(int), bg=0, connectivity=8, monochromatic=True, return_format="dict")
    parsed: list[dict] = []
    for comp in comps:
        info = _classify_prompt_component(comp["coords"])
        if info is not None:
            parsed.append(info)

    h_lefts = [x for x in parsed if x["mode"] == "h_left"]
    h_rights = [x for x in parsed if x["mode"] == "h_right"]
    v_tops = [x for x in parsed if x["mode"] == "v_top"]
    v_bottoms = [x for x in parsed if x["mode"] == "v_bottom"]

    hint_cells: set[tuple[int, int]] = set()
    repairs: list[tuple[str, int, int, int]] = []
    used_ids: set[int] = set()

    for left in h_lefts:
        for right in h_rights:
            if id(left) in used_ids or id(right) in used_ids:
                continue
            if left["top"] != right["top"] or left["arm_len"] != right["arm_len"]:
                continue
            if right["anchor"] - left["anchor"] != GAP_SIZE - 1:
                continue
            gap_row = left["top"] - 2
            gap0 = left["anchor"]
            repairs.append(("h", gap_row, gap0, left["arm_len"]))
            hint_cells |= left["coords"] | right["coords"]
            used_ids.add(id(left))
            used_ids.add(id(right))
            break

    for top in v_tops:
        for bottom in v_bottoms:
            if id(top) in used_ids or id(bottom) in used_ids:
                continue
            if top["left"] != bottom["left"] or top["arm_len"] != bottom["arm_len"]:
                continue
            if bottom["anchor"] - top["anchor"] != GAP_SIZE - 1:
                continue
            gap_col = top["left"] - 2
            gap0 = top["anchor"]
            repairs.append(("v", gap_col, gap0, top["arm_len"]))
            hint_cells |= top["coords"] | bottom["coords"]
            used_ids.add(id(top))
            used_ids.add(id(bottom))
            break

    return hint_cells, repairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove mirrored diagonal-L prompts and reconnect the broken line network.

    Input structure:
    - The grid has one background color and one foreground color forming only axis-aligned segments.
    - Each true gap is exactly 3 cells long and is marked by two mirrored diagonal-L prompts in the same fg color.
    - A horizontal break uses a mirrored pair placed two rows below the gap; a vertical break uses a mirrored pair placed two columns to the right.

    Rule:
    - Detect small isolated fg components that match one of the mirrored diagonal-L prompt templates.
    - Pair matching Ls, erase them, and use their arm length as the required distance from the gap to a real T-crossing on the surviving line network.
    - When the encoded distance is consistent with such a crossing, fill the 3-cell horizontal or vertical gap between the pair.

    Key insight:
    - The prompt is no longer "any diagonal noise near a gap"; it is a paired code. Mirror placement locates the center gap, component orientation tells whether the missing segment is horizontal or vertical, and arm length disambiguates the intended gap by naming how far the nearby junction must be.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

    fg_mask = g == fg
    hint_cells, repairs = _extract_gap_repairs(fg_mask)

    core = fg_mask.copy()
    for r, c in hint_cells:
        core[r, c] = False

    out = core.copy()
    h, w = g.shape
    for kind, fixed, gap0, arm_len in repairs:
        if kind == "h":
            row = fixed
            if not (0 <= row < h and 0 <= gap0 - 1 and gap0 + GAP_SIZE < w):
                continue
            left_ok = _has_vertical_crossing(core, row, gap0 - 1 - arm_len)
            right_ok = _has_vertical_crossing(core, row, gap0 + GAP_SIZE + arm_len)
            if bool(core[row, gap0 - 1]) and bool(core[row, gap0 + GAP_SIZE]) and (left_ok or right_ok):
                out[row, gap0 : gap0 + GAP_SIZE] = True
        else:
            col = fixed
            if not (0 <= col < w and 0 <= gap0 - 1 and gap0 + GAP_SIZE < h):
                continue
            top_ok = _has_horizontal_crossing(core, gap0 - 1 - arm_len, col)
            bottom_ok = _has_horizontal_crossing(core, gap0 + GAP_SIZE + arm_len, col)
            if bool(core[gap0 - 1, col]) and bool(core[gap0 + GAP_SIZE, col]) and (top_ok or bottom_ok):
                out[gap0 : gap0 + GAP_SIZE, col] = True

    result = np.full_like(g, bg)
    result[out] = fg
    return result


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])

    built = _build_output(rng, h, w, bg, fg)
    if built is None:
        return None
    out, h_segments, v_segments = built
    inp = out.copy()
    occupied: set[tuple[int, int]] = set()
    encoded_lengths: list[int] = []

    h_ready = [(seg, _h_gap_options(seg, out, fg, bg, occupied)) for seg in h_segments if seg["blocked"]]
    h_ready = [(seg, opts) for seg, opts in h_ready if opts]
    v_ready = [(seg, _v_gap_options(seg, out, fg, bg, occupied)) for seg in v_segments if seg["blocked"]]
    v_ready = [(seg, opts) for seg, opts in v_ready if opts]
    if not h_ready or not v_ready:
        return None

    rng.shuffle(h_ready)
    rng.shuffle(v_ready)
    chosen: list[tuple[str, dict, tuple[int, int]]] = []

    h_seg, h_opts = h_ready[0]
    chosen.append(("h", h_seg, rng.choice(h_opts)))

    v_seg, v_opts = v_ready[0]
    chosen.append(("v", v_seg, rng.choice(v_opts)))

    extra_pool: list[tuple[str, dict, list[tuple[int, int]]]] = []
    extra_pool.extend([("h", seg, opts) for seg, opts in h_ready[1:]])
    extra_pool.extend([("v", seg, opts) for seg, opts in v_ready[1:]])
    rng.shuffle(extra_pool)
    if extra_pool and rng.random() < 0.6:
        kind, seg, opts = extra_pool[0]
        chosen.append((kind, seg, rng.choice(opts)))

    used_segments: set[int] = set()
    for kind, seg, choice in chosen:
        if id(seg) in used_segments:
            return None
        used_segments.add(id(seg))
        gap_start, arm_len = choice
        encoded_lengths.append(arm_len)
        if kind == "h":
            row = seg["row"]
            left, right = _h_prompt_coords(row, gap_start, arm_len)
            inp[row, gap_start : gap_start + GAP_SIZE] = bg
            for r, c in left + right:
                inp[r, c] = fg
            occupied.update(left)
            occupied.update(right)
        else:
            col = seg["col"]
            top, bottom = _v_prompt_coords(col, gap_start, arm_len)
            inp[gap_start : gap_start + GAP_SIZE, col] = bg
            for r, c in top + bottom:
                inp[r, c] = fg
            occupied.update(top)
            occupied.update(bottom)

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp, out, encoded_lengths


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        made = _gen_attempt()
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo variant")
