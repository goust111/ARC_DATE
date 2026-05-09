# Combo variant of 7b80bb43 (index 19)
# combo_spec: D3/1+D11/2+D14/2
# Name: 着色方向编码精确匹配

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


# Design analysis
# a) Concrete behavior on a tiny example:
#    - Original task: a row like `fg fg bg bg bg fg fg` plus a same-color diagonal chain
#      always filled the 3-cell gap after the chain was deleted.
#    - Fused task: the diagonal chain is red or green instead of fg. Red means "this chain
#      talks about a horizontal gap", green means "this chain talks about a vertical gap".
#      The chain is not deleted in the output. Instead it is recolored to fg. Only if the
#      chain length is exactly gap_size - 1 does the nearby gap get filled. So a red length-2
#      chain near a 3-cell horizontal gap fills, while a red length-3 chain near that same
#      kind of gap is merely recolored and the gap stays open.
# b) Complexity calibration:
#    - The change must be visually obvious, so the generator always uses two visible hint
#      colors (red=2, green=3), repeats each color multiple times, and mixes matched and
#      mismatched lengths in the same sample. This prevents the task from collapsing into a
#      trivial "all colored hints fill" or "all colored hints are recolored" rule.
# c) Hidden assumptions from the original main():
#    - The original solver assumed exactly one non-bg color. That no longer holds.
#    - It assumed every diagonal-only pixel should be removed. That is false here because
#      hint pixels survive and are recolored.
#    - It assumed chain span only sets a permissive max gap. That is replaced by an exact
#      equality test against gap_size - 1.
#    - It assumed post-fill pruning was needed to restore clean orthogonal lines. Here that
#      pruning would destroy the recolored hints, so it must be removed.
# d) Placement legality:
#    - Gap sites are still cut only from internal portions of existing straight segments.
#    - Candidate chains are checked for bounds, background occupancy, zero orthogonal fg
#      neighbors before recoloring, and separation from previously placed chains/gaps.
# e) Visual observability:
#    - Each sample contains both hint colors, at least one horizontal and one vertical fill,
#      and at least one same-color non-fill caused only by wrong chain length. A human can
#      see color->axis and length->fill from 2-4 examples without relying on numeric ids.
# f) Visual cleanliness:
#    - Open-space distractors are omitted here because they would add noise without helping
#      the fused rule. Instead, non-filling examples are placed at real gaps so the decisive
#      difference is local and easy to compare.
# g) Color-change inferability:
#    - Only four colors appear: bg, fg, red horizontal hints, green vertical hints.
#      The output always merges both hint colors into fg, so the recolor rule is simple and
#      strongly repeated.


H_HINT = 2
V_HINT = 3
HINT_COLORS = (H_HINT, V_HINT)


def _chain_coords(row: int, gap_start: int, gap_size: int, anchor: str, length: int) -> list[tuple[int, int]]:
    """Diagonal chain coordinates for a horizontal gap."""
    if anchor == "left":
        return [(row + i, gap_start - 1 + i) for i in range(1, length + 1)]
    return [(row + i, gap_start + gap_size - i) for i in range(1, length + 1)]


def _v_chain_coords(col: int, gap_start: int, gap_size: int, anchor: str, dc: int,
                    length: int) -> list[tuple[int, int]]:
    """Diagonal chain coordinates for a vertical gap."""
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(length)]
    return [(gap_start + gap_size - 1 - i, col + dc * (i + 1)) for i in range(length)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int,
                    occupied: set[tuple[int, int]]) -> bool:
    """Check that chain coords sit on bg, have no orthogonal fg neighbor, and have diagonal support."""
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


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _build_hint_components(grid: np.ndarray) -> tuple[dict[int, dict[int, set[tuple[int, int]]]], dict[tuple[int, int], tuple[int, int]]]:
    component_cells: dict[int, dict[int, set[tuple[int, int]]]] = {H_HINT: {}, V_HINT: {}}
    component_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for color in HINT_COLORS:
        for comp_id, coords in enumerate(components_mask(grid == int(color), connectivity=8)):
            cell_set = {tuple(map(int, rc)) for rc in coords}
            if not cell_set:
                continue
            component_cells[int(color)][int(comp_id)] = cell_set
            for cell in cell_set:
                component_lookup[cell] = (int(color), int(comp_id))
    return component_cells, component_lookup


def _match_component(color: int, coords: list[tuple[int, int]],
                     component_cells: dict[int, dict[int, set[tuple[int, int]]]],
                     component_lookup: dict[tuple[int, int], tuple[int, int]]) -> tuple[int, int] | None:
    coord_set = {tuple(map(int, rc)) for rc in coords}
    if not coord_set:
        return None
    first = component_lookup.get(next(iter(coord_set)))
    if first is None or int(first[0]) != int(color):
        return None
    for cell in coord_set:
        if component_lookup.get(cell) != first:
            return None
    if component_cells[int(color)][int(first[1])] != coord_set:
        return None
    return int(first[0]), int(first[1])


def _segments_from_indices(indices: np.ndarray) -> list[tuple[int, int]]:
    if int(indices.size) == 0:
        return []
    segs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for x in indices[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            segs.append((int(start), int(prev)))
            start = prev = x
    segs.append((int(start), int(prev)))
    return segs


def _find_fillable_gap_data(grid: np.ndarray, fg: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return gap cells to fill and hint components that triggered those fills."""
    h, w = grid.shape
    line_mask = grid == int(fg)
    component_cells, component_lookup = _build_hint_components(grid)
    fill_cells: set[tuple[int, int]] = set()
    matched_components: set[tuple[int, int]] = set()

    for r in range(h):
        segs = _segments_from_indices(np.where(line_mask[int(r), :])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            gap_size = int(gap1 - gap0 + 1)
            chain_len = int(gap_size - 1)
            if chain_len < 1:
                continue
            for anchor in ("left", "right"):
                coords = _chain_coords(int(r), gap0, gap_size, anchor, chain_len)
                key = _match_component(H_HINT, coords, component_cells, component_lookup)
                if key is None:
                    continue
                matched_components.add(key)
                for c in range(gap0, gap1 + 1):
                    fill_cells.add((int(r), int(c)))
                break

    for c in range(w):
        segs = _segments_from_indices(np.where(line_mask[:, int(c)])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if gap1 < gap0:
                continue
            gap_size = int(gap1 - gap0 + 1)
            chain_len = int(gap_size - 1)
            if chain_len < 1:
                continue
            found = False
            for anchor in ("top", "bottom"):
                for dc in (-1, 1):
                    coords = _v_chain_coords(int(c), gap0, gap_size, anchor, dc, chain_len)
                    key = _match_component(V_HINT, coords, component_cells, component_lookup)
                    if key is None:
                        continue
                    matched_components.add(key)
                    for r in range(gap0, gap1 + 1):
                        fill_cells.add((int(r), int(c)))
                    found = True
                    break
                if found:
                    break

    return fill_cells, matched_components


def analyze_fusion_behavior(grid: np.ndarray) -> dict[str, int]:
    """Utility for validation scripts: count matched vs unmatched hint chains."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    colors.difference_update(HINT_COLORS)
    assert len(colors) == 1
    fg = int(next(iter(colors)))
    fill_cells, matched = _find_fillable_gap_data(g, fg)
    component_cells, _component_lookup = _build_hint_components(g)
    h_total = len(component_cells[H_HINT])
    v_total = len(component_cells[V_HINT])
    h_matched = sum(1 for key in matched if int(key[0]) == int(H_HINT))
    v_matched = sum(1 for key in matched if int(key[0]) == int(V_HINT))
    return {
        "fill_cells": len(fill_cells),
        "h_total": h_total,
        "v_total": v_total,
        "h_matched": h_matched,
        "v_matched": v_matched,
        "h_unmatched": h_total - h_matched,
        "v_unmatched": v_total - v_matched,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor direction-coded diagonal hints and bridge only the exactly matched gaps.

    Input structure:
    - bg plus one main line color fg forming axis-aligned segments.
    - Red diagonal chains mark horizontal-gap candidates; green diagonal chains mark vertical-gap candidates.
    - Every hint chain is still diagonally arranged away from a nearby line break, but different chains may have different lengths.

    Rule:
    - Recolor every hint pixel into fg instead of deleting it.
    - For each horizontal or vertical gap, consult only the hint color assigned to that axis.
    - Fill the gap iff the nearby hint chain occupies the exact expected diagonal footprint for length gap_size - 1.

    Key insight: The solver must validate three things in order: hint color chooses the axis, chain length must match the broken span exactly, and only then does recoloring fuse the hint into the restored segment.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    colors.difference_update(HINT_COLORS)
    assert len(colors) == 1, "expected one line color plus two hint colors"
    fg = int(next(iter(colors)))

    fill_cells, _matched = _find_fillable_gap_data(g, fg)

    out = g.copy()
    for hint_color in HINT_COLORS:
        out[out == int(hint_color)] = int(fg)
    for r, c in fill_cells:
        out[int(r), int(c)] = int(fg)
    return out


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

# Gap / chain placement
GAP_SIZES = (3, 4)
CHAIN_LENGTHS = (2, 3)
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

# Retry budget
GEN_RETRIES = 400


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


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int,
                       branch_col: int, end_col: int) -> tuple[dict, dict]:
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


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    w = out.shape[1]
    c0 = 0 if side == "left" else w - length
    c1 = length - 1 if side == "left" else w - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": row, "c0": c0, "c1": c1, "blocked": set()}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    h = out.shape[0]
    r0 = 0 if side == "top" else h - length
    r1 = length - 1 if side == "top" else h - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": col, "r0": r0, "r1": r1, "blocked": set()}


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


def _gap_cells_h(row: int, gap_start: int, gap_size: int) -> list[tuple[int, int]]:
    return [(int(row), int(c)) for c in range(int(gap_start), int(gap_start + gap_size))]


def _gap_cells_v(col: int, gap_start: int, gap_size: int) -> list[tuple[int, int]]:
    return [(int(r), int(col)) for r in range(int(gap_start), int(gap_start + gap_size))]


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   gap_size: int, chain_len: int, occupied: set[tuple[int, int]],
                   forbidden: set[tuple[int, int]]) -> list[dict]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[dict] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - gap_size):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        gap_cells = _gap_cells_h(row, g, gap_size)
        if any(cell in forbidden for cell in gap_cells):
            continue
        temp = out.copy()
        temp[int(row), int(g): int(g + gap_size)] = int(bg)
        for anchor in ("left", "right"):
            coords = _chain_coords(int(row), int(g), int(gap_size), anchor, int(chain_len))
            if not _cells_are_color(temp, coords, bg):
                continue
            if any(cell in forbidden for cell in coords):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append(
                {
                    "orientation": "H",
                    "row": int(row),
                    "gap_start": int(g),
                    "gap_size": int(gap_size),
                    "chain_len": int(chain_len),
                    "anchor": anchor,
                    "coords": coords,
                    "gap_cells": gap_cells,
                }
            )
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   gap_size: int, chain_len: int, occupied: set[tuple[int, int]],
                   forbidden: set[tuple[int, int]]) -> list[dict]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[dict] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - gap_size):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        gap_cells = _gap_cells_v(col, g, gap_size)
        if any(cell in forbidden for cell in gap_cells):
            continue
        temp = out.copy()
        temp[int(g): int(g + gap_size), int(col)] = int(bg)
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(int(col), int(g), int(gap_size), anchor, int(dc), int(chain_len))
                if not _cells_are_color(temp, coords, bg):
                    continue
                if any(cell in forbidden for cell in coords):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                options.append(
                    {
                        "orientation": "V",
                        "col": int(col),
                        "gap_start": int(g),
                        "gap_size": int(gap_size),
                        "chain_len": int(chain_len),
                        "anchor": anchor,
                        "dc": int(dc),
                        "coords": coords,
                        "gap_cells": gap_cells,
                    }
                )
    return options


def _pick_placement(rng: random.Random, segments: list[dict], orientation: str, out: np.ndarray,
                    fg: int, bg: int, gap_size: int, chain_len: int,
                    occupied: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> dict | None:
    segs = segments[:]
    rng.shuffle(segs)
    options: list[dict] = []
    for seg in segs:
        if orientation == "H":
            options.extend(_h_gap_options(seg, out, fg, bg, gap_size, chain_len, occupied, forbidden))
        else:
            options.extend(_v_gap_options(seg, out, fg, bg, gap_size, chain_len, occupied, forbidden))
    if not options:
        return None
    return rng.choice(options)


def _apply_placement(inp: np.ndarray, out: np.ndarray, bg: int, fg: int, placement: dict, valid: bool) -> None:
    hint_color = H_HINT if placement["orientation"] == "H" else V_HINT
    for r, c in placement["gap_cells"]:
        inp[int(r), int(c)] = int(bg)
        if not valid:
            out[int(r), int(c)] = int(bg)
    for r, c in placement["coords"]:
        inp[int(r), int(c)] = int(hint_color)
        out[int(r), int(c)] = int(fg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    palette = [c for c in range(N_COLORS) if c not in HINT_COLORS]
    bg = rng.choice(palette)
    fg = rng.choice([c for c in palette if c != bg])
    cfg = rng.choice(VALID_CONFIGS)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    base, h_segments, v_segments = built

    inp = base.copy()
    out = base.copy()
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    specs = [("H", True), ("H", False), ("V", True), ("V", False)]
    rng.shuffle(specs)
    valid_pairs = [(3, 2), (4, 3)]
    invalid_pairs = [(3, 3), (4, 2)]
    chosen: list[tuple[bool, dict]] = []

    for orientation, valid in specs:
        pair_choices = valid_pairs[:] if valid else invalid_pairs[:]
        rng.shuffle(pair_choices)
        placement = None
        for gap_size, chain_len in pair_choices:
            placement = _pick_placement(
                rng,
                h_segments if orientation == "H" else v_segments,
                orientation,
                base,
                fg,
                bg,
                gap_size,
                chain_len,
                occupied,
                forbidden,
            )
            if placement is not None:
                break
        if placement is None:
            return None
        _apply_placement(inp, out, bg, fg, placement, valid)
        occupied.update(placement["coords"])
        _reserve_neighborhood(forbidden, placement["coords"])
        _reserve_neighborhood(forbidden, placement["gap_cells"])
        chosen.append((valid, placement))

    gap_sizes_used = {int(p["gap_size"]) for _valid, p in chosen}
    chain_lengths_used = {int(p["chain_len"]) for _valid, p in chosen}
    if gap_sizes_used != set(GAP_SIZES) or chain_lengths_used != set(CHAIN_LENGTHS):
        return None

    stats = analyze_fusion_behavior(inp)
    if stats["h_matched"] < 1 or stats["v_matched"] < 1:
        return None
    if stats["h_unmatched"] < 1 or stats["v_unmatched"] < 1:
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo variant 19")
