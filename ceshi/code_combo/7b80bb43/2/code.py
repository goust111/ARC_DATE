# Combo variant of 7b80bb43 (index 2)
# combo_spec: D1/2+D3/1+D11/1
# Name: 多色着色填充
#
# Design analysis
# a) Concrete behavior trace:
#    Original task: one-color orthogonal line network; a 2-pixel diagonal chain marks a nearby 3-cell
#    gap; solver deletes the chain and restores the line with the same foreground color.
#    This combo changes three linked stages at once:
#    - input lines are split across two line colors by connected component;
#    - the diagonal hint chain uses a third foreground color in the input;
#    - output keeps the hint pixels, but recolors them to the color of the line they point to,
#      while the 3-cell straight gap is filled by that third color instead of the line color.
#    Small example:
#    input:  blue blue . . . blue blue   plus a yellow diagonal 2-chain near the left endpoint
#    output: blue blue yellow yellow yellow blue blue, and the two diagonal pixels become blue.
#    The geometric cue is unchanged; only the color roles are redistributed.
# b) Complexity calibration:
#    Using exactly two line colors plus one shared marker/fill color keeps the change visually obvious
#    without making the color system arbitrary. The special color repeats in every training example
#    both as input hints and as output gap fills, so the rule is supportable from few examples.
# c) Implicit-assumption audit:
#    The original solver assumed a single foreground color and identified hints topologically.
#    That breaks here. The new solver instead infers a unique "prompt color" by component shape:
#    every component of that color is a short diagonal chain with zero orthogonal same-color neighbors.
#    Gap filling is then done per owned line color, so single-fg scans are no longer used.
# d) Placement legality:
#    The generator still builds the original spaced line-network geometry first, then chooses legal
#    gap sites with the original non-overlap checks. Prompt chains are placed only on background cells,
#    remain 4-disconnected from lines, and are prevented from merging with one another.
# e) Visual observability:
#    A human can see that the input's special color appears only on diagonal hints, while in the output
#    that same color moves onto the repaired straight 3-pixel gaps. The diagonal pixels remain but adopt
#    the surrounding line color, so the rule is observable without relying on numeric color ids.
# f) Visual cleanliness:
#    Distractor chains are removed in this combo. They would have no line ownership and would make the
#    recoloring rule ambiguous. Keeping only real gap hints preserves a clean one-hint/one-gap mapping.
# g) Inferability of color changes:
#    The mapping is structural, not arbitrary: "special color" means hint/fill, while line colors are
#    inherited from the connected component touched by the hint. The generator ensures both line colors
#    recur across components and that selected gaps cover both colors whenever possible.

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


GAP_SIZE = 3
CHAIN_LENGTH = 2


def _deg4_same(grid: np.ndarray, color: int, r: int, c: int) -> int:
    h, w = grid.shape
    deg = 0
    if r > 0 and int(grid[r - 1, c]) == int(color):
        deg += 1
    if r + 1 < h and int(grid[r + 1, c]) == int(color):
        deg += 1
    if c > 0 and int(grid[r, c - 1]) == int(color):
        deg += 1
    if c + 1 < w and int(grid[r, c + 1]) == int(color):
        deg += 1
    return int(deg)


def _diag_neighbor_colors(grid: np.ndarray, r: int, c: int) -> list[int]:
    h, w = grid.shape
    found: list[int] = []
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            found.append(int(grid[rr, cc]))
    return found


def _has_diag_same(grid: np.ndarray, color: int, r: int, c: int) -> bool:
    return any(int(col) == int(color) for col in _diag_neighbor_colors(grid, r, c))


def _segments_from_sorted(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            segs.append((int(start), int(prev)))
            start = prev = idx
    segs.append((int(start), int(prev)))
    return segs


def _is_prompt_component(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != CHAIN_LENGTH:
        return False
    for r, c in coords:
        r = int(r)
        c = int(c)
        if _deg4_same(grid, color, r, c) != 0:
            return False
        if not _has_diag_same(grid, color, r, c):
            return False
    return True


def _detect_special_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != int(bg)):
        comps = components_mask(grid == int(color), connectivity=8)
        if comps and all(_is_prompt_component(grid, color, coords) for coords in comps):
            candidates.append(int(color))
    assert len(candidates) == 1, f"expected one prompt color, got {candidates}"
    return int(candidates[0])


def _owner_color_for_prompt(
    grid: np.ndarray, coords: np.ndarray, line_colors: set[int]
) -> int:
    counts: Counter[int] = Counter()
    for r, c in coords:
        for color in _diag_neighbor_colors(grid, int(r), int(c)):
            if int(color) in line_colors:
                counts[int(color)] += 1
    assert counts, "prompt chain does not touch any line color diagonally"
    owner, top = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    tied = [color for color, count in counts.items() if int(count) == int(top)]
    assert len(tied) == 1, f"ambiguous line ownership for prompt: {counts}"
    return int(owner)


def _locate_gap_for_prompt(mask: np.ndarray, prompt_mask: np.ndarray) -> tuple[str, int, int, int]:
    h, w = mask.shape
    candidates: list[tuple[str, int, int, int]] = []

    for r in range(h):
        segs = _segments_from_sorted(np.where(mask[r, :])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if int(gap1 - gap0 + 1) != GAP_SIZE:
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(h - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(w - 1), int(gap1 + 1))
            if bool(prompt_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                candidates.append(("h", int(r), int(gap0), int(gap1)))

    for c in range(w):
        segs = _segments_from_sorted(np.where(mask[:, c])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if int(gap1 - gap0 + 1) != GAP_SIZE:
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(h - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(w - 1), int(c + 1))
            if bool(prompt_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                candidates.append(("v", int(c), int(gap0), int(gap1)))

    assert len(candidates) == 1, f"expected one gap near prompt, got {candidates}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor diagonal hints onto their owner lines and use the hint color to bridge each gap.

    Input structure:
    - bg = background; two line colors form separated axis-aligned segment networks.
    - One special foreground color appears only as 2-pixel diagonal hint chains beside 3-cell gaps.

    Rule:
    - Infer which line color each diagonal hint chain belongs to from diagonal adjacency.
    - Recolor the hint pixels to that owner line color instead of deleting them.
    - Fill the nearby horizontal/vertical 3-cell gap with the special hint color.

    Key insight: The special color does not represent a third line network; it is a role marker that
    moves from diagonal hints in the input to straight bridge cells in the output, while the hint cells
    themselves inherit the color of the line they annotate.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    special = int(_detect_special_color(g, bg))
    line_colors = {int(x) for x in np.unique(g) if int(x) not in (int(bg), int(special))}
    assert len(line_colors) >= 1

    out = g.copy()
    special_mask = g == int(special)
    for coords in components_mask(special_mask, connectivity=8):
        owner = _owner_color_for_prompt(g, coords, line_colors)
        prompt_mask = np.zeros_like(special_mask, dtype=bool)
        for r, c in coords:
            prompt_mask[int(r), int(c)] = True
            out[int(r), int(c)] = int(owner)

        gap = _locate_gap_for_prompt(g == int(owner), prompt_mask)
        if gap[0] == "h":
            _tag, row, gap0, gap1 = gap
            out[int(row), int(gap0) : int(gap1) + 1] = int(special)
        else:
            _tag, col, gap0, gap1 = gap
            out[int(gap0) : int(gap1) + 1, int(col)] = int(special)

    return out


# --------------- generator (combo output-first) ---------------

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

GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

GEN_RETRIES = 400


def _pick_distinct(
    rng: random.Random, values: list[int], k: int, gap: int = ROW_COL_GAP
) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - p) >= int(gap) for p in picked):
            picked.append(int(v))
            if len(picked) == int(k):
                return sorted(picked)
    return []


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
    grid: np.ndarray, coords: list[tuple[int, int]], fg: int, occupied: set[tuple[int, int]]
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


def _place_t_component(
    out: np.ndarray, fg: int, slot: str, row: int, branch_col: int, end_col: int
) -> tuple[dict, dict]:
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


def _h_gap_options(
    seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]
) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= int(b) <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = int(bg)
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((int(g), anchor))
    return options


def _v_gap_options(
    seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]
) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= int(b) <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = int(bg)
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_chain(temp, coords, fg, occupied):
                    continue
                options.append((int(g), anchor, int(dc)))
    return options


def _build_output(
    rng: random.Random, h: int, w: int, bg: int, fg: int, cfg: dict
) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), int(bg), dtype=int)
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

    simple_h_rows = (
        _pick_distinct(rng, list(range(center_start, center_end + 1)), h_simple) if h_simple else []
    )
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


def _component_id_grid(base: np.ndarray, bg: int) -> tuple[np.ndarray, list[np.ndarray]]:
    comps = components_mask(base != int(bg), connectivity=4)
    comp_grid = np.full(base.shape, -1, dtype=int)
    for idx, coords in enumerate(comps):
        for r, c in coords:
            comp_grid[int(r), int(c)] = int(idx)
    return comp_grid, comps


def _annotate_segment_components(
    h_segments: list[dict], v_segments: list[dict], comp_grid: np.ndarray
) -> None:
    for seg in h_segments:
        seg["comp"] = int(comp_grid[int(seg["row"]), int(seg["c0"])])
    for seg in v_segments:
        seg["comp"] = int(comp_grid[int(seg["r0"]), int(seg["col"])])


def _assign_component_colors(
    rng: random.Random, n_comp: int, line_colors: tuple[int, int]
) -> dict[int, int]:
    order = list(range(n_comp))
    rng.shuffle(order)
    palette = list(line_colors)
    rng.shuffle(palette)
    comp_colors: dict[int, int] = {}
    for idx, comp in enumerate(order):
        comp_colors[int(comp)] = int(palette[idx % len(palette)])
    return comp_colors


def _paint_components(
    base: np.ndarray, bg: int, comps: list[np.ndarray], comp_colors: dict[int, int]
) -> np.ndarray:
    out = np.full(base.shape, int(bg), dtype=int)
    for comp_id, coords in enumerate(comps):
        color = int(comp_colors[int(comp_id)])
        for r, c in coords:
            out[int(r), int(c)] = int(color)
    return out


def _segment_color(seg: dict, comp_colors: dict[int, int]) -> int:
    return int(comp_colors[int(seg["comp"])])


def _choose_gaps(
    rng: random.Random,
    base: np.ndarray,
    bg: int,
    fg: int,
    h_segments: list[dict],
    v_segments: list[dict],
    comp_colors: dict[int, int],
    gap_count: int,
) -> list[dict] | None:
    v_gap_count = int(gap_count - 1)
    if len(v_segments) < v_gap_count:
        return None

    for _ in range(80):
        occupied: set[tuple[int, int]] = set()
        chosen: list[dict] = []

        h_order = h_segments[:]
        v_order = v_segments[:]
        rng.shuffle(h_order)
        rng.shuffle(v_order)

        found_h = False
        for seg in h_order:
            opts = _h_gap_options(seg, base, fg, bg, occupied)
            if not opts:
                continue
            gap_start, anchor = rng.choice(opts)
            chain = _chain_coords(int(seg["row"]), int(gap_start), anchor)
            chosen.append(
                {
                    "axis": "h",
                    "seg": seg,
                    "gap_start": int(gap_start),
                    "anchor": anchor,
                    "chain": chain,
                    "color": _segment_color(seg, comp_colors),
                }
            )
            occupied.update(chain)
            found_h = True
            break
        if not found_h:
            continue

        for seg in v_order:
            if len([item for item in chosen if item["axis"] == "v"]) >= v_gap_count:
                break
            opts = _v_gap_options(seg, base, fg, bg, occupied)
            if not opts:
                continue
            gap_start, anchor, dc = rng.choice(opts)
            chain = _v_chain_coords(int(seg["col"]), int(gap_start), anchor, int(dc))
            chosen.append(
                {
                    "axis": "v",
                    "seg": seg,
                    "gap_start": int(gap_start),
                    "anchor": anchor,
                    "dc": int(dc),
                    "chain": chain,
                    "color": _segment_color(seg, comp_colors),
                }
            )
            occupied.update(chain)

        if len(chosen) != gap_count:
            continue
        if len({int(item["color"]) for item in chosen}) < 2:
            continue
        return chosen

    return None


def _apply_combo_transformation(
    base_out: np.ndarray, bg: int, special: int, chosen_gaps: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    output = base_out.copy()
    inp = base_out.copy()

    for gap in chosen_gaps:
        color = int(gap["color"])
        gap_start = int(gap["gap_start"])
        chain = gap["chain"]
        if gap["axis"] == "h":
            row = int(gap["seg"]["row"])
            output[row, gap_start : gap_start + GAP_SIZE] = int(special)
            inp[row, gap_start : gap_start + GAP_SIZE] = int(bg)
        else:
            col = int(gap["seg"]["col"])
            output[gap_start : gap_start + GAP_SIZE, col] = int(special)
            inp[gap_start : gap_start + GAP_SIZE, col] = int(bg)

        for r, c in chain:
            output[int(r), int(c)] = int(color)
            inp[int(r), int(c)] = int(special)

    return inp, output


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    bg = rng.randrange(N_COLORS)
    usable = [c for c in range(N_COLORS) if c != bg]
    special = rng.choice(usable)
    usable = [c for c in usable if c != special]
    line_colors = tuple(rng.sample(usable, 2))
    base_fg = int(line_colors[0])

    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])

    built = _build_output(rng, h, w, bg, base_fg, cfg)
    if built is None:
        return None
    base, h_segments, v_segments = built

    comp_grid, comps = _component_id_grid(base, bg)
    if len(comps) < 2:
        return None
    _annotate_segment_components(h_segments, v_segments, comp_grid)
    comp_colors = _assign_component_colors(rng, len(comps), line_colors)
    colored_base = _paint_components(base, bg, comps, comp_colors)

    chosen_gaps = _choose_gaps(rng, base, bg, base_fg, h_segments, v_segments, comp_colors, gap_count)
    if chosen_gaps is None:
        return None

    inp, output = _apply_combo_transformation(colored_base, bg, special, chosen_gaps)

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid combo input for 7b80bb43")
