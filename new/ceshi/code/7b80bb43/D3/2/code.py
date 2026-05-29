# Variant of 7b80bb43
# Dimension: D3 — 对角提示的删除逻辑 (transform)
# Change: 识别所有无正交前景邻居但有对角前景邻居的像素，统一删除... → 选择性删除：仅删除与间隙相邻的提示链

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT / "solvers"))

from arc_utils import components_mask, draw_line, find_components, get_bg


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Original input contains one fg line network plus diagonal-only fg chains.
#      The original solver deletes every diagonal-only fg pixel, so both
#      functional gap hints and independent distractor chains disappear.
#    - In this variant the input still contains two diagonal-chain categories:
#      1. gap_chains: 2-pixel diagonal staircases attached to a 3-cell hole in a
#         horizontal or vertical line. One chain pixel is within Chebyshev-1 of
#         the hole; deleting the whole 8-connected chain and filling that exact
#         3-cell hole restores the line.
#      2. distractor_chains: isolated 2-3 pixel diagonal staircases in open
#         background. They satisfy the same local "diagonal-only" test, but they
#         are not within Chebyshev-1 of any restorable 3-cell straight gap.
#    - Small trace:
#      old logic: mark every diagonal-only pixel -> delete both chain types ->
#      fill any nearby 3-cell line holes -> prune around all deleted diagonals.
#      new logic: mark every diagonal-only pixel as a candidate -> scan rows and
#      columns for exact 3-cell gaps in the axis-aligned network after removing
#      all candidate diagonals from consideration -> only candidate components
#      that touch one of those gap neighborhoods are labeled "functional" ->
#      delete only those components, fill only those gaps, prune only near those
#      deleted components. Distractor chains survive unchanged into the output.
#    - Observable difference: output is no longer a pure orthogonal network; it
#      preserves some diagonal islands, so the solver must classify diagonals.
# b) Complexity calibration
#    - To keep the rule visible, every generated sample includes 2-3 functional
#      gap_chains and 1-2 distractor_chains. This prevents degeneration into the
#      original "delete all diagonals" task.
#    - Distractors remain isolated and small (length 2 or 3), so the scene is
#      still readable and does not introduce unrelated object types.
#    - Functional chains keep the original exact gap size of 3, which gives a
#      stable geometric cue and avoids ambiguous long-gap matching.
# c) Implicit-assumption review of the original main()
#    - `diag[...] = True` for every diagonal-only pixel is still safe as an
#      initial candidate extraction, but no longer safe as the final deletion
#      mask. The variant therefore splits "candidate diagonals" from
#      "functional diagonals to delete".
#    - `out[diag] = bg` depended on all diagonal-only pixels being disposable.
#      That fails now because distractors must survive, so deletion is deferred
#      until after component classification.
#    - The original MAXG inference used all diagonal components. That would let
#      distractor span affect restoration, so the variant keys on the original
#      spec's exact 3-cell gaps instead.
#    - The original `near` mask was built from all diagonals. In the variant
#      pruning near distractors would incorrectly erase preserved output, so
#      `near` is built only from deleted gap_chains.
# d) Placement legality
#    - Generator keeps output-first construction of the orthogonal network.
#    - Distractors are placed in open background with a reserved Chebyshev-1
#      buffer around all existing fg, so they neither overlap nor touch the
#      network, and they do not accidentally look like gap hints.
#    - Gap hints are then added at valid line gaps only, with occupancy tracking
#      so they cannot overlap distractors or one another.
#    - Oracle validation still checks `main(input) == output`.
# e) Visual observability
#    - From 2-4 examples, a human can now see that some diagonal chains vanish
#      and repair nearby lines, while others remain untouched because no broken
#      line is nearby. This is the new discriminative cue.
# f) Visual cleanliness
#    - Distractors remain separated from the line network by reserved space.
#      Functional chains remain close to exactly one gap, preventing cluttered
#      multi-cause regions.
# g) Color inference
#    - No color remapping is introduced; the task remains one fg on one bg.


GAP_SIZE = 3
CHAIN_LENGTH = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete only diagonal hint chains that certify a nearby 3-cell line gap.

    The output preserves isolated diagonal distractor chains.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors
    fg = int(next(iter(colors)))

    mask = g == int(fg)
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
        return int(d)

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and bool(m[rr, cc]):
                return True
        return False

    # Step 1: extract every diagonal-only fg pixel as a candidate chain pixel.
    diag_like = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag_like[r, c] = True

    diag_comps = components_mask(diag_like, connectivity=8)
    comp_at = np.full((H, W), -1, dtype=int)
    for cid, coords in enumerate(diag_comps):
        for r, c in coords:
            comp_at[int(r), int(c)] = int(cid)

    functional_diag = np.zeros((H, W), dtype=bool)
    near_deleted = np.zeros((H, W), dtype=bool)
    out = g.copy()

    def mark_component(cid: int) -> None:
        if cid < 0:
            return
        coords = diag_comps[int(cid)]
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            functional_diag[rr, cc] = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        near_deleted[nr, nc] = True

    # Step 2: reason about the orthogonal network after ignoring all diagonal
    # candidate pixels. This assumes only diagonal chains are 4-isolated fg; the
    # generator preserves that property for both functional hints and distractors.
    base_mask = mask & (~diag_like)

    def mark_gap_neighbors(r0: int, r1: int, c0: int, c1: int) -> bool:
        touched = False
        seen: set[int] = set()
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                cid = int(comp_at[rr, cc])
                if cid >= 0:
                    seen.add(int(cid))
        for cid in seen:
            mark_component(int(cid))
            touched = True
        return touched

    # Fill horizontal 3-cell gaps only when a candidate diagonal component lies
    # in the gap's Chebyshev-1 neighborhood.
    for r in range(H):
        cols = np.where(base_mask[r, :])[0]
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
            if int(gap1 - gap0 + 1) != int(GAP_SIZE):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(H - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(W - 1), int(gap1 + 1))
            if mark_gap_neighbors(rr0, rr1, cc0, cc1):
                out[r, gap0 : gap1 + 1] = int(fg)

    # Fill vertical 3-cell gaps only when a candidate diagonal component lies
    # in the gap's Chebyshev-1 neighborhood.
    for c in range(W):
        rows = np.where(base_mask[:, c])[0]
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
            if int(gap1 - gap0 + 1) != int(GAP_SIZE):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(H - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(W - 1), int(c + 1))
            if mark_gap_neighbors(rr0, rr1, cc0, cc1):
                out[gap0 : gap1 + 1, c] = int(fg)

    out[functional_diag] = int(bg)

    # Step 3: prune only around deleted functional hints. Preserved distractors
    # must not trigger cleanup, so the neighborhood mask is selective.
    m = out == int(fg)
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not bool(m[r, c]) or not bool(near_deleted[r, c]):
                    continue
                if deg4(m, r, c) != 1:
                    continue
                if not has_diag(m, r, c):
                    continue
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out2 = np.full((H, W), int(bg), dtype=int)
    out2[m] = int(fg)
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

# Simple H-segment sizing
SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

# Gap / chain placement
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

# Distractor chains
DISTRACTOR_LENGTHS = (3, 2)
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
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != color:
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
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg:
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg):
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
            coords = _chain_coords(row, g, anchor)
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


def _reserve_neighborhood(cells: set[tuple[int, int]], coords) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


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


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.choice([1, 2])

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    network, h_segments, v_segments = built

    out = network.copy()
    inp = network.copy()

    # Place distractors into both input and output so the new rule is visible.
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(network == fg))))
    distractors = _distractor_options(out, bg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any(cell in forbidden for cell in coords):
            continue
        for r, c in coords:
            out[r, c] = fg
            inp[r, c] = fg
        occupied.update(coords)
        _reserve_neighborhood(forbidden, coords)
        placed += 1
    if placed != distractor_count:
        return None

    # Collect gap candidates after reserving distractor space.
    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, out, fg, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, out, fg, bg, occupied)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    # Apply one horizontal functional gap.
    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, out, fg, bg, occupied) or h_initial
    gap_start, anchor = rng.choice(h_opts)
    row = h_seg["row"]
    chain = _chain_coords(row, gap_start, anchor)
    inp[row, gap_start : gap_start + GAP_SIZE] = bg
    for r, c in chain:
        inp[r, c] = fg
    occupied.update(chain)

    # Apply the remaining vertical functional gaps.
    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, bg, occupied) or v_initial
        gap_start, anchor, dc = rng.choice(v_opts)
        col = v_seg["col"]
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D3 variant")
