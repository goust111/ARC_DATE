# Variant of 7b80bb43
# Dimension: D12 — 对角像素的检测条件 (transform)
# Change: 同时满足两个条件：(1)无正交前景邻居(deg4=0)，(2)有至少一个对角前景邻居... → 基于连通分量大小检测

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg

# -----------------------------------------------------------------------------
# Design analysis for the D12 variant
#
# 0. Source-task note
#    - The prompt cites `b99e7126` as the anomaly-detection inspiration.
#    - In this workspace only `/ARC-data/.../b99e7126.json` is present; no solver
#      source is available to read. The implementation therefore uses the prompt's
#      stated mechanism directly: anomaly detection by connected-component size.
#
# 1. Concretized behavior
#    - The clean output is still a one-color network of axis-aligned H/V line
#      segments with 2-3 broken sites repaired in the output.
#    - The input no longer uses per-pixel "diagonal-only" stair pixels touching
#      the broken line by 8-connectivity. Instead, every cue/noise object is a
#      separate 8-connected foreground component of area 2 or 3.
#    - Gap cues are tiny floating elbow/domino blobs placed near a 3-cell gap but
#      kept at least one Chebyshev cell away from the line network. Distractors
#      use the same size class in open background.
#    - Solver consequence: classify fg by 8-connected component size. Components
#      with area <= 3 are "anomalies" and are deleted wholesale; larger
#      components are the real line network.
#
# 2. Small trace of old vs new logic
#    - Old rule on a 3-pixel elbow cue:
#        XX
#         X
#      The two orthogonally adjacent pixels have deg4>0, so the old
#      per-pixel detector would not delete the whole cue. The variant must.
#    - New rule: the elbow is one 8-connected component of area 3, so it is
#      removed entirely. The nearby 3-cell gap is then filled, because a deleted
#      small component sits within a radius-2 cue zone around the gap.
#
# 3. Complexity calibration / visibility
#    - All cue/noise components are tiny (area 2 or 3); all true network
#      components are forced by generation to have area > 3.
#    - This creates a visible categorical split that a human can infer from a few
#      examples: "small floating blobs disappear; long line structures remain."
#    - Some small blobs deliberately contain orthogonal adjacency, so the variant
#      is visibly different from the original diagonal-only staircase rule.
#
# 4. Hidden assumptions audited from the original main()
#    - Original `diag` construction assumed hint pixels can be recognized
#      independently. This is no longer valid: a cue must be removed as a whole
#      component, not pixel-by-pixel.
#    - Original gap filling searched for deleted pixels inside a radius-1 box
#      around the gap because cues touched the line diagonally. New cues are
#      8-disconnected, so the cue search radius must expand to 2.
#    - Original local pruning was needed because deleting touching diagonal chains
#      could leave leaf pixels. New cues are disconnected, so pruning should be a
#      no-op in valid generated instances. It is kept in a conservative form.
#
# 5. Placement legality requirements enforced by the generator
#    - Every inserted cue/distractor must:
#      * lie on background cells,
#      * stay within bounds,
#      * be 8-disconnected from the main network and from other small blobs,
#      * be internally 8-connected and have area <= 3.
#    - Every genuine network component in both the clean output and the corrupted
#      input must have area > 3, otherwise the solver's size classifier would
#      delete real structure. The generator validates this explicitly.
#
# 6. Human-solvability / neatness
#    - The grid still contains only bg+fg.
#    - No color-id semantics are used.
#    - Small blobs are sparse and separated from the network, so the display
#      remains readable: long orthogonal lines vs tiny floating anomalies.
# -----------------------------------------------------------------------------


SMALL_COMPONENT_MAX = 3
GAP_SIZE = 3
CUE_RADIUS = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove tiny disconnected cue/noise components and complete nearby 3-cell gaps
    in an axis-aligned foreground line network.
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
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and bool(m[rr, cc]):
                    return True
        return False

    # Variant core: classify the foreground by 8-connected component size.
    small = np.zeros((H, W), dtype=bool)
    for coords in components_mask(mask, connectivity=8):
        if 0 < len(coords) <= SMALL_COMPONENT_MAX:
            small[tuple(coords.T)] = True

    # Radius-1 neighborhood is enough for conservative pruning; radius-2 is used
    # separately for cue-guided gap filling because the cue no longer touches the
    # line network diagonally.
    near = np.zeros((H, W), dtype=bool)
    cue = np.zeros((H, W), dtype=bool)
    for coords in components_mask(small, connectivity=8):
        if coords.size == 0:
            continue
        for r, c in coords:
            for dr in range(-CUE_RADIUS, CUE_RADIUS + 1):
                for dc in range(-CUE_RADIUS, CUE_RADIUS + 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        cue[rr, cc] = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    out = g.copy()
    out[small] = bg

    network = mask & (~small)

    # Fill short horizontal gaps between network segments when a deleted small
    # component sits near the gap.
    for r in range(H):
        cols = np.where(network[r, :])[0]
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
            if gap1 < gap0:
                continue
            if gap1 - gap0 + 1 > GAP_SIZE:
                continue
            rr0 = max(0, r - CUE_RADIUS)
            rr1 = min(H - 1, r + CUE_RADIUS)
            cc0 = max(0, gap0 - CUE_RADIUS)
            cc1 = min(W - 1, gap1 + CUE_RADIUS)
            if bool(cue[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[r, gap0 : gap1 + 1] = fg

    # Fill short vertical gaps between network segments when a deleted small
    # component sits near the gap.
    for c in range(W):
        rows = np.where(network[:, c])[0]
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
            if gap1 < gap0:
                continue
            if gap1 - gap0 + 1 > GAP_SIZE:
                continue
            rr0 = max(0, gap0 - CUE_RADIUS)
            rr1 = min(H - 1, gap1 + CUE_RADIUS)
            cc0 = max(0, c - CUE_RADIUS)
            cc1 = min(W - 1, c + CUE_RADIUS)
            if bool(cue[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[gap0 : gap1 + 1, c] = fg

    # Conservative carry-over from the original logic. In valid generated
    # examples this should do nothing because deleted blobs are disconnected.
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


# --------------- generator (D12 variant) ---------------

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
SIMPLE_V_LEN_MIN = 4
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

DISTRACTOR_MARGIN = 1
GEN_RETRIES = 600


H_HINT_PATTERNS = {
    "left": [
        ((2, 0), (2, 1), (3, 1)),
        ((2, 1), (3, 0), (3, 1)),
    ],
    "right": [
        ((2, 1), (2, 2), (3, 1)),
        ((2, 1), (3, 1), (3, 2)),
    ],
}

V_HINT_PATTERNS = {
    "left": [
        ((0, -2), (1, -2), (1, -3)),
        ((1, -2), (1, -3), (2, -3)),
    ],
    "right": [
        ((0, 2), (1, 2), (1, 3)),
        ((1, 2), (1, 3), (2, 3)),
    ],
}

DISTRACTOR_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
)


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
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != color:
            return False
    return True


def _shape_coords(anchor_r: int, anchor_c: int, pattern: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [(anchor_r + dr, anchor_c + dc) for dr, dc in pattern]


def _can_place_small_component(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    bg: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != bg:
            return False
        if (r, c) in forbidden:
            return False
    return True


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _h_hint_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, list[tuple[int, int]]]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, list[tuple[int, int]]]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        left_len = g - c0
        right_len = c1 - (g + GAP_SIZE) + 1
        left_has_blocked = any(c0 <= b < g for b in blocked)
        right_has_blocked = any(g + GAP_SIZE <= b <= c1 for b in blocked)
        if not left_has_blocked and left_len <= SMALL_COMPONENT_MAX:
            continue
        if not right_has_blocked and right_len <= SMALL_COMPONENT_MAX:
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = bg
        for anchor, patterns in H_HINT_PATTERNS.items():
            for pattern in patterns:
                coords = _shape_coords(row, g, pattern)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _can_place_small_component(temp, coords, bg, forbidden):
                    continue
                options.append((g, coords))
    return options


def _v_hint_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, list[tuple[int, int]]]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, list[tuple[int, int]]]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        top_len = g - r0
        bottom_len = r1 - (g + GAP_SIZE) + 1
        top_has_blocked = any(r0 <= b < g for b in blocked)
        bottom_has_blocked = any(g + GAP_SIZE <= b <= r1 for b in blocked)
        if not top_has_blocked and top_len <= SMALL_COMPONENT_MAX:
            continue
        if not bottom_has_blocked and bottom_len <= SMALL_COMPONENT_MAX:
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = bg
        for side, patterns in V_HINT_PATTERNS.items():
            for pattern in patterns:
                coords = _shape_coords(g, col, pattern)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _can_place_small_component(temp, coords, bg, forbidden):
                    continue
                options.append((g, coords))
    return options


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
        if top_limit < SIMPLE_V_LEN_MIN:
            return None
        v_segments.append(_place_simple_v(out, fg, "top", col, rng.randint(SIMPLE_V_LEN_MIN, top_limit)))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < SIMPLE_V_LEN_MIN:
            return None
        v_segments.append(_place_simple_v(out, fg, "bottom", col, rng.randint(SIMPLE_V_LEN_MIN, max_len)))

    comps4 = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps4) != cfg["segments"] - cfg["junctions"]:
        return None
    if any(len(coords) <= SMALL_COMPONENT_MAX for coords in components_mask(out == fg, connectivity=8)):
        return None
    return out, h_segments, v_segments


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for pattern in DISTRACTOR_SHAPES:
        height = max(dr for dr, _dc in pattern) + 1
        min_dc = min(dc for _dr, dc in pattern)
        max_dc = max(dc for _dr, dc in pattern)
        for r in range(DISTRACTOR_MARGIN, h - height + 1):
            for c in range(DISTRACTOR_MARGIN - min_dc, w - DISTRACTOR_MARGIN - max_dc):
                coords = _shape_coords(r, c, pattern)
                if not _can_place_small_component(inp, coords, bg, forbidden):
                    continue
                options.append(coords)
    return options


def _small_component_signature(coords: np.ndarray | list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    if isinstance(coords, np.ndarray):
        return frozenset((int(r), int(c)) for r, c in coords.tolist())
    return frozenset((int(r), int(c)) for r, c in coords)


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
    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(out == fg))))
    expected_small: list[list[tuple[int, int]]] = []

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_hint_options(seg, out, fg, bg, forbidden)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_hint_options(seg, out, fg, bg, forbidden)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_hint_options(h_seg, out, fg, bg, forbidden) or h_initial
    gap_start, hint_coords = rng.choice(h_opts)
    row = h_seg["row"]
    inp[row, gap_start : gap_start + GAP_SIZE] = bg
    for r, c in hint_coords:
        inp[r, c] = fg
    expected_small.append(hint_coords)
    _reserve_neighborhood(forbidden, hint_coords)

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_hint_options(v_seg, out, fg, bg, forbidden) or v_initial
        if not v_opts:
            return None
        gap_start, hint_coords = rng.choice(v_opts)
        col = v_seg["col"]
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in hint_coords:
            inp[r, c] = fg
        expected_small.append(hint_coords)
        _reserve_neighborhood(forbidden, hint_coords)

    distractors = _distractor_options(inp, bg, forbidden)
    rng.shuffle(distractors)
    for coords in distractors:
        if len(expected_small) >= gap_count + distractor_count:
            break
        if not _can_place_small_component(inp, coords, bg, forbidden):
            continue
        for r, c in coords:
            inp[r, c] = fg
        expected_small.append(coords)
        _reserve_neighborhood(forbidden, coords)

    if len(expected_small) != gap_count + distractor_count:
        return None

    expected_signatures = {_small_component_signature(coords) for coords in expected_small}
    actual_components = components_mask(inp == fg, connectivity=8)
    actual_small = {_small_component_signature(coords) for coords in actual_components if 0 < len(coords) <= SMALL_COMPONENT_MAX}
    if actual_small != expected_signatures:
        return None
    if any(len(coords) <= SMALL_COMPONENT_MAX and _small_component_signature(coords) not in expected_signatures for coords in actual_components):
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D12 variant")
