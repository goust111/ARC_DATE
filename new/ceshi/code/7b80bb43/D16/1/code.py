# Variant of 7b80bb43
# Dimension: D16 — 提示链与间隙端点的空间关系 (src)
# Change: 对角链从间隙端点的对角方向延伸，链的第一个像素位于端点的对角位置
#   → 提示链与间隙在同一行/列（平行偏移）

from __future__ import annotations

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import draw_line, find_components, get_bg


# Design analysis for D16 parallel-offset variant
# -----------------------------------------------
# a) Concrete behavior / trace
#    - Original task: a 3-cell straight gap is hinted by a detached 2-pixel diagonal chain whose
#      first pixel sits diagonally off a gap endpoint. Solver can therefore delete all diagonal-only
#      pixels, then scan for short gaps with any deleted diagonal in the local 3xN neighborhood.
#    - Variant: the hint is still a 2-pixel diagonal chain, but it is no longer detached from the
#      gap endpoint. Instead, the first chain pixel is placed in the adjacent parallel strip:
#        horizontal gap on row r:
#          left-anchored hint below:  (r+1,g-1),(r+2,g)
#          right-anchored hint below: (r+1,g+3),(r+2,g+2)
#          and symmetric above-row versions are also allowed.
#        vertical gap on col c:
#          top-anchored hint right:   (g-1,c+1),(g,c+2)
#          bottom-anchored hint right:(g+3,c+1),(g+2,c+2)
#          and symmetric left-column versions are also allowed.
#      The first chain pixel now touches the line endpoint orthogonally, so the gap-hint becomes a
#      2-pixel spur attached to an elbow cell. A concrete horizontal trace:
#        clean output:   ####___####  on row r with a 3-cell missing region g..g+2
#        old corruption: endpoint at g-1, hint starts at (r+1,g)
#        new corruption: endpoint at g-1, hint starts at (r+1,g-1), then (r+2,g)
#      Old solver would classify only detached diagonal-only pixels; here the attached spur pixel is
#      not diagonal-only, so the old pipeline misses the hint. The new solver must detect an elbow
#      root cell with one line continuation and one diagonal spur.
# b) Complexity calibration
#    - The variant introduces exactly one new local motif: an L-shaped endpoint + 2-pixel diagonal
#      spur. That is visually distinct from the original detached chains and from the clean output,
#      which contains only straight lines and T-junctions, never elbows.
#    - The allowed side choices are small and repeated: horizontal hints use row offsets {-1,+1},
#      vertical hints use column offsets {-1,+1}. This keeps the family learnable while avoiding a
#      single frozen placement.
# c) Original-code assumptions that break
#    - `deg4==0 and has_diag(...)` no longer captures all true gap hints because the first hint
#      pixel has one orthogonal fg neighbor by construction.
#    - The original gap-fill trigger "any removed diagonal within rows r-1..r+1 / cols gap-1..gap+1"
#      assumes hints sit next to the gap body. The variant instead encodes the gap through an elbow
#      attached to one endpoint, so filling must be inferred from the root endpoint geometry.
#    - The original root-count assert for detached diagonal components is tied to the old detached
#      placement and should not be reused for attached spurs.
# d) Placement legality
#    - Generator still builds the clean output first, then cuts 3-cell gaps.
#    - A valid hint requires: the two chain cells are currently bg, the first cell has exactly one
#      orthogonal fg neighbor (the chosen endpoint), the second cell has no orthogonal fg neighbors,
#      and the second cell is diagonally connected to the first while remaining away from other
#      segments/hints.
#    - Hints are neighborhood-reserved after placement so two local motifs cannot overlap or create
#      ambiguous elbows.
# e) Visual observability
#    - Across examples, the solver-visible clue is now "every broken line endpoint has a little
#      elbow-plus-diagonal spur hanging in the adjacent row/column", not "a detached staircase near
#      the gap". This is visible from 2-3 examples because the clean outputs never contain elbows.
# f) Visual cleanliness
#    - The network topology and color scheme stay unchanged; only the hint placement changes.
#      Spacing constraints are preserved, and hints remain 2 pixels long, so the grids stay sparse.
# g) Color reasoning
#    - Colors are unchanged: one fg, one bg. No color-based branching is introduced.
# h) New solver assumptions to audit against the generator
#    - Solver assumes every true gap-hint root is an elbow fg cell with exactly one horizontal and
#      one vertical neighbor; generator guarantees clean outputs have no such elbows, so any elbow
#      in input must come from a hint.
#    - Solver assumes the spur neighbor has deg4==1 and the diagonal leaf has deg4==0, and that a
#      3-cell bg gap plus a continuing segment exists in the inferred missing direction; generator
#      enforces exactly that for every placed hint.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: remove both detached diagonal distractors and attached parallel-offset hint spurs,
    then restore the 3-cell line gaps indicated by those spurs.
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

    def ortho_fg(m: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        if r > 0 and bool(m[r - 1, c]):
            out.append((r - 1, c))
        if r + 1 < H and bool(m[r + 1, c]):
            out.append((r + 1, c))
        if c > 0 and bool(m[r, c - 1]):
            out.append((r, c - 1))
        if c + 1 < W and bool(m[r, c + 1]):
            out.append((r, c + 1))
        return out

    def diag_fg(m: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W and bool(m[rr, cc]):
                out.append((rr, cc))
        return out

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        return bool(diag_fg(m, r, c))

    attached: set[tuple[int, int]] = set()
    gap_fill: set[tuple[int, int]] = set()
    used_roots: set[tuple[int, int]] = set()

    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or deg4(mask, r, c) != 2:
                continue
            nbrs = ortho_fg(mask, r, c)
            horiz = [p for p in nbrs if p[0] == r]
            vert = [p for p in nbrs if p[1] == c]
            if len(horiz) != 1 or len(vert) != 1:
                continue

            root = (r, c)
            if root in used_roots:
                continue

            found = False
            for spur in nbrs:
                if deg4(mask, spur[0], spur[1]) != 1:
                    continue
                for leaf in diag_fg(mask, spur[0], spur[1]):
                    if leaf == root or deg4(mask, leaf[0], leaf[1]) != 0:
                        continue

                    sr = spur[0] - r
                    sc = spur[1] - c
                    dr = leaf[0] - spur[0]
                    dc = leaf[1] - spur[1]

                    gap_cells: list[tuple[int, int]] | None = None
                    cont: tuple[int, int] | None = None
                    line_neighbor: tuple[int, int] | None = None

                    if sc == 0 and abs(sr) == 1 and dr == sr and abs(dc) == 1:
                        step = dc
                        line_neighbor = (r, c - step)
                        gap_cells = [(r, c + step * i) for i in range(1, 4)]
                        cont = (r, c + step * 4)
                    elif sr == 0 and abs(sc) == 1 and dc == sc and abs(dr) == 1:
                        step = dr
                        line_neighbor = (r - step, c)
                        gap_cells = [(r + step * i, c) for i in range(1, 4)]
                        cont = (r + step * 4, c)
                    else:
                        continue

                    if line_neighbor not in nbrs:
                        continue
                    if cont is None or not (0 <= cont[0] < H and 0 <= cont[1] < W):
                        continue
                    if not bool(mask[cont[0], cont[1]]):
                        continue
                    if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in gap_cells):
                        continue
                    if any(bool(mask[rr, cc]) for rr, cc in gap_cells):
                        continue

                    attached.add(spur)
                    attached.add(leaf)
                    gap_fill.update(gap_cells)
                    used_roots.add(root)
                    found = True
                    break
                if found:
                    break

    diag_only = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if (r, c) in attached:
                continue
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag_only[r, c] = True

    remove = diag_only.copy()
    for r, c in attached:
        remove[r, c] = True

    near = np.zeros((H, W), dtype=bool)
    for r, c in np.argwhere(remove):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    near[rr, cc] = True

    out = g.copy()
    out[remove] = bg
    for r, c in gap_fill:
        out[r, c] = fg

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

# Gap / chain placement
GAP_SIZE = 3
CHAIN_LENGTH = 2
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


def _chain_coords(row: int, gap_start: int, anchor: str, dr: int) -> list[tuple[int, int]]:
    """Attached diagonal chain for a horizontal gap; first pixel sits in the adjacent row."""
    if anchor == "left":
        return [(row + dr, gap_start - 1), (row + 2 * dr, gap_start)]
    return [(row + dr, gap_start + GAP_SIZE), (row + 2 * dr, gap_start + GAP_SIZE - 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    """Attached diagonal chain for a vertical gap; first pixel sits in the adjacent column."""
    if anchor == "top":
        return [(gap_start - 1, col + dc), (gap_start, col + 2 * dc)]
    return [(gap_start + GAP_SIZE, col + dc), (gap_start + GAP_SIZE - 1, col + 2 * dc)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or grid[r, c] != color:
            return False
    return True


def _deg4_grid(grid: np.ndarray, fg: int, r: int, c: int) -> int:
    h, w = grid.shape
    d = 0
    if r > 0 and grid[r - 1, c] == fg:
        d += 1
    if r + 1 < h and grid[r + 1, c] == fg:
        d += 1
    if c > 0 and grid[r, c - 1] == fg:
        d += 1
    if c + 1 < w and grid[r, c + 1] == fg:
        d += 1
    return d


def _diag_fg_grid(grid: np.ndarray, fg: int, r: int, c: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg:
            out.append((rr, cc))
    return out


def _validate_parallel_chain(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    fg: int,
    reserved: set[tuple[int, int]],
    root: tuple[int, int],
) -> bool:
    """Validate the attached hint motif required by the new solver."""
    if len(coords) != CHAIN_LENGTH:
        return False
    h, w = grid.shape
    coord_set = set(coords)
    if any(cell in reserved for cell in coords):
        return False
    if any(not (0 <= r < h and 0 <= c < w) for r, c in coords):
        return False
    if any(grid[r, c] != get_bg(grid) for r, c in coords):
        return False

    spur, leaf = coords
    if _deg4_grid(grid, fg, spur[0], spur[1]) != 1:
        return False
    if _deg4_grid(grid, fg, leaf[0], leaf[1]) != 0:
        return False

    root_neighbors = []
    for rr, cc in ((spur[0] - 1, spur[1]), (spur[0] + 1, spur[1]), (spur[0], spur[1] - 1), (spur[0], spur[1] + 1)):
        if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == fg:
            root_neighbors.append((rr, cc))
    if root_neighbors != [root]:
        return False

    diag_spur = set(_diag_fg_grid(grid, fg, spur[0], spur[1])) | {
        cell for cell in coord_set if abs(cell[0] - spur[0]) == 1 and abs(cell[1] - spur[1]) == 1
    }
    diag_leaf = set(_diag_fg_grid(grid, fg, leaf[0], leaf[1])) | {
        cell for cell in coord_set if abs(cell[0] - leaf[0]) == 1 and abs(cell[1] - leaf[1]) == 1
    }
    return leaf in diag_spur and spur in diag_leaf


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


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, reserved: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            root = (row, g - 1) if anchor == "left" else (row, g + GAP_SIZE)
            for dr in (-1, 1):
                coords = _chain_coords(row, g, anchor, dr)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_parallel_chain(temp, coords, fg, reserved, root):
                    continue
                options.append((g, anchor, dr))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, reserved: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            root = (g - 1, col) if anchor == "top" else (g + GAP_SIZE, col)
            for dc in (-1, 1):
                coords = _v_chain_coords(col, g, anchor, dc)
                if not _cells_are_color(temp, coords, bg):
                    continue
                if not _validate_parallel_chain(temp, coords, fg, reserved, root):
                    continue
                options.append((g, anchor, dc))
    return options


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
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
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    reserved: set[tuple[int, int]] = set()

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, out, fg, bg, reserved)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, out, fg, bg, reserved)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, out, fg, bg, reserved) or h_initial
    gap_start, anchor, dr = rng.choice(h_opts)
    row = h_seg["row"]
    root = (row, gap_start - 1) if anchor == "left" else (row, gap_start + GAP_SIZE)
    chain = _chain_coords(row, gap_start, anchor, dr)
    inp[row, gap_start : gap_start + GAP_SIZE] = bg
    for rr, cc in chain:
        inp[rr, cc] = fg
    _reserve_neighborhood(reserved, [root, *chain])

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, bg, reserved) or v_initial
        gap_start, anchor, dc = rng.choice(v_opts)
        col = v_seg["col"]
        root = (gap_start - 1, col) if anchor == "top" else (gap_start + GAP_SIZE, col)
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for rr, cc in chain:
            inp[rr, cc] = fg
        _reserve_neighborhood(reserved, [root, *chain])

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D16 variant")
