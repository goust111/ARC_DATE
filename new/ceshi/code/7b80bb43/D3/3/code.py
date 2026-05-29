# Variant of 7b80bb43
# Dimension: D3 — 对角提示的删除逻辑 (transform)
# Change: 识别所有无正交前景邻居但有对角前景邻居的像素，统一删除
#   → 对角提示像素被拉直为正交像素
#
# Design analysis
# - Reference task check: 7b5033c1 "linearize" turns a bent 4-connected path into a
#   straight strip while preserving local order. This variant borrows only the
#   "straighten the bend into the axis-aligned target" idea; it keeps the original
#   2D line-network setting of 7b80bb43.
# - a) Pixel concretization / behavior trace:
#   Original corruption on a horizontal line removed a 3-cell gap and placed a
#   2-cell diagonal hint, e.g. output row `..####..` could become input
#   `..#...#.` with a hint chain below it. The old solver deleted the diagonal
#   pixels and refilled the gap. In this variant the corruption is changed:
#   take the last 2 cells of a line end, remove them from the line, and place
#   them as a 45-degree chain, e.g. output has `(r,c+1),(r,c+2)` but input has
#   `(r+1,c+1),(r+2,c+2)` instead. The new solver deletes those diagonal cells
#   from their bent positions and reprojects them to `(r,c+1),(r,c+2)`, so the
#   output line becomes longer than simply deleting the chain.
# - b) Complexity calibration:
#   The bend length stays fixed at 2 cells. This keeps the mapping visible and
#   repeated across examples, while avoiding too many free degrees. The network
#   layout, number of segments, colors, and T-junction structure stay close to
#   the original task, so the visual change is attributable to D3 rather than a
#   different object family.
# - c) Hidden assumptions in the old main():
#   `out[diag]=bg` assumed every diagonal-only pixel should disappear.
#   Horizontal/vertical gap-filling loops assumed diagonal chains only mark
#   missing interior gaps; that premise is false here because chains now encode
#   bent endpoints. The pruning loop assumed leftover diagonal-adjacent leaves
#   near hints are noise; here those pixels are valid projected targets and must
#   not be pruned. Therefore those three blocks are removed/replaced.
# - d) Placement legality:
#   The generator bends only segment ends with a unique straight continuation.
#   For each candidate end, the 2 orthogonal target cells must currently be fg,
#   and the diagonal replacement cells must be in-bounds, bg, and have no
#   orthogonal fg neighbor after removing the targets. This guarantees the input
#   still matches the "diagonal-only chain" signature and the solver can recover
#   a unique root pixel.
# - e) Visual observability:
#   In the examples, the output visibly extends lines beyond the input endpoint.
#   The diagonal chain stays adjacent to that endpoint, so a human can infer
#   "straighten the bent end back into the row/column" from 2-4 samples.
# - f) Visual cleanliness:
#   Detached distractor chains are removed from the generator because under this
#   variant they would become ambiguous: there is no source line to straighten
#   them onto. Keeping only line-attached bend hints avoids clutter.
# - g) Color inference:
#   Colors are unchanged from the original task: one fg on one bg. No color-based
#   reasoning is introduced.
# - h) Quantitative variant signal used during validation:
#   Each generated input contains 2-3 diagonal hint components, and the new
#   output differs from the old-delete behavior by exactly the projected cells of
#   those components. We compare the new solver with the original 7b80bb43 solver
#   on generated samples to confirm non-degeneration.
# - i) New solver assumptions that the generator must satisfy:
#   Each diagonal hint component has exactly one diagonal-adjacent non-diagonal
#   root pixel; that root has exactly one orthogonal fg neighbor, revealing the
#   underlying axis. The generator enforces this by bending only free segment
#   ends, never junction interiors or detached noise.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten diagonal endpoint hints back into axis-aligned line extensions.

    Input structure:
    - bg = background; one foreground color fg forming axis-aligned lines.
    - Some fg pixels form 45-degree diagonal chains with no orthogonal fg
      neighbors. Each such chain is attached diagonally to a line endpoint.

    Rule:
    - Identify diagonal-only fg pixels.
    - For each diagonal component with a unique diagonal-adjacent root line pixel,
      infer whether that root belongs to a horizontal or vertical segment.
    - Move each diagonal pixel onto the same row/column as the root, preserving
      its distance from the root along the extension direction.

    Key insight: the diagonal chain is not noise to delete; it is a bent tail
    whose pixels should be orthogonally projected back onto the source segment.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors
    fg = int(next(iter(colors)))

    mask = g == fg
    h, w = g.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < h and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < w and bool(m[r, c + 1]):
            d += 1
        return d

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(m[rr, cc]):
                return True
        return False

    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag[r, c] = True

    out = g.copy()
    out[diag] = bg
    base = out == fg

    projected: set[tuple[int, int]] = set()
    for coords in components_mask(diag, connectivity=8):
        if coords.size == 0:
            continue

        roots: set[tuple[int, int]] = set()
        for r, c in coords:
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not bool(diag[rr, cc]):
                    roots.add((rr, cc))

        if len(roots) != 1:
            continue
        root_r, root_c = next(iter(roots))

        ortho = []
        for rr, cc in ((root_r - 1, root_c), (root_r + 1, root_c), (root_r, root_c - 1), (root_r, root_c + 1)):
            if 0 <= rr < h and 0 <= cc < w and bool(base[rr, cc]):
                ortho.append((rr, cc))
        if len(ortho) != 1:
            continue

        nbr_r, nbr_c = ortho[0]
        if nbr_r == root_r:
            axis = "h"
            dir_sign = 1 if float(np.mean(coords[:, 1] - root_c)) > 0 else -1
        elif nbr_c == root_c:
            axis = "v"
            dir_sign = 1 if float(np.mean(coords[:, 0] - root_r)) > 0 else -1
        else:
            continue

        for r, c in coords:
            if axis == "h":
                k = abs(int(c) - root_c)
                if k == 0:
                    continue
                target = (root_r, int(root_c + dir_sign * k))
            else:
                k = abs(int(r) - root_r)
                if k == 0:
                    continue
                target = (int(root_r + dir_sign * k), root_c)
            tr, tc = target
            if 0 <= tr < h and 0 <= tc < w:
                projected.add((tr, tc))

    result = np.full((h, w), bg, dtype=int)
    result[base] = fg
    for r, c in projected:
        result[r, c] = fg
    return result


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

# Simple segment sizing
SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

# Bend placement
BEND_LENGTH = 2
MIN_SEG_INTERIOR = 2

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


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    hh, ww = grid.shape
    for r, c in coords:
        if not (0 <= r < hh and 0 <= c < ww) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int, bg: int) -> bool:
    hh, ww = grid.shape
    coord_set = set(coords)
    for r, c in coords:
        if not (0 <= r < hh and 0 <= c < ww):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < hh and 0 <= cc < ww and int(grid[rr, cc]) == int(fg):
                return False
        has_diag_neighbor = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < hh and 0 <= cc < ww and int(grid[rr, cc]) == int(fg)):
                has_diag_neighbor = True
                break
        if not has_diag_neighbor:
            return False
    return True


def _h_chain_coords(row: int, root_c: int, dir_sign: int) -> list[tuple[int, int]]:
    return [(row + i, root_c + dir_sign * i) for i in range(1, BEND_LENGTH + 1)]


def _v_chain_coords(root_r: int, col: int, dir_sign: int, dc: int) -> list[tuple[int, int]]:
    return [(root_r + dir_sign * i, col + dc * i) for i in range(1, BEND_LENGTH + 1)]


def _place_t_component(
    out: np.ndarray, fg: int, slot: str, row: int, branch_col: int, end_col: int
) -> tuple[dict, dict]:
    hh, ww = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else ww - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else hh - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {
        "id": None,
        "axis": "h",
        "row": row,
        "c0": h_c0,
        "c1": h_c1,
        "free_dir": 1 if slot[0] == "L" else -1,
        "kind": "t_h",
    }
    v_seg = {
        "id": None,
        "axis": "v",
        "col": branch_col,
        "r0": v_r0,
        "r1": v_r1,
        "free_dir": None,
        "kind": "t_v",
    }
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    ww = out.shape[1]
    c0 = 0 if side == "left" else ww - length
    c1 = length - 1 if side == "left" else ww - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {
        "id": None,
        "axis": "h",
        "row": row,
        "c0": c0,
        "c1": c1,
        "free_dir": 1 if side == "left" else -1,
        "kind": "simple_h",
    }


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    hh = out.shape[0]
    r0 = 0 if side == "top" else hh - length
    r1 = length - 1 if side == "top" else hh - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {
        "id": None,
        "axis": "v",
        "col": col,
        "r0": r0,
        "r1": r1,
        "free_dir": 1 if side == "top" else -1,
        "kind": "simple_v",
    }


def _h_bend_options(seg: dict, grid: np.ndarray, fg: int, bg: int) -> list[dict]:
    row = int(seg["row"])
    dir_sign = int(seg["free_dir"])
    c0 = int(seg["c0"])
    c1 = int(seg["c1"])
    free_end = c1 if dir_sign == 1 else c0
    root_c = int(free_end - dir_sign * BEND_LENGTH)
    inner_c = int(root_c - dir_sign)
    if not (c0 <= root_c <= c1 and c0 <= inner_c <= c1):
        return []
    targets = [(row, int(root_c + dir_sign * i)) for i in range(1, BEND_LENGTH + 1)]
    if not _cells_are_color(grid, targets, fg):
        return []
    temp = grid.copy()
    for r, c in targets:
        temp[r, c] = bg
    coords = _h_chain_coords(row, root_c, dir_sign)
    if not _validate_chain(temp, coords, fg, bg):
        return []
    return [{"targets": targets, "chain": coords}]


def _v_bend_options(seg: dict, grid: np.ndarray, fg: int, bg: int) -> list[dict]:
    free_dir = seg["free_dir"]
    if free_dir is None:
        return []
    free_dir = int(free_dir)
    col = int(seg["col"])
    r0 = int(seg["r0"])
    r1 = int(seg["r1"])
    if int(r1 - r0 + 1) < int(BEND_LENGTH + MIN_SEG_INTERIOR):
        return []
    free_end = r1 if free_dir == 1 else r0
    root_r = int(free_end - free_dir * BEND_LENGTH)
    inner_r = int(root_r - free_dir)
    if not (r0 <= root_r <= r1 and r0 <= inner_r <= r1):
        return []
    targets = [(int(root_r + free_dir * i), col) for i in range(1, BEND_LENGTH + 1)]
    if not _cells_are_color(grid, targets, fg):
        return []
    options: list[dict] = []
    temp = grid.copy()
    for r, c in targets:
        temp[r, c] = bg
    for dc in (-1, 1):
        coords = _v_chain_coords(root_r, col, free_dir, dc)
        if _validate_chain(temp, coords, fg, bg):
            options.append({"targets": targets, "chain": coords})
    return options


def _build_output(
    rng: random.Random, h: int, w: int, bg: int, fg: int, cfg: dict
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

    for idx, seg in enumerate(h_segments + v_segments):
        seg["id"] = idx

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    return out, h_segments, v_segments


def _apply_bend(inp: np.ndarray, fg: int, option: dict) -> None:
    bg = int(get_bg(inp))
    for r, c in option["targets"]:
        inp[r, c] = bg
    for r, c in option["chain"]:
        inp[r, c] = fg


def _choose_bends(
    rng: random.Random,
    inp: np.ndarray,
    fg: int,
    bg: int,
    h_segments: list[dict],
    v_segments: list[dict],
    bend_count: int,
) -> bool:
    used_ids: set[int] = set()
    used_axes: list[str] = []

    for bend_index in range(bend_count):
        h_ready = []
        v_ready = []
        for seg in h_segments:
            if seg["id"] in used_ids:
                continue
            opts = _h_bend_options(seg, inp, fg, bg)
            if opts:
                h_ready.append((seg, opts))
        for seg in v_segments:
            if seg["id"] in used_ids:
                continue
            opts = _v_bend_options(seg, inp, fg, bg)
            if opts:
                v_ready.append((seg, opts))

        if bend_index == 0 and not h_ready:
            return False

        pool = h_ready[:]
        if v_ready and (bend_index > 0 or not h_ready):
            pool += v_ready
        if not pool:
            return False

        if bend_index > 0 and v_ready and "v" not in used_axes:
            pool = v_ready
        elif h_ready and (bend_index == 0 or rng.random() < 0.7):
            pool = h_ready

        seg, opts = rng.choice(pool)
        option = rng.choice(opts)
        _apply_bend(inp, fg, option)
        used_ids.add(int(seg["id"]))
        used_axes.append(str(seg["axis"]))

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    bend_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    if not _choose_bends(rng, inp, fg, bg, h_segments, v_segments, bend_count):
        return None

    if np.array_equal(inp, out):
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D3 variant")
