# Variant of 7b80bb43
# Dimension: D1 — 间隙填充变换方式 (transform)
# Change: 删除对角提示像素后，在提示相邻处填补水平/垂直直线上的3像素间隙，恢复断裂的线段... → 间隙扩展而非填充
#
# Design analysis
# ---------------
# variant_spec:
#   source_task: none
#   retained_structure:
#     - grid仍然只有1个背景色和1个前景色
#     - 前景主体仍然是若干轴对齐的水平/垂直线段网络
#     - 输入里仍然有2-3个“真实提示链”与0-2个开放背景中的干扰对角链
#     - 对角链像素仍满足：没有正交前景邻居，但至少有一个对角前景邻居
#   changed_behavior:
#     - 原题：删掉提示链后，把其旁边的3格线段缺口补回去
#     - 变体：删掉提示链后，不修复缺口，反而继续把缺口左右/上下各扩大1格
#     - 因此真实缺口在输入中长3格，在输出中长5格
#   concrete_pixel_trace:
#     - 设一条水平线在输入中为 `###...###`，其中 `...` 是3格缺口，缺口一端旁边还有2格45度提示链
#     - 原题输出：提示链删除，`...` 被涂回，得到连续直线
#     - 本变体输出：提示链删除，同时把缺口两端相邻线像素也删除，得到 `##.....##`
#     - 差异的关键不在“识别提示链”，而在“提示链附近的线段缺口应该被放大而非修补”
#   complexity_calibration:
#     - 若直接让生成器沿用原题的完整输出网络，则很多样例会退化为“只是删掉几颗提示像素”，看不出反转规则
#     - 为了让D1变化肉眼可见，生成器固定真实缺口数为2-3，并保证每个真实缺口都会从3格变成5格
#     - 同时保留H/V两种朝向，避免解题者将规则误判为只影响某一方向
#   implicit_assumption_review:
#     - 原solver扫描同一行/列上由提示链邻接的短缺口，然后无条件 `out[...] = fg`
#       旧前提：短缺口都应被修复
#       新变体下该前提失效；同一定位步骤仍可复用，但动作必须改成清除缺口两侧端点
#     - 原generator把“干净完整线网”当作输出，再反推输入
#       新变体下输出不再是完整线网，因此必须改成：
#       先造完整线网作为隐藏母体，再同时导出
#         1. 输入：3格缺口 + 提示链
#         2. 输出：对应位置的5格缺口
#     - 原gap候选允许缺口靠左端只留2格
#       在本变体中若再多删1格，残段只剩1格，视觉上会过短
#       因此生成器收紧候选，保证5格缺口两边至少各保留2格
#   placement_legality:
#     - 真实缺口仍不能贴近T结点或线段自由端，否则扩张时会误删结点/端点
#     - 对角链仍放在3格输入缺口的一端外侧，且链像素必须落在背景上、无正交前景邻接
#     - 干扰链仍只放在开放背景区，并保留Chebyshev-1缓冲，避免与真实线网混淆
#   visual_observability:
#     - 在2-4个样例中，解题者可以直接观察到：所有提示链都消失了，但提示附近的断口不是消失，而是更宽
#     - 5格缺口比3格缺口明显，且在H/V两个方向都会重复出现，足以支撑“反向操作”归纳
#   cleanliness:
#     - 仍然只使用单一前景色，提示链与线网同色，避免引入额外颜色规则
#     - 通过原有布局约束与额外的缺口边距约束，保证输出仍是清晰的线网而非碎片噪声
#   solver_generator_contract:
#     - solver假设真实提示只标记3格缺口；generator通过固定 `GAP_SIZE == 3` 保证这一点
#     - solver假设扩张后的5格缺口不会碰到T结点；generator通过blocked区和边距约束保证这一点

from __future__ import annotations

import random

import numpy as np

from arc_utils import draw_line, find_components, get_bg


BASE_GAP_SIZE = 3
EXPANSION = 1
EXPANDED_GAP_SIZE = BASE_GAP_SIZE + 2 * EXPANSION


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
    if r > 0 and c > 0 and bool(mask[r - 1, c - 1]):
        return True
    if r > 0 and c + 1 < w and bool(mask[r - 1, c + 1]):
        return True
    if r + 1 < h and c > 0 and bool(mask[r + 1, c - 1]):
        return True
    if r + 1 < h and c + 1 < w and bool(mask[r + 1, c + 1]):
        return True
    return False


def _diag_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and _deg4(mask, r, c) == 0 and _has_diag(mask, r, c):
                diag[r, c] = True
    return diag


def _near_mask(diag: np.ndarray) -> np.ndarray:
    h, w = diag.shape
    near = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(diag):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    near[rr, cc] = True
    return near


def _segments_1d(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            segs.append((start, prev))
            start = prev = value
    segs.append((start, prev))
    return segs


def _collect_expansion_sites(mask: np.ndarray, diag: np.ndarray) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """
    Return hint-marked 3-cell gaps to be expanded to 5 cells.

    Horizontal site tuple: (row, gap0, gap1)
    Vertical site tuple: (col, gap0, gap1)
    """
    h, w = mask.shape
    core = mask & (~diag)
    horizontal: list[tuple[int, int, int]] = []
    vertical: list[tuple[int, int, int]] = []

    for r in range(h):
        segs = _segments_1d(np.where(core[r, :])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if int(gap1 - gap0 + 1) != BASE_GAP_SIZE:
                continue
            rr0 = max(0, r - 1)
            rr1 = min(h - 1, r + 1)
            cc0 = max(0, gap0 - 1)
            cc1 = min(w - 1, gap1 + 1)
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                horizontal.append((r, gap0, gap1))

    for c in range(w):
        segs = _segments_1d(np.where(core[:, c])[0])
        for (_a1, b1), (a2, _b2) in zip(segs, segs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            if int(gap1 - gap0 + 1) != BASE_GAP_SIZE:
                continue
            rr0 = max(0, gap0 - 1)
            rr1 = min(h - 1, gap1 + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w - 1, c + 1)
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                vertical.append((c, gap0, gap1))

    return horizontal, vertical


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise and widen marked 3-cell gaps to 5 cells.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

    mask = g == fg
    diag = _diag_mask(mask)
    near = _near_mask(diag)
    h_sites, v_sites = _collect_expansion_sites(mask, diag)

    out = g.copy()
    out[diag] = bg
    for r, gap0, gap1 in h_sites:
        out[r, gap0 - EXPANSION : gap1 + EXPANSION + 1] = bg
    for c, gap0, gap1 in v_sites:
        out[gap0 - EXPANSION : gap1 + EXPANSION + 1, c] = bg

    m = out == fg
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                if not bool(m[r, c]) or not bool(near[r, c]):
                    continue
                if _deg4(m, r, c) != 1:
                    continue
                if not _has_diag(m, r, c):
                    continue
                to_clear.append((r, c))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out2 = np.full(g.shape, bg, dtype=int)
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

# Simple segment sizing
SIMPLE_H_LEN_MIN = 7
SIMPLE_H_LEN_MAX = 11
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

# Gap / chain placement
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
OUTPUT_SIDE_KEEP = 2

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
    return [(row + i, gap_start + BASE_GAP_SIZE - i) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(gap_start + i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]
    return [(gap_start + BASE_GAP_SIZE - 1 - i, col + dc * (i + 1)) for i in range(CHAIN_LENGTH)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int, occupied: set[tuple[int, int]]) -> bool:
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


def _h_gap_options(seg: dict, full: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    stop = c1 - OUTPUT_SIDE_KEEP - BASE_GAP_SIZE + 1
    for g in range(c0 + OUTPUT_SIDE_KEEP + 1, stop):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = full.copy()
        temp[row, g : g + BASE_GAP_SIZE] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((g, anchor))
    return options


def _v_gap_options(seg: dict, full: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int]] = []
    stop = r1 - OUTPUT_SIDE_KEEP - BASE_GAP_SIZE + 1
    for g in range(r0 + OUTPUT_SIDE_KEEP + 1, stop):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = full.copy()
        temp[g : g + BASE_GAP_SIZE, col] = bg
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


def _build_full_output(rng: random.Random, h: int, w: int, bg: int, fg: int, cfg: dict) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    full = np.full((h, w), bg, dtype=int)
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
        h_seg, v_seg = _place_t_component(full, fg, slot, row, branch_col, end_col)
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
        h_segments.append(_place_simple_h(full, fg, side, row, rng.randint(SIMPLE_H_LEN_MIN, max_len)))

    top_v = min(2, v_simple)
    top_v_cols = simple_v_cols[:top_v]
    bottom_v_cols = simple_v_cols[top_v:]
    top_limit = min(min(top_rows) - 2 if top_rows else center_start - 2, center_start - 2)
    bottom_start = max(max(bottom_rows) + 2 if bottom_rows else center_end + 2, center_end + 2)
    for col in top_v_cols:
        if top_limit < 2:
            return None
        v_segments.append(_place_simple_v(full, fg, "top", col, rng.randint(2, top_limit)))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < 2:
            return None
        v_segments.append(_place_simple_v(full, fg, "bottom", col, rng.randint(2, max_len)))

    comps = find_components(full, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != cfg["segments"] - cfg["junctions"]:
        return None
    return full, h_segments, v_segments


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(0, 2)

    built = _build_full_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    full, h_segments, v_segments = built

    inp = full.copy()
    out = full.copy()
    occupied: set[tuple[int, int]] = set()

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, full, fg, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, full, fg, bg, occupied)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, full, fg, bg, occupied) or h_initial
    gap_start, anchor = rng.choice(h_opts)
    row = h_seg["row"]
    chain = _chain_coords(row, gap_start, anchor)
    inp[row, gap_start : gap_start + BASE_GAP_SIZE] = bg
    out[row, gap_start - EXPANSION : gap_start + BASE_GAP_SIZE + EXPANSION] = bg
    for r, c in chain:
        inp[r, c] = fg
    occupied.update(chain)

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, full, fg, bg, occupied) or v_initial
        gap_start, anchor, dc = rng.choice(v_opts)
        col = v_seg["col"]
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start : gap_start + BASE_GAP_SIZE, col] = bg
        out[gap_start - EXPANSION : gap_start + BASE_GAP_SIZE + EXPANSION, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp == fg))))
    _reserve_neighborhood(forbidden, list(occupied))
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D1 variant")
