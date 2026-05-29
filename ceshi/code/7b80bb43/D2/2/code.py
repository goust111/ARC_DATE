# Variant of 7b80bb43
# Dimension: D2 — 对角提示像素的角色 (src)
# Change: 45度阶梯状对角像素（无正交前景邻居，有对角前景邻居）作为间隙位置的提示标记
#   -> 网格边缘控制像素指示间隙位置
#
# Design analysis
# yaml:
#   variant_core:
#     input:
#       - 保留原题的单色 H/V 线段网络，以及 3 像素 gap。
#       - 与原题不同，真正用于“指出哪条线有 gap”的信息不再出现在 gap 附近；
#         改为第三种颜色 ctrl，且 ctrl 只出现在网格边缘。
#       - 左/右边缘上的 ctrl 表示“该行存在一个需要修复的水平 gap”。
#       - 上/下边缘上的 ctrl 表示“该列存在一个需要修复的垂直 gap”。
#       - 实际 gap 位置仍在内部，长度固定 3；gap 附近不再放提示对角链。
#       - 0-2 组纯 fg 对角链仍可作为干扰噪声，solver 需要删除它们，但不能依赖它们补 gap。
#     output:
#       - 仍是干净的轴对齐 fg 线段网络；ctrl 和对角噪声都消失。
#   behavior_trace:
#     original:
#       - 先找 diag-only fg 像素 -> 用其邻域判断哪一行/列的短 gap 应被补上。
#     variant:
#       - 先识别“仅出现于边缘的 ctrl 颜色” -> 从 ctrl 的边缘位置读出目标行/列 ->
#         删除 diag-only 噪声 -> 仅在被 ctrl 点名的行/列上补 3 像素 gap。
#     difference:
#       - 原题是局部提示；变体是全局坐标提示。相同输入若没有边缘 ctrl，新的补线逻辑不会触发。
#   complexity_calibration:
#     - ctrl 只使用一种颜色，避免颜色规则过多。
#     - ctrl 的方向信息只靠所在边缘表达：左右=行，上下=列，档位很少，2-4 个样例即可观察。
#     - 每题仍有 2-3 个 gap，至少 1 个水平和 1 个垂直，保证行/列控制都出现。
#   implicit_assumption_review:
#     - 原 solver 中 `diag` 同时承担“删除噪声”和“定位 gap”两种职责；变体后只保留前者。
#     - 原 `near` 只在对角提示周围修剪叶子；现在它只服务于纯干扰链，不能再参与 gap 定位。
#     - 原按 `diag` 邻域补线的代码假设 gap 附近必有对角提示；该前提在变体中失效，必须替换为 ctrl 行/列。
#     - 新 solver 假设存在且仅存在一种 ctrl 颜色，并且该颜色所有像素都在边缘；generator 保证这一点。
#     - 新 solver 假设被 ctrl 标记的每条行/列上至多一个长度为 3 的 gap；generator 保证这一点。
#   placement_legality:
#     - 水平 gap 的 ctrl 放在该行未被线段占用的一侧边缘；垂直 gap 同理放在未占用的一侧边缘。
#     - 输出网络原本就要求自由端锚定到边缘，因此每个带 gap 的 segment 在其“另一侧”边缘一定留有 bg 控制位。
#     - ctrl 与线段、对角噪声都不重叠；干扰对角链仍放在开阔 bg 中。
#   visual_observability:
#     - 人类可直接看到：内部 3 格断裂附近不再有斜阶梯，而边缘新增孤立 ctrl 像素；
#       这些 ctrl 与断裂行/列对齐，跨样例重复出现，足以推断规则。
#     - 若退化回原题，旧 solver 会因为缺少 gap 附近对角提示而无法补线，因此变体行为是可区分的。
#   cleanliness:
#     - ctrl 仅在边缘，一眼可与内部 fg 网络分层区分。
#     - 干扰对角链远离线网和 ctrl，避免多层叠加导致读图混乱。

from __future__ import annotations

import random

import numpy as np

from arc_utils import draw_line, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal fg noise, read border control pixels, and fill the marked gaps.

    Variant rule:
    - fg still forms axis-aligned lines on bg.
    - ctrl is a third color used only on the outer border.
    - A ctrl pixel on the left/right border marks a row containing one 3-cell horizontal gap.
    - A ctrl pixel on the top/bottom border marks a column containing one 3-cell vertical gap.
    - Diagonal-only fg chains are now distractors only: delete them, but do not use them to infer gaps.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    colors = sorted({int(x) for x in np.unique(g) if int(x) != int(bg)})
    assert len(colors) >= 2

    def on_border(r: int, c: int) -> bool:
        return int(r) in (0, H - 1) or int(c) in (0, W - 1)

    ctrl_candidates: list[int] = []
    for col in colors:
        coords = np.argwhere(g == int(col))
        if coords.size > 0 and all(on_border(int(r), int(c)) for r, c in coords):
            ctrl_candidates.append(int(col))
    assert len(ctrl_candidates) == 1
    ctrl = int(ctrl_candidates[0])

    fg_candidates = [int(col) for col in colors if int(col) != int(ctrl)]
    assert len(fg_candidates) == 1
    fg = int(fg_candidates[0])

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    row_marks = {int(r) for r, c in ctrl_pts if int(c) in (0, W - 1)}
    col_marks = {int(c) for r, c in ctrl_pts if int(r) in (0, H - 1)}
    assert row_marks and col_marks

    mask = g == int(fg)

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

    diag = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag[r, c] = True

    near = np.zeros((H, W), dtype=bool)
    for r, c in np.argwhere(diag):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    near[rr, cc] = True

    out = g.copy()
    out[g == int(ctrl)] = int(bg)
    out[diag] = int(bg)

    GAP_SIZE = 3

    def fill_marked_row(r: int) -> None:
        cols = np.where(out[r, :] == int(fg))[0]
        if cols.size == 0:
            return
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
            if int(gap1 - gap0 + 1) == GAP_SIZE:
                out[r, gap0 : gap1 + 1] = int(fg)

    def fill_marked_col(c: int) -> None:
        rows = np.where(out[:, c] == int(fg))[0]
        if rows.size == 0:
            return
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
            if int(gap1 - gap0 + 1) == GAP_SIZE:
                out[gap0 : gap1 + 1, c] = int(fg)

    for r in sorted(row_marks):
        fill_marked_row(int(r))
    for c in sorted(col_marks):
        fill_marked_col(int(c))

    m = out == int(fg)
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
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out2 = np.full((H, W), int(bg), dtype=int)
    out2[m] = int(fg)
    return out2


# --------------- generator ---------------

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

CHAIN_LENGTH = 2
DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1

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


def _chain_coords(row: int, start_col: int, direction: int, length: int) -> list[tuple[int, int]]:
    return [(int(row + i), int(start_col + direction * i)) for i in range(int(length))]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
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
                cells.add((int(r + dr), int(c + dc)))


def _distractor_options(inp: np.ndarray, bg: int, fg: int,
                        forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, h - length):
            for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
                for direction in (-1, 1):
                    coords = _chain_coords(r, c, direction, length)
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    if not _validate_chain(inp, coords, fg):
                        continue
                    options.append(coords)
    return options


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


def _h_control_pos(seg: dict, w: int) -> tuple[int, int]:
    return (int(seg["row"]), int(w - 1 if int(seg["c0"]) == 0 else 0))


def _v_control_pos(seg: dict, h: int) -> tuple[int, int]:
    return (int(h - 1 if int(seg["r0"]) == 0 else 0), int(seg["col"]))


def _h_gap_options(seg: dict, out: np.ndarray, bg: int) -> list[int]:
    row, c0, c1 = int(seg["row"]), int(seg["c0"]), int(seg["c1"])
    ctrl_r, ctrl_c = _h_control_pos(seg, out.shape[1])
    if int(out[ctrl_r, ctrl_c]) != int(bg):
        return []
    blocked = seg["blocked"]
    options: list[int] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE):
        if any(g - GAP_BLOCKED_LO <= int(b) <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        if np.all(out[row, g : g + GAP_SIZE] != bg):
            options.append(int(g))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, bg: int) -> list[int]:
    col, r0, r1 = int(seg["col"]), int(seg["r0"]), int(seg["r1"])
    ctrl_r, ctrl_c = _v_control_pos(seg, out.shape[0])
    if int(out[ctrl_r, ctrl_c]) != int(bg):
        return []
    blocked = seg["blocked"]
    options: list[int] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE):
        if any(g - GAP_BLOCKED_LO <= int(b) <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        if np.all(out[g : g + GAP_SIZE, col] != bg):
            options.append(int(g))
    return options


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
    ctrl = rng.choice([c for c in range(N_COLORS) if c not in (bg, fg)])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()

    h_ready = [(seg, _h_gap_options(seg, out, bg)) for seg in h_segments]
    h_ready = [(seg, opts) for seg, opts in h_ready if opts]
    v_ready = [(seg, _v_gap_options(seg, out, bg)) for seg in v_segments]
    v_ready = [(seg, opts) for seg, opts in v_ready if opts]
    if not h_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    used_ctrl: set[tuple[int, int]] = set()
    gap_cells: list[tuple[int, int]] = []

    h_seg, h_opts = h_ready[0]
    h_gap = int(rng.choice(h_opts))
    h_row = int(h_seg["row"])
    inp[h_row, h_gap : h_gap + GAP_SIZE] = int(bg)
    gap_cells.extend((int(h_row), int(c)) for c in range(h_gap, h_gap + GAP_SIZE))
    h_ctrl = _h_control_pos(h_seg, w)
    if h_ctrl in used_ctrl:
        return None
    inp[h_ctrl] = int(ctrl)
    used_ctrl.add(h_ctrl)

    for v_seg, v_opts in v_ready[:v_gap_count]:
        v_gap = int(rng.choice(v_opts))
        v_col = int(v_seg["col"])
        inp[v_gap : v_gap + GAP_SIZE, v_col] = int(bg)
        gap_cells.extend((int(r), int(v_col)) for r in range(v_gap, v_gap + GAP_SIZE))
        v_ctrl = _v_control_pos(v_seg, h)
        if v_ctrl in used_ctrl:
            return None
        inp[v_ctrl] = int(ctrl)
        used_ctrl.add(v_ctrl)

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, [tuple(cell) for cell in np.argwhere(inp == fg)])
    _reserve_neighborhood(forbidden, list(used_ctrl))
    _reserve_neighborhood(forbidden, gap_cells)
    distractors = _distractor_options(inp, bg, fg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any(cell in forbidden for cell in coords):
            continue
        for r, c in coords:
            inp[r, c] = int(fg)
        _reserve_neighborhood(forbidden, coords)
        placed += 1

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7b80bb43 D2 variant")
