# Combo variant of 7b80bb43 (index 17)
# combo_spec: D4/2+D10/1
# Name: 仅垂直单像素间隙

from __future__ import annotations

import random

import numpy as np

from arc_utils import draw_line, find_components, get_bg


# Design analysis for combo D4/2 + D10/1
# a) 具体化与行为追踪
#    - 原题的真实提示是 2 像素对角链，对应 3 像素 gap，solver 用“删掉所有 diagonal-only 像素，
#      然后在行/列上补附近的小 gap”的弱规则恢复线段。
#    - 本融合版把真实 gap 收缩到“仅垂直方向、且恰好 1 像素”。长度信息不再重要，方向信息变成核心。
#    - 一个具体局部例子：
#        clean output: 列 c 上有 fg 于 (r-1,c),(r,c),(r+1,c)
#        原题 input:   常见做法会删去 (r..r+2,c) 三格，并放两格对角链
#        融合 input:   只删去单格 (r,c)，再在 (r,c+1) 或 (r,c-1) 放 1 个提示像素
#      该提示像素与上下两个根像素仅做对角接触，不与线段 4 连通；它形成一个非常稳定的 3x2/3x2 局部图样：
#        fg .    或    . fg
#        .  x          x .
#        fg .          . fg
#      其中 x 是待删除提示，真正要补的是它内侧那一列的中心单格。
#    - 与原题差异：不再扫描水平修复；也不再从链长推 gap 长度，而是从“单点夹在上下两个根像素的侧边”
#      这个局部几何直接推出“补 1 格垂直 gap”。
# b) 复杂度校准
#    - 若只保留单点提示而完全没有干扰，会过于简单；因此仍保留 0-2 条开放背景里的 2-3 像素对角干扰链。
#    - 真实提示固定为 2-3 个，且都作用在不同垂直线段上。这样每题都会重复呈现同一种局部模式，足够让人类
#      在少量样例里看出“只有竖线中断，且只补 1 格”的规律。
#    - 干扰链长度故意比真实提示长，视觉上与单点提示有明显差异，避免把“单点=真实、长链=噪声”做成隐性规则。
#      solver 仍按拓扑处理：所有 diagonal-only 像素都删除，但只有满足双根模式的单点才触发补线。
# c) 隐式假设审查
#    - 原 `MAXG = 3` 的全局阈值假设失效：这里 gap 大小不需要推断，固定为 1，且只允许垂直修复。
#    - 原水平补线循环依赖“输入里至少可能有水平 gap”；融合后不成立，必须删除该分支，否则会把无关小缺口也补上。
#    - 原 `components_mask(diag, connectivity=8)` / 链跨度决定 gap 长度 的假设失效：真实提示只剩单点，8 连通分量
#      不再承载数值信息。
#    - 原“附近有任意 diag 就可补一个短 gap”的条件过弱：长干扰链若偶然靠近某列，会导致误补。新 solver 改为：
#      只有当某个被删的 diagonal-only 单点，其四个对角位置中恰好存在“同一列、上下相隔 2 的两个结构根像素”
#      时，才补该列中央那 1 格。
# d) 放置合法性
#    - generator 仍先构造干净输出网络，再回写 corruption。
#    - 真实 gap 只切在垂直线段内部，要求上下剩余长度都 >= 2，并避开 T 结点附近，保证局部模式清晰。
#    - 真实单点提示必须落在背景上，且没有任何正交 fg 邻居；它在“删掉 gap 后”的网格里，其结构性对角邻居
#      必须恰好是目标列上的上下两个根像素。
#    - 干扰链与所有既有前景及真实提示保留 Chebyshev-1 间隔，避免视觉粘连，也保证 solver 的新局部假设成立。
# e) 视觉可观察性
#    - 题面变化会体现在：输入里看不到任何水平断口；所有输出新增像素都位于垂直线段中间；提示缩成单点。
#    - 这种“只修竖线 + 单点贴边提示”的变化，在 2-4 组样例中肉眼可见，不依赖颜色编号或排序等不可视因素。
# f) 可视化整洁度
#    - 真实提示只占 1 格，干扰链又被放到开放背景，整体比原题更干净。
#    - 同一区域不会同时叠加水平 gap、垂直 gap、长链、交叉结构等多层因素；T 结点附近也禁止切 gap，避免难读。
# g) 变化可推断性
#    - 本融合版没有新增颜色映射；所有推理线索都来自几何关系。
#    - “单点在竖线侧边、上下对角各有一个根像素”这一模式会重复出现 2-3 次，足以支撑归纳。
# 新 solver 的输入假设，以及 generator 的对应保证
#    - 假设 1：所有需要修复的真实提示都是 diagonal-only 单点。generator 只用这一种真实提示形状。
#    - 假设 2：真实提示删掉后，目标 gap 正好位于其内侧 1 列，且该 gap 上下各有一个结构根像素。generator 用
#      `_validate_single_hint` 显式检查这一点。
#    - 假设 3：开放背景干扰链不会在删链后伪造出上述双根模式。generator 对干扰链使用背景 moat 保证这一点。


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
SIMPLE_V_MARGIN_NEAR = 7
SIMPLE_V_MARGIN_FAR = 3

GAP_SEG_INSET = 2
GAP_BLOCKED_RADIUS = 2
REAL_GAP_COUNTS = (2, 3)

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


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(color):
            return False
    return True


def _reserve_neighborhood(cells: set[tuple[int, int]], coords) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _hint_coord(col: int, gap_row: int, dc: int) -> tuple[int, int]:
    return int(gap_row), int(col + dc)


def _structural_diag_neighbors(mask: np.ndarray, r: int, c: int) -> set[tuple[int, int]]:
    h, w = mask.shape
    out: set[tuple[int, int]] = set()
    for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            out.add((int(rr), int(cc)))
    return out


def _validate_single_hint(
    grid: np.ndarray,
    bg: int,
    fg: int,
    hint: tuple[int, int],
    gap_col: int,
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    r, c = hint
    if not (0 < r < h - 1 and 0 <= c < w):
        return False
    if (r, c) in occupied or int(grid[r, c]) != int(bg):
        return False

    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg):
            return False

    expected = {(int(r - 1), int(gap_col)), (int(r + 1), int(gap_col))}
    actual = _structural_diag_neighbors(grid == int(fg), r, c)
    return actual == expected


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


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    col, r0, r1 = int(seg["col"]), int(seg["r0"]), int(seg["r1"])
    blocked = {int(x) for x in seg["blocked"]}
    options: list[tuple[int, int]] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SEG_INSET + 1):
        if any(abs(g - b) <= GAP_BLOCKED_RADIUS for b in blocked):
            continue
        temp = out.copy()
        temp[g, col] = int(bg)
        for dc in (-1, 1):
            hint = _hint_coord(col, g, dc)
            if _validate_single_hint(temp, bg, fg, hint, col, occupied):
                options.append((int(g), int(dc)))
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

    h_simple = int(cfg["h"] - cfg["junctions"])
    v_simple = int(cfg["v"] - cfg["junctions"])
    t_slots = rng.sample(list(T_SLOTS), int(cfg["junctions"]))

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
    if len(comps) != int(cfg["segments"] - cfg["junctions"]):
        return None
    return out, h_segments, v_segments


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
    return int(d)


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
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
    for r in range(h):
        for c in range(w):
            if not bool(diag[r, c]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w:
                        near[rr, cc] = True
    return near


def _detect_vertical_targets(base: np.ndarray, diag: np.ndarray) -> set[tuple[int, int]]:
    h, w = base.shape
    targets: set[tuple[int, int]] = set()
    for r, c in np.argwhere(diag):
        r = int(r)
        c = int(c)
        for dc in (-1, 1):
            gap_c = int(c - dc)
            if not (0 < r < h - 1 and 0 <= gap_c < w):
                continue
            if bool(base[r, gap_c]):
                continue
            if not (bool(base[r - 1, gap_c]) and bool(base[r + 1, gap_c])):
                continue
            expected = {(int(r - 1), int(gap_c)), (int(r + 1), int(gap_c))}
            if _structural_diag_neighbors(base, r, c) == expected:
                targets.add((int(r), int(gap_c)))
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete detached diagonal hints and restore only the tiny vertical breaks they encode.

    Input structure:
    - One foreground color forms an axis-aligned line network on a background color.
    - Horizontal segments are intact; every real break is a single missing pixel on a vertical segment.
    - Real hints are diagonal-only single pixels placed immediately to the left or right of that missing spot.
    - Longer diagonal-only distractor chains may also appear in empty background.

    Rule:
    - Remove every diagonal-only foreground pixel.
    - Refill a background cell only when a removed hint had exactly two structural diagonal neighbors:
      one above-left/above-right and one below-left/below-right, both belonging to the same column.
    - Prune any leftover diagonal-adjacent leaf pixels near the deleted hint region.

    Key insight:
    - In this fused variant the hint no longer carries gap length; its precise local geometry encodes
      a fixed one-cell vertical repair, while horizontal repair is never allowed.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors
    fg = int(next(iter(colors)))

    mask = g == int(fg)
    diag = _diag_mask(mask)
    near = _near_mask(diag)

    out = g.copy()
    out[diag] = int(bg)
    base = out == int(fg)

    for r, c in _detect_vertical_targets(base, diag):
        out[int(r), int(c)] = int(fg)

    m = out == int(fg)
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        h, w = m.shape
        for r in range(h):
            for c in range(w):
                if not bool(m[r, c]) or not bool(near[r, c]):
                    continue
                if _deg4(m, r, c) != 1:
                    continue
                if not _has_diag(m, r, c):
                    continue
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[r, c] = False

    out2 = np.full_like(g, int(bg))
    out2[m] = int(fg)
    return out2


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, _h_segments, v_segments = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()

    v_ready = [(seg, _v_gap_options(seg, out, fg, bg, occupied)) for seg in v_segments]
    v_ready = [(seg, opts) for seg, opts in v_ready if opts]
    if len(v_ready) < 2:
        return None

    gap_count = rng.choice(REAL_GAP_COUNTS)
    if len(v_ready) < gap_count:
        gap_count = 2
    if len(v_ready) < gap_count:
        return None

    rng.shuffle(v_ready)
    chosen = v_ready[:gap_count]
    for seg, initial_opts in chosen:
        opts = _v_gap_options(seg, out, fg, bg, occupied) or initial_opts
        if not opts:
            return None
        gap_row, dc = rng.choice(opts)
        col = int(seg["col"])
        hint = _hint_coord(col, gap_row, dc)
        inp[gap_row, col] = int(bg)
        inp[hint] = int(fg)
        occupied.add(hint)

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
            inp[r, c] = int(fg)
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo D4/2+D10/1")
