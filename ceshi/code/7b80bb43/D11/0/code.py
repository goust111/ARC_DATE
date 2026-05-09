# Variant of 7b80bb43
# Dimension: D11 — 颜色系统 (meta)
# Change: 单一前景色+单一背景色，共2种颜色；提示像素与线段同色... → 多色线段网络
#
# Design analysis
# ---
# variant_spec:
#   core_change:
#     original: 所有轴对齐线段、断裂提示链、干扰对角链都共用一个前景色
#     new: 每个4连通线段网络分量使用不同前景色；该分量上的断裂提示链也使用同色
#   concrete_visual:
#     output:
#       - 仍然是 3-5 个彼此分离的轴对齐线段网络，含 2-3 个 T 形连接
#       - 但每个连通分量整块染成不同颜色，因此同一网格里会同时出现多条不同颜色的线网
#     input:
#       - 在若干水平/垂直线段上切出 3 像素缺口
#       - 缺口旁边放 2 像素的45度同色提示链
#       - 另有 0-2 条开放背景中的同色干扰对角链
#     output_action:
#       - 删除所有“仅靠对角相连”的提示/干扰像素
#       - 只在与该提示链同色的直线中补回 3 像素缺口
#       - 因此补线颜色必须和所属分量匹配，不能再用单一 fg 覆盖
#   behavior_trace:
#     original_local_example:
#       - 一条红色? 不存在；原题里只有一种 fg，所以 gap 一旦满足条件，直接写回 fg
#     new_local_example:
#       - 蓝色组件的水平线被切断，旁边有蓝色对角提示链；红色组件在别处也有缺口
#       - solver 必须先按颜色分开看：蓝链只触发蓝线补全，红链只触发红线补全
#       - 若仍沿用原代码的 `fg = next(iter(colors))`，就只会恢复一种颜色，其他颜色会丢失
#   complexity_calibration:
#     - 不改拓扑生成方式，保留原题的线网密度、gap 数量、干扰链规则
#     - 颜色变化只作用于“哪个分量用什么色、补线写回什么色”，避免引入额外几何自由度
#     - 为防止多色退化回“只是装饰”，生成器强制至少两个不同颜色都参与 gap 修复
#   implicit_assumption_audit:
#     old_fg_assumptions:
#       - `fg = next(iter(colors))`: 假设非背景色只有一种；在变体中会漏掉其他颜色
#       - `mask = g == fg`: 假设所有需要处理的像素共用一个布尔掩码
#       - `out[...] = fg`: 假设补线时写什么颜色都一样；变体中会错误覆盖成单色
#       - `m = out == fg` / `out2[m] = fg`: 假设最终保留下来的前景也只有一种颜色
#     new_solver_fix:
#       - 将原始单色逻辑封装成“单色子求解器”
#       - 对每个前景色独立执行：识别同色对角链、计算同色 gap、修剪同色叶子
#       - 最后把各颜色结果合并回同一个输出网格
#   placement_legality:
#     - 先按原题方式生成单色 clean output，确保几何结构合法
#     - 再按4连通分量逐块重着色；因为整块重着色，不会破坏 T 连接共享像素
#     - 每个分量使用唯一颜色，避免“两个不相连但同色的分量”在 solver 中产生伪 gap / 伪提示交互
#     - 提示链验证仍要求：链像素位于背景、无正交前景邻居、至少有同色对角邻居
#   visual_observability:
#     - 输入输出对里能直接看到：蓝线缺口补回蓝色、红线缺口补回红色、其他颜色同理
#     - 颜色不是按数值大小排序，而是跟随可见的局部结构分量
#   cleanliness:
#     - 保持原始间距约束；干扰链仍放在开放背景并避开所有前景的 Chebyshev-1 邻域
#     - 因为每个分量唯一着色，视觉上不会出现同色远距对象误被看成一组
#   new_solver_assumptions_checked:
#     - 每种颜色对应至多一个线网分量；generator 通过“分量唯一色”保证
#     - 同色提示链只服务同色线段；generator 保证链颜色继承自被切断分量
#     - 多颜色结果合并时不会重叠；generator 保证对象原本不重叠

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


def _solve_single_color(grid: np.ndarray, bg: int, color: int) -> np.ndarray:
    """Original 7b80bb43 logic, applied to exactly one foreground color."""
    mask = grid == int(color)
    H, W = grid.shape

    def deg4(m: np.ndarray, r: int, c: int) -> int:
        d = 0
        if r > 0 and bool(m[int(r - 1), int(c)]):
            d += 1
        if r + 1 < H and bool(m[int(r + 1), int(c)]):
            d += 1
        if c > 0 and bool(m[int(r), int(c - 1)]):
            d += 1
        if c + 1 < W and bool(m[int(r), int(c + 1)]):
            d += 1
        return int(d)

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[int(r - 1), int(c - 1)]):
            return True
        if r > 0 and c + 1 < W and bool(m[int(r - 1), int(c + 1)]):
            return True
        if r + 1 < H and c > 0 and bool(m[int(r + 1), int(c - 1)]):
            return True
        if r + 1 < H and c + 1 < W and bool(m[int(r + 1), int(c + 1)]):
            return True
        return False

    diag = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[int(r), int(c)]) and deg4(mask, int(r), int(c)) == 0 and has_diag(mask, int(r), int(c)):
                diag[int(r), int(c)] = True

    near = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not bool(diag[int(r), int(c)]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    base = mask & (~diag)
    out = base.copy()

    max_gap = 3
    if bool(diag.any()):
        for coords in components_mask(diag, connectivity=8):
            if coords.size == 0:
                continue
            roots = set()
            for r, c in coords:
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and not bool(diag[rr, cc]):
                        roots.add((rr, cc))
            assert len(coords) + len(roots) <= 3, f"diagonal chain too large ({len(coords)} diag + {len(roots)} root)"
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            max_gap = max(int(max_gap), int(r1 - r0 + 1), int(c1 - c0 + 1))

    for r in range(H):
        cols = np.where(base[int(r), :])[0]
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
            if gap1 < gap0 or int(gap1 - gap0 + 1) > int(max_gap):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(H - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(W - 1), int(gap1 + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[int(r), int(gap0) : int(gap1) + 1] = True

    for c in range(W):
        rows = np.where(base[:, int(c)])[0]
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
            if gap1 < gap0 or int(gap1 - gap0 + 1) > int(max_gap):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(H - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(W - 1), int(c + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[int(gap0) : int(gap1) + 1, int(c)] = True

    m = out.copy()
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not bool(m[int(r), int(c)]) or not bool(near[int(r), int(c)]):
                    continue
                if deg4(m, int(r), int(c)) != 1:
                    continue
                if not has_diag(m, int(r), int(c)):
                    continue
                to_clear.append((int(r), int(c)))
        if to_clear:
            changed = True
            for r, c in to_clear:
                m[int(r), int(c)] = False

    return m


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = sorted(int(x) for x in np.unique(g) if int(x) != int(bg))
    assert colors

    out = np.full(g.shape, int(bg), dtype=int)
    for color in colors:
        solved_mask = _solve_single_color(g, bg, color)
        assert not np.any(solved_mask & (out != int(bg))), "color layers overlap unexpectedly"
        out[solved_mask] = int(color)
    return out


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
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
    bg: int,
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
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != int(bg):
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(color)):
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


def _h_gap_options(seg: dict, out: np.ndarray, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    color = seg["color"]
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
            if not _validate_chain(temp, coords, color, bg, occupied):
                continue
            options.append((g, anchor))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, bg: int, occupied: set[tuple[int, int]]) -> list[tuple[int, str, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    color = seg["color"]
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
                if not _validate_chain(temp, coords, color, bg, occupied):
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


def _build_output(rng: random.Random, h: int, w: int, bg: int, fg: int, cfg: dict) -> tuple[np.ndarray, list[dict], list[dict]] | None:
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


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    labels = np.zeros(mask.shape, dtype=int)
    comps = components_mask(mask, connectivity=4)
    for idx, coords in enumerate(comps, start=1):
        labels[coords[:, 0], coords[:, 1]] = int(idx)
    return labels, comps


def _recolor_output(
    rng: random.Random,
    out: np.ndarray,
    bg: int,
    h_segments: list[dict],
    v_segments: list[dict],
) -> tuple[np.ndarray, dict[int, int]] | None:
    labels, comps = _label_components(out != int(bg))
    if len(comps) < 2:
        return None
    palette = rng.sample([c for c in range(N_COLORS) if c != int(bg)], len(comps))
    comp_colors: dict[int, int] = {}
    recolored = np.full_like(out, int(bg))
    for comp_id, color in enumerate(palette, start=1):
        comp_colors[int(comp_id)] = int(color)
        recolored[labels == int(comp_id)] = int(color)

    for seg in h_segments:
        comp_id = int(labels[int(seg["row"]), int(seg["c0"])])
        if comp_id == 0:
            return None
        seg["component"] = comp_id
        seg["color"] = int(comp_colors[comp_id])
    for seg in v_segments:
        comp_id = int(labels[int(seg["r0"]), int(seg["col"])])
        if comp_id == 0:
            return None
        seg["component"] = comp_id
        seg["color"] = int(comp_colors[comp_id])
    return recolored, comp_colors


def _choose_gap_segments(
    rng: random.Random,
    h_ready: list[tuple[dict, list[tuple[int, str]]]],
    v_ready: list[tuple[dict, list[tuple[int, str, int]]]],
    v_gap_count: int,
) -> tuple[int, tuple[int, ...]] | None:
    h_indices = list(range(len(h_ready)))
    v_combos = list(itertools.combinations(range(len(v_ready)), v_gap_count))
    rng.shuffle(h_indices)
    rng.shuffle(v_combos)
    for h_idx in h_indices:
        for combo in v_combos:
            used_colors = {int(h_ready[h_idx][0]["color"])}
            used_colors.update(int(v_ready[idx][0]["color"]) for idx in combo)
            if len(used_colors) >= 2:
                return h_idx, combo
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(0, 2)
    line_token = 1 if bg != 1 else 2

    built = _build_output(rng, h, w, bg, line_token, cfg)
    if built is None:
        return None
    mono_out, h_segments, v_segments = built

    recolored = _recolor_output(rng, mono_out, bg, h_segments, v_segments)
    if recolored is None:
        return None
    out, comp_colors = recolored

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, out, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, out, bg, occupied)] if opts]
    if not h_ready:
        return None
    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None

    chosen = _choose_gap_segments(rng, h_ready, v_ready, v_gap_count)
    if chosen is None:
        return None
    h_idx, v_indices = chosen

    used_gap_colors: set[int] = set()

    h_seg, h_initial = h_ready[h_idx]
    h_opts = _h_gap_options(h_seg, out, bg, occupied) or h_initial
    if not h_opts:
        return None
    gap_start, anchor = rng.choice(h_opts)
    row = int(h_seg["row"])
    chain = _chain_coords(row, int(gap_start), anchor)
    inp[row, int(gap_start) : int(gap_start) + GAP_SIZE] = int(bg)
    for r, c in chain:
        inp[int(r), int(c)] = int(h_seg["color"])
    occupied.update(chain)
    used_gap_colors.add(int(h_seg["color"]))

    for v_idx in v_indices:
        v_seg, v_initial = v_ready[int(v_idx)]
        v_opts = _v_gap_options(v_seg, out, bg, occupied) or v_initial
        if not v_opts:
            return None
        gap_start, anchor, dc = rng.choice(v_opts)
        col = int(v_seg["col"])
        chain = _v_chain_coords(col, int(gap_start), anchor, int(dc))
        inp[int(gap_start) : int(gap_start) + GAP_SIZE, col] = int(bg)
        for r, c in chain:
            inp[int(r), int(c)] = int(v_seg["color"])
        occupied.update(chain)
        used_gap_colors.add(int(v_seg["color"]))

    if len(used_gap_colors) < 2:
        return None

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, [tuple(cell) for cell in np.argwhere(inp != int(bg))])
    distractors = _distractor_options(inp, bg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    palette = list(comp_colors.values())
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any(cell in forbidden for cell in coords):
            continue
        color = int(rng.choice(palette))
        for r, c in coords:
            inp[int(r), int(c)] = color
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D11 variant")
