# Variant of 7b80bb43
# Dimension: D1 — 间隙填充变换方式 (transform)
# Change: 删除对角提示像素后，在提示相邻处填补水平/垂直直线上的3像素间隙，恢复断裂的线段... → 间隙处绘制垂直交叉线段
#
# Design analysis
# ===============
# variant_spec:
#   preserved_structure:
#     - 仍然只有 1 个前景色，主体仍然是轴对齐线段网络
#     - 输入中的提示仍然是 45 度对角链；solver 仍先删掉所有 diagonal-only 像素
#     - 干扰链仍可存在，且仍应在修枝阶段被清掉
#   changed_structure:
#     - 原题输出只把 3 像素 gap 补回直线
#     - 本变体把每个真实 gap 变成一个新的十字节点：
#       * 若 gap 在水平线段上，输出先补齐这 3 个水平像素，再以 gap 中心为轴补一条长度 3 的竖线
#       * 若 gap 在垂直线段上，输出先补齐这 3 个垂直像素，再以 gap 中心为轴补一条长度 3 的横线
#   concrete_trace:
#     - 原逻辑：row=r 上的 "...XX...XXX..." + 邻近对角链 -> 删除链后把中间 3 格补成 "XXXXXXXX"
#     - 新逻辑：同一个 gap 删除链后不仅补齐水平 3 格，还在中心格上下各长 1 格，得到一个 '+' 形交叉
#     - 差异：输出拓扑新增 degree-4 节点；它不是简单去噪，而是利用提示把断口升级为连接器
#   complexity_calibration:
#     - 参考任务 3e6067c3 的核心是“由提示触发新连接结构，而不是只恢复原像素”
#     - 这里保持同一复杂度层级，但把构造限制为固定长度 3 的垂直支线，避免多档位规则导致样例难归纳
#   implicit_assumptions_review:
#     - 原 solver 在 gap 处无条件 `out[...] = fg` 只恢复同向线，这依赖“输出不新增其他拓扑”
#     - 新规则下，gap 处理必须额外记录哪些像素是新生成的交叉臂，否则后续局部修枝会把合法叶子端点误删
#     - 原 pruning 条件 `deg4==1 and has_diag` 对新交叉臂端点同样成立，因此新增 `protected` 掩码是必要的
#     - 原 generator 只需保证 gap 邻域能放 diagonal chain；新 generator 还必须保证 gap 中心的垂直方向有净空，
#       且支线两端不会碰到其他线段，否则“长度等于 gap 宽度”的视觉规则会被破坏
#   placement_legality:
#     - 只在可放长度 3 交叉臂的位置开 gap：支线两端必须在界内、为空白，并且支线末端的其余正交方向也为空白
#     - 每条真实 gap 仍避开 T-junction 的 blocked 位置，并且不同 gap 的 chain / cross neighborhood 彼此避让
#     - 输入通过 output-first 反推：先造最终带交叉节点的输出，再删除主线 gap 和交叉臂，最后放回对角提示链
#   visual_observability:
#     - 每个样本强制包含 1 个水平 gap 和 1-2 个垂直 gap，因此训练样例里能同时看到“水平断口长竖臂”和“垂直断口长横臂”
#     - 与原题相比，输出会稳定出现新的十字节点；旧 solver 在同一输入上只会补直线，因此两者可直接区分
#   cleanliness:
#     - 交叉臂长度固定为 3，且 generator 要求支线周围净空，避免和既有线段叠出杂乱粗块
#     - 颜色不变，推理只依赖可见的几何结构，不依赖颜色编号

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
for extra in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    extra_s = str(extra)
    if extra_s not in sys.path:
        sys.path.append(extra_s)

from arc_utils import components_mask, draw_line, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise and convert hinted 3-pixel gaps into cross nodes.

    Input structure:
    - bg = background; one foreground color fg forming axis-aligned lines.
    - Some fg pixels form 45° diagonal chains (8-connected but 4-disconnected from lines).

    Rule:
    - Delete all diagonal-only fg pixels (those with no orthogonal fg neighbor, only diagonal ones).
    - Find short horizontal/vertical gaps next to deleted hints.
    - Restore the broken line and add a perpendicular segment of the same length through the gap center.
    - Prune only leftover diagonal-adjacent leaf pixels that are not part of a generated cross.
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

    def mark_vertical_cross(out: np.ndarray, protected: np.ndarray, row: int, gap0: int, gap1: int) -> None:
        gap_len = int(gap1 - gap0 + 1)
        center_c = int((gap0 + gap1) // 2)
        start_r = int(row - gap_len // 2)
        out[int(row), int(gap0) : int(gap1) + 1] = int(fg)
        protected[int(row), int(gap0) : int(gap1) + 1] = True
        for rr in range(int(start_r), int(start_r + gap_len)):
            if 0 <= rr < H:
                out[int(rr), int(center_c)] = int(fg)
                protected[int(rr), int(center_c)] = True

    def mark_horizontal_cross(out: np.ndarray, protected: np.ndarray, col: int, gap0: int, gap1: int) -> None:
        gap_len = int(gap1 - gap0 + 1)
        center_r = int((gap0 + gap1) // 2)
        start_c = int(col - gap_len // 2)
        out[int(gap0) : int(gap1) + 1, int(col)] = int(fg)
        protected[int(gap0) : int(gap1) + 1, int(col)] = True
        for cc in range(int(start_c), int(start_c + gap_len)):
            if 0 <= cc < W:
                out[int(center_r), int(cc)] = int(fg)
                protected[int(center_r), int(cc)] = True

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

    out = g.copy()
    out[diag] = int(bg)
    protected = np.zeros((H, W), dtype=bool)

    MAXG = 3
    if bool(diag.any()):
        for coords in components_mask(diag, connectivity=8):
            if coords.size == 0:
                continue
            roots = set()
            for r, c in coords:
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and not bool(diag[rr, cc]):
                        roots.add((int(rr), int(cc)))
            assert len(coords) + len(roots) <= 3, (
                f"diagonal chain too large ({len(coords)} diag + {len(roots)} root)"
            )
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            MAXG = max(int(MAXG), int(r1 - r0 + 1), int(c1 - c0 + 1))

    for r in range(H):
        cols = np.where(mask[int(r), :] & (~diag[int(r), :]))[0]
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
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(MAXG):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(H - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(W - 1), int(gap1 + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                mark_vertical_cross(out, protected, int(r), int(gap0), int(gap1))

    for c in range(W):
        rows = np.where(mask[:, int(c)] & (~diag[:, int(c)]))[0]
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
            if gap1 < gap0:
                continue
            if int(gap1 - gap0 + 1) > int(MAXG):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(H - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(W - 1), int(c + 1))
            if bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                mark_horizontal_cross(out, protected, int(c), int(gap0), int(gap1))

    m = out == int(fg)
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if not bool(m[int(r), int(c)]) or not bool(near[int(r), int(c)]):
                    continue
                if bool(protected[int(r), int(c)]):
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
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1

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
        return [(int(row + i), int(gap_start - 1 + i)) for i in range(1, CHAIN_LENGTH + 1)]
    return [(int(row + i), int(gap_start + GAP_SIZE - i)) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        return [(int(gap_start + i), int(col + dc * (i + 1))) for i in range(CHAIN_LENGTH)]
    return [(int(gap_start + GAP_SIZE - 1 - i), int(col + dc * (i + 1))) for i in range(CHAIN_LENGTH)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or int(grid[int(r), int(c)]) != int(color):
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
            if 0 <= rr < h and 0 <= cc < w and int(grid[int(rr), int(cc)]) == int(fg):
                return False
        has_diag = False
        for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
            if (rr, cc) in coord_set or (0 <= rr < h and 0 <= cc < w and int(grid[int(rr), int(cc)]) == int(fg)):
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
    h_seg = {"row": int(row), "c0": int(h_c0), "c1": int(h_c1), "blocked": {int(branch_col)}}
    v_seg = {"col": int(branch_col), "r0": int(v_r0), "r1": int(v_r1), "blocked": {int(row)}}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    W = out.shape[1]
    c0 = 0 if side == "left" else W - length
    c1 = length - 1 if side == "left" else W - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": int(row), "c0": int(c0), "c1": int(c1), "blocked": set()}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    H = out.shape[0]
    r0 = 0 if side == "top" else H - length
    r1 = length - 1 if side == "top" else H - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": int(col), "r0": int(r0), "r1": int(r1), "blocked": set()}


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _distractor_options(inp: np.ndarray, bg: int, forbidden: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(DISTRACTOR_MARGIN, h - length):
            for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
                for direction in (-1, 1):
                    coords = [(int(r + i), int(c + direction * i)) for i in range(length)]
                    if not _cells_are_color(inp, coords, bg):
                        continue
                    if any(cell in forbidden for cell in coords):
                        continue
                    options.append(coords)
    return options


def _build_base_output(
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
        end_col = rng.randint(int(min_end), int(max_end))
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
        h_segments.append(_place_simple_h(out, fg, side, int(row), rng.randint(SIMPLE_H_LEN_MIN, int(max_len))))

    top_v = min(2, v_simple)
    top_v_cols = simple_v_cols[:top_v]
    bottom_v_cols = simple_v_cols[top_v:]
    top_limit = min(min(top_rows) - 2 if top_rows else center_start - 2, center_start - 2)
    bottom_start = max(max(bottom_rows) + 2 if bottom_rows else center_end + 2, center_end + 2)
    for col in top_v_cols:
        if top_limit < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "top", int(col), rng.randint(2, int(top_limit))))
    for col in bottom_v_cols:
        max_len = h - bottom_start
        if max_len < 2:
            return None
        v_segments.append(_place_simple_v(out, fg, "bottom", int(col), rng.randint(2, int(max_len))))

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(comps) != int(cfg["segments"] - cfg["junctions"]):
        return None
    return out, h_segments, v_segments


def _h_cross_cells(row: int, gap_start: int) -> list[tuple[int, int]]:
    center_c = int(gap_start + GAP_SIZE // 2)
    start_r = int(row - GAP_SIZE // 2)
    return [(int(start_r + i), int(center_c)) for i in range(GAP_SIZE)]


def _v_cross_cells(col: int, gap_start: int) -> list[tuple[int, int]]:
    center_r = int(gap_start + GAP_SIZE // 2)
    start_c = int(col - GAP_SIZE // 2)
    return [(int(center_r), int(start_c + i)) for i in range(GAP_SIZE)]


def _endpoint_clear(base: np.ndarray, fg: int, cell: tuple[int, int], checks: list[tuple[int, int]]) -> bool:
    h, w = base.shape
    r, c = cell
    for dr, dc in checks:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w and int(base[int(rr), int(cc)]) == int(fg):
            return False
    return True


def _cross_is_clear(base: np.ndarray, fg: int, bg: int, arm_cells: list[tuple[int, int]], center: tuple[int, int], orient: str) -> bool:
    h, w = base.shape
    for r, c in arm_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (int(r), int(c)) != center and int(base[int(r), int(c)]) != int(bg):
            return False

    if orient == "vertical":
        top = arm_cells[0]
        bottom = arm_cells[-1]
        return _endpoint_clear(base, fg, top, [(-1, 0), (0, -1), (0, 1)]) and _endpoint_clear(
            base, fg, bottom, [(1, 0), (0, -1), (0, 1)]
        )

    left = arm_cells[0]
    right = arm_cells[-1]
    return _endpoint_clear(base, fg, left, [(0, -1), (-1, 0), (1, 0)]) and _endpoint_clear(
        base, fg, right, [(0, 1), (-1, 0), (1, 0)]
    )


def _h_cross_options(
    seg: dict,
    base: np.ndarray,
    fg: int,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> list[dict]:
    row, c0, c1 = int(seg["row"]), int(seg["c0"]), int(seg["c1"])
    blocked = seg["blocked"]
    options: list[dict] = []
    for g in range(c0 + GAP_SEG_INSET, c1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = base.copy()
        temp[int(row), int(g) : int(g + GAP_SIZE)] = int(bg)
        arm_cells = _h_cross_cells(int(row), int(g))
        center = (int(row), int(g + GAP_SIZE // 2))
        if not _cross_is_clear(base, fg, bg, arm_cells, center, "vertical"):
            continue
        arm_set = set(arm_cells)
        if arm_set & forbidden:
            continue
        for anchor in ("left", "right"):
            chain = _chain_coords(int(row), int(g), anchor)
            if not _cells_are_color(temp, chain, bg):
                continue
            if not _validate_chain(temp, chain, fg, forbidden):
                continue
            options.append(
                {
                    "kind": "h",
                    "row": int(row),
                    "gap_start": int(g),
                    "chain": chain,
                    "arm": arm_cells,
                }
            )
    return options


def _v_cross_options(
    seg: dict,
    base: np.ndarray,
    fg: int,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> list[dict]:
    col, r0, r1 = int(seg["col"]), int(seg["r0"]), int(seg["r1"])
    blocked = seg["blocked"]
    options: list[dict] = []
    for g in range(r0 + GAP_SEG_INSET, r1 - GAP_SIZE - 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = base.copy()
        temp[int(g) : int(g + GAP_SIZE), int(col)] = int(bg)
        arm_cells = _v_cross_cells(int(col), int(g))
        center = (int(g + GAP_SIZE // 2), int(col))
        if not _cross_is_clear(base, fg, bg, arm_cells, center, "horizontal"):
            continue
        arm_set = set(arm_cells)
        if arm_set & forbidden:
            continue
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                chain = _v_chain_coords(int(col), int(g), anchor, int(dc))
                if not _cells_are_color(temp, chain, bg):
                    continue
                if not _validate_chain(temp, chain, fg, forbidden):
                    continue
                options.append(
                    {
                        "kind": "v",
                        "col": int(col),
                        "gap_start": int(g),
                        "chain": chain,
                        "arm": arm_cells,
                    }
                )
    return options


def _choose_gap(
    rng: random.Random,
    segments: list[dict],
    base: np.ndarray,
    fg: int,
    bg: int,
    forbidden: set[tuple[int, int]],
    kind: str,
) -> tuple[dict, set[tuple[int, int]]] | None:
    segs = segments[:]
    rng.shuffle(segs)
    for seg in segs:
        if kind == "h":
            opts = _h_cross_options(seg, base, fg, bg, forbidden)
        else:
            opts = _v_cross_options(seg, base, fg, bg, forbidden)
        if not opts:
            continue
        choice = rng.choice(opts)
        reserve = set(choice["chain"]) | set(choice["arm"])
        _reserve_neighborhood(reserve, choice["chain"] + choice["arm"])
        return choice, reserve
    return None


def _apply_gap_to_input(inp: np.ndarray, bg: int, fg: int, rec: dict) -> None:
    if rec["kind"] == "h":
        row = int(rec["row"])
        g = int(rec["gap_start"])
        inp[int(row), int(g) : int(g + GAP_SIZE)] = int(bg)
    else:
        col = int(rec["col"])
        g = int(rec["gap_start"])
        inp[int(g) : int(g + GAP_SIZE), int(col)] = int(bg)
    for r, c in rec["arm"]:
        inp[int(r), int(c)] = int(bg)
    for r, c in rec["chain"]:
        inp[int(r), int(c)] = int(fg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(N_COLORS)
    fg = rng.choice([c for c in range(N_COLORS) if c != bg])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if int(cfg["h"]) == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(0, 2)

    built = _build_base_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    base, h_segments, v_segments = built

    if not h_segments or len(v_segments) < gap_count - 1:
        return None

    selected: list[dict] = []
    forbidden: set[tuple[int, int]] = set()

    chosen_h = _choose_gap(rng, h_segments, base, fg, bg, forbidden, "h")
    if chosen_h is None:
        return None
    rec, reserve = chosen_h
    selected.append(rec)
    forbidden |= reserve

    remaining_v = v_segments[:]
    rng.shuffle(remaining_v)
    for _ in range(gap_count - 1):
        chosen_v = _choose_gap(rng, remaining_v, base, fg, bg, forbidden, "v")
        if chosen_v is None:
            return None
        rec, reserve = chosen_v
        selected.append(rec)
        forbidden |= reserve
        remaining_v = [seg for seg in remaining_v if seg is not None and int(seg["col"]) != int(rec["col"])]

    out = base.copy()
    for rec in selected:
        for r, c in rec["arm"]:
            out[int(r), int(c)] = int(fg)

    inp = out.copy()
    for rec in selected:
        _apply_gap_to_input(inp, bg, fg, rec)

    forbidden_distractors: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden_distractors, list(map(tuple, np.argwhere(inp == fg))))
    for rec in selected:
        _reserve_neighborhood(forbidden_distractors, rec["arm"])

    distractors = _distractor_options(inp, bg, forbidden_distractors)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any(cell in forbidden_distractors for cell in coords):
            continue
        for r, c in coords:
            inp[int(r), int(c)] = int(fg)
        _reserve_neighborhood(forbidden_distractors, coords)
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
