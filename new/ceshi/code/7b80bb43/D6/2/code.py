# Variant of 7b80bb43
# Dimension: D6 — 干扰对角链的角色 (transform)
# Change: 0-2条独立干扰对角链放置在空白背景区域，不与任何线段相邻；输出中被删除（及其叶子像素被修剪）... → 干扰链保留在输出中
#
# Design analysis for the D6 variant
# ----------------------------------
# variant_spec:
#   concrete_behavior:
#     - 输入仍然包含一张单色轴对齐线段网络，以及2-3条贴着3像素缺口的2像素对角提示链。
#     - 另外再放置1-2条独立干扰对角链；它们位于纯背景区，只与自身对角相连，不与任何直线端点、
#       gap 提示链或其 Chebyshev-1 邻域接触。
#     - 输出恢复直线网络时，只删除“与直线网络对角相邻”的提示链；独立干扰链原样保留。
#   flow_trace:
#     - 原逻辑：只要像素“无正交邻居且有对角邻居”，就被归入 diag 并删除；因此 gap 提示链和独立干扰链
#       都会一起消失。
#     - 新逻辑：先找出所有 diagonal-only 像素，再按8连通分量分组。若该分量中至少有一个像素与某个
#       非对角 fg 像素形成对角接触，则它是“提示链”，会被删除并触发补线；否则它是“独立干扰链”，
#       在输出中保留。
#     - 小例子：一条水平线被挖掉3格，缺口右端旁边挂着2像素对角链；远处另有一条3像素对角链。
#       旧规则会删掉两条链；新规则只删靠近缺口的那条，并把远处那条保留到输出。
#   complexity_calibration:
#     - 不引入新的颜色、形状类别或多步变换；只增加“是否贴近结构线段”的二分类。
#     - 为了让2-4个样例足以看出差异，生成器默认要求每个成功样例至少有1条独立干扰链。
#     - 仍然只保留少量档位：gap 仍是2-3个，干扰链仍是1-2条，长度仍是2或3，避免组合爆炸。
#   implicit_assumptions_review:
#     - 原 `diag[...] = True` 依赖“所有 diagonal-only 像素都该删除”；变体下不成立，因此必须改成
#       “先找候选，再按是否接触直线根像素分类”。
#     - 原 `near` 基于全部 diag 计算；若继续这样做，会把独立干扰链周围的保留像素也纳入修剪区，
#       导致本应保留的干扰链被误删，所以 `near` 只能来自 actionable 提示链。
#     - 原 gap fill 读取 `mask & (~diag)` 来忽略所有对角链。这个前提仍成立：即使独立干扰链保留在输出，
#       它们也不是直线结构的一部分，所以补线阶段仍应忽略所有 diagonal-only 像素。
#     - 原 `MAXG` 由所有 diag 分量跨度决定；若把独立干扰链也算进去，会把远处3像素链错误当成更大的
#       gap 提示。变体下它只能由 actionable 提示链决定。
#   placement_legality:
#     - gap 提示链仍要求：自身在 bg 上、无正交 fg 邻居、至少有一个对角 fg 邻居、长度固定2。
#     - 独立干扰链要求：位于背景区，和任意现有 fg（包括线段、gap 提示链、其他干扰链）保持 Chebyshev
#       距离至少2，这样它不会意外获得“root”或被 prune 机制波及。
#     - 输出先构造干净网络，再把独立干扰链写入 output；输入则在此基础上再加入缺口和提示链。
#   visual_observability:
#     - 人类可直接看到：有些对角链消失并补上线，有些远处对角链在输出里完全没变。
#     - 关键对比是“靠近线段的删，远离线段的留”，而不是颜色值或隐式计数。
#   visual_cleanliness:
#     - 通过禁止 distractor 进入任何 fg 的 Chebyshev-1 邻域，避免对角链贴着线段、hint 链或彼此堆叠，
#       保证画面仍是“直线网络 + 少量远处斜链”的清晰布局。
#   color_inference:
#     - 仍然只有 bg/fg 两色，没有新增颜色映射规则，因此不存在颜色不可推断问题。
#   generator_guarantees_for_new_main:
#     - `main()` 假设独立干扰链对任一非对角 fg 分量都没有 diagonal root；生成器通过 `forbidden`
#       邻域预留来保证。
#     - `main()` 假设需要补的 gap 附近至少有一条 actionable 提示链；生成器沿用 output-first +
#       oracle 验证来保证。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, draw_line, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Diagonal-only fg components that touch the line network diagonally are actionable hint chains:
      delete them, fill the nearby short H/V gap, and prune local diagonal leaf artifacts.
    - Diagonal-only fg components with no such contact are independent distractors:
      keep them unchanged in the output.
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
        return int(d)

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[r - 1, c - 1]):
            return True
        if r > 0 and c + 1 < W and bool(m[r - 1, c + 1]):
            return True
        if r + 1 < H and c > 0 and bool(m[r + 1, c - 1]):
            return True
        if r + 1 < H and c + 1 < W and bool(m[r + 1, c + 1]):
            return True
        return False

    def diagonal_roots(diag_mask: np.ndarray, coords: np.ndarray) -> set[tuple[int, int]]:
        roots: set[tuple[int, int]] = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(diag_mask[nr, nc]):
                    roots.add((nr, nc))
        return roots

    # All diagonal-only pixels are still "non-structural", but only some are actionable.
    diag_candidate = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag_candidate[r, c] = True

    actionable = np.zeros((H, W), dtype=bool)
    for coords in components_mask(diag_candidate, connectivity=8):
        if coords.size == 0:
            continue
        roots = diagonal_roots(diag_candidate, coords)
        if roots:
            rows, cols = coords[:, 0], coords[:, 1]
            actionable[rows, cols] = True

    # Neighborhood used by the post-fill leaf-pruning step must come only from actionable hints.
    near = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not bool(actionable[r, c]):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        near[rr, cc] = True

    # Delete only actionable hint-chain pixels. Independent distractors remain in place.
    out = g.copy()
    out[actionable] = bg

    # Gap size is inferred only from actionable hint chains; preserved distractors are irrelevant.
    MAXG = 3
    if bool(actionable.any()):
        for coords in components_mask(actionable, connectivity=8):
            if coords.size == 0:
                continue
            roots = diagonal_roots(diag_candidate, coords)
            assert len(coords) + len(roots) <= 3, (
                f"actionable chain too large ({len(coords)} diag + {len(roots)} root)"
            )
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            MAXG = max(int(MAXG), int(r1 - r0 + 1), int(c1 - c0 + 1))

    # Structural rows/cols are derived from the original line network, ignoring all diagonal-only pixels.
    structural = mask & (~diag_candidate)

    for r in range(H):
        cols = np.where(structural[r, :])[0]
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
            if gap1 < gap0 or int(gap1 - gap0 + 1) > int(MAXG):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(H - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(W - 1), int(gap1 + 1))
            if bool(actionable[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[r, gap0 : gap1 + 1] = fg

    for c in range(W):
        rows = np.where(structural[:, c])[0]
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
            if gap1 < gap0 or int(gap1 - gap0 + 1) > int(MAXG):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(H - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(W - 1), int(c + 1))
            if bool(actionable[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                out[gap0 : gap1 + 1, c] = fg

    # Prune only around actionable hint neighborhoods so preserved distractors are untouched.
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


# --------------- generator (D6 variant) ---------------

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

GAP_SIZE = 3
CHAIN_LENGTH = 2
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2

DISTRACTOR_LENGTHS = (3, 2)
DISTRACTOR_MARGIN = 1
DISTRACTOR_COUNT_CHOICES = (1, 2)

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


def _place_t_component(
    out: np.ndarray,
    fg: int,
    slot: str,
    row: int,
    branch_col: int,
    end_col: int,
) -> tuple[dict, dict]:
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


def _h_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, str]]:
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


def _v_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
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


def _reserve_neighborhood(cells: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((r + dr, c + dc))


def _distractor_options(
    inp: np.ndarray,
    bg: int,
    forbidden: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
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
    distractor_count = rng.choice(DISTRACTOR_COUNT_CHOICES)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, out, fg, bg, occupied)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, out, fg, bg, occupied)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, out, fg, bg, occupied) or h_initial
    gap_start, anchor = rng.choice(h_opts)
    row = h_seg["row"]
    chain = _chain_coords(row, gap_start, anchor)
    inp[row, gap_start : gap_start + GAP_SIZE] = bg
    for r, c in chain:
        inp[r, c] = fg
    occupied.update(chain)

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, bg, occupied) or v_initial
        gap_start, anchor, dc = rng.choice(v_opts)
        col = v_seg["col"]
        chain = _v_chain_coords(col, gap_start, anchor, dc)
        inp[gap_start : gap_start + GAP_SIZE, col] = bg
        for r, c in chain:
            inp[r, c] = fg
        occupied.update(chain)

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
            out[r, c] = fg
        _reserve_neighborhood(forbidden, coords)
        placed += 1
    if placed != distractor_count:
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D6/2")
