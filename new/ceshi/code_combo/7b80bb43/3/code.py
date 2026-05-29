# Combo variant of 7b80bb43 (index 3)
# combo_spec: D1/4+D4/1
# Name: 方向编码间隙
#
# Design analysis
# ===============
# variant_spec:
#   input_structure:
#     - 仍然只有一个前景色 fg 和一个背景色 bg。
#     - 底层结构仍是由水平/竖直线段组成的网络，保留 2-3 个 T 形连接，竖直线段主要承担结构背景作用。
#     - 输入里有 2-3 条真实 45 度对角提示链，以及 0-2 条远离主结构的纯噪声对角链。
#   concretized_behavior_trace:
#     - 取一条水平线，原始完整形态为:
#         .. F F F F F F ..
#     - 若在中间挖出 3 格，且在左端放一条 down-right 的 3 像素链，输入会变成:
#         row r:     .. F . . . F F ..
#         row r+1:      . F
#         row r+2:        . F
#         row r+3:          . F
#     - 原题会把 3 格全补满，输出回到整条直线。
#     - D1/4 单变体会只补右侧两格，保留 gap 左端 1 格为空。
#     - 本融合版继续保留这个“方向决定补哪一侧”的行为，但 gap 不再固定为 3：
#       * 链长 2 -> 总 gap 长 2，只补其中 1 格
#       * 链长 3 -> 总 gap 长 3，只补其中 2 格
#       * down-right 链补右侧，down-left 链补左侧
#   why_this_interpretation:
#     - 若把长度只用于“整段补满多长 gap”，那么同一长度下两种方向会得到完全相同的输出，D1 维度会退化成不可观察的隐变量。
#     - 因此这里采用“长度决定总 gap 长度，方向决定哪一侧被保留为空”的联合规则，确保两个控制变量都真实改变输出。
#   complexity_calibration:
#     - gap_size 只允许 2 和 3 两档，避免自由度过大。
#     - 真实 gap 只放在水平线段上；若同时对竖直 gap 定义“左半/右半”的类比规则，人在 2-4 个样例里不容易稳定归纳，反而降低可见性。
#     - 每题优先同时覆盖两种方向，并尽量同时出现大小 2 和 3，避免样例退化成单一情况。
#   implicit_assumption_audit:
#     - 原 solver 依赖“附近有 diag 就把短 gap 整段补满”；融合后不成立，因为方向和长度都必须被单独解码。
#     - 原 solver 依赖“gap 固定为 3、链固定为 2”；融合后不成立，需要按组件长度逐条恢复。
#     - 原 solver 的“只看全局 MAXG”会抹平链长信息；融合后必须改成“按单条链模板精确匹配”。
#     - D1/4 中“链长固定，输出永远补两格”的假设也不成立；现在补回数量是 `chain_len - 1`。
#   placement_legality:
#     - 真实 gap 只切在线段内部，左右残段都至少保留 2 格。
#     - 真实链长度必须与 gap_size 相等，所有链像素都落在背景上，且不与任一前景像素正交接触。
#     - 多个 gap 允许共存，但其链邻域与行内 gap 带状区域必须分离，避免链之间粘连，也避免输出上的多个缺口视觉重叠。
#   visual_observability:
#     - 输出不再是完全修复的满线，而是保留一个规则性 1 格缺口；这样“方向决定保留哪一边”可直接肉眼看到。
#     - 同时让 gap 有 2/3 两种长度，解题者也能看到“链越长，被挖掉的总缺口越长，但最终只剩 1 格”。
#   cleanliness:
#     - 输出仍只包含轴对齐线段与规则性缺口，没有任何对角像素。
#     - 干扰链继续放在开放背景里，并通过局部修剪被彻底清掉，不和真实提示混在一起。
#   generator_guarantees_for_new_solver:
#     - 每条真实链恰好只有 1 个对角相邻根像素，且与某个水平 gap 模板一一对应。
#     - 根像素另一侧在 gap 外仍有前景，因此 solver 可以验证“这是可恢复的真实 gap，而非噪声链”。
#     - 真实链长度只取 2 或 3；distractor 虽可同长，但不会靠近任何可恢复的水平 gap 模板。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, find_components, get_bg


GAP_SIZE_VALUES = (2, 3)


def _chain_coords(row: int, gap_start: int, gap_size: int, anchor: str) -> list[tuple[int, int]]:
    """Diagonal chain for a horizontal gap; chain length equals gap size."""
    if anchor == "left":
        return [(int(row + i), int(gap_start - 1 + i)) for i in range(1, int(gap_size) + 1)]
    return [(int(row + i), int(gap_start + gap_size - i)) for i in range(1, int(gap_size) + 1)]


def _cells_are_color(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _validate_chain(grid: np.ndarray, coords: list[tuple[int, int]], fg: int, occupied: set[tuple[int, int]]) -> bool:
    """Check that chain coords sit on bg, have no ortho fg neighbor, and have diagonal fg."""
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


def _reserve_neighborhood(cells: set[tuple[int, int]], coords) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _output_hole_col(gap_start: int, gap_size: int, anchor: str) -> int:
    return int(gap_start) if anchor == "left" else int(gap_start + gap_size - 1)


def _gap_forbidden_band(row: int, gap_start: int, gap_size: int) -> set[tuple[int, int]]:
    return {(int(row), int(c)) for c in range(int(gap_start - 1), int(gap_start + gap_size + 1))}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal staircase noise and rebuild direction-coded partial gaps.

    Input structure:
    - bg is the background and a single fg color forms an axis-aligned line network.
    - Some fg pixels form short 45-degree chains that are 8-connected internally but 4-disconnected from the lines.
    - Real chains sit beside horizontal gaps only; chain length is 2 or 3 and equals the total gap length.

    Rule:
    - Delete every diagonal-only fg pixel component.
    - For each deleted chain with exactly one diagonal root on a horizontal line, infer the missing gap length from the chain length.
    - If the chain goes down-right, refill all but the leftmost gap cell; if it goes down-left, refill all but the rightmost gap cell.
    - Prune any remaining diagonal-adjacent leaf pixels near former chain locations to remove distractor residue.

    Key insight:
    - A real hint is not just “some diagonal nearby”: its exact template matters twice.
    - The chain length tells how many total cells were removed, while the chain direction tells which side of that run should stay open by one cell.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(int(bg))
    assert colors
    fg = int(next(iter(colors)))

    mask = g == int(fg)
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
        return int(d)

    def has_diag(m: np.ndarray, r: int, c: int) -> bool:
        if r > 0 and c > 0 and bool(m[r - 1, c - 1]):
            return True
        if r > 0 and c + 1 < w and bool(m[r - 1, c + 1]):
            return True
        if r + 1 < h and c > 0 and bool(m[r + 1, c - 1]):
            return True
        if r + 1 < h and c + 1 < w and bool(m[r + 1, c + 1]):
            return True
        return False

    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) and deg4(mask, r, c) == 0 and has_diag(mask, r, c):
                diag[r, c] = True

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

    base = mask & (~diag)
    out = np.full((h, w), int(bg), dtype=int)
    out[base] = int(fg)

    for coords in components_mask(diag, connectivity=8):
        if coords.size == 0:
            continue
        pts = {(int(r), int(c)) for r, c in coords}
        chain_len = int(len(pts))
        if chain_len not in GAP_SIZE_VALUES:
            continue

        roots: set[tuple[int, int]] = set()
        for r, c in pts:
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and bool(base[rr, cc]):
                    roots.add((rr, cc))
        if len(roots) != 1:
            continue

        root_r, root_c = next(iter(roots))

        left_gap_start = int(root_c + 1)
        left_chain = set(_chain_coords(root_r, left_gap_start, chain_len, "left"))
        if pts == left_chain:
            gap_cols = [int(root_c + i) for i in range(1, chain_len + 1)]
            fill_cols = gap_cols[1:]
            far_col = int(root_c + chain_len + 1)
            if far_col < w and bool(base[root_r, far_col]) and all(not bool(base[root_r, cc]) for cc in gap_cols):
                for cc in fill_cols:
                    out[root_r, cc] = int(fg)
            continue

        right_gap_start = int(root_c - chain_len)
        right_chain = set(_chain_coords(root_r, right_gap_start, chain_len, "right"))
        if pts == right_chain:
            gap_cols = [int(root_c - chain_len + i) for i in range(chain_len)]
            fill_cols = gap_cols[:-1]
            far_col = int(root_c - chain_len - 1)
            if far_col >= 0 and bool(base[root_r, far_col]) and all(not bool(base[root_r, cc]) for cc in gap_cols):
                for cc in fill_cols:
                    out[root_r, cc] = int(fg)

    m = out == int(fg)
    changed = True
    while changed:
        changed = False
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
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

    out2 = np.full((h, w), int(bg), dtype=int)
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
GAP_BLOCKED_MARGIN = 2
GAP_SEG_INSET = 2
TARGET_GAP_COUNTS = (2, 3)

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


def _h_gap_options(
    seg: dict,
    out: np.ndarray,
    fg: int,
    bg: int,
    occupied: set[tuple[int, int]],
    gap_size: int,
) -> list[tuple[int, str]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str]] = []
    for g in range(int(c0 + GAP_SEG_INSET), int(c1 - gap_size)):
        if any(int(g - GAP_BLOCKED_MARGIN) <= int(b) <= int(g + gap_size - 1 + GAP_BLOCKED_MARGIN) for b in blocked):
            continue
        temp = out.copy()
        temp[row, g : g + gap_size] = bg
        for anchor in ("left", "right"):
            coords = _chain_coords(row, g, gap_size, anchor)
            if not _cells_are_color(temp, coords, bg):
                continue
            if not _validate_chain(temp, coords, fg, occupied):
                continue
            options.append((int(g), anchor))
    return options


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


def _collect_h_candidates(clean: np.ndarray, h_segments: list[dict], fg: int, bg: int) -> list[dict]:
    candidates: list[dict] = []
    for seg in h_segments:
        row = int(seg["row"])
        for gap_size in GAP_SIZE_VALUES:
            for gap_start, anchor in _h_gap_options(seg, clean, fg, bg, set(), gap_size):
                chain = _chain_coords(row, gap_start, gap_size, anchor)
                candidates.append(
                    {
                        "row": row,
                        "gap_start": int(gap_start),
                        "gap_size": int(gap_size),
                        "anchor": anchor,
                        "chain": chain,
                        "band": _gap_forbidden_band(row, int(gap_start), int(gap_size)),
                    }
                )
    return candidates


def _choose_h_gaps(
    rng: random.Random,
    clean: np.ndarray,
    h_segments: list[dict],
    fg: int,
    bg: int,
) -> list[dict] | None:
    candidates = _collect_h_candidates(clean, h_segments, fg, bg)
    if not candidates:
        return None

    best: list[dict] | None = None
    best_score = (-1, -1, -1, -1)

    for target_count in (3, 2):
        for _ in range(400):
            order = candidates[:]
            rng.shuffle(order)
            chosen: list[dict] = []
            forbidden_chain_cells: set[tuple[int, int]] = set()
            forbidden_bands: set[tuple[int, int]] = set()

            for cand in order:
                if any(cell in forbidden_chain_cells for cell in cand["chain"]):
                    continue
                if any(cell in forbidden_bands for cell in cand["band"]):
                    continue
                chosen.append(cand)
                _reserve_neighborhood(forbidden_chain_cells, cand["chain"])
                forbidden_bands.update(cand["band"])
                if len(chosen) == target_count:
                    break

            if len(chosen) != target_count:
                continue

            dirs = {cand["anchor"] for cand in chosen}
            sizes = {cand["gap_size"] for cand in chosen}
            score = (
                int(len(dirs) >= 2),
                int(len(sizes) >= 2),
                int(max(sizes)),
                int(len(chosen)),
            )
            if score > best_score:
                best = chosen
                best_score = score
            if score == (1, 1, 3, target_count):
                return chosen

    return best


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
    clean, h_segments, _v_segments = built

    chosen = _choose_h_gaps(rng, clean, h_segments, fg, bg)
    if chosen is None or len(chosen) < min(TARGET_GAP_COUNTS):
        return None

    out = clean.copy()
    inp = clean.copy()
    occupied: set[tuple[int, int]] = set()
    for item in chosen:
        row = int(item["row"])
        gap_start = int(item["gap_start"])
        gap_size = int(item["gap_size"])
        anchor = str(item["anchor"])
        out[row, _output_hole_col(gap_start, gap_size, anchor)] = bg
        inp[row, gap_start : gap_start + gap_size] = bg
        chain = _chain_coords(row, gap_start, gap_size, anchor)
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo variant")
