# Variant of 7b80bb43
# Dimension: D2 — 对角提示像素的角色 (src)
# Change: 45度阶梯状对角像素（无正交前景邻居，有对角前景邻居）作为间隙位置的提示标记
#   → 种子像素在间隙内部；提示不再位于间隙外侧，而是位于 3 格断口的中心。
#
# Design analysis
# spec_variant:
#   a_concretization_and_trace:
#     input_objects:
#       - 输出仍然是单色前景 fg 构成的轴对齐线段网络，布局约束保持原题。
#       - 每个被破坏的线段断口仍然长度固定为 3。
#       - 原题输入在断口外侧放 2 像素对角 fg 链；本变体改成把断口中间像素染成第三色 seed，
#         于是局部模式变成：
#         horizontal: fg fg bg seed bg fg fg
#         vertical:   fg/seed/bg 模式沿列同理
#       - 另有 0-2 个 seed 干扰像素，位于开阔背景，最终直接消失。
#     behavior_trace:
#       - 原逻辑：先找“仅对角连接的 fg 链”并删除，再利用其邻域判断附近 3 格断口应回填。
#       - 新逻辑：直接以 seed 为起点，检查它是否正好位于某个 3 格断口中心；若是，就把该 3 格全部改回 fg。
#       - 差异：提示源从“断口外的对角 fg 形状”改成“断口内部的第三色像素”，求解入口完全改为 seed-centered。
#   b_complexity_calibration:
#     - 保留 2-3 个断口，且同时覆盖 H/V 两种方向，避免任务退化成单一局部替换。
#     - 保留 0-2 个同色 seed 干扰点，避免“所有第三色都直接变 fg”的过度简化。
#     - seed 颜色固定为单一第三色，减少颜色自由度，保证 2-4 个样例足以归纳。
#   c_old_main_assumption_audit:
#     - 原 `main()` 假设“所有提示像素也是 fg”，并通过 `deg4==0 and has_diag` 找噪声；此假设失效。
#     - 原 `main()` 假设“提示与断口的联系通过空间邻近（Chebyshev-1）建立”；此假设失效。
#     - 原 `main()` 依赖“删除对角链后再按行/列找小间隙”；本变体必须先识别 seed 所在方向，再定点回填。
#     - 原局部修枝依赖残余 fg 叶子与对角噪声共存；本变体不再生成该类残余，因此去掉该流程。
#   d_placement_legality:
#     - 断口仍只放在线段内部，避开端点和 T 结点。
#     - 每个断口两侧都至少保留 2 个连续 fg，保证 seed 周围存在稳定的 `fg fg bg seed bg fg fg` /
#       竖直同构模式，便于 solver 唯一判定方向。
#     - gap 支撑窗口及其 Chebyshev-1 邻域会被预留，避免多个 seed 互相干扰或形成双向歧义。
#     - 干扰 seed 只能放在开放背景中，且插入后不得满足任何有效 gap 模式。
#   e_visual_observability:
#     - 人类可直接看到：第三色只出现在断线中点或空白背景；位于断线中点的会触发整段补全，背景中的则消失。
#     - 不依赖颜色编号大小，只依赖“第三色在断口中心且与线段的几何关系”。
#   f_visual_cleanliness:
#     - 保留原网络的稀疏布局；seed 只是一颗单像素标记，不再叠加对角链，局部视觉比原题更整洁。
#     - 干扰 seed 与任何 fg 至少隔开一个 Chebyshev-1 邻域，防止看上去像断口的一部分。
#   g_color_inference:
#     - 只有 bg / fg / seed 三类颜色；seed 的作用完全由位置结构决定，不需要额外颜色映射。
#   new_main_assumptions_checked_against_generator:
#     - `main()` 假设 seed 像素不是 fg 也不是 bg；generator 始终用单一第三色放置。
#     - `main()` 假设有效 seed 在且仅在一个轴向上满足 7 格局部模板；generator 用候选检测和邻域预留保证这一点。
#     - `main()` 假设无效 seed 不会偶然匹配模板；generator 放置干扰点前会显式排除这种情况。

from __future__ import annotations

import random

import numpy as np

from arc_utils import draw_line, find_components, get_bg


GAP_SIZE = 3
SEED_TEST_COLOR = 10


def _infer_fg_color(grid: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(grid)} - {int(bg)})
    assert colors
    H, W = grid.shape

    def score(color: int) -> tuple[int, int, int]:
        mask = grid == int(color)
        pts = np.argwhere(mask)
        adj_cells = 0
        adj_sum = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            d = 0
            if r > 0 and bool(mask[r - 1, c]):
                d += 1
            if r + 1 < H and bool(mask[r + 1, c]):
                d += 1
            if c > 0 and bool(mask[r, c - 1]):
                d += 1
            if c + 1 < W and bool(mask[r, c + 1]):
                d += 1
            if d > 0:
                adj_cells += 1
                adj_sum += d
        return int(adj_cells), int(adj_sum), int(mask.sum())

    return int(max(colors, key=score))


def _seed_gap_axes(grid: np.ndarray, r: int, c: int, fg: int, bg: int) -> tuple[bool, bool]:
    H, W = grid.shape
    val = int(grid[int(r), int(c)])
    if val == int(bg) or val == int(fg):
        return False, False

    h_ok = (
        c - 3 >= 0
        and c + 3 < W
        and int(grid[int(r), int(c - 3)]) == int(fg)
        and int(grid[int(r), int(c - 2)]) == int(fg)
        and int(grid[int(r), int(c - 1)]) == int(bg)
        and int(grid[int(r), int(c + 1)]) == int(bg)
        and int(grid[int(r), int(c + 2)]) == int(fg)
        and int(grid[int(r), int(c + 3)]) == int(fg)
        and (r == 0 or int(grid[int(r - 1), int(c)]) != int(fg))
        and (r == H - 1 or int(grid[int(r + 1), int(c)]) != int(fg))
    )
    v_ok = (
        r - 3 >= 0
        and r + 3 < H
        and int(grid[int(r - 3), int(c)]) == int(fg)
        and int(grid[int(r - 2), int(c)]) == int(fg)
        and int(grid[int(r - 1), int(c)]) == int(bg)
        and int(grid[int(r + 1), int(c)]) == int(bg)
        and int(grid[int(r + 2), int(c)]) == int(fg)
        and int(grid[int(r + 3), int(c)]) == int(fg)
        and (c == 0 or int(grid[int(r), int(c - 1)]) != int(fg))
        and (c == W - 1 or int(grid[int(r), int(c + 1)]) != int(fg))
    )
    return bool(h_ok), bool(v_ok)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete broken straight lines using seed pixels placed inside each gap.

    Input structure:
    - bg = background.
    - One foreground color fg forms axis-aligned line segments.
    - A second non-bg color seed may appear:
      * at the center of a 3-cell horizontal/vertical gap, or
      * as an isolated distractor in open background.

    Rule:
    - Infer fg as the non-bg color that forms the orthogonal network.
    - For each seed pixel, test whether it is the center of a 3-cell gap on exactly one axis.
    - If yes, fill the whole 3-cell gap with fg.
    - Remove all remaining seed pixels.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    fg = int(_infer_fg_color(g, bg))

    out_mask = g == int(fg)
    seed_cells = np.argwhere((g != int(bg)) & (g != int(fg)))
    for r, c in seed_cells:
        r = int(r)
        c = int(c)
        h_ok, v_ok = _seed_gap_axes(g, r, c, fg, bg)
        if h_ok == v_ok:
            continue
        if h_ok:
            out_mask[int(r), int(c - 1) : int(c) + 2] = True
        else:
            out_mask[int(r) - 1 : int(r) + 2, int(c)] = True

    out = np.full(g.shape, int(bg), dtype=int)
    out[out_mask] = int(fg)
    return out


# --------------- generator (seed-in-gap variant) ---------------

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

GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 2
DISTRACTOR_MARGIN = 2

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


def _h_support_cells(row: int, gap_start: int) -> list[tuple[int, int]]:
    return [(int(row), int(c)) for c in range(int(gap_start - 2), int(gap_start + 5))]


def _v_support_cells(col: int, gap_start: int) -> list[tuple[int, int]]:
    return [(int(r), int(col)) for r in range(int(gap_start - 2), int(gap_start + 5))]


def _reserve_neighborhood(cells: set[tuple[int, int]], coords) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cells.add((int(r + dr), int(c + dc)))


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   reserved: set[tuple[int, int]]) -> list[int]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[int] = []
    for g in range(int(c0 + GAP_SEG_INSET), int(c1 - GAP_SIZE - 1)):
        if any(int(g - GAP_BLOCKED_LO) <= int(b) <= int(g + GAP_BLOCKED_HI) for b in blocked):
            continue
        support = _h_support_cells(int(row), int(g))
        if any(cell in reserved for cell in support):
            continue
        temp = out.copy()
        temp[int(row), int(g) : int(g) + GAP_SIZE] = int(bg)
        temp[int(row), int(g + 1)] = SEED_TEST_COLOR
        h_ok, v_ok = _seed_gap_axes(temp, int(row), int(g + 1), fg, bg)
        if h_ok and not v_ok:
            options.append(int(g))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, bg: int,
                   reserved: set[tuple[int, int]]) -> list[int]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[int] = []
    for g in range(int(r0 + GAP_SEG_INSET), int(r1 - GAP_SIZE - 1)):
        if any(int(g - GAP_BLOCKED_LO) <= int(b) <= int(g + GAP_BLOCKED_HI) for b in blocked):
            continue
        support = _v_support_cells(int(col), int(g))
        if any(cell in reserved for cell in support):
            continue
        temp = out.copy()
        temp[int(g) : int(g) + GAP_SIZE, int(col)] = int(bg)
        temp[int(g + 1), int(col)] = SEED_TEST_COLOR
        h_ok, v_ok = _seed_gap_axes(temp, int(g + 1), int(col), fg, bg)
        if v_ok and not h_ok:
            options.append(int(g))
    return options


def _distractor_options(inp: np.ndarray, fg: int, bg: int,
                        forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = inp.shape
    options: list[tuple[int, int]] = []
    for r in range(DISTRACTOR_MARGIN, h - DISTRACTOR_MARGIN):
        for c in range(DISTRACTOR_MARGIN, w - DISTRACTOR_MARGIN):
            if int(inp[int(r), int(c)]) != int(bg):
                continue
            if (int(r), int(c)) in forbidden:
                continue
            temp = inp.copy()
            temp[int(r), int(c)] = SEED_TEST_COLOR
            h_ok, v_ok = _seed_gap_axes(temp, int(r), int(c), fg, bg)
            if h_ok or v_ok:
                continue
            options.append((int(r), int(c)))
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
    seed = rng.choice([c for c in range(N_COLORS) if c not in {bg, fg}])
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(0, 2)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    reserved: set[tuple[int, int]] = set()
    placed_gap_seeds: list[tuple[int, int]] = []

    h_candidates = [(seg, _h_gap_options(seg, out, fg, bg, reserved)) for seg in h_segments]
    h_candidates = [(seg, opts) for seg, opts in h_candidates if opts]
    if not h_candidates:
        return None
    rng.shuffle(h_candidates)
    h_seg, h_opts = h_candidates[0]
    h_gap = rng.choice(h_opts)
    h_row = int(h_seg["row"])
    inp[h_row, int(h_gap) : int(h_gap) + GAP_SIZE] = int(bg)
    inp[h_row, int(h_gap) + 1] = int(seed)
    placed_gap_seeds.append((int(h_row), int(h_gap) + 1))
    _reserve_neighborhood(reserved, _h_support_cells(h_row, int(h_gap)))

    need_v = gap_count - 1
    rng.shuffle(v_segments)
    placed_v = 0
    for v_seg in v_segments:
        if placed_v >= need_v:
            break
        v_opts = _v_gap_options(v_seg, out, fg, bg, reserved)
        if not v_opts:
            continue
        v_gap = rng.choice(v_opts)
        v_col = int(v_seg["col"])
        inp[int(v_gap) : int(v_gap) + GAP_SIZE, v_col] = int(bg)
        inp[int(v_gap) + 1, v_col] = int(seed)
        placed_gap_seeds.append((int(v_gap) + 1, v_col))
        _reserve_neighborhood(reserved, _v_support_cells(v_col, int(v_gap)))
        placed_v += 1
    if placed_v != need_v:
        return None

    forbidden: set[tuple[int, int]] = set()
    _reserve_neighborhood(forbidden, list(map(tuple, np.argwhere(inp != bg))))
    distractors = _distractor_options(inp, fg, bg, forbidden)
    rng.shuffle(distractors)
    placed = 0
    for r, c in distractors:
        if placed >= distractor_count:
            break
        if (int(r), int(c)) in forbidden:
            continue
        inp[int(r), int(c)] = int(seed)
        _reserve_neighborhood(forbidden, [(int(r), int(c))])
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D2 seed variant")
