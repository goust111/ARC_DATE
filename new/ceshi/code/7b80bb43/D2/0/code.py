from __future__ import annotations

# Variant of 7b80bb43
# Dimension: D2 — 对角提示像素的角色 (src)
# Change: 45度阶梯状对角像素（无正交前景邻居，有对角前景邻居）作为间隙位置的提示标记
#         → 正交突出像素作为提示：从线段端点向垂直方向伸出的 1-2 像素短臂。
#
# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     before: 输入中的提示是脱离主线的 45 度对角链；solver 通过“无正交邻居、有对角邻居”的像素来识别它们。
#     after: 输入中的提示改成附着在轴对齐线段端点上的短支臂；solver 需要识别“端点处的直角拐点”。
#   concrete_shape:
#     output:
#       仍是单色、轴对齐的 H/V 线段网络，包含 2-3 个 T 形交点，3-5 个 4 连通组件。
#     input_gap_hint:
#       对一条水平线段切出 3 像素横向缺口后，在缺口一侧留下的线端上，加一条向上或向下的 1-2 像素竖臂。
#       对一条竖直线段切出 3 像素纵向缺口后，在缺口一侧留下的线端上，加一条向左或向右的 1-2 像素横臂。
#     input_distractor:
#       在本来就存在的自由端点上，也可附加同类短支臂，但它旁边没有待修复缺口；solver 只删除支臂，不补线。
#   behavior_trace:
#     example:
#       original:
#         水平段 "...###...###" 的缺口附近放一条对角链，删除对角链后根据其邻域把中间 3 格补回。
#       variant:
#         水平段 "...###___###" 的左半端点处变成一个 L 形：根像素连着左侧主干和上/下短臂。
#         solver 先识别该 L 形根部：两邻居正交，一侧是短叶臂(1-2)，另一侧是较长主干。
#         删除短臂后，沿主干的缺失延长方向查看 1-3 格背景后是否重新遇到同一行/列的前景；
#         若遇到，则把中间背景补满。
#       difference:
#         原逻辑依赖“对角像素完全与主线 4 断开”；新逻辑依赖“提示像素与主线正交连接并形成短叶臂”。
#   complexity_calibration:
#     kept:
#       仍然只有两种颜色、仍然是 3 像素缺口、仍然是 2-3 个提示缺口、仍然保留少量无补线的干扰提示。
#     adapted:
#       为了让短臂和主干可区分，生成器要求被打断线段在提示根部向主干方向至少保留 3 像素可见长度。
#       这样 1-2 像素短臂在视觉上稳定更短，不会与真正线段端部混淆。
#   implicit_assumption_review:
#     old_solver_assumptions:
#       - deg4==0 且有对角邻居即可判定提示像素：新变体不成立，因为短臂像素是 4 连通的。
#       - 提示本体从主线完全分离：新变体不成立，短臂直接挂在线端。
#       - 用对角组件 span 推断最大补线距离：新变体无此组件，因此改回固定 3 格缺口上限。
#       - 局部修剪依赖“提示附近的对角叶子”：新变体的残余是正交短臂，因此必须显式整臂删除。
#     new_solver_assumptions:
#       - 输出网络本身没有拐角 elbow，只有直线和 T 交点；因此输入中的 L 形根部可以视为提示根部。
#       - 根部两条分支中，真正的提示臂长度至多 2，且是 straight leaf；主干分支长度至少 3。
#       - 若该根部真是缺口提示，则沿主干缺失方向在 3 格背景后会重新遇到同轴线段。
#     generator_guarantees:
#       - gap 位置距剩余主干两端至少 3 像素；
#       - 提示臂长度只取 1 或 2；
#       - 提示臂所有像素都落在空背景上，且不与其他提示重叠；
#       - 干扰提示只放在真实自由端点上，因此删除后不影响目标输出。
#   placement_legality:
#     - 所有短臂都在切口后验证：臂像素在边界内、原本为背景、除根部外不接触其他前景。
#     - gap 仍固定为 3 格，不跨越 T 交点，不切到线段端点。
#     - 干扰提示复用相同合法性检查，并避开已有提示根部/臂像素。
#   visual_observability:
#     - 输入里最显著的新现象是“线端长出一个垂直小刺”，而不是附近飘着对角链。
#     - 多个样例中会重复出现水平线配竖刺、竖线配横刺两种情况，足够让人归纳“垂直突出=修线提示”。
#     - 干扰提示与真实提示长相相同，但只有部分旁边存在 3 格缺口，因此可从 input→output 对比看出：
#       所有短臂都删除，只有能对接另一段直线的会触发补线。
#   visual_cleanliness:
#     - 保持原始网络稀疏布局，提示臂最长 2 格，避免在同一区域堆叠多种形状。
#     - 不再混用对角链，整题视觉语言统一成“端点小刺”。
#   color_reasoning:
#     - 不引入新的颜色规则；颜色仍然只是一个前景色 + 一个背景色，因此不存在不可见的颜色编码。

import random
import numpy as np

from arc_utils import draw_line, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove short orthogonal prompt arms and complete the 3-cell gaps they mark.

    Prompt geometry in this variant:
    - A prompt is a 1-2 pixel terminal arm attached orthogonally to a line endpoint.
    - The root pixel is an elbow: one neighbor continues the real line, one neighbor leads into the short arm.
    - If a matching straight segment reappears within 3 background cells in the missing continuation direction,
      that gap is filled after the arm is deleted.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = {int(x) for x in np.unique(g)}
    colors.discard(bg)
    assert colors
    fg = int(next(iter(colors)))

    H, W = g.shape
    mask = g == fg
    MAX_GAP = 3
    ARM_MAX = 2

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    def fg_neighbors(m: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if in_bounds(rr, cc) and bool(m[rr, cc]):
                out.append((rr, cc))
        return out

    def ray_info(m: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[list[tuple[int, int]], bool]:
        """
        Follow a straight 4-connected ray starting from the neighbor in (dr,dc).
        Returns (cells, is_straight_leaf_path).
        """
        rr, cc = r + dr, c + dc
        if not (in_bounds(rr, cc) and bool(m[rr, cc])):
            return [], False

        cells: list[tuple[int, int]] = []
        prev = (r, c)
        while True:
            cells.append((rr, cc))
            nxt = [p for p in fg_neighbors(m, rr, cc) if p != prev]
            if not nxt:
                return cells, True
            if len(nxt) != 1:
                return cells, False
            nr, nc = nxt[0]
            if (nr - rr, nc - cc) != (dr, dc):
                return cells, False
            prev = (rr, cc)
            rr, cc = nr, nc

    prompt_roots: list[tuple[int, int, int, int]] = []
    prompt_pixels = np.zeros((H, W), dtype=bool)

    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]):
                continue

            up = r > 0 and bool(mask[r - 1, c])
            down = r + 1 < H and bool(mask[r + 1, c])
            left = c > 0 and bool(mask[r, c - 1])
            right = c + 1 < W and bool(mask[r, c + 1])
            h_dirs = [d for d, ok in (((0, -1), left), ((0, 1), right)) if ok]
            v_dirs = [d for d, ok in (((-1, 0), up), ((1, 0), down)) if ok]

            # A prompt root is an elbow: exactly one horizontal and one vertical neighbor.
            if len(h_dirs) != 1 or len(v_dirs) != 1:
                continue

            candidates: list[tuple[list[tuple[int, int]], tuple[int, int], bool]] = []
            for dr, dc in (h_dirs[0], v_dirs[0]):
                cells, is_leaf = ray_info(mask, r, c, dr, dc)
                candidates.append((cells, (dr, dc), is_leaf))

            short_leaf = [(cells, d) for cells, d, is_leaf in candidates if is_leaf and 1 <= len(cells) <= ARM_MAX]
            long_branch = [(cells, d) for cells, d, _is_leaf in candidates if len(cells) >= ARM_MAX + 1]
            if len(short_leaf) != 1 or len(long_branch) != 1:
                continue

            arm_cells, arm_dir = short_leaf[0]
            trunk_cells, trunk_dir = long_branch[0]
            if arm_dir == trunk_dir:
                continue

            for rr, cc in arm_cells:
                prompt_pixels[rr, cc] = True
            prompt_roots.append((r, c, trunk_dir[0], trunk_dir[1]))

    out = g.copy()
    out[prompt_pixels] = bg

    for r, c, trunk_dr, trunk_dc in prompt_roots:
        miss_dr, miss_dc = -trunk_dr, -trunk_dc
        hit_dist = None
        for dist in range(2, MAX_GAP + 2):
            rr = r + miss_dr * dist
            cc = c + miss_dc * dist
            if not in_bounds(rr, cc):
                break
            mid_ok = True
            for step in range(1, dist):
                mr = r + miss_dr * step
                mc = c + miss_dc * step
                if bool(out[mr, mc] == fg):
                    mid_ok = False
                    break
            if not mid_ok:
                break
            if bool(out[rr, cc] == fg):
                next_rr = rr + miss_dr
                next_cc = cc + miss_dc
                if not (in_bounds(next_rr, next_cc) and bool(out[next_rr, next_cc] == fg)):
                    continue
                hit_dist = dist
                break
        if hit_dist is None:
            continue
        for step in range(1, hit_dist):
            rr = r + miss_dr * step
            cc = c + miss_dc * step
            out[rr, cc] = fg

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

# Gap / prompt placement
GAP_SIZE = 3
ARM_LENGTHS = (1, 2)
GAP_BLOCKED_LO = 2
GAP_BLOCKED_HI = 4
GAP_SEG_INSET = 3

# Distractor prompt count
DISTRACTOR_MAX = 2

# Retry budget
GEN_RETRIES = 1000


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
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != int(color):
            return False
    return True


def _h_prompt_coords(row: int, root_col: int, dr: int, arm_len: int) -> list[tuple[int, int]]:
    return [(row + dr * i, root_col) for i in range(1, arm_len + 1)]


def _v_prompt_coords(root_row: int, col: int, dc: int, arm_len: int) -> list[tuple[int, int]]:
    return [(root_row, col + dc * i) for i in range(1, arm_len + 1)]


def _validate_prompt_arm(grid: np.ndarray, fg: int, root: tuple[int, int],
                         coords: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    """
    Validate a prompt arm against the solver's assumptions:
    - coordinates are straight and on background,
    - the arm is a pure leaf chain attached only through the root.
    """
    if not coords:
        return False
    h, w = grid.shape
    root_r, root_c = root
    coord_set = set(coords)
    if any(cell in occupied for cell in coords):
        return False
    if any(not (0 <= r < h and 0 <= c < w) for r, c in coords):
        return False
    bg = int(get_bg(grid))
    if any(int(grid[r, c]) != bg for r, c in coords):
        return False

    dr = coords[0][0] - root_r
    dc = coords[0][1] - root_c
    if abs(dr) + abs(dc) != 1:
        return False
    dr = 0 if dr == 0 else dr // abs(dr)
    dc = 0 if dc == 0 else dc // abs(dc)

    for i, cell in enumerate(coords, start=1):
        if cell != (root_r + dr * i, root_c + dc * i):
            return False

    temp = np.array(grid, copy=True)
    for r, c in coords:
        temp[r, c] = fg

    for i, (r, c) in enumerate(coords):
        expected = {(root_r, root_c) if i == 0 else coords[i - 1]}
        if i + 1 < len(coords):
            expected.add(coords[i + 1])
        actual = set()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and int(temp[rr, cc]) == int(fg):
                actual.add((rr, cc))
        if actual != expected:
            return False

    return True


def _place_t_component(out: np.ndarray, fg: int, slot: str, row: int,
                       branch_col: int, end_col: int) -> tuple[dict, dict]:
    H, W = out.shape
    h_c0 = 0 if slot[0] == "L" else end_col
    h_c1 = end_col if slot[0] == "L" else W - 1
    v_r0 = 0 if slot[1] == "T" else row
    v_r1 = row if slot[1] == "T" else H - 1
    draw_line(out, row, h_c0, end_r=row, end_c=h_c1, color=fg)
    draw_line(out, v_r0, branch_col, end_r=v_r1, end_c=branch_col, color=fg)
    h_seg = {"row": row, "c0": h_c0, "c1": h_c1, "blocked": {branch_col}, "side": slot[0]}
    v_seg = {"col": branch_col, "r0": v_r0, "r1": v_r1, "blocked": {row}, "side": slot[1]}
    return h_seg, v_seg


def _place_simple_h(out: np.ndarray, fg: int, side: str, row: int, length: int) -> dict:
    W = out.shape[1]
    c0 = 0 if side == "left" else W - length
    c1 = length - 1 if side == "left" else W - 1
    draw_line(out, row, c0, end_r=row, end_c=c1, color=fg)
    return {"row": row, "c0": c0, "c1": c1, "blocked": set(), "side": "L" if side == "left" else "R"}


def _place_simple_v(out: np.ndarray, fg: int, side: str, col: int, length: int) -> dict:
    H = out.shape[0]
    r0 = 0 if side == "top" else H - length
    r1 = length - 1 if side == "top" else H - 1
    draw_line(out, r0, col, end_r=r1, end_c=col, color=fg)
    return {"col": col, "r0": r0, "r1": r1, "blocked": set(), "side": "T" if side == "top" else "B"}


def _h_gap_options(seg: dict, out: np.ndarray, fg: int, occupied: set[tuple[int, int]],
                   used_roots: set[tuple[int, int]]) -> list[tuple[int, str, int, int]]:
    row, c0, c1 = seg["row"], seg["c0"], seg["c1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int, int]] = []
    start_g = c0 + GAP_SEG_INSET
    end_g = c1 - GAP_SIZE - GAP_SEG_INSET + 1
    for g in range(start_g, end_g + 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        bg = int(get_bg(temp))
        temp[row, g:g + GAP_SIZE] = bg
        for anchor in ("left", "right"):
            root = (row, g - 1) if anchor == "left" else (row, g + GAP_SIZE)
            if root in used_roots:
                continue
            for dr in (-1, 1):
                for arm_len in ARM_LENGTHS:
                    coords = _h_prompt_coords(row, root[1], dr, arm_len)
                    if _validate_prompt_arm(temp, fg, root, coords, occupied):
                        options.append((g, anchor, dr, arm_len))
    return options


def _v_gap_options(seg: dict, out: np.ndarray, fg: int, occupied: set[tuple[int, int]],
                   used_roots: set[tuple[int, int]]) -> list[tuple[int, str, int, int]]:
    col, r0, r1 = seg["col"], seg["r0"], seg["r1"]
    blocked = seg["blocked"]
    options: list[tuple[int, str, int, int]] = []
    start_g = r0 + GAP_SEG_INSET
    end_g = r1 - GAP_SIZE - GAP_SEG_INSET + 1
    for g in range(start_g, end_g + 1):
        if any(g - GAP_BLOCKED_LO <= b <= g + GAP_BLOCKED_HI for b in blocked):
            continue
        temp = out.copy()
        bg = int(get_bg(temp))
        temp[g:g + GAP_SIZE, col] = bg
        for anchor in ("top", "bottom"):
            root = (g - 1, col) if anchor == "top" else (g + GAP_SIZE, col)
            if root in used_roots:
                continue
            for dc in (-1, 1):
                for arm_len in ARM_LENGTHS:
                    coords = _v_prompt_coords(root[0], col, dc, arm_len)
                    if _validate_prompt_arm(temp, fg, root, coords, occupied):
                        options.append((g, anchor, dc, arm_len))
    return options


def _segment_endpoint(seg: dict, orientation: str) -> tuple[int, int]:
    if orientation == "h":
        root_col = seg["c1"] if seg["side"] == "L" else seg["c0"]
        return seg["row"], root_col
    root_row = seg["r1"] if seg["side"] == "T" else seg["r0"]
    return root_row, seg["col"]


def _segment_inward_len(seg: dict, orientation: str) -> int:
    if orientation == "h":
        return int(seg["c1"] - seg["c0"])
    return int(seg["r1"] - seg["r0"])


def _endpoint_prompt_options(segments: list[dict], orientation: str, inp: np.ndarray, fg: int,
                             occupied: set[tuple[int, int]],
                             used_roots: set[tuple[int, int]]) -> list[tuple[str, int, tuple[int, int], int]]:
    options: list[tuple[str, int, tuple[int, int], int]] = []
    for idx, seg in enumerate(segments):
        if _segment_inward_len(seg, orientation) < 3:
            continue
        root = _segment_endpoint(seg, orientation)
        if root in used_roots:
            continue
        if orientation == "h":
            for dr in (-1, 1):
                for arm_len in ARM_LENGTHS:
                    coords = _h_prompt_coords(root[0], root[1], dr, arm_len)
                    if _validate_prompt_arm(inp, fg, root, coords, occupied):
                        options.append((orientation, idx, root, dr * arm_len))
        else:
            for dc in (-1, 1):
                for arm_len in ARM_LENGTHS:
                    coords = _v_prompt_coords(root[0], root[1], dc, arm_len)
                    if _validate_prompt_arm(inp, fg, root, coords, occupied):
                        options.append((orientation, idx, root, dc * arm_len))
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
    cfg = rng.choice(VALID_CONFIGS)
    gap_count = 2 if cfg["h"] == 2 else rng.choice([2, 3])
    distractor_count = rng.randint(0, DISTRACTOR_MAX)

    built = _build_output(rng, h, w, bg, fg, cfg)
    if built is None:
        return None
    out, h_segments, v_segments = built

    inp = out.copy()
    occupied: set[tuple[int, int]] = set()
    used_roots: set[tuple[int, int]] = set()

    h_ready = [(seg, opts) for seg in h_segments for opts in [_h_gap_options(seg, out, fg, occupied, used_roots)] if opts]
    v_ready = [(seg, opts) for seg in v_segments for opts in [_v_gap_options(seg, out, fg, occupied, used_roots)] if opts]
    if not h_ready or not v_ready:
        return None

    v_gap_count = gap_count - 1
    if len(v_ready) < v_gap_count:
        return None
    rng.shuffle(h_ready)
    rng.shuffle(v_ready)

    h_seg, h_initial = h_ready[0]
    h_opts = _h_gap_options(h_seg, out, fg, occupied, used_roots) or h_initial
    g, anchor, dr, arm_len = rng.choice(h_opts)
    row = h_seg["row"]
    root = (row, g - 1) if anchor == "left" else (row, g + GAP_SIZE)
    coords = _h_prompt_coords(row, root[1], dr, arm_len)
    inp[row, g:g + GAP_SIZE] = bg
    for rr, cc in coords:
        inp[rr, cc] = fg
    occupied.update(coords)
    used_roots.add(root)

    for v_seg, v_initial in v_ready[:v_gap_count]:
        v_opts = _v_gap_options(v_seg, out, fg, occupied, used_roots) or v_initial
        if not v_opts:
            return None
        g, anchor, dc, arm_len = rng.choice(v_opts)
        col = v_seg["col"]
        root = (g - 1, col) if anchor == "top" else (g + GAP_SIZE, col)
        coords = _v_prompt_coords(root[0], col, dc, arm_len)
        inp[g:g + GAP_SIZE, col] = bg
        for rr, cc in coords:
            inp[rr, cc] = fg
        occupied.update(coords)
        used_roots.add(root)

    distractor_options = _endpoint_prompt_options(h_segments, "h", inp, fg, occupied, used_roots)
    distractor_options += _endpoint_prompt_options(v_segments, "v", inp, fg, occupied, used_roots)
    rng.shuffle(distractor_options)
    placed = 0
    for orientation, idx, root, signed_len in distractor_options:
        if placed >= distractor_count:
            break
        arm_len = abs(signed_len)
        if orientation == "h":
            coords = _h_prompt_coords(root[0], root[1], 1 if signed_len > 0 else -1, arm_len)
        else:
            coords = _v_prompt_coords(root[0], root[1], 1 if signed_len > 0 else -1, arm_len)
        if not _validate_prompt_arm(inp, fg, root, coords, occupied):
            continue
        for rr, cc in coords:
            inp[rr, cc] = fg
        occupied.update(coords)
        used_roots.add(root)
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
    raise RuntimeError("failed to generate valid input for 7b80bb43 D2 variant")
