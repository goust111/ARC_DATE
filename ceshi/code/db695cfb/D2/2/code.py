# Variant of db695cfb
# Dimension: D2 — 延伸路径方向规则 (path)
# Change: 垂直对角延伸（perpendicular_extension）——从品红像素出发，沿原对角线旋转90度的两个方向延伸至网格边界 → 反射式延伸

# design_analysis:
#   a_concretization_and_trace:
#     original_scene:
#       - 蓝色端点定义 45 度对角线段；品红像素若落在线段上，就向垂直对角线两端各打一条品红射线。
#       - 例如 "\" 线段经过一个品红点时，原题只会额外生成 "/" 方向两条品红线。
#     variant_scene:
#       - 仍先把同色端点之间的主对角线补全为蓝线。
#       - 但线段穿过品红像素时，品红像素被视为半透明反射墙：
#         1. 穿透分支继续沿主对角线方向向两端延伸到边界；
#         2. 反射分支沿另一组对角线方向向两端延伸到边界。
#       - 所有新增分支都保持品红色，只填背景；蓝色主线和已有非背景像素不被覆盖。
#     concrete_trace:
#       - 若蓝色端点在 (2,2) 和 (6,6)，品红触发点在 (4,4)。
#       - 原题输出：补出 "\" 上的蓝线，并从 (4,4) 向 (3,5)/(5,3)/(2,6)... 发出 "/" 品红射线。
#       - 变体输出：保留上面的内容，同时再从 (4,4) 沿 "\" 方向向 (1,1),(0,0) 以及 (7,7),... 继续延伸品红穿透线。
#       - 差异在于：原题只有“垂直叉”，变体是“原方向穿透 + 垂直反射”的四向分叉。
#   b_complexity_calibration:
#     - 反射只引入一种新分支机制，不再叠加多次弹跳或变色，复杂度仍接近原题。
#     - 为保证人眼能看见 D2 变化，生成器要求整题至少有一个在线触发点；否则会退化成只连蓝线。
#     - 保留 1-4 条线段和 0-2 个触发点/段的原始规模，但通过占位约束避免大面积重叠。
#   c_implicit_assumption_audit:
#     - 原 `draw_segment()` 假设触发点只需记录垂直方向；变体下必须改成记录“原方向 + 反方向”共四个半射线。
#     - 原发射循环只知道 `perp_dirs`；若继续沿用会丢失穿透分支。
#     - 原代码依赖“线段之间互不共享对角线”才不会把多个端点误连成一条更长线；生成器显式保证每条对角常数只对应一段。
#     - 原代码只在 `bg` 上色，这一前提仍需保留，否则穿透线会覆盖蓝端点或别的触发点，破坏可读性。
#   d_placement_legality:
#     - 线段路径本身不能与已有输入像素或既有输出轨迹重叠。
#     - 新分支轨迹除允许经过本段自身路径外，不能碰到别段的输入像素或输出轨迹。
#     - 触发点必须在线段内部；离线品红干扰点只能放在不会被任何线段或分支触及的位置。
#   e_visual_observability:
#     - 至少一个真实触发点确保样本不退化。
#     - 新旧规则在同一输入上的差别可直接看成：端点之外是否出现沿“同一主对角线”继续外射的品红线。
#     - 该差别不依赖颜色 id 排序，也不依赖隐藏元数据。
#   f_visual_cleanliness:
#     - 生成器避免多段共享同一对角线，也避免分支穿过别段端点，减少“看不出哪条线属于谁”的混叠。
#     - 分支仍只用两种前景色：蓝主线、品红分支，视觉层次稳定。
#   g_color_inference:
#     - 颜色规则不变：蓝色负责定义主线，品红色负责触发并染色所有新增分支。
#     - 因而解题者只需推断几何变化，不需要额外学习新的颜色映射。

import random

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete same-color diagonal connectors, then split their path at trigger pixels into transmitted and reflected diagonal branches.

    Input structure:
    - The background is the dominant color. One non-background color marks endpoint pairs that lie on 45-degree diagonals, while other colored pixels may sit directly on those connector diagonals as triggers.

    Rule:
    - Fill each diagonal connector between the extreme same-color endpoints on a shared diagonal.
    - Whenever that connector passes through a trigger pixel, keep the connector intact and emit trigger-colored branches both along the original diagonal direction and along the reflected diagonal direction, extending to the grid boundary.

    Key insight:
    - A trigger behaves like a semi-transparent diagonal mirror: the main beam continues through it while a reflected copy leaves along the other diagonal, so each on-line trigger creates a four-way split centered on the trigger.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        dr = 1 if r1 > r0 else -1
        dc = 1 if c1 > c0 else -1
        steps = abs(r1 - r0)
        return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]

    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        branch_dirs = [(dr, dc), (-dr, -dc), (dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                key = (r, c, v)
                triggers.setdefault(key, set()).update(branch_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _compute_branch_cells(
    triggers: list[tuple[int, int]],
    branch_dirs: list[tuple[int, int]],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    """Compute all cells touched by transmitted + reflected branches from the triggers."""
    cells: set[tuple[int, int]] = set()
    for tr, tc in triggers:
        for ddr, ddc in branch_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    """Build the expected output grid by filling diagonals and emitting split branches."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, branch_dirs in segments:
        for r, c in path:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in branch_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _build_baseline_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    """Build the original-task output for the same planned segments."""
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, branch_dirs in segments:
        for r, c in path:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        baseline_dirs = branch_dirs[2:]
        for tr, tc in trigger_positions:
            for ddr, ddc in baseline_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    used_diag_keys: set[tuple[str, int]],
):
    """Try to place one diagonal segment, returning (path, dc, diag_key) or None."""
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
            diag_key = ("backslash", r0 - c0)
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1
            diag_key = ("slash", r0 + c0)

        if diag_key in used_diag_keys:
            continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc, diag_key
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    used_diag_keys: set[tuple[str, int]] = set()
    segments = []
    total_triggers = 0

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn, used_diag_keys)
        if result is None:
            continue
        path, dc, diag_key = result
        path_set = set(path)

        path_interior = path[1:-1]
        max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
        min_triggers = min(_TRIGGERS_PER_SEG[0], max_triggers)
        n_triggers = random.randint(min_triggers, max_triggers)
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 else []

        branch_dirs = [(1, dc), (-1, -dc), (1, -dc), (-1, dc)]
        branch_cells = _compute_branch_cells(trigger_positions, branch_dirs, H, W)
        new_branch_cells = branch_cells - path_set
        if any(rc in occupied or rc in drawn for rc in new_branch_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path, branch_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(path_set - {ep0, ep1} - set(trigger_positions))
        drawn.update(new_branch_cells)
        used_diag_keys.add(diag_key)
        total_triggers += len(trigger_positions)

    if not segments or total_triggers == 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _path, _branch_dirs in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    baseline_expected = _build_baseline_expected(inp, bg, segments)
    if np.array_equal(result, expected) and not np.array_equal(expected, baseline_expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
