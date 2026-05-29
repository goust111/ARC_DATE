# Combo variant of 7ed72f31 (index 3)
# combo_spec: D1/0+D2/2+D8/1
# Name: 全局轴旋转链

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


# design_analysis:
#   fused_rule:
#     original:
#       - 每个局部红标记自己定义镜像轴/中心，payload 从静态输入里就地读取，各组独立叠加。
#     combo:
#       - 红色不再附着在局部 sprite 上，而是形成 3-4 条完整的全局分隔线。
#       - 若分隔线是整行，则当前 payload 绕该行中点顺时针旋转 90 度。
#       - 若分隔线是整列，则当前 payload 绕该列中点顺时针旋转 90 度。
#       - 第一步读取输入中的唯一 seed payload；之后每一步都只旋转“上一步刚生成的 payload”，形成链。
#   concrete_trace:
#     row_mode_example:
#       - 宽度取奇数，中心列 m=9；第一条全红轴在第 6 行，seed 的一个像素在 (4,11)，即相对轴为“上 2、右 2”。
#       - 原题若把红线看作局部镜像轴，这个像素甚至不会被处理，因为它和红线不连通。
#       - 本变体第 1 步把它绕整行中点旋到 (8,11)；若下一条全红轴在第 11 行，则该像素变成“上 3、右 2”，
#         第 2 步再旋到 (13,12)。输出因此出现 seed + 第 1 轮 + 第 2 轮 + ... 的连续轨迹。
#       - 差异不只在“镜像改旋转”，还在“轴从局部改全局”以及“后续轴消费前一步结果而非静态输入”。
#   complexity_calibration:
#     - 每个样例只使用一种轴朝向：要么全是整行，要么全是整列。这样顺序可直接由上到下或左到右读出，避免行列混排造成歧义。
#     - 每个样例固定 3-4 条轴，保证链式传递一定可见，不会退化成单步。
#     - seed 使用单色、4 连通、2x2 以上且非 90 度对称的小形状，避免旋转后看不出变化。
#     - 轴之间使用固定 gap=5；seed 与各轮状态都被限制在靠近全局中心的一侧小象限里，便于留白和防碰撞。
#   implicit_assumption_audit:
#     original_main_breakage:
#       - 原 `label_mask(g != bg, connectivity=4)` 默认红标记和 payload 在同一局部组件内；融合后全局红线与 seed 分离，这个前提失效。
#       - 原 `axis = ("row"/"col"/"point")` 由局部红像素几何决定；融合后必须扫描整行/整列红线并按空间顺序执行。
#       - 原 solver 从静态输入 `g` 反复筛 payload；融合后后续步骤必须只读取 `current_payload`，不能回扫整张输入。
#       - 原孤立红点分支完全不适用；组合版只保留全局整行/整列轴。
#     new_main_assumptions:
#       - 至少存在 1 条完整红轴，且样例中只会出现“全行们”或“全列们”二选一，不混用。
#       - 行轴模式要求网格宽度为奇数，列轴模式要求网格高度为奇数，否则 90 度旋转会落到半格；solver 对此通过整点检查自动跳过。
#       - 输入中所有非背景非红像素共同构成链的初始 payload；generator 保证只有这一个 seed 组件。
#       - 每轮 payload 必须整体位于当前轴的一侧；若输入违背这一点，solver 停止后续链，避免产生不符合规格的混合结果。
#   placement_legality:
#     - generator 先在局部偏移坐标里构造整条链，再整体映射到网格；每一步都检查所有 payload 状态与红轴不重叠、不越界、彼此不重叠。
#     - 整行模式把所有状态放在“轴上方且中心列右侧”的小区域；整列模式放在“中心行上方且轴左侧”的小区域。
#     - 红轴是整行/整列，因此它们自身不能穿过任何 payload；固定 gap 和小偏移盒保证这一点。
#   observability_and_cleanliness:
#     - 输入里只有一个 seed payload，但有 3-4 条醒目的全局红线；输出中同一形状沿这些线逐步传播，人类能直接看到“沿顺序多次旋转”。
#     - 不使用颜色值排序，也不加入噪声精灵；顺序只由可见空间顺序决定。
#     - 单色 seed + 大留白让链条比原题更干净，且“旋转而非镜像”的方向变化很明显。
#   anti_degeneration_check:
#     - 量化指标可用 `added_states = (output_colored - input_colored) / seed_size`。本 generator 强制其等于轴数，至少为 3。
#     - 若退化成单步或静态全局镜像，该指标会降到 1 或输出与“只执行第一条轴”一致；测试阶段会专门统计这一点。


MARK = 2
GAP = 5
OFFSET_MAX = 4


def _rot90_cw_center2(center2: tuple[int, int], r: int, c: int) -> tuple[int, int] | None:
    cr2, cc2 = map(int, center2)
    rr2 = cr2 + 2 * int(c) - cc2
    cc2_new = cc2 - 2 * int(r) + cr2
    if rr2 % 2 != 0 or cc2_new % 2 != 0:
        return None
    return rr2 // 2, cc2_new // 2


def _detect_axes(grid: np.ndarray) -> list[tuple[str, int]]:
    g = np.array(grid, dtype=int)
    rows = [int(r) for r in range(g.shape[0]) if np.all(g[int(r), :] == int(MARK))]
    cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, int(c)] == int(MARK))]
    if rows and not cols:
        return [("row", int(r)) for r in sorted(rows)]
    if cols and not rows:
        return [("col", int(c)) for c in sorted(cols)]
    return []


def _initial_payload(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    return sorted(
        (int(r), int(c), int(grid[int(r), int(c)]))
        for r, c in np.argwhere((grid != int(bg)) & (grid != int(MARK)))
    )


def _rotate_payload(
    payload: list[tuple[int, int, int]],
    axis: tuple[str, int],
    shape: tuple[int, int],
) -> list[tuple[int, int, int]]:
    if not payload:
        return []

    h, w = map(int, shape)
    kind, pos = axis
    rows = [int(r) for r, _c, _v in payload]
    cols = [int(c) for _r, c, _v in payload]
    if kind == "row":
        if any(int(r) == int(pos) for r in rows):
            return []
        if not (max(rows) < int(pos) or min(rows) > int(pos)):
            return []
        center2 = (2 * int(pos), int(w - 1))
    else:
        if any(int(c) == int(pos) for c in cols):
            return []
        if not (max(cols) < int(pos) or min(cols) > int(pos)):
            return []
        center2 = (int(h - 1), 2 * int(pos))

    rotated: dict[tuple[int, int], int] = {}
    for r, c, v in payload:
        dst = _rot90_cw_center2(center2, int(r), int(c))
        if dst is None:
            return []
        rr, cc = dst
        if not (0 <= int(rr) < int(h) and 0 <= int(cc) < int(w)):
            return []
        rotated[(int(rr), int(cc))] = int(v)
    if len(rotated) != len(payload):
        return []
    return sorted((int(r), int(c), int(v)) for (r, c), v in rotated.items())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain a single seed payload through multiple full red separator lines.

    Each full red row/column is a global rotation axis.
    Axes are processed in visible spatial order: top-to-bottom for rows,
    left-to-right for columns. Each step rotates the current payload 90 degrees
    clockwise around the axis midpoint, and the newly rotated payload becomes the
    input to the next step.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    axes = _detect_axes(g)
    out = g.copy()
    if not axes:
        return out

    current = _initial_payload(g, int(bg))
    if not current:
        return out

    for axis in axes:
        current = _rotate_payload(current, axis, g.shape)
        if not current:
            break
        for r, c, v in current:
            out[int(r), int(c)] = int(v)
    return out


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(int(r) for r, _c in cells)
    min_c = min(int(c) for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _rotate_square4(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(3 - c), int(r)) for r, c in cells}


def _interesting_seed(cells: set[tuple[int, int]]) -> bool:
    rs = [int(r) for r, _c in cells]
    cs = [int(c) for _r, c in cells]
    if len(cells) < 4:
        return False
    if max(rs) == min(rs) or max(cs) == min(cs):
        return False
    rot = _rotate_square4(cells)
    if cells == rot:
        return False
    if cells == {(int(3 - r), int(3 - c)) for r, c in cells}:
        return False
    return True


def _random_seed_offsets(rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(300):
        target = rng.randint(4, 7)
        start = (rng.randint(0, 3), rng.randint(0, 3))
        cells = {start}
        while len(cells) < int(target):
            frontier = set()
            for cell in cells:
                for nxt in _neighbors4(cell):
                    rr, cc = nxt
                    if not (0 <= int(rr) < 4 and 0 <= int(cc) < 4):
                        continue
                    if nxt not in cells:
                        frontier.add((int(rr), int(cc)))
            if not frontier:
                break
            cells.add(rng.choice(sorted(frontier)))
        norm = _normalize(cells)
        if any(int(r) > 3 or int(c) > 3 for r, c in norm):
            continue
        if not _interesting_seed(norm):
            continue
        return {(int(r + 1), int(c + 1)) for r, c in norm}
    raise RuntimeError("failed to generate an asymmetric seed shape")


def _row_abs_positions(offsets: set[tuple[int, int]], axis_row: int, center_col: int) -> set[tuple[int, int]]:
    return {(int(axis_row - dr), int(center_col + dc)) for dr, dc in offsets}


def _row_step_positions(offsets: set[tuple[int, int]], axis_row: int, center_col: int) -> set[tuple[int, int]]:
    return {(int(axis_row + dc), int(center_col + dr)) for dr, dc in offsets}


def _row_next_offsets(offsets: set[tuple[int, int]], gap: int) -> set[tuple[int, int]]:
    return {(int(gap - dc), int(dr)) for dr, dc in offsets}


def _col_abs_positions(offsets: set[tuple[int, int]], center_row: int, axis_col: int) -> set[tuple[int, int]]:
    return {(int(center_row - dr), int(axis_col - dc)) for dr, dc in offsets}


def _col_step_positions(offsets: set[tuple[int, int]], center_row: int, axis_col: int) -> set[tuple[int, int]]:
    return {(int(center_row - dc), int(axis_col + dr)) for dr, dc in offsets}


def _col_next_offsets(offsets: set[tuple[int, int]], gap: int) -> set[tuple[int, int]]:
    return {(int(dc), int(gap - dr)) for dr, dc in offsets}


def _validate_layout(
    shape: tuple[int, int],
    axes: list[tuple[str, int]],
    seed_cells: set[tuple[int, int]],
    states: list[set[tuple[int, int]]],
) -> bool:
    h, w = map(int, shape)
    occupied: set[tuple[int, int]] = set()
    for r, c in seed_cells:
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
        occupied.add((int(r), int(c)))
    for state in states:
        for r, c in state:
            if not (0 <= int(r) < h and 0 <= int(c) < w):
                return False
            if (int(r), int(c)) in occupied:
                return False
            occupied.add((int(r), int(c)))

    if axes and axes[0][0] == "row":
        axis_rows = {int(pos) for _kind, pos in axes}
        if any(int(r) in axis_rows for r, _c in occupied):
            return False
    else:
        axis_cols = {int(pos) for _kind, pos in axes}
        if any(int(c) in axis_cols for _r, c in occupied):
            return False
    return True


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    orient = rng.choice(["row", "col"])
    chain_len = rng.randint(3, 4)
    bg = int(rng.choice([c for c in range(10) if c != int(MARK)]))
    color = int(rng.choice([c for c in range(10) if c not in (int(MARK), int(bg))]))
    offsets0 = _random_seed_offsets(rng)

    if orient == "row":
        w = int(rng.choice([v for v in range(19, 28) if v % 2 == 1]))
        min_h = 6 + GAP * (int(chain_len) - 1) + OFFSET_MAX + 1
        h = rng.randint(max(18, min_h), 28)
        center_col = w // 2
        max_first = h - (GAP * (int(chain_len) - 1) + OFFSET_MAX + 2)
        first_axis = rng.randint(OFFSET_MAX + 1, max_first)
        axis_positions = [int(first_axis + GAP * i) for i in range(int(chain_len))]
        axes = [("row", int(pos)) for pos in axis_positions]

        current_offsets = set(offsets0)
        seed_cells = _row_abs_positions(current_offsets, axis_positions[0], center_col)
        states: list[set[tuple[int, int]]] = []
        for idx, axis_row in enumerate(axis_positions):
            nxt = _row_step_positions(current_offsets, axis_row, center_col)
            states.append(nxt)
            if idx + 1 < len(axis_positions):
                current_offsets = _row_next_offsets(current_offsets, GAP)
    else:
        h = int(rng.choice([v for v in range(19, 28) if v % 2 == 1]))
        min_w = 6 + GAP * (int(chain_len) - 1) + OFFSET_MAX + 1
        w = rng.randint(max(18, min_w), 28)
        center_row = h // 2
        max_first = w - (GAP * (int(chain_len) - 1) + OFFSET_MAX + 2)
        first_axis = rng.randint(OFFSET_MAX + 1, max_first)
        axis_positions = [int(first_axis + GAP * i) for i in range(int(chain_len))]
        axes = [("col", int(pos)) for pos in axis_positions]

        current_offsets = set(offsets0)
        seed_cells = _col_abs_positions(current_offsets, center_row, axis_positions[0])
        states = []
        for idx, axis_col in enumerate(axis_positions):
            nxt = _col_step_positions(current_offsets, center_row, axis_col)
            states.append(nxt)
            if idx + 1 < len(axis_positions):
                current_offsets = _col_next_offsets(current_offsets, GAP)

    if not _validate_layout((int(h), int(w)), axes, seed_cells, states):
        raise RuntimeError("invalid chain layout")

    inp = np.full((int(h), int(w)), int(bg), dtype=int)
    out = np.full((int(h), int(w)), int(bg), dtype=int)
    for kind, pos in axes:
        if kind == "row":
            inp[int(pos), :] = int(MARK)
            out[int(pos), :] = int(MARK)
        else:
            inp[:, int(pos)] = int(MARK)
            out[:, int(pos)] = int(MARK)

    for r, c in seed_cells:
        inp[int(r), int(c)] = int(color)
        out[int(r), int(c)] = int(color)
    for state in states:
        for r, c in state:
            out[int(r), int(c)] = int(color)

    meta = {
        "orient": orient,
        "chain_len": int(chain_len),
        "seed_size": int(len(seed_cells)),
        "shape": (int(h), int(w)),
        "axes": [int(pos) for _kind, pos in axes],
        "bg": int(bg),
        "color": int(color),
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        try:
            inp, out, _meta = _build_case(rng)
        except RuntimeError:
            continue
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid combo instance")
