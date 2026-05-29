# Combo variant of da515329 (index 24)
# combo_spec: D7/0+D16/2
# Name: 先延伸N步再螺旋 + 固定圈数终止

# Design analysis
# ----------------
# spec:
#   fused_rule:
#     preserved_core:
#       - 输入仍然只有一个青色十字，中心为空洞，四个臂等长，颜色仍为 8。
#       - 四个端点仍由同一个顺时针自避让螺旋机制驱动：
#         右转优先，其次直行，必要时左转；转向后走 1，普通直行走 2，
#         若长度 2 直行受阻则退化成 1。
#     fused_changes:
#       - D16/2: 每个端点不再立刻右转入螺旋，而是先沿自身臂方向继续直线延伸 N 步，
#         其中 N = 臂长。于是臂端从半径 arm 推到半径 2*arm，螺旋的第一处拐弯发生在更外层。
#       - D7/0: 螺旋不再铺满整块画布，而只允许在固定圈数的外侧环带中展开。
#         圈数由臂长直接编码：rings = arm_len。外半径 = 2*arm_len + 2*rings = 4*arm_len。
#   behavior_trace:
#     example:
#       arm_len: 2
#       original:
#         - 端点在半径 2 处立刻按右转优先展开，路径会继续缠绕直到真实边界附近，
#           之后还要依赖 parity/border 的补边收尾。
#       fused:
#         - 端点先从半径 2 沿原臂方向走到半径 4，再开始第一次顺时针拐弯。
#         - 之后只允许在内半径 4、外半径 8 之间的方形环带内继续走；
#           进入更内侧会破坏“十字周围留空”，越过更外侧则违反固定圈数终止。
#       visible_difference:
#         - 十字附近保留一块可见黑色空域，只剩四条向外伸出的直线辐条。
#         - 外围不再一直填到真实边界，而是停在 arm_len 对应的有限圈数处，边框外留黑。
#   complexity_check:
#     - 自由度只有 arm_len=1..4 四档，且它同时控制“先延伸多远”和“外圈有几层”，
#       结构性强，不需要额外颜色或隐藏编码。
#     - 不做更复杂的 arm->rings 查表，直接用 arm_len 本身编码，样例间重复充分。
#   assumptions_review:
#     - 原版用 row/col argmax 找十字中心，这里仍成立，因为输入对象没有变化。
#     - 原版 parity/border 后处理假设最外层最终碰到真实边界；融合后螺旋故意停在内部，
#       若保留这些收尾会伪造像素，因此必须整体删除。
#     - 单独加入“先延伸 N 步”后，后续螺旋仍可能回卷到十字附近，肉眼上留白不稳定；
#       因而融合实现把固定圈数解释成一个外侧有限环带，螺旋阶段不得重新进入内盒，
#       这样 D16/2 的视觉效应才会稳定可观察。
#   placement_legality:
#     - 生成器先由 arm_len 推出内外半径，再选择中心，使外半径到真实边界至少还有
#       2 格黑色缓冲。这样直线延伸和有限圈数都不会越界，也不会误触真实边界逻辑。
#     - 输出通过 solver 自检，并额外检查外圈半径、中心留白、边框留白都满足融合行为。
#   visual_observability:
#     - 人类可直接看到两层规律：
#       1. 十字四端都先直着伸出去一段，第一拐点明显比原题更远；
#       2. 螺旋只在外环出现若干层，外侧还有整圈黑边，不再填满整图。
#   visual_cleanliness:
#     - 仍只用一种前景色，避免叠色噪音。
#     - 内盒禁入让中心区域保持干净，外环终止让轮廓整齐，图像读感比“延伸后又卷回中心”更清楚。
#   new_main_assumptions_and_generator_guarantees:
#     - main() 新假设：输入是唯一的 4 臂等长十字；generate_input() 只构造这种输入。
#     - main() 新假设：螺旋阶段只在内外半径之间活动；generate_input() 通过尺寸公式与边距
#       保证这个环带完整地落在画布内。
#     - main() 的左转分支仍以真实边界为门槛；generate_input() 保证大多数样例里外环离真实边界
#       至少 2 格，因此主要终止因素始终是融合规则本身而非偶然撞边。

import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8


def _add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _extract_arms(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]]]:
    h, w = grid.shape
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while inb(r + dr, c + dc) and int(grid[r + dr, c + dc]) == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))
    return cr, cc, arms


def _rings_from_arm(arm_len: int) -> int:
    return int(arm_len)


def _extension_steps_from_arm(arm_len: int) -> int:
    return int(arm_len)


def _inner_radius_from_arm(arm_len: int) -> int:
    return arm_len + _extension_steps_from_arm(arm_len)


def _outer_radius_from_arm(arm_len: int) -> int:
    return _inner_radius_from_arm(arm_len) + 2 * _rings_from_arm(arm_len)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each arm tip outward, then let a finite clockwise spiral grow only in the outer ring band.

    Input structure:
    - bg = 0 (black). A single teal (8) cross with a 1-cell empty hub sits somewhere inside the grid.
    - All four cardinal arms are present and have the same length; that arm length controls both the pre-extension distance and the number of spiral rings.

    Rule:
    - Each tip first walks straight outward for N steps, where N equals the arm length.
    - After that, the same local spiral policy takes over: prefer turning right, otherwise go straight, otherwise turn left; turned moves use length 1 and ordinary straights use 2 unless reduced to 1 by blockage.
    - New spiral cells must keep a 1-cell orthogonal gap from existing walls, must stay inside the outer fixed-ring box, and may not re-enter the inner box enclosed by the straight extensions.

    Key insight:
    - One visible quantity in the input drives two geometric effects at once. Arm length first pushes the launch point farther away from the cross, then also sets how thick the allowed spiral shell is, so the output becomes "inner spokes + outer finite rings" instead of a border-filling maze.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    cr, cc, arms = _extract_arms(grid)
    if len(arms) != 4:
        raise ValueError("main: expected a 4-arm teal cross")

    arm_lengths = {len(cells) for _, cells in arms}
    if len(arm_lengths) != 1:
        raise ValueError("main: expected equal arm lengths")

    arm_len = next(iter(arm_lengths))
    inner_radius = _inner_radius_from_arm(arm_len)
    outer_radius = _outer_radius_from_arm(arm_len)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def within_outer_limit(cell: tuple[int, int]) -> bool:
        r, c = cell
        return cr - outer_radius <= r <= cr + outer_radius and cc - outer_radius <= c <= cc + outer_radius

    def outside_inner_keepout(cell: tuple[int, int]) -> bool:
        r, c = cell
        return max(abs(r - cr), abs(c - cc)) >= inner_radius

    def spacing_ok(
        cell: tuple[int, int],
        prev: tuple[int, int],
        start: tuple[int, int],
    ) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(
        cur: tuple[int, int],
        d: tuple[int, int],
        length: int,
        phase: str,
    ) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = _add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not within_outer_limit((rr, cc2)):
                return False
            if phase == "spiral" and not outside_inner_keepout((rr, cc2)):
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents: list[dict[str, object]] = []
    extend_steps = _extension_steps_from_arm(arm_len)
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": outward,
                "out_dir": outward,
                "extend_remaining": extend_steps,
                "done": False,
                "p": 0,
            }
        )

    def step_agent(agent: dict[str, object]) -> bool:
        if agent["done"]:
            return False

        if agent["extend_remaining"] > 0:
            if can_move(agent["cur"], agent["out_dir"], 1, "extend"):
                do_move(agent, agent["out_dir"], 1)
                agent["extend_remaining"] -= 1
                agent["dir"] = agent["out_dir"]
                agent["p"] = 0
                return True
            agent["done"] = True
            return False

        cur = agent["cur"]
        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*cur):
            dirs.append(_left(agent["dir"]))

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            if can_move(agent["cur"], nd, length, "spiral"):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1, "spiral"):
                do_move(agent, nd, 1)
                agent["p"] = 0
                agent["dir"] = nd
                return True

        agent["done"] = True
        return False

    changed = True
    guard = 0
    while changed and guard < 200000:
        guard += 1
        changed = False
        for ag in agents:
            if step_agent(ag):
                changed = True
        if all(ag["done"] for ag in agents):
            break

    return out


# --------------- generator (cx) ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_MARGIN = 2
GRID_EXTRA = 8
MAX_OUTER_RETRIES = 100
FILL_LO, FILL_HI = 0.17, 0.34


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _axial_reach(inp: np.ndarray, out: np.ndarray) -> int:
    cr = int(np.argmax(np.sum(inp == CROSS_COLOR, axis=1)))
    cc = int(np.argmax(np.sum(inp == CROSS_COLOR, axis=0)))
    best = 0
    for d in _DIRS_URDL:
        r, c = cr, cc
        reach = 0
        while 0 <= r + d[0] < out.shape[0] and 0 <= c + d[1] < out.shape[1] and out[r + d[0], c + d[1]] == CROSS_COLOR:
            r += d[0]
            c += d[1]
            reach += 1
        best = max(best, reach)
    return int(best)


def _output_metrics(inp: np.ndarray, out: np.ndarray) -> tuple[float, int, int, int, int]:
    cr = int(np.argmax(np.sum(inp == CROSS_COLOR, axis=1)))
    cc = int(np.argmax(np.sum(inp == CROSS_COLOR, axis=0)))
    arm_len = int(np.sum(inp[cr] == CROSS_COLOR) // 2)
    ys, xs = np.where(out == CROSS_COLOR)
    used_radius = int(max(np.max(np.abs(ys - cr)), np.max(np.abs(xs - cc))))
    border_ink = int(
        np.sum(out[0] == CROSS_COLOR)
        + np.sum(out[-1] == CROSS_COLOR)
        + np.sum(out[1:-1, 0] == CROSS_COLOR)
        + np.sum(out[1:-1, -1] == CROSS_COLOR)
    )
    inner_offaxis_ink = 0
    inner_radius = _inner_radius_from_arm(arm_len)
    for r, c in zip(ys, xs):
        dr = abs(int(r) - cr)
        dc = abs(int(c) - cc)
        if max(dr, dc) < inner_radius and dr != 0 and dc != 0:
            inner_offaxis_ink += 1
    return (
        float(np.mean(out == CROSS_COLOR)),
        used_radius,
        border_ink,
        _axial_reach(inp, out),
        inner_offaxis_ink,
    )


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    outer_radius = _outer_radius_from_arm(arm)
    size = 2 * outer_radius + GRID_EXTRA

    lo = outer_radius + GRID_MARGIN
    hi = size - outer_radius - GRID_MARGIN - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        try:
            out = main(inp)
        except Exception:
            continue
        fill, used_radius, border_ink, axial_reach, inner_offaxis_ink = _output_metrics(inp, out)
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if border_ink != 0:
            continue
        if used_radius != outer_radius:
            continue
        if axial_reach != _inner_radius_from_arm(arm):
            continue
        if inner_offaxis_ink != 0:
            continue
        if np.array_equal(main(inp), out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
