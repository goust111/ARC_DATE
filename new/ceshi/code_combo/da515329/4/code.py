# Combo variant of da515329 (index 4)
# combo_spec: D2/2+D7/0
# Name: 间距由臂长编码 + 固定圈数终止

# Design analysis
# a) 具体化与行为追踪
#    - 仍然只有一个青色十字输入，中心为空，四臂等长；臂长记为 L。
#    - 原题里相邻螺旋臂之间只留 1 格，且会一直长到真实边界；这里改成两处都由 L 控制：
#      1) 相邻平行臂之间留 L 格黑色；
#      2) 只允许扩张 L 圈，外侧保留黑框。
#    - 以 L=2 为例：
#      原题：第一条水平臂贴着十字外缘起跑，后续按 2/1 步长节奏把整块画布几乎铺满。
#      融合后：四个 agent 先各自向外补 1 步，把第一条横/竖臂抬到离十字主横/主竖条正好 2 格空隙的位置；
#      之后直行步长推广为 3/2（即 spacing+1 / spacing），于是同心矩形通道之间始终隔 2 格；
#      同时限制半径 = L + (L+1)*L = 8，只画 2 圈，到达该包围盒后即停止，外围黑色留白仍保留。
# b) 复杂度校准
#    - 自由度仍只有 4 档 arm_len=1..4；spacing=L、rings=L 使用同一离散量，不会引入额外映射表。
#    - 视觉差异明显：L 越大，内圈之间留白越宽，且最外层半径按 L(L+1) 级别增大，但始终停在黑框之内。
# c) 隐式假设审查
#    - 原题里 `spacing_ok` 只需豁免 predecessor，因为 spacing=1；现在 spacing=L>1 时，同一条路径尾迹在 L 格内也会被看到，
#      所以必须豁免“候选格子后方连续的本路径尾迹”，否则 agent 会被自己的起始十字或刚走出的直线错误阻塞。
#    - 原题的 parity / border 后处理依赖“最外圈顶到真实边界”；固定圈数后这一前提不再成立，所以全部删除。
#    - 原题只在 arm_len=1 时额外先向外走一步；这里若不对所有 arm_len 都先 outward 1 步，
#      第一条侧向通道会离十字主横/主竖条过近，无法满足“留白=L”的新规则。
# d) 放置合法性
#    - 生成器先由 arm_len 推出 `limit_radius = L + (L+1)*L`，再选画布大小与中心位置，
#      保证限制盒四周至少还有 2 格纯黑缓冲，因此轨迹既不会越界，也不会碰到真实边框。
# e) 视觉可观察性
#    - 人类可以直接从样例里同时读出两件事：靠近中心的首条横/竖通道与十字之间的空隙就是 L；
#      最外层终止半径也随 L 变化，但不会碰边，因此“臂长双重编码”能在 2-4 个样例中重复出现。
# f) 可视化整洁度
#    - 仍然只有一种前景色，不添加任何辅助标记；结构是规整的同心矩形螺旋，不会出现多对象互相遮挡。
# g) 变化的可推断性
#    - 这里没有颜色映射；所有可推断性都来自几何重复模式，因此避免了“颜色值本身不可见”的问题。

import random

import numpy as np

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _spacing_from_arm(arm_len: int) -> int:
    return int(arm_len)


def _rings_from_arm(arm_len: int) -> int:
    return int(arm_len)


def _limit_radius_from_arm(arm_len: int) -> int:
    spacing = _spacing_from_arm(arm_len)
    rings = _rings_from_arm(arm_len)
    return arm_len + (spacing + 1) * rings


def _extract_cross_geometry(grid: np.ndarray):
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    arm_len = max((len(cells) for _, cells in arms), default=0)
    return cr, cc, arms, arm_len


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise rectangular spirals from the tips of a teal cross, with both spacing and lifetime encoded by arm length.

    Input structure:
    - Background is black (0). The only foreground is one teal (8) plus-sign with a 1-cell hole at its center.
    - All four arms have the same length L, and that same visible length is the only parameter source for the output.

    Rule:
    - Each arm tip first takes one outward step, then launches a clockwise turtle whose initial heading is 90° right of the arm direction.
    - Turning priority stays right > straight > left, but the spacing target changes: parallel spiral lanes must stay L blank cells apart.
    - To realize that larger gap, straight runs alternate between lengths L+1 and L, while a turning step still advances by 1 cell.
    - Drawing is additionally clipped to a square limit box whose radius is determined by L and exactly L spiral rings are allowed.

    Key insight: The arm length has a double role. The same L that measures the input cross also sets the corridor spacing and the number of rings, so solvers must infer one quantity and apply it twice.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    cr, cc, arms, arm_len = _extract_cross_geometry(grid)
    if arm_len == 0 or len(arms) == 0:
        return out

    spacing = _spacing_from_arm(arm_len)
    rings = _rings_from_arm(arm_len)
    limit_radius = _limit_radius_from_arm(arm_len)
    limit_r0 = cr - limit_radius
    limit_r1 = cr + limit_radius
    limit_c0 = cc - limit_radius
    limit_c1 = cc + limit_radius

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def within_limit(cell) -> bool:
        r, c = cell
        return limit_r0 <= r <= limit_r1 and limit_c0 <= c <= limit_c1

    def trail_cells(cur, d, planned_steps: int):
        trail = {add(cur, d, k) for k in range(planned_steps)}
        back = (-d[0], -d[1])
        pos = cur
        # For spacing > 1, allow the candidate to "see" its own contiguous tail.
        for _ in range(max(0, spacing - planned_steps)):
            pos = add(pos, back, 1)
            if not inb(*pos) or out[pos] == 0:
                break
            trail.add(pos)
        return trail

    def spacing_ok(cur, d, planned_steps: int) -> bool:
        cell = add(cur, d, planned_steps)
        trail = trail_cells(cur, d, planned_steps)
        r, c = cell
        for dr, dc in _DIRS_URDL:
            for dist in range(1, spacing + 1):
                nr, nc = r + dr * dist, c + dc * dist
                if not inb(nr, nc):
                    break
                if out[nr, nc] != 0 and (nr, nc) not in trail:
                    return False
        return True

    def can_move(cur, d, length: int) -> bool:
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not within_limit((rr, cc2)):
                return False
            if not spacing_ok(cur, d, t):
                return False
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    long_step = spacing + 1
    short_step = spacing
    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first",
                "done": False,
                "p": 0,
            }
        )

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1)
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            return True

        cur = agent["cur"]
        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*cur):
            dirs.append(_left(agent["dir"]))

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (long_step if agent["p"] == 0 else short_step)
            if can_move(agent["cur"], nd, length):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == long_step and can_move(agent["cur"], nd, short_step):
                do_move(agent, nd, short_step)
                agent["p"] = 0
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
MAX_OUTER_RETRIES = 200
FILL_LO, FILL_HI = 0.10, 0.25


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _measure_launch_gap(inp: np.ndarray, out: np.ndarray) -> int:
    cr, cc, _, _ = _extract_cross_geometry(inp)
    probe_col = cc + 1
    if probe_col >= out.shape[1]:
        return -1
    above = [r for r in range(cr - 1, -1, -1) if out[r, probe_col] == CROSS_COLOR]
    if not above:
        return -1
    return cr - above[0] - 1


def _output_metrics(inp: np.ndarray, out: np.ndarray) -> tuple[float, int, int, int]:
    cr, cc, _, _ = _extract_cross_geometry(inp)
    ys, xs = np.where(out == CROSS_COLOR)
    used_radius = int(max(np.max(np.abs(ys - cr)), np.max(np.abs(xs - cc))))
    border_ink = int(
        np.sum(out[0] == CROSS_COLOR)
        + np.sum(out[-1] == CROSS_COLOR)
        + np.sum(out[1:-1, 0] == CROSS_COLOR)
        + np.sum(out[1:-1, -1] == CROSS_COLOR)
    )
    fill = float(np.mean(out == CROSS_COLOR))
    launch_gap = _measure_launch_gap(inp, out)
    return fill, used_radius, border_ink, launch_gap


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    limit_radius = _limit_radius_from_arm(arm)
    size = 2 * limit_radius + GRID_EXTRA

    lo = limit_radius + GRID_MARGIN
    hi = size - limit_radius - GRID_MARGIN - 1
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
        fill, used_radius, border_ink, launch_gap = _output_metrics(inp, out)
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if border_ink != 0:
            continue
        if used_radius != limit_radius:
            continue
        if launch_gap != arm:
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
