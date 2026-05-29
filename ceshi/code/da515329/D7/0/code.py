# Variant of da515329
# Dimension: D7 — 终止条件 (dst)
# Change: 螺旋一直展开直到网格空间被完全占据（无空白可继续画）... → 固定圈数终止

# Design analysis
# spec:
#   variant_core:
#     preserved_mechanism:
#       - 输入仍然只有一个青色十字，中心为空洞，四臂等长。
#       - 四个 agent 仍从四个臂端同时出发，仍按“右转优先，其次直行，必要时左转”的局部规则行走。
#       - 步长规则与原题一致：转向后走 1，普通直行走 2，长度 2 被阻挡时仍允许退化成 1。
#     changed_termination:
#       - 不再以“直到真实网格无处可走”为终止条件。
#       - 改为“只允许在一个以十字中心为空洞、按圈数展开得到的最大包围盒内作画”；一旦下一步越出这个包围盒，该步就非法。
#       - 圈数由臂长直接编码：1->1, 2->2, 3->3, 4->4。
#       - 包围盒半径 = arm_len + 2 * rings，因此每多 1 圈，四向外扩半径恰好多 2 像素，和原螺旋的 1 像素留缝节奏一致。
#   behavior_trace:
#     example:
#       arm_len: 2
#       original:
#         - 在 16x16 左右的画布上，螺旋会继续向外扩，直到接触真实边界并触发补边/修边逻辑。
#       variant:
#         - 同样从四个臂端开始扩张，但只要候选步会越出半径 6 的限制盒，就停止接受该步。
#         - 因而螺旋只形成 2 层同心矩形通道，外圈之外保留整片黑色留白。
#       difference:
#         - 差异发生在 solver 的“can_move 判定”环节，而不是事后裁剪已有输出。
#   complexity_check:
#     - 圈数只有 4 个档位，离散且可见，不会引入过多自由度。
#     - 输出仍是单色几何图案，但外侧留白宽度与十字臂长同步变化，2-4 组样例足以观察“臂越长，圈越多”。
#   assumptions_review:
#     - row/col 的 argmax 仍能恢复中心空洞，因为输入唯一的 8 仍是十字四臂。
#     - 原题的 parity/border 后处理依赖“最外层会碰到真实边界”，在本变体中不再成立；若保留会错误补边，破坏外围留白，因此删除。
#     - 左转仅在真实边界外禁用的写法仍然无害；生成器保证限制盒离真实边界至少 2 格，所以绝大多数运动都由限制盒而非真实边界终止。
#   placement_legality:
#     - 生成器先按 arm 计算 rings 和限制盒半径，再选择中心位置，使限制盒到四条真实边界都至少有 2 格黑色缓冲。
#     - 这样输出既不会越界，也一定留下可观察的外围空白带。
#   visual_cleanliness:
#     - 保持单一颜色和原始路径连通性，避免添加额外标记。
#     - 通过固定外侧黑色缓冲区，让“停在第几圈”能直接从图像外轮廓读出。

import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _rings_from_arm(arm_len: int) -> int:
    return int(arm_len)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spiral wall paths from the tips of a teal cross, but stop after a fixed number of rings.

    Input structure:
    - bg = 0 (black). A single teal (8) cross with a 1-cell hole at the center.
    - All four arms have the same length; that arm length encodes how many spiral rings may be drawn.

    Rule:
    - Each arm tip launches a turtle heading 90° clockwise from its outward direction.
    - Movement priority remains right turn > straight > left turn (left only if not on the true border).
    - Normal straight steps use length 2; a move immediately after a turn uses length 1.
    - A candidate step is legal only if it keeps a 1-cell orthogonal gap from existing walls and stays inside the ring-limit box implied by the arm length.

    Key insight: The local spiral rule is unchanged; only the stopping condition changes. Arm length tells you how large the allowed expansion box is, so the same self-avoiding spiral stalls early and leaves a clean black outer frame.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == 8:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    arm_len = max((len(cells) for _, cells in arms), default=0)
    ring_count = _rings_from_arm(arm_len)
    limit_radius = arm_len + 2 * ring_count
    limit_r0 = cr - limit_radius
    limit_r1 = cr + limit_radius
    limit_c0 = cc - limit_radius
    limit_c1 = cc + limit_radius

    def within_limit(cell) -> bool:
        r, c = cell
        return limit_r0 <= r <= limit_r1 and limit_c0 <= c <= limit_c1

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not within_limit((rr, cc2)):
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = 8
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
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
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            if can_move(agent["cur"], nd, length):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                do_move(agent, nd, 1)
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
CROSS_COLOR = 8
FILL_LO, FILL_HI = 0.10, 0.34


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _limit_radius_from_arm(arm: int) -> int:
    return arm + 2 * _rings_from_arm(arm)


def _output_metrics(inp: np.ndarray, out: np.ndarray) -> tuple[float, int, int]:
    cr = int(np.argmax(np.sum(inp == CROSS_COLOR, axis=1)))
    cc = int(np.argmax(np.sum(inp == CROSS_COLOR, axis=0)))
    ys, xs = np.where(out == CROSS_COLOR)
    used_radius = int(max(np.max(np.abs(ys - cr)), np.max(np.abs(xs - cc))))
    border_ink = int(
        np.sum(out[0] == CROSS_COLOR)
        + np.sum(out[-1] == CROSS_COLOR)
        + np.sum(out[1:-1, 0] == CROSS_COLOR)
        + np.sum(out[1:-1, -1] == CROSS_COLOR)
    )
    return float(np.mean(out == CROSS_COLOR)), used_radius, border_ink


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    limit_radius = _limit_radius_from_arm(arm)
    size = 2 * limit_radius + GRID_EXTRA

    lo = limit_radius + GRID_MARGIN
    hi = size - limit_radius - GRID_MARGIN - 1
    if lo > hi:
        return None

    candidates = [
        (r, c)
        for r in range(lo, hi + 1)
        for c in range(lo, hi + 1)
        if (r + c) % 2 == 0
    ]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        try:
            out = main(inp)
        except Exception:
            continue
        fill, used_radius, border_ink = _output_metrics(inp, out)
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if border_ink != 0:
            continue
        if used_radius < limit_radius - 1:
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
