# Variant of 9bbf930d
# Dimension: D14 — 符号编码与图例 (symbol_instructions)
# Change: 无图例或符号编码，所有规则由网格结构隐式表达。原题在遇到彩色阻挡时，
# 通过“只有一个垂直走廊是合法的”来隐式决定转弯；本变体改为：底边 2 行放置按颜色
# 编码的小型 L 形图例，同色图例指示该颜色阻挡对应的转向偏好（CCW 或 CW）。
# 因而在真实转弯点上，生成器会故意把两个候选转口都做成合法，solver 必须先读图例。
#
# 设计分析（按题目要求先写清楚再实现）:
# 1) 具体化与行为追踪
#    - 主体迷宫仍是原题的横条纹走廊：偶数行是彩条，奇数行是粒子通道；粒子从最左列向右进入。
#    - 新增对象是“边缘图例带”：固定放在最底部 2 行。每个颜色在这两行中占一个 2x2 小块，
#      其中 3 个同色像素构成一个 L 形符号：
#        * {(0,0),(1,0),(1,1)} 形状表示 CCW
#        * {(0,1),(1,0),(1,1)} 镜像形状表示 CW
#      这样人眼能直接看到“同色小拐角”，solver 也能稳定解析。
#    - 原题在一个水平通道里遇到阻挡色块时，只会存在一个可转的垂直口；本变体则把上/下两个口都打开，
#      然后按阻挡颜色的图例决定是向上还是向下（从右行进时：CCW->上，CW->下）。
#      若偏好方向不存在，粒子停止；不会再自动退而求其次选另一边。
#    - 小例子：粒子在某行向右，到达 c_v-1，前方 (r,c_v) 是颜色 4 的阻挡。原题这里必须只有上口或下口
#      之一为真；本题中两者都为真，若底边颜色 4 的 L 形是 CCW，则转到上口，否则转到下口。
#
# 2) 复杂度校准
#    - 图例只有两类离散含义（CW/CCW），自由度很小，不会让规则过度发散。
#    - 同一颜色会在若干条真实走廊上重复出现，给人类提供“同色=同转向偏好”的重复支撑。
#    - 生成器保留直行/L/U 三类轨迹，避免变成单一一跳题；但图例只接管“转哪边”，不改其余几何。
#
# 3) 对原 main() 的逐段隐式假设审查
#    - _has_corridor()：原本只判断“某个候选口是不是合法走廊”。这个前提仍成立，直接复用。
#    - _trace()：原本依赖“len(options)<=1”，即结构本身消除歧义；本变体不成立，因为我们故意制造双口。
#      所以这里必须改成：读取前方阻挡色 -> 查图例 -> 算出 CW/CCW 后的目标方向 -> 只尝试那个方向。
#    - main() 的入口检测：原本假设整张网格都属于迷宫。现在底边 2 行是图例带，所以 solver 必须先裁出
#      迷宫区再跑原逻辑；输出时图例带保持不变。
#    - main() 最终写回粒子位置：仍然只在迷宫区内写，不能污染底边图例带。
#
# 4) 放置合法性
#    - 图例带与迷宫彻底分离：总高度=迷宫高度+2；迷宫只在前 H-2 行构造，底边 2 行只放图例。
#    - 竖向通道仍使用原生成器的 vcol 预留窗口，避免不同转弯共用同一列导致串线。
#    - 新增的“未被图例选中的假支路”只在源转角处开口，不继续向外雕刻；因此它至多前进一小段就被下一条彩条堵住，
#      不会跨区域干扰别的通道。
#
# 5) 视觉可观察性
#    - 变体的关键变化是“同样的彩色阻挡，在不同样例里可以因为边缘图例不同而选择相反转法”。
#      这能从 2-3 组样例中肉眼观察出来；且图例和主体同色绑定，不依赖颜色 ID 的数值大小。
#    - 生成器不再让结构唯一决定方向，而是稳定地产生若干“上下都能转”的节点，保证 D14 行为实际发生。
#
# 6) 可视化整洁度
#    - 图例统一在底边 2 行，不与走廊叠层。
#    - 每个图例条目就是一个 2x2 的单色 L 形，背景仍是原背景色 7；不会引入额外装饰噪声。
#
# 7) 颜色变化可推断性
#    - 没有做颜色映射；颜色只承担“哪个图例条目控制哪个阻挡”的绑定作用。
#    - 每种颜色只有两种可能符号，且同色在一个样例里含义恒定。
#
# 8) 变体行为量化指标
#    - 代码里提供 _count_legend_controlled_junctions()：统计迷宫中“前方被颜色阻挡，且两条垂直候选口都合法，
#      并且该颜色在图例中有定义”的节点数。这个值 > 0 才说明样本真的依赖图例。
#
# 9) 新 solver 的隐式假设审查
#    - 假设 A：底边正好 2 行是图例带。generate_input() 始终按此构造。
#    - 假设 B：每个图例颜色在底边形成一个 4 连通、3 像素、2x2 bbox 的 L 形。生成器严格保证。
#    - 假设 C：真实转弯遇到的阻挡颜色都在图例里。生成器给所有迷宫条纹颜色都放图例。
#    - 假设 D：未选支路不会和别的竖向走廊打通。vcol 预留窗口保证同列不复用。

from __future__ import annotations

import random
import numpy as np

BG = 7
PARTICLE = 6
LEGEND_H = 2

_DELTAS = {
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
    "up": (-1, 0),
}
_DIR_ORDER = ["up", "right", "down", "left"]
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}
_TURN_FROM_SHAPE = {
    frozenset({(0, 0), (1, 0), (1, 1)}): "ccw",
    frozenset({(0, 1), (1, 0), (1, 1)}): "cw",
}


def _rotate(direction: str, turn: str) -> str:
    idx = _DIR_ORDER.index(direction)
    step = -1 if turn == "ccw" else 1
    return _DIR_ORDER[(idx + step) % 4]


def _same_color_component(legend: np.ndarray, start_r: int, start_c: int, seen: np.ndarray) -> list[tuple[int, int]]:
    color = int(legend[start_r, start_c])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    cells: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for dr, dc in _DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < legend.shape[0] and 0 <= nc < legend.shape[1] and not seen[nr, nc]:
                if int(legend[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return cells


def _parse_legend(legend: np.ndarray) -> dict[int, str]:
    """Read 2x2 bottom-edge L-shape symbols as color -> {ccw,cw}."""
    assert legend.shape[0] == LEGEND_H, "generator always builds a 2-row legend band"
    seen = np.zeros_like(legend, dtype=bool)
    turn_map: dict[int, str] = {}
    for r in range(legend.shape[0]):
        for c in range(legend.shape[1]):
            color = int(legend[r, c])
            if color == BG or seen[r, c]:
                continue
            comp = _same_color_component(legend, r, c, seen)
            if len(comp) != 3:
                continue
            rs = [rr for rr, _ in comp]
            cs = [cc for _, cc in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if (r0, r1) != (0, 1) or c1 - c0 != 1:
                continue
            shape = frozenset((rr - r0, cc - c0) for rr, cc in comp)
            turn = _TURN_FROM_SHAPE.get(shape)
            if turn is not None:
                turn_map[color] = turn
    return turn_map


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True iff (r,c) is a passable cell flanked by same-color bars perpendicular to direction."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {PARTICLE, BG}


def _trace(grid: np.ndarray, turn_map: dict[int, str], r: int, c: int) -> tuple[int, int]:
    """
    Trace one particle.

    Difference from the original task:
    - We no longer assert that at most one perpendicular turn exists.
    - When blocked by color C, we compute the preferred 90-degree turn from the bottom legend:
      CCW means rotate current heading one step left in URDL order; CW means one step right.
    - If that preferred side is not a legal corridor, the particle stops even if the other side exists.
    """
    H, W = grid.shape
    direction = "right"
    seen: set[tuple[int, int, str]] = set()
    for _ in range(600):
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        ahead = int(grid[nr, nc])
        if ahead in {PARTICLE, BG}:
            r, c = nr, nc
            continue

        turn = turn_map.get(ahead)
        if turn is None:
            break
        target_dir = _rotate(direction, turn)
        if target_dir not in _PERPENDICULAR[direction]:
            break
        tdr, tdc = _DELTAS[target_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, target_dir):
            r, c, direction = tr, tc, target_dir
            continue
        break
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through stripe corridors using bottom-edge symbol instructions."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    assert H > LEGEND_H + 2

    maze = g[:-LEGEND_H, :]
    legend = g[-LEGEND_H:, :]
    turn_map = _parse_legend(legend)

    out = g.copy()
    for r in range(1, maze.shape[0] - 1):
        if int(maze[r, 0]) != PARTICLE:
            continue

        entrance_col = None
        for c in range(2, W):
            if int(maze[r, c]) != BG:
                continue
            above, below = int(maze[r - 1, c]), int(maze[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_color = int(maze[r - 1, entrance_col])
        bar_above = next((c for c in range(W) if int(maze[r - 1, c]) == bar_color), W)
        bar_below = next((c for c in range(W) if int(maze[r + 1, c]) == bar_color), W)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(maze, turn_map, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_PLAY_HEIGHT_OPTIONS = [17, 19, 21, 23]     # original maze heights
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_LEGEND_CONTROLLED = 1
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
    turn_pref: dict[int, str],
) -> None:
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)

    for color, chans in by_color.items():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
        for p in range(n_pairs):
            a, b = sorted(chans[2 * p:2 * p + 2])
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                break
            internal_stop = random.random() < _U_INTERNAL_STOP_PROB
            if turn_pref[color] == "ccw":
                source, target, direction = b, a, "up"
            else:
                source, target, direction = a, b, "down"
            plans[source] = {
                "kind": "u",
                "target": target,
                "direction": direction,
                "c_v": c_v,
                "internal_stop": internal_stop,
            }
            plans[target] = {"kind": "u_target"}


def _plan_remaining(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
    turn_pref: dict[int, str],
) -> None:
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        color = bar_color[bar_rows[i]]
        direction = "up" if turn_pref[color] == "ccw" else "down"
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    """Render an L-turn whose chosen branch follows the legend, while the opposite side remains a short valid decoy."""
    g[row, c_v] = color

    # Open both immediate turn mouths so geometry alone is ambiguous.
    for br in (top, bot):
        g[br, c_v - 1] = BG
        g[br, c_v - 2] = bar_color[br]
        g[br, c_v] = bar_color[br]

    if direction == "up":
        for r_br in range(0, top, 2):
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g: np.ndarray,
    H: int,
    W: int,
    top: int,
    bot: int,
    row: int,
    color: int,
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
    internal_stop: bool,
) -> None:
    """
    Render a U-turn.

    The first decision point is legend-controlled because both source mouths are open.
    The second turn (vertical -> horizontal re-entry) remains structurally unique as in the original task.
    """
    g[row, c_v] = color

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = BG
        g[top, c_v - 2] = bar_color[top]
        g[top, c_v] = bar_color[top]
        g[bot, c_v - 2] = bar_color[bot]
        g[bot, c_v] = bar_color[bot]
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = BG
        g[top, c_v - 2] = bar_color[top]
        g[top, c_v] = bar_color[top]
        g[bot, c_v - 2] = bar_color[bot]
        g[bot, c_v] = bar_color[bot]
        g[j_bot, c_v - 2] = BG

    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    extras: list[int],
) -> None:
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if kind in {"l_edge", "l_internal"}:
            stop_at = None
            direction = str(plan["direction"])
            if kind == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        else:
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g, H, W, top, bot, row, color, c_v, str(plan["direction"]),
                j_top, j_bot, bar_color, bar_start, extras, bool(plan.get("internal_stop", False))
            )


def _place_legend(g: np.ndarray, maze_h: int, colors: list[int], turn_pref: dict[int, str]) -> None:
    band = g[maze_h:, :]
    band[:, :] = BG

    legend_colors = colors[:]
    random.shuffle(legend_colors)
    total_w = len(legend_colors) * 2 + max(0, len(legend_colors) - 1)
    max_start = g.shape[1] - total_w
    start = 0 if max_start <= 0 else random.randint(0 if max_start == 0 else 1, max_start)
    c = start
    for color in legend_colors:
        shape = {(0, 0), (1, 0), (1, 1)} if turn_pref[color] == "ccw" else {(0, 1), (1, 0), (1, 1)}
        for dr, dc in shape:
            band[dr, c + dc] = color
        c += 3


def _count_legend_controlled_junctions(grid: np.ndarray) -> int:
    maze = grid[:-LEGEND_H, :]
    turn_map = _parse_legend(grid[-LEGEND_H:, :])
    H, W = maze.shape
    count = 0
    for r in range(H):
        for c in range(W):
            if int(maze[r, c]) not in {BG, PARTICLE}:
                continue
            for direction, (dr, dc) in _DELTAS.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                ahead = int(maze[nr, nc])
                if ahead in {BG, PARTICLE} or ahead not in turn_map:
                    continue
                options = 0
                for turn_dir in _PERPENDICULAR[direction]:
                    tdr, tdc = _DELTAS[turn_dir]
                    tr, tc = r + tdr, c + tdc
                    if 0 <= tr < H and 0 <= tc < W and _has_corridor(maze, tr, tc, turn_dir):
                        options += 1
                if options == 2:
                    count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    maze_h = random.choice(_PLAY_HEIGHT_OPTIONS)
    H = maze_h + LEGEND_H
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    maze = g[:maze_h, :]
    maze[:, 0] = PARTICLE

    bar_rows = list(range(0, maze_h, 2))
    n_channels = len(bar_rows) - 1

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = random.choice([c for c in extras if c != bar_color[top]])

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    turn_pref = {color: random.choice(["ccw", "cw"]) for color in set(bar_color.values())}

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        maze[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans, turn_pref)
    _plan_remaining(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans, turn_pref)
    _execute_plans(maze, maze_h, W, plans, bar_rows, bar_color, bar_start, extras)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if int(maze[r, c_decor]) == BG:
                maze[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    maze[:, 0] = PARTICLE
    maze[:, 1] = BG
    _place_legend(g, maze_h, sorted(set(bar_color.values())), turn_pref)

    try:
        out = main(g)
    except Exception:
        return None

    n_moved = sum(
        1 for r in range(maze_h)
        if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if _count_legend_controlled_junctions(g) < _MIN_LEGEND_CONTROLLED:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
