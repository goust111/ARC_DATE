# Combo variant of 9bbf930d (index 1)
# combo_spec: D2/1+D9/2
# Name: 障碍色转向+交叉口信号灯

from __future__ import annotations

import random

import numpy as np

"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 保留原题的条纹几何：偶数行是彩条，奇数行是水平通道，col 0 全是粒子，col 1 保持背景。
   - 融合后的决策分成两层：
       1. 普通挡块层：当前进一步若撞到彩色挡块，则该挡块颜色决定相对转向。
          本实现固定两种挡块提示色：
            * 2 = 逆时针转 90 度（right->up, up->left, left->down, down->right）
            * 3 = 顺时针转 90 度（right->down, down->left, left->up, up->right）
       2. 信号灯层：如果前方一步是特殊信号像素，则它先决定“此交叉口是直行还是转弯”。
            * 8 = 在该交叉口转弯
            * 9 = 在该交叉口直行
   - 关键融合点放成“先有信号、后有挡块”的串联结构：
       * 粒子向右时，先经过一个带侧支路的信号像素。
       * 若信号是 9，粒子忽略侧支，继续向右，随后再遇到颜色 2/3 的挡块，并按挡块颜色拐弯。
       * 若信号是 8，粒子在信号处就转入侧支，不再走到后面的挡块。
   - 具体小例子：
       * 原题：... BG BG [彩色挡块] ...，只有撞到挡块时才看上下哪边可走。
       * D2 单变体：... BG BG [2] ...，即使上下两侧都开口，也由 2/3 决定上拐还是下拐。
       * 融合后：... BG [9] BG [2] ...，9 让粒子先直行，随后 2 再决定上拐；
                  ... BG [8] ... 则在 8 这个交叉口直接转入唯一侧支，后面的挡块即使存在也不触发。
   - 差异因此是显式可见的：同样是 T 字交叉口，8/9 决定“转不转”；而 2/3 决定“若转，则朝哪边转”。

b) 复杂度校准
   - 只引入四个有语义的新颜色：两个挡块提示色 + 两个信号色。条色仍只用两种。
   - 这样每个样例总共 8 种颜色，虽然比原题多，但语义分工清晰：
       * 0/5: 走廊条色
       * 2/3: 挡块转向语义
       * 8/9: 交叉口信号语义
       * 6/7: 粒子/背景
   - generator 强制每个样例同时出现：
       * 至少一个 8 事件和一个 9 事件
       * 至少一个颜色 2 挡块和一个颜色 3 挡块
       * 至少一条路径会先过 9 再撞到 2/3，确保两套规则在同一路径上协同出现，而不是互不相关地分散在不同粒子上。

c) 隐式假设审查（main 相关）
   - 原 _has_corridor 假设通道中心只能是 BG/PARTICLE；D9 融合后信号像素本身也必须可视为可通行，否则粒子会把信号灯当墙。
   - 原 _trace 看到任何非 BG 颜色都当作挡块；现在必须先分流处理信号色，否则 8/9 完全不起作用。
   - 原题/原 solver 依赖“挡块处最多只有一个垂直选项”；D2 融合后我故意在挡块列把上下都开口，因此不能再用唯一性断言。
   - 新增假设 1：若信号色为 8，则信号交叉口必须存在可转入的合法侧支。
     generator 用唯一开口支路保证这一点。
   - 新增假设 2：若要让 9 事件和后续挡块同路出现，则信号列之后必须仍有水平前进空间。
     generator 在分配 `signal_then_blocker` 计划时保证 `signal_x < blocker_x - 3`。
   - 新增假设 3：若信号处要借前方挡块颜色决定转向，则信号前方一格必须真的是 2/3。
     solver 支持这个协同逻辑，但 generator 只把它作为可选增强，不把正确性建立在该条件上。

d) 放置合法性
   - 所有竖直结构统一按 3 列窗口预留，避免不同交叉口的竖直通道彼此贴边重叠。
   - 挡块转弯使用原题式几何：挡块在 `c_v`，真正竖直通道在 `c_v-1`，上下都先开一个可选入口，再只向被选择方向继续延展。
   - 信号交叉口使用更直接的 D9 式几何：信号像素就在奇数行通道单元，只有一个上下入口被打开，因此“转弯”是唯一可执行动作。
   - `signal_then_blocker` 行会同时占用两个相距足够远的列，避免信号支路与后续挡块支路互相覆盖。
   - 所有内部终止位置都选在合法偶数行上，不越界。

e) 视觉可观察性
   - 解题者可以直接在像素层观察到：
       * 8/9 都出现在“有侧支但主路未断”的位置，且一类会拐入侧支、另一类会继续直行。
       * 2/3 都出现在“主路被挡住且上下都可尝试”的位置，输出粒子会稳定地朝不同方向离开。
   - 样例里至少有一条粒子路径先经过 9 再撞到 2/3，因此人类能看到“先看信号决定是否进交叉口，再看挡块颜色决定向哪边转”。
   - 这里没有使用颜色 id 的数值大小、排序或黑色特殊地位；全部规律都由重复的局部几何和输出终点共同支持。

f) 可视化整洁度
   - 条纹主体仍然规整；所有新语义都只占单像素（信号）或单挡块（2/3）。
   - 信号和挡块使用与条色完全不同的颜色，避免“这是墙还是标记”的角色混淆。
   - 每个真实通道只放 0-2 个关键事件，且关键列有全局间隔，避免整张图出现密集交织的十字路口。

g) 变化的可推断性
   - 挡块色映射只有两类，信号灯也只有两类，且每类在每个有效样本中都至少被访问一次。
   - 颜色意义是固定的，不在样例间置换，因此少量示例就足以归纳：
       * 8/9 控制“是否在交叉口转”
       * 2/3 控制“若需转，则是逆时针还是顺时针”

9) 对新 main() 假设的逐项复查
   - 入口扫描仍只把“横向同色条对夹着的空格”当作真实入口；generator 保持左侧入口纯净，不把信号/挡块放在最左几列。
   - `_choose_signal_turn()` 假设信号交叉口若有多个候选方向，则前方挡块色可消歧；generator 当前主要生成唯一侧支，所以这个回退分支只是容错。
   - `_turn_from_blocker()` 假设真正被活动路径撞到的有语义挡块只会是 2 或 3；generator 不会让活动路径正撞到普通条色。
   - 对于双开口挡块，solver 会只检查颜色指定的一侧是否为合法走廊；generator 保证该侧至少有一步合法 corridor。
"""

BG = 7
PARTICLE = 6
BAR_COLORS = (0, 5)
BLOCK_CCW = 2
BLOCK_CW = 3
TURN_SIGNAL = 8
STRAIGHT_SIGNAL = 9

_TRAVERSABLE = {BG, PARTICLE, TURN_SIGNAL, STRAIGHT_SIGNAL}
_BLOCKER_CUES = {BLOCK_CCW, BLOCK_CW}
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}
_CCW_TURN = {"right": "up", "up": "left", "left": "down", "down": "right"}
_CW_TURN = {"right": "down", "down": "left", "left": "up", "up": "right"}


def _is_bar_color(value: int) -> bool:
    return value in BAR_COLORS


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is traversable and flanked by matching colored bars."""
    H, W = grid.shape
    if int(grid[r, c]) not in _TRAVERSABLE:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and _is_bar_color(a)


def _turn_from_blocker(color: int, direction: str) -> str | None:
    if color == BLOCK_CCW:
        return _CCW_TURN[direction]
    if color == BLOCK_CW:
        return _CW_TURN[direction]
    return None


def _cue_color(from_dir: str, to_dir: str) -> int:
    if _CCW_TURN[from_dir] == to_dir:
        return BLOCK_CCW
    if _CW_TURN[from_dir] == to_dir:
        return BLOCK_CW
    raise ValueError(f"invalid cue turn {from_dir}->{to_dir}")


def _turn_options_from(grid: np.ndarray, r: int, c: int, direction: str) -> list[str]:
    H, W = grid.shape
    options: list[str] = []
    for turn_dir in _PERPENDICULAR[direction]:
        dr, dc = _DELTAS[turn_dir]
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and _has_corridor(grid, nr, nc, turn_dir):
            options.append(turn_dir)
    return options


def _choose_signal_turn(grid: np.ndarray, r: int, c: int, direction: str) -> str | None:
    """Choose the turn direction after stepping onto a TURN_SIGNAL cell."""
    options = _turn_options_from(grid, r, c, direction)
    dr, dc = _DELTAS[direction]
    nr, nc = r + dr, c + dc
    H, W = grid.shape
    if 0 <= nr < H and 0 <= nc < W:
        hinted = _turn_from_blocker(int(grid[nr, nc]), direction)
        if hinted in options:
            return hinted
    if len(options) == 1:
        return options[0]
    return None


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle under fused signal-first and blocker-color turning rules."""
    H, W = grid.shape
    direction = "right"
    for _ in range(700):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break

        nxt = int(grid[nr, nc])
        if nxt == STRAIGHT_SIGNAL:
            r, c = nr, nc
            continue
        if nxt == TURN_SIGNAL:
            turn_dir = _choose_signal_turn(grid, nr, nc, direction)
            if turn_dir is None:
                break
            r, c = nr, nc
            direction = turn_dir
            continue
        if nxt in {BG, PARTICLE}:
            r, c = nr, nc
            continue

        turn_dir = _turn_from_blocker(nxt, direction)
        if turn_dir is None:
            break
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
            r, c = tr, tc
            direction = turn_dir
            continue
        break
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through striped corridors with fused local instructions.

    Input structure:
    - bg=7, particle=6. Even rows are colored bars; odd rows are horizontal channels.
    - Signal pixels 8/9 can appear on traversable channel cells at cross-junctions.
    - Blocker cue colors 2/3 can appear ahead of a corridor to encode relative turn direction.

    Rule:
    - A particle starts in col 0 and moves right through traversable cells.
    - If the next cell is signal color 9, it steps onto that cell and keeps going straight.
    - If the next cell is signal color 8, the particle turns at that junction; signal logic is
      checked before any blocker logic.
    - Otherwise, if the next cell is blocker color 2 or 3, that color determines whether the
      particle turns counter-clockwise or clockwise, provided the target side is a legal corridor.
    - The particle stops at the edge or when the instructed move is impossible.

    Key insight:
    - The task separates two decisions that used to be conflated: signal colors decide whether
      a junction is taken, while blocker colors decide which way a forced 90-degree turn points.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and _is_bar_color(above):
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_above = next(
            (c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])),
            W,
        )
        bar_below = next(
            (c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])),
            W,
        )
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (15, 21)
_BAR_START_OPTIONS = [2, 3]
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_FEATURE_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)
_MIN_MOVED = 4
_MAX_MOVED = 6
_MAX_RETRIES = 500
_L_INTERNAL_PROB = 0.35
_SIGNAL_INTERNAL_PROB = 0.35


def _other_bar_color(color: int) -> int:
    return BAR_COLORS[1] if color == BAR_COLORS[0] else BAR_COLORS[0]


def _reserve_cols(used_cols: set[int], *cols: int) -> None:
    for col in cols:
        used_cols.add(col)


def _allocate_signal_col(
    bar_start: int,
    W: int,
    used_cols: set[int],
    min_col: int | None = None,
    max_col: int | None = None,
) -> int | None:
    c_min = max(bar_start + 3, min_col if min_col is not None else bar_start + 3)
    c_max = min(W - 3, max_col if max_col is not None else W - 3)
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_cols for d in _FEATURE_WINDOW)
    ]
    if not candidates:
        return None
    x = random.choice(candidates)
    _reserve_cols(used_cols, x - 1, x, x + 1)
    return x


def _allocate_blocker_col(
    bar_start: int,
    W: int,
    used_cols: set[int],
    min_col: int | None = None,
    max_col: int | None = None,
) -> int | None:
    c_min = max(bar_start + 4, min_col if min_col is not None else bar_start + 4)
    c_max = min(W - 2, max_col if max_col is not None else W - 2)
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_cols for d in _FEATURE_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    _reserve_cols(used_cols, c_v - 1, c_v, c_v + 1)
    return c_v


def _pick_internal_stop(direction: str, top: int, bot: int, H: int) -> int | None:
    if direction == "up":
        cands = list(range(2, top, 2))
    else:
        cands = list(range(bot + 2, H - 1, 2))
    return random.choice(cands) if cands else None


def _paint_base_grid(H: int, W: int, bar_start: int, channel_spec: list[str]):
    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))

    bar_color = {bar_rows[0]: random.choice(BAR_COLORS)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = _other_bar_color(bar_color[top])

    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]
    return g, bar_rows, bar_color


def _extend_vertical(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    x: int,
    direction: str,
    stop_at: int | None = None,
) -> None:
    if direction == "up":
        for r_br in range(0, top + 1, 2):
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, x] = BG
    else:
        for r_br in range(bot, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, x] = BG


def _open_dual_blocker_junction(
    g: np.ndarray,
    top: int,
    bot: int,
    c_v: int,
    bar_color: dict[int, int],
) -> None:
    x = c_v - 1
    g[top, x] = BG
    g[bot, x] = BG
    g[top, c_v - 2] = bar_color[top]
    g[top, c_v] = bar_color[top]
    g[bot, c_v - 2] = bar_color[bot]
    g[bot, c_v] = bar_color[bot]


def _build_blocker_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    bar_color: dict[int, int],
    c_v: int,
    direction: str,
    stop_at: int | None = None,
) -> None:
    g[row, c_v] = _cue_color("right", direction)
    _open_dual_blocker_junction(g, top, bot, c_v, bar_color)
    _extend_vertical(g, H, top, bot, c_v - 1, direction, stop_at)


def _build_signal_branch(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    x: int,
    signal_color: int,
    direction: str,
    stop_at: int | None = None,
) -> None:
    g[row, x] = signal_color
    if direction == "up":
        g[top, x] = BG
    else:
        g[bot, x] = BG
    _extend_vertical(g, H, top, bot, x, direction, stop_at)


def _plan_rows(real_indices: list[int]) -> dict[int, dict[str, object]]:
    rows = random.sample(real_indices, len(real_indices))
    plans: dict[int, dict[str, object]] = {}
    plans[rows[0]] = {"kind": "blocker", "direction": "up"}
    plans[rows[1]] = {"kind": "blocker", "direction": "down"}
    plans[rows[2]] = {"kind": "signal_turn"}
    plans[rows[3]] = {"kind": "signal_then_blocker"}
    for idx in rows[4:]:
        choice = random.choices(
            ["straight", "blocker", "signal_turn", "signal_then_blocker"],
            weights=[1, 3, 2, 2],
            k=1,
        )[0]
        if choice == "blocker":
            plans[idx] = {"kind": "blocker", "direction": random.choice(["up", "down"])}
        else:
            plans[idx] = {"kind": choice}
    return plans


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    bar_start: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
) -> bool:
    used_cols: set[int] = set()
    for idx, plan in plans.items():
        top, bot = bar_rows[idx], bar_rows[idx + 1]
        row = top + 1
        kind = plan["kind"]

        if kind == "straight":
            continue

        if kind == "blocker":
            c_v = _allocate_blocker_col(bar_start, W, used_cols)
            if c_v is None:
                return False
            stop_at = None
            if random.random() < _L_INTERNAL_PROB:
                stop_at = _pick_internal_stop(str(plan["direction"]), top, bot, H)
            _build_blocker_turn(
                g,
                H,
                top,
                bot,
                row,
                bar_color,
                c_v,
                str(plan["direction"]),
                stop_at,
            )
            continue

        if kind == "signal_turn":
            x = _allocate_signal_col(bar_start, W, used_cols)
            if x is None:
                return False
            direction = random.choice(["up", "down"])
            stop_at = None
            if random.random() < _SIGNAL_INTERNAL_PROB:
                stop_at = _pick_internal_stop(direction, top, bot, H)
            _build_signal_branch(g, H, top, bot, row, x, TURN_SIGNAL, direction, stop_at)
            continue

        if kind == "signal_then_blocker":
            x = _allocate_signal_col(bar_start, W, used_cols, max_col=W - 7)
            if x is None:
                return False
            signal_dir = random.choice(["up", "down"])
            _build_signal_branch(g, H, top, bot, row, x, STRAIGHT_SIGNAL, signal_dir, None)
            c_v = _allocate_blocker_col(bar_start, W, used_cols, min_col=x + 4)
            if c_v is None:
                return False
            blocker_dir = random.choice(["up", "down"])
            stop_at = None
            if random.random() < _L_INTERNAL_PROB:
                stop_at = _pick_internal_stop(blocker_dir, top, bot, H)
            _build_blocker_turn(
                g,
                H,
                top,
                bot,
                row,
                bar_color,
                c_v,
                blocker_dir,
                stop_at,
            )
            continue

        return False
    return True


def _iter_active_rows(g: np.ndarray):
    H, W = g.shape
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and _is_bar_color(above):
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), W)
        if bar_above == bar_below and bar_above <= 3:
            yield r


def _trace_with_events(grid: np.ndarray, r: int, c: int):
    H, W = grid.shape
    direction = "right"
    events: list[dict[str, object]] = []
    for _ in range(700):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt == STRAIGHT_SIGNAL:
            events.append(
                {
                    "type": "signal",
                    "color": nxt,
                    "action": "straight",
                    "from_dir": direction,
                    "chosen": None,
                }
            )
            r, c = nr, nc
            continue
        if nxt == TURN_SIGNAL:
            options = _turn_options_from(grid, nr, nc, direction)
            chosen = _choose_signal_turn(grid, nr, nc, direction)
            events.append(
                {
                    "type": "signal",
                    "color": nxt,
                    "action": "turn",
                    "from_dir": direction,
                    "options": tuple(options),
                    "chosen": chosen,
                }
            )
            if chosen is None:
                break
            r, c = nr, nc
            direction = chosen
            continue
        if nxt in {BG, PARTICLE}:
            r, c = nr, nc
            continue

        options = _turn_options_from(grid, r, c, direction)
        chosen = _turn_from_blocker(nxt, direction)
        events.append(
            {
                "type": "blocker",
                "color": nxt,
                "from_dir": direction,
                "valid_dirs": tuple(options),
                "chosen": chosen,
            }
        )
        if chosen is None:
            break
        tdr, tdc = _DELTAS[chosen]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, chosen):
            r, c = tr, tc
            direction = chosen
            continue
        break
    return (r, c), events


def _trace_no_signals(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    H, W = grid.shape
    direction = "right"
    for _ in range(700):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt in _TRAVERSABLE:
            r, c = nr, nc
            continue
        chosen = _turn_from_blocker(nxt, direction)
        if chosen is None:
            break
        tdr, tdc = _DELTAS[chosen]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, chosen):
            r, c = tr, tc
            direction = chosen
            continue
        break
    return r, c


def _trace_unique_only(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    H, W = grid.shape
    direction = "right"
    for _ in range(700):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt in _TRAVERSABLE:
            r, c = nr, nc
            continue
        options = _turn_options_from(grid, r, c, direction)
        if len(options) != 1:
            break
        direction = options[0]
        dr, dc = _DELTAS[direction]
        r, c = r + dr, c + dc
    return r, c


def _variant_metrics(g: np.ndarray):
    metrics = {
        "moved": 0,
        "blocker_counts": {BLOCK_CCW: 0, BLOCK_CW: 0},
        "signal_counts": {TURN_SIGNAL: 0, STRAIGHT_SIGNAL: 0},
        "signal_turns": 0,
        "signal_straights": 0,
        "ambiguous_blockers": 0,
        "signal_then_blocker_paths": 0,
        "no_signal_differences": 0,
        "unique_only_differences": 0,
    }
    finals: dict[int, tuple[int, int]] = {}
    for r in _iter_active_rows(g):
        final_pos, events = _trace_with_events(g, r, 0)
        no_signal_pos = _trace_no_signals(g, r, 0)
        unique_only_pos = _trace_unique_only(g, r, 0)
        finals[r] = final_pos
        if final_pos != (r, 0):
            metrics["moved"] += 1
        if final_pos != no_signal_pos:
            metrics["no_signal_differences"] += 1
        if final_pos != unique_only_pos:
            metrics["unique_only_differences"] += 1

        saw_straight_signal = False
        for ev in events:
            if ev["type"] == "signal":
                metrics["signal_counts"][int(ev["color"])] += 1
                if ev["action"] == "turn":
                    metrics["signal_turns"] += 1
                else:
                    metrics["signal_straights"] += 1
                    saw_straight_signal = True
            else:
                color = int(ev["color"])
                if color in metrics["blocker_counts"]:
                    metrics["blocker_counts"][color] += 1
                if len(ev["valid_dirs"]) == 2:
                    metrics["ambiguous_blockers"] += 1
                if saw_straight_signal:
                    metrics["signal_then_blocker_paths"] += 1
                    saw_straight_signal = False
    return metrics, finals


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    g, bar_rows, bar_color = _paint_base_grid(H, W, bar_start, channel_spec)
    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    plans = _plan_rows(real_indices)
    if not _execute_plans(g, H, W, bar_start, plans, bar_rows, bar_color):
        return None

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
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    metrics, finals = _variant_metrics(g)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["blocker_counts"][BLOCK_CCW] < 1 or metrics["blocker_counts"][BLOCK_CW] < 1:
        return None
    if metrics["signal_counts"][TURN_SIGNAL] < 1 or metrics["signal_counts"][STRAIGHT_SIGNAL] < 1:
        return None
    if metrics["signal_turns"] < 1 or metrics["signal_straights"] < 1:
        return None
    if metrics["ambiguous_blockers"] < 2:
        return None
    if metrics["signal_then_blocker_paths"] < 1:
        return None
    if metrics["no_signal_differences"] < 1:
        return None
    if metrics["unique_only_differences"] < 1:
        return None
    if set(np.unique(g).tolist()) != {
        BG,
        PARTICLE,
        BAR_COLORS[0],
        BAR_COLORS[1],
        BLOCK_CCW,
        BLOCK_CW,
        TURN_SIGNAL,
        STRAIGHT_SIGNAL,
    }:
        return None

    for rr, (fr, fc) in finals.items():
        if (fr, fc) != (rr, 0) and int(out[fr, fc]) != PARTICLE:
            return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
