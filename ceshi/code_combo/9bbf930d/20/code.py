# Combo variant of 9bbf930d (index 20)
# combo_spec: D2/1+D5/3
# Name: 障碍色转向+多色粒子匹配走廊

from __future__ import annotations

import random

import numpy as np

"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 画面仍然是原题的条纹式布局：偶数行是彩色条，奇数行是 1-cell 通道；col 1
     仍为空背景，真实入口从奇数行的左侧开始。
   - 融合后有两套同时生效的颜色语义：
       * 粒子颜色 = 允许进入哪一种走廊。只有当通道两侧条色都等于该粒子色时，
         该通道才对它开放。
       * 障碍颜色 = 被挡住后向哪边拐。这里固定用两个提示色：
           - 2: 逆时针 90 度（右->上, 上->左, 左->下, 下->右）
           - 9: 顺时针 90 度（右->下, 下->左, 左->上, 上->右）
   - 具体追踪一个小例子：
       * 原题：row 5 左边有一个统一的品红粒子，右行撞到彩色格后，检查上下两边，
         哪边存在唯一同色条对走廊就往哪边转。
       * D5 单变体：row 5 的粒子可能是红色。如果该行入口条对是蓝色，则它连入口
         都不能进，停在 col 0。
       * 本融合版：row 5 的红粒子只能沿红色走廊前进。它向右撞到颜色 2 的提示块时，
         即便上下两边都被显式打开成红色走廊，也必须按颜色 2 选择“上转”；若换成
         颜色 9，则在同样的双分支结构中会“下转”。
   - 差异因此发生在两个链路：
       * 入口/转入资格：由粒子自身颜色过滤。
       * 真正选择哪条垂直支路：由障碍色决定，而不是由“唯一可转方向”决定。

b) 复杂度校准
   - 为了让人类可从少量样例中看出两条规则，颜色角色被严格分离：
       * 走廊/源粒子色只从 {1, 3, 4, 8} 取。
       * 转向提示色固定是 {2, 9}。
       * 背景固定是 7。
   - 这样每个样例最多 7 种颜色，但颜色职责很清楚，不会把“哪种颜色可进走廊”
     和“哪种颜色控制拐弯”混在一起。
   - generator 强制保留：
       * 至少两种真实粒子颜色实际移动；
       * 至少一个颜色匹配失败的真实入口行；
       * 至少一个 U-turn，让垂直阶段再次遇到提示色，从而支持上下/左右两个阶段
         都由同一套提示色控制；
       * 至少两个“双侧都可转”的事件，防止样例退化回原题的唯一分支。

c) 隐式假设审查（main 逐段）
   - 原版 `PARTICLE = 6` 的假设失效：现在 col 0 的粒子不再是单一颜色，因此
     `main()` 必须把“非背景的奇数行源颜色”当成粒子身份，并在写回结果时保留该色。
   - 原版 `_trace()` 假设“碰到彩色格时，只要扫描两个垂直方向，最多一个合法”。
     融合版故意把很多真实拐点做成双侧都开口，因此必须移除该假设，改成只看提示色
     决定的那个方向。
   - D5 参考实现里把 `particle_color` 当成一般可通行像素；这在融合版中会造成
     “与粒子同色的横条/竖条也被当成路” 的副作用，从而破坏原题的条纹几何。
     因此本实现收紧为：真正可走的路径单元仍然只有 BG，粒子色只参与“条对是否匹配”
     的资格判定。
   - 原版入口检测只验证“上下同色且 bar_start<=3”；融合后还必须要求这个颜色等于
     源粒子色，否则该行不启动。
   - 新 `_trace()` 还依赖两个前提：
       * 活动路径上的提示块颜色必须属于 {2, 9}；
       * 提示色指定的目标侧必须真的是该粒子颜色的合法走廊。
     generator 会在真实路径的每个活动拐点显式保证这两点。

d) 放置合法性
   - 所有垂直走廊仍通过 3 列窗口保留 (`c_v-1, c_v, c_v+1`)，避免不同真实轨迹的
     竖直结构互相覆盖。
   - 水平拐点使用对称开口模板：在源行的上下两个 bar row 同时打开 turn cell，
     使两侧都看起来像合法支路；真正要走哪边仅由提示色决定。
   - U-turn 只连接同色真实通道，目标通道的左入口会被破坏，避免其自身粒子也从左侧
     重复进入同一条轨迹。
   - 颜色分配在几何完成后才写入 col 0，因此不会引入新的重叠或越界。

e) 视觉可观察性
   - 人类可以从左边一列立即看到“粒子有多种颜色”；再对照其所在行左右条对颜色，
     会发现只有同色者才移动。
   - 在真正移动的那些行里，又能看到某些拐点前方的提示块是 2、某些是 9，且这些
     提示块往往位于上下双开口的岔口前，输出位置显示二者分别稳定地把粒子导向相反
     的 90 度转向。
   - 每个样例都会同时出现 2 和 9，且至少一个被访问到的“垂直再转回水平”事件，
     这样解题者无需跨很多样例拼图。

f) 可视化整洁度
   - 视觉层次被控制为三类：源粒子色、走廊条色、提示色。提示色不会充当走廊色，
     走廊色也不会充当提示色。
   - 双分支只发生在真实拐点的局部区域，背景仍占据大部分通道空间，整体图案保持
     原题那种整齐的横条+局部竖井风格。

g) 变化的可推断性
   - 这里没有依赖任何人类不可见的颜色顺序或数值大小。颜色 2/9 的行为是“固定语义”
     而非按样例重排，走廊准入规则则完全由肉眼可见的“粒子色是否等于两侧条色”决定。
   - generator 还会统计两个退化指标：
       * 若按原题“唯一合法转向”规则模拟，结果应与融合版不同；
       * 若忽略粒子色准入，只按提示色转向，结果也应与融合版不同。
     二者都成立时，样例才能证明融合行为确实发生。

h) 对新 main() 假设的复查
   - 假设 1：源对象只在奇数行 col 0。generator 保证如此，并保持整列 col 1 为 BG。
   - 假设 2：真实路径上的 turn target 一定是 BG。builder 在所有实际转入的格子上都
     显式挖出 BG，不会要求 solver 踩到彩色条上。
   - 假设 3：若提示色要求上/下/左/右转，目标侧通道的条对颜色与粒子色一致。
     generator 只在同色真实通道之间连通，且 U-turn 只配对同色真实行。
   - 假设 4：若某个源颜色与入口条色不匹配，solver 不会再尝试“跳过当前入口去找别的
     颜色入口”。generator 也正是围绕“第一条可见真实入口是否匹配”来定义源行为。
"""

BG = 7
TURN_CCW = 2
TURN_CW = 9
BAR_COLORS = (1, 3, 4, 8)

_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}
_CCW_TURN = {"right": "up", "up": "left", "left": "down", "down": "right"}
_CW_TURN = {"right": "down", "down": "left", "left": "up", "up": "right"}
_TURN_CLASS = {TURN_CCW: "ccw", TURN_CW: "cw"}


def _has_corridor(
    grid: np.ndarray, r: int, c: int, direction: str, particle_color: int
) -> bool:
    """True iff (r, c) is a BG path cell flanked by bars of the particle color."""
    H, W = grid.shape
    if int(grid[r, c]) != BG:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b == particle_color


def _has_corridor_any(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """Color-agnostic corridor check used for degeneration metrics."""
    H, W = grid.shape
    if int(grid[r, c]) != BG:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {BG, TURN_CCW, TURN_CW}


def _turn_from_blocker(color: int, direction: str) -> str | None:
    cls = _TURN_CLASS.get(color)
    if cls == "ccw":
        return _CCW_TURN[direction]
    if cls == "cw":
        return _CW_TURN[direction]
    return None


def _cue_color(from_dir: str, to_dir: str) -> int:
    if _CCW_TURN[from_dir] == to_dir:
        return TURN_CCW
    if _CW_TURN[from_dir] == to_dir:
        return TURN_CW
    raise ValueError(f"invalid turn {from_dir}->{to_dir}")


def _trace(grid: np.ndarray, r: int, c: int, particle_color: int) -> tuple[int, int]:
    """Trace one colored particle using cue-colored 90-degree turns."""
    H, W = grid.shape
    direction = "right"
    for _ in range(600):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt == BG:
            r, c = nr, nc
            continue
        turn_dir = _turn_from_blocker(nxt, direction)
        if turn_dir is None:
            break
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir, particle_color):
            r, c, direction = tr, tc, turn_dir
            continue
        break
    return r, c


def _trace_old_turn_rule(
    grid: np.ndarray, r: int, c: int, particle_color: int
) -> tuple[int, int]:
    """Ablation: keep color gating but turn only on a unique perpendicular option."""
    H, W = grid.shape
    direction = "right"
    for _ in range(600):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) == BG:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir, particle_color):
                options.append((tr, tc, turn_dir))
        if len(options) != 1:
            break
        r, c, direction = options[0]
    return r, c


def _trace_without_color_gate(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Ablation: keep cue-colored turning but let any same-color corridor be usable."""
    H, W = grid.shape
    direction = "right"
    for _ in range(600):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt == BG:
            r, c = nr, nc
            continue
        turn_dir = _turn_from_blocker(nxt, direction)
        if turn_dir is None:
            break
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor_any(grid, tr, tc, turn_dir):
            r, c, direction = tr, tc, turn_dir
            continue
        break
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move colored particles through same-color corridors using cue-colored turns.

    Input structure:
    - bg=7. Odd-row sources sit in col 0 and can have several colors.
    - Even rows contain colored bars; real corridors are BG cells flanked by matching
      bar colors, while cue blocks use colors 2 and 9.

    Rule:
    - A source may enter only a corridor whose two flank bars match the source color.
    - The particle moves straight through BG cells.
    - When blocked by a colored cell, that blocker color decides the 90-degree turn:
      2 means counterclockwise, 9 means clockwise relative to current travel.
    - The chosen target cell must be a BG corridor cell flanked by the source color.
      Otherwise the particle stops.

    Key insight:
    - Color plays two different roles at once: source color filters which corridors
      are legal, while blocker color controls which perpendicular branch is taken.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1, 2):
        particle_color = int(g[r, 0])
        if particle_color in {BG, TURN_CCW, TURN_CW}:
            continue

        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below == particle_color:
                entrance_col = c
                break
        if entrance_col is None:
            continue

        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == particle_color), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == particle_color), W)
        if bar_above != bar_below or bar_above > 3:
            continue

        end_r, end_c = _trace(g, r, 0, particle_color)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = particle_color

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]

_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.12
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.75

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MAX_RETRIES = 500
_MIN_AMBIGUOUS_EVENTS = 2
_MIN_COLOR_GATE_DIFFERENCES = 1
_MIN_TURN_RULE_DIFFERENCES = 1
_MIN_MISMATCHED_REAL_SOURCES = 1


def _other_bar_color(color: int) -> int:
    options = [c for c in BAR_COLORS if c != color]
    return random.choice(options)


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c for c in range(c_min, c_max + 1) if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
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
    plans: dict[int, dict[str, int | str | bool]],
) -> None:
    """Force at least one same-color U-turn when the palette allows it."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)

    placed_any = False
    color_groups = list(by_color.values())
    random.shuffle(color_groups)
    for chans in color_groups:
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        max_pairs = len(chans) // 2
        min_pairs = 1 if not placed_any else 0
        n_pairs = random.randint(min_pairs, max_pairs)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p : 2 * p + 2])
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                break
            internal_stop = random.random() < _U_INTERNAL_STOP_PROB
            reentry = random.choice(["left", "right"])
            if random.random() < 0.5:
                plans[j] = {
                    "kind": "u",
                    "target": i,
                    "direction": "up",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                    "reentry": reentry,
                }
                plans[i] = {"kind": "u_target"}
            else:
                plans[i] = {
                    "kind": "u",
                    "target": j,
                    "direction": "down",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                    "reentry": reentry,
                }
                plans[j] = {"kind": "u_target"}
            placed_any = True


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, int | str | bool]],
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
        direction = random.choice(["up", "down"])
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _paint_base_grid(H: int, W: int, bar_start: int, channel_spec: list[str]):
    g = np.full((H, W), BG, dtype=int)
    bar_rows = list(range(0, H, 2))

    bar_color = {bar_rows[0]: random.choice(BAR_COLORS)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = random.choice([c for c in BAR_COLORS if c != bar_color[top]])

    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]
    return g, bar_rows, bar_color


def _open_source_split(g: np.ndarray, top: int, bot: int, c_v: int, bar_color: dict[int, int]) -> None:
    """Open both perpendicular candidates at a horizontal blocker."""
    g[top, c_v - 1] = BG
    g[bot, c_v - 1] = BG
    g[top, c_v - 2] = bar_color[top]
    g[top, c_v] = bar_color[top]
    g[bot, c_v - 2] = bar_color[bot]
    g[bot, c_v] = bar_color[bot]


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    g[row, c_v] = _cue_color("right", direction)
    _open_source_split(g, top, bot, c_v, bar_color)

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
    c_v: int,
    direction: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    internal_stop: bool,
    reentry: str,
) -> None:
    g[row, c_v] = _cue_color("right", direction)
    _open_source_split(g, top, bot, c_v, bar_color)

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        j_row = j_top + 1
        block_r = j_top
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        j_row = j_top + 1
        block_r = j_bot

    g[block_r, c_v - 1] = _cue_color(direction, reentry)
    g[j_row, c_v - 2] = BG
    g[j_row, c_v] = BG

    g[j_top, bar_start] = _other_bar_color(bar_color[j_top])

    if internal_stop:
        if reentry == "left":
            lo, hi = bar_start + 2, c_v - 3
        else:
            lo, hi = c_v + 2, W - 1
        if lo <= hi:
            g[j_row, random.randint(lo, hi)] = random.choice((TURN_CCW, TURN_CW))


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, int | str | bool]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
) -> None:
    for i, plan in plans.items():
        if plan["kind"] in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        c_v = int(plan["c_v"])
        if plan["kind"] in {"l_edge", "l_internal"}:
            stop_at = None
            if plan["kind"] == "l_internal":
                if plan["direction"] == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, c_v, str(plan["direction"]), bar_color, stop_at)
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                H,
                W,
                top,
                bot,
                row,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                bool(plan.get("internal_stop", False)),
                str(plan.get("reentry", "left")),
            )


def _candidate_source_rows(g: np.ndarray, bar_rows: list[int]) -> list[tuple[int, int]]:
    """Rows with a geometrically valid left entrance, ignoring source color."""
    H, W = g.shape
    candidates: list[tuple[int, int]] = []
    for i, top in enumerate(bar_rows[:-1]):
        row = top + 1
        if row <= 0 or row >= H - 1:
            continue
        entrance_col = None
        corridor_color = None
        for c in range(2, W):
            if int(g[row, c]) != BG:
                continue
            above, below = int(g[row - 1, c]), int(g[row + 1, c])
            if above == below and above not in {BG, TURN_CCW, TURN_CW}:
                entrance_col = c
                corridor_color = above
                break
        if entrance_col is None or corridor_color is None:
            continue
        bar_above = next((c for c in range(W) if int(g[row - 1, c]) == corridor_color), W)
        bar_below = next((c for c in range(W) if int(g[row + 1, c]) == corridor_color), W)
        if bar_above == bar_below and bar_above <= 3:
            candidates.append((i, corridor_color))
    return candidates


def _assign_particle_colors(
    g: np.ndarray,
    channel_spec: list[str],
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
) -> tuple[dict[int, int], dict[str, int]] | tuple[None, None]:
    real_colors = sorted({bar_color[bar_rows[i]] for i in real_indices})
    if len(real_colors) < 2:
        return None, None

    candidates = _candidate_source_rows(g, bar_rows)
    candidate_indices = [i for i, _ in candidates]
    candidate_colors = {corridor_color for _, corridor_color in candidates}
    if len(candidate_indices) < _MIN_MOVED + _MIN_MISMATCHED_REAL_SOURCES:
        return None, None
    if len(candidate_colors) < 2:
        return None, None

    max_match = min(_MAX_MOVED, len(candidate_indices) - _MIN_MISMATCHED_REAL_SOURCES)
    if max_match < _MIN_MOVED:
        return None, None
    n_match = random.randint(_MIN_MOVED, max_match)
    matched = set(random.sample(candidate_indices, n_match))

    particle_color: dict[int, int] = {}
    for i in range(len(channel_spec)):
        corridor_color = bar_color[bar_rows[i]]
        if i in matched:
            particle_color[i] = corridor_color
        else:
            particle_color[i] = random.choice([c for c in real_colors if c != corridor_color])

    mismatched_real = sum(1 for i in candidate_indices if particle_color[i] != bar_color[bar_rows[i]])
    matched_real_colors = {particle_color[i] for i in matched}
    if mismatched_real < _MIN_MISMATCHED_REAL_SOURCES:
        return None, None
    if len(matched_real_colors) < 2:
        return None, None

    stats = {
        "candidate_real": len(candidate_indices),
        "matched_real": len(matched),
        "mismatched_real": mismatched_real,
        "matched_real_colors": len(matched_real_colors),
    }
    return particle_color, stats


def _paint_sources(g: np.ndarray, bar_rows: list[int], particle_color: dict[int, int]) -> None:
    g[:, 0] = BG
    for i, top in enumerate(bar_rows[:-1]):
        g[top + 1, 0] = particle_color[i]
    g[:, 1] = BG


def _iter_combo_rows(g: np.ndarray):
    H, W = g.shape
    for r in range(1, H - 1, 2):
        particle_color = int(g[r, 0])
        if particle_color in {BG, TURN_CCW, TURN_CW}:
            continue
        entrance_col = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below == particle_color:
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == particle_color), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == particle_color), W)
        if bar_above == bar_below and bar_above <= 3:
            yield r, particle_color


def _iter_geom_rows(g: np.ndarray):
    H, W = g.shape
    for r in range(1, H - 1, 2):
        if int(g[r, 0]) == BG:
            continue
        entrance_col = None
        corridor_color = None
        for c in range(2, W):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and above not in {BG, TURN_CCW, TURN_CW}:
                entrance_col = c
                corridor_color = above
                break
        if entrance_col is None or corridor_color is None:
            continue
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == corridor_color), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == corridor_color), W)
        if bar_above == bar_below and bar_above <= 3:
            yield r, corridor_color


def _trace_with_events(grid: np.ndarray, r: int, c: int, particle_color: int):
    H, W = grid.shape
    direction = "right"
    events = []
    for _ in range(600):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt == BG:
            r, c = nr, nc
            continue
        valid_dirs = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir, particle_color):
                valid_dirs.append(turn_dir)
        chosen = _turn_from_blocker(nxt, direction)
        events.append(
            {
                "from_dir": direction,
                "blocker_color": nxt,
                "valid_dirs": tuple(valid_dirs),
                "chosen": chosen,
            }
        )
        if chosen is None:
            break
        tdr, tdc = _DELTAS[chosen]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, chosen, particle_color):
            r, c, direction = tr, tc, chosen
            continue
        break
    return (r, c), events


def _variant_metrics(g: np.ndarray):
    metrics = {
        "moved": 0,
        "ambiguous_events": 0,
        "cue_counts": {TURN_CCW: 0, TURN_CW: 0},
        "vertical_turns": 0,
        "turn_rule_differences": 0,
        "color_gate_differences": 0,
    }

    combo_finals: dict[int, tuple[int, int]] = {}
    for r, particle_color in _iter_combo_rows(g):
        final_pos, events = _trace_with_events(g, r, 0, particle_color)
        legacy_turn_pos = _trace_old_turn_rule(g, r, 0, particle_color)
        combo_finals[r] = final_pos
        if final_pos != (r, 0):
            metrics["moved"] += 1
        if final_pos != legacy_turn_pos:
            metrics["turn_rule_differences"] += 1
        for ev in events:
            if len(ev["valid_dirs"]) == 2:
                metrics["ambiguous_events"] += 1
            if ev["blocker_color"] in metrics["cue_counts"]:
                metrics["cue_counts"][ev["blocker_color"]] += 1
            if ev["chosen"] is not None and ev["from_dir"] in {"up", "down"}:
                metrics["vertical_turns"] += 1

    for r, _corridor_color in _iter_geom_rows(g):
        combo_pos = combo_finals.get(r, (r, 0))
        ungated_pos = _trace_without_color_gate(g, r, 0)
        if combo_pos != ungated_pos:
            metrics["color_gate_differences"] += 1

    return metrics


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]
    bar_start = random.choice(_BAR_START_OPTIONS)

    g, bar_rows, bar_color = _paint_base_grid(H, W, bar_start, channel_spec)
    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str | bool]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    if not any(plan.get("kind") == "u" for plan in plans.values()):
        return None

    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start)

    particle_color, stats = _assign_particle_colors(g, channel_spec, real_indices, bar_rows, bar_color)
    if particle_color is None or stats is None:
        return None

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if g[row, c_decor] == BG:
                g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    _paint_sources(g, bar_rows, particle_color)

    try:
        out = main(g)
    except Exception:
        return None

    metrics = _variant_metrics(g)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["moved"] != stats["matched_real"]:
        return None
    if metrics["ambiguous_events"] < _MIN_AMBIGUOUS_EVENTS:
        return None
    if metrics["vertical_turns"] < 1:
        return None
    if metrics["turn_rule_differences"] < _MIN_TURN_RULE_DIFFERENCES:
        return None
    if metrics["color_gate_differences"] < _MIN_COLOR_GATE_DIFFERENCES:
        return None
    if stats["mismatched_real"] < _MIN_MISMATCHED_REAL_SOURCES:
        return None
    if metrics["cue_counts"][TURN_CCW] < 1 or metrics["cue_counts"][TURN_CW] < 1:
        return None

    observed_colors = set(np.unique(g).tolist())
    if not observed_colors.issubset({BG, TURN_CCW, TURN_CW, *BAR_COLORS}):
        return None
    if len(observed_colors) < 6:
        return None

    moved_out = sum(1 for r in range(1, H - 1, 2) if int(g[r, 0]) != BG and int(out[r, 0]) == BG)
    if moved_out != metrics["moved"]:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
