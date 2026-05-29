# Variant of 9bbf930d
# Dimension: D14 — 符号编码与图例 (symbol_instructions)
# Change: 无图例或符号编码，所有规则由网格结构隐式表达。... → 控制像素标记可转弯位置
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题里“是否转弯”完全由几何决定：粒子在水平/垂直走廊中前进，只要前方被彩色墙挡住，
#   且某个垂直于当前方向的邻格满足“该格是通路、两侧是同色条对”，solver 就会转弯。
# - 变体把“转弯许可”改成显式符号：使用红色控制像素（ARC 红色=2）放在粒子当前所在的通路格。
#   该红点本身可通行，但只有当粒子正站在红点上且前方被墙挡住时，才允许检查垂直方向并转弯。
# - 因此控制像素放在“转弯决策格”而不是拐过去后的目标格：
#   对水平 -> 垂直的 L 转弯，红点在 `(channel_row, blocker_col-1)`；
#   对 U 转弯的第二次拐弯，红点在目标横向走廊与竖直通道的交点 `(target_row, vcol)`。
# - 小例子：
#   原逻辑：粒子在 `(5, 8)`，前方 `(5, 9)` 是彩墙，上方 `(4, 8)` 是合法竖直走廊入口，
#   所以直接向上转。
#   变体逻辑：若 `(5, 8)` 是红点，则同样向上转；若 `(5, 8)` 只是背景色，则即使上方存在合法竖直
#   走廊，也直接停在 `(5, 8)`。这就是“显式标记替代结构推断”的核心差异。
#
# b) 复杂度校准
# - 只改 D14，不改走廊几何本体：同色条对、假走廊、L/U 形结构都保留。
# - 为了让人能看出变体，不只生成“有红点才拐弯”的正例，也强制生成至少一个“几何上可拐、但没红点
#   所以不拐”的抑制案例。否则样例可能退化成“原题 + 无关红点装饰”。
# - 控制符号只用一个颜色、一个像素大小，避免额外编码自由度。
#
# c) 原 main() 的隐式前提审查
# - `_has_corridor()` 原先把 `{BG, PARTICLE}` 当成可通路格；变体下红点也必须可通行，
#   否则粒子会被控制符号本身挡住，所以要扩成 `{BG, PARTICLE, CONTROL}`。
# - `_trace()` 原先“遇阻就尝试转弯”隐含前提是：所有合法转弯都应执行。变体下这不成立，
#   因此必须先检查“当前格是否为 CONTROL”，否则直接停止。
# - `main()` 的入口检测原先只认 BG。变体里控制点也可能落在真实走廊上，所以入口扫描要把 CONTROL
#   当成可通路格。其余关于“上下条颜色相同、条起点在前 3 列”的前提仍然成立。
# - 输出阶段仍旧只移动粒子。生成器因此要避免把最终停点设计在控制像素上，否则 solver 会把红点覆盖掉，
#   产生不必要的新变化。
#
# d) 放置合法性
# - 控制点只放在原本就是通路的格子上，不与条纹重叠。
# - 所有竖直转弯列继续使用 3 列窗口预约，避免不同转弯结构互相踩踏。
# - U 转弯目标行仍打断自身入口，避免该行左侧粒子也进入同一走廊导致歧义。
#
# e) 视觉可观察性
# - 输入中会同时出现：
#   1. 有红点的拐弯点：粒子会拐。
#   2. 结构上几乎一样、但无红点的拐弯点：粒子不会拐。
# - 这样 2-3 组样例就能让人直接归纳“红点是转弯许可”。
#
# f) 可视化整洁度
# - 控制点只出现在走廊中心线，不放在彩条上，也不做多色图例；视觉层次仍是“彩条/橙色通道/红点/品红粒子”。
# - 红点数量保持少量，只在真正关键转折处出现，避免满图噪声。
#
# g) 颜色变化可推断性
# - 没有新的颜色映射规则；红色只承担单一语义“允许在此转弯”。
# - 为保持总颜色数与原题接近，走廊色减为 3 种，加上 BG、PARTICLE、CONTROL 共 6 色。
#
# h) 变体量化指标
# - `suppressed_rows`: 若忽略控制像素、沿原题规则追踪，某些粒子的最终位置会不同；这些行数必须 >= 1。
# - `controlled_turn_rows`: 在新规则下实际发生至少一次转弯的行数必须 >= 1。
# - 上述两个指标一起保证：红点既在起作用，也不是所有转弯都被禁用。
#
# i) 新 main() 假设审查
# - 假设 1：所有需要执行的转弯，其“决策格”上都有控制像素。生成器在 `_build_l_turn()` 和
#   `_build_u_turn()` 中显式放置。
# - 假设 2：没有控制像素的可转弯结构可以安全存在，solver 会停下而不是断言。生成器用
#   `blocked_turn` 计划专门构造这种案例。
# - 假设 3：控制像素不会占据最终停点。生成器只把控制点放在转弯前一格，且后续至少再走一步。

from __future__ import annotations

import random
import numpy as np

BG = 7
PARTICLE = 6
CONTROL = 2

_PATH_VALUES = {BG, PARTICLE, CONTROL}
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _is_path(value: int) -> bool:
    return int(value) in _PATH_VALUES


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a traversable cell flanked by same-color bars perpendicular to direction."""
    H, W = grid.shape
    if not _is_path(int(grid[r, c])):
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in _PATH_VALUES


def _trace_state(
    grid: np.ndarray, r: int, c: int, allow_unmarked_turns: bool = False
) -> tuple[int, int, int]:
    """Trace a particle and return `(final_r, final_c, n_turns)`."""
    H, W = grid.shape
    direction = "right"
    turns = 0
    for _ in range(H * W * 4):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if _is_path(int(grid[nr, nc])):
            r, c = nr, nc
            continue

        if (not allow_unmarked_turns) and int(grid[r, c]) != CONTROL:
            break

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
        turns += 1
    return r, c, turns


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    end_r, end_c, _ = _trace_state(grid, r, c, allow_unmarked_turns=False)
    return end_r, end_c


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    H, W = grid.shape
    if r <= 0 or r >= H - 1:
        return None
    for c in range(2, W):
        if not _is_path(int(grid[r, c])):
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in _PATH_VALUES:
            return c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through bar corridors, but turn only on red control pixels."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue

        entrance_col = _find_entrance_col(g, r)
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
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 3

_N_REAL_OPTIONS = [3, 4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2, 1]
_MIN_PALETTE_DIVERSITY = 3

_STRAIGHT_PROB = 0.12
_BLOCKED_TURN_PROB = 0.28
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.80

_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

_FAKE_DECOR_RANGE = (0, 2)

_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor; return the wall column c_v."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c
        for c in range(c_min, c_max + 1)
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
) -> None:
    """Pair some same-color real channels into U-turn plans."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        max_pairs = len(chans) // 2
        n_pairs = random.randint(0, max_pairs)
        for p in range(n_pairs):
            i, j = sorted(chans[2 * p : 2 * p + 2])
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if c_v is None:
                break
            internal_stop = random.random() < _U_INTERNAL_STOP_PROB
            if random.random() < 0.5:
                plans[j] = {
                    "kind": "u",
                    "target": i,
                    "direction": "up",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                }
                plans[i] = {"kind": "u_target"}
            else:
                plans[i] = {
                    "kind": "u",
                    "target": j,
                    "direction": "down",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                }
                plans[j] = {"kind": "u_target"}


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    """Assign straight, blocked-turn, or controlled L-turn plans to unplanned real channels."""
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
        roll = random.random()
        if roll < _BLOCKED_TURN_PROB:
            plans[i] = {"kind": "blocked_turn", "direction": direction, "c_v": c_v}
        else:
            kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
            plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _ensure_plan_mix(
    real_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> bool:
    """Force at least one controlled-turn row and one blocked-turn row when possible."""

    def kinds_of(keys: set[str]) -> list[int]:
        return [i for i in real_indices if plans.get(i, {}).get("kind") in keys]

    def add_from_straight(kind: str) -> bool:
        straight_rows = kinds_of({"straight"})
        if not straight_rows:
            return False
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            return False
        i = random.choice(straight_rows)
        plans[i] = {"kind": kind, "direction": random.choice(["up", "down"]), "c_v": c_v}
        return True

    controlled_rows = kinds_of({"l_edge", "l_internal", "u"})
    blocked_rows = kinds_of({"blocked_turn"})

    if not controlled_rows:
        if not add_from_straight("l_edge"):
            blocked_rows = kinds_of({"blocked_turn"})
            if not blocked_rows:
                return False
            i = random.choice(blocked_rows)
            plans[i]["kind"] = "l_edge"
        controlled_rows = kinds_of({"l_edge", "l_internal", "u"})

    if not blocked_rows:
        if not add_from_straight("blocked_turn"):
            l_rows = kinds_of({"l_edge", "l_internal"})
            if not l_rows:
                return False
            if len(controlled_rows) <= 1 and not kinds_of({"u"}):
                return False
            i = random.choice(l_rows)
            plans[i]["kind"] = "blocked_turn"

    return True


def _paint_control(g: np.ndarray, r: int, c: int) -> None:
    assert int(g[r, c]) in {BG, CONTROL}, "control pixel must sit on a path cell"
    g[r, c] = CONTROL


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
    add_control: bool,
) -> None:
    """Build an L-turn; optionally add a control pixel on the turn-decision cell."""
    g[row, c_v] = color

    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = bar_color[turn_bar]
    g[opp_bar, c_v - 1] = bar_color[opp_bar]

    if add_control:
        _paint_control(g, row, c_v - 1)

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
    """Build a U-turn and place control pixels at both turn-decision cells."""
    g[row, c_v] = color
    _paint_control(g, row, c_v - 1)

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[bot, c_v - 1] = bar_color[bot]
        g[j_top, c_v - 2] = BG
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        g[top, c_v - 1] = bar_color[top]
        g[j_bot, c_v - 2] = BG

    _paint_control(g, j_top + 1, c_v - 1)

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
    """Render all planned turns and control pixels onto the grid."""
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
            continue

        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if plan["kind"] in ("l_edge", "l_internal", "blocked_turn"):
            direction = str(plan["direction"])
            stop_at = None
            if plan["kind"] == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(
                g,
                H,
                top,
                bot,
                row,
                color,
                c_v,
                direction,
                bar_color,
                stop_at,
                add_control=(plan["kind"] != "blocked_turn"),
            )
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                bool(plan.get("internal_stop", False)),
            )


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether control pixels materially change behavior."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    controlled_turn_rows = 0
    legacy_turn_rows = 0
    suppressed_rows = 0
    moved_rows = 0

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue

        end_r, end_c, turns = _trace_state(g, r, 0, allow_unmarked_turns=False)
        legacy_r, legacy_c, legacy_turns = _trace_state(g, r, 0, allow_unmarked_turns=True)

        if (end_r, end_c) != (r, 0):
            moved_rows += 1
        if turns > 0:
            controlled_turn_rows += 1
        if legacy_turns > 0:
            legacy_turn_rows += 1
        if (end_r, end_c) != (legacy_r, legacy_c):
            suppressed_rows += 1

    return {
        "control_pixels": int(np.count_nonzero(g == CONTROL)),
        "controlled_turn_rows": controlled_turn_rows,
        "legacy_turn_rows": legacy_turn_rows,
        "suppressed_rows": suppressed_rows,
        "moved_rows": moved_rows,
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample(
        [c for c in range(10) if c not in (BG, PARTICLE, CONTROL)],
        _N_EXTRA_COLORS,
    )

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
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

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    if not _ensure_plan_mix(real_indices, bar_start, W, used_vcols, plans):
        return None
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, extras)

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

    n_moved = sum(
        1 for r in range(H) if int(g[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    )
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None

    try:
        metrics = _variant_metrics(g)
    except Exception:
        return None
    if metrics["controlled_turn_rows"] < 1:
        return None
    if metrics["suppressed_rows"] < 1:
        return None
    if metrics["control_pixels"] < 1:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
