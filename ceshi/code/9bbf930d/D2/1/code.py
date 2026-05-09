# Variant of 9bbf930d
# Dimension: D2 — 转弯规则 (path)
# Change: 被彩色格挡住时检查两个垂直方向，选择有同色条对走廊的一侧转90度；必须恰好一个方向有效（无歧义断裂）。... → 转弯方向由障碍色决定

from __future__ import annotations

import random
import numpy as np

"""
Design analysis for the D2 variant
==================================

a) 具体化与行为追踪
   - 仍然保留原题的“偶数行是彩色条，奇数行是通道”的条纹布局，粒子仍从最左列出发。
   - 变体的唯一核心变化是：真正触发转弯的不是“哪一侧有唯一走廊”，而是“挡在正前方的提示色”。
   - 本实现将提示色固定为两类：
       * 颜色 2: 逆时针转弯。右->上， 上->左， 左->下， 下->右。
       * 颜色 9: 顺时针转弯。右->下， 下->左， 左->上， 上->右。
   - 为了让这个变化在像素上可观察，generator 不再把拐点做成“只有一侧可转”，而是故意在许多拐点两侧都开口：
       * 例如粒子向右前进，在 (r, c+1) 被颜色 2 挡住时，(r-1, c) 和 (r+1, c) 两边都满足同色条对走廊。
       * 原题逻辑在这里会因为双侧都可转而失效/歧义；变体逻辑则直接根据颜色 2 选上转。
       * 另一处颜色 9 的挡块会在同样的双分支结构上选下转。
   - 小例子追踪：
       * 原逻辑：向右撞到彩色格 -> 扫描上/下 -> 两边都合法 -> 发生歧义。
       * 新逻辑：向右撞到颜色 2 -> 颜色 2 表示逆时针 -> 右->上 -> 只检查上方那一支是否合法 -> 合法则进入。
       * 差异不在“有没有路”，而在“同样两边都有路时到底选哪边”。

b) 复杂度校准
   - 颜色种类控制为 6 种，尽量贴近原题：bg=7，particle=6，走廊条色仅 1/8，转向提示色仅 2/9。
   - 两个提示色足够让人从少量样例归纳映射，同时不会引入过多类别导致记忆负担。
   - 保留直行、L 转弯、U 转弯三类路径，使“颜色控制转向”在水平->垂直和垂直->水平两个阶段都能出现。

c) 隐式假设审查（main 相关）
   - 原 _trace 依赖前提：被挡住时，两个垂直方向至多一个合法；否则 assert。
     变体中这个前提被刻意打破，因此必须删掉“扫描全部再断言唯一”的逻辑。
   - 原 main 只要求入口是横向同色条对，这一点仍成立，因此入口识别代码可保留。
   - 新 _trace 的新前提：真正出现在活动路径上的挡块颜色必须属于 {2, 9} 之一，否则粒子会停下。
     generator 通过只在真实拐点上放 2/9 来保证这一点。
   - 新 _trace 还假设：提示色所指定的那一侧如果存在，就应该是合法走廊。
     generator 在构造每个拐点时都会显式开出该侧走廊。

d) 放置合法性
   - 垂直走廊列仍然用 3 列窗口预留，避免不同拐点的竖直结构互相覆盖。
   - 双分支只在同一个拐点处额外打开“未被选择”的那一侧，而且只做短 stub 或独立长走廊，不与其它 vcol 重叠。
   - U 转弯仍只连接同色水平通道；目标通道入口会被破坏，避免其自身粒子也进入同一路径。

e) 视觉可观察性
   - 提示色和走廊条色完全分离：1/8 负责定义走廊，2/9 负责决定转向。
   - 每个样本都要求实际被访问到的提示色同时出现 2 和 9，并且至少出现一次垂直阶段的转弯，让“上/下”和“左/右”都能被样例支持。
   - 另外要求至少两个“双方都可转”的事件，防止样本退化成原题那种唯一可转的情形。

f) 可视化整洁度
   - 仍然是整齐的横条 + 局部竖向缺口；双分支仅在拐点附近增加一个对称开口，不引入散乱标记。
   - 假走廊继续只由上下不同色条形成，不使用提示色，避免视觉角色混淆。

g) 颜色变化可推断性
   - 提示色映射是“固定颜色语义”，不是按颜色 id 排序，也不是样例内随机置换。
   - 只有两个提示色，且二者在每个样本中都出现在真实路径上，解题者可以直接从输出粒子的去向归纳：
       * 颜色 2 总是把粒子拐向“上/左”这一侧。
       * 颜色 9 总是把粒子拐向“下/右”这一侧。

9) 对新 main() 的假设复查
   - 每个真实拐点前一格必须是 bg，挡块必须正好在前方一格；generator 按原题几何模板构造。
   - 选择后的目标单元必须是 _has_corridor 为真的通道；builder 会显式打开该 turn cell 与延长段。
   - 如果某条路径需要第二次转弯（U 转弯回接），垂直终点处的挡块也必须是提示色 2/9；generator 会单独着色。
   - 如果挡块不是提示色，solver 会停下；因此 generator 不允许活动路径把普通条色当作“应当转弯”的挡块。
"""

BG = 7
PARTICLE = 6
BAR_COLORS = (1, 8)
TURN_CCW = 2
TURN_CW = 9

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


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r, c) is a passable cell flanked by same-color bars."""
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


def _turn_from_blocker(color: int, direction: str) -> str | None:
    """Map a blocker color to a relative 90-degree turn."""
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


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle using color-controlled turns."""
    H, W = grid.shape
    direction = "right"
    for _ in range(600):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt in {PARTICLE, BG}:
            r, c = nr, nc
            continue
        turn_dir = _turn_from_blocker(nxt, direction)
        if turn_dir is None:
            break
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
            r, c, direction = tr, tc, turn_dir
            continue
        break
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles along same-color corridors using cue-colored turns."""
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
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next((c for c in range(W) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), W)
        bar_below = next((c for c in range(W) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), W)
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
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_STRAIGHT_PROB = 0.12
_L_INTERNAL_PROB = 0.30
_U_INTERNAL_STOP_PROB = 0.75
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)
_MIN_MOVED = 3
_MAX_MOVED = 6
_MAX_RETRIES = 400
_MIN_AMBIGUOUS_EVENTS = 2


def _other_bar_color(color: int) -> int:
    return BAR_COLORS[1] if color == BAR_COLORS[0] else BAR_COLORS[0]


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


def _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans):
    """Force at least one U-turn when possible so vertical-stage cue colors appear."""
    by_color = {}
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
            i, j = sorted(chans[2 * p:2 * p + 2])
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


def _plan_remaining(real_indices, bar_start, W, used_vcols, plans):
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


def _paint_base_grid(H: int, W: int, bar_start: int, channel_spec):
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


def _open_source_split(g, top, bot, c_v, bar_color):
    """Open both up/down candidates at a horizontal blocker."""
    g[top, c_v - 1] = BG
    g[bot, c_v - 1] = BG
    g[top, c_v - 2] = bar_color[top]
    g[top, c_v] = bar_color[top]
    g[bot, c_v - 2] = bar_color[bot]
    g[bot, c_v] = bar_color[bot]


def _build_l_turn(g, H, top, bot, row, bar_color, c_v, direction, stop_at):
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
    g, H, W, top, bot, row, c_v, direction, j_top, j_bot, bar_color, bar_start, internal_stop, reentry
):
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
    if c_v - 2 >= 0:
        g[j_row, c_v - 2] = BG
    if c_v < W:
        g[j_row, c_v] = BG

    # Break the target channel entrance so its own particle does not also trace the path.
    g[j_top, bar_start] = _other_bar_color(bar_color[j_top])

    if internal_stop:
        if reentry == "left":
            lo, hi = bar_start + 2, c_v - 3
        else:
            lo, hi = c_v + 2, W - 1
        if lo <= hi:
            g[j_row, random.randint(lo, hi)] = bar_color[j_top]


def _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start):
    for i, plan in plans.items():
        if plan["kind"] in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        c_v = plan["c_v"]
        if plan["kind"] in {"l_edge", "l_internal"}:
            stop_at = None
            if plan["kind"] == "l_internal":
                if plan["direction"] == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, bar_color, c_v, plan["direction"], stop_at)
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[plan["target"]], bar_rows[plan["target"] + 1]
            _build_u_turn(
                g,
                H,
                W,
                top,
                bot,
                row,
                c_v,
                plan["direction"],
                j_top,
                j_bot,
                bar_color,
                bar_start,
                plan.get("internal_stop", False),
                plan.get("reentry", "left"),
            )


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
            if above == below and above not in {PARTICLE, BG}:
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
    events = []
    for _ in range(600):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt in {PARTICLE, BG}:
            r, c = nr, nc
            continue
        valid_dirs = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
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
        if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, chosen):
            r, c, direction = tr, tc, chosen
            continue
        break
    return (r, c), events


def _trace_legacy(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Old rule without assert: turn only if exactly one perpendicular side works."""
    H, W = grid.shape
    direction = "right"
    for _ in range(600):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        if len(options) != 1:
            break
        r, c, direction = options[0]
    return r, c


def _variant_metrics(g: np.ndarray):
    metrics = {
        "moved": 0,
        "ambiguous_events": 0,
        "cue_counts": {TURN_CCW: 0, TURN_CW: 0},
        "vertical_turns": 0,
        "legacy_differences": 0,
    }
    finals = {}
    legacy_finals = {}
    for r in _iter_active_rows(g):
        final_pos, events = _trace_with_events(g, r, 0)
        legacy_pos = _trace_legacy(g, r, 0)
        finals[r] = final_pos
        legacy_finals[r] = legacy_pos
        if final_pos != (r, 0):
            metrics["moved"] += 1
        if final_pos != legacy_pos:
            metrics["legacy_differences"] += 1
        for ev in events:
            if len(ev["valid_dirs"]) == 2:
                metrics["ambiguous_events"] += 1
            if ev["blocker_color"] in metrics["cue_counts"]:
                metrics["cue_counts"][ev["blocker_color"]] += 1
            if ev["chosen"] is not None and ev["from_dir"] in {"up", "down"}:
                metrics["vertical_turns"] += 1
    return metrics, finals, legacy_finals


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
    if set(bar_color.values()) != set(BAR_COLORS):
        return None

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols = set()
    plans = {}
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    if not any(plan.get("kind") == "u" for plan in plans.values()):
        return None

    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start)

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
    metrics, finals, legacy_finals = _variant_metrics(g)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["ambiguous_events"] < _MIN_AMBIGUOUS_EVENTS:
        return None
    if metrics["vertical_turns"] < 1:
        return None
    if metrics["legacy_differences"] < 1:
        return None
    if metrics["cue_counts"][TURN_CCW] < 1 or metrics["cue_counts"][TURN_CW] < 1:
        return None
    if set(np.unique(g).tolist()) != {BG, PARTICLE, *BAR_COLORS, TURN_CCW, TURN_CW}:
        return None
    # Ensure moved particles really land where the traced simulation says they should.
    for r, pos in finals.items():
        rr, cc = pos
        if pos != legacy_finals[r] and int(out[rr, cc]) != PARTICLE:
            return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
