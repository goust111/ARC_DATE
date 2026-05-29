# Variant of 9bbf930d
# Dimension: D6 — 终点条件 (dst)
# Change: 粒子在到达网格边缘或遇到完全阻塞（无有效垂直走廊可转）时停止。... → 碰到异色条对（假走廊）停止

from __future__ import annotations

import random

import numpy as np

BG = 7
PARTICLE = 6
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}

# Design analysis for the D6 variant
# ==================================
# a) Concrete behavior:
#    原题里，粒子只把“被同色条对夹住的 BG 单元”当作真走廊；异色条对只是视觉噪声。
#    本变体保持入口识别、直行、撞墙后尝试 90 度转弯这些机制不变，只改终点：
#    粒子每前进一步后，都要检查“自己当前所在格子相对于当前运动方向，是否正好夹在一对异色条之间”。
#    如果是，它已经进入假走廊，因此立即停在这个 BG 格子里。
#    小例子：某条水平真走廊一开始上下都是红色，走到第 9 列后，下方条改成蓝色。
#    原逻辑：粒子会继续把这些 BG 当普通空白，一路滑到边缘或后续转角。
#    新逻辑：粒子进入第 9 列这个“上红下蓝”的 BG 格子时就停住，输出中的 6 落在异色条对之间。
# b) Complexity calibration:
#    只靠 solver 改停止条件不够，因为如果 generator 仍主要产出“全程真走廊直到边缘/死路”，
#    输入输出会退化成原题。因此 generator 强制至少 2 条真实入口通道在中途变成短假走廊补丁，
#    让新终点在 2-4 个样例里肉眼可见。其余通道仍保留直行 / L / U 的原始风格，避免任务过度简化。
# c) Hidden assumptions audit for main():
#    - 入口筛选依赖“最左侧真正可进入的通道在起始列附近必须是同色条对”。
#      该前提在变体中仍成立，所以入口逻辑不改；generator 保证假走廊补丁不会落在入口附近。
#    - 原 _trace 在 step-to-BG 后立即 continue，默认“所有可穿过的 BG 都没有额外语义”。
#      这个前提失效，因此改为“先移动，再检测当前格是否是假走廊终点”。
#    - 原转弯逻辑只接受同色条对，这是题目核心，不应扩大为“也能拐进假走廊”。
#      变体只把假走廊变成终点，不把它变成可转入的新路径，所以 _has_corridor 仍只认真走廊。
# d) Placement legality:
#    真实入口通道选择为“尽量互不相邻”的奇数行通道，原因是相邻通道共享一条横条。
#    假终点补丁要改写某一侧横条的颜色；若两个活动真通道相邻，会互相破坏对方的结构。
#    非相邻布局可以把改色影响限制到相邻的 fake channel，不会制造新的活动入口。
# e) Visual observability:
#    输出中的终点 6 将明显停在“上下(或左右)颜色不一致”的位置，而不是边界或转角封堵前。
#    人类观察几组样例时能看到：有些粒子停在走廊中途，且停点恰好是异色夹层的第一格。
# f) Visual cleanliness:
#    假终点补丁只在单条活动水平通道上做 2-4 列短色块，不叠加额外标记，不改变粒子颜色，
#    仍保留清晰的条带结构。活动通道彼此隔开，减少共享边界导致的视觉噪声。
# g) Color inferability:
#    没有引入颜色映射或基于颜色 id 的排序。唯一用到的视觉属性是“同色 / 异色是否匹配”，
#    这是直接可见的结构关系。
# h) Variant-specific metric used during validation:
#    统计所有发生移动的粒子中，stop_reason == 'fake' 的数量；要求每个样本至少 2 个。
#    这保证 generator 不是偶然产出仍按原题停止的退化样本。
# i) New solver assumptions checked against generator:
#    - _corridor_kind 只在当前格是 BG/粒子且两侧都在网格内时有意义；generator 的假终点补丁放在内部列。
#    - trace 假设入口通道起步是真走廊；generator 保证 bar_start 到补丁前保持同色。
#    - main 仍可能在多粒子场景下并行写输出；generator 通过分离通道与唯一终点设计，避免两个粒子争同一终点。


def _corridor_flanks(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int] | None:
    """Return the two bar colors perpendicular to direction, or None if absent."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        return int(grid[r - 1, c]), int(grid[r + 1, c])
    if c == 0 or c == W - 1:
        return None
    return int(grid[r, c - 1]), int(grid[r, c + 1])


def _corridor_kind(grid: np.ndarray, r: int, c: int, direction: str) -> str:
    """Classify a passable cell relative to current travel direction."""
    flanks = _corridor_flanks(grid, r, c, direction)
    if flanks is None:
        return "none"
    a, b = flanks
    if a in {PARTICLE, BG} or b in {PARTICLE, BG}:
        return "none"
    return "real" if a == b else "fake"


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True iff (r,c) is a real corridor cell flanked by same-color bars."""
    return _corridor_kind(grid, r, c, direction) == "real"


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int, str]:
    """Trace one particle and return final position plus stop reason."""
    H, W = grid.shape
    direction = "right"
    for _ in range(500):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            return r, c, "edge"
        if int(grid[nr, nc]) in {PARTICLE, BG}:
            r, c = nr, nc
            if _corridor_kind(grid, r, c, direction) == "fake":
                return r, c, "fake"
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            return r, c, "blocked"
        r, c, direction = options[0]
        if _corridor_kind(grid, r, c, direction) == "fake":
            return r, c, "fake"
    return r, c, "loop_guard"


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    """Return the first genuine horizontal corridor entrance for row r."""
    _, W = grid.shape
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            bar_above = next((k for k in range(W) if int(grid[r - 1, k]) == above), W)
            bar_below = next((k for k in range(W) if int(grid[r + 1, k]) == below), W)
            if bar_above == bar_below and bar_above <= 3:
                return c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Move particles through real corridors; entering a fake corridor stops them."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(g, r) is None:
            continue
        end_r, end_c, _ = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

# Channel assignment
_N_REAL_OPTIONS = [3, 4, 5]
_N_REAL_WEIGHTS = [5, 3, 1]
_MIN_PALETTE_DIVERSITY = 3

# Plan mix
_MIN_FAKE_STOP_CHANNELS = 2
_MAX_FAKE_STOP_CHANNELS = 3
_STRAIGHT_PROB = 0.20
_L_INTERNAL_PROB = 0.25
_U_INTERNAL_STOP_PROB = 0.70

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Fake-stop patch placement
_FAKE_STOP_LEFT_PAD = 4
_FAKE_STOP_RIGHT_PAD = 3
_FAKE_STOP_PATCH_RANGE = (2, 4)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_FAKE_STOPS = 2
_MAX_RETRIES = 400


def _pick_real_channels(n_channels: int, n_real: int) -> set[int]:
    """Pick mostly non-adjacent real channels so fake-stop patches stay local."""
    n_real = min(n_real, (n_channels + 1) // 2)
    best: list[int] = []
    for _ in range(80):
        order = list(range(n_channels))
        random.shuffle(order)
        chosen: list[int] = []
        for idx in order:
            if all(abs(idx - prev) > 1 for prev in chosen):
                chosen.append(idx)
                if len(chosen) == n_real:
                    return set(chosen)
        if len(chosen) > len(best):
            best = chosen
    return set(best)


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor wall column."""
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


def _choose_fake_stop_channels(real_indices: list[int]) -> list[int]:
    """Reserve at least two active channels for the new fake-corridor endpoint."""
    if len(real_indices) <= _MIN_FAKE_STOP_CHANNELS:
        return real_indices[:]
    hi = min(_MAX_FAKE_STOP_CHANNELS, len(real_indices) - 1)
    k = random.randint(_MIN_FAKE_STOP_CHANNELS, hi)
    return random.sample(real_indices, k)


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    """Pair same-color real channels into U-turn plans."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        by_color.setdefault(bar_color[bar_rows[i]], []).append(i)
    for chans in by_color.values():
        if len(chans) < 2:
            continue
        random.shuffle(chans)
        n_pairs = random.randint(0, len(chans) // 2)
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
    """Assign straight or L-turn plans to real channels not already planned."""
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


def _trace_all_particles(grid: np.ndarray) -> list[dict[str, object]]:
    """Collect stop data for all rows that have a genuine entrance."""
    H, _ = grid.shape
    traces: list[dict[str, object]] = []
    for r in range(1, H - 1):
        if int(grid[r, 0]) != PARTICLE:
            continue
        if _find_entrance_col(grid, r) is None:
            continue
        end_r, end_c, reason = _trace(grid, r, 0)
        traces.append(
            {
                "start": (r, 0),
                "end": (end_r, end_c),
                "reason": reason,
                "moved": (end_r, end_c) != (r, 0),
            }
        )
    return traces


def _gen_attempt() -> np.ndarray | None:
    """Generate one D6-variant input for task 9bbf930d."""
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    real_set = _pick_real_channels(n_channels, n_real)
    if len(real_set) < _MIN_MOVED:
        return None
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color: dict[int, int] = {bar_rows[0]: random.choice(extras)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            choices = [c for c in extras if c != bar_color[top]]
            bar_color[bot] = random.choice(choices)

    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    fake_stop_indices = _choose_fake_stop_channels(real_indices)
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {i: {"kind": "fake_stop"} for i in fake_stop_indices}
    remaining = [i for i in real_indices if i not in plans]
    _plan_u_turns(remaining, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(remaining, bar_start, W, used_vcols, plans)
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

    traces = _trace_all_particles(g)
    moved = [t for t in traces if bool(t["moved"])]
    fake_stops = sum(1 for t in moved if t["reason"] == "fake")
    if not (_MIN_MOVED <= len(moved) <= _MAX_MOVED):
        return None
    if fake_stops < _MIN_FAKE_STOPS:
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if not any(int(g[r, 0]) != int(out[r, 0]) for r in range(H)):
        return None
    return g


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
    """Render all planned turns and fake-stop patches onto the grid."""
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]

        if kind == "fake_stop":
            _build_fake_stop(g, W, top, bot, color, bar_start, extras)
            continue

        c_v = int(plan["c_v"])
        if kind in ("l_edge", "l_internal"):
            direction = str(plan["direction"])
            stop_at = None
            if kind == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif kind == "u":
            j_top = bar_rows[int(plan["target"])]
            j_bot = bar_rows[int(plan["target"]) + 1]
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


def _build_fake_stop(
    g: np.ndarray,
    W: int,
    top: int,
    bot: int,
    color: int,
    bar_start: int,
    extras: list[int],
) -> None:
    """Turn part of a real horizontal corridor into a short fake-corridor endpoint."""
    c_min = bar_start + _FAKE_STOP_LEFT_PAD
    c_max = W - _FAKE_STOP_RIGHT_PAD
    if c_min >= c_max:
        c_min = min(W - 2, bar_start + 3)
        c_max = max(c_min + 1, W - 2)
    c_stop = random.randint(c_min, c_max)
    patch_len = random.randint(*_FAKE_STOP_PATCH_RANGE)
    c_end = min(W, c_stop + patch_len)
    changed_row = random.choice([top, bot])
    other_color = random.choice([c for c in extras if c != color])
    g[changed_row, c_stop:c_end] = other_color


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
    """Build an L-turn at channel (top, bot) with a wall at (row, c_v)."""
    g[row, c_v] = color
    if direction == "up":
        turn_bar, opp_bar = top, bot
    else:
        turn_bar, opp_bar = bot, top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = bar_color[turn_bar]
    g[opp_bar, c_v - 1] = bar_color[opp_bar]

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
    """Build a U-turn between source channel (top..bot) and target (j_top..j_bot)."""
    g[row, c_v] = color

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

    other_colors = [c for c in extras if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, W - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
