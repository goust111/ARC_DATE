# Variant of 9bbf930d
# Dimension: D13 — 多步处理与级联 (multi_step)
# Change: 单步处理——所有粒子独立计算终点，互不影响，结果一次性写入输出。→ 逐个粒子按从上到下顺序处理；
#         已经落定的粒子会留在工作网格中，成为后续粒子的真实障碍。

from __future__ import annotations

import random
import numpy as np


# ---------------------------------------------------------------------------
# Design analysis for the D13 variant
#
# concrete_behavior:
#   input_visual:
#     - 仍然是原题的条纹场景：偶数行是彩色横条，奇数行是走廊；第 0 列是品红粒子，第 1 列是背景。
#     - 变体不增加新标记，顺序依赖完全通过“后处理粒子看见前处理粒子的终点”来体现。
#   output_visual:
#     - 原题里，多条路径即使汇入同一尾段，也会按“各自独立算终点”一次性落子。
#     - 本变体里，上方粒子先落到尾段终点后，会堵住后面的粒子；因此后一个粒子会停在终点前一格
#       （或把该粒子当作普通障碍再尝试转弯）。
#   traced_example:
#     - 设源通道 A 在更上方，源通道 B 在其下方，二者都通过各自的 U-turn 进入同一目标通道 T。
#     - 原题：A 与 B 都独立在原始网格上追踪，都会落到 T 的最右端。
#     - 变体：先处理 A，A 先占住 T 的最右端；再处理 B，B 进入 T 后向右前进，下一格遇到品红障碍，
#       若无可转的垂直走廊则停在最右端前一格。
#
# complexity_calibration:
#   - 只把“粒子是否可穿过已存在的品红格”从 yes 改成 no，并显式维护工作网格。
#   - 不引入颜色排序、隐藏计数或额外状态编码；人类只需观察“谁先走、谁后被堵”。
#   - generator 强制至少一组共享尾段的双源结构，避免样本退化成与原题相同的独立运动。
#
# implicit_assumption_audit_of_original_main:
#   - _has_corridor 原本把 PARTICLE 当作可站立单元，这在原题无害，因为路径内部只有起点列含粒子。
#     在变体中若继续允许，会导致后续粒子穿过已落定粒子，顺序依赖消失，因此改为仅 BG 可进入。
#   - _trace 原本在原始输入 g 上独立运行，默认“其他粒子永不影响当前粒子”。
#     变体里必须在不断更新的 work grid 上追踪。
#   - main 原本使用 out 单独写结果，而 g 不变；这依赖“每个粒子彼此独立”。
#     变体中必须把成功移动后的终点立即写回 work，否则后续粒子看不到障碍。
#   - 行是否为真实入口的判定仍基于原始结构 grid，而不是 work；否则一个被前序粒子占住的入口
#     会让后续粒子连“尝试移动”都不发生，偏离“它只是被障碍挡住”的语义。
#
# placement_legality_and_visual_cleanliness:
#   - 生成器继续使用原题的 L/U 走廊构造函数，避免越界和二义转弯。
#   - 特制级联结构使用两个上方源通道 + 一个下方目标通道，中间插入一个 fake spacer channel，
#     防止 target 入口被打断后反过来破坏紧邻上方源通道的入口合法性。
#   - 两个源通道使用不同的垂直列进入同一目标通道，肉眼能直接看到“共享尾段”。
#
# observability:
#   - 不依赖颜色 id 或不可见的 tie-break；排序仅由从上到下的可见位置决定。
#   - 样例中会稳定出现“更高的源粒子占住右端，较低的源粒子停在它左边一格”的形态，
#     这是 2-3 组示例内可归纳的顺序规则。
# ---------------------------------------------------------------------------

BG = 7
PARTICLE = 6
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a bg cell flanked by same-color bars perpendicular to direction."""
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
    return a == b and a not in {PARTICLE, BG}


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle on the current working grid."""
    H, W = grid.shape
    direction = "right"
    for _ in range(500):
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
            if 0 <= tr < H and 0 <= tc < W and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _find_entrance_col(grid: np.ndarray, r: int) -> int | None:
    H, W = grid.shape
    if r <= 0 or r >= H - 1:
        return None
    for c in range(2, W):
        if int(grid[r, c]) != BG:
            continue
        above, below = int(grid[r - 1, c]), int(grid[r + 1, c])
        if above == below and above not in {PARTICLE, BG}:
            return c
    return None


def _is_real_channel(grid: np.ndarray, r: int) -> bool:
    W = grid.shape[1]
    entrance_col = _find_entrance_col(grid, r)
    if entrance_col is None:
        return False
    color = int(grid[r - 1, entrance_col])
    bar_above = next((c for c in range(W) if int(grid[r - 1, c]) == color), W)
    bar_below = next((c for c in range(W) if int(grid[r + 1, c]) == int(grid[r + 1, entrance_col])), W)
    return bar_above == bar_below and bar_above <= 3


def _simulate(grid: np.ndarray, sequential: bool) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Common engine for sequential variant and original parallel baseline."""
    base = np.array(grid, dtype=int)
    H, _ = base.shape
    out = base.copy()
    work = base.copy()
    moves: list[dict[str, object]] = []

    for r in range(1, H - 1):
        if int(work[r, 0]) != PARTICLE:
            continue
        if not _is_real_channel(base, r):
            continue
        trace_grid = work if sequential else base
        end_r, end_c = _trace(trace_grid, r, 0)
        moved = (end_r, end_c) != (r, 0)
        moves.append({"row": r, "end": (end_r, end_c), "moved": moved})
        if not moved:
            continue
        if sequential:
            work[r, 0] = BG
            work[end_r, end_c] = PARTICLE
        else:
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return (work if sequential else out), moves


def _parallel_main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, sequential=False)[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Process particles from top to bottom; settled particles block later ones."""
    return _simulate(grid, sequential=True)[0]


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 1]
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
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
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
        if plan["kind"] in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if plan["kind"] in ("l_edge", "l_internal"):
            direction = str(plan["direction"])
            stop_at = None
            if plan["kind"] == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, color, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
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


def _pick_special_cluster(n_channels: int) -> dict[str, object]:
    hub = random.randint(3, n_channels - 2)
    return {"sources": [hub - 3, hub - 2], "spacer": hub - 1, "target": hub}


def _interaction_metrics(
    g: np.ndarray,
    bar_rows: list[int],
    special: dict[str, object],
) -> dict[str, object]:
    seq_out, seq_moves = _simulate(g, sequential=True)
    par_out, par_moves = _simulate(g, sequential=False)
    seq_by_row = {int(m["row"]): m for m in seq_moves}
    par_by_row = {int(m["row"]): m for m in par_moves}
    interaction_rows = sorted(r for r in seq_by_row if seq_by_row[r]["end"] != par_by_row[r]["end"])
    source_channels = list(special["sources"])
    source_rows = [bar_rows[i] + 1 for i in source_channels]
    target = int(special["target"])
    target_row = bar_rows[target] + 1
    return {
        "seq_out": seq_out,
        "par_out": par_out,
        "seq_by_row": seq_by_row,
        "par_by_row": par_by_row,
        "interaction_rows": interaction_rows,
        "target_row": target_row,
        "source_channels": source_channels,
        "source_rows": source_rows,
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    special = _pick_special_cluster(n_channels)
    sources = list(special["sources"])
    spacer = int(special["spacer"])
    target = int(special["target"])

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels - 1)

    forced_real = set(sources + [target])
    forced_fake = {spacer}
    pool = [i for i in range(n_channels) if i not in forced_real and i not in forced_fake]
    if n_real < len(forced_real):
        n_real = len(forced_real)
    extra_needed = n_real - len(forced_real)
    extra_reals = set(random.sample(pool, extra_needed))
    real_set = forced_real | extra_reals
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]
    channel_spec[spacer] = "fake"

    bar_color: dict[int, int] = {bar_rows[0]: random.choice(extras)}
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

    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {target: {"kind": "u_target"}}

    special_cols = []
    for _ in sources:
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            return None
        special_cols.append(c_v)
    special_cols.sort()
    for src, c_v in zip(sources, special_cols):
        plans[src] = {
            "kind": "u",
            "target": target,
            "direction": "down",
            "c_v": c_v,
            "internal_stop": False,
        }

    remaining_reals = [i for i, s in enumerate(channel_spec) if s == "real" and i not in plans]
    _plan_u_turns(remaining_reals, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(remaining_reals, bar_start, W, used_vcols, plans)
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
        seq_out = main(g)
        par_out = _parallel_main(g)
    except Exception:
        return None

    metrics = _interaction_metrics(g, bar_rows, special)
    seq_by_row = metrics["seq_by_row"]
    par_by_row = metrics["par_by_row"]
    src0, src1 = metrics["source_rows"]
    if src0 not in seq_by_row or src1 not in seq_by_row:
        return None

    first_seq = seq_by_row[src0]
    second_seq = seq_by_row[src1]
    first_par = par_by_row[src0]
    second_par = par_by_row[src1]
    if not all(bool(m["moved"]) for m in (first_seq, second_seq, first_par, second_par)):
        return None
    if tuple(first_par["end"]) != tuple(second_par["end"]):
        return None
    if tuple(first_seq["end"]) != tuple(first_par["end"]):
        return None
    if tuple(second_seq["end"]) == tuple(second_par["end"]):
        return None
    if tuple(second_seq["end"])[0] != int(metrics["target_row"]):
        return None
    if tuple(second_seq["end"])[1] != tuple(second_par["end"])[1] - 1:
        return None
    if seq_out.shape != par_out.shape or not np.array_equal(seq_out, metrics["seq_out"]):
        return None
    if not metrics["interaction_rows"]:
        return None

    n_moved = sum(1 for m in metrics["seq_by_row"].values() if bool(m["moved"]))
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
