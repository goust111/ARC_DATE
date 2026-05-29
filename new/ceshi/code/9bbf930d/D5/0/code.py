# Variant of 9bbf930d
# Dimension: D5 — 源对象：粒子定义 (src)
# Change: 品红色粒子位于最左列（col 0），从有效走廊入口的奇数行出发，初始方向固定向右。... → 粒子位于网格边缘任意一侧

from __future__ import annotations

import random

import numpy as np

"""
design_analysis:
  concretization:
    original_visual:
      - 原题是横向条带布局: 偶数行是彩色横条, 奇数行是背景通道.
      - 左边界的品红粒子从 col 0 向右进入走廊; 竖向拐弯由同色条在某列处开缺口实现.
    variant_visual:
      - 走廊几何仍保持同一套条带/竖向拐弯机制, 只改变“粒子放在哪个边界端点”.
      - 同一张图中可以同时看到四类起点: 左侧粒子向右, 右侧粒子向左, 顶侧粒子向下, 底侧粒子向上.
      - 顶/底起点只放在真正通向边缘的竖直走廊端点; 右侧起点只放在无转弯的直通道右端, 避免逆向进入时无解.
    traced_example:
      - 原逻辑: 某奇数行的左粒子先穿过 BG, 在 (row, c_v) 被彩色墙挡住, 于是检查 up/down, 进入一条同色夹出的竖直走廊.
      - 变体逻辑: 若粒子在顶边 (0, c_v-1), 初始方向必须是 down. 它沿竖直走廊下行, 在底部封口前被挡, 再按规则检查 left/right, 转入同色横向走廊.
      - 差异核心不在转弯机制, 而在“起始方向由边界位置推断”, 不再写死为 right.
  complexity_calibration:
    - 不引入额外标记或颜色编码; 仍只靠粒子所在边决定初始方向, 规则肉眼可见.
    - 为了让 D5 在少量样例里明显可见, generator 强制每题至少出现 left/right/top/bottom 四个边界起点中的四类.
    - 仍保留 straight/L/U 三种通道类型与 fake corridor, 避免任务退化成纯直线搬运.
  implicit_assumptions_audit:
    - `_trace` 原先写死 `direction = "right"`; 变体下必须改为参数.
    - `main` 原先只扫描 `g[r, 0] == PARTICLE`; 变体下必须扫描四条边.
    - 原先的“入口检查”假设真实入口在左侧, 所以需要检查条带起点是否在 col <= 3; 变体下不能继续把这个约束绑死在左边, 改为沿粒子朝内方向寻找首个可达 corridor cell.
    - 原 generator 会无条件执行 `g[:, 0] = PARTICLE`; 变体下这会覆盖已有正确边界状态, 必须删除.
    - 原 generator 用 `n_moved` 统计 col 0 被清空的数量; 变体下应按全部边界粒子统计.
  placement_legality:
    - 右侧粒子只放在 straight real channel 的右端, 因为这类路径从右向左进入时不会遇到“镜像缺失”的转弯歧义.
    - 顶/底粒子只放在 `l_edge` 计划产生的真实竖直边缘端点, 这些端点一定是 BG 且与同色侧墙匹配.
    - 每条真实路径至多放一个粒子, 避免同一路径两端同时放粒子造成输入输出相同的退化.
    - 竖直走廊列仍复用原始的窗口占用规则, 防止相邻拐弯重叠.
  visual_observability:
    - 解题者能直接看到品红粒子散布在四条边, 且输出中的终点落在路径另一端; 这比“仍全在左边”更容易肉眼归纳出 D5.
    - 不使用颜色 id 排序或隐藏状态. 方向完全由几何边界位置决定.
  cleanliness:
    - 边界粒子只落在原本就属于通路端点的 BG/gap 上, 不叠压到彩色墙体主体.
    - 同一真实路径只放一个粒子, 终点不会与其他路径共享, 视觉上清楚.
  color_inference:
    - 颜色机制与原题相同: 是否可转弯仍由两侧同色条决定, 不引入新的颜色映射问题.
  new_solver_assumptions_and_generator_guarantees:
    - solver 假设边界粒子不在角上, 或者若在角上也只能有一个朝内方向能连到真实走廊.
      generator 保证不放角点.
    - solver 假设每个边界粒子沿朝内射线都能到达至少一个 passable corridor cell.
      generator 只在 `_candidate_starts` 返回的位置上放粒子.
    - solver 假设不同粒子的终点不冲突.
      generator 每条路径最多一个起点, 且转弯列/通道行互异.
"""

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
    """True if (r,c) is a passable cell flanked by same-color bars perpendicular to direction."""
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


def _trace(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int]:
    """Trace one particle from an edge source with a provided inward direction."""
    H, W = grid.shape
    for _ in range(max(500, 4 * H * W)):
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
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _ray_has_corridor_entry(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """Check that an edge particle can actually enter a real corridor when moving inward."""
    H, W = grid.shape
    dr, dc = _DELTAS[direction]
    rr, cc = r, c
    for _ in range(max(H, W)):
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if int(grid[rr, cc]) not in {PARTICLE, BG}:
            return False
        if _has_corridor(grid, rr, cc, direction):
            return True
        rr += dr
        cc += dc
    return False


def _infer_inward_direction(grid: np.ndarray, r: int, c: int) -> str | None:
    """Infer initial direction from which edge the particle lies on."""
    H, W = grid.shape
    candidates = []
    if c == 0:
        candidates.append("right")
    if c == W - 1:
        candidates.append("left")
    if r == 0:
        candidates.append("down")
    if r == H - 1:
        candidates.append("up")
    valid = [d for d in candidates if _ray_has_corridor_entry(grid, r, c, d)]
    if len(valid) == 1:
        return valid[0]
    return None


def _edge_particles(grid: np.ndarray) -> list[tuple[int, int, str]]:
    H, W = grid.shape
    starts = []
    seen = set()
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != PARTICLE:
                continue
            if r not in {0, H - 1} and c not in {0, W - 1}:
                continue
            if (r, c) in seen:
                continue
            direction = _infer_inward_direction(grid, r, c)
            if direction is None:
                continue
            starts.append((r, c, direction))
            seen.add((r, c))
    starts.sort()
    return starts


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original left-edge-only solver, kept for anti-degeneration checks in generation."""
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

        end_r, end_c = _trace(g, r, 0, "right")
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta edge particles along same-color corridors; initial direction is inward from the edge."""
    g = np.array(grid, dtype=int)
    out = g.copy()

    for r, c, direction in _edge_particles(g):
        end_r, end_c = _trace(g, r, c, direction)
        if (end_r, end_c) != (r, c):
            out[r, c] = BG
            out[end_r, end_c] = PARTICLE

    return out


# --------------- generator (variant-aware) ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

# Channel assignment
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_MIN_PALETTE_DIVERSITY = 3

# Turn probabilities
_STRAIGHT_PROB = 0.20
_L_INTERNAL_PROB = 0.25
_U_INTERNAL_STOP_PROB = 0.80

# Vertical column allocation
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)

# Decoration
_FAKE_DECOR_RANGE = (0, 2)

# Validation
_MIN_MOVED = 4
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MAX_RETRIES = 400


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall column."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [c for c in range(c_min, c_max + 1) if all((c + d) not in used_vcols for d in _VCOL_WINDOW)]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _reserve_seed_plans(real_indices: list[int], bar_start: int, W: int, used_vcols: set[int], plans: dict[int, dict]) -> bool:
    """Guarantee visible starts on right/top/bottom by reserving three real paths."""
    if len(real_indices) < 3:
        return False
    straight_i, up_i, down_i = random.sample(real_indices, 3)
    c_up = _allocate_vcol(bar_start, W, used_vcols)
    c_down = _allocate_vcol(bar_start, W, used_vcols)
    if c_up is None or c_down is None:
        return False
    plans[straight_i] = {"kind": "straight", "preferred_start": "right"}
    plans[up_i] = {"kind": "l_edge", "direction": "up", "c_v": c_up, "stop_at": None, "preferred_start": "top"}
    plans[down_i] = {"kind": "l_edge", "direction": "down", "c_v": c_down, "stop_at": None, "preferred_start": "bottom"}
    return True


def _plan_u_turns(real_indices: list[int], bar_rows: list[int], bar_color: dict[int, int], bar_start: int, W: int, used_vcols: set[int], plans: dict[int, dict]) -> None:
    """Pair same-color unplanned real channels into U-turn plans."""
    by_color: dict[int, list[int]] = {}
    for i in real_indices:
        if i in plans:
            continue
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
                    "preferred_start": "left",
                }
                plans[i] = {"kind": "u_target"}
            else:
                plans[i] = {
                    "kind": "u",
                    "target": j,
                    "direction": "down",
                    "c_v": c_v,
                    "internal_stop": internal_stop,
                    "preferred_start": "left",
                }
                plans[j] = {"kind": "u_target"}


def _plan_remaining(real_indices: list[int], bar_rows: list[int], bar_start: int, W: int, used_vcols: set[int], plans: dict[int, dict]) -> None:
    """Assign straight or L-turn plans to remaining real channels."""
    H = bar_rows[-1] + 1
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight", "preferred_start": "left"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight", "preferred_start": "left"}
            continue
        direction = random.choice(["up", "down"])
        if random.random() < _L_INTERNAL_PROB:
            top, bot = bar_rows[i], bar_rows[i + 1]
            if direction == "up":
                cands = list(range(2, top, 2))
            else:
                cands = list(range(bot + 2, H - 1, 2))
            stop_at = random.choice(cands) if cands else None
            plans[i] = {
                "kind": "l_internal",
                "direction": direction,
                "c_v": c_v,
                "stop_at": stop_at,
                "preferred_start": "left",
            }
        else:
            plans[i] = {
                "kind": "l_edge",
                "direction": direction,
                "c_v": c_v,
                "stop_at": None,
                "preferred_start": "left",
            }


def _build_l_turn(g: np.ndarray, H: int, top: int, bot: int, row: int, color: int, c_v: int, direction: str, bar_color: dict[int, int], stop_at: int | None) -> None:
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


def _build_u_turn(g: np.ndarray, W: int, top: int, bot: int, row: int, color: int, c_v: int, direction: str, j_top: int, j_bot: int, bar_color: dict[int, int], bar_start: int, extras: list[int], internal_stop: bool) -> None:
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


def _execute_plans(g: np.ndarray, H: int, W: int, plans: dict[int, dict], bar_rows: list[int], bar_color: dict[int, int], bar_start: int, extras: list[int]) -> None:
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        if plan["kind"] in ("straight", "u_target"):
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = plan["c_v"]

        if plan["kind"] in ("l_edge", "l_internal"):
            _build_l_turn(g, H, top, bot, row, color, c_v, plan["direction"], bar_color, plan.get("stop_at"))
        elif plan["kind"] == "u":
            j_top, j_bot = bar_rows[plan["target"]], bar_rows[plan["target"] + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                color,
                c_v,
                plan["direction"],
                j_top,
                j_bot,
                bar_color,
                bar_start,
                extras,
                plan.get("internal_stop", False),
            )


def _candidate_starts_for_plan(i: int, plan: dict, bar_rows: list[int], H: int, W: int) -> list[dict]:
    """List legal boundary starts for one movable path."""
    if plan["kind"] == "u_target":
        return []
    row = bar_rows[i] + 1
    starts = [{"side": "left", "pos": (row, 0), "direction": "right"}]
    if plan["kind"] == "straight":
        starts.append({"side": "right", "pos": (row, W - 1), "direction": "left"})
    elif plan["kind"] == "l_edge":
        if plan["direction"] == "up":
            starts.append({"side": "top", "pos": (0, plan["c_v"] - 1), "direction": "down"})
        else:
            starts.append({"side": "bottom", "pos": (H - 1, plan["c_v"] - 1), "direction": "up"})
    return starts


def _pick_start(plan: dict, options: list[dict], preferred_side: str) -> dict | None:
    for option in options:
        if option["side"] == preferred_side:
            return option
    for option in options:
        if option["side"] == plan.get("preferred_start"):
            return option
    return options[0] if options else None


def _select_particle_starts(plans: dict[int, dict], bar_rows: list[int], H: int, W: int) -> list[tuple[int, int, str]]:
    """Choose 4-6 starts, forcing visible use of four edge sides."""
    options_by_path = {i: _candidate_starts_for_plan(i, plan, bar_rows, H, W) for i, plan in plans.items()}
    movable_paths = [i for i, opts in options_by_path.items() if opts]
    if len(movable_paths) < _MIN_MOVED:
        return []

    chosen: list[tuple[int, int, str]] = []
    used_paths: set[int] = set()

    required = [("right", "right"), ("top", "top"), ("bottom", "bottom")]
    for side, preferred in required:
        candidates = [i for i in movable_paths if i not in used_paths and any(opt["side"] == side for opt in options_by_path[i])]
        if not candidates:
            return []
        path_id = random.choice(candidates)
        start = _pick_start(plans[path_id], options_by_path[path_id], preferred)
        if start is None:
            return []
        chosen.append((*start["pos"], start["direction"]))
        used_paths.add(path_id)

    left_candidates = [i for i in movable_paths if i not in used_paths and any(opt["side"] == "left" for opt in options_by_path[i])]
    if not left_candidates:
        return []
    left_path = random.choice(left_candidates)
    left_start = _pick_start(plans[left_path], options_by_path[left_path], "left")
    if left_start is None:
        return []
    chosen.append((*left_start["pos"], left_start["direction"]))
    used_paths.add(left_path)

    target_n = random.randint(_MIN_MOVED, min(_MAX_MOVED, len(movable_paths)))
    remaining = [i for i in movable_paths if i not in used_paths]
    random.shuffle(remaining)
    for path_id in remaining:
        if len(chosen) >= target_n:
            break
        opts = options_by_path[path_id]
        if not opts:
            continue
        preferred = plans[path_id].get("preferred_start", "left")
        start = _pick_start(plans[path_id], opts, preferred)
        if start is None:
            start = random.choice(opts)
        chosen.append((*start["pos"], start["direction"]))
        used_paths.add(path_id)

    return chosen


def _place_particles(g: np.ndarray, starts: list[tuple[int, int, str]]) -> None:
    for r, c, _ in starts:
        g[r, c] = PARTICLE


def _count_moved(inp: np.ndarray, out: np.ndarray) -> int:
    src = {tuple(map(int, rc)) for rc in np.argwhere(inp == PARTICLE)}
    dst = {tuple(map(int, rc)) for rc in np.argwhere(out == PARTICLE)}
    return len(src - dst)


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    g = np.full((H, W), BG, dtype=int)
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

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict] = {}
    if not _reserve_seed_plans(real_indices, bar_start, W, used_vcols, plans):
        return None
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, W, used_vcols, plans)
    _plan_remaining(real_indices, bar_rows, bar_start, W, used_vcols, plans)
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

    starts = _select_particle_starts(plans, bar_rows, H, W)
    if not starts:
        return None
    _place_particles(g, starts)

    try:
        out = main(g)
    except Exception:
        return None
    if np.array_equal(out, _legacy_main(g)):
        return None

    n_moved = _count_moved(g, out)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if len(set(g.flatten().tolist())) < _MIN_DISTINCT_COLORS:
        return None

    sides = {
        "left" if c == 0 else "right" if c == W - 1 else "top" if r == 0 else "bottom"
        for r, c, _ in starts
    }
    if sides != {"left", "right", "top", "bottom"}:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
