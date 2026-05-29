# Combo variant of 9bbf930d (index 4)
# combo_spec: D5/0+D15/2
# Name: 四边粒子+边缘标记指定方向

from __future__ import annotations

import random

import numpy as np

"""
design_analysis:
  concretization:
    original_visual:
      - 原题是横向条带布局: 偶数行是彩色横条, 奇数行是背景通道, 品红粒子只在左边界, 初始方向固定向右。
      - 粒子沿 BG 前进, 在彩色阻挡前检查两个垂直方向, 若恰有一个方向存在“同色夹墙通道”就转弯。
    combo_visual:
      - 仍保留同一套横条/竖直缺口几何, 但粒子可以落在四条边上的真实路径端点。
      - 每个粒子旁边多一个固定颜色的箭头像素; 粒子到箭头像素的相对偏移就是初始方向。
      - 左边起点表现为 `(row,0)=particle, (row,1)=arrow`; 右边是 `(row,W-1)=particle, (row,W-2)=arrow`;
        顶/底边分别把箭头放在粒子正下/正上。
    traced_example:
      - 原题: `(r,0)` 的粒子直接按 `right` 开始, 穿过水平通道, 被 `(r,c_v)` 彩墙挡住后转入竖通道。
      - 融合后: 若粒子在顶边 `(0,c_v-1)`, 箭头在 `(1,c_v-1)`, solver 先从箭头偏移读出 `down`,
        然后沿竖通道下行, 在封口处被阻挡, 再按原规则检查 `left/right` 转入水平通道。
      - 差异是“起点位置 + 相邻标记”共同定义初始状态; 真正的走廊判定与转弯判定仍与原题一致。
  complexity_calibration:
    - 不改变走廊几何, 只把“起点来自哪条边”和“方向如何给出”这两个自由度显式化。
    - 箭头只用一个专用颜色, 避免引入额外颜色映射。
    - 生成器强制四条边都至少出现一次起点, 让 D5 的变化肉眼可见。
    - 箭头保持朝向网格内部的真实通路端点; 若让箭头沿边切向移动, 需要额外边界引导结构, 会把任务改得过重。
  implicit_assumptions_audit:
    - 原 `_trace` 写死 `right`; 组合后必须改成参数, 且箭头像素必须视为可通行。
    - 原 `main` 只扫描 `col 0`; 组合后必须扫描四条边, 并从唯一相邻箭头恢复方向。
    - 原 `_has_corridor` 只把 `BG/PARTICLE` 当作可通行; 组合后若不把 `ARROW` 也算 passable, 起步时就会被箭头挡住。
    - 原 generator 会无条件铺满 `col 0` 粒子; 组合后绝不能这样做, 否则会覆盖右/顶/底起点和箭头。
    - 原 `n_moved` 只统计左边被清空的粒子; 组合后要按全部边界起点统计。
    - 新 solver 额外假设: 每个有效边界粒子恰有一个 4 邻箭头。generator 通过专门放置并校验 `_edge_starts` 来保证。
  placement_legality:
    - 右边起点只放在 `straight` 通道的右端, 逆向进入不会遇到镜像歧义。
    - 顶/底边起点只放在 `l_edge` 计划生成的真实竖通道端点, 对应端点天然是 BG 且可直连内部。
    - 每条真实可移动路径最多放一个粒子, 避免同一路径两端同时放粒子导致输入输出退化。
    - 箭头永远落在粒子的内侧相邻可通行格, 不与彩墙重叠, 不越界。
  visual_observability:
    - 样例中会同时看到左/右/上/下四种边界起点, 且每个起点都带同色箭头标记, 人类可直接归纳“箭头给初始方向”。
    - 输出只移动粒子, 不删除箭头, 因此输入中的起始证据在输出里仍可对照。
    - 不使用颜色编号、排序或隐藏状态; 方向只由相邻位置关系表达。
  cleanliness:
    - 箭头只出现在边界起点旁, 不进入内部转弯区, 不与假走廊装饰混叠。
    - 走廊仍是单层结构: 彩条=墙, BG/箭头=可走区域, 品红=被移动对象。
  color_inference:
    - 颜色机制只保留原题的“同色夹墙可转弯”; 箭头颜色固定且不参与颜色映射。
  new_solver_assumptions_and_generator_guarantees:
    - solver 假设有效起点都在边界且恰有一个箭头邻居; generator 在放置后调用 `_edge_starts` 验证。
    - solver 假设不同路径终点不冲突; generator 每条路径最多选一个起点, 并沿用原始转弯列占用规则。
    - solver 假设箭头方向确实通向可通行路径; generator 只从 `_candidate_starts_for_plan` 产出的合法端点放置起点。
"""

BG = 7
PARTICLE = 6
ARROW = 2
PASSABLE = {BG, PARTICLE, ARROW}
_DELTAS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a passable cell flanked by same-color bars perpendicular to direction."""
    h, w = grid.shape
    if int(grid[r, c]) not in PASSABLE:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == h - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == w - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in PASSABLE


def _trace(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int]:
    """Trace one particle from an edge source with a provided initial direction."""
    h, w = grid.shape
    for _ in range(max(500, 4 * h * w)):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break
        if int(grid[nr, nc]) in PASSABLE:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < h and 0 <= tc < w and _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _edge_starts(grid: np.ndarray) -> list[tuple[int, int, str]]:
    """Return border particles whose unique adjacent arrow pixel defines the start direction."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts: list[tuple[int, int, str]] = []
    edge_cells: list[tuple[int, int]] = []
    for c in range(w):
        edge_cells.append((0, c))
        edge_cells.append((h - 1, c))
    for r in range(1, h - 1):
        edge_cells.append((r, 0))
        edge_cells.append((r, w - 1))

    seen = set()
    for r, c in edge_cells:
        if (r, c) in seen or int(g[r, c]) != PARTICLE:
            continue
        seen.add((r, c))
        dirs = []
        for name, (dr, dc) in _DELTAS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == ARROW:
                dirs.append(name)
        if len(dirs) == 1:
            starts.append((r, c, dirs[0]))
    starts.sort()
    return starts


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original left-edge-only solver, used only for anti-degeneration checks in generation."""
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for r in range(1, h - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, w):
            if int(g[r, c]) != BG:
                continue
            above, below = int(g[r - 1, c]), int(g[r + 1, c])
            if above == below and above not in {PARTICLE, BG}:
                entrance_col = c
                break
        if entrance_col is None:
            continue
        bar_above = next(
            (c for c in range(w) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])),
            w,
        )
        bar_below = next(
            (c for c in range(w) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])),
            w,
        )
        if bar_above != bar_below or bar_above > 3:
            continue
        end_r, end_c = _trace(g, r, 0, "right")
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move arrow-marked edge particles through same-color corridors.

    Input structure:
    - bg=7, particle=6, arrow marker=2.
    - Corridors are still formed by same-color bar pairs in the original striped layout.
    - Particles may appear on any of the four grid edges, each with exactly one adjacent arrow.

    Rule:
    - The arrow offset relative to a border particle gives its initial direction.
    - The particle moves straight through passable cells (bg, particle, arrow).
    - When a colored wall blocks it, it turns 90 degrees into the unique perpendicular
      direction whose next cell is a same-color-flanked corridor. If no such turn exists,
      it stops; if it would leave the grid, it also stops.

    Key insight:
    - Start state is no longer "every particle begins from the left moving right".
      You must first read each edge particle's adjacent arrow to recover the initial
      heading, then apply the original same-color corridor-turning rule unchanged.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    starts = _edge_starts(g)
    end_positions = [_trace(g, r, c, direction) for r, c, direction in starts]

    for r, c, _ in starts:
        out[r, c] = BG
    for end_r, end_c in end_positions:
        out[end_r, end_c] = PARTICLE
    return out


# --------------- generator ---------------

# Grid dimensions
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_BAR_COLORS = 3

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
_MAX_RETRIES = 500


def _allocate_vcol(bar_start: int, w: int, used_vcols: set[int]) -> int | None:
    """Reserve a 3-column window for a vertical corridor and return the wall column."""
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = w - _VCOL_RIGHT_PAD
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


def _reserve_seed_plans(
    real_indices: list[int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> bool:
    """Guarantee one straight, one top-entry, and one bottom-entry real path."""
    if len(real_indices) < 3:
        return False
    straight_i, up_i, down_i = random.sample(real_indices, 3)
    c_up = _allocate_vcol(bar_start, w, used_vcols)
    c_down = _allocate_vcol(bar_start, w, used_vcols)
    if c_up is None or c_down is None:
        return False
    plans[straight_i] = {"kind": "straight", "preferred_start": "right"}
    plans[up_i] = {
        "kind": "l_edge",
        "direction": "up",
        "c_v": c_up,
        "stop_at": None,
        "preferred_start": "top",
    }
    plans[down_i] = {
        "kind": "l_edge",
        "direction": "down",
        "c_v": c_down,
        "stop_at": None,
        "preferred_start": "bottom",
    }
    return True


def _plan_u_turns(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
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
            c_v = _allocate_vcol(bar_start, w, used_vcols)
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


def _plan_remaining(
    real_indices: list[int],
    bar_rows: list[int],
    bar_start: int,
    w: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    """Assign straight or L-turn plans to remaining real channels."""
    h = bar_rows[-1] + 1
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight", "preferred_start": "left"}
            continue
        c_v = _allocate_vcol(bar_start, w, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight", "preferred_start": "left"}
            continue
        direction = random.choice(["up", "down"])
        if random.random() < _L_INTERNAL_PROB:
            top, bot = bar_rows[i], bar_rows[i + 1]
            if direction == "up":
                cands = list(range(2, top, 2))
            else:
                cands = list(range(bot + 2, h - 1, 2))
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


def _build_l_turn(
    g: np.ndarray,
    h: int,
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
        for r_br in range(bot + 2, h, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g: np.ndarray,
    w: int,
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
    bar_colors: list[int],
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

    other_colors = [c for c in bar_colors if c != color]
    if other_colors:
        g[j_top, bar_start] = random.choice(other_colors)

    if internal_stop:
        j_row = j_top + 1
        c_stop_min, c_stop_max = c_v + 3, w - 1
        if c_stop_min <= c_stop_max:
            g[j_row, random.randint(c_stop_min, c_stop_max)] = color


def _execute_plans(
    g: np.ndarray,
    h: int,
    w: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    bar_colors: list[int],
) -> None:
    """Render all planned turns onto the grid."""
    for i, plan in plans.items():
        kind = str(plan["kind"])
        if kind in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        color = bar_color[top]
        c_v = int(plan["c_v"])

        if kind in {"l_edge", "l_internal"}:
            _build_l_turn(
                g,
                h,
                top,
                bot,
                row,
                color,
                c_v,
                str(plan["direction"]),
                bar_color,
                plan.get("stop_at"),
            )
        elif kind == "u":
            j_top, j_bot = bar_rows[int(plan["target"])], bar_rows[int(plan["target"]) + 1]
            _build_u_turn(
                g,
                w,
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
                bar_colors,
                bool(plan.get("internal_stop", False)),
            )


def _candidate_starts_for_plan(
    i: int,
    plan: dict[str, object],
    bar_rows: list[int],
    h: int,
    w: int,
) -> list[dict[str, object]]:
    """List legal boundary starts for one movable path."""
    if str(plan["kind"]) == "u_target":
        return []
    row = bar_rows[i] + 1
    starts: list[dict[str, object]] = [
        {"side": "left", "pos": (row, 0), "direction": "right"}
    ]
    if str(plan["kind"]) == "straight":
        starts.append({"side": "right", "pos": (row, w - 1), "direction": "left"})
    elif str(plan["kind"]) == "l_edge":
        if str(plan["direction"]) == "up":
            starts.append({"side": "top", "pos": (0, int(plan["c_v"]) - 1), "direction": "down"})
        else:
            starts.append(
                {"side": "bottom", "pos": (h - 1, int(plan["c_v"]) - 1), "direction": "up"}
            )
    return starts


def _pick_start(
    plan: dict[str, object],
    options: list[dict[str, object]],
    preferred_side: str,
) -> dict[str, object] | None:
    for option in options:
        if option["side"] == preferred_side:
            return option
    for option in options:
        if option["side"] == plan.get("preferred_start"):
            return option
    return options[0] if options else None


def _select_particle_starts(
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    h: int,
    w: int,
) -> list[tuple[int, int, str]]:
    """Choose 4-6 starts, forcing visible use of all four edge sides."""
    options_by_path = {
        i: _candidate_starts_for_plan(i, plan, bar_rows, h, w)
        for i, plan in plans.items()
    }
    movable_paths = [i for i, opts in options_by_path.items() if opts]
    if len(movable_paths) < _MIN_MOVED:
        return []

    chosen: list[tuple[int, int, str]] = []
    used_paths: set[int] = set()

    required = [("right", "right"), ("top", "top"), ("bottom", "bottom")]
    for side, preferred in required:
        candidates = [
            i
            for i in movable_paths
            if i not in used_paths and any(opt["side"] == side for opt in options_by_path[i])
        ]
        if not candidates:
            return []
        path_id = random.choice(candidates)
        start = _pick_start(plans[path_id], options_by_path[path_id], preferred)
        if start is None:
            return []
        pos = start["pos"]
        chosen.append((int(pos[0]), int(pos[1]), str(start["direction"])))
        used_paths.add(path_id)

    left_candidates = [
        i
        for i in movable_paths
        if i not in used_paths and any(opt["side"] == "left" for opt in options_by_path[i])
    ]
    if not left_candidates:
        return []
    left_path = random.choice(left_candidates)
    left_start = _pick_start(plans[left_path], options_by_path[left_path], "left")
    if left_start is None:
        return []
    pos = left_start["pos"]
    chosen.append((int(pos[0]), int(pos[1]), str(left_start["direction"])))
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
        preferred = str(plans[path_id].get("preferred_start", "left"))
        start = _pick_start(plans[path_id], opts, preferred)
        if start is None:
            start = random.choice(opts)
        pos = start["pos"]
        chosen.append((int(pos[0]), int(pos[1]), str(start["direction"])))
        used_paths.add(path_id)

    return chosen


def _place_particles_and_arrows(g: np.ndarray, starts: list[tuple[int, int, str]]) -> bool:
    """Place particles on the border and their unique adjacent arrow markers."""
    h, w = g.shape
    arrow_cells: set[tuple[int, int]] = set()
    for r, c, direction in starts:
        dr, dc = _DELTAS[direction]
        ar, ac = r + dr, c + dc
        if not (0 <= ar < h and 0 <= ac < w):
            return False
        if int(g[ar, ac]) not in {BG, PARTICLE}:
            return False
        if (ar, ac) in arrow_cells:
            return False
        g[r, c] = PARTICLE
        g[ar, ac] = ARROW
        arrow_cells.add((ar, ac))
    return True


def _count_moved(inp: np.ndarray, out: np.ndarray) -> int:
    src = {tuple(map(int, rc)) for rc in np.argwhere(inp == PARTICLE)}
    dst = {tuple(map(int, rc)) for rc in np.argwhere(out == PARTICLE)}
    return len(src - dst)


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(_HEIGHT_OPTIONS)
    w = random.randint(*_WIDTH_RANGE)
    pool = [c for c in range(10) if c not in (BG, PARTICLE, ARROW)]
    bar_colors = random.sample(pool, _N_BAR_COLORS)

    g = np.full((h, w), BG, dtype=int)
    bar_rows = list(range(0, h, 2))
    n_channels = len(bar_rows) - 1

    n_real = random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0]
    n_real = min(n_real, n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]

    bar_color = {bar_rows[0]: random.choice(bar_colors)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            bar_color[bot] = random.choice([c for c in bar_colors if c != bar_color[top]])
    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    bar_start = random.choice(_BAR_START_OPTIONS)
    for br in bar_rows:
        g[br, bar_start:w] = bar_color[br]

    real_indices = [i for i, spec in enumerate(channel_spec) if spec == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}
    if not _reserve_seed_plans(real_indices, bar_start, w, used_vcols, plans):
        return None
    _plan_u_turns(real_indices, bar_rows, bar_color, bar_start, w, used_vcols, plans)
    _plan_remaining(real_indices, bar_rows, bar_start, w, used_vcols, plans)
    _execute_plans(g, h, w, plans, bar_rows, bar_color, bar_start, bar_colors)

    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, w), min(n_decor, w - bar_start - 1))
        for c_decor in cols:
            if g[row, c_decor] == BG:
                g[row, c_decor] = random.choice([bar_color[top], bar_color[bot]])

    starts = _select_particle_starts(plans, bar_rows, h, w)
    if not starts:
        return None
    if not _place_particles_and_arrows(g, starts):
        return None

    parsed_starts = _edge_starts(g)
    if len(parsed_starts) != len(starts):
        return None

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
        "left" if c == 0 else "right" if c == w - 1 else "top" if r == 0 else "bottom"
        for r, c, _ in parsed_starts
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
