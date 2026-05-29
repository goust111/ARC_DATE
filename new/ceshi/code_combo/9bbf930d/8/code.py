# Combo variant of 9bbf930d (index 8)
# combo_spec: D8/0+D10/2
# Name: 条对部分断裂+间距区分真假走廊

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

# ---------------------------------------------------------------------------
# Design analysis for the combo variant D8/0 + D10/2
#
# a) 具体化与行为追踪
# - 原题: 真水平走廊 = 单行 BG，且同一列的上下 bar 同色；假走廊 = 同结构但异色。
# - D8/0 后: bar row 不再整行连续，而是断成虚线；人类会把 1 格缺口视作同一条 rail。
# - D10/2 后: 真假不再看颜色，而看通道宽度。真走廊宽 1 格，假走廊宽 2 格。
# - 本融合版: 同时应用两点。于是一个“真水平走廊”长成:
#     top bar : C . C . C . C
#     channel : . . . . . . .
#     bot bar : . C . C . C .
#   这里上下并不在同一列重合，所以旧 solver 会看漏；但人类能看出它们围着同一条 1 格窄通道。
# - 对比的小例子:
#     top : [1, 7, 1]
#     mid : [7, 7, 7]
#     bot : [7, 2, 7]
#   原题逻辑在 mid[1] 失败，因为同列既不连续也不同色。
#   融合逻辑在 mid[1] 成功，因为上/下 bar 在半径 1 内都可见，且中间通道宽度仍是 1 格。
# - 对宽假走廊:
#     top : [1, 7, 1]
#     mid1: [7, 7, 7]
#     mid2: [7, 7, 7]
#     bot : [7, 1, 7]
#   无论颜色如何，mid1 和 mid2 都不是“被上下相邻 bar 紧夹的一行”，因此是假走廊。
#
# b) 复杂度校准
# - 断裂只允许 1 格节拍式虚线，不引入不规则碎片；宽度只取 1/2 两档，不引入更多档位。
# - 每个样例都强制同时出现:
#   1. 至少一个真走廊但上下 bar 异色；
#   2. 至少一个假走廊但上下 bar 同色。
#   这样颜色规则会被明确推翻，必须改用“宽度 + 断裂支撑”规则。
# - 真走廊仍保留直行和 L 转弯两类，避免退化成只看某一行是否移动的静态分类题。
#
# c) 隐式假设审查
# - 原 _has_corridor(水平方向) 假设上下 bar 必须在同一列精确出现；融合后失效，改为“半径 1 内可见 bar 支撑”。
# - 原 _has_corridor(竖直方向) 假设左右必须同色；融合后颜色无语义，改为“左右都为彩色即可”。
# - 原 main() 的入口检测依赖 `above == below`；融合后不再比较颜色，只看窄走廊几何结构。
# - 原 main() 的左边界校验假设上下 bar 左端完全同列；断裂后允许相差 1 列，因为相邻 bar row 采用相反相位。
# - 新冲突点: 若生成器随意在 turn 附近断裂，竖直转入会失效。为此生成器对 turn flank 用 must_keep/must_bg 约束。
#
# d) 放置合法性
# - 行布局沿用 D10/2: 宽 1 lane 占 2 行高度(top bar, channel, bot bar)，宽 2 lane 占 3 行高度。
# - 所有真转弯列都预留 3 列窗口 [c-1, c, c+1]，避免不同竖井互相覆盖。
# - turn cell 只在一侧 bar 打开，另一侧 bar 保持封口，因此 `_trace()` 中不会出现上下两个方向同时可转的歧义。
# - bar row 最终统一渲染为虚线，再叠加 must_keep/must_bg，保证不越界也不破坏必要结构。
#
# e) 视觉可观察性
# - 在 2-4 个样例里，人类会同时看到:
#   1. 上下异色但粒子照样进入并前进的窄走廊；
#   2. 上下同色但粒子停在起点的宽走廊；
#   3. 真走廊的 rail 是断裂虚线，不能按“同列实线配对”来识别。
# - 规则完全来自可见几何，不依赖颜色编号、排序、计数器或隐藏状态。
#
# f) 可视化整洁度
# - 每个 bar row 只使用一种颜色，背景缺口统一是 BG，视觉上仍是“几条彩轨 + 少量阻挡”的骨架。
# - 本融合版不再额外加入复杂 fake U-turn 或装饰块，避免虚线、宽 lane、转弯三者叠加后过乱。
#
# g) 变化的可推断性
# - 颜色只保留外观，不承担真假语义。
# - 为了让这一点在少量样例中可归纳，生成器强制出现“真异色 / 假同色”反例。
# - 断裂模式在所有 bar row 上重复出现，人类容易归纳出“断一格仍算同一条 rail”。
#
# h) 新 solver 假设与 generator 对应保证
# - solver 假设真水平走廊一定是单行 BG，且其相邻两条 bar row 在半径 1 内都可见彩色支撑。
#   generator 通过宽 1 lane + 相反相位虚线保证这一点。
# - solver 假设真竖直走廊一定是单列 BG 缺口，且该格左右紧邻彩色 flank。
#   generator 通过 must_keep/must_bg 约束 turn 列保证这一点。
# - solver 假设入口附近两条 bar 的最左可见位置只会相差 0 或 1。
#   generator 通过全局交错虚线相位保证这一点。
# ---------------------------------------------------------------------------


def _is_colored(value: int) -> bool:
    return int(value) not in {BG, PARTICLE}


def _row_has_local_support(grid: np.ndarray, bar_row: int, c: int, radius: int = 1) -> bool:
    if not (0 <= bar_row < grid.shape[0]):
        return False
    left = max(0, c - radius)
    right = min(grid.shape[1] - 1, c + radius)
    for cc in range(left, right + 1):
        if _is_colored(int(grid[bar_row, cc])):
            return True
    return False


def _first_colored_col(row: np.ndarray) -> int:
    for c, value in enumerate(row):
        if _is_colored(int(value)):
            return c
    return len(row)


def _dominant_row_color(row: np.ndarray) -> int | None:
    for value in row:
        iv = int(value)
        if _is_colored(iv):
            return iv
    return None


def _has_true_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True iff (r,c) is inside a real narrow corridor under the fused rule."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        return _row_has_local_support(grid, r - 1, c) and _row_has_local_support(grid, r + 1, c)
    if c == 0 or c == W - 1:
        return False
    return _is_colored(int(grid[r, c - 1])) and _is_colored(int(grid[r, c + 1]))


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle using the fused corridor predicate."""
    H, W = grid.shape
    direction = "right"
    for _ in range(500):
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
            if 0 <= tr < H and 0 <= tc < W and _has_true_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through broken narrow corridors.

    Input structure:
    - bg=7, particle=6. Column 0 is filled with particles, column 1 is bg.
    - Colored bar rows are rendered as dashed horizontal rails.
    - Real corridors are 1-row channels between adjacent bar rows; fake corridors are 2-row gaps.

    Rule:
    - A particle starts from each row's col 0 and moves right through bg cells.
    - When blocked by a colored cell, it may turn 90 degrees into the unique perpendicular
      1-cell-wide corridor. Horizontal corridor recognition tolerates 1-cell rail gaps.
    - The particle stops at the grid edge or when no valid narrow turn exists.

    Key insight:
    - Corridor truth is determined jointly by geometry and local support: a real hallway is
      the narrow 1-row gap whose two dashed rails are still visible within radius 1, even if
      the rails are not aligned column-by-column and even if their colors differ.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()

    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue

        entrance_col = None
        for c in range(2, W):
            if _has_true_corridor(g, r, c, "right"):
                entrance_col = c
                break
        if entrance_col is None:
            continue

        top_start = _first_colored_col(g[r - 1])
        bot_start = _first_colored_col(g[r + 1])
        if max(top_start, bot_start) > 4 or abs(top_start - bot_start) > 1:
            continue

        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


# ---------------------------------------------------------------------------
# Baselines used only for generator-side anti-degeneration checks.
# ---------------------------------------------------------------------------


def _exact_width_has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """Width-based corridor test without D8-style gap tolerance."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return False
        return _is_colored(int(grid[r - 1, c])) and _is_colored(int(grid[r + 1, c]))
    if c == 0 or c == W - 1:
        return False
    return _is_colored(int(grid[r, c - 1])) and _is_colored(int(grid[r, c + 1]))


def _exact_width_trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    direction = "right"
    H, W = grid.shape
    for _ in range(500):
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
            if 0 <= tr < H and 0 <= tc < W and _exact_width_has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        if len(options) > 1:
            raise AssertionError("ambiguous turn: both perpendicular directions valid")
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _exact_width_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    for r in range(1, H - 1):
        if int(g[r, 0]) != PARTICLE:
            continue
        entrance_col = None
        for c in range(2, W):
            if _exact_width_has_corridor(g, r, c, "right"):
                entrance_col = c
                break
        if entrance_col is None:
            continue
        top_start = _first_colored_col(g[r - 1])
        bot_start = _first_colored_col(g[r + 1])
        if max(top_start, bot_start) > 4 or abs(top_start - bot_start) > 1:
            continue
        end_r, end_c = _exact_width_trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

_STRAIGHT_PROB = 0.30
_MIN_MOVED = 3
_MAX_MOVED = 6
_MAX_RETRIES = 500
_MIN_VARIANT_ONLY_MOVED = 2


def _row_layout_solutions(H: int) -> list[tuple[int, int]]:
    solutions = []
    for n_true in range(_MIN_MOVED, _MAX_MOVED + 1):
        rem = (H - 1) - 2 * n_true
        if rem >= 0 and rem % 3 == 0:
            n_fake = rem // 3
            if n_fake >= 2:
                solutions.append((n_true, n_fake))
    return solutions


def _build_lanes(H: int) -> tuple[list[dict], list[int]]:
    n_true, n_fake = random.choice(_row_layout_solutions(H))
    widths = [1] * n_true + [2] * n_fake
    random.shuffle(widths)

    lanes = []
    bar_rows = [0]
    cur = 0
    for idx, width in enumerate(widths):
        top = cur
        bottom = cur + width + 1
        lane = {
            "index": idx,
            "width": width,
            "top": top,
            "bottom": bottom,
            "rows": list(range(top + 1, bottom)),
        }
        if width == 1:
            lane["row"] = top + 1
        lanes.append(lane)
        bar_rows.append(bottom)
        cur = bottom
    assert cur == H - 1
    return lanes, bar_rows


def _lane_relations(lanes: list[dict]) -> list[str]:
    relations = [random.choice(["same", "diff"]) for _ in lanes]
    true_indices = [lane["index"] for lane in lanes if lane["width"] == 1]
    fake_indices = [lane["index"] for lane in lanes if lane["width"] == 2]
    relations[random.choice(true_indices)] = "diff"
    relations[random.choice(fake_indices)] = "same"
    return relations


def _assign_bar_colors(bar_rows: list[int], lanes: list[dict], extras: list[int]) -> dict[int, int]:
    relations = _lane_relations(lanes)
    colors = {bar_rows[0]: random.choice(extras)}
    for lane, relation in zip(lanes, relations):
        top = lane["top"]
        bottom = lane["bottom"]
        if relation == "same":
            colors[bottom] = colors[top]
        else:
            colors[bottom] = random.choice([c for c in extras if c != colors[top]])
    return colors


def _turn_phase_for_lane(lane: dict, direction: str) -> int:
    return lane["index"] % 2 if direction == "up" else (lane["index"] + 1) % 2


def _allocate_turn_col(
    bar_start: int, W: int, used_cols: set[int], required_gap_phase: int
) -> int | None:
    candidates = []
    for c in range(bar_start + 4, W - 2):
        if (c - bar_start) % 2 != required_gap_phase:
            continue
        if all(col not in used_cols for col in range(c - 1, c + 2)):
            candidates.append(c)
    if not candidates:
        return None
    c = random.choice(candidates)
    used_cols.update(range(c - 1, c + 2))
    return c


def _plan_true_lanes(lanes: list[dict], bar_start: int, W: int) -> dict[int, dict]:
    true_indices = [lane["index"] for lane in lanes if lane["width"] == 1]
    plans: dict[int, dict] = {}
    used_cols: set[int] = set()
    made_turn = False

    shuffled = true_indices[:]
    random.shuffle(shuffled)
    for idx in shuffled:
        lane = lanes[idx]
        if random.random() < _STRAIGHT_PROB:
            plans[idx] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        c = _allocate_turn_col(bar_start, W, used_cols, _turn_phase_for_lane(lane, direction))
        if c is None:
            plans[idx] = {"kind": "straight"}
            continue
        plans[idx] = {"kind": "l", "direction": direction, "c": c}
        made_turn = True

    if not made_turn and true_indices:
        for idx in true_indices:
            lane = lanes[idx]
            for direction in random.sample(["up", "down"], 2):
                c = _allocate_turn_col(bar_start, W, used_cols, _turn_phase_for_lane(lane, direction))
                if c is None:
                    continue
                plans[idx] = {"kind": "l", "direction": direction, "c": c}
                made_turn = True
                break
            if made_turn:
                break
    return plans


def _build_l_turn(
    g: np.ndarray,
    lane: dict,
    bar_rows: list[int],
    bar_color: dict[int, int],
    c: int,
    direction: str,
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
) -> None:
    top, bottom, row = lane["top"], lane["bottom"], lane["row"]
    turn_bar = top if direction == "up" else bottom
    opp_bar = bottom if direction == "up" else top

    g[row, c + 1] = bar_color[turn_bar]
    must_bg[turn_bar].add(c)
    must_keep[turn_bar].update({c - 1, c + 1})
    must_keep[opp_bar].add(c)

    if direction == "up":
        shaft_rows = [br for br in bar_rows if br < top]
    else:
        shaft_rows = [br for br in bar_rows if br > bottom]
    for br in shaft_rows:
        must_bg[br].add(c)
        must_keep[br].update({c - 1, c + 1})


def _render_bar_rows(
    g: np.ndarray,
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    must_keep: dict[int, set[int]],
    must_bg: dict[int, set[int]],
) -> None:
    W = g.shape[1]
    for idx, br in enumerate(bar_rows):
        phase = idx % 2
        color = bar_color[br]
        for c in range(bar_start, W):
            if c in must_bg[br]:
                g[br, c] = BG
            elif c in must_keep[br] or (c - bar_start) % 2 == phase:
                g[br, c] = color
            else:
                g[br, c] = BG


def _moved_rows(inp: np.ndarray, out: np.ndarray) -> set[int]:
    return {
        r
        for r in range(inp.shape[0])
        if int(inp[r, 0]) == PARTICLE and int(out[r, 0]) != PARTICLE
    }


def analyze_variant(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    colored_counts = np.sum((g != BG) & (g != PARTICLE), axis=1)
    bar_rows = [r for r, count in enumerate(colored_counts) if int(count) >= 3]

    metrics = {
        "true_corridors": 0,
        "fake_corridors": 0,
        "true_diff": 0,
        "fake_same": 0,
        "true_diff_moved": 0,
        "variant_only_moved": 0,
    }

    fused_out = main(g)
    exact_out = _exact_width_main(g)
    fused_moved = _moved_rows(g, fused_out)
    exact_moved = _moved_rows(g, exact_out)
    metrics["variant_only_moved"] = len(fused_moved - exact_moved)

    for top, bottom in zip(bar_rows, bar_rows[1:]):
        gap = bottom - top - 1
        top_color = _dominant_row_color(g[top])
        bottom_color = _dominant_row_color(g[bottom])
        if gap == 1:
            metrics["true_corridors"] += 1
            if top_color is not None and bottom_color is not None and top_color != bottom_color:
                metrics["true_diff"] += 1
                if top + 1 in fused_moved:
                    metrics["true_diff_moved"] += 1
        elif gap == 2:
            metrics["fake_corridors"] += 1
            if top_color is not None and top_color == bottom_color:
                metrics["fake_same"] += 1

    return metrics


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in {BG, PARTICLE}], _N_EXTRA_COLORS)

    lanes, bar_rows = _build_lanes(H)
    bar_start = random.choice(_BAR_START_OPTIONS)
    bar_color = _assign_bar_colors(bar_rows, lanes, extras)
    if len(set(bar_color.values())) < _N_EXTRA_COLORS:
        return None

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    must_keep = {br: set() for br in bar_rows}
    must_bg = {br: set() for br in bar_rows}

    plans = _plan_true_lanes(lanes, bar_start, W)
    for lane in lanes:
        if lane["width"] != 1:
            continue
        plan = plans[lane["index"]]
        if plan["kind"] == "l":
            _build_l_turn(
                g,
                lane,
                bar_rows,
                bar_color,
                plan["c"],
                plan["direction"],
                must_keep,
                must_bg,
            )

    _render_bar_rows(g, bar_rows, bar_color, bar_start, must_keep, must_bg)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    try:
        out = main(g)
    except Exception:
        return None

    moved = _moved_rows(g, out)
    if not (_MIN_MOVED <= len(moved) <= _MAX_MOVED):
        return None
    if len(set(map(int, g.flatten()))) < 6:
        return None

    metrics = analyze_variant(g)
    if metrics["true_diff"] < 1:
        return None
    if metrics["fake_same"] < 1:
        return None
    if metrics["true_diff_moved"] < 1:
        return None
    if metrics["variant_only_moved"] < _MIN_VARIANT_ONLY_MOVED:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
