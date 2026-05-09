# Variant of 9bbf930d
# Dimension: D10 — 真走廊与假走廊的区分 (obstacle)
# Change: 真走廊/假走廊不再靠“两侧同色/异色”区分，而是改成“1 格宽窄走廊 / 2 格宽宽走廊”。
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 仍保留原题的“横向条带 + 局部竖向转弯”的视觉骨架。
# - 真水平走廊: 两条彩色横条之间只有 1 行 BG；假水平走廊: 两条彩色横条之间有 2 行 BG。
# - 真竖向走廊: 在若干横条上挖出 1 列宽的缺口；假竖向走廊: 挖出 2 列宽的缺口。
# - 粒子仍从第 0 列向右走；遇到前方彩色阻挡时，只会转进“窄的 1 格真走廊”，不会转进“宽的 2 格假走廊”。
# - 具体小例子:
#   原题中，粒子在某行遇到阻挡后，若上方相邻格是 BG 且该格左右同色，就上转。
#   本变体中，粒子在某行遇到阻挡后，若上方相邻格是 BG 且其左右 1 格处都是彩色，就上转；
#   若下方是 2 列宽缺口，则相邻格的一侧仍是 BG，不能算真走廊，因此不转。
#
# b) 复杂度校准
# - 宽度只取 1 和 2 两档，视觉上足够明显，也不会引入过多自由度。
# - 每个样例都强制同时出现:
#   1. 至少一个“真走廊但上下横条异色”；
#   2. 至少一个“假走廊但上下横条同色”。
#   这样可直接否定原题的颜色判据，迫使观察者改用宽度判据。
#
# c) 原 main() 中依赖旧属性的隐式假设
# - _has_corridor 原先要求两侧颜色相同；本变体里该前提失效，必须改成“相邻两侧都是彩色即可”。
# - main() 的入口检测原先靠 `above == below` 找真走廊；现在必须只看“相邻上下均为彩色”。
# - 原代码后续又通过同色条的最左起点过滤入口；本变体保留“上下条从同一左边界开始”这个几何约束，
#   但不再比较颜色值，只比较左右边界是否对齐。
# - 原生成器把真假完全编码进颜色；本变体若继续这么做，就会退化回原题，因此必须在生成器里显式制造
#   “真异色 / 假同色”的反例。
#
# d) 放置合法性
# - 行布局由单行 bar 与宽度为 1/2 的 corridor 交替组成，天然不越界。
# - 真竖向走廊只占 1 列缺口，假竖向走廊占 2 列缺口；转弯列统一远离左入口，避免破坏入口对齐。
# - U-turn 会破坏目标行的左入口，避免同一真走廊被两个粒子同时启动造成歧义。
# - 所有转弯列做窗口级占用，避免两个竖向结构互相覆盖。
#
# e) 视觉可观察性
# - 2-4 个样例中，人能直接看到: 有些上下同色的“宽走廊”不参与转弯，而一些上下异色的“窄走廊”反而会转弯。
# - 规则完全基于可见的几何宽度，不依赖颜色 id、排序或任何不可见属性。
#
# f) 可视化整洁度
# - 主要对象仍是整行横条和少量竖向缺口；元素类型少，层次清楚。
# - 假走廊通过“更宽的缺口/更高的通道”制造干扰，但不会和真走廊重叠到难以辨认。
#
# g) 颜色变化可推断性
# - 颜色不再承载真假语义，只保留为障碍外观。
# - 为了显式打破旧规则，生成器强制出现“真异色 / 假同色”，但不引入新的颜色映射任务。
#
# h) 变体后的 main() 新假设
# - 真水平走廊一定是单行 BG，且其上下 1 格是彩色条。
# - 真竖向走廊一定是单列 BG 缺口，且在转入点的左右 1 格是彩色。
# - U-turn 的重入方向通过破坏目标行一侧的 bar 单元来消歧。
# - generator 通过固定的窄/宽布局、入口对齐和列占用保证这些假设成立。

from __future__ import annotations

import random
from collections import Counter

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

# Grid / palette
_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4

# Movement mix
_STRAIGHT_PROB = 0.15
_L_STOP_PROB = 0.35
_U_INTERNAL_STOP_PROB = 0.65
_OPPOSITE_FAKE_PROB = 0.65
_FAKE_VERTICAL_RANGE = (1, 3)

# Validation
_MIN_MOVED = 3
_MAX_MOVED = 6
_MAX_RETRIES = 400


def _is_colored(value: int) -> bool:
    return int(value) not in {BG, PARTICLE}


def _has_true_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True iff (r,c) lies in a 1-cell-wide corridor perpendicular to direction."""
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


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle. Straight motion is unchanged; turning now uses width, not color."""
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


def _first_colored_col(row: np.ndarray) -> int:
    for c, value in enumerate(row):
        if _is_colored(int(value)):
            return c
    return len(row)


def main(grid: np.ndarray) -> np.ndarray:
    """Move each magenta particle along aligned narrow corridors."""
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
        if top_start != bot_start or top_start > 3:
            continue

        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE

    return out


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


def _allocate_col(bar_start: int, W: int, used_cols: set[int], *, fake: bool = False) -> int | None:
    right_pad = 4 if fake else 3
    candidates = []
    for c in range(bar_start + 4, W - right_pad):
        window = range(c - 1, c + (4 if fake else 3))
        if all(col not in used_cols for col in window):
            candidates.append(c)
    if not candidates:
        return None
    c = random.choice(candidates)
    used_cols.update(range(c - 1, c + (4 if fake else 3)))
    return c


def _fake_stop_options(lanes: list[dict], lane_idx: int, direction: str) -> list[int]:
    options = []
    if direction == "up":
        for lane in lanes[:lane_idx]:
            if lane["width"] == 2:
                options.append(lane["top"])
    else:
        for lane in lanes[lane_idx + 1 :]:
            if lane["width"] == 2:
                options.append(lane["bottom"])
    return options


def _build_fake_shaft(
    g: np.ndarray,
    bar_rows: list[int],
    start_bar: int,
    c: int,
    direction: str,
    stop_bar: int | None = None,
) -> None:
    for br in bar_rows:
        if direction == "up":
            if br > start_bar:
                continue
            if stop_bar is not None and br <= stop_bar:
                continue
        else:
            if br < start_bar:
                continue
            if stop_bar is not None and br >= stop_bar:
                continue
        g[br, c : c + 2] = BG


def _build_l_turn(
    g: np.ndarray,
    lane: dict,
    bar_rows: list[int],
    bar_color: dict[int, int],
    c: int,
    direction: str,
    stop_bar: int | None,
    opposite_fake: bool,
) -> None:
    top, bottom, row = lane["top"], lane["bottom"], lane["row"]
    turn_bar = top if direction == "up" else bottom
    opp_bar = bottom if direction == "up" else top
    blocker_color = bar_color[turn_bar]
    g[row, c + 1] = blocker_color

    g[turn_bar, c] = BG
    if opposite_fake:
        _build_fake_shaft(g, bar_rows, opp_bar, c, "down" if direction == "up" else "up")
    else:
        g[opp_bar, c] = bar_color[opp_bar]

    for br in bar_rows:
        if direction == "up":
            if br >= top:
                continue
            if stop_bar is not None and br <= stop_bar:
                continue
        else:
            if br <= bottom:
                continue
            if stop_bar is not None and br >= stop_bar:
                continue
        g[br, c] = BG


def _build_u_turn(
    g: np.ndarray,
    W: int,
    lane: dict,
    target: dict,
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    c: int,
    direction: str,
    internal_stop: bool,
    opposite_fake: bool,
) -> None:
    top, bottom, row = lane["top"], lane["bottom"], lane["row"]
    j_top, j_bottom, j_row = target["top"], target["bottom"], target["row"]

    turn_bar = top if direction == "up" else bottom
    opp_bar = bottom if direction == "up" else top
    blocker_color = bar_color[turn_bar]
    g[row, c + 1] = blocker_color

    if opposite_fake:
        _build_fake_shaft(g, bar_rows, opp_bar, c, "down" if direction == "up" else "up")
    else:
        g[opp_bar, c] = bar_color[opp_bar]

    if direction == "up":
        for br in bar_rows:
            if j_bottom <= br <= top and br != j_top:
                g[br, c] = BG
        g[j_top, c] = bar_color[j_top]
        g[j_top, c - 1] = BG
    else:
        for br in bar_rows:
            if bottom <= br <= j_top and br != j_bottom:
                g[br, c] = BG
        g[j_bottom, c] = bar_color[j_bottom]
        g[j_bottom, c - 1] = BG

    # In the original task, recoloring this cell was enough because truth depended on
    # same-color matching. Here truth depends on width, so the target entrance must be
    # broken geometrically by shifting the top bar's left edge.
    g[j_top, bar_start] = BG

    if internal_stop:
        c_stop_min = c + 3
        if c_stop_min < W:
            g[j_row, random.randint(c_stop_min, W - 1)] = blocker_color


def _plan_true_lanes(lanes: list[dict], bar_start: int, W: int) -> tuple[dict[int, dict], set[int]]:
    true_indices = [lane["index"] for lane in lanes if lane["width"] == 1]
    plans: dict[int, dict] = {}
    used_cols: set[int] = set()

    max_pairs = max(0, len(true_indices) - _MIN_MOVED)
    pair_cap = min(len(true_indices) // 2, max_pairs)
    n_pairs = random.randint(0, pair_cap) if pair_cap > 0 else 0
    available = true_indices[:]
    random.shuffle(available)
    for _ in range(n_pairs):
        if len(available) < 2:
            break
        a = available.pop()
        b = available.pop()
        i, j = sorted((a, b))
        c = _allocate_col(bar_start, W, used_cols)
        if c is None:
            available.extend([a, b])
            break
        internal_stop = random.random() < _U_INTERNAL_STOP_PROB
        opposite_fake = random.random() < _OPPOSITE_FAKE_PROB
        if random.random() < 0.5:
            plans[j] = {
                "kind": "u",
                "target": i,
                "direction": "up",
                "c": c,
                "internal_stop": internal_stop,
                "opposite_fake": opposite_fake,
            }
            plans[i] = {"kind": "u_target"}
        else:
            plans[i] = {
                "kind": "u",
                "target": j,
                "direction": "down",
                "c": c,
                "internal_stop": internal_stop,
                "opposite_fake": opposite_fake,
            }
            plans[j] = {"kind": "u_target"}

    for idx in true_indices:
        if idx in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[idx] = {"kind": "straight"}
            continue
        c = _allocate_col(bar_start, W, used_cols)
        if c is None:
            plans[idx] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        stop_options = _fake_stop_options(lanes, idx, direction)
        stop_bar = None
        if stop_options and random.random() < _L_STOP_PROB:
            stop_bar = random.choice(stop_options)
        plans[idx] = {
            "kind": "l",
            "direction": direction,
            "c": c,
            "stop_bar": stop_bar,
            "opposite_fake": random.random() < _OPPOSITE_FAKE_PROB,
        }

    return plans, used_cols


def _execute_plans(
    g: np.ndarray,
    W: int,
    lanes: list[dict],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    plans: dict[int, dict],
) -> None:
    for idx, plan in plans.items():
        if plan["kind"] in {"straight", "u_target"}:
            continue
        lane = lanes[idx]
        if plan["kind"] == "l":
            _build_l_turn(
                g,
                lane,
                bar_rows,
                bar_color,
                plan["c"],
                plan["direction"],
                plan["stop_bar"],
                plan["opposite_fake"],
            )
        else:
            _build_u_turn(
                g,
                W,
                lane,
                lanes[plan["target"]],
                bar_rows,
                bar_color,
                bar_start,
                plan["c"],
                plan["direction"],
                plan["internal_stop"],
                plan["opposite_fake"],
            )


def _add_fake_verticals(
    g: np.ndarray,
    bar_rows: list[int],
    bar_start: int,
    W: int,
    used_cols: set[int],
) -> None:
    for _ in range(random.randint(*_FAKE_VERTICAL_RANGE)):
        c = _allocate_col(bar_start, W, used_cols, fake=True)
        if c is None:
            return
        top_idx = random.randint(0, len(bar_rows) - 2)
        bottom_idx = random.randint(top_idx + 1, len(bar_rows) - 1)
        for br in bar_rows[top_idx : bottom_idx + 1]:
            g[br, c : c + 2] = BG


def _dominant_bar_color(row: np.ndarray) -> int | None:
    vals = [int(v) for v in row if _is_colored(int(v))]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def analyze_variant(grid: np.ndarray) -> dict[str, int]:
    """Lightweight inspection helper used by tests to confirm the D10 behavior is present."""
    g = np.array(grid, dtype=int)
    color_counts = np.sum((g != BG) & (g != PARTICLE), axis=1)
    bar_rows = [r for r, count in enumerate(color_counts) if count >= max(4, g.shape[1] // 2)]
    true_rows = 0
    fake_rows = 0
    true_diff = 0
    fake_same = 0
    for top, bottom in zip(bar_rows, bar_rows[1:]):
        gap = bottom - top - 1
        top_color = _dominant_bar_color(g[top])
        bottom_color = _dominant_bar_color(g[bottom])
        if gap == 1:
            true_rows += 1
            if top_color is not None and bottom_color is not None and top_color != bottom_color:
                true_diff += 1
        elif gap == 2:
            fake_rows += 1
            if top_color is not None and bottom_color is not None and top_color == bottom_color:
                fake_same += 1
    return {
        "true_corridors": true_rows,
        "fake_corridors": fake_rows,
        "true_diff": true_diff,
        "fake_same": fake_same,
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)
    lanes, bar_rows = _build_lanes(H)
    bar_start = random.choice(_BAR_START_OPTIONS)
    bar_color = _assign_bar_colors(bar_rows, lanes, extras)

    if len(set(bar_color.values())) < _N_EXTRA_COLORS:
        return None

    g = np.full((H, W), BG, dtype=int)
    g[:, 0] = PARTICLE
    g[:, 1] = BG

    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]

    plans, used_cols = _plan_true_lanes(lanes, bar_start, W)
    _execute_plans(g, W, lanes, bar_rows, bar_color, bar_start, plans)
    _add_fake_verticals(g, bar_rows, bar_start, W, used_cols)

    try:
        out = main(g)
    except Exception:
        return None

    n_moved = int(np.sum((g[:, 0] == PARTICLE) & (out[:, 0] != PARTICLE)))
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None

    metrics = analyze_variant(g)
    if metrics["true_diff"] < 1 or metrics["fake_same"] < 1:
        return None
    if len(set(map(int, g.flatten()))) < 6:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
