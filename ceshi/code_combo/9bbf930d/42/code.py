# Combo variant of 9bbf930d (index 42)
# combo_spec: D11/1+D16/2
# Name: 非均匀行间距+间隙中减速
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 视觉上仍是“左侧品红粒子 + 右侧水平彩条/竖向转弯”，但相邻彩条之间不再固定只隔 1 行，
#   而是隔 1/2/3 行背景带。宽 band 内的任意一行都可能是合法水平走廊。
# - 原题水平识别依赖 `(r-1, c)` / `(r+1, c)` 恰好是两堵墙；D11 后必须改成“沿当前列向上/向下找最近的彩条行”。
# - D16 的减速规则改成：只要沿当前方向前方 1 格是背景，仍不能直接落在那里，必须再看前方第 2 格也可通行，
#   只有两格都可通行时才跳到第 2 格；于是背景间隙中的奇数位永远不可落脚。
# - 具体小例子：起点在一个 height=3 的真走廊中间行，原题会 `0->1->2->3...`；融合后会 `0->2->4...`。
#   若前方奇数宽度只剩 1 格背景再接边界/墙，原题还能走到那格，融合后会提前停住。
#   若该走廊在某个奇数墙列处向上转弯，原题会一格一格爬到边界；融合后竖向也两格一跳，
#   宽 band 会让粒子在背景带内部多次落脚，最后可能停在距离边界 1 格处。
# b) 复杂度校准
# - 为了把变化集中在“几何识别 + 落脚奇偶性”这两个维度，生成器只保留 straight 和单次 L-turn。
#   不再生成 U-turn，避免在非均匀 band 上叠加多次垂直重入导致规则过脆。
# - 每个样例强制包含：
#   1. 至少一个 height=3 的真走廊，且粒子从 band 中间行出发，证明 solver 不能只看相邻一行。
#   2. 至少一个来自宽 band 的转弯路径，证明垂直 traversal 也会受“只走偶数步”约束。
#   3. 至少两个粒子的终点和“无减速版本”不同，避免退化成纯 D11。
# c) 隐式假设审查
# - 原 `_has_corridor(..., left/right)` 假设上下相邻就是墙；融合后改为搜索最近水平墙。
# - 原 `_trace` 假设每个背景格都可落脚；融合后只有隔 1 格的 landing 才可达。
# - 原入口扫描接受任意早期走廊格；融合后必须固定 `bar_start=2`，并仅接受偶数列入口，
#   否则从 `col0` 两格一跳无法稳定落到入口。
# - 原竖向走廊只需在穿过的 bar row 上挖洞即可；融合后 landing 可能落在宽 band 的背景行，
#   但这些行天然就是背景，所以生成器只需继续保证所有穿过的 bar row 在同一列为 BG。
# d) 放置合法性
# - 所有转弯墙列都限制为奇数列，确保最后一个水平落脚点 `c_v-1` 为偶数可达列。
# - 转弯列用不重叠的 3 列窗口预留，避免两个 L-turn 互相覆盖。
# - 每个真走廊最多放一个起始粒子；这避免宽 band 中同一对墙之间多粒子同时触发造成歧义。
# - 假走廊和装饰像素避开真实竖向走廊列，防止无意中阻断真实跳跃路径。
# e) 视觉可观察性
# - 看到训练样例时，人类能直接观察到：有些粒子明明离上下彩条不只 1 格，却仍被那对彩条引导；
#   同时不少粒子会在边界前 1 格或在宽竖向 gap 内提前停下，说明背景内并非每格可达。
# - 这些现象都来自可见几何结构，不依赖颜色编号、隐藏计数或不可视排序。
# f) 可视化整洁度
# - 颜色仍只表示“上下是否同色形成真走廊”；没有额外颜色映射。
# - 同一区域最多叠加一条竖向走廊和一条水平走廊，且通过固定窗口分隔转弯列，避免画面发散。
# g) 变化的可推断性
# - 没有引入新的颜色变换规则；解题者只需推断两件事：
#   1. 宽背景带依然属于同一走廊；
#   2. 粒子在背景中按隔格跳跃，奇数位不可达。

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


def _is_passable(value: int) -> bool:
    return value in {BG, PARTICLE}


def _nearest_horizontal_walls(
    grid: np.ndarray, r: int, c: int
) -> tuple[int, int, int] | None:
    """Return the nearest colored walls above/below `(r, c)` if they match."""
    h, _ = grid.shape
    if not _is_passable(int(grid[r, c])):
        return None

    top = None
    for rr in range(r - 1, -1, -1):
        value = int(grid[rr, c])
        if value not in {BG, PARTICLE}:
            top = rr
            break

    bottom = None
    for rr in range(r + 1, h):
        value = int(grid[rr, c])
        if value not in {BG, PARTICLE}:
            bottom = rr
            break

    if top is None or bottom is None:
        return None
    a, b = int(grid[top, c]), int(grid[bottom, c])
    if a != b or a in {BG, PARTICLE}:
        return None
    return top, bottom, a


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if `(r, c)` is a legal turn target for the requested direction."""
    h, w = grid.shape
    if not _is_passable(int(grid[r, c])):
        return False
    if direction in {"left", "right"}:
        return _nearest_horizontal_walls(grid, r, c) is not None
    if c == 0 or c == w - 1:
        return False
    a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {BG, PARTICLE}


def _choose_turn(
    grid: np.ndarray, r: int, c: int, direction: str
) -> tuple[int, int, str] | None:
    options = []
    h, w = grid.shape
    for turn_dir in _PERPENDICULAR[direction]:
        tdr, tdc = _DELTAS[turn_dir]
        tr, tc = r + tdr, c + tdc
        if 0 <= tr < h and 0 <= tc < w and _has_corridor(grid, tr, tc, turn_dir):
            options.append((tr, tc, turn_dir))
    assert len(options) <= 1, "ambiguous turn: both perpendicular directions valid"
    return options[0] if options else None


def _trace_slow(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle with jump motion through all background gaps."""
    h, w = grid.shape
    direction = "right"
    for _ in range(h * w * 4):
        dr, dc = _DELTAS[direction]
        mr, mc = r + dr, c + dc
        if not (0 <= mr < h and 0 <= mc < w):
            break
        if not _is_passable(int(grid[mr, mc])):
            turn = _choose_turn(grid, r, c, direction)
            if turn is None:
                break
            r, c, direction = turn
            continue

        lr, lc = mr + dr, mc + dc
        if not (0 <= lr < h and 0 <= lc < w):
            break
        if not _is_passable(int(grid[lr, lc])):
            break
        r, c = lr, lc
    return r, c


def _trace_regular(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Reference movement with irregular walls but without jump slowdown."""
    h, w = grid.shape
    direction = "right"
    for _ in range(h * w * 4):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if _is_passable(int(grid[nr, nc])):
            r, c = nr, nc
            continue
        turn = _choose_turn(grid, r, c, direction)
        if turn is None:
            break
        r, c, direction = turn
    return r, c


def _first_color_col(grid: np.ndarray, row: int, color: int) -> int | None:
    cols = np.flatnonzero(grid[row] == color)
    if cols.size == 0:
        return None
    return int(cols[0])


def _find_entrance(grid: np.ndarray, r: int) -> tuple[int, int, int] | None:
    """Return `(entrance_col, top_row, bottom_row)` for a valid starter row."""
    _, w = grid.shape
    for c in range(2, w, 2):
        walls = _nearest_horizontal_walls(grid, r, c)
        if walls is None:
            continue
        top, bottom, color = walls
        top_start = _first_color_col(grid, top, color)
        bottom_start = _first_color_col(grid, bottom, color)
        if top_start is None or bottom_start is None:
            continue
        if top_start == bottom_start == 2:
            return c, top, bottom
    return None


def _main_common(grid: np.ndarray, trace_fn) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, _ = g.shape
    out = g.copy()

    for r in range(h):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance(g, r) is None:
            continue
        end_r, end_c = trace_fn(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through irregular-height corridors with jump motion.

    Input structure:
    - `bg=7`, `particle=6`. Only selected rows in `col 0` contain particles.
    - Colored horizontal bars begin at `col 2`, but adjacent bar rows are separated by
      1-3 background rows instead of a strict even/odd stripe pattern.

    Rule:
    - A horizontal corridor is any passable cell whose nearest colored row above and
      nearest colored row below have the same non-background color.
    - The particle keeps its current direction, but while traversing background it may
      only land every second cell. If the intermediate cell is open but the second cell
      is blocked or outside the grid, the particle stops instead of landing on the odd cell.
    - When the immediately adjacent cell in the current direction is colored, the particle
      tries to turn 90 degrees into the unique perpendicular corridor. If none exists, it stops.

    Key insight:
    - The solver must combine two geometry cues at once: wide background bands still act
      like one corridor because the nearest matching walls matter, and the particle advances
      by parity-constrained jumps rather than by unit steps.
    """
    return _main_common(grid, _trace_slow)


def _main_regular(grid: np.ndarray) -> np.ndarray:
    return _main_common(grid, _trace_regular)


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_OPTIONS = [14, 16, 18, 20]
_BAR_START = 2
_N_EXTRA_COLORS = 4
_N_BARS_RANGE = (5, 7)
_MAX_RETRIES = 1500

_MIN_MOVED = 3
_MAX_MOVED = 5

_VCOL_LEFT_PAD = 3
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)
_EXTRA_REAL_RANGE = (0, 2)
_FAKE_DECOR_RANGE = (0, 2)


def _sample_layout() -> tuple[int, list[int], list[int]] | None:
    """Sample irregular bar rows with at least one height-3 band."""
    for _ in range(300):
        n_bars = random.randint(*_N_BARS_RANGE)
        gaps = [random.randint(2, 4) for _ in range(n_bars - 1)]
        band_heights = [gap - 1 for gap in gaps]
        if max(band_heights) < 3:
            continue
        if sum(height > 1 for height in band_heights) < 2:
            continue
        if len(set(band_heights)) < 2:
            continue

        bar_rows = [0]
        cur = 0
        for gap in gaps:
            cur += gap
            bar_rows.append(cur)
        min_h = bar_rows[-1] + 1
        valid_heights = [h for h in _HEIGHT_OPTIONS if h >= min_h]
        if not valid_heights:
            continue
        h = random.choice(valid_heights)
        return h, bar_rows, band_heights
    return None


def _allocate_vcol(w: int, used_vcols: set[int]) -> int | None:
    """Reserve an odd wall column so the landing cell `c_v-1` is reachable."""
    c_min = max(_BAR_START + _VCOL_LEFT_PAD, 5)
    c_max = w - _VCOL_RIGHT_PAD
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if c % 2 == 1 and all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _band_rows(top: int, bottom: int) -> list[int]:
    return list(range(top + 1, bottom))


def _band_middle_row(top: int, bottom: int) -> int:
    rows = _band_rows(top, bottom)
    return rows[len(rows) // 2]


def _pick_turn_direction(top: int, bottom: int, h: int) -> str:
    up_len = top
    down_len = h - 1 - bottom
    if up_len == 0:
        return "down"
    if down_len == 0:
        return "up"
    return "up" if up_len >= down_len else "down"


def _paint_bars(
    g: np.ndarray, bar_rows: list[int], bar_color: dict[int, int]
) -> None:
    for br in bar_rows:
        g[br, _BAR_START:] = bar_color[br]


def _build_l_turn(
    g: np.ndarray,
    bar_rows: list[int],
    bar_color: dict[int, int],
    top: int,
    bottom: int,
    start_row: int,
    c_v: int,
    direction: str,
) -> None:
    """Build a single-turn real corridor that exits vertically to an edge."""
    color = bar_color[top]
    g[start_row, c_v] = color

    turn_bar = top if direction == "up" else bottom
    opp_bar = bottom if direction == "up" else top

    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = bar_color[turn_bar]
    g[opp_bar, c_v - 1] = bar_color[opp_bar]

    for br in bar_rows:
        if direction == "up" and br < top:
            g[br, c_v - 1] = BG
        if direction == "down" and br > bottom:
            g[br, c_v - 1] = BG


def _plan_active_bands(
    h: int, bar_rows: list[int], band_heights: list[int], w: int
) -> tuple[dict[int, dict[str, int | str]], set[int]] | None:
    n_bands = len(band_heights)
    if n_bands < _MIN_MOVED:
        return None

    height3 = [i for i, height in enumerate(band_heights) if height == 3]
    wide = [i for i, height in enumerate(band_heights) if height > 1]
    if not height3 or len(wide) < 2:
        return None

    n_active = random.randint(_MIN_MOVED, min(_MAX_MOVED, n_bands))
    showcase_mid = random.choice(height3)
    wide_turn_choices = [i for i in wide if i != showcase_mid] or wide
    showcase_turn = random.choice(wide_turn_choices)

    active = {showcase_mid, showcase_turn}
    remaining = [i for i in range(n_bands) if i not in active]
    random.shuffle(remaining)
    while len(active) < n_active and remaining:
        active.add(remaining.pop())

    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str]] = {}
    turn_count = 0

    for i in sorted(active):
        top, bottom = bar_rows[i], bar_rows[i + 1]
        height = band_heights[i]
        if i == showcase_mid:
            plans[i] = {"kind": "straight", "start_row": _band_middle_row(top, bottom)}
            continue

        make_turn = i == showcase_turn or (turn_count == 0 and random.random() < 0.5) or random.random() < 0.35
        if make_turn:
            c_v = _allocate_vcol(w, used_vcols)
            if c_v is not None:
                direction = _pick_turn_direction(top, bottom, h)
                start_row = top + 1 if direction == "up" else bottom - 1
                plans[i] = {
                    "kind": "turn",
                    "start_row": start_row,
                    "direction": direction,
                    "c_v": c_v,
                }
                turn_count += 1
                continue

        if height == 3 and random.random() < 0.6:
            start_row = _band_middle_row(top, bottom)
        else:
            start_row = random.choice(_band_rows(top, bottom))
        plans[i] = {"kind": "straight", "start_row": start_row}

    if showcase_turn not in plans or plans[showcase_turn]["kind"] != "turn":
        return None
    return plans, active


def _assign_bar_colors(
    bar_rows: list[int], real_bands: set[int], extras: list[int]
) -> dict[int, int] | None:
    bar_color = {bar_rows[0]: random.choice(extras)}
    for i in range(len(bar_rows) - 1):
        top, bottom = bar_rows[i], bar_rows[i + 1]
        if i in real_bands:
            bar_color[bottom] = bar_color[top]
        else:
            choices = [c for c in extras if c != bar_color[top]]
            if not choices:
                return None
            bar_color[bottom] = random.choice(choices)
    if len(set(bar_color.values())) < 3:
        return None
    return bar_color


def _add_fake_decor(
    g: np.ndarray,
    band_heights: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    real_bands: set[int],
    reserved_cols: set[int],
) -> None:
    for i, height in enumerate(band_heights):
        if i in real_bands:
            continue
        rows = _band_rows(bar_rows[i], bar_rows[i + 1])
        free_cols = [c for c in range(_BAR_START + 1, g.shape[1]) if c not in reserved_cols]
        if not rows or not free_cols:
            continue
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        for _ in range(n_decor):
            rr = random.choice(rows)
            cc = random.choice(free_cols)
            if g[rr, cc] == BG:
                g[rr, cc] = random.choice([bar_color[bar_rows[i]], bar_color[bar_rows[i + 1]]])


def _trace_starters(
    grid: np.ndarray, trace_fn
) -> list[tuple[int, tuple[int, int], tuple[int, int, int]]]:
    traced = []
    h, _ = grid.shape
    for r in range(h):
        if int(grid[r, 0]) != PARTICLE:
            continue
        entrance = _find_entrance(grid, r)
        if entrance is None:
            continue
        traced.append((r, trace_fn(grid, r, 0), entrance))
    return traced


def variant_metrics(
    grid: np.ndarray,
    out: np.ndarray | None = None,
    out_regular: np.ndarray | None = None,
) -> dict[str, int]:
    """Quantify both D11 and D16 signals in one sample."""
    if out is None:
        out = main(grid)
    if out_regular is None:
        out_regular = _main_regular(grid)

    slow = {r: end for r, end, _ in _trace_starters(grid, _trace_slow)}
    regular = {r: end for r, end, _ in _trace_starters(grid, _trace_regular)}

    moved = 0
    wide_moved = 0
    middle_row_moved = 0
    wide_turned = 0
    slow_diff_particles = 0

    for r, _, entrance in _trace_starters(grid, _trace_slow):
        top_row = entrance[1]
        bottom_row = entrance[2]
        band_height = bottom_row - top_row - 1
        end = slow[r]
        regular_end = regular[r]
        if end != (r, 0):
            moved += 1
            if band_height > 1:
                wide_moved += 1
            if r not in {top_row + 1, bottom_row - 1}:
                middle_row_moved += 1
            if band_height > 1 and end[0] != r:
                wide_turned += 1
        if end != regular_end:
            slow_diff_particles += 1

    return {
        "moved": moved,
        "wide_moved": wide_moved,
        "middle_row_moved": middle_row_moved,
        "wide_turned": wide_turned,
        "slow_diff_particles": slow_diff_particles,
        "output_diff_cells": int(np.count_nonzero(out != out_regular)),
    }


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    h, bar_rows, band_heights = layout
    w = random.choice(_WIDTH_OPTIONS)

    plans_bundle = _plan_active_bands(h, bar_rows, band_heights, w)
    if plans_bundle is None:
        return None
    plans, active_bands = plans_bundle

    n_bands = len(band_heights)
    remaining = [i for i in range(n_bands) if i not in active_bands]
    n_extra_real = random.randint(_EXTRA_REAL_RANGE[0], min(_EXTRA_REAL_RANGE[1], len(remaining)))
    extra_real = set(random.sample(remaining, n_extra_real))
    real_bands = set(active_bands) | extra_real

    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)
    bar_color = _assign_bar_colors(bar_rows, real_bands, extras)
    if bar_color is None:
        return None

    g = np.full((h, w), BG, dtype=int)
    _paint_bars(g, bar_rows, bar_color)

    reserved_cols: set[int] = set()
    for i, plan in plans.items():
        start_row = int(plan["start_row"])
        g[start_row, 0] = PARTICLE
        if plan["kind"] != "turn":
            continue
        top, bottom = bar_rows[i], bar_rows[i + 1]
        c_v = int(plan["c_v"])
        reserved_cols.update({c_v - 1, c_v})
        _build_l_turn(
            g,
            bar_rows,
            bar_color,
            top,
            bottom,
            start_row,
            c_v,
            str(plan["direction"]),
        )

    g[:, 1] = BG
    _add_fake_decor(g, band_heights, bar_rows, bar_color, real_bands, reserved_cols)

    try:
        out = main(g)
        out_regular = _main_regular(g)
        metrics = variant_metrics(g, out, out_regular)
    except Exception:
        return None

    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["wide_moved"] < 2:
        return None
    if metrics["middle_row_moved"] < 1:
        return None
    if metrics["wide_turned"] < 1:
        return None
    if metrics["slow_diff_particles"] < 2:
        return None
    if metrics["output_diff_cells"] < 2:
        return None
    if len(set(map(int, g.flatten()))) < 6:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
