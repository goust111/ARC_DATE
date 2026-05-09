# Variant of 9bbf930d
# Dimension: D11 — 网格布局：条纹结构 (grid_shape)
# Change: 严格条纹布局——偶数行为条行，奇数行为通道行/背景行；输出与输入同尺寸。
#         → 非均匀行间距：条行之间改为 1-3 行不等的背景带，走廊高度随位置变化。
#
# Design analysis
# ---------------
# variant_spec:
#   core_visual:
#     - 仍然是“水平彩条 + 左侧粒子 + 局部竖向转弯”的题，但条行不再固定在偶数行。
#     - 任意相邻两条彩条之间可以有 1/2/3 行背景带；这整片背景带肉眼看起来属于同一个水平走廊区间。
#     - 真走廊仍由上下同色条行夹出；假走廊由上下异色条行夹出。
#     - 为了避免宽走廊里整列粒子都被激活，生成器只在每个真走廊里放一个起始粒子。
#   behavior_trace:
#     - 原题在 `main()` 中默认“当前行的上一行/下一行就是走廊的两堵墙”。
#     - 现在若粒子位于一个 3 行高的走廊带中间行，那么上一行和下一行都还是背景，原判定会把它误判为无走廊。
#     - 新逻辑改成：在当前列向上、向下搜索最近的非背景条行；若二者同色，则当前格属于水平走廊。
#     - 转弯语义不改：只有在被前方彩色块挡住时，才检查垂直方向的候选格；候选格仍要求左右紧邻同色墙。
#   complexity_calibration:
#     - 只改变“如何识别水平走廊”的几何前提，不引入新的颜色映射或多义规则。
#     - 每个样例强制同时出现至少两种 band 高度，并且至少两个已移动粒子来自高度 > 1 的 band，
#       否则视觉上会退化回原题。
#     - 路径类型压到 `straight` / `L-turn-to-edge` 两类，避免为了 D11 再额外引入多次转弯造成杂讯。
#   implicit_assumption_audit:
#     - 原 `_has_corridor(..., left/right)` 依赖 `r-1` / `r+1` 恰好是条行；在本变体中失效，必须搜索最近条行。
#     - 原 `main()` 的入口检测也直接读 `r-1` / `r+1`；同样改成基于最近条行的入口检测。
#     - 原生成器用“全列都是粒子”且只有奇数行会合法进入走廊；现在如果保留这一点，
#       宽 band 会让多行一起被激活，变成与 D11 无关的数量变化，因此生成器改成“只在选定起始行放粒子”。
#   placement_legality:
#     - 条行始终从同一个 `bar_start` 向右延展，因此每个 band 的上下边界清晰。
#     - 每个转弯列使用互不重叠的 3 列窗口，避免不同 L-turn 的缺口互相覆盖。
#     - L-turn 只通向边缘，不做内部死胡同或 U-turn，避免竖向运动中被其它真走廊再次吸入而产生歧义。
#   observability:
#     - 人类从 2-3 个样例就能看到：有的粒子位于“离上下彩条很远”的背景带里，仍然会被同一对彩条引导。
#     - 若用原题规则，只看相邻一行，这些粒子不会动；因此 D11 的变化在输入/输出上可见。
#   cleanliness:
#     - 颜色只承担“同色/异色夹墙”的可见结构作用，不参与排序或隐藏编码。
#     - 不在同一区域叠太多机制：宽 band、假走廊、单次 L-turn 已足够体现新布局。

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


def _nearest_horizontal_walls(
    grid: np.ndarray, r: int, c: int
) -> tuple[int, int, int] | None:
    """Return (top_row, bottom_row, color) of the nearest matching bar pair."""
    H, _ = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return None

    top = None
    for rr in range(r - 1, -1, -1):
        val = int(grid[rr, c])
        if val not in {PARTICLE, BG}:
            top = rr
            break

    bottom = None
    for rr in range(r + 1, H):
        val = int(grid[rr, c])
        if val not in {PARTICLE, BG}:
            bottom = rr
            break

    if top is None or bottom is None:
        return None
    a, b = int(grid[top, c]), int(grid[bottom, c])
    if a != b or a in {PARTICLE, BG}:
        return None
    return top, bottom, a


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    """True if (r,c) is a valid turn target for the requested direction."""
    H, W = grid.shape
    if int(grid[r, c]) not in {PARTICLE, BG}:
        return False
    if direction in {"left", "right"}:
        return _nearest_horizontal_walls(grid, r, c) is not None
    if c == 0 or c == W - 1:
        return False
    a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in {PARTICLE, BG}


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle from (r, c) until edge or full blockage."""
    H, W = grid.shape
    direction = "right"
    for _ in range(H * W * 4):
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


def _first_color_col(grid: np.ndarray, row: int, color: int) -> int | None:
    cols = np.flatnonzero(grid[row] == color)
    if cols.size == 0:
        return None
    return int(cols[0])


def _find_entrance(grid: np.ndarray, r: int) -> tuple[int, int, int] | None:
    """Return (entrance_col, top_row, bottom_row) for a valid starter row."""
    _, W = grid.shape
    for c in range(2, W):
        walls = _nearest_horizontal_walls(grid, r, c)
        if walls is None:
            continue
        top, bottom, color = walls
        top_start = _first_color_col(grid, top, color)
        bottom_start = _first_color_col(grid, bottom, color)
        if top_start is None or bottom_start is None:
            continue
        if top_start == bottom_start and top_start <= 3:
            return c, top, bottom
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through irregular-height corridors."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    out = g.copy()

    for r in range(H):
        if int(g[r, 0]) != PARTICLE:
            continue
        if _find_entrance(g, r) is None:
            continue
        end_r, end_c = _trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


# --------------- generator ---------------

_HEIGHT_RANGE = (17, 23)
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_N_BARS_RANGE = (5, 7)
_MIN_MOVED = 3
_MAX_MOVED = 5
_STRAIGHT_PROB = 0.35
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)
_MAX_RETRIES = 1500


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


def _sample_layout() -> tuple[int, list[int], list[int]] | None:
    for _ in range(200):
        n_bars = random.randint(*_N_BARS_RANGE)
        gaps = [random.randint(2, 4) for _ in range(n_bars - 1)]
        if len(set(gaps)) < 2 or sum(g > 2 for g in gaps) < 2:
            continue
        bar_rows = [0]
        cur = 0
        for gap in gaps:
            cur += gap
            bar_rows.append(cur)
        min_h = bar_rows[-1] + 1
        if min_h > _HEIGHT_RANGE[1]:
            continue
        H = random.randint(max(_HEIGHT_RANGE[0], min_h), _HEIGHT_RANGE[1])
        return H, bar_rows, [gap - 1 for gap in gaps]
    return None


def _paint_bars(
    g: np.ndarray, bar_rows: list[int], bar_start: int, bar_color: dict[int, int]
) -> None:
    for br in bar_rows:
        g[br, bar_start:] = bar_color[br]


def _build_l_turn(
    g: np.ndarray,
    bar_rows: list[int],
    bar_color: dict[int, int],
    top: int,
    bot: int,
    start_row: int,
    c_v: int,
    direction: str,
) -> None:
    color = bar_color[top]
    g[start_row, c_v] = color
    turn_bar = top if direction == "up" else bot
    g[turn_bar, c_v - 1] = BG
    g[turn_bar, c_v - 2] = bar_color[turn_bar]
    g[turn_bar, c_v] = bar_color[turn_bar]

    if direction == "up":
        for br in bar_rows:
            if br < top:
                g[br, c_v - 1] = BG
    else:
        for br in bar_rows:
            if br > bot:
                g[br, c_v - 1] = BG


def _band_rows(top: int, bot: int) -> list[int]:
    return list(range(top + 1, bot))


def _band_height_from_row(grid: np.ndarray, r: int) -> int | None:
    entrance = _find_entrance(grid, r)
    if entrance is None:
        return None
    _, top, bottom = entrance
    return bottom - top - 1


def variant_metrics(grid: np.ndarray, out: np.ndarray | None = None) -> dict[str, int]:
    """Quantify whether generated samples really use irregular-height corridors."""
    if out is None:
        out = main(grid)
    H, W = grid.shape
    starters = [r for r in range(H) if int(grid[r, 0]) == PARTICLE]
    moved = [r for r in starters if int(out[r, 0]) != PARTICLE]
    wide_starters = 0
    wide_moved = 0
    for r in starters:
        height = _band_height_from_row(grid, r)
        if height is not None and height > 1:
            wide_starters += 1
            if r in moved:
                wide_moved += 1
    destinations = [(int(r), int(c)) for r, c in np.argwhere(out == PARTICLE) if int(grid[r, c]) != PARTICLE]
    turned_moved = sum(1 for _, c in destinations if c != W - 1)
    return {
        "starters": len(starters),
        "moved": len(moved),
        "wide_starters": wide_starters,
        "wide_moved": wide_moved,
        "turned_moved": turned_moved,
    }


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    H, bar_rows, band_heights = layout
    n_bands = len(bar_rows) - 1
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    wide_bands = [i for i, h in enumerate(band_heights) if h > 1]
    if len(wide_bands) < 2:
        return None

    n_active = random.randint(_MIN_MOVED, min(_MAX_MOVED, n_bands))
    forced_wide = random.sample(wide_bands, min(2, n_active))
    remaining = [i for i in range(n_bands) if i not in forced_wide]
    active_bands = set(forced_wide)
    while len(active_bands) < n_active:
        active_bands.add(random.choice(remaining))
        remaining = [i for i in remaining if i not in active_bands]

    spare_real_pool = [i for i in range(n_bands) if i not in active_bands]
    extra_real = random.sample(spare_real_pool, random.randint(0, min(2, len(spare_real_pool))))
    real_bands = set(active_bands) | set(extra_real)

    bar_color = {bar_rows[0]: random.choice(extras)}
    for i in range(n_bands):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if i in real_bands:
            bar_color[bot] = bar_color[top]
        else:
            choices = [c for c in extras if c != bar_color[top]]
            bar_color[bot] = random.choice(choices)
    if len(set(bar_color.values())) < 3:
        return None

    g = np.full((H, W), BG, dtype=int)
    _paint_bars(g, bar_rows, bar_start, bar_color)
    g[:, 0] = BG
    g[:, 1] = BG

    used_vcols: set[int] = set()
    plans: dict[int, dict[str, int | str]] = {}
    wide_active = [i for i in active_bands if band_heights[i] > 1]
    forced_turn_band = random.choice(wide_active)

    for i in sorted(active_bands):
        top, bot = bar_rows[i], bar_rows[i + 1]
        rows = _band_rows(top, bot)
        if not rows:
            return None

        make_turn = i == forced_turn_band or random.random() > _STRAIGHT_PROB
        if make_turn:
            possible_dirs = []
            if top > 0:
                possible_dirs.append("up")
            if bot < H - 1:
                possible_dirs.append("down")
            c_v = _allocate_vcol(bar_start, W, used_vcols)
            if possible_dirs and c_v is not None:
                direction = random.choice(possible_dirs)
                start_row = top + 1 if direction == "up" else bot - 1
                plans[i] = {"kind": "turn", "direction": direction, "start_row": start_row, "c_v": c_v}
            else:
                start_row = rows[len(rows) // 2]
                plans[i] = {"kind": "straight", "start_row": start_row}
        else:
            start_row = rows[len(rows) // 2]
            plans[i] = {"kind": "straight", "start_row": start_row}

    if not any(plan["kind"] == "turn" for plan in plans.values()):
        return None

    for i, plan in plans.items():
        start_row = int(plan["start_row"])
        g[start_row, 0] = PARTICLE
        if plan["kind"] == "turn":
            top, bot = bar_rows[i], bar_rows[i + 1]
            _build_l_turn(
                g,
                bar_rows,
                bar_color,
                top,
                bot,
                start_row,
                int(plan["c_v"]),
                str(plan["direction"]),
            )

    for i in range(n_bands):
        if i in real_bands:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        band_rows = _band_rows(top, bot)
        if not band_rows:
            continue
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        for _ in range(n_decor):
            rr = random.choice(band_rows)
            cc = random.randint(bar_start + 1, W - 1)
            if g[rr, cc] == BG:
                g[rr, cc] = random.choice([bar_color[top], bar_color[bot]])

    try:
        out = main(g)
    except Exception:
        return None

    metrics = variant_metrics(g, out)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["wide_moved"] < 2:
        return None
    if metrics["turned_moved"] < 1:
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
