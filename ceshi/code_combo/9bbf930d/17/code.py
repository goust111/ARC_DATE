from __future__ import annotations

# Combo variant of 9bbf930d (index 17)
# combo_spec: D11/1+D3/1
# Name: 非均匀行间距+三格宽走廊
#
# Design analysis
# ---------------
# variant_spec:
#   core_visual:
#     - 水平彩条不再按固定偶数行排布；相邻两条彩条之间可隔 1/2/3 行背景，整张图因此出现宽窄不一的横向带。
#     - 真正可通行的水平走廊只出现在“上下同色且中间恰好 3 行背景”的宽带里；1/2 行的窄带只是布局噪声。
#     - 转弯后的竖向走廊也做成 3 列宽：中间 3 列是背景，两侧更外层才是同色竖墙。
#     - 生成器每个真走廊只放 1 个起始粒子，避免把 3 行宽走廊误读成“整条左边界都会一起移动”的新机制。
#   behavior_trace:
#     - 具体例子：若第 0 行和第 4 行都是红条，则第 1/2/3 行形成一个三格高走廊；粒子放在第 2 行从左往右走。
#     - 原 solver 在水平运动时只检查 `r-1` 和 `r+1`，因此面对 3 行高走廊会把中间那一行误判成“上下都是背景，不是走廊”。
#     - 融合后改成：沿当前列向上/向下搜索最近的彩条，并要求两条彩条同色且间距正好为 3；竖向转弯时沿当前行向左/向右做同样搜索，并要求列间距也正好为 3。
#     - 被彩色障碍挡住时，只在被挑中的那一侧入口行画出竖向走廊签名，因此 `up/down` 里最多只有一个合法转向。
#   complexity_calibration:
#     - 真实规则只引入两件事：不规则条带布局、以及“宽度必须为 3”的走廊判定；不再叠 U-turn 或内部死胡同，避免样例噪声盖过融合点。
#     - 每个样例强制至少出现两种条带高度，且至少一个粒子发生 L 转弯，否则视觉上会退化成 D11-only 或接近原题。
#     - 宽度 1/2 的带保留在同一张图里当作显式对照，让人类能在少量样例中看出“不是所有同色条带都能走，必须是三格宽”。
#   implicit_assumption_audit:
#     - 原 `_has_corridor(..., left/right)` 假设墙就在相邻行；融合后必须改成最近墙搜索，否则三格宽水平走廊完全失效。
#     - 原 `main()` 的入口检测同样默认 `r-1/r+1` 就是上下墙；现在入口要基于最近同色条行，并额外检查间距是否为 3。
#     - 原题的竖向转弯只需要 1 列宽缺口；现在竖向目标必须识别“左墙 + 3 列背景 + 右墙”的签名，否则会把普通背景误当成转弯口。
#     - 竖向墙在穿过别的宽走廊时不能把那些走廊意外堵死，所以生成器只在源走廊入口行保留竖墙，在其它宽走廊的 3 行内部把竖墙挖空做直通 crossing。
#   placement_legality:
#     - 条行始终从同一个 `bar_start` 向右延展，且行号严格递增，保证每个横向带的上下边界唯一且不重叠。
#     - 每个转弯窗口独占 5 列：`墙 | 背景 背景 背景 | 墙`，避免两条三格宽竖向走廊互相覆盖。
#     - 竖向走廊内部 3 列会一直挖到边缘；源走廊只在选中的入口行保留侧墙和水平阻挡色块，确保不会出现双向可转的歧义。
#   observability:
#     - 肉眼能直接看到：有些粒子位于 3 行高同色条带中并发生移动，而旁边 1/2 行高的条带或异色的 3 行高条带不会触发同样行为。
#     - 多组样例叠加后，解题者既能归纳“不规则条带布局”，也能归纳“只有三格宽同色走廊才是真走廊”。
#   cleanliness:
#     - 颜色只承担“同色墙配对”的结构作用，不承担排序或映射编码。
#     - 版面只保留少量宽竖井和窄假带，不在同一区域叠太多机制，因此结构仍然清楚。
#   new_solver_assumptions:
#     - 起始粒子只会出现在存在合法入口的 3 行高真走廊的中间行；生成器显式保证这一点。
#     - 所有转弯都从 3 行高走廊的中间行出发，因此合法转向入口固定是 `top+1` 或 `bottom-1`。
#     - 生成样本若被旧版 1 格宽 solver 处理后得到相同输出，则视为退化并直接丢弃。

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
    """Return the nearest colored rows above/below `(r, c)` if they match."""
    H, _ = grid.shape
    if int(grid[r, c]) not in {BG, PARTICLE}:
        return None

    top = None
    for rr in range(r - 1, -1, -1):
        val = int(grid[rr, c])
        if val not in {BG, PARTICLE}:
            top = rr
            break

    bottom = None
    for rr in range(r + 1, H):
        val = int(grid[rr, c])
        if val not in {BG, PARTICLE}:
            bottom = rr
            break

    if top is None or bottom is None:
        return None
    a, b = int(grid[top, c]), int(grid[bottom, c])
    if a != b or a in {BG, PARTICLE}:
        return None
    return top, bottom, a


def _nearest_vertical_walls(
    grid: np.ndarray, r: int, c: int
) -> tuple[int, int, int] | None:
    """Return the nearest colored cols left/right of `(r, c)` if they match."""
    _, W = grid.shape
    if int(grid[r, c]) not in {BG, PARTICLE}:
        return None

    left = None
    for cc in range(c - 1, -1, -1):
        val = int(grid[r, cc])
        if val not in {BG, PARTICLE}:
            left = cc
            break

    right = None
    for cc in range(c + 1, W):
        val = int(grid[r, cc])
        if val not in {BG, PARTICLE}:
            right = cc
            break

    if left is None or right is None:
        return None
    a, b = int(grid[r, left]), int(grid[r, right])
    if a != b or a in {BG, PARTICLE}:
        return None
    return left, right, a


def _is_horizontal_corridor(grid: np.ndarray, r: int, c: int) -> bool:
    walls = _nearest_horizontal_walls(grid, r, c)
    return walls is not None and walls[1] - walls[0] - 1 == 3


def _is_vertical_corridor(grid: np.ndarray, r: int, c: int) -> bool:
    walls = _nearest_vertical_walls(grid, r, c)
    return walls is not None and walls[1] - walls[0] - 1 == 3


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return False
    if direction in {"left", "right"}:
        return _is_horizontal_corridor(grid, r, c)
    return _is_vertical_corridor(grid, r, c)


def _trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
    """Trace one particle from `(r, c)` until the edge or full blockage."""
    H, W = grid.shape
    direction = "right"
    for _ in range(H * W * 4):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) in {BG, PARTICLE}:
            r, c = nr, nc
            continue

        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if _has_corridor(grid, tr, tc, turn_dir):
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
    """Return `(entrance_col, top_row, bottom_row)` for a valid starter row."""
    _, W = grid.shape
    for c in range(2, W):
        if not _is_horizontal_corridor(grid, r, c):
            continue
        walls = _nearest_horizontal_walls(grid, r, c)
        assert walls is not None
        top, bottom, color = walls
        top_start = _first_color_col(grid, top, color)
        bottom_start = _first_color_col(grid, bottom, color)
        if top_start is None or bottom_start is None:
            continue
        if top_start == bottom_start and top_start <= 3:
            return c, top, bottom
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta particles through irregularly spaced three-cell corridors.

    Input structure:
    - `bg=7`, `particle=6`. Horizontal colored bars start at column 2 or 3.
    - Consecutive bar rows are separated by 1-3 background rows; only some 3-row bands are true same-color corridors.
    - Vertical turns are marked by a 3-column-wide shaft whose colored side walls sit farther out.

    Rule:
    - A particle in column 0 may enter only a same-color horizontal band whose nearest top/bottom walls are exactly 3 rows apart.
    - It moves straight through background cells.
    - When a colored blocker is directly ahead, it turns 90 degrees only if one adjacent cell belongs to a 3-column-wide vertical corridor with matching side walls.
    - It stops at the grid edge or when blocked with no legal turn.

    Key insight:
    - The old puzzle only needed adjacent walls; this fusion requires searching for the nearest colored walls and rejecting bands whose span is not exactly 3.
    """
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

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (15, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_N_BANDS_RANGE = (4, 6)
_MIN_MOVED = 3
_MAX_MOVED = 5
_STRAIGHT_PROB = 0.35
_MAX_RETRIES = 1200


def _legacy_has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
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


def _legacy_trace(grid: np.ndarray, r: int, c: int) -> tuple[int, int]:
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
            if 0 <= tr < H and 0 <= tc < W and _legacy_has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        if len(options) > 1:
            break
        if not options:
            break
        r, c, direction = options[0]
    return r, c


def _legacy_main(grid: np.ndarray) -> np.ndarray:
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
        end_r, end_c = _legacy_trace(g, r, 0)
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def _allocate_turn_col(bar_start: int, W: int, used_cols: set[int]) -> int | None:
    c_min = bar_start + 4
    c_max = W - 2
    candidates = []
    for c_block in range(c_min, c_max + 1):
        window = range(c_block - 3, c_block + 2)
        if all(col not in used_cols for col in window):
            candidates.append(c_block)
    if not candidates:
        return None
    c_block = random.choice(candidates)
    used_cols.update(range(c_block - 3, c_block + 2))
    return c_block


def _sample_layout() -> tuple[int, list[int], list[int]] | None:
    for _ in range(400):
        n_bands = random.randint(*_N_BANDS_RANGE)
        heights = [random.choice([1, 2, 3]) for _ in range(n_bands)]
        if heights.count(3) < 3:
            continue
        if len(set(heights)) < 2:
            continue
        bar_rows = [0]
        cur = 0
        for h in heights:
            cur += h + 1
            bar_rows.append(cur)
        base_h = cur + 1
        viable = [H for H in _HEIGHT_OPTIONS if H >= base_h]
        if not viable:
            continue
        H = random.choice(viable)
        return H, bar_rows, heights
    return None


def _paint_bars(
    g: np.ndarray, bar_rows: list[int], bar_start: int, bar_color: dict[int, int]
) -> None:
    for br in bar_rows:
        g[br, bar_start:] = bar_color[br]


def _wide_band_inner_rows(bar_rows: list[int], heights: list[int]) -> set[int]:
    rows = set()
    for i, h in enumerate(heights):
        if h != 3:
            continue
        top, bottom = bar_rows[i], bar_rows[i + 1]
        rows.update(range(top + 1, bottom))
    return rows


def _build_l_turn(
    g: np.ndarray,
    H: int,
    bar_rows: list[int],
    heights: list[int],
    top: int,
    bottom: int,
    travel_row: int,
    color: int,
    c_block: int,
    direction: str,
) -> None:
    """Build a one-turn path from a 3-row horizontal band into a 3-col shaft."""
    left_wall = c_block - 3
    inside_cols = [c_block - 2, c_block - 1, c_block]
    right_wall = c_block + 1
    source_entry = top + 1 if direction == "up" else bottom - 1
    wide_rows = _wide_band_inner_rows(bar_rows, heights)

    g[travel_row, c_block] = color

    if direction == "up":
        span_rows = range(0, travel_row)
    else:
        span_rows = range(travel_row + 1, H)

    for rr in span_rows:
        g[rr, inside_cols] = BG
        if rr == source_entry or rr not in wide_rows:
            g[rr, left_wall] = color
            g[rr, right_wall] = color
        else:
            g[rr, left_wall] = BG
            g[rr, right_wall] = BG


def _infer_bar_start(grid: np.ndarray) -> int:
    H, W = grid.shape
    if W <= 2:
        return 2
    col2_count = sum(int(grid[r, 2]) not in {BG, PARTICLE} for r in range(H))
    return 2 if col2_count >= 3 else 3


def _layout_heights_from_grid(grid: np.ndarray) -> list[int]:
    H, _ = grid.shape
    bar_start = _infer_bar_start(grid)
    bar_rows = [r for r in range(H) if int(grid[r, bar_start]) not in {BG, PARTICLE}]
    return [bar_rows[i + 1] - bar_rows[i] - 1 for i in range(len(bar_rows) - 1)]


def variant_metrics(grid: np.ndarray, out: np.ndarray | None = None) -> dict[str, int]:
    """Quantify whether generated samples actually exhibit the combo behavior."""
    if out is None:
        out = main(grid)
    H, W = grid.shape
    starters = [r for r in range(H) if int(grid[r, 0]) == PARTICLE]
    moved = [r for r in starters if int(out[r, 0]) != PARTICLE]
    wide_moved = 0
    for r in moved:
        entrance = _find_entrance(grid, r)
        if entrance is not None and entrance[2] - entrance[1] - 1 == 3:
            wide_moved += 1
    destinations = [
        (int(r), int(c))
        for r, c in np.argwhere(out == PARTICLE)
        if int(grid[r, c]) != PARTICLE
    ]
    turned_moved = sum(1 for _, c in destinations if c != W - 1)
    heights = _layout_heights_from_grid(grid)
    return {
        "starters": len(starters),
        "moved": len(moved),
        "wide_moved": wide_moved,
        "turned_moved": turned_moved,
        "wide_band_count": sum(h == 3 for h in heights),
        "height_variety": len(set(heights)),
        "legacy_same": int(np.array_equal(out, _legacy_main(grid))),
    }


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    H, bar_rows, heights = layout
    W = random.randint(*_WIDTH_RANGE)
    bar_start = random.choice(_BAR_START_OPTIONS)
    extras = random.sample([c for c in range(10) if c not in (BG, PARTICLE)], _N_EXTRA_COLORS)

    wide_bands = [i for i, h in enumerate(heights) if h == 3]
    n_active = random.randint(_MIN_MOVED, min(_MAX_MOVED, len(wide_bands)))
    active_bands = set(random.sample(wide_bands, n_active))
    spare_wide = [i for i in wide_bands if i not in active_bands]
    extra_real = set(
        random.sample(spare_wide, random.randint(0, min(1, len(spare_wide))))
    )
    real_bands = active_bands | extra_real

    bar_color = {bar_rows[0]: random.choice(extras)}
    used_bar_colors = {bar_color[bar_rows[0]]}
    for i in range(len(heights)):
        top, bottom = bar_rows[i], bar_rows[i + 1]
        if i in real_bands:
            bar_color[bottom] = bar_color[top]
        else:
            choices = [c for c in extras if c != bar_color[top]]
            unused = [c for c in choices if c not in used_bar_colors]
            bar_color[bottom] = random.choice(unused or choices)
        used_bar_colors.add(bar_color[bottom])
    if len(set(bar_color.values())) < _N_EXTRA_COLORS:
        return None

    g = np.full((H, W), BG, dtype=int)
    _paint_bars(g, bar_rows, bar_start, bar_color)
    g[:, 0] = BG
    g[:, 1] = BG

    plans: dict[int, dict[str, int | str]] = {}
    used_cols: set[int] = set()
    turnable = []
    for i in sorted(active_bands):
        top, bottom = bar_rows[i], bar_rows[i + 1]
        dirs = []
        if top > 0:
            dirs.append("up")
        if bottom < H - 1:
            dirs.append("down")
        if dirs:
            turnable.append(i)
    if not turnable:
        return None
    forced_turn_band = random.choice(turnable)

    for i in sorted(active_bands):
        top, bottom = bar_rows[i], bar_rows[i + 1]
        travel_row = top + 2
        dirs = []
        if top > 0:
            dirs.append("up")
        if bottom < H - 1:
            dirs.append("down")

        make_turn = i == forced_turn_band or random.random() > _STRAIGHT_PROB
        if make_turn and dirs:
            c_block = _allocate_turn_col(bar_start, W, used_cols)
            if c_block is not None:
                plans[i] = {
                    "kind": "turn",
                    "direction": random.choice(dirs),
                    "c_block": c_block,
                    "travel_row": travel_row,
                }
                continue
            if i == forced_turn_band:
                return None

        plans[i] = {"kind": "straight", "travel_row": travel_row}

    if not any(plan["kind"] == "turn" for plan in plans.values()):
        return None

    for i, plan in plans.items():
        travel_row = int(plan["travel_row"])
        g[travel_row, 0] = PARTICLE
        if plan["kind"] == "turn":
            top, bottom = bar_rows[i], bar_rows[i + 1]
            _build_l_turn(
                g,
                H,
                bar_rows,
                heights,
                top,
                bottom,
                travel_row,
                bar_color[top],
                int(plan["c_block"]),
                str(plan["direction"]),
            )

    out = main(g)
    metrics = variant_metrics(g, out)
    if not (_MIN_MOVED <= metrics["moved"] <= _MAX_MOVED):
        return None
    if metrics["wide_moved"] != metrics["moved"]:
        return None
    if metrics["turned_moved"] < 1:
        return None
    if metrics["height_variety"] < 2:
        return None
    if metrics["wide_band_count"] < 3:
        return None
    if metrics["legacy_same"]:
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
