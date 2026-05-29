# Combo variant of 9bbf930d (index 29)
# combo_spec: D5/0+D15/2+D20/2
# Name: 四边粒子+方向标记+螺旋走廊

from __future__ import annotations

import random

import numpy as np

"""
design_analysis:
  concretization:
    original_task:
      - 原题是全局横条带布局: 偶数行是彩色条, 奇数行是通道, 所有粒子都在左边界并固定向右。
      - solver 的真正物理规则很局部: 粒子在可通行格上直行; 前方若撞到彩色墙, 就检查左右/上下两个垂直候选里是否存在“被同色墙夹住”的通道格; 若唯一存在则拐弯。
      - generator 的限制比 solver 更强, 只造 0/1/2 转的 straight/L/U 结构。
    fused_variant:
      - 我保留“同色夹道决定可转弯”的像素物理, 但把全局单一条带改成 4 个互不重叠的局部条带组件。
      - 每个组件都是一个 5 行高的双色带, 内部是一条固定 3 转蛇形/螺旋路径:
        起点 -> 横向前进 -> 向内竖转 -> 反向横走 -> 再竖转 -> 到内部死胡同停止。
      - 然后把这个 canonical 组件分别旋转到四个朝向, 贴到全局网格四条边, 于是同一题中同时出现 left/right/top/bottom 四类边缘起点。
      - 每个起点边上紧贴一个箭头色像素, 箭头偏移方向就是初始方向; solver 不再从边界位置硬推方向。
    trace_example:
      - 以左侧组件为例, 粒子在 `(r,0)` 且 `(r,1)` 是箭头, 因此先向右。
      - 在上层通道直行到 `x1` 时, 前方 `x1+1` 被同色墙封住, 所以下拐进中间缺口。
      - 下到下层通道后继续前进, 再被 `x2-1` 的同色墙挡住, 于是向上拐回上层。
      - 回到上层后前方仍是彩色墙, 且左右都被 breaker 破坏, 因此停止在内部终点。
      - 旋转后的 top/right/bottom 组件完全沿用同一局部机制, 只是整块几何旋转, 所以肉眼能看出“同一种螺旋走廊从不同边进入”。
  complexity_calibration:
    - 螺旋路径固定为 3 转, 明显区别于原题的 0-2 转, 但仍保持单一路径、无回路、无分叉, 不会把规则做得过散。
    - 同一张图里固定出现四条边的起点, 且四个组件共享同一个可见模板, 便于人类从 2-4 个样例里归纳“边缘粒子 + 箭头 + 长路径追踪”。
    - 颜色自由度被压到少量档位: 4 个走廊色、1 个箭头色、1 个背景色、1 个粒子色。颜色不编码隐藏语义, 只承担“同色墙/异色 breaker”的可见结构作用。
  implicit_assumptions_audit:
    - 原 `_trace` 把初始方向写死为 `right`; 融合后必须改成读取箭头。
    - 原 `_has_corridor` 只把 `{BG, PARTICLE}` 视作可走; 融合后箭头像素必须是可通行的, 否则粒子一出发就撞上标记。
    - 原 `main` 只扫 `col 0`; 融合后必须枚举四条边上的所有粒子。
    - 原 `main` 通过“左边入口在 col 2/3”筛真通道; 融合后不再依赖全局条带入口, 因为四个局部组件经过旋转后不共享这一几何朝向。
    - 原 generator 假设所有有效路径都从左边进入; 融合后用“先造 canonical 3 转组件, 再旋转放到四条边”的方式统一解决四向起点与长路径。
    - D5 与 D15 叠加后的潜在冲突是“方向会不会又退化成由边缘唯一决定”; 这里用显式箭头作为 solver 的唯一方向来源, 代码层面不再读取边的位置来推方向。
  placement_legality:
    - 四个真实组件被放在左上、右下、右上、左下四个互不重叠区域; 组件之间留出空白背景, 防止走廊互相连通。
    - canonical 组件内部只改动 5×W 的局部矩形, 所有 blocker/gap/breaker 都在边界内, 并满足唯一转弯约束。
    - 旋转后仍是矩形贴片; 贴片粘贴前检查非背景像素不重叠。
    - 每条真实路径只放 1 个粒子, 所以不会发生多个粒子竞争同一通道或终点冲突。
  visual_observability:
    - 人类能直接看到四条边上各有一个品红粒子, 且粒子旁都有同色箭头标记。
    - 四个局部带状组件外观相似但朝向不同, 输出中四个粒子都沿着长蛇形/螺旋轨迹进入内部终点, 与原题“左边短路径”差异明显。
    - 规则完全由相邻关系、夹道同色、箭头偏移三种可见线索决定, 不依赖颜色编号或隐式排序。
  cleanliness:
    - 箭头只出现在起点旁, 不会与内部墙体混在一起。
    - 每个组件都限制在自己的小矩形里, 周围留大块背景, 避免整图彩条交织成噪声。
    - fake corridor 在这个融合版里被弱化掉, 因为真正的核心变化是多边入口和 3 转以上路径; 继续保留大量假走廊只会降低可读性。
  color_inference:
    - 颜色没有映射题; “是否可转弯”只取决于局部两侧是否同色。
    - breaker 颜色只用于 visibly 打断一个候选出口, 其作用靠空间位置就能看出, 不要求人类记忆某个固定色号的特殊语义。
  new_solver_assumptions_and_generator_guarantees:
    - solver 假设有效起点是边界粒子且拥有唯一相邻箭头。
      generator 保证每个真实起点都满足这一点, 且没有额外伪箭头贴在粒子周围。
    - solver 假设所有拐弯点在“当前方向的两个垂直候选”里至多有一个合法通道。
      generator 通过 breaker 像素破坏另一侧出口, 保证 `assert` 不触发。
    - solver 假设路径长度可能明显超过原题, 因此步数上限不能写死为 500。
      这里改成 `4 * H * W`, 足够覆盖 3 转蛇形与旋转后的长路径。
    - solver 不依赖组件模板; 只依赖局部走廊规则。
      generator 之所以选择旋转组件, 是为了让多方向与螺旋路径在视觉上稳定、可控。
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
    """Return True when (r,c) is a passable cell flanked by same-color walls."""
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


def _trace_with_turns(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int, int]:
    """Trace one particle and also count the number of 90-degree turns taken."""
    h, w = grid.shape
    turns = 0
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
        turns += 1
    return r, c, turns


def _trace(grid: np.ndarray, r: int, c: int, direction: str) -> tuple[int, int]:
    end_r, end_c, _ = _trace_with_turns(grid, r, c, direction)
    return end_r, end_c


def _edge_starts(grid: np.ndarray) -> list[tuple[int, int, str]]:
    """Return border particles that have exactly one adjacent arrow marker."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts: list[tuple[int, int, str]] = []
    edge_cells = []
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
    """Original left-edge-only behavior, kept for anti-degeneration validation."""
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
        bar_above = next((c for c in range(w) if int(g[r - 1, c]) == int(g[r - 1, entrance_col])), w)
        bar_below = next((c for c in range(w) if int(g[r + 1, c]) == int(g[r + 1, entrance_col])), w)
        if bar_above != bar_below or bar_above > 3:
            continue
        end_r, end_c = _trace(g, r, 0, "right")
        if (end_r, end_c) != (r, 0):
            out[r, 0] = BG
            out[end_r, end_c] = PARTICLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move magenta edge particles by following explicit arrows through spiral corridors.

    Input structure:
    - bg=7, particle=6, arrow=2.
    - Particles lie on any grid edge, not only the left edge.
    - Each valid particle has exactly one adjacent arrow pixel; that offset gives the initial direction.
    - Corridors are 1-cell-wide passable paths whose turns are defined by same-color wall pairs.

    Rule:
    - Start each edge particle in the arrow-indicated direction.
    - Move straight through passable cells (`bg`, `particle`, `arrow`).
    - When blocked by a colored wall, try the two perpendicular directions.
    - Turn iff exactly one perpendicular neighbor is itself a passable cell flanked by same-color walls.
    - Stop at the grid edge or when no legal perpendicular turn exists.

    Key insight:
    - The path geometry is purely local: a turn is readable from the two same-color flank pixels around
      the candidate next cell. Rotating the same 3-turn corridor template to different edges does not
      change the tracing rule; only the arrow-selected initial direction changes.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    starts = _edge_starts(g)
    end_positions = []
    for r, c, direction in starts:
        end_positions.append(_trace(g, r, c, direction))

    for r, c, _ in starts:
        out[r, c] = BG
    for end_r, end_c in end_positions:
        out[end_r, end_c] = PARTICLE
    return out


# --------------- generator ---------------

_HEIGHT = 23
_WIDTH = 21
_N_PATH_COLORS = 4
_COMPONENT_WIDTHS = [11, 12, 13]
_BAR_START_OPTIONS = [2, 3]
_MAX_RETRIES = 200


def _build_canonical_component(width: int, color: int, breaker: int, bar_start: int) -> np.ndarray | None:
    """Create one 3-turn left-entry snake component before rotation."""
    g = np.full((5, width), BG, dtype=int)
    g[0, bar_start:width] = color
    g[2, bar_start:width] = color
    g[4, bar_start:width] = color

    x1_min = bar_start + 5
    x1_max = width - 3
    if x1_min > x1_max:
        return None
    x1 = random.randint(x1_min, x1_max)
    x2_min = bar_start + 2
    x2_max = x1 - 3
    if x2_min > x2_max:
        return None
    x2 = random.randint(x2_min, x2_max)

    # Start pair.
    g[1, 0] = PARTICLE
    g[1, 1] = ARROW

    # First turn: right -> down at x1.
    g[1, x1 + 1] = color
    g[2, x1] = BG
    g[4, x1 + 1] = breaker

    # Second/third turns: down -> left at x1, then left -> up at x2.
    g[3, x2 - 1] = color
    g[2, x2] = BG

    # Internal dead-end after re-entry.
    g[0, x2 - 1] = breaker
    g[0, x2 + 1] = breaker
    return g


def _paste_patch(base: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    """Paste non-background cells; return False on collision or OOB."""
    h, w = patch.shape
    base_h, base_w = base.shape
    if top < 0 or left < 0 or top + h > base_h or left + w > base_w:
        return False
    region = base[top : top + h, left : left + w]
    mask = patch != BG
    if np.any((region != BG) & mask):
        return False
    region[mask] = patch[mask]
    return True


def _side_of_start(start: tuple[int, int, str], h: int, w: int) -> str:
    r, c, _ = start
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    if r == 0:
        return "top"
    return "bottom"


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    """Quantify that the generated sample really uses the combo behavior."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts = _edge_starts(g)
    sides = [_side_of_start(s, h, w) for s in starts]
    turns = []
    for r, c, direction in starts:
        _, _, n_turns = _trace_with_turns(g, r, c, direction)
        turns.append(n_turns)
    return {
        "n_starts": len(starts),
        "sides": tuple(sorted(sides)),
        "turns": tuple(turns),
        "min_turns": min(turns) if turns else 0,
        "long_paths": sum(t >= 3 for t in turns),
    }


def _gen_attempt() -> np.ndarray | None:
    h, w = _HEIGHT, _WIDTH
    g = np.full((h, w), BG, dtype=int)

    pool = [c for c in range(10) if c not in (BG, PARTICLE, ARROW)]
    path_colors = random.sample(pool, _N_PATH_COLORS)

    left_w, top_w, right_w, bottom_w = [random.choice(_COMPONENT_WIDTHS) for _ in range(4)]
    bar_starts = [random.choice(_BAR_START_OPTIONS) for _ in range(4)]

    # Randomized but non-overlapping placements. The rotated components preserve the same local rule.
    specs = [
        ("left", 0, left_w, random.randint(1, 4), 0),
        ("top", 3, top_w, 0, random.randint(w - 7, w - 5)),
        ("right", 2, right_w, random.randint(15, 17), w - right_w),
        ("bottom", 1, bottom_w, h - bottom_w, random.randint(0, 2)),
    ]

    for idx, (side, rot_k, width, top, left) in enumerate(specs):
        color = path_colors[idx]
        breaker_choices = [c for c in pool if c != color]
        breaker = random.choice(breaker_choices)
        patch = _build_canonical_component(width, color, breaker, bar_starts[idx])
        if patch is None:
            return None
        patch = np.rot90(patch, rot_k)
        if side == "top" and patch.shape[1] != 5:
            return None
        if not _paste_patch(g, patch, top, left):
            return None

    starts = _edge_starts(g)
    if len(starts) != 4:
        return None
    sides = {_side_of_start(s, h, w) for s in starts}
    if sides != {"left", "right", "top", "bottom"}:
        return None

    try:
        out = main(g)
    except Exception:
        return None
    if np.array_equal(out, _legacy_main(g)):
        return None

    metrics = _fusion_metrics(g)
    if metrics["n_starts"] != 4:
        return None
    if metrics["min_turns"] < 3 or metrics["long_paths"] != 4:
        return None
    if len(set(g.flatten().tolist())) < 7:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
