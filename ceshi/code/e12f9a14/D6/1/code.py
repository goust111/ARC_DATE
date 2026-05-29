# Variant of e12f9a14
# Dimension: D6 — 射线终止条件 (dst)
# Change: 射线延伸到网格边界或碰到另一条射线的路径时终止... → 碰到任意前景像素即停止

# Design analysis
# design:
#   a_concretization_and_trace:
#     objects:
#       - 输入仍然是若干个 2x2 纯色块，每个块被共享颜色的 4x4 边框包住。
#       - 边框上的背景缺口是射线起点；缺口在边上时发射水平/垂直射线，在角上时发射对角射线。
#       - 输出仍把缺口本身染成内部块颜色，然后沿着缺口朝外延伸。
#     core_change:
#       - 原题在“射线遇到别的射线”时会触发 45/90 度碰撞、转向、锯齿和空洞。
#       - 本变体取消这些后续动力学；射线现在只会直行，并且下一个像素只要不是背景就立即停下。
#       - 障碍物范围扩大为：其他框架边框、其他框架内部 2x2 色块、已经画出的射线路径，以及同步一步里被多条射线争抢的同一空白格。
#     worked_example:
#       - 例：左框在同一行向右开口，右框与它水平对齐，两个 4x4 框之间留 2 个背景格。
#       - 原逻辑：生成器必须避免这种布局，因为射线撞到框架会触发 solver 断言；若改成两条射线相撞，则会继续转向。
#       - 新逻辑：左射线会涂色“缺口 + 两个背景格”，然后在右框左侧墙像素前停止，不会发生转向或继续传播。
#       - 视觉差异就是：轨迹在已有前景前戛然而止，而不是一路冲到边界或发生复杂碰撞。
#   b_complexity_calibration:
#     - 如果 generator 继续使用原题的“所有射线都只会穿过背景”的约束，这个 D6 改动几乎不可见。
#     - 因此样例里必须稳定出现“射线被已有前景提前截断”的情况。
#     - 为了避免同时引入太多新自由度，generator 收缩为“每个框恰好 1 个缺口”：
#       这样每个块颜色只对应一条射线，观察者不需要分辨同色多束射线之间的关系。
#   c_implicit_assumption_audit:
#     - 原 main() 的大部分代码都假设“下一格若不是背景，只可能是被碰撞逻辑处理过的 blocked 格”；这在 D6 下不成立。
#     - 原 is_free() 只看输入网格是否是背景，并把几何 crossing 当成有效碰撞对象；D6 需要改成直接把任何 out 上的非背景都当作障碍。
#     - 原 resolve_cell_collision / resolve_cross_collision / resolve_segment_cross_collision 都依赖旧规则，因此整段必须删除。
#     - 原 generator 依赖“射线绝不会碰到框架/色块”；D6 反而需要主动制造这种情况。
#   d_placement_legality:
#     - 所有 4x4 框仍保持不重叠，并维持至少 2 格 bbox 间距，保证物体结构清晰、solver 的框检测稳定。
#     - generator 先放一个强制成对布局：两框水平或垂直对齐，并保证一条射线在至少 2 格背景后撞到对方框架。
#     - 其余框随机放置，但其射线路径必须与已有路径集合不相交，避免动态遮挡把样例变乱。
#   e_visual_observability:
#     - 每个样例至少有 1 条射线在到达边界前被前景截断，这个是 D6 的定量可见证据。
#     - 因为每个块颜色唯一、每块只发 1 条射线，所以人眼可以直接把“某个颜色块”和“它那条被截断的射线”对应起来。
#   f_visual_tidiness:
#     - 限制为单缺口 + 路径互不相交，避免输出里出现大面积同色重叠、转向带、锯齿带。
#     - 这样输出主要由若干条干净的直线段组成，终点通常紧贴某个框架边缘，规则更整洁。
#   g_color_inference:
#     - 不做任何颜色映射；射线颜色仍等于对应 2x2 内块颜色。
#     - 颜色只承担“绑定哪条射线属于哪个块”的作用，没有不可见排序或 id 依赖。
#   post_change_solver_assumptions:
#     - solver 现在假设：框结构可由 2x2 块和其 4x4 边框唯一恢复；generator 继续保证这一点。
#     - solver 对同时争抢同一空白格的多条射线采用“全停且不落格”的同步规则；generator 额外构造路径互斥，避免样例长期依赖这个次要分支。

import numpy as np

from arc_utils import get_bg, paste


_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))

    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    return out


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0 and dc == 0:
        return (0, 0)
    if dr != 0 and dc != 0:
        return (int(np.sign(dr)), int(np.sign(dc)))
    if dr != 0:
        return (int(np.sign(dr)), 0)
    return (0, int(np.sign(dc)))


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return _normalize_dir(dr, dc)


class _Beam:
    __slots__ = ("color", "pos", "direction", "active")

    def __init__(self, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.color = int(color)
        self.pos = pos
        self.direction = direction
        self.active = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow straight rays from frame gaps until the first occupied pixel blocks them.

    Input structure:
    - The background is the most common color in the grid.
    - Distinct monochrome 2x2 blocks sit inside shared-color 4x4 frames.
    - Each frame has one or more border gaps where the wall is replaced by background.

    Rule:
    - Every gap is recolored with its block's color and launches one ray in the gap's outward cardinal or diagonal direction.
    - Rays advance synchronously through background only; the next cell stops a ray as soon as it is out of bounds or already non-background.
    - Obstacles include original foreground objects and any ray pixels painted earlier; if several rays try to enter the same empty cell on the same step, none of them paints it.

    Key insight: The new behavior is pure obstacle-limited straight growth, so the informative pattern is where each ray terminates against existing foreground rather than how rays would collide and turn.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert gaps, "expected each block frame to contain at least one structural gap"
        for gr, gc in gaps:
            d = _normalize_dir(
                -1 if gr == br - 1 else (1 if gr == br + 2 else 0),
                -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0),
            )
            assert d != (0, 0), "expected each gap to face outward"
            assert int(out[gr, gc]) == bg, "expected a background gap cell"
            out[gr, gc] = int(col)
            beams[next_bid] = _Beam(int(col), (gr, gc), d)
            next_bid += 1

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    max_iters = h * w
    for _ in range(max_iters):
        proposals: dict[tuple[int, int], list[int]] = {}
        any_active = False
        for bid, beam in beams.items():
            if not beam.active:
                continue
            any_active = True
            nr = beam.pos[0] + beam.direction[0]
            nc = beam.pos[1] + beam.direction[1]
            nxt = (nr, nc)
            if not in_bounds(nxt) or int(out[nr, nc]) != bg:
                beam.active = False
                continue
            proposals.setdefault(nxt, []).append(bid)

        if not any_active:
            break
        if not proposals:
            break

        moved = False
        for cell, bidders in proposals.items():
            if len(bidders) > 1:
                for bid in bidders:
                    beams[bid].active = False
                continue
            bid = bidders[0]
            out[cell[0], cell[1]] = beams[bid].color
            beams[bid].pos = cell
            moved = True

        if not moved:
            break
    else:
        raise AssertionError("beam simulation did not converge")

    return out


# --------------- generator (foreground-stop variant) ---------------

BLOCK_COUNT_RANGE = (2, 6)  # uniform [lo, hi)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 4: 20}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 2
MIN_BEAM_PIXELS_PER_GAP = 2
FORCED_RUNWAY_MAX = 6
GEN_RETRIES = 500


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _grid_size_min(block_count: int) -> int:
    result = 14
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(map(int, rng.choice(palette, size=block_count, replace=False)))
    return bg, frame_color, block_colors


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _trace_ray_cells(
    obstacle_grid: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = obstacle_grid.shape
    cells = [seed]
    nr, nc = seed[0] + direction[0], seed[1] + direction[1]
    while 0 <= nr < h and 0 <= nc < w and int(obstacle_grid[nr, nc]) == bg:
        cells.append((nr, nc))
        nr += direction[0]
        nc += direction[1]
    return cells


def _boundary_steps(shape: tuple[int, int], seed: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    steps = 0
    nr, nc = seed[0] + direction[0], seed[1] + direction[1]
    while 0 <= nr < h and 0 <= nc < w:
        steps += 1
        nr += direction[0]
        nc += direction[1]
    return steps


def _candidate_single_gaps(
    obstacle_grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    forbidden = forbidden or set()
    options: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for gap in _BORDER_ORDER:
        if gap in forbidden:
            continue
        seed = (r0 + gap[0], c0 + gap[1])
        direction = _gap_dir(gap)
        cells = _trace_ray_cells(obstacle_grid, bg, seed, direction)
        if len(cells) - 1 >= MIN_RAY_STEPS:
            options.append((gap, cells))
    return options


def _all_positions(n: int) -> list[tuple[int, int]]:
    hi = n - 4 - PLACEMENT_MARGIN
    return [(r, c) for r in range(PLACEMENT_MARGIN, hi + 1) for c in range(PLACEMENT_MARGIN, hi + 1)]


def _choose_extra_positions(
    n: int,
    count: int,
    existing: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    selected = list(existing)
    positions = _all_positions(n)
    while len(selected) < len(existing) + count:
        viable = [p for p in positions if p not in selected and all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])
    return selected[len(existing):]


def _place_forced_pair(
    n: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], dict[str, object]] | None:
    max_top_left = n - 4 - PLACEMENT_MARGIN
    if max_top_left < PLACEMENT_MARGIN:
        return None

    for _ in range(200):
        runway = int(rng.integers(MIN_RAY_STEPS, FORCED_RUNWAY_MAX + 1))
        if int(rng.integers(2)) == 0:
            r0 = int(rng.integers(PLACEMENT_MARGIN, max_top_left + 1))
            c0_max = max_top_left - 4 - runway
            if c0_max < PLACEMENT_MARGIN:
                continue
            c0 = int(rng.integers(PLACEMENT_MARGIN, c0_max + 1))
            left = (r0, c0)
            right = (r0, c0 + 4 + runway)
            rowoff = int(rng.choice([1, 2]))
            meta = {
                "forced_index": 0,
                "forced_gap": (rowoff, 3),
                "forbidden_on_partner": {(rowoff, 0)},
            }
            return [left, right], meta
        c0 = int(rng.integers(PLACEMENT_MARGIN, max_top_left + 1))
        r0_max = max_top_left - 4 - runway
        if r0_max < PLACEMENT_MARGIN:
            continue
        r0 = int(rng.integers(PLACEMENT_MARGIN, r0_max + 1))
        top = (r0, c0)
        bottom = (r0 + 4 + runway, c0)
        coloff = int(rng.choice([1, 2]))
        meta = {
            "forced_index": 0,
            "forced_gap": (3, coloff),
            "forbidden_on_partner": {(0, coloff)},
        }
        return [top, bottom], meta
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    forced = _place_forced_pair(n, rng)
    if forced is None:
        return None
    pair_positions, forced_meta = forced

    extras = _choose_extra_positions(n, block_count - 2, pair_positions, rng)
    if extras is None:
        return None
    positions = pair_positions + extras

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    chosen_gaps: list[tuple[tuple[int, int], ...]] = []
    used_ray_cells: set[tuple[int, int]] = set()

    forced_idx = int(forced_meta["forced_index"])
    forced_gap = tuple(forced_meta["forced_gap"])
    forced_pos = positions[forced_idx]
    forced_seed = (forced_pos[0] + forced_gap[0], forced_pos[1] + forced_gap[1])
    forced_dir = _gap_dir(forced_gap)
    forced_cells = _trace_ray_cells(obstacles, bg, forced_seed, forced_dir)
    if len(forced_cells) - 1 < MIN_RAY_STEPS:
        return None
    if len(forced_cells) - 1 >= _boundary_steps(obstacles.shape, forced_seed, forced_dir):
        return None

    for idx, pos in enumerate(positions):
        r0, c0 = pos
        if idx == forced_idx:
            gap = forced_gap
            cells = forced_cells
        else:
            forbidden = forced_meta["forbidden_on_partner"] if idx == 1 else set()
            candidates = _candidate_single_gaps(obstacles, bg, r0, c0, forbidden=set(forbidden))
            candidates = [(gap, cells) for gap, cells in candidates if all(cell not in used_ray_cells for cell in cells)]
            if not candidates:
                return None
            gap, cells = candidates[int(rng.integers(len(candidates)))]

        if any(cell in used_ray_cells for cell in cells):
            return None
        used_ray_cells.update(cells)
        chosen_gaps.append((gap,))

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    gap_total = len(chosen_gaps)
    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None

    forced_color = int(block_colors[forced_idx])
    forced_beam_pixels = int(np.count_nonzero(out == forced_color)) - 4
    if forced_beam_pixels != len(forced_cells):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D6 input")
