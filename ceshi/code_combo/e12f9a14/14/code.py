# Combo variant of e12f9a14 (index 14)
# combo_spec: D6/1+D15/0
# Name: 额外障碍物+碰任意前景停

# Design analysis
# design:
#   a_concretization_and_trace:
#     objects:
#       - 输入保留原题的主体：若干个 2x2 单色块，每个块被同色 4x4 边框包住，边框上有缺口。
#       - 缺口本身仍是背景色，输出中会先被对应块颜色填上，然后沿缺口朝外发射一条直线射线。
#       - 新增若干个孤立单像素障碍物；它们和框、内块一样，都是“非背景前景”。
#     worked_example:
#       - 例：左框右侧开口，向右直射；两格之后路径上有一个孤立灰点，再更远处还有另一个框。
#       - 原题：这种输入不合法，因为额外前景不存在，而且射线若遇其他射线会继续发生 45/90 度碰撞转向。
#       - 仅 D15/0 的参考实现：灰点会把射线顺时针折走。
#       - 本融合版：灰点不提供特殊转向，射线会在灰点前一格停止；若路径上先遇到别的框、内块或已画射线，也同样停止。
#       - 因此“所有前景都是墙”成为统一规则：孤立障碍物只是新增了一类静态墙，而不是新的动力学节点。
#   b_complexity_calibration:
#     - 若 generator 沿用原题“射线只穿背景直到边界”的约束，那么 D15 的新增障碍物完全不可见，任务会退化成普通 D6。
#     - 为了稳定显出融合效应，生成器强制至少一条射线被孤立障碍物截断，并额外保留一条被框体截断的射线，让解题者看出“障碍物和原有前景同样会挡住射线”。
#     - 为减少无关复杂度，每个框固定只放 1 个缺口。这样每个 2x2 颜色只对应 1 条射线，观察者不必处理同色多束线。
#   c_implicit_assumption_audit:
#     - 原 solver 假设“下一格若非背景，只可能进入复杂碰撞分支”；融合后这个前提失效，任何非背景都直接终止。
#     - 原 solver 的几何 crossing、45 度合流、90 度留空洞等逻辑都必须删除，否则会把“前景即墙”错误解释成可继续传播的碰撞。
#     - `_find_blocks_2x2` 默认每个有意义的 2x2 非背景方块都是发射器；因此新增障碍物必须保持孤立，不能拼出 2x2。
#     - `_infer_frame_color` / `_frame_gaps_for_block` 默认框周围没有贴边额外前景；生成器需要保证障碍物与框保持 8 邻域分离。
#     - 同步更新里，两个射线若同一步争抢同一背景格，输入里那一格事先不是前景。这里采用 D6 单变体的自然同步扩展：双方都停下且该格不着色。
#   d_placement_legality:
#     - 所有 4x4 框继续保持不重叠，且 bbox 之间至少留 2 格，确保结构清晰、框检测稳定。
#     - 生成器先构造一对强制对齐的框，使第一条射线在至少 2 格背景后撞到另一个框，从而保留“框也是墙”的例子。
#     - 其余框只选单缺口且要求原始直线路径互不相交，避免多个停止原因堆在同一条轨迹上。
#     - 障碍物只放在这些原始直线路径上，并且必须是 8 邻域孤立的背景格，保证不会破坏框识别，也不会伪造新的 2x2 物体。
#   e_visual_observability:
#     - 每个样例至少出现一次“射线在孤立前景点前停下”的现象，这是 D15/0 在本融合规则里的核心证据。
#     - 同时至少保留一次“射线在别的框前停下”的现象，帮助人把规则归纳成“所有前景都挡路”，而不是“只有灰点挡路”。
#     - 障碍物颜色固定为灰色 5，且不参与框或内块颜色，视觉上容易识别为新增环境元素。
#   f_visual_tidiness:
#     - 单缺口、路径互斥、孤立障碍物三条约束一起控制杂乱度，输出主要是几条干净的直线段加若干明显的停点。
#     - 障碍物本身保持原样不被覆盖，终点总紧贴在它们或框墙之前一格，因此人眼能直接看出“没穿过去”。
#   g_color_inference:
#     - 不做颜色映射：射线颜色仍等于其 2x2 内块颜色。
#     - 灰色 5 只用于“额外墙”，不承担隐藏排序或类别 id。
#   post_change_solver_assumptions:
#     - solver 新假设：额外障碍物若存在，一定是孤立单像素且不会贴框；generator 通过 `_isolated_obstacle_site()` 保证这一点。
#     - solver 支持射线同一步争抢同一空格时的“全停”分支；generator 继续尽量避免这种次要情况，让样例主要展示静态前景阻挡。

import numpy as np

from arc_utils import get_bg, paste


OBSTACLE_COLOR = 5

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
        ok = True
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
                and 0 <= c < w
                and 0 <= r < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))

    out_set = set(out)
    if len(out_set) >= 3 and (fr1, fc1) in out_set and (fr0, bc + 1) in out_set:
        out = [p for p in out if p != (fr1, fc1)]
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


def _simulate(grid: np.ndarray, collect_stats: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    frame_colors = {_infer_frame_color(g, bg, br, bc, col) for col, br, bc in blocks}
    assert len(frame_colors) == 1, "expected all frames to share a color"
    frame_color = next(iter(frame_colors))
    block_colors = {int(col) for col, _br, _bc in blocks}
    assert bg != OBSTACLE_COLOR and frame_color != OBSTACLE_COLOR and OBSTACLE_COLOR not in block_colors, (
        "gray(5) is reserved for singleton obstacles"
    )

    beams: dict[int, _Beam] = {}
    next_bid = 0
    for col, br, bc in blocks:
        gaps = _frame_gaps_for_block(g, bg, frame_color, br, bc)
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

    stats = {
        "obstacle_hits": 0,
        "static_foreground_hits": 0,
        "ray_hits": 0,
        "boundary_stops": 0,
        "shared_stop_events": 0,
        "painted_pixels": 0,
    }

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
            if not in_bounds(nxt):
                stats["boundary_stops"] += 1
                beam.active = False
                continue
            if int(out[nr, nc]) != bg:
                if int(g[nr, nc]) == OBSTACLE_COLOR:
                    stats["obstacle_hits"] += 1
                elif int(g[nr, nc]) != bg:
                    stats["static_foreground_hits"] += 1
                else:
                    stats["ray_hits"] += 1
                beam.active = False
                continue
            proposals.setdefault(nxt, []).append(bid)

        if not any_active or not proposals:
            break

        moved = False
        for cell, bidders in proposals.items():
            if len(bidders) > 1:
                stats["shared_stop_events"] += 1
                for bid in bidders:
                    beams[bid].active = False
                continue
            bid = bidders[0]
            out[cell[0], cell[1]] = beams[bid].color
            beams[bid].pos = cell
            stats["painted_pixels"] += 1
            moved = True

        if not moved:
            break
    else:
        raise AssertionError("beam simulation did not converge")

    if not collect_stats:
        return out, {}
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend straight rays from framed color blocks, but treat every foreground pixel as a hard wall.

    Input structure:
    - The most common color is the background.
    - Distinct monochrome 2x2 blocks sit inside shared-color 4x4 frames, and each frame has one or more background gaps on its border.
    - Extra gray singleton pixels may also appear in the background; they are independent obstacles, not emitters.

    Rule:
    - Each frame gap is recolored with its inner block color and launches one outward cardinal-or-diagonal ray.
    - Rays advance synchronously only through background cells.
    - A ray stops as soon as its next cell is outside the grid or any non-background foreground pixel, including a gray obstacle, a frame, another block, or a ray cell painted earlier. If multiple rays try to enter the same empty cell on the same step, none of them paints it.

    Key insight: The fusion removes all turning/collision geometry and replaces it with a single principle: every foreground mark in the input or output acts as an opaque wall, so the challenge is to infer which straight ray gets cut short by which obstacle.
    """
    out, _stats = _simulate(grid, collect_stats=False)
    return out


# --------------- generator (extra-obstacle + foreground-stop combo) ---------------

BLOCK_COUNT_RANGE = (3, 6)  # uniform [lo, hi)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {3: 16, 5: 22}
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
FORCED_RUNWAY_MAX = 7
MAX_OBSTACLES_PER_GRID = 3
MIN_OBSTACLE_RUNUP = 2
GEN_RETRIES = 800


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _grid_size_min(block_count: int) -> int:
    result = 16
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c not in (frame_color, OBSTACLE_COLOR)]
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
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    h, w = obstacle_grid.shape
    cells = [seed]
    nr, nc = seed[0] + direction[0], seed[1] + direction[1]
    while 0 <= nr < h and 0 <= nc < w and int(obstacle_grid[nr, nc]) == bg:
        cells.append((nr, nc))
        nr += direction[0]
        nc += direction[1]
    if 0 <= nr < h and 0 <= nc < w:
        return cells, (nr, nc)
    return cells, None


def _candidate_single_gaps(
    obstacle_grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int] | None]]:
    forbidden = forbidden or set()
    options: list[tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int] | None]] = []
    for gap in _BORDER_ORDER:
        if gap in forbidden:
            continue
        seed = (r0 + gap[0], c0 + gap[1])
        direction = _gap_dir(gap)
        cells, stopper = _trace_ray_cells(obstacle_grid, bg, seed, direction)
        if len(cells) - 1 >= MIN_RAY_STEPS:
            options.append((gap, cells, stopper))
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

    for _ in range(250):
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
            return [
                left,
                right,
            ], {
                "forced_index": 0,
                "forced_gap": (rowoff, 3),
                "forbidden_on_partner": {(rowoff, 0)},
            }

        c0 = int(rng.integers(PLACEMENT_MARGIN, max_top_left + 1))
        r0_max = max_top_left - 4 - runway
        if r0_max < PLACEMENT_MARGIN:
            continue
        r0 = int(rng.integers(PLACEMENT_MARGIN, r0_max + 1))
        top = (r0, c0)
        bottom = (r0 + 4 + runway, c0)
        coloff = int(rng.choice([1, 2]))
        return [
            top,
            bottom,
        ], {
            "forced_index": 0,
            "forced_gap": (3, coloff),
            "forbidden_on_partner": {(0, coloff)},
        }
    return None


def _isolated_obstacle_site(
    grid: np.ndarray,
    bg: int,
    cell: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    r, c = cell
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if int(grid[r, c]) != bg or cell in reserved:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            here = (rr, cc)
            if here == cell:
                continue
            if int(grid[rr, cc]) != bg or here in reserved:
                return False
    return True


def _obstacle_sites_for_path(
    grid: np.ndarray,
    bg: int,
    cells: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    if len(cells) < MIN_OBSTACLE_RUNUP + 4:
        return []
    sites: list[tuple[int, int]] = []
    for idx in range(MIN_OBSTACLE_RUNUP + 1, len(cells) - 1):
        cell = cells[idx]
        if _isolated_obstacle_site(grid, bg, cell, reserved):
            sites.append(cell)
    return sites


def _plan_obstacles(
    grid: np.ndarray,
    bg: int,
    raw_paths: list[list[tuple[int, int]]],
    forced_idx: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    reserved: set[tuple[int, int]] = set()
    candidates_by_ray: dict[int, list[tuple[int, int]]] = {}
    for ridx, cells in enumerate(raw_paths):
        if ridx == forced_idx:
            continue
        candidates = _obstacle_sites_for_path(grid, bg, cells, reserved)
        if candidates:
            candidates_by_ray[ridx] = candidates

    if not candidates_by_ray:
        return None

    ray_ids = list(candidates_by_ray)
    rng.shuffle(ray_ids)
    target = int(rng.integers(1, min(MAX_OBSTACLES_PER_GRID, len(ray_ids)) + 1))

    chosen: list[tuple[int, int]] = []
    for ridx in ray_ids:
        candidates = _obstacle_sites_for_path(grid, bg, raw_paths[ridx], set(chosen))
        if not candidates:
            continue
        chosen.append(candidates[int(rng.integers(len(candidates)))])
        if len(chosen) == target:
            break

    if not chosen:
        return None
    return chosen


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

    obstacle_grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacle_grid, r0, c0, bg, frame_color, int(color), ())

    chosen_gaps: list[tuple[tuple[int, int], ...]] = []
    raw_paths: list[list[tuple[int, int]]] = []
    stoppers: list[tuple[int, int] | None] = []
    used_ray_cells: set[tuple[int, int]] = set()

    forced_idx = int(forced_meta["forced_index"])
    forced_gap = tuple(forced_meta["forced_gap"])
    forced_pos = positions[forced_idx]
    forced_seed = (forced_pos[0] + forced_gap[0], forced_pos[1] + forced_gap[1])
    forced_dir = _gap_dir(forced_gap)
    forced_cells, forced_stopper = _trace_ray_cells(obstacle_grid, bg, forced_seed, forced_dir)
    if len(forced_cells) - 1 < MIN_RAY_STEPS or forced_stopper is None:
        return None

    for idx, pos in enumerate(positions):
        r0, c0 = pos
        if idx == forced_idx:
            gap = forced_gap
            cells = forced_cells
            stopper = forced_stopper
        else:
            forbidden = forced_meta["forbidden_on_partner"] if idx == 1 else set()
            candidates = _candidate_single_gaps(obstacle_grid, bg, r0, c0, forbidden=set(forbidden))
            candidates = [
                (gap, cells, stopper)
                for gap, cells, stopper in candidates
                if all(cell not in used_ray_cells for cell in cells)
            ]
            if not candidates:
                return None

            obstacle_ready = [
                cand for cand in candidates
                if _obstacle_sites_for_path(obstacle_grid, bg, cand[1], set())
            ]
            if idx != forced_idx and obstacle_ready and int(rng.integers(100)) < 70:
                candidates = obstacle_ready

            gap, cells, stopper = candidates[int(rng.integers(len(candidates)))]

        if any(cell in used_ray_cells for cell in cells):
            return None
        used_ray_cells.update(cells)
        chosen_gaps.append((gap,))
        raw_paths.append(cells)
        stoppers.append(stopper)

    if stoppers[forced_idx] is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)

    obstacle_cells = _plan_obstacles(grid, bg, raw_paths, forced_idx, rng)
    if not obstacle_cells:
        return None
    for r, c in obstacle_cells:
        grid[r, c] = OBSTACLE_COLOR

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return None

    gap_total = len(chosen_gaps)
    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["obstacle_hits"] < len(obstacle_cells):
        return None
    if stats["static_foreground_hits"] < 1:
        return None
    if stats["shared_stop_events"] != 0:
        return None

    # Quantitative anti-degeneracy checks:
    # - at least one obstacle actually shortens its ray before the natural terminus;
    # - the forced frame-facing ray still ends at another frame or block.
    shortened_by_obstacle = 0
    for cells, color in zip(raw_paths, block_colors):
        painted = int(np.count_nonzero(out == int(color))) - 4
        if painted < len(cells):
            shortened_by_obstacle += 1
    if shortened_by_obstacle < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
