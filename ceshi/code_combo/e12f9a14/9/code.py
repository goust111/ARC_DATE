# Combo variant of e12f9a14 (index 9)
# combo_spec: D6/1+D7/1
# Name: 碰任意前景停+框架作障碍物

# Design analysis
# design:
#   a_concretization_and_trace:
#     objects:
#       - 输入仍由若干个 2x2 纯色块组成，每个色块外包一个共享的 4x4 绿色框。
#       - 框边上的背景缺口仍是射线起点；角缺口发对角线，边缺口发水平/垂直线。
#       - 输出仍先把缺口本身染成对应块色，再沿该方向同步延伸。
#     fused_change:
#       - D6/1 把终止条件改成“下一格只要不是背景就停”；因此原题的 45/90 度碰撞、转向、锯齿、空洞全部删除。
#       - D7/1 规定其他框的绿色边框也属于“非背景障碍物”；同一个框既能发射，也能挡住别人的射线。
#       - 因为 D6/1 已经把所有前景统一成硬障碍，所以 D7/1 在融合版里的可见作用不再是反弹，而是“绿框会截断轨迹”。
#     worked_example:
#       - 例：左框右侧开一个缺口，右框与它同一行，两个 4x4 框之间留 3 个背景格；右框左边同一行保持实墙。
#       - 原题：这类输入会被 generator 回避，否则 solver 会把撞框视作非法，或若改造成射线相撞则继续转向。
#       - D6 单变体：左射线会在右框实墙前停止，但若样例里没有“撞框”布局，这个变化不显眼。
#       - 本融合版：左射线稳定表现为“缺口 + 若干背景格”后在右框绿墙前戛然而止；右框同时还会发出自己的一条射线。
#       - 差异核心不是颜色映射，而是观察到绿色框既是发射器，又是其它射线的终点屏障。
#   b_complexity_calibration:
#     - 若继续沿用原题 generator 的“所有射线都只穿背景到边界”约束，这个融合会退化成普通 D6 直线版，D7 完全不可见。
#     - 因此 generator 强制至少制造 1 组“框挡射线”事件，并把每个框限制为恰好 1 个缺口，避免多束同色射线造成视觉歧义。
#     - 其余框只选择会一直通到边界的缺口，保证画面里既有典型撞框截断，也有对照性的长直线。
#   c_implicit_assumption_audit:
#     - 原 main() 默认“下一格非背景”只可能意味着旧碰撞系统里的 blocked/cell_owner 事件；融合后这个前提失效，必须直接把 `out[nxt] != bg` 解释为停止。
#     - 原 `resolve_cell_collision`、`resolve_cross_collision`、`resolve_segment_cross_collision` 全依赖旧动力学，融合后应整段删除。
#     - 原 generator 默认框绝不能落在射线路径上；融合后恰恰要故意安排一条路径撞上别框的实墙，但仍需避免撞上别框的缺口。
#     - 新 solver 还隐含一个同步假设：若多条射线同一步争抢同一空白格，则都停且不落格。generator 通过路径互斥避免样例依赖这一细枝末节。
#   d_placement_legality:
#     - 所有 4x4 框仍保持不重叠，bbox 之间至少留 2 格，且整体不越界。
#     - generator 先放一对强制布局：源框与目标框水平或垂直对齐，源框缺口朝向目标框的某个保留墙格，且中间至少有 2 格背景跑道。
#     - 目标框自己的缺口禁止开在被撞的墙格上；其余框额外随机放置，但只选边界向外的单缺口并要求所有射线路径互不相交。
#   e_visual_observability:
#     - 每个样例至少有 1 条射线在边界之前被另一个绿色框截断，这是融合规则最重要的肉眼证据。
#     - 同时保留若干直达边界的射线，让解题者能对比“为什么有的线变短，有的线很长”，从而归纳出障碍物作用。
#     - 颜色只用于绑定“哪条线属于哪个 2x2 块”，不承担排序、优先级或隐藏编码。
#   f_visual_tidiness:
#     - 单缺口、单直线、路径互斥，使输出主要是几条干净的细线段，终点紧贴别框边缘或网格边界。
#     - 禁止 beam-beam 覆盖、同步抢格和撞到内部 2x2 色块，避免同一区域出现多层解释。
#   g_color_inference:
#     - 框统一为绿色 3，进一步强化“绿色是公用墙体”的视觉角色。
#     - 块色从非背景且非绿色的调色板中抽样，保持每块唯一；没有颜色变换，只是原色向外延伸。
#   post_change_solver_assumptions:
#     - solver 假设每个 2x2 块周围都能恢复出唯一共享框色和至少一个结构性缺口；generator 显式满足这一点。
#     - solver 允许射线因边界、框墙、色块或已画射线而停止；generator 额外保证训练样例几乎总是“边界或框墙”两类终止，提升可推断性。

import numpy as np

from arc_utils import get_bg, paste


FRAME_COLOR = 3

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


def _build_frame_walls(g: np.ndarray, bg: int, blocks: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    walls: set[tuple[int, int]] = set()
    for col, br, bc in blocks:
        wall = _infer_frame_color(g, bg, br, bc, col)
        gaps = set(_frame_gaps_for_block(g, bg, wall, br, bc))
        fr0, fc0 = br - 1, bc - 1
        for lr in range(4):
            for lc in range(4):
                if not (lr in (0, 3) or lc in (0, 3)):
                    continue
                cell = (fr0 + lr, fc0 + lc)
                if cell in gaps:
                    continue
                if int(g[cell]) == wall:
                    walls.add(cell)
    return walls


def _simulate(grid: np.ndarray, collect_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"
    frame_walls = _build_frame_walls(g, bg, blocks)

    beams: dict[int, _Beam] = {}
    next_bid = 0

    stats = None
    if collect_stats:
        stats = {
            "beams_spawned": 0,
            "frame_stops": 0,
            "boundary_stops": 0,
            "painted_ray_stops": 0,
            "input_block_stops": 0,
            "simultaneous_conflicts": 0,
        }

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
            if stats is not None:
                stats["beams_spawned"] += 1

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
                beam.active = False
                if stats is not None:
                    stats["boundary_stops"] += 1
                continue
            if int(out[nr, nc]) != bg:
                beam.active = False
                if stats is not None:
                    if nxt in frame_walls:
                        stats["frame_stops"] += 1
                    elif int(g[nr, nc]) != bg:
                        stats["input_block_stops"] += 1
                    else:
                        stats["painted_ray_stops"] += 1
                continue
            proposals.setdefault(nxt, []).append(bid)

        if not any_active or not proposals:
            break

        moved = False
        for cell, bidders in sorted(proposals.items()):
            if len(bidders) > 1:
                for bid in bidders:
                    beams[bid].active = False
                if stats is not None:
                    stats["simultaneous_conflicts"] += 1
                continue
            bid = bidders[0]
            out[cell[0], cell[1]] = beams[bid].color
            beams[bid].pos = cell
            moved = True

        if not moved:
            break
    else:
        raise AssertionError("beam simulation did not converge")

    if stats is None:
        return out
    stats["painted_pixels"] = int(np.count_nonzero(out != g))
    return out, stats


def analyze_behavior(grid: np.ndarray) -> dict[str, int]:
    _, stats = _simulate(grid, collect_stats=True)
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow straight rays from framed 2x2 blocks; any foreground pixel, including other green frames, halts the ray.

    Input structure:
    - The background is the most common color in the grid.
    - Each object is a monochrome 2x2 block inside a shared green 4x4 frame.
    - Every frame contains one or more structural border gaps made of background cells.

    Rule:
    - Each gap is recolored with its block's color and emits one outward cardinal or diagonal ray according to the gap position.
    - Rays advance synchronously through background only; the next step immediately stops a ray if it leaves the grid or if the destination is any non-background pixel.
    - Obstacles include other green frame walls, inner 2x2 blocks, and ray pixels painted earlier; if multiple rays try to enter the same empty cell on the same step, none of them paints that cell.

    Key insight:
    - The important new pattern is that frames have a dual role: they still define beam sources through their gaps, but their intact green border cells also act as hard blockers that truncate neighboring rays.
    """
    return _simulate(grid, collect_stats=False)


# --------------- generator (combo D6/1 + D7/1) ---------------

BLOCK_COUNT_RANGE = (3, 6)  # uniform [lo, hi)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {3: 16, 4: 20, 5: 23}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 4, 6, 7, 8, 9],
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
    result = 16
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    palette = PALETTE_FOR_BG[bg]
    block_colors = list(map(int, rng.choice(palette, size=block_count, replace=False)))
    return bg, block_colors


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), FRAME_COLOR, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _trace_until_stop(
    obstacle_grid: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    h, w = obstacle_grid.shape
    cells = [seed]
    nr, nc = seed[0] + direction[0], seed[1] + direction[1]
    while 0 <= nr < h and 0 <= nc < w:
        if int(obstacle_grid[nr, nc]) != bg:
            return cells, (nr, nc)
        cells.append((nr, nc))
        nr += direction[0]
        nc += direction[1]
    return cells, None


def _candidate_single_gaps(
    obstacle_grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    forbidden: set[tuple[int, int]] | None = None,
    require_hit: bool | None = None,
) -> list[tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int] | None]]:
    forbidden = forbidden or set()
    options: list[tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int] | None]] = []
    for gap in _BORDER_ORDER:
        if gap in forbidden:
            continue
        seed = (r0 + gap[0], c0 + gap[1])
        cells, hit = _trace_until_stop(obstacle_grid, bg, seed, _gap_dir(gap))
        if len(cells) - 1 < MIN_RAY_STEPS:
            continue
        if require_hit is True and hit is None:
            continue
        if require_hit is False and hit is not None:
            continue
        options.append((gap, cells, hit))
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
            source = (r0, c0)
            target = (r0, c0 + 4 + runway)
            rowoff = int(rng.choice([1, 2]))
            hit_abs = (target[0] + rowoff, target[1])
            meta = {
                "source_index": 0,
                "target_index": 1,
                "forced_gap": (rowoff, 3),
                "forbidden_on_target": {(rowoff, 0)},
                "expected_hit": hit_abs,
            }
            return [source, target], meta

        c0 = int(rng.integers(PLACEMENT_MARGIN, max_top_left + 1))
        r0_max = max_top_left - 4 - runway
        if r0_max < PLACEMENT_MARGIN:
            continue
        r0 = int(rng.integers(PLACEMENT_MARGIN, r0_max + 1))
        source = (r0, c0)
        target = (r0 + 4 + runway, c0)
        coloff = int(rng.choice([1, 2]))
        hit_abs = (target[0], target[1] + coloff)
        meta = {
            "source_index": 0,
            "target_index": 1,
            "forced_gap": (3, coloff),
            "forbidden_on_target": {(0, coloff)},
            "expected_hit": hit_abs,
        }
        return [source, target], meta
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))
    bg, block_colors = _pick_colors(rng, block_count)

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
        _draw_framed_block(obstacles, r0, c0, bg, int(color), ())

    chosen_gaps: list[tuple[tuple[int, int], ...]] = [((-1, -1),)] * block_count
    used_ray_cells: set[tuple[int, int]] = set()

    source_idx = int(forced_meta["source_index"])
    target_idx = int(forced_meta["target_index"])
    forced_gap = tuple(forced_meta["forced_gap"])
    forced_pos = positions[source_idx]
    forced_seed = (forced_pos[0] + forced_gap[0], forced_pos[1] + forced_gap[1])
    forced_cells, forced_hit = _trace_until_stop(obstacles, bg, forced_seed, _gap_dir(forced_gap))
    if forced_hit != tuple(forced_meta["expected_hit"]):
        return None
    if len(forced_cells) - 1 < MIN_RAY_STEPS:
        return None
    used_ray_cells.update(forced_cells)
    chosen_gaps[source_idx] = (forced_gap,)

    for idx, pos in enumerate(positions):
        if idx == source_idx:
            continue
        forbidden = set(forced_meta["forbidden_on_target"]) if idx == target_idx else set()
        candidates = _candidate_single_gaps(obstacles, bg, pos[0], pos[1], forbidden=forbidden, require_hit=False)
        candidates = [(gap, cells) for gap, cells, _hit in candidates if all(cell not in used_ray_cells for cell in cells)]
        if not candidates:
            return None
        gap, cells = candidates[int(rng.integers(len(candidates)))]
        used_ray_cells.update(cells)
        chosen_gaps[idx] = (gap,)

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        _draw_framed_block(grid, r0, c0, bg, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return None

    gap_total = block_count
    if stats["painted_pixels"] < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["frame_stops"] < 1:
        return None
    if stats["painted_ray_stops"] != 0 or stats["input_block_stops"] != 0 or stats["simultaneous_conflicts"] != 0:
        return None

    forced_color = int(block_colors[source_idx])
    forced_beam_pixels = int(np.count_nonzero(out == forced_color)) - 4
    if forced_beam_pixels != len(forced_cells):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
