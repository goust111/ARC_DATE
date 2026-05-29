from __future__ import annotations

# Combo variant of 7491f3cf (index 15)
# combo_spec: D3/0+D7/0
# Name: 曲线墙+任意边缘指示像素
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入仍是四个 5x5 面板横排：tile0 给出 cue，tile1/tile2 是两个单色对称素材，tile3 输入为空白。
#    - 原题里 indicator 只能放在 8 个“角点/边中点”，墙只能是 5 像素直线或 L 弯，还要求关于 indicator→center 轴对称。
#    - 融合后，indicator 放宽到任意边缘格。比如 indicator=(0,1) 也合法；同时墙改成经过中心的 7 像素阶梯折线，
#      例如 {(0,3),(1,3),(2,2),(2,3),(3,1),(3,2),(4,1)}。
#    - 在这个例子里，solver 不再靠“对称轴”猜分界，而是先完整识别整条折线路径，再从 indicator=(0,1)
#      做 4 连通 flood-fill。inside/outside 的划分不再是简单半平面，而是沿阶梯线形成不规则区域。
#    - 输出合成规则保持原题：inside 取 tile1，outside 取 tile2，wall 仍优先 tile1，若 tile1 为空且 wall 为 4 连通则回退 tile2。
# b) 复杂度校准
#    - 只改两处：墙形状族与 indicator 位置族；布局、颜色角色、tile1/tile2 的对称采样都保持原题，避免规则漂移过大。
#    - 曲线墙限制为“无分叉、无闭环、单像素宽、4 连通简单路径、至少 3 段方向 run、固定 7 像素、经过中心并触边至少两侧”。
#      这样视觉上明显区别于原题直线/L 弯，但仍足够规整，解题者可从少量样例稳定归纳。
#    - indicator 虽扩到 16 个边缘位，但 generator 会优先采样新增的 8 个非旧位边缘点，以保证在 2-4 个样例中更容易看出变化。
# c) 隐式假设审查
#    - 原 main() 依赖“indicator 只可能在 8 个特殊边缘位”，这在融合后失效，必须改为接受全部 16 个边缘格。
#    - 原 main() 依赖“wall 关于 indicator→center 轴对称”，曲线墙 + 任意边缘位会打破这条假设，必须删除。
#    - 原 main() 的真正必要前提其实是：存在唯一 8-隔离 indicator；wall 是穿过中心的单条障碍；wall 能把 5x5 分成 inside/outside。
#      这些前提在融合后仍成立，因此 flood-fill 与合成代码可复用。
#    - 原生成器按 wall 族直接硬编码 20 个配置，无法覆盖 16 个边缘位和更多曲线形状；这里改为“从中心向两侧边界枚举分支并筛合法曲线墙”。
# d) 放置合法性
#    - 所有 wall 都通过预枚举得到：路径不出界、不自交、不分叉，中心必在 wall 中，且 wall 至少触碰两条边。
#    - indicator 必须在边缘、不能落在 wall 上，且与 wall 的全部像素 8-隔离。
#    - 只有当 flood-fill 得到的 inside 和 outside 都非空时，配置才保留，所以 tile3 的分区始终有效。
#    - 墙 7 像素 + indicator 1 像素，总 cue 像素为 8，tile0 背景仍是多数色，不会破坏背景识别。
# e) 视觉可观察性
#    - 曲线墙一定至少弯两次；相比原题的直线/L 弯，tile0 的分界线会出现明显的 S/Z 式台阶轮廓。
#    - indicator 出现在任意边缘位时，例如 (0,1)、(1,0)、(3,4)，解题者会直接看到“不是只在角点/中点”这一视觉线索。
#    - 由于 inside/outside 由折线真实阻挡决定，tile3 中两种素材的分区边界会跟 tile0 中的阶梯墙一一对应，变化规律可肉眼追踪。
# f) 可视化整洁度
#    - cue 仍只有一种颜色；wall 与 indicator 只占 tile0，不与素材重叠。
#    - tile1/tile2 继续使用单色对称图案，tile3 只显示两者的合成结果；元素层级与原题一致，不引入额外标记或多层编码。
# g) 变化的可推断性
#    - 颜色职责完全沿用原题：cue_color 负责提示几何结构，tile1_color / tile2_color 分别代表两侧素材来源。
#    - 融合变化不依赖颜色编号，只依赖可见的“折线墙形状 + indicator 边缘位置 + flood-fill 分区”，因此人类能从少量样例归纳。
# 9) 新的 solver 假设与 generator 保证
#    - 新 solver 只假设：唯一 8-隔离 edge indicator；wall 是经过中心的单条 4 连通阶梯路径；inside/outside 都非空。
#    - generator 预枚举全部合法曲线墙，再过滤掉仍可被旧规则解释的旧兼容配置；这样样本不会退化回原题。

import random
from collections import Counter

import numpy as np

from arc_utils import flood_fill, in_bounds, label_mask, neighbors8


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300

CENTER = (2, 2)
MIN_CURVE_RUNS = 3
MIN_WALL_LEN = 7
MAX_BRANCH_LEN = 5
NON_LEGACY_INDICATOR_PROB = 0.7

LEGACY_INDICATORS = (
    (0, 0), (0, 2), (0, 4),
    (2, 0),         (2, 4),
    (4, 0), (4, 2), (4, 4),
)
LEGACY_INDICATOR_SET = frozenset(LEGACY_INDICATORS)
ALL_EDGE_INDICATORS = tuple(
    (r, c)
    for r in range(TILE_SZ)
    for c in range(TILE_SZ)
    if r in (0, TILE_SZ - 1) or c in (0, TILE_SZ - 1)
)
ALL_EDGE_INDICATOR_SET = frozenset(ALL_EDGE_INDICATORS)
NON_LEGACY_INDICATORS = tuple(cell for cell in ALL_EDGE_INDICATORS if cell not in LEGACY_INDICATOR_SET)
ALL_CELLS = frozenset((r, c) for r in range(TILE_SZ) for c in range(TILE_SZ))


_ORBITS = [
    [(0, 0), (4, 4)],
    [(0, 4), (4, 0)],
    [(2, 2)],
    [(1, 1), (3, 3)],
    [(1, 3), (3, 1)],
    [(0, 1), (1, 0), (3, 4), (4, 3)],
    [(0, 2), (2, 0), (2, 4), (4, 2)],
    [(0, 3), (3, 0), (1, 4), (4, 1)],
    [(1, 2), (2, 1), (2, 3), (3, 2)],
]


def _neighbors4(r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if in_bounds((TILE_SZ, TILE_SZ), nr, nc):
            yield nr, nc


def _is_boundary(cell: tuple[int, int]) -> bool:
    r, c = cell
    return r in (0, TILE_SZ - 1) or c in (0, TILE_SZ - 1)


def _wall_side_tags(wall: frozenset[tuple[int, int]]) -> frozenset[str]:
    tags = set()
    for r, c in wall:
        if r == 0:
            tags.add("top")
        if r == TILE_SZ - 1:
            tags.add("bottom")
        if c == 0:
            tags.add("left")
        if c == TILE_SZ - 1:
            tags.add("right")
    return frozenset(tags)


def _flood_inside(wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _ordered_wall_path(wall: frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...] | None:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    if n_comp != 1:
        return None

    degrees = {cell: sum((nbr in wall) for nbr in _neighbors4(*cell)) for cell in wall}
    if any(deg > 2 for deg in degrees.values()):
        return None

    ends = [cell for cell, deg in degrees.items() if deg == 1]
    if len(ends) != 2:
        return None

    path = [ends[0]]
    prev = None
    cur = ends[0]
    while True:
        nxt = [nbr for nbr in _neighbors4(*cur) if nbr in wall and nbr != prev]
        if not nxt:
            break
        if len(nxt) > 1:
            return None
        prev, cur = cur, nxt[0]
        path.append(cur)

    if len(path) != len(wall) or path[-1] != ends[1]:
        return None
    return tuple(path)


def _run_count(path: tuple[tuple[int, int], ...]) -> int:
    dirs = [(r1 - r0, c1 - c0) for (r0, c0), (r1, c1) in zip(path, path[1:])]
    if not dirs:
        return 0
    runs = 1
    for d0, d1 in zip(dirs, dirs[1:]):
        if d0 != d1:
            runs += 1
    return runs


def _legacy_reflect(indicator: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    ir, ic = indicator
    r, c = cell
    dr, dc = CENTER[0] - ir, CENTER[1] - ic
    if dr == 0:
        return TILE_SZ - 1 - r, c
    if dc == 0:
        return r, TILE_SZ - 1 - c
    if dr * dc > 0:
        return c, r
    return TILE_SZ - 1 - c, TILE_SZ - 1 - r


def _is_legacy_compatible(indicator: tuple[int, int], wall: frozenset[tuple[int, int]]) -> bool:
    if indicator not in LEGACY_INDICATOR_SET:
        return False
    return wall == frozenset(_legacy_reflect(indicator, cell) for cell in wall)


def _enumerate_center_to_boundary_paths() -> list[tuple[tuple[int, int], ...]]:
    paths: list[tuple[tuple[int, int], ...]] = []

    def dfs(path: list[tuple[int, int]]) -> None:
        if len(path) > MAX_BRANCH_LEN:
            return
        if len(path) > 1 and _is_boundary(path[-1]):
            paths.append(tuple(path))
            return
        for nbr in _neighbors4(*path[-1]):
            if nbr in path:
                continue
            dfs(path + [nbr])

    dfs([CENTER])
    return paths


def _build_curve_configs() -> list[tuple[tuple[int, int], frozenset[tuple[int, int]]]]:
    branches = _enumerate_center_to_boundary_paths()
    configs: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], tuple[tuple[int, int], frozenset[tuple[int, int]]]] = {}
    for i, path_a in enumerate(branches):
        end_a = path_a[-1]
        cells_a = set(path_a)
        for path_b in branches[i + 1 :]:
            end_b = path_b[-1]
            cells_b = set(path_b)
            if end_a == end_b:
                continue
            if cells_a & cells_b != {CENTER}:
                continue

            wall = frozenset(cells_a | cells_b)
            if len(wall) < MIN_WALL_LEN:
                continue
            if CENTER not in wall:
                continue
            if len(_wall_side_tags(wall)) < 2:
                continue

            ordered = _ordered_wall_path(wall)
            if ordered is None:
                continue
            if _run_count(ordered) < MIN_CURVE_RUNS:
                continue

            for indicator in ALL_EDGE_INDICATORS:
                if indicator in wall:
                    continue
                if any(nbr in wall for nbr in neighbors8(*indicator)):
                    continue
                inside = _flood_inside(wall, indicator)
                outside = ALL_CELLS - wall - inside
                if not inside or not outside:
                    continue
                if _is_legacy_compatible(indicator, wall):
                    continue
                key = (indicator, tuple(sorted(wall)))
                configs[key] = (indicator, wall)

    if not configs:
        raise RuntimeError("failed to enumerate curved wall configs")
    return sorted(configs.values(), key=lambda item: (item[0], tuple(sorted(item[1]))))


CURVE_CONFIGS = _build_curve_configs()
CONFIGS_BY_INDICATOR: dict[tuple[int, int], list[frozenset[tuple[int, int]]]] = {}
for _indicator, _wall in CURVE_CONFIGS:
    CONFIGS_BY_INDICATOR.setdefault(_indicator, []).append(_wall)
for _indicator in ALL_EDGE_INDICATORS:
    assert _indicator in CONFIGS_BY_INDICATOR, f"missing curved wall configs for {_indicator}"


def _sample_symmetric_pattern(
    bg: int,
    fg: int,
    rng: random.Random,
    *,
    min_px: int,
    max_px: int,
) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(_ORBITS)))
        rng.shuffle(order)
        p = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for i in order:
            orbit = _ORBITS[i]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    p[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return p
    return None


def _wall_is_4_connected(wall: frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return n_comp == 1


def _composite_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    tile_bg: int,
    inside: set[tuple[int, int]],
    wall: frozenset[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != tile_bg:
                    out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != tile_bg:
                    out3[r, c] = v1
                elif wall_4conn and v2 != tile_bg:
                    out3[r, c] = v2
            else:
                if v2 != tile_bg:
                    out3[r, c] = v2
    return out3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a fourth panel using a curved wall and an edge indicator.

    Input structure:
    - Four 5x5 panels in one row, border-separated.
    - Panel 0: cue pixels = one 8-isolated indicator on any tile edge, plus a center-crossing
      single-pixel staircase wall.
    - Panels 1-2: source patterns. Panel 3: blank.

    Rule:
    - Flood-fill (4-conn) from the indicator, blocked by the wall, to recover the indicator-side region.
    - Inside region -> copy non-bg pixels from panel 1; outside region -> copy non-bg pixels from panel 2.
    - Wall pixels -> panel 1 if non-bg there; otherwise panel 2 may fill them when the wall is 4-connected.

    Key insight:
    - The cue tile does not encode a symmetry axis anymore. The solver must identify the full curved barrier,
      then use the indicator's actual edge location as the flood-fill seed to decide which side belongs to panel 1.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    tile_bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != tile_bg, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"
    assert indicator in ALL_EDGE_INDICATOR_SET, f"indicator {indicator} not on tile edge"

    wall = frozenset(cue_set - {indicator})
    assert wall, "missing wall"
    assert CENTER in wall, "wall must pass through the grid center (2,2)"
    assert len(wall) >= MIN_WALL_LEN, "wall too short for curved variant"
    assert not any((nr, nc) in wall for nr, nc in neighbors8(*indicator)), "indicator must be 8-isolated from wall"

    sides_touched = _wall_side_tags(wall)
    assert len(sides_touched) >= 2, f"wall must span edge-to-edge, only touches {sorted(sides_touched)}"

    ordered = _ordered_wall_path(wall)
    assert ordered is not None, "wall must be a single simple 4-connected staircase path"
    assert _run_count(ordered) >= MIN_CURVE_RUNS, "wall must bend at least twice to form a curved staircase"

    inside = _flood_inside(wall, indicator)
    outside = ALL_CELLS - wall - inside
    assert inside, "inside region is empty"
    assert outside, "wall must partition the grid — outside region is empty"

    out3 = _composite_tile(t1, t2, tile_bg, inside, wall, _wall_is_4_connected(wall))
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _pick_indicator(rng: random.Random) -> tuple[int, int]:
    if rng.random() < NON_LEGACY_INDICATOR_PROB:
        return rng.choice(NON_LEGACY_INDICATORS)
    return rng.choice(LEGACY_INDICATORS)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    indicator = _pick_indicator(rng)
    wall = rng.choice(CONFIGS_BY_INDICATOR[indicator])
    inside = _flood_inside(wall, indicator)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _composite_tile(t1, t2, tile_bg, inside, wall, _wall_is_4_connected(wall))

    t3_in = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    g = np.full((GRID_ROWS, GRID_COLS), border, dtype=int)
    for j, tile in enumerate([t0, t1, t2, t3_in]):
        g[1:6, 1 + 6 * j : 1 + 6 * j + 5] = tile

    expected = g.copy()
    expected[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    got = main(g)
    if not np.array_equal(got, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
