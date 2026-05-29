from __future__ import annotations

# Combo variant of 7491f3cf (index 3)
# combo_spec: D3/0+D10/2
# Name: 曲线墙+墙上叠合规则
#
# Design analysis
# a) 具体化与行为追踪
#    - 仍是 4 个 5x5 面板横排：tile0 提供“墙 + 指示像素”，tile1/tile2 是两个单色对称素材，tile3 输入为空。
#    - 原题里，wall 多为直线/L 弯；这里改成穿过中心的阶梯曲线墙，并且要求路径里至少出现一次对角步。
#      例如 indicator=(0,0) 时，墙可以是
#      {(0,4),(1,3),(2,2),(2,1),(2,0),(3,1),(4,2)} 这类 Z/S 感的 8 连通折线。
#    - flood-fill 仍从 indicator 出发、按 4 连通被墙阻挡。变化点在输出 tile3 的 wall 区：
#      原题是“tile1 优先，tile2 只在部分墙型下回退”；现在是固定 OR：
#      wall 上若 tile1/tile2 都有前景，则显示 tile1 颜色；若只有一个有前景，则显示那个；都无则保留背景。
#    - 差异的关键不是把两条规则并排放，而是让曲线墙含对角步，从而天然形成非 4 连通的墙带。
#      这样 wall 上的 tile2-only 像素会在新规则下保留下来，而在旧规则里会被吞掉，融合行为才真正可见。
# b) 复杂度校准
#    - 仍固定 5x5、单 indicator、两个单色对称素材，不改颜色角色和整体布局，避免多维自由度同时暴涨。
#    - 曲线墙只允许“单像素宽、简单 8 连通、无分叉、经过中心、至少两次转向、含对角步”的少量族；
#      这样肉眼能看出明显区别，但 solver 不需要引入模糊匹配。
#    - 生成器再额外要求 wall 上同时出现 tile1 颜色和 tile2-only 颜色，保证融合带不是名义存在。
# c) 隐式假设审查
#    - 原 solver 里“wall 关于 indicator->center 轴对称”的断言依赖旧墙族；曲线 Z/S 墙不再满足，必须删除。
#    - 原 solver 里“是否允许 tile2 落在 wall 上”取决于 wall 的 4 连通性；融合后这条前提失效，wall 合成必须改成独立规则。
#    - flood-fill 的核心前提仍成立：wall 需要真实分区，indicator 需要唯一且与 wall 8 隔离。
#    - 新 solver 额外依赖“wall 可以被还原成单条简单 8 连通路径，并且至少含一次对角步”；generator 会显式保证。
# d) 放置合法性
#    - 曲线墙通过预枚举合法路径生成：不越界、不自交、不过度贴近 indicator、经过中心、触达至少两侧，并能分出 inside/outside。
#    - wall 长度控制在 6-7，配合 indicator 后 tile0 仍以背景为多数色，不会破坏 cue 解析。
#    - tile1/tile2 保持原题的单色对称生成方式，不引入额外对象，也不与边框重叠。
# e) 视觉可观察性
#    - 解题者能从 tile0 直接看出“这不是直线/直角墙，而是阶梯曲线”。
#    - tile3 中除了 inside/outside 的常规分区，还能在 wall 上看到两种素材颜色混入同一条带，形成额外线索。
#    - 生成器强制 wall 至少有一个 tile2-only 像素，因此“墙上 OR”不会退化成原题的纯 tile1 边界。
# f) 可视化整洁度
#    - 叠合只发生在 wall 这条单像素带上；inside/outside 仍各自由一种素材主导，因此层次清楚。
#    - 输出虽然会在 wall 上出现两种颜色，但带宽只有 1，且两侧区域规则稳定，不会造成整块画面混乱。
# g) 变化的可推断性
#    - 颜色语义保持最小：cue_color 仍只用于 tile0；tile1_color/tile2_color 仍分别来自两块素材。
#    - 新增的不是颜色映射，而是 wall 位置的取值逻辑，因此人类可以靠“位置=墙带”而不是靠颜色编号去归纳规则。
# 9) 新的 solver 假设与 generator 保证
#    - solver 假设：唯一 8 隔离 indicator、合法边界位置、wall 经过中心、wall 是简单 8 连通曲线路径、inside/outside 非空。
#    - generator 保证：只从满足这些结构条件的墙中采样，并且额外筛选出 wall 上确实出现融合行为的样本。

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
GEN_RETRIES = 400

CENTER = (2, 2)
MIN_WALL_LEN = 6
MAX_WALL_LEN = 7
VALID_INDICATORS = (
    (0, 0), (0, 2), (0, 4),
    (2, 0),         (2, 4),
    (4, 0), (4, 2), (4, 4),
)
ALL_CELLS = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)}
BOUNDARY_CELLS = tuple(
    (r, c)
    for r in range(TILE_SZ)
    for c in range(TILE_SZ)
    if r in (0, TILE_SZ - 1) or c in (0, TILE_SZ - 1)
)

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


def _neighbors8_in_bounds(r: int, c: int):
    for nr, nc in neighbors8(r, c):
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
    _, n_comp = label_mask(wall_mask, connectivity=8)
    if n_comp != 1:
        return None

    degrees = {}
    for cell in wall:
        degrees[cell] = sum((nbr in wall) for nbr in _neighbors8_in_bounds(*cell))
    if any(deg > 2 for deg in degrees.values()):
        return None

    ends = [cell for cell, deg in degrees.items() if deg == 1]
    if len(ends) != 2:
        return None

    path = [ends[0]]
    prev = None
    cur = ends[0]
    while True:
        nxt = [nbr for nbr in _neighbors8_in_bounds(*cur) if nbr in wall and nbr != prev]
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


def _has_diagonal_step(path: tuple[tuple[int, int], ...]) -> bool:
    return any(abs(r1 - r0) == 1 and abs(c1 - c0) == 1 for (r0, c0), (r1, c1) in zip(path, path[1:]))


def _build_curve_configs() -> list[tuple[tuple[int, int], frozenset[tuple[int, int]]]]:
    walls: dict[tuple[tuple[int, int], ...], frozenset[tuple[int, int]]] = {}

    def dfs(start: tuple[int, int], path: list[tuple[int, int]]) -> None:
        if len(path) > MAX_WALL_LEN:
            return
        cur = path[-1]
        if len(path) >= MIN_WALL_LEN and cur != start and _is_boundary(cur) and CENTER in path:
            wall = frozenset(path)
            ordered = _ordered_wall_path(wall)
            if ordered is None:
                return
            if _run_count(ordered) < 3:
                return
            if not _has_diagonal_step(ordered):
                return
            if len(_wall_side_tags(wall)) < 2:
                return
            walls[tuple(sorted(wall))] = wall
            return

        for nbr in _neighbors8_in_bounds(*cur):
            if nbr in path:
                continue
            dfs(start, path + [nbr])

    for start in BOUNDARY_CELLS:
        dfs(start, [start])

    configs: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], tuple[tuple[int, int], frozenset[tuple[int, int]]]] = {}
    for wall in walls.values():
        for indicator in VALID_INDICATORS:
            if indicator in wall:
                continue
            if any(nbr in wall for nbr in neighbors8(*indicator)):
                continue
            inside = _flood_inside(wall, indicator)
            outside = ALL_CELLS - wall - inside
            if not inside or not outside:
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


def _compose_output(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: frozenset[tuple[int, int]],
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != bg:
                    out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != bg:
                    out3[r, c] = v1
                elif v2 != bg:
                    out3[r, c] = v2
            else:
                if v2 != bg:
                    out3[r, c] = v2
    return out3


def _fusion_metrics(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: frozenset[tuple[int, int]],
) -> dict[str, int]:
    outside = ALL_CELLS - inside - wall
    inside_hits = sum(int(t1[r, c] != bg) for r, c in inside)
    outside_hits = sum(int(t2[r, c] != bg) for r, c in outside)
    wall_t1_hits = 0
    wall_t2_only = 0
    wall_overlap = 0
    for r, c in wall:
        has1 = int(t1[r, c] != bg)
        has2 = int(t2[r, c] != bg)
        wall_t1_hits += has1
        wall_t2_only += int((not has1) and has2)
        wall_overlap += int(has1 and has2)
    return {
        "inside_hits": inside_hits,
        "outside_hits": outside_hits,
        "wall_t1_hits": wall_t1_hits,
        "wall_t2_only": wall_t2_only,
        "wall_overlap": wall_overlap,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a fourth panel using a curved wall and an indicator pixel.

    Input structure:
    - Four 5x5 panels in one row, border-separated.
    - Panel 0: cue pixels = one 8-isolated indicator plus one center-crossing curved wall.
      The wall is a single-pixel staircase path: simple, 8-connected, edge-to-edge, and contains a diagonal step.
    - Panels 1-2: source patterns. Panel 3: blank.

    Rule:
    - Flood-fill (4-conn) from the indicator, blocked by the wall, to get the inside region.
    - Inside -> take non-bg pixels from panel 1. Outside -> take non-bg pixels from panel 2.
    - Wall -> OR-combine the two panels at that position:
      panel 1 wins if both are non-bg, otherwise keep whichever panel contributes a non-bg pixel.

    Key insight:
    - The curved wall is both the separator and a fusion band: it is not owned by just one source panel.
      You must infer the irregular partition from flood-fill, then apply a distinct wall-overlay rule on top of it.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(TILE_SZ) for c in range(TILE_SZ) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"
    assert indicator in set(VALID_INDICATORS), f"indicator {indicator} not at corner/edge-midpoint"

    wall = frozenset(cue_set - {indicator})
    assert wall, "missing wall"
    assert CENTER in wall, "wall must pass through the grid center (2,2)"
    assert not any((nr, nc) in wall for nr, nc in neighbors8(*indicator)), "indicator must be 8-isolated from wall"
    assert len(_wall_side_tags(wall)) >= 2, "wall must span at least two border sides"

    ordered = _ordered_wall_path(wall)
    assert ordered is not None, "wall must be a single simple 8-connected path"
    assert MIN_WALL_LEN <= len(ordered) <= MAX_WALL_LEN, "wall length must match the curved-wall family"
    assert _run_count(ordered) >= 3, "wall must bend at least twice"
    assert _has_diagonal_step(ordered), "wall must include a diagonal step"

    inside = _flood_inside(wall, indicator)
    outside = ALL_CELLS - wall - inside
    assert inside, "inside region is empty"
    assert outside, "outside region is empty"

    out3 = _compose_output(t1, t2, bg, inside, wall)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    indicator = rng.choice(list(CONFIGS_BY_INDICATOR))
    wall = rng.choice(CONFIGS_BY_INDICATOR[indicator])
    inside = _flood_inside(wall, indicator)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    metrics = _fusion_metrics(t1, t2, tile_bg, inside, wall)
    if metrics["inside_hits"] == 0 or metrics["outside_hits"] == 0:
        return None
    if metrics["wall_t1_hits"] == 0 or metrics["wall_t2_only"] == 0:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _compose_output(t1, t2, tile_bg, inside, wall)

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
