from __future__ import annotations

# Variant of 7491f3cf
# Dimension: D3 — 墙线形状类型 (match_cond)
# Change: 直线（行/列/对角线/反对角线）或L弯，经过中心(2,2)，关于指示像素到中心连线对称...
#         → 曲线墙（阶梯状折线）
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入仍是四个 5x5 面板横向排列；tile0 给出“墙 + 指示像素”，tile1/tile2 为素材，tile3 为空白。
#    - 变体的核心变化只发生在 tile0 的墙：原题墙只有 5 个像素，且只能是直线或 L 弯。
#      这里把墙改成 7 像素左右的阶梯状折线，仍经过中心 (2,2)，仍把 5x5 分割成指示像素侧和另一侧。
#    - 具体例子：原题中 indicator=(0,2) 时，墙可能是整条第 2 行；flood-fill 后 inside 就是上半区。
#      变体中 indicator=(0,2) 时，墙可能变成
#      {(0,1),(1,1),(1,2),(2,2),(3,2),(3,3),(4,3)}。
#      flood-fill 仍从 (0,2) 出发，但 inside/outside 会沿着阶梯边界被不规则地切开。
#      输出合成规则不变，只是“哪一格属于 inside”变复杂了。
# b) 复杂度校准
#    - 仍固定 5x5、单个 indicator、两个单色对称素材，避免同时改动多个维度。
#    - 墙只允许简单的 4 连通折线，不允许分叉和闭环；这样视觉上能明显看出“弯折”，但不会乱成多段障碍。
#    - 预计算合法曲线墙配置，样例中会重复出现“经过中心、阶梯分割”的规律，便于从 2-4 个样例归纳。
# c) 隐式假设审查
#    - 原 solver 中“wall 必须关于 indicator→center 轴对称”的断言依赖旧墙族；曲线墙允许 Z/S 式非对称折线，
#      所以这条断言必须删除，否则新样例会被误判为非法。
#    - 原 solver 的 flood-fill / inside-outside 合成逻辑并不依赖直线或 L 弯，只依赖“wall 能分割平面且 indicator 8-隔离”，
#      因此这些代码可保留。
#    - 原 generator 默认墙只有 5 像素；变体里墙更长，但仍控制在 7 像素，使 tile0 背景继续是多数色。
# d) 放置合法性
#    - 所有墙都预先枚举为：4 连通、无分叉、经过中心、触边至少两侧、能把 indicator 一侧与另一侧分开。
#    - indicator 仍限制在四角或四个边中点，并要求与整条墙 8-隔离。
#    - tile1/tile2/tile3 的布局和颜色约束维持原题，不重叠、不越界。
# e) 视觉可观察性
#    - 相比原题“直线/L 弯”，新墙会出现 3 段及以上方向 run 的阶梯边界，肉眼可见更曲折。
#    - 输出仍只在第 4 个面板变化，解题者可以直接对照 tile0 的折线与 tile3 的分区边界。
# f) 可视化整洁度
#    - 墙仍是单色单像素宽折线，不引入额外标记；tile1/tile2 仍是单色对称图案，层次与原题一致。
# g) 颜色变化可推断性
#    - 颜色角色完全沿用原题：cue_color 只负责墙/indicator，tile1_color 与 tile2_color 分别负责两侧素材。
#      不新增颜色映射规则。
# 9) 新的 solver 假设与 generator 保证
#    - solver 现在只假设：存在唯一 8-隔离 indicator；wall 经过中心；wall 触边至少两侧；wall 能分割平面。
#    - generator 通过预枚举曲线墙配置来保证这些条件，同时保证墙是简单阶梯折线而不是旧式 5 像素直线/L 弯。

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
MAX_BRANCH_LEN = 5  # center->boundary cells, inclusive


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

    degrees = {}
    for cell in wall:
        degrees[cell] = sum((nbr in wall) for nbr in _neighbors4(*cell))
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
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    if not dirs:
        return 0
    runs = 1
    for d0, d1 in zip(dirs, dirs[1:]):
        if d0 != d1:
            runs += 1
    return runs


def _enumerate_center_to_boundary_paths() -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []

    def dfs(path: list[tuple[int, int]]) -> None:
        r, c = path[-1]
        if len(path) > MAX_BRANCH_LEN:
            return
        if len(path) > 1 and _is_boundary((r, c)):
            out.append(tuple(path))
            return
        for nbr in _neighbors4(r, c):
            if nbr in path:
                continue
            dfs(path + [nbr])

    dfs([CENTER])
    return out


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
            if len(wall) < 7:
                continue
            if CENTER not in wall:
                continue
            if len(_wall_side_tags(wall)) < 2:
                continue

            ordered = _ordered_wall_path(wall)
            if ordered is None:
                continue
            if _run_count(ordered) < 3:
                continue

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two pattern tiles into a fourth panel using flood-fill from an indicator pixel.

    Input structure:
    - Four 5x5 panels in one row, border-separated.
    - Panel 0: cue pixels = a center-crossing curved wall (single-pixel staircase)
      plus one 8-isolated indicator pixel.
    - Panels 1-2: source patterns. Panel 3: blank.

    Rule:
    - Flood-fill (4-conn) from indicator, blocked by wall -> inside region.
    - Inside -> non-bg from panel 1; outside -> non-bg from panel 2.
    - Wall -> panel 1 if non-bg; otherwise panel 2 fallback when the wall is 4-connected.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    b = int(Counter(t1.flatten()).most_common(1)[0][0])
    assert border != b, "border color must differ from tile background"

    t0_bg = int(Counter(t0.flatten()).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(5) for c in range(5) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"
    assert indicator in set(VALID_INDICATORS), f"indicator {indicator} not at corner/edge-midpoint"

    wall = cue_set - {indicator}
    assert wall, "missing wall"
    assert CENTER in wall, "wall must pass through the grid center (2,2)"
    assert not any((nr, nc) in wall for nr, nc in neighbors8(*indicator)), "indicator must be 8-isolated from wall"

    sides_touched = _wall_side_tags(frozenset(wall))
    assert len(sides_touched) >= 2, f"wall must span edge-to-edge, only touches {sorted(sides_touched)}"

    ordered = _ordered_wall_path(frozenset(wall))
    assert ordered is not None, "wall must be a single simple 4-connected staircase path"
    assert _run_count(ordered) >= 3, "wall must bend at least twice to form a curved staircase"

    EMPTY, WALL, INSIDE = 0, 1, 2
    tmp = np.zeros((5, 5), dtype=int)
    for r, c in wall:
        tmp[r, c] = WALL
    flood_fill(tmp, indicator[0], indicator[1], INSIDE, connectivity=4)
    inside = {(int(r), int(c)) for r, c in np.argwhere(tmp == INSIDE)}

    outside = {(r, c) for r in range(5) for c in range(5)} - inside - wall
    assert inside, "inside region is empty"
    assert outside, "wall must partition the grid — outside region is empty"

    wall_mask = np.zeros((5, 5), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = n_comp == 1

    out3 = np.full((5, 5), b, dtype=int)
    for r in range(5):
        for c in range(5):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != b:
                    out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != b:
                    out3[r, c] = v1
                elif wall_4conn and v2 != b:
                    out3[r, c] = v2
            else:
                if v2 != b:
                    out3[r, c] = v2

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

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = n_comp == 1

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
