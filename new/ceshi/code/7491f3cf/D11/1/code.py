# Variant of 7491f3cf
# Dimension: D11 — 面板间的颜色约束 (meta)
# Change: cue_color、t1_color、t2_color三色互不相同且都不等于tile_bg；border可以是任意非tile_bg颜色
#   → 多色墙线：墙线由两种颜色沿 5 像素路径交替出现，指示像素使用第三种颜色。

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import flood_fill, label_mask, neighbors8

"""
Design analysis for the D11 variant
===================================

a) 具体化与行为追踪
- 版式、墙形状、tile1/tile2 的合成规则都不变；只有 tile0 的 cue 配色变化。
- tile0 仍然只有 6 个非背景像素：5 个墙像素 + 1 个指示像素。
- 原题：这 6 个像素里，墙和指示点共享同一个 cue_color。
- 变体：墙的 5 个像素按路径顺序交替着色为 A/B/A/B/A 或 B/A/B/A/B；指示像素用第三种颜色 I。
- 小例子：若墙是横线 {(2,0),(2,1),(2,2),(2,3),(2,4)}，指示点在 (0,2)。
  原题 tile0 是 `C C C C C` 一条纯色墙 + 1 个同色指示点。
  变体 tile0 是 `A B A B A` 一条双色交替墙 + 1 个颜色 I 的指示点。
  对 flood-fill 而言，正确逻辑仍应把 5 个墙像素的并集当成一整条隔墙；
  若只把颜色 A 或只把颜色 B 当成墙，填充会穿过“空隙”，inside/outside 会被算错。

b) 复杂度校准
- 只增加一个 cue 颜色，不引入新对象、不改面板尺寸、不改组合规则，避免偏离原题。
- 两种墙色固定为“沿同一条 5 像素路径交替”，而不是任意花纹；自由度只有两色与起始奇偶位，复杂度可控。
- 这个变化肉眼可见：tile0 会稳定出现“双色条纹墙 + 第三色指示点”。

c) 隐式假设审查
- 原 `cue_set = {t0 != t0_bg}` 这一步本身并不要求 cue 单色，因此“取所有非背景像素的并集”仍然成立。
- 但原题默认 cue 单色，导致“按颜色分对象”与“按几何并集分对象”在视觉上没有冲突；变体后不再如此。
- 新 solver 需要显式承认：tile0 中会有 3 个 cue 颜色，其中 1 个是 singleton 指示点颜色，另外 2 个颜色属于同一条墙。
- 原来“把任一非背景颜色都看作独立对象”在单色墙下恰好无害；现在会错，因为两种墙色单独都不足以阻挡 flood-fill。

d) 放置合法性
- 沿用原题已验证的 5 像素墙形状与 8 个直线 / 12 个 L 弯配置，不改变坐标集合，只改变其着色。
- 因为只是给既有 wall coords 重新分配 A/B 颜色，不会新增重叠、越界或破坏 indicator 的 8-孤立约束。
- generator 继续用 output-first 检查：先手工合成 out3，再要求 `main(input) == expected`。

e) 视觉可观察性
- tile1/tile2 仍保持单色对称图案，视觉复杂度集中在 tile0；解题者会先看到“墙不再纯色，但几何位置没变”。
- 交替色沿着单条直线/L 形路径分布，而不是散点分布，因此人类容易归纳“两色其实是一条墙”。

f) 可视化整洁度
- 不增加额外标记，不改变面板排布，只把旧墙重涂成双色条纹，因此整体仍然干净。
- 指示点仍然只有 1 个，而且是 8-孤立像素，和墙在视觉上分离清楚。

g) 颜色变化可推断性
- cue 侧只新增 1 个颜色维度：墙从 1 色变 2 色；指示点继续单独成色。
- generator 保证 wall_a、wall_b、indicator_color、t1_color、t2_color 五者互不相同且都不等于 tile_bg。
- border 维持原题规则：仍可为任意非 tile_bg 颜色，不把 border 也卷入新规则。

对新 solver 的额外假设审查
- 它现在假设 tile0 恰有 3 种非背景 cue 颜色，计数为 3/2/1。
- 它还假设那 5 个墙像素组成一条无分叉的 8-连通路径，并且路径颜色严格交替。
- generate_input() 通过固定 5 像素 wall 模板和交替着色来保证这些前提成立。
"""


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
LBEND_PROB = 0.2

VALID_INDICATORS = {
    (0, 0),
    (0, 2),
    (0, 4),
    (2, 0),
    (2, 4),
    (4, 0),
    (4, 2),
    (4, 4),
}

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

_STRAIGHT_CONFIGS = [
    ((0, 2), frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})),
    ((4, 2), frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)})),
    ((2, 0), frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)})),
    ((2, 4), frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)})),
    ((0, 0), frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)})),
    ((4, 4), frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)})),
    ((0, 4), frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})),
    ((4, 0), frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)})),
]

_LBEND_CONFIGS = [
    ((0, 0), frozenset({(2, 2), (1, 2), (0, 2), (2, 1), (2, 0)})),
    ((0, 0), frozenset({(2, 2), (3, 2), (4, 2), (2, 3), (2, 4)})),
    ((4, 4), frozenset({(2, 2), (1, 2), (0, 2), (2, 1), (2, 0)})),
    ((4, 4), frozenset({(2, 2), (3, 2), (4, 2), (2, 3), (2, 4)})),
    ((0, 4), frozenset({(2, 2), (1, 2), (0, 2), (2, 3), (2, 4)})),
    ((0, 4), frozenset({(2, 2), (3, 2), (4, 2), (2, 1), (2, 0)})),
    ((4, 0), frozenset({(2, 2), (1, 2), (0, 2), (2, 3), (2, 4)})),
    ((4, 0), frozenset({(2, 2), (3, 2), (4, 2), (2, 1), (2, 0)})),
    ((4, 2), frozenset({(2, 2), (1, 1), (0, 0), (1, 3), (0, 4)})),
    ((0, 2), frozenset({(2, 2), (3, 1), (4, 0), (3, 3), (4, 4)})),
    ((2, 4), frozenset({(2, 2), (1, 1), (0, 0), (3, 1), (4, 0)})),
    ((2, 0), frozenset({(2, 2), (1, 3), (0, 4), (3, 3), (4, 4)})),
]


def _majority_color(tile: np.ndarray) -> int:
    return int(Counter(map(int, tile.flatten())).most_common(1)[0][0])


def _ordered_wall_path(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> list[tuple[int, int]]:
    wall = set(wall)
    assert len(wall) == 5, "wall must have exactly 5 pixels"
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in wall:
        graph[(r, c)] = sorted((nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in wall)
    endpoints = sorted(node for node, ns in graph.items() if len(ns) == 1)
    starts = endpoints if endpoints else sorted(graph)

    def dfs(path: list[tuple[int, int]], seen: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == len(wall):
            return path
        cur = path[-1]
        for nxt in graph[cur]:
            if nxt in seen:
                continue
            got = dfs(path + [nxt], seen | {nxt})
            if got is not None:
                return got
        return None

    for start in starts:
        path = dfs([start], {start})
        if path is not None:
            return path
    raise AssertionError(f"failed to recover a 5-pixel wall path from {sorted(wall)}")


def _reflection_for_indicator(indicator: tuple[int, int]):
    ir, ic = indicator
    dr, dc = 2 - ir, 2 - ic
    if dr == 0:
        return lambda r, c: (4 - r, c)
    if dc == 0:
        return lambda r, c: (r, 4 - c)
    if dr * dc > 0:
        return lambda r, c: (c, r)
    return lambda r, c: (4 - c, 4 - r)


def _flood_inside(blockers: set[tuple[int, int]] | frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in blockers:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _compose_output_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    inside: set[tuple[int, int]],
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    wall_4conn: bool,
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    wall = set(wall)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != bg:
                    out3[r, c] = v1
            elif (r, c) in wall:
                if v1 != bg:
                    out3[r, c] = v1
                elif wall_4conn and v2 != bg:
                    out3[r, c] = v2
            else:
                if v2 != bg:
                    out3[r, c] = v2
    return out3


def _extract_indicator_and_wall(
    t0: np.ndarray,
) -> tuple[int, tuple[int, int], set[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]:
    t0_bg = _majority_color(t0)
    color_to_coords: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v = int(t0[r, c])
            if v != t0_bg:
                color_to_coords[v].append((r, c))

    cue_pixels = {coord for coords in color_to_coords.values() for coord in coords}
    assert len(cue_pixels) == 6, f"tile0 must contain 6 cue pixels, got {len(cue_pixels)}"
    assert len(color_to_coords) == 3, f"tile0 must use 3 cue colors, got {sorted(color_to_coords)}"

    singleton_colors = [color for color, coords in color_to_coords.items() if len(coords) == 1]
    assert len(singleton_colors) == 1, f"expected one singleton indicator color, got {singleton_colors}"
    indicator_color = singleton_colors[0]
    indicator = color_to_coords[indicator_color][0]

    assert indicator in VALID_INDICATORS, f"indicator {indicator} not at corner/edge-midpoint"
    assert not any((nr, nc) in cue_pixels for nr, nc in neighbors8(indicator[0], indicator[1])), (
        "indicator must be 8-isolated from the wall"
    )

    wall_colors = tuple(sorted(color for color in color_to_coords if color != indicator_color))
    wall_color_counts = sorted(len(color_to_coords[color]) for color in wall_colors)
    assert wall_color_counts == [2, 3], f"wall colors must contribute 2 and 3 pixels, got {wall_color_counts}"

    wall = cue_pixels - {indicator}
    path = _ordered_wall_path(wall)
    path_colors = [int(t0[r, c]) for r, c in path]
    assert len(set(path_colors)) == 2, "wall must use exactly two colors"
    assert all(path_colors[i] != path_colors[i + 1] for i in range(len(path_colors) - 1)), (
        "wall colors must alternate along the path"
    )

    return t0_bg, indicator, wall, wall_colors, path


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
    Composite tile1/tile2 into tile3 using tile0 as a partition cue.

    Variant change:
    - The wall in tile0 is not monochrome anymore.
    - Two wall colors alternate along one 5-pixel 8-connected path.
    - The indicator is a third color and remains 8-isolated.
    """
    g = np.array(grid, dtype=int)

    border = int(g[0, 0])
    tiles = [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]
    t0, t1, t2, _t3 = tiles
    bg = _majority_color(t1)
    assert border != bg, "border color must differ from tile background"

    _t0_bg, indicator, wall, _wall_colors, _path = _extract_indicator_and_wall(t0)

    assert (2, 2) in wall, "wall must pass through the grid center (2,2)"
    refl = _reflection_for_indicator(indicator)
    assert wall == {refl(r, c) for r, c in wall}, "wall must be symmetric about indicator→center axis"

    sides_touched = set()
    for r, c in wall:
        if r == 0:
            sides_touched.add("top")
        if r == 4:
            sides_touched.add("bottom")
        if c == 0:
            sides_touched.add("left")
        if c == 4:
            sides_touched.add("right")
    assert len(sides_touched) >= 2, f"wall must span edge-to-edge, only touches {sides_touched}"

    inside = _flood_inside(wall, indicator)
    outside = {(r, c) for r in range(TILE_SZ) for c in range(TILE_SZ)} - inside - wall
    assert outside, "wall must partition the grid — outside region is empty"

    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = n_comp == 1

    out3 = _compose_output_tile(t1, t2, bg, inside, wall, wall_4conn)
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]

    wall_a, wall_b, indicator_color, t1_color, t2_color = rng.sample(other, 5)
    border = rng.choice(other)

    pool = _LBEND_CONFIGS if rng.random() < LBEND_PROB else _STRAIGHT_CONFIGS
    indicator, wall = rng.choice(pool)
    wall_path = _ordered_wall_path(wall)
    start_color, next_color = (wall_a, wall_b) if rng.random() < 0.5 else (wall_b, wall_a)

    inside = _flood_inside(wall, indicator)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for idx, (r, c) in enumerate(wall_path):
        t0[r, c] = start_color if idx % 2 == 0 else next_color
    t0[indicator[0], indicator[1]] = indicator_color

    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = n_comp == 1

    out3 = _compose_output_tile(t1, t2, tile_bg, inside, wall, wall_4conn)

    color_a_cells = {(r, c) for r, c in wall if int(t0[r, c]) == wall_a}
    color_b_cells = {(r, c) for r, c in wall if int(t0[r, c]) == wall_b}
    assert color_a_cells and color_b_cells, "generator must use both wall colors"
    if _flood_inside(color_a_cells, indicator) == inside:
        return None
    if _flood_inside(color_b_cells, indicator) == inside:
        return None

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
