# Combo variant of 7491f3cf (index 2)
# combo_spec: D2/0+D7/2
# Name: 颜色匹配对应+多指示像素
#
# Design analysis:
# a) 具体化与行为追踪
#    - 原题的 tile0 是“5 像素 wall + 1 个孤立 indicator”，并且默认 indicator 所在侧一定取 tile1。
#    - 本融合版的 tile0 改成“5 像素 wall + 2 个孤立 marker”：
#      * 一个 marker 的颜色等于 tile1 的前景色
#      * 另一个 marker 的颜色等于 tile2 的前景色
#      * 两个 marker 分别位于 wall 划分出的两个 4 连通开放区域中
#    - 解题流程不再是“找到唯一 indicator -> inside 固定用 tile1”，而是：
#      1. 识别 tile0 中除 wall 外的两个 marker
#      2. 从两个 marker 各自 flood-fill，得到 wall 两侧的两个区域
#      3. 用 marker 颜色和 tile1/tile2 的前景色做匹配，决定哪个区域取 tile1、哪个区域取 tile2
#    - 例子：若上侧 marker 颜色等于 tile2 前景色、下侧 marker 颜色等于 tile1 前景色，
#      那么上侧区域必须取 tile2，下侧区域必须取 tile1；这与原题“指示侧恒取 tile1”显式不同。
#
# b) 复杂度校准
#    - 只增加 1 个 marker，颜色种类仍控制在 border、tile_bg、wall_color、tile1_color、tile2_color 共 5 类。
#    - 区域到 source 的对应关系只有两种分配：region A->tile1 / region B->tile2，或反过来。
#    - generator 会随机翻转两侧 marker 的颜色分配，并过滤“翻转后输出几乎不变”的样本，
#      避免颜色匹配退化成装饰信息。
#
# c) 隐式假设审查
#    - 原 main() 假设“恰有一个 8-孤立 cue 像素”，融合后失效；现在必须允许两个孤立 marker。
#    - 原 main() 假设“wall = cue_set - {indicator}”，融合后失效；现在必须按颜色拆出 wall 与 markers。
#    - 原 main() 假设“wall 关于 indicator->center 轴对称”，这是单 indicator 语义的产物；融合后不再适用。
#    - 原 main() 假设“inside 固定对应 tile1”；融合后必须改成“marker 颜色匹配决定 region->tile 对应”。
#    - 原题的 wall 混合规则保留：wall 仍优先取 tile1，只有 wall 为 4 连通时 tile2 才能补位。
#      这里不沿用 D2/0 单变体里的“inside-source 主导 wall”，因为融合后不存在唯一 inside-source；
#      两个 marker 只定义两个开放区域，wall 继续沿用原题的固定规则更一致、也更易观察。
#
# d) 放置合法性
#    - wall 只从已验证可分区的 5 像素形状池中取样，保证经过中心且把 5x5 分成恰好两个开放区域。
#    - marker 只放在 8 个原始 anchor 位置，且必须：
#      * 不与 wall 重叠
#      * 与 wall 8-孤立
#      * 两个 marker 落在不同开放区域
#    - generator 先算开放区域标签，再从各区域合法 anchor 中取样，因此不会越界或重叠。
#
# e) 视觉可观察性
#    - 解题者在 tile0 中能直接看到 wall_color 之外还有 tile1_color 和 tile2_color 两个 marker。
#    - marker 颜色与 source tile 的整块图案颜色一致，因此“这个区域对应哪张 source tile”有直接视觉线索。
#    - marker 的位置分配会在样本间翻转，避免人类误学成“上/左总是 tile1，下/右总是 tile2”。
#
# f) 可视化整洁度
#    - marker 仍限制在边角/边中点 8 个位置，且与 wall 8-隔离，不会和 wall 糊成一个对象。
#    - tile0 中至多出现 7 个非背景像素，元素数很小；source tiles 仍是单前景色对称图案，整体视觉清楚。
#
# g) 变化的可推断性
#    - 颜色变化不是任意置换：marker 颜色必须精确复用 tile1/tile2 的前景色。
#    - 颜色匹配总是伴随“落在哪个开放区域”这一结构线索，因此人类可以在少量样例中推出规则。
#    - generator 使用量化指标 `swap_diff = count_nonzero(out3 != swapped_out3)` 检查核心融合行为：
#      若交换两个 marker 对应区域后输出几乎不变，则该样本退化，直接丢弃。
#
# New main() assumption audit
# - tile1 与 tile2 必须共享背景，且各自恰好有一种前景色；generator 的对称图案采样器保证这一点。
# - tile0 的非背景像素必须由 5 个 monochrome wall 像素和 2 个 marker 构成；generator 固定如此生成。
# - 两个 marker 的颜色必须分别等于 tile1/tile2 前景色，且这两种颜色不同；generator 显式采样不同颜色。
# - wall 必须把开放格划成恰好两个 4 连通区域；wall 池与 `_open_region_labels()` 共同保证这一点。
# - 两个 marker 必须位于不同区域；generator 先按区域选 anchor，再由 main() 断言复核。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import flood_fill, get_bg, label_mask, neighbors8


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300
LBEND_PROB = 0.2
MIN_SWAP_DIFF = 2

VALID_MARKER_POS = {
    (0, 0), (0, 2), (0, 4),
    (2, 0),         (2, 4),
    (4, 0), (4, 2), (4, 4),
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

_STRAIGHT_WALLS = [
    frozenset({(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}),
    frozenset({(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)}),
    frozenset({(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)}),
    frozenset({(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}),
]

_LBEND_WALLS = [
    frozenset({(2, 2), (1, 2), (0, 2), (2, 1), (2, 0)}),
    frozenset({(2, 2), (3, 2), (4, 2), (2, 3), (2, 4)}),
    frozenset({(2, 2), (1, 2), (0, 2), (2, 3), (2, 4)}),
    frozenset({(2, 2), (3, 2), (4, 2), (2, 1), (2, 0)}),
    frozenset({(2, 2), (1, 1), (0, 0), (1, 3), (0, 4)}),
    frozenset({(2, 2), (3, 1), (4, 0), (3, 3), (4, 4)}),
    frozenset({(2, 2), (1, 1), (0, 0), (3, 1), (4, 0)}),
    frozenset({(2, 2), (1, 3), (0, 4), (3, 3), (4, 4)}),
]


def _extract_tiles(g: np.ndarray) -> list[np.ndarray]:
    return [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]


def _single_fg_color(tile: np.ndarray, bg: int, *, name: str) -> int:
    fg = {int(v) for v in np.unique(tile) if int(v) != int(bg)}
    assert len(fg) == 1, f"{name} must use exactly one foreground color, got {fg}"
    return next(iter(fg))


def _wall_is_4_connected(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return bool(n_comp == 1)


def _open_region_labels(
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]]
) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    open_mask = np.ones((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        open_mask[r, c] = False
    labels, n_comp = label_mask(open_mask, connectivity=4)
    assert n_comp == 2, f"wall must partition the tile into exactly 2 open regions, got {n_comp}"
    regions: dict[int, set[tuple[int, int]]] = {}
    for k in range(1, n_comp + 1):
        regions[k] = {(int(r), int(c)) for r, c in np.argwhere(labels == k)}
    return labels, regions


def _cue_parts(
    t0: np.ndarray, bg: int, t1_color: int, t2_color: int
) -> tuple[set[tuple[int, int]], dict[int, tuple[int, int]]]:
    wall: set[tuple[int, int]] = set()
    markers: dict[int, tuple[int, int]] = {}
    wall_color = None

    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v = int(t0[r, c])
            if v == bg:
                continue
            pos = (r, c)
            if v == t1_color:
                assert t1_color not in markers, "multiple tile1-colored markers in cue tile"
                markers[t1_color] = pos
            elif v == t2_color:
                assert t2_color not in markers, "multiple tile2-colored markers in cue tile"
                markers[t2_color] = pos
            else:
                if wall_color is None:
                    wall_color = v
                assert v == wall_color, "wall must be monochrome"
                wall.add(pos)

    assert len(wall) == 5, f"expected 5 wall pixels, got {len(wall)}"
    assert set(markers) == {t1_color, t2_color}, "missing one or both color-matched markers"
    return wall, markers


def _composite_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    t1_region: set[tuple[int, int]],
    t2_region: set[tuple[int, int]],
) -> np.ndarray:
    wall_4conn = _wall_is_4_connected(wall)
    out3 = np.full((TILE_SZ, TILE_SZ), int(bg), dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            pos = (r, c)
            v1 = int(t1[r, c])
            v2 = int(t2[r, c])
            if pos in t1_region:
                if v1 != int(bg):
                    out3[r, c] = v1
            elif pos in wall:
                if v1 != int(bg):
                    out3[r, c] = v1
                elif wall_4conn and v2 != int(bg):
                    out3[r, c] = v2
            else:
                assert pos in t2_region, "every non-wall cell must belong to exactly one matched region"
                if v2 != int(bg):
                    out3[r, c] = v2
    return out3


def _swap_diff_metric(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    t1_region: set[tuple[int, int]],
    t2_region: set[tuple[int, int]],
) -> int:
    forward = _composite_tile(t1, t2, bg, wall, t1_region, t2_region)
    swapped = _composite_tile(t1, t2, bg, wall, t2_region, t1_region)
    return int(np.count_nonzero(forward != swapped))


def _sample_symmetric_pattern(
    bg: int, fg: int, rng: random.Random, *, min_px: int, max_px: int
) -> np.ndarray | None:
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(_ORBITS)))
        rng.shuffle(order)
        tile = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
        count = 0
        for i in order:
            orbit = _ORBITS[i]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    tile[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return tile
    return None


def _marker_candidates_by_region(
    wall: frozenset[tuple[int, int]]
) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    labels, regions = _open_region_labels(wall)
    candidates: dict[int, list[tuple[int, int]]] = {k: [] for k in regions}
    for pos in sorted(VALID_MARKER_POS):
        if pos in wall:
            continue
        if any((nr, nc) in wall for nr, nc in neighbors8(*pos)):
            continue
        label = int(labels[pos[0], pos[1]])
        if label != 0:
            candidates[label].append(pos)
    return labels, regions, candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite tile1 and tile2 into tile3 using a wall plus two color-matched markers.

    Input structure:
    - Four 5x5 panels in one row, border-separated.
    - tile0 contains a 5-pixel wall in one cue color and two 8-isolated markers.
    - The two markers lie in different wall-defined open regions and their colors
      equal the foreground colors of tile1 and tile2 respectively.
    - tile1 and tile2 are single-foreground-color symmetric source patterns.
    - tile3 is blank in the input.

    Rule:
    - Recover the wall and the two markers from tile0.
    - Flood-fill / label the two open regions separated by the wall.
    - The region containing the tile1-colored marker copies non-bg pixels from tile1.
    - The region containing the tile2-colored marker copies non-bg pixels from tile2.
    - Wall pixels keep the original task's blending rule: tile1 has priority, and
      tile2 only fills uncovered wall pixels when the wall is 4-connected.

    Key insight:
    - Geometry alone no longer tells you which side is tile1 or tile2. You must first
      detect the two isolated markers, then match each marker's color to the source-tile
      foreground color, and only then assign the wall-separated regions.
    """
    g = np.asarray(grid, dtype=int)
    border = int(g[0, 0])
    t0, t1, t2, _t3 = _extract_tiles(g)

    bg = get_bg(t1)
    assert border != bg, "border color must differ from tile background"
    assert get_bg(t2) == bg, "tile1 and tile2 must share the same background"
    assert get_bg(t0) == bg, "cue tile background must match source tiles"

    t1_color = _single_fg_color(t1, bg, name="tile1")
    t2_color = _single_fg_color(t2, bg, name="tile2")
    assert t1_color != t2_color, "tile1 and tile2 foreground colors must differ"

    wall, markers = _cue_parts(t0, bg, t1_color, t2_color)

    assert (2, 2) in wall, "wall must pass through the center"
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

    for color, pos in markers.items():
        assert pos in VALID_MARKER_POS, f"marker {color} at illegal anchor {pos}"
        assert pos not in wall, "marker cannot overlap wall"
        assert not any((nr, nc) in wall for nr, nc in neighbors8(*pos)), "marker must be 8-isolated from wall"

    labels, regions = _open_region_labels(wall)
    t1_label = int(labels[markers[t1_color][0], markers[t1_color][1]])
    t2_label = int(labels[markers[t2_color][0], markers[t2_color][1]])
    assert t1_label != 0 and t2_label != 0, "markers must lie inside open regions"
    assert t1_label != t2_label, "the two markers must occupy different regions"

    t1_region = regions[t1_label]
    t2_region = regions[t2_label]
    assert t1_region.isdisjoint(t2_region), "matched regions must be disjoint"

    # Explicit dual flood-fill sanity check: each marker should recover exactly one open region.
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, markers[t1_color][0], markers[t1_color][1], 2, connectivity=4)
    flood_fill(tmp, markers[t2_color][0], markers[t2_color][1], 3, connectivity=4)
    ff_t1 = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    ff_t2 = {(int(r), int(c)) for r, c in np.argwhere(tmp == 3)}
    assert ff_t1 == t1_region and ff_t2 == t2_region, "marker flood-fills must match the wall partition"

    out3 = _composite_tile(t1, t2, bg, wall, t1_region, t2_region)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    wall_color, t1_color, t2_color, border = rng.sample(other, 4)

    wall_pool = _LBEND_WALLS if rng.random() < LBEND_PROB else _STRAIGHT_WALLS
    wall = rng.choice(wall_pool)

    labels, regions, candidates = _marker_candidates_by_region(wall)
    region_ids = sorted(regions)
    if len(region_ids) != 2:
        return None
    if any(not candidates[k] for k in region_ids):
        return None

    pos_a = rng.choice(candidates[region_ids[0]])
    pos_b = rng.choice(candidates[region_ids[1]])
    if pos_a == pos_b:
        return None

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    if np.array_equal(t1 == tile_bg, t2 == tile_bg):
        return None

    if rng.random() < 0.5:
        markers = {t1_color: pos_a, t2_color: pos_b}
    else:
        markers = {t1_color: pos_b, t2_color: pos_a}

    t1_label = int(labels[markers[t1_color][0], markers[t1_color][1]])
    t2_label = int(labels[markers[t2_color][0], markers[t2_color][1]])
    if t1_label == 0 or t2_label == 0 or t1_label == t2_label:
        return None

    t1_region = regions[t1_label]
    t2_region = regions[t2_label]
    swap_diff = _swap_diff_metric(t1, t2, tile_bg, wall, t1_region, t2_region)
    if swap_diff < MIN_SWAP_DIFF:
        return None

    out3 = _composite_tile(t1, t2, tile_bg, wall, t1_region, t2_region)

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = wall_color
    t0[markers[t1_color][0], markers[t1_color][1]] = t1_color
    t0[markers[t2_color][0], markers[t2_color][1]] = t2_color

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
