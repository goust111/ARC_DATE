# Variant of 7491f3cf
# Dimension: D7 — 指示像素的位置约束 (symbol_instructions)
# Change: 指示像素必须在面板角落{(0,0),(0,4),(4,0),(4,4)}或边中点{(0,2),(2,0),(2,4),(4,2)}之一
#   且与墙像素8-孤立
#   → 多个指示像素：在墙两侧各放 1 个 marker，marker 颜色分别等于 tile1/tile2 的前景色，
#     解题时需要通过 marker 颜色与 source tile 颜色匹配，决定每个区域该取哪一个 tile。
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入仍是 4 个 5x5 面板横排。tile0 不再是“5 像素墙 + 1 个同色 indicator”，而是
#      “5 像素墙(单独的 cue_color) + 2 个 marker 像素(marker 颜色分别等于 t1/t2 前景色)”。
#    - 两个 marker 仍放在原 D7 的 8 个象征性 anchor 位置中，并保持与墙 8-孤立，视觉上仍然像
#      “边缘提示”，但现在提示的是“这个区域该用哪个 source tile”。
#    - 原流程：找到唯一孤立 indicator -> flood-fill 得 inside -> inside 用 tile1，outside 用 tile2。
#    - 变体流程：找到两个 marker -> 看 marker 颜色分别匹配 tile1/tile2 的前景色 -> marker 所在的
#      两个 4 连通区域分别取对应 tile。也就是说，“哪一侧是 tile1”不再由几何上的单指示点固定，
#      而由颜色匹配决定，且该对应关系在样本间会翻转。
# b) 复杂度校准
#    - 只新增 1 个像素，不引入额外对象类别或额外面板，复杂度接近原题。
#    - 为避免退化成原题，generator 会随机交换两侧 marker 的颜色分配，并拒绝“交换 marker 后输出不变”
#      的样本，确保 marker 颜色真地控制了结果。
# c) 隐式假设审查（原 main 中与 D7 相关的旧假设）
#    - “存在唯一一个 8-孤立 cue 像素”不再成立：现在是两个 marker，且颜色不同。
#    - “墙关于 indicator->center 轴对称”不再适用：该约束绑定了单个 indicator 的几何意义。
#      变体保留原墙形集合，但去掉这个依赖单指示点的断言。
#    - “wall = cue_set - {indicator}”不再成立：现在需要按颜色拆出 wall 与两个 marker。
#    - “inside 固定对应 tile1”不再成立：区域与 tile 的对应关系由 marker 颜色匹配决定。
# d) 放置合法性
#    - 仍使用原题可行的 5 像素墙集合，保证穿过中心且把 5x5 切成两个 4 连通区域。
#    - marker 只从 8 个 anchor 位置中选，且必须位于某个区域内部、与墙 8-孤立、彼此不重合。
#    - 每个区域必须至少有 1 个合法 anchor，否则该墙形不使用。
# e) 视觉可观察性
#    - tile0 现在同时出现“墙色 + t1 色 + t2 色”，解题者能直接看到两个 colored marker 分别在墙两侧。
#    - 因为 marker 颜色与 tile1/tile2 的图案颜色完全相同，所以颜色匹配规则可以从少量样例中归纳。
#    - generator 会让“左/上侧不总是 tile1”，否则第二个 marker 只是装饰。
# f) 可视化整洁度
#    - 只在 tile0 增加 1 个像素，且仍沿用边缘 anchor，避免在中心区域制造遮挡。
#    - marker 与墙保持 8-孤立，避免 cue 混成一个杂乱团块。
# g) 颜色变化可推断性
#    - 颜色种类仍受控：border、tile_bg、cue_color、t1_color、t2_color 共 5 类。
#    - marker 颜色不是任意映射，而是直接复用 tile1/tile2 的前景色，因此“颜色如何决定取哪张 tile”
#      有明确结构线索。
#
# New main() assumption audit
# - tile1/tile2 必须各自是单前景色图案，且两种前景色不同；generator 显式构造并断言这一点。
# - tile0 的非背景像素必须恰好由 5 个 wall 像素 + 2 个 marker 组成；generator 固定如此放置。
# - 两个 marker 必须落在 wall 分出的两个不同 4 连通区域中；generator 通过连通域候选位保证。
# - 交换两个 marker 颜色后，输出应发生可见变化；generator 过滤退化样本。

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


def _single_fg_color(tile: np.ndarray, bg: int) -> int:
    fg = {int(v) for v in np.unique(tile) if int(v) != bg}
    assert len(fg) == 1, f"expected exactly one foreground color, got {fg}"
    return next(iter(fg))


def _wall_connectivity_flag(wall: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return n_comp == 1


def _open_region_labels(
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]]
) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    open_mask = np.ones((TILE_SZ, TILE_SZ), dtype=bool)
    for r, c in wall:
        open_mask[r, c] = False
    lab, n_comp = label_mask(open_mask, connectivity=4)
    assert n_comp == 2, f"wall must partition the tile into exactly 2 regions, got {n_comp}"
    regions: dict[int, set[tuple[int, int]]] = {}
    for k in range(1, n_comp + 1):
        regions[k] = {(int(r), int(c)) for r, c in np.argwhere(lab == k)}
    return lab, regions


def _cue_parts(
    t0: np.ndarray, tile_bg: int, t1_color: int, t2_color: int
) -> tuple[set[tuple[int, int]], dict[int, tuple[int, int]]]:
    wall: set[tuple[int, int]] = set()
    markers: dict[int, tuple[int, int]] = {}
    wall_color = None
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v = int(t0[r, c])
            if v == tile_bg:
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
    assert set(markers) == {t1_color, t2_color}, "missing one or both colored markers"
    return wall, markers


def _composite_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: set[tuple[int, int]] | frozenset[tuple[int, int]],
    t1_region: set[tuple[int, int]],
    t2_region: set[tuple[int, int]],
) -> np.ndarray:
    wall_4conn = _wall_connectivity_flag(wall)
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            pos = (r, c)
            v1 = int(t1[r, c])
            v2 = int(t2[r, c])
            if pos in t1_region:
                if v1 != bg:
                    out3[r, c] = v1
            elif pos in wall:
                if v1 != bg:
                    out3[r, c] = v1
                elif wall_4conn and v2 != bg:
                    out3[r, c] = v2
            else:
                assert pos in t2_region, "non-wall cells must belong to one of the two regions"
                if v2 != bg:
                    out3[r, c] = v2
    return out3


def _sample_symmetric_pattern(
    bg: int, fg: int, rng: random.Random, *, min_px: int, max_px: int
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


def _marker_candidates_by_region(
    wall: frozenset[tuple[int, int]]
) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    lab, regions = _open_region_labels(wall)
    candidates: dict[int, list[tuple[int, int]]] = {k: [] for k in regions}
    for pos in sorted(VALID_MARKER_POS):
        if pos in wall:
            continue
        if any((nr, nc) in wall for nr, nc in neighbors8(*pos)):
            continue
        label = int(lab[pos[0], pos[1]])
        if label != 0:
            candidates[label].append(pos)
    return lab, regions, candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite tile1/tile2 into tile3 using a wall plus two color-coded markers.

    Input structure:
    - Four 5x5 panels in one row, border-separated.
    - tile0 contains:
      * a 5-pixel wall in a dedicated cue color
      * one tile1-colored marker in one wall-defined region
      * one tile2-colored marker in the other wall-defined region
    - tile1 and tile2 are single-color symmetric source patterns.
    - tile3 is blank.

    Rule:
    - The region containing the tile1-colored marker takes non-bg pixels from tile1.
    - The region containing the tile2-colored marker takes non-bg pixels from tile2.
    - Wall blending is unchanged from the original task:
      tile1 has priority on wall pixels; tile2 can fill wall pixels only when the wall is 4-connected.
    """
    g = np.asarray(grid, dtype=int)
    border = int(g[0, 0])
    t0, t1, t2, _t3 = _extract_tiles(g)

    bg = get_bg(t1)
    assert border != bg, "border color must differ from tile background"
    assert get_bg(t2) == bg, "tile backgrounds must match"
    assert get_bg(t0) == bg, "cue tile background must match source tiles"

    t1_color = _single_fg_color(t1, bg)
    t2_color = _single_fg_color(t2, bg)
    assert t1_color != t2_color, "tile1 and tile2 colors must differ"

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

    lab, regions = _open_region_labels(wall)
    t1_label = int(lab[markers[t1_color][0], markers[t1_color][1]])
    t2_label = int(lab[markers[t2_color][0], markers[t2_color][1]])
    assert t1_label != 0 and t2_label != 0, "markers must lie in open regions"
    assert t1_label != t2_label, "the two markers must occupy different regions"

    t1_region = regions[t1_label]
    t2_region = regions[t2_label]
    assert t1_region.isdisjoint(t2_region), "regions must be disjoint"

    # Equivalent to dual flood-fill from the two markers, retained as an explicit sanity check.
    tmp = np.zeros((TILE_SZ, TILE_SZ), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, markers[t1_color][0], markers[t1_color][1], 2, connectivity=4)
    flood_fill(tmp, markers[t2_color][0], markers[t2_color][1], 3, connectivity=4)
    ff_t1 = {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}
    ff_t2 = {(int(r), int(c)) for r, c in np.argwhere(tmp == 3)}
    assert ff_t1 == t1_region and ff_t2 == t2_region, "marker flood-fill must match wall partition"

    out3 = _composite_tile(t1, t2, bg, wall, t1_region, t2_region)
    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    wall_pool = _LBEND_WALLS if rng.random() < LBEND_PROB else _STRAIGHT_WALLS
    wall = rng.choice(wall_pool)

    lab, regions, candidates = _marker_candidates_by_region(wall)
    labels = sorted(regions)
    if len(labels) != 2:
        return None
    if any(not candidates[k] for k in labels):
        return None

    pos_a = rng.choice(candidates[labels[0]])
    pos_b = rng.choice(candidates[labels[1]])
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

    t1_label = int(lab[markers[t1_color][0], markers[t1_color][1]])
    t2_label = int(lab[markers[t2_color][0], markers[t2_color][1]])
    if t1_label == 0 or t2_label == 0 or t1_label == t2_label:
        return None

    t1_region = regions[t1_label]
    t2_region = regions[t2_label]
    out3 = _composite_tile(t1, t2, tile_bg, wall, t1_region, t2_region)

    swapped = _composite_tile(t1, t2, tile_bg, wall, t2_region, t1_region)
    if int(np.count_nonzero(out3 != swapped)) < MIN_SWAP_DIFF:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
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
