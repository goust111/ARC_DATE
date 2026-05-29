# Variant of 7491f3cf
# Dimension: D8 — 面板布局与网格结构 (grid_shape)
# Change: 4个5x5面板在同一行排列，1像素宽的边框分隔，总网格7x25... → 可变面板尺寸

from __future__ import annotations

import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT / "solvers"))

from arc_utils import flood_fill, label_mask, neighbors8


# -----------------------------------------------------------------------------
# Design analysis for the D8 variant
#
# variant_spec:
#   core_change:
#     original: |
#       输入恒为 7x25：四个 5x5 面板横向排布，面板间/外圈都有 1 像素边框。
#       solver 直接用固定切片 [1:6] 和步长 6 提取面板。
#     variant: |
#       输入仍是“四面板横排 + 1 像素边框”的同一视觉语法，但单个面板边长变成可变奇数 n。
#       本实现的 generator 取 n ∈ {5,7,9}，solver 对任意奇数 n>=3 的布局都能解析。
#       整体网格尺寸因此变为 (n+2) x (4n+5)。
#   concrete_visualization:
#     - 顶边、底边、5 条竖向分隔线全部是同一种 border 颜色。
#     - 四个内容面板共享同一个 tile_bg 背景色。
#     - panel0 仍包含“墙线 + 指示像素”；panel1 和 panel2 仍是两个源图案；panel3 输入仍为空白。
#     - 变化只体现在：5x5 被任意奇数 n x n 取代，中心点从 (2,2) 变成 (n//2,n//2)。
#   behavior_trace:
#     original_example: |
#       n=5 时，solver 默认 panel0 = g[1:6,1:6]，panel3 = g[1:6,19:24]。
#       如果指示点在 (0,2)，墙是中间横线 row=2，则 flood-fill 得到上半区为 inside，
#       输出 panel3 在 inside 位置复制 panel1 非背景像素，在 outside 位置复制 panel2。
#     variant_example: |
#       n=7 时，同样的结构变成 panel0 = g[1:8,1:8]，panel3 = g[1:8,25:32]，
#       中心点改为 (3,3)，中间横线 wall 变成整条 row=3。
#       规则完全不变，只是 flood-fill、合法指示点、对称墙线、输出拼接全部按 7x7 执行。
#     key_difference:
#       - 原逻辑把尺寸和切片写死在 5、6、7x25。
#       - 新逻辑先由 grid.shape 反推 n，再统一在 n 上运行所有几何规则。
#   complexity_calibration:
#     - 若允许任意很多尺寸，训练样例会过散，解题者很难从 2-4 组样例归纳布局公式。
#     - 原题还要求“指示点位于角/边中点且与穿过中心的墙线 8-隔离”；这一组约束在 3x3 上不可同时满足，
#       因为中心像素必然紧邻任一合法指示点。
#     - 因此 generator 只使用 {5,7,9} 三档，既能明显看出“尺寸可变”，又不破坏原题核心机制。
#     - 图案仍保持双对角对称，墙线仍限制为少量家族（直线 / L 弯），避免自由度失控。
#   implicit_assumption_audit_main:
#     - 原 `tiles = [g[1:6, 1 + 6*j : 1 + 6*j + 5]]` 假设 n=5；变体中必须改为基于 n 的切片。
#     - 原 `valid_pos` 枚举 8 个固定坐标；变体中必须改为四角 + 四边中点的通式。
#     - 原 `(2,2) in wall`、`dr/dc` 反射、边界触碰检测都依赖固定中心；现统一改为 `mid=n//2`。
#     - 原 flood-fill 临时网格、inside/outside 枚举、输出面板赋值都写死为 5x5；现全部按 n 构造。
#     - 原输出回填 `out[1:6, 19:24]` 仅对 7x25 生效；现改为第 4 块面板的动态切片。
#   placement_legality:
#     - 面板切片由公式 `(1 : 1+n, 1 + j*(n+1) : 1 + j*(n+1) + n)` 给出，不会重叠且恰好落在边框之间。
#     - 墙线候选由程序生成并过滤：必须经过中心、与指示点 8-隔离、触边、能把网格分成 inside/outside。
#     - 图案像素数上限严格低于半数，保证 tile_bg 仍是多数色，solver 可稳健识别背景。
#   visual_observability:
#     - 仅靠看样例，解题者能立即看到总网格高度/宽度在变化，但四面板关系未变。
#     - n=5 / 7 / 9 的混合样例会迫使解题者先识别“单面板尺寸”再复用原合成规则。
#   cleanliness_and_color_inference:
#     - 颜色角色不变：border、tile_bg、cue、t1、t2 各自单一且彼此区分。
#     - 变体不引入新的颜色映射；差异只来自几何尺度，因此不会制造额外颜色推断负担。
#   generator_guarantees_for_new_solver:
#     - grid.shape 总满足 `(n+2, 4n+5)` 且 n 为奇数。
#     - panel0 恰有一个 8-孤立指示像素；其余 cue 像素构成合法 wall。
#     - panel1/tile2 的非背景像素数都低于半数，且 inside 与 outside 至少各能看到一个来源像素。
# -----------------------------------------------------------------------------


NUM_COLORS = 10
SIZE_CHOICES = (5, 7, 9)
PATTERN_RETRIES = 300
GEN_RETRIES = 300
ORBIT_INCLUDE_PROB = 0.55
LBEND_PROB = 0.2

PATTERN_BOUNDS = {
    5: {"t1": (8, 12), "t2": (9, 12)},
    7: {"t1": (14, 20), "t2": (16, 22)},
    9: {"t1": (22, 32), "t2": (24, 36)},
}


def _infer_tile_size(g: np.ndarray) -> int:
    h, w = map(int, g.shape)
    n = h - 2
    assert n >= 3 and n % 2 == 1, f"tile size must be odd >= 3, got n={n}"
    assert w == 4 * n + 5, f"grid width {w} incompatible with tile size {n}"
    return n


def _tile_slice(n: int, j: int) -> tuple[slice, slice]:
    return slice(1, 1 + n), slice(1 + (n + 1) * j, 1 + (n + 1) * j + n)


def _extract_tiles(g: np.ndarray) -> tuple[int, list[np.ndarray]]:
    n = _infer_tile_size(g)
    border = int(g[0, 0])
    assert np.all(g[0, :] == border) and np.all(g[-1, :] == border), "top/bottom border broken"
    for c in range(0, g.shape[1], n + 1):
        assert np.all(g[:, c] == border), f"separator column {c} broken"
    tiles = []
    for j in range(4):
        rs, cs = _tile_slice(n, j)
        tiles.append(g[rs, cs].copy())
    return n, tiles


def _valid_indicator_positions(n: int) -> set[tuple[int, int]]:
    last = n - 1
    mid = n // 2
    return {
        (0, 0),
        (0, last),
        (last, 0),
        (last, last),
        (0, mid),
        (mid, 0),
        (mid, last),
        (last, mid),
    }


def _reflection_for_indicator(n: int, indicator: tuple[int, int]):
    last = n - 1
    mid = n // 2
    ir, ic = indicator
    dr, dc = mid - ir, mid - ic
    if dr == 0:
        return lambda r, c: (last - r, c)
    if dc == 0:
        return lambda r, c: (r, last - c)
    if dr * dc > 0:
        return lambda r, c: (c, r)
    return lambda r, c: (last - c, last - r)


def _flood_inside(n: int, wall: frozenset[tuple[int, int]], indicator: tuple[int, int]) -> set[tuple[int, int]]:
    tmp = np.zeros((n, n), dtype=int)
    for r, c in wall:
        tmp[r, c] = 1
    flood_fill(tmp, indicator[0], indicator[1], 2, connectivity=4)
    return {(int(r), int(c)) for r, c in np.argwhere(tmp == 2)}


def _wall_is_valid(n: int, indicator: tuple[int, int], wall: frozenset[tuple[int, int]]) -> bool:
    mid = n // 2
    last = n - 1
    if indicator in wall or (mid, mid) not in wall:
        return False
    if indicator not in _valid_indicator_positions(n):
        return False
    if any((nr, nc) in wall for nr, nc in neighbors8(*indicator)):
        return False
    refl = _reflection_for_indicator(n, indicator)
    if wall != {refl(r, c) for r, c in wall}:
        return False
    touched = set()
    for r, c in wall:
        if r == 0:
            touched.add("top")
        if r == last:
            touched.add("bottom")
        if c == 0:
            touched.add("left")
        if c == last:
            touched.add("right")
    if len(touched) < 2:
        return False
    inside = _flood_inside(n, wall, indicator)
    outside = {(r, c) for r in range(n) for c in range(n)} - inside - wall
    return bool(inside) and bool(outside)


def _ray(n: int, start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    out = []
    r, c = start
    while 0 <= r < n and 0 <= c < n:
        out.append((r, c))
        r += dr
        c += dc
    return out


@lru_cache(maxsize=None)
def _wall_configs(n: int) -> tuple[tuple[tuple[int, int], frozenset[tuple[int, int]]], ...]:
    mid = n // 2
    last = n - 1

    row = frozenset((mid, c) for c in range(n))
    col = frozenset((r, mid) for r in range(n))
    diag = frozenset((i, i) for i in range(n))
    anti = frozenset((i, last - i) for i in range(n))

    recipes = [
        ((0, mid), row),
        ((last, mid), row),
        ((mid, 0), col),
        ((mid, last), col),
        ((0, 0), anti),
        ((last, last), anti),
        ((0, last), diag),
        ((last, 0), diag),
        ((0, 0), frozenset(_ray(n, (mid, mid), -1, 0) + _ray(n, (mid, mid), 0, -1))),
        ((0, 0), frozenset(_ray(n, (mid, mid), 1, 0) + _ray(n, (mid, mid), 0, 1))),
        ((last, last), frozenset(_ray(n, (mid, mid), -1, 0) + _ray(n, (mid, mid), 0, -1))),
        ((last, last), frozenset(_ray(n, (mid, mid), 1, 0) + _ray(n, (mid, mid), 0, 1))),
        ((0, last), frozenset(_ray(n, (mid, mid), -1, 0) + _ray(n, (mid, mid), 0, 1))),
        ((0, last), frozenset(_ray(n, (mid, mid), 1, 0) + _ray(n, (mid, mid), 0, -1))),
        ((last, 0), frozenset(_ray(n, (mid, mid), -1, 0) + _ray(n, (mid, mid), 0, 1))),
        ((last, 0), frozenset(_ray(n, (mid, mid), 1, 0) + _ray(n, (mid, mid), 0, -1))),
        ((last, mid), frozenset(_ray(n, (mid, mid), -1, -1) + _ray(n, (mid, mid), -1, 1))),
        ((0, mid), frozenset(_ray(n, (mid, mid), 1, -1) + _ray(n, (mid, mid), 1, 1))),
        ((mid, last), frozenset(_ray(n, (mid, mid), -1, -1) + _ray(n, (mid, mid), 1, -1))),
        ((mid, 0), frozenset(_ray(n, (mid, mid), -1, 1) + _ray(n, (mid, mid), 1, 1))),
    ]

    unique = []
    seen = set()
    for indicator, wall in recipes:
        key = (indicator, tuple(sorted(wall)))
        if key in seen or not _wall_is_valid(n, indicator, wall):
            continue
        seen.add(key)
        unique.append((indicator, wall))
    assert unique, f"no valid wall configs for n={n}"
    return tuple(unique)


def _is_straight_wall(n: int, wall: frozenset[tuple[int, int]]) -> bool:
    last = n - 1
    return (
        all(r == n // 2 for r, _ in wall)
        or all(c == n // 2 for _, c in wall)
        or all(r == c for r, c in wall)
        or all(r + c == last for r, c in wall)
    )


@lru_cache(maxsize=None)
def _orbits(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    last = n - 1
    seen: set[tuple[int, int]] = set()
    out = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = sorted({(r, c), (c, r), (last - c, last - r), (last - r, last - c)})
            orbit_t = tuple(orbit)
            seen.update(orbit_t)
            out.append(orbit_t)
    return tuple(out)


def _sample_symmetric_pattern(
    n: int,
    bg: int,
    fg: int,
    rng: random.Random,
    *,
    min_px: int,
    max_px: int,
) -> np.ndarray | None:
    area = n * n
    assert max_px < area - max_px, "background must stay majority"
    orbits = list(_orbits(n))
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(orbits)))
        rng.shuffle(order)
        p = np.full((n, n), bg, dtype=int)
        count = 0
        for idx in order:
            orbit = orbits[idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    p[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return p
    return None


def _compose_output_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    wall: frozenset[tuple[int, int]],
    inside: set[tuple[int, int]],
) -> np.ndarray:
    n = t1.shape[0]
    wall_mask = np.zeros((n, n), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    wall_4conn = n_comp == 1

    out = np.full((n, n), bg, dtype=int)
    for r in range(n):
        for c in range(n):
            v1, v2 = int(t1[r, c]), int(t2[r, c])
            if (r, c) in inside:
                if v1 != bg:
                    out[r, c] = v1
            elif (r, c) in wall:
                if v1 != bg:
                    out[r, c] = v1
                elif wall_4conn and v2 != bg:
                    out[r, c] = v2
            elif v2 != bg:
                out[r, c] = v2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two source tiles into the fourth tile.

    D8 variant parsing rule:
    - Still 4 panels in one row with 1-cell separators.
    - If tile size is n (odd), total grid shape is (n+2, 4n+5).
    """
    g = np.array(grid, dtype=int)
    n, tiles = _extract_tiles(g)
    t0, t1, t2, _t3 = tiles

    border = int(g[0, 0])
    bg = int(Counter(map(int, t1.flatten())).most_common(1)[0][0])
    assert border != bg, "border color must differ from tile background"

    t0_bg = int(Counter(map(int, t0.flatten())).most_common(1)[0][0])
    cue_set = {(int(r), int(c)) for r in range(n) for c in range(n) if int(t0[r, c]) != t0_bg}
    assert cue_set, "no cue pixels found"

    indicator = None
    for r, c in cue_set:
        if not any((nr, nc) in cue_set for nr, nc in neighbors8(r, c)):
            assert indicator is None, "multiple 8-isolated cue pixels"
            indicator = (r, c)
    assert indicator is not None, "no 8-isolated cue pixel found"
    assert indicator in _valid_indicator_positions(n), f"indicator {indicator} not at corner/edge-midpoint"

    wall = frozenset(cue_set - {indicator})
    assert _wall_is_valid(n, indicator, wall), "invalid wall geometry"

    inside = _flood_inside(n, wall, indicator)
    out3 = _compose_output_tile(t1, t2, bg, wall, inside)

    out = g.copy()
    rs, cs = _tile_slice(n, 3)
    out[rs, cs] = out3
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(SIZE_CHOICES)
    tile_bg = rng.randrange(NUM_COLORS)
    other = [c for c in range(NUM_COLORS) if c != tile_bg]
    cue_color, t1_color, t2_color = rng.sample(other, 3)
    border = rng.choice(other)

    configs = _wall_configs(n)
    straight = [(indicator, wall) for indicator, wall in configs if len(wall) == n and _is_straight_wall(n, wall)]
    lbends = [cfg for cfg in configs if cfg not in straight]
    pool = lbends if lbends and rng.random() < LBEND_PROB else straight or configs
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(n, wall, indicator)
    outside = {(r, c) for r in range(n) for c in range(n)} - inside - wall

    bounds = PATTERN_BOUNDS[n]
    t1 = _sample_symmetric_pattern(n, tile_bg, t1_color, rng, min_px=bounds["t1"][0], max_px=bounds["t1"][1])
    t2 = _sample_symmetric_pattern(n, tile_bg, t2_color, rng, min_px=bounds["t2"][0], max_px=bounds["t2"][1])
    if t1 is None or t2 is None:
        return None

    if not any(int(t1[r, c]) != tile_bg for r, c in inside):
        return None
    if not any(int(t2[r, c]) != tile_bg for r, c in outside):
        return None

    t0 = np.full((n, n), tile_bg, dtype=int)
    for r, c in wall:
        t0[r, c] = cue_color
    t0[indicator[0], indicator[1]] = cue_color

    out3 = _compose_output_tile(t1, t2, tile_bg, wall, inside)
    if np.array_equal(out3, np.full((n, n), tile_bg, dtype=int)):
        return None

    g = np.full((n + 2, 4 * n + 5), border, dtype=int)
    tiles = [t0, t1, t2, np.full((n, n), tile_bg, dtype=int)]
    for j, tile in enumerate(tiles):
        rs, cs = _tile_slice(n, j)
        g[rs, cs] = tile

    expected = g.copy()
    rs, cs = _tile_slice(n, 3)
    expected[rs, cs] = out3
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
