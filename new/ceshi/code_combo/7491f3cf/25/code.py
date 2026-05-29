# Combo variant of 7491f3cf (index 25)
# combo_spec: D8/1+D15/2+D16/0
# Name: 可变面板尺寸+可变宽度边框+面板顺序可变

from __future__ import annotations

import random
import sys
from functools import lru_cache

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import flood_fill, get_bg, label_mask, neighbors8


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    - 原题的视觉语法是：4 个正方形面板横向排布，外圈和分隔线是统一边框色；其中
#      一个面板是 cue（wall + isolated indicator），两个面板是 source，另一个面板为空白输出位。
#    - 融合后这套视觉语法保留，但三项元信息不再固定：
#      1. 面板边长 n 不固定；
#      2. 边框宽度 b 不固定；
#      3. cue / source / blank 的顺序不固定。
#    - solver 的新执行链因此变成：
#      1. 先由整网格尺寸反推 (n, b)；
#      2. 按 (n, b) 切出四个面板；
#      3. 找到唯一 blank 面板以恢复共享背景色；
#      4. 找到唯一 cue 面板；
#      5. 在剩余两个 source 中，按与 cue 的线性距离分配：
#         距离更近的是 tile1（inside 来源），更远的是 tile2（outside 来源）；
#      6. 再执行原题的 flood-fill 合成规则。
#    - 具体例子：若输入总尺寸是 11x38，则必须满足
#      H = n + 2b, W = 4n + 5b，因此可解得 n=7, b=2。
#      此时四个面板切片依次是：
#      rows [2:9], cols [2:9] / [11:18] / [20:27] / [29:36]。
#      若 blank 位于索引 0、cue 位于索引 2，则剩余 source 位于 1 和 3；
#      因为 |1-2| = 1 < |3-2| = 1 不成立，所以这种布局会被 generator 排除。
#      generator 只保留 source 到 cue 距离不同的布局，例如 blank=0, cue=1, source=2/3，
#      这时索引 2 是 tile1，索引 3 是 tile2。
#    - 与原题相比，真正变化的是“先解析容器，再应用规则”；合成规则本身不变。
#
# b) Complexity calibration
#    - 面板尺寸若机械照搬“3/5/7”会与原题的 cue 几何约束冲突：
#      原题继承下来的 indicator 必须位于角/边中点，且与穿过中心的 wall 8-隔离。
#      在 3x3 中，任何合法 indicator 位置都与中心 8 邻接，因此不可能同时满足。
#    - 为了保持视觉规则一致而不是让 solver 对 3x3 用另一套 cue 语义，
#      本融合版将 generator 的尺寸集合收紧为 {5, 7}。solver 对任意 odd n>=5 都能解析。
#    - 边框宽度只取 {1,2,3}，角色顺序只通过 cue 相对距离决定，不引入隐藏 tie-break。
#    - source 图案继续保持双对角对称，wall 继续限制在直线/L 弯少数家族，避免视觉过乱。
#
# c) Implicit-assumption audit
#    - 原 `g[1:6, 1+6*j : 1+6*j+5]` 假设 n=5,b=1；现在必须先由整网格解出 (n,b) 后切片。
#    - 原 `t0,t1,t2,t3 = tiles` 假设角色顺序固定；现在必须先识别 blank 与 cue。
#    - 原 `b = most_common(t1)` 假设 panel1 是 source；现在背景色必须从 blank 面板恢复，
#      再验证其他面板的多数色一致。
#    - 原 `valid_pos`、`(2,2) in wall`、反射轴判断、flood-fill 网格大小都写死在 5x5；
#      现在全部改成 n 的通式，中心改为 `(n//2, n//2)`。
#    - 原输出固定写回 panel3；现在改为写回唯一 blank 面板。
#    - 新增隐式假设是：总尺寸必须满足 `H=n+2b, W=4n+5b`。generator 用固定布局公式保证这一点。
#
# d) Placement legality
#    - 4 个面板始终用 `_tile_slice(n, b, j)` 放置，不重叠、不越界，剩余区域全部填 border。
#    - wall 候选由程序化配方生成并过滤：必须穿过中心、8 连通、与 indicator 8-隔离、
#      触达至少两侧边界，并且确实把面板切成非空 inside/outside。
#    - 图案像素数始终小于半数，保证 tile_bg 在各面板中仍是可见且可推断的多数色。
#
# e) Visual observability
#    - 解题者从 2-4 组样例中可以直接看到：网格总高宽在变，说明 panel size 和 border width 在变。
#    - blank 面板是唯一纯色面板，cue 面板是唯一“单色 wall+indicator”面板，顺序变化可直接观察。
#    - 输出仍由 cue 分出的 inside/outside 决定，因此人类可以先看懂原题主规则，再补上前置解析步骤。
#
# f) Clean visual layout
#    - 没有新增编码层。边框、空白、cue、两个 source 仍是四种清楚的区域角色。
#    - border、bg、cue、tile1、tile2 强制为 5 个不同颜色，避免视觉歧义。
#
# g) Inferability of structure and color
#    - 颜色不参与任何排序或隐藏判断，只用于可视区分。
#    - 核心可推断变量全部是肉眼可见的结构量：网格尺寸、边框厚度、cue 位置、blank 位置、
#      以及 source 相对 cue 的左右距离。
#
# New main() assumptions and generator guarantees
#    - Assumption: 输入是 4 面板横排，且整体尺寸满足 H=n+2b, W=4n+5b。
#      Generator guarantee: 始终按该公式组装。
#    - Assumption: 恰有一个 blank 面板，且所有面板共享同一背景色。
#      Generator guarantee: 是。
#    - Assumption: 恰有一个 cue 面板满足原题几何约束。
#      Generator guarantee: 是，并拒绝会让 source 被误判为 cue 的样本。
#    - Assumption: 两个 source 到 cue 的距离不同。
#      Generator guarantee: 只采样无歧义布局。
# -----------------------------------------------------------------------------


NUM_TILES = 4
NUM_COLORS = 10
SIZE_CHOICES = (5, 7)
BORDER_CHOICES = (1, 2, 3)

PATTERN_RETRIES = 300
GEN_RETRIES = 400
ORBIT_INCLUDE_PROB = 0.55
LBEND_PROB = 0.2

PATTERN_BOUNDS = {
    5: {"t1": (8, 12), "t2": (9, 12)},
    7: {"t1": (14, 20), "t2": (16, 22)},
}


def _tile_slice(n: int, b: int, j: int) -> tuple[slice, slice]:
    return slice(b, b + n), slice(b + j * (n + b), b + j * (n + b) + n)


def _infer_layout(g: np.ndarray) -> tuple[int, int]:
    h, w = map(int, g.shape)
    b_num = 4 * h - w
    assert b_num > 0 and b_num % 3 == 0, f"grid shape {g.shape} is incompatible with 4 panels"
    b = b_num // 3
    n = h - 2 * b
    assert b in BORDER_CHOICES, f"border width must be 1..3, got {b}"
    assert n >= 5 and n % 2 == 1, f"tile size must be odd and >=5, got {n}"
    assert w == 4 * n + 5 * b, "layout algebra mismatch"

    border = int(g[0, 0])
    panel_mask = np.zeros(g.shape, dtype=bool)
    for j in range(NUM_TILES):
        rs, cs = _tile_slice(n, b, j)
        panel_mask[rs, cs] = True
    assert np.all(g[~panel_mask] == border), "all non-panel cells must be border-colored"
    return n, b


def _extract_tiles(g: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    n, b = _infer_layout(g)
    return n, b, [g[_tile_slice(n, b, j)].copy() for j in range(NUM_TILES)]


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


def _wall_is_8_connected(n: int, wall: frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((n, n), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=8)
    return n_comp == 1


def _wall_is_4_connected(n: int, wall: frozenset[tuple[int, int]]) -> bool:
    wall_mask = np.zeros((n, n), dtype=bool)
    for r, c in wall:
        wall_mask[r, c] = True
    _, n_comp = label_mask(wall_mask, connectivity=4)
    return n_comp == 1


def _wall_is_valid(n: int, indicator: tuple[int, int], wall: frozenset[tuple[int, int]]) -> bool:
    mid = n // 2
    last = n - 1
    if indicator in wall or len(wall) != n or (mid, mid) not in wall:
        return False
    if indicator not in _valid_indicator_positions(n):
        return False
    if any((nr, nc) in wall for nr, nc in neighbors8(*indicator)):
        return False
    if not _wall_is_8_connected(n, wall):
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
    mid = n // 2
    return (
        all(r == mid for r, _ in wall)
        or all(c == mid for _, c in wall)
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
            orbit = tuple(sorted({(r, c), (c, r), (last - c, last - r), (last - r, last - c)}))
            seen.update(orbit)
            out.append(orbit)
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
    assert max_px < area - max_px, "background must remain the majority color"
    orbits = list(_orbits(n))
    for _ in range(PATTERN_RETRIES):
        order = list(range(len(orbits)))
        rng.shuffle(order)
        pattern = np.full((n, n), bg, dtype=int)
        count = 0
        for idx in order:
            orbit = orbits[idx]
            if count + len(orbit) > max_px:
                continue
            if rng.random() < ORBIT_INCLUDE_PROB:
                for r, c in orbit:
                    pattern[r, c] = fg
                count += len(orbit)
        if min_px <= count <= max_px:
            return pattern
    return None


def _compose_output_tile(
    tile_bg: int,
    tile1: np.ndarray,
    tile2: np.ndarray,
    inside: set[tuple[int, int]],
    wall: frozenset[tuple[int, int]],
) -> np.ndarray:
    n = int(tile1.shape[0])
    wall_4conn = _wall_is_4_connected(n, wall)
    out = np.full((n, n), tile_bg, dtype=int)
    for r in range(n):
        for c in range(n):
            v1, v2 = int(tile1[r, c]), int(tile2[r, c])
            if (r, c) in inside:
                if v1 != tile_bg:
                    out[r, c] = v1
            elif (r, c) in wall:
                if v1 != tile_bg:
                    out[r, c] = v1
                elif wall_4conn and v2 != tile_bg:
                    out[r, c] = v2
            elif v2 != tile_bg:
                out[r, c] = v2
    return out


def _parse_cue(tile: np.ndarray, bg: int) -> tuple[tuple[int, int], frozenset[tuple[int, int]]] | None:
    n = int(tile.shape[0])
    if tile.shape != (n, n) or get_bg(tile) != bg:
        return None

    cue_pixels = {(int(r), int(c)) for r in range(n) for c in range(n) if int(tile[r, c]) != bg}
    if len(cue_pixels) != n + 1:
        return None
    if len({int(tile[r, c]) for r, c in cue_pixels}) != 1:
        return None

    isolated = [
        (r, c)
        for r, c in cue_pixels
        if (r, c) in _valid_indicator_positions(n)
        and not any((nr, nc) in cue_pixels for nr, nc in neighbors8(r, c))
    ]
    if len(isolated) != 1:
        return None

    indicator = isolated[0]
    wall = frozenset(cue_pixels - {indicator})
    if not _wall_is_valid(n, indicator, wall):
        return None
    return indicator, wall


def _find_blank_tile(tiles: list[np.ndarray]) -> tuple[int, int]:
    candidates = [idx for idx, tile in enumerate(tiles) if np.unique(tile).size == 1]
    assert len(candidates) == 1, "expected exactly one blank panel"
    blank_idx = candidates[0]
    return blank_idx, int(tiles[blank_idx][0, 0])


def _find_cue_tile(
    tiles: list[np.ndarray],
    bg: int,
    blank_idx: int,
) -> tuple[int, tuple[int, int], frozenset[tuple[int, int]]]:
    matches: list[tuple[int, tuple[int, int], frozenset[tuple[int, int]]]] = []
    for idx, tile in enumerate(tiles):
        if idx == blank_idx:
            continue
        parsed = _parse_cue(tile, bg)
        if parsed is not None:
            indicator, wall = parsed
            matches.append((idx, indicator, wall))
    assert len(matches) == 1, "expected exactly one cue panel"
    return matches[0]


def _source_role_indices(cue_idx: int, blank_idx: int) -> tuple[int, int]:
    source_indices = [idx for idx in range(NUM_TILES) if idx not in (cue_idx, blank_idx)]
    ranked = sorted((abs(idx - cue_idx), idx) for idx in source_indices)
    assert len(ranked) == 2, "expected exactly two source panels"
    assert ranked[0][0] != ranked[1][0], "ambiguous source-role distances"
    return ranked[0][1], ranked[1][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite two source panels into the blank panel after jointly inferring geometry and roles.

    Input structure:
    - Four square panels appear in one row inside a uniform border-colored frame.
    - Panel size is variable and odd; generator uses 5x5 or 7x7.
    - Border width is variable; generator uses widths 1, 2, or 3.
    - Exactly one panel is blank, exactly one panel is the cue, and the other two are source patterns.
    - The cue panel contains one wall plus one 8-isolated indicator, all in a single cue color.

    Rule:
    - Recover `(panel_size, border_width)` from the whole-grid shape, then slice out the four panels.
    - Use the unique blank panel to identify the shared tile background color.
    - Use cue geometry to find the unique cue panel. Among the two remaining panels, the one closer to the cue index is tile1 and the farther one is tile2.
    - Flood-fill from the indicator with 4-connectivity, blocked by the wall.
    - Inside cells copy tile1 non-background pixels; outside cells copy tile2; wall cells prefer tile1 and use tile2 only as 4-connected-wall fallback.

    Key insight:
    - The painting rule is still the original task's inside/outside composite.
    - What changed is the prerequisite parsing: the solver must infer scale, separator thickness, and panel roles before it can apply that rule.
    """
    g = np.array(grid, dtype=int)
    n, b, tiles = _extract_tiles(g)
    border = int(g[0, 0])

    blank_idx, tile_bg = _find_blank_tile(tiles)
    assert border != tile_bg, "border color must differ from tile background"
    for tile in tiles:
        assert get_bg(tile) == tile_bg, "all panels must share one background color"

    cue_idx, indicator, wall = _find_cue_tile(tiles, tile_bg, blank_idx)
    tile1_idx, tile2_idx = _source_role_indices(cue_idx, blank_idx)
    tile1, tile2 = tiles[tile1_idx], tiles[tile2_idx]

    inside = _flood_inside(n, wall, indicator)
    outside = {(r, c) for r in range(n) for c in range(n)} - inside - wall
    assert outside, "wall must partition the tile"

    out_tile = _compose_output_tile(tile_bg, tile1, tile2, inside, wall)
    out = g.copy()
    out[_tile_slice(n, b, blank_idx)] = out_tile
    return out


def _valid_blank_positions(cue_idx: int) -> list[int]:
    valid = []
    for blank_idx in range(NUM_TILES):
        if blank_idx == cue_idx:
            continue
        try:
            _source_role_indices(cue_idx, blank_idx)
        except AssertionError:
            continue
        valid.append(blank_idx)
    return valid


def _choose_layout(rng: random.Random) -> tuple[int, int, int, int]:
    cue_idx = rng.randrange(NUM_TILES)
    blank_idx = rng.choice(_valid_blank_positions(cue_idx))
    tile1_idx, tile2_idx = _source_role_indices(cue_idx, blank_idx)
    return cue_idx, blank_idx, tile1_idx, tile2_idx


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(SIZE_CHOICES)
    b = rng.choice(BORDER_CHOICES)
    tile_bg, border, cue_color, t1_color, t2_color = rng.sample(range(NUM_COLORS), 5)

    configs = _wall_configs(n)
    straight = [(indicator, wall) for indicator, wall in configs if _is_straight_wall(n, wall)]
    bends = [cfg for cfg in configs if cfg not in straight]
    pool = bends if bends and rng.random() < LBEND_PROB else straight or configs
    indicator, wall = rng.choice(pool)
    inside = _flood_inside(n, wall, indicator)
    outside = {(r, c) for r in range(n) for c in range(n)} - inside - wall

    bounds = PATTERN_BOUNDS[n]
    tile1 = _sample_symmetric_pattern(n, tile_bg, t1_color, rng, min_px=bounds["t1"][0], max_px=bounds["t1"][1])
    tile2 = _sample_symmetric_pattern(n, tile_bg, t2_color, rng, min_px=bounds["t2"][0], max_px=bounds["t2"][1])
    if tile1 is None or tile2 is None:
        return None
    if _parse_cue(tile1, tile_bg) is not None or _parse_cue(tile2, tile_bg) is not None:
        return None

    if not any(int(tile1[r, c]) != tile_bg for r, c in inside):
        return None
    if not any(int(tile2[r, c]) != tile_bg for r, c in outside):
        return None

    cue_tile = np.full((n, n), tile_bg, dtype=int)
    for r, c in wall:
        cue_tile[r, c] = cue_color
    cue_tile[indicator[0], indicator[1]] = cue_color

    blank_tile = np.full((n, n), tile_bg, dtype=int)
    out_tile = _compose_output_tile(tile_bg, tile1, tile2, inside, wall)
    if np.array_equal(out_tile, blank_tile):
        return None

    cue_idx, blank_idx, tile1_idx, tile2_idx = _choose_layout(rng)
    tiles: list[np.ndarray | None] = [None] * NUM_TILES
    tiles[cue_idx] = cue_tile
    tiles[blank_idx] = blank_tile
    tiles[tile1_idx] = tile1
    tiles[tile2_idx] = tile2

    g = np.full((n + 2 * b, 4 * n + 5 * b), border, dtype=int)
    for idx, tile in enumerate(tiles):
        assert tile is not None
        g[_tile_slice(n, b, idx)] = tile

    expected = g.copy()
    expected[_tile_slice(n, b, blank_idx)] = out_tile
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
