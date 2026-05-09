# Combo variant of 7491f3cf (index 22)
# combo_spec: D1/0+D2/0+D7/2
# Name: 十字分割+颜色匹配+多指示像素
#
# Design analysis:
#   a) 具体化与行为追踪
#      - 原题 tile0 是“5 像素墙 + 1 个孤立 indicator”，墙把 5x5 切成两个区域，indicator 所在侧固定取 tile1。
#      - 本融合版把 tile0 改成“9 像素十字墙 + 4 个角 marker”：
#        十字墙固定为中心行 row=2 与中心列 col=2 的并集，把 5x5 分成四个 2x2 象限。
#        四个 marker 固定放在四个角，各自落在对应象限内，且颜色直接复用 tile1/tile2 的前景色。
#      - 输出 tile3 时，不再存在全局 inside/outside。四个象限分别独立判断：
#        marker 颜色 == tile1 前景色 -> 该象限从 tile1 取非背景像素；
#        marker 颜色 == tile2 前景色 -> 该象限从 tile2 取非背景像素。
#        墙本身不属于任何象限，因此保持背景色，避免再引入一套难以观察的 wall 混合规则。
#      - 具体追踪示例：若左上/右下 marker 颜色匹配 tile1，右上/左下匹配 tile2，
#        则输出会呈 checkerboard 式四象限拼接；这与原题的“整片 inside 用同一源、outside 用另一源”有本质区别。
#
#   b) 复杂度校准
#      - 颜色种类仍限制为 5 类：border、tile_bg、wall_color、tile1_color、tile2_color。
#      - marker 只使用 tile1/tile2 两种前景色，不引入额外颜色语义；这样“颜色匹配”线索强而稳定。
#      - 虽然融合说明允许 2-4 个 marker，但这里固定使用 4 个，恰好一象限一个。
#        这是有意收束自由度：如果少于 4 个 marker，则未标记象限需要额外补规则，样例里会变得欠定。
#      - generator 只采样“非纯横分/非纯竖分”的象限分配，确保十字分割真地参与推理，而不是退化成单条直线切分。
#
#   c) 隐式假设审查
#      - 原 main() 假设存在唯一 8-孤立 indicator；现在不成立，四个角 marker 都是合法提示。
#      - 原 main() 假设 wall = cue_set - {indicator}；现在不成立，必须按“角上 marker / 十字墙”拆分。
#      - 原 main() 假设 inside 固定对应 tile1；现在不成立，对应关系逐象限由 marker 颜色决定。
#      - 原 main() 假设 wall 像素也按 inside/outside 规则写出；现在不成立，因为四象限不存在唯一 inside-source。
#      - 新 main() 依赖的新前提：
#        1. tile1/tile2 各自恰有一种前景色，且两种前景色不同；
#        2. tile0 的墙恰好是完整十字墙；
#        3. 四个角各有一个 marker，且颜色都属于 {tile1_color, tile2_color}；
#        4. wall_color 与两种 source 颜色不同。
#        这些都由 generator 显式构造并在 oracle 校验中验证。
#
#   d) 放置合法性
#      - 十字墙位置固定，不存在越界或形状歧义。
#      - 四个 marker 放在四个角，天然与十字墙 8-分离，不会重叠或贴边混淆。
#      - source patterns 仍使用原题的 5x5 单色对称采样器，因此颜色与背景结构保持稳定。
#
#   e) 视觉可观察性
#      - 解题者能直接看到：tile0 中四个角 marker 的颜色分别对应 tile1/tile2 的图案颜色。
#      - 输出不是单一左右/上下拼接，而是四个局部象限分别选源；对比 2-4 组样例即可归纳“局部颜色匹配”规则。
#      - generator 轮换多种象限赋值，并过滤纯横分/纯竖分，避免样例误导成人只需要看某一条轴。
#
#   f) 可视化整洁度
#      - tile0 只包含一条清晰的十字墙和四个角点，没有中心区域叠加。
#      - 输出 tile3 的墙保持背景色，相当于保留清晰分隔线，四个象限拼接关系更容易肉眼辨认。
#
#   g) 变化的可推断性
#      - marker 颜色不是任意编码，而是直接等于 tile1/tile2 的前景色，因此颜色语义完全可见。
#      - 每个样例中两种 source 颜色都必须出现于 marker 分配中，且每个象限都必须对 tile3 产生可见贡献，
#        否则 generator 会丢弃该样本。
#
# New main() assumption audit:
#   - `_parse_cue_tile()` 假设十字墙完整存在、四角 marker 完整存在、其余非背景像素不存在；
#     generator 固定只生成这 13 个 cue 像素。
#   - `_composite_tile()` 假设每个象限对应唯一 marker；generator 固定四角各放一个 marker。
#   - 退化审查假设“非纯横分/非纯竖分”足以让十字逻辑显性化；
#     generator 进一步要求输出与 uniform / 横分 / 竖分 / 颜色全翻转基线有足够差异。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


TILE_SZ = 5
GRID_ROWS, GRID_COLS = 7, 25
NUM_COLORS = 10

T1_MIN_PX, T1_MAX_PX = 8, 12
T2_MIN_PX, T2_MAX_PX = 9, 12
ORBIT_INCLUDE_PROB = 0.55
PATTERN_RETRIES = 200
GEN_RETRIES = 300

MIN_BASELINE_DIFF = 3
MIN_INVERT_DIFF = 4

CORNER_ORDER = ((0, 0), (0, 4), (4, 0), (4, 4))
CORNER_TO_NAME = {
    (0, 0): "nw",
    (0, 4): "ne",
    (4, 0): "sw",
    (4, 4): "se",
}
QUADRANT_REGIONS = {
    "nw": frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    "ne": frozenset({(0, 3), (0, 4), (1, 3), (1, 4)}),
    "sw": frozenset({(3, 0), (3, 1), (4, 0), (4, 1)}),
    "se": frozenset({(3, 3), (3, 4), (4, 3), (4, 4)}),
}
CROSS_WALL = frozenset({(2, c) for c in range(TILE_SZ)} | {(r, 2) for r in range(TILE_SZ)})

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

_ALL_ASSIGNMENTS = []
for bits in range(16):
    flags = tuple(bool((bits >> i) & 1) for i in range(4))
    top_same = flags[0] == flags[1]
    bottom_same = flags[2] == flags[3]
    left_same = flags[0] == flags[2]
    right_same = flags[1] == flags[3]
    if len(set(flags)) == 1:
        continue
    if top_same and bottom_same:
        continue
    if left_same and right_same:
        continue
    _ALL_ASSIGNMENTS.append(flags)
VALID_ASSIGNMENTS = tuple(_ALL_ASSIGNMENTS)
_NEXT_ASSIGNMENT_IDX = 0


def _extract_tiles(g: np.ndarray) -> list[np.ndarray]:
    return [g[1:6, 1 + 6 * j : 1 + 6 * j + 5].copy() for j in range(4)]


def _single_fg_color(tile: np.ndarray, bg: int, *, name: str) -> int:
    fg = {int(v) for v in np.unique(tile) if int(v) != int(bg)}
    assert len(fg) == 1, f"{name} must use exactly one foreground color"
    return next(iter(fg))


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


def _parse_cue_tile(
    t0: np.ndarray,
    bg: int,
    t1_color: int,
    t2_color: int,
) -> tuple[int, dict[str, int]]:
    wall: set[tuple[int, int]] = set()
    markers: dict[str, int] = {}
    wall_color = None

    for r in range(TILE_SZ):
        for c in range(TILE_SZ):
            v = int(t0[r, c])
            if v == bg:
                continue
            pos = (r, c)
            if pos in CORNER_TO_NAME:
                assert v in {t1_color, t2_color}, "corner marker must match tile1/tile2 color"
                quadrant = CORNER_TO_NAME[pos]
                assert quadrant not in markers, "multiple markers for one quadrant"
                markers[quadrant] = v
            else:
                if wall_color is None:
                    wall_color = v
                assert v == wall_color, "wall must be monochrome"
                wall.add(pos)

    assert wall == set(CROSS_WALL), "cue wall must be the full center-row/center-col cross"
    assert wall_color is not None, "missing wall color"
    assert wall_color not in {t1_color, t2_color}, "wall color must differ from source colors"
    assert set(markers) == set(QUADRANT_REGIONS), "all four quadrants must have markers"
    return wall_color, markers


def _composite_tile(
    t1: np.ndarray,
    t2: np.ndarray,
    bg: int,
    quadrant_uses_t1: dict[str, bool],
) -> np.ndarray:
    out3 = np.full((TILE_SZ, TILE_SZ), bg, dtype=int)
    for quadrant, region in QUADRANT_REGIONS.items():
        src = t1 if quadrant_uses_t1[quadrant] else t2
        for r, c in region:
            v = int(src[r, c])
            if v != bg:
                out3[r, c] = v
    return out3


def _assignment_from_markers(
    markers: dict[str, int],
    t1_color: int,
    t2_color: int,
) -> dict[str, bool]:
    quadrant_uses_t1: dict[str, bool] = {}
    for quadrant, color in markers.items():
        assert color in {t1_color, t2_color}, "marker color must match tile1 or tile2"
        quadrant_uses_t1[quadrant] = color == t1_color
    assert any(quadrant_uses_t1.values()), "at least one quadrant must use tile1"
    assert not all(quadrant_uses_t1.values()), "at least one quadrant must use tile2"
    return quadrant_uses_t1


def _flags_to_assignment(flags: tuple[bool, bool, bool, bool]) -> dict[str, bool]:
    return {
        "nw": flags[0],
        "ne": flags[1],
        "sw": flags[2],
        "se": flags[3],
    }


def _assignment_to_flags(quadrant_uses_t1: dict[str, bool]) -> tuple[bool, bool, bool, bool]:
    return (
        bool(quadrant_uses_t1["nw"]),
        bool(quadrant_uses_t1["ne"]),
        bool(quadrant_uses_t1["sw"]),
        bool(quadrant_uses_t1["se"]),
    )


def _quadrant_is_informative(t1: np.ndarray, t2: np.ndarray, bg: int, quadrant: str) -> bool:
    return any(
        int(t1[r, c]) != bg or int(t2[r, c]) != bg
        for r, c in QUADRANT_REGIONS[quadrant]
    )


def _baseline_outputs(t1: np.ndarray, t2: np.ndarray, bg: int) -> list[np.ndarray]:
    baselines = [
        _composite_tile(t1, t2, bg, _flags_to_assignment((True, True, True, True))),
        _composite_tile(t1, t2, bg, _flags_to_assignment((False, False, False, False))),
        _composite_tile(t1, t2, bg, _flags_to_assignment((True, True, False, False))),
        _composite_tile(t1, t2, bg, _flags_to_assignment((False, False, True, True))),
        _composite_tile(t1, t2, bg, _flags_to_assignment((True, False, True, False))),
        _composite_tile(t1, t2, bg, _flags_to_assignment((False, True, False, True))),
    ]
    return baselines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite tile1/tile2 into tile3 using a cross wall and four color-coded quadrant markers.

    Input structure:
    - Four 5x5 panels in one row, border-separated.
    - tile0 contains:
      * a 9-pixel cross wall (center row + center column) in one cue color
      * four corner markers, one per quadrant
      * each marker color equals either tile1's foreground color or tile2's foreground color
    - tile1 and tile2 are single-color symmetric source patterns.
    - tile3 is blank.

    Rule:
    - Treat the cross wall as splitting the 5x5 tile into four 2x2 quadrants.
    - For each quadrant independently, compare its corner marker color to tile1/tile2.
    - If the marker matches tile1, copy tile1's non-bg pixels in that quadrant.
    - If the marker matches tile2, copy tile2's non-bg pixels in that quadrant.
    - Wall pixels stay background.

    Key insight:
    - The output is a four-way local mosaic: marker colors do not choose one global "inside" side,
      they choose the source panel separately for each quadrant.
    """
    g = np.asarray(grid, dtype=int)
    border = int(g[0, 0])
    t0, t1, t2, _t3 = _extract_tiles(g)

    bg = get_bg(t1)
    assert border != bg, "border color must differ from tile background"
    assert get_bg(t2) == bg, "tile1 and tile2 must share a background"
    assert get_bg(t0) == bg, "cue tile must share the source background"

    t1_color = _single_fg_color(t1, bg, name="tile1")
    t2_color = _single_fg_color(t2, bg, name="tile2")
    assert t1_color != t2_color, "tile1 and tile2 foreground colors must differ"

    _wall_color, markers = _parse_cue_tile(t0, bg, t1_color, t2_color)
    quadrant_uses_t1 = _assignment_from_markers(markers, t1_color, t2_color)

    out3 = _composite_tile(t1, t2, bg, quadrant_uses_t1)

    out = g.copy()
    out[1:6, 1 + 6 * 3 : 1 + 6 * 3 + 5] = out3
    return out


def _gen_attempt(rng: random.Random, *, assignment_flags: tuple[bool, bool, bool, bool]) -> np.ndarray | None:
    palette = list(range(NUM_COLORS))
    tile_bg = rng.choice(palette)
    other = [c for c in palette if c != tile_bg]
    wall_color, t1_color, t2_color, border = rng.sample(other, 4)

    t1 = _sample_symmetric_pattern(tile_bg, t1_color, rng, min_px=T1_MIN_PX, max_px=T1_MAX_PX)
    t2 = _sample_symmetric_pattern(tile_bg, t2_color, rng, min_px=T2_MIN_PX, max_px=T2_MAX_PX)
    if t1 is None or t2 is None:
        return None

    quadrant_uses_t1 = _flags_to_assignment(assignment_flags)
    if not all(_quadrant_is_informative(t1, t2, tile_bg, quadrant) for quadrant in QUADRANT_REGIONS):
        return None

    out3 = _composite_tile(t1, t2, tile_bg, quadrant_uses_t1)

    min_baseline_diff = min(
        int(np.count_nonzero(out3 != baseline))
        for baseline in _baseline_outputs(t1, t2, tile_bg)
    )
    if min_baseline_diff < MIN_BASELINE_DIFF:
        return None

    inverted = _composite_tile(
        t1,
        t2,
        tile_bg,
        {quadrant: not use_t1 for quadrant, use_t1 in quadrant_uses_t1.items()},
    )
    if int(np.count_nonzero(out3 != inverted)) < MIN_INVERT_DIFF:
        return None

    t0 = np.full((TILE_SZ, TILE_SZ), tile_bg, dtype=int)
    for r, c in CROSS_WALL:
        t0[r, c] = wall_color
    for pos in CORNER_ORDER:
        quadrant = CORNER_TO_NAME[pos]
        t0[pos[0], pos[1]] = t1_color if quadrant_uses_t1[quadrant] else t2_color

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
    global _NEXT_ASSIGNMENT_IDX

    assignment_flags = VALID_ASSIGNMENTS[_NEXT_ASSIGNMENT_IDX % len(VALID_ASSIGNMENTS)]
    _NEXT_ASSIGNMENT_IDX += 1

    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng, assignment_flags=assignment_flags)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
