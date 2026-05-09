# Variant of 9aaea919
# Dimension: D5 — 标记条形态与位置 (symbol_instructions)
# Change: 1x5水平条状标记，位于第29行(最底行)，红(2)标记源车道，绿(3)标记目标车道... → 三色标记系统
#
# Design analysis
# analysis:
#   concrete_form:
#     - 仍然使用 30x30 画布、4 条固定纵向车道、3x5 十字精灵、最底行 1x5 横条标记。
#     - 本变体把“未标记车道”改为“黄色(4)冻结车道”，因此四条车道在输入里全部有标记：
#       1-2 条红条=源车道，1 条绿条=目标车道，其余 1-2 条黄条=冻结车道。
#     - 输出中：红车道精灵变灰(5)；绿车道自底向上补入与所有红车道总数相同的新精灵；
#       黄车道精灵完全不动；三种标记条全部消失。
#     - 具体追踪例子：若 lane0 有 2 个红源精灵，lane2 有 1 个绿目标精灵，lane1/lane3 为黄冻结，
#       则原题只会处理 lane0→lane2，lane1/lane3 只是“没有标记”；本变体里 lane1/lane3 明确带黄条，
#       solver 必须识别它们是“显式不参与”的角色，并在输出中删掉黄条。
#   complexity_calibration:
#     - 只增加一个新角色“冻结”，不引入新的几何操作；核心难度来自三分类，而非额外形变。
#     - 为避免黄色角色沦为摆设，生成器强制每个样例至少有 1 条黄车道，且不再保留无标记车道。
#     - 标记仍然是统一的 1x5 底部横条，视觉上能直接比较三种颜色的语义差异。
#   implicit_assumptions_review:
#     - 原版 `red_markers = _find_marker_bars(g, 2)` 与 `green_markers = _find_marker_bars(g, 3)` 默认
#       “只有红/绿角色存在”；现在必须额外查找黄色条，否则黄色条会残留在输出中。
#     - 原版结尾只擦除红/绿条；在新规则下这会错误保留黄色条，因此必须统一擦除三类标记。
#     - 原版把所有非红非绿车道都视作无关背景；变体后这些车道虽然逻辑不变，但它们有显式符号，
#       所以生成器要保证黄色条与精灵分离、solver 要保证不把黄条当成待移动精灵。
#     - 目标车道与源车道的抽取逻辑仍依赖“同一车道内精灵大小一致、标记与精灵不接触、目标车道原本就有精灵”，
#       这些前提需要继续由生成器保证。
#   placement_legality:
#     - 纵向周期、底部 top 行、最大总堆叠数维持原值，保证新增精灵仍能落在合法空槽里。
#     - 黄色条仍放在第 29 行，底部精灵最低只到第 27 行，中间保留 1 行空隙，不会与精灵连通。
#     - 背景色改为排除 {2,3,4,5}，并且精灵颜色排除 4 和 5，从而让黄色条始终是孤立标记而非车道内容。
#   visual_observability:
#     - 输入中四条车道全部带条，解题者可直接观察到红/绿/黄三类符号并对照输出差异：
#       红条对应“本车道变灰”，绿条对应“接收补入”，黄条对应“车道不变但标记消失”。
#     - 2-4 个样例里只需比较角色颜色与输出行为，不需要借助颜色编号排序或隐含计数技巧。
#   visual_cleanliness:
#     - 车道横向间隔、精灵形状、堆叠节奏都不变，新增信息只集中在底部标记行，不会制造重叠噪声。
#     - 所有标记同形同宽，仅颜色不同，读图时很整洁。
#   color_change_inference:
#     - 真正发生重着色的仍只有红源车道→灰、复制到绿目标车道→目标色；黄色不触发颜色映射。
#     - 黄色角色的结构性线索来自位置固定的底部横条，而不是依赖颜色 id 的数值意义。
#   solver_assumptions_after_change:
#     - `_find_marker_bars` 仍假设标记是细长 4 连通分量；生成器固定产出 1x5 横条满足此点。
#     - 绿车道至少有一个原始精灵，才能估计周期和目标颜色；生成器仍令每条车道输入精灵数为 1-3。
#     - 红车道转移后的总数必须能装入目标车道；生成器继续用 `_MAX_LANE_TOTAL` 约束。

import numpy as np

from itertools import combinations

from arc_utils import bbox_coords, components_mask, get_bg, paste


_SRC_MARKER = 2
_DST_MARKER = 3
_FROZEN_MARKER = 4
_GREY = 5


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """
    Marker bars are thin 4-connected components of the given marker color.
    """
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, -r1, hh * ww, len(pts), pts))
    cands.sort()
    return [t[4] for t in cands]


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    m = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        cols_in = {int(g[r, c]) for r, c in pts}
        if len(cols_in) == 1:
            out.append(pts)
    return out


def _extract_lane_sprites(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> tuple[float, list[list[tuple[int, int]]]]:
    marker_arr = np.array(marker_pts)
    _, lane_c0, _, lane_c1 = bbox_coords(marker_arr)
    lane_cols = range(lane_c0, lane_c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return (lane_c0 + lane_c1) / 2.0, []

    sprite_sizes = [len(p) for p in comps if len(p) != len(marker_pts)]
    if not sprite_sizes:
        return (lane_c0 + lane_c1) / 2.0, []

    sprite_sz = max(sprite_sizes)
    sprites = [p for p in comps if len(p) == sprite_sz]
    return (lane_c0 + lane_c1) / 2.0, sprites


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    items = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        items.append((rmin, cmin, pts))
    items.sort()
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a three-color lane legend: red bars donate sprites, the green bar receives them, and yellow bars freeze their lanes.

    Input structure:
    - The dominant color is the background. Each of the four fixed lanes has a 1x5 marker bar on the last row: red (2) for source, green (3) for destination, yellow (4) for frozen.
    - Every lane contains repeated monochrome plus sprites laid out on the same vertical lattice.

    Rule:
    - Sprites in each red-marked lane turn grey (5), and the same number of copies are inserted into the nearest green-marked lane from the bottom upward using that lane's own color.
    - Yellow-marked lanes keep their sprites unchanged; all three kinds of marker bars are erased.

    Key insight:
    - The geometry is unchanged from the base task; the only new reasoning step is to classify all lanes into three explicit roles and ignore yellow lanes during transfer while still removing their markers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, _SRC_MARKER)
    green_markers = _find_marker_bars(g, _DST_MARKER)
    yellow_markers = _find_marker_bars(g, _FROZEN_MARKER)
    assert red_markers and green_markers and yellow_markers, "expected red, green, and yellow marker bars"

    green_state = {}
    for gm in green_markers:
        gc, dst_sprites = _extract_lane_sprites(g, gm, bg)
        if not dst_sprites:
            continue

        dst_sprites_sorted = _sorted_sprites(dst_sprites)
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        green_state[id(gm)] = {
            "center": gc,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        rc, src_sprites = _extract_lane_sprites(g, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = _GREY

        best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = _sorted_sprites(src_sprites)
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]

        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
            new_top = best["bottom0"] - k * best["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in best["occupied"]:
                continue
            ok = True
            for dr, c in src_offsets:
                rr = new_top + dr
                if not (0 <= rr < h and 0 <= c < w):
                    ok = False
                    break
                if int(out[rr, c]) != bg:
                    ok = False
                    break
            if not ok:
                continue
            for dr, c in src_offsets:
                out[new_top + dr, c] = best["dst_color"]
            best["occupied"].add(new_top)
            placed += 1

    for markers in (red_markers, green_markers, yellow_markers):
        for marker in markers:
            for r, c in marker:
                out[r, c] = bg

    return out


# --------------- generator ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 6, 7, 8, 9]

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _FROZEN_MARKER, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    frozen_set = {i for i in range(_NUM_LANES) if i not in red_set and i != green}
    if not frozen_set:
        return None
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = _DST_MARKER

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = _SRC_MARKER

    for lane_idx in frozen_set:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = _FROZEN_MARKER

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
