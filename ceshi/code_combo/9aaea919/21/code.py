# Combo variant of 9aaea919 (index 21)
# combo_spec: D10/1+D16/1
# Name: 紧密排列+精灵数量扩大

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for combo D10/1 + D16/1
# a) 具体化与行为追踪:
#    - 原题里每个 3x5 十字精灵之间隔 1 行背景, 所以同一车道的多个精灵会被 4 连通分开;
#      solver 依赖这一点, 直接把每个连通块当成一个 sprite。
#    - 本融合把周期从 4 改到 3: 例如同车道有 3 个精灵时, 行模式会变成
#      `.XXX. / XXXXX / .XXX. / .XXX. / XXXXX / .XXX. / .XXX. / XXXXX / .XXX.`。
#      相邻两个精灵在两条 `.XXX.` 行上会垂直接触, 整列车道变成一个高组件, 不能再按连通块数 sprite。
#    - 同时 D16/1 把每车道输入数量扩大到 1-5。比如源车道有 5 个、目标车道有 4 个, 输出目标车道会有 9 个
#      紧密堆叠的十字, 顶行来到 row 1; 这在原题 period=4 的布局下物理上放不下, 所以两个维度是强耦合的。
#    - 因此新 solver 改为利用“固定车道宽 5 + 固定精灵面积 11 + 紧密周期 3”的结构去解析每条车道的数量,
#      然后直接重绘红车道为灰色紧密栈、重绘绿车道为扩容后的目标色紧密栈。
#
# b) 复杂度校准:
#    - 变化只有两个: 间隙消失、数量上限变大; 形状、车道位置、底部标记语义都保持不变, 不会引入额外自由度。
#    - 生成器强制样本里至少有一个输入车道 count >= 4, 且目标输出总数优先达到 8 或 9, 让 D16/1 在肉眼上稳定可见。
#    - 紧密排列会产生重复的 3/5/3 行宽节奏, 人类仍可凭重复图样辨认单个十字的边界, 但比原题更拥挤。
#
# c) 隐式假设审查:
#    - 原 `_single_color_components_in_cols(...)` 假设“一个连通块≈一个 sprite”; 本融合下失效, 必须弃用。
#    - 原目标放置循环假设存在 period-4 的空槽并逐槽试探; 本融合的生成器保证每条车道始终是底部对齐的紧密连续栈,
#      所以更稳妥的实现是按数量直接整列重绘, 不再依赖空槽搜索。
#    - 新 solver 的前提变成: 每条车道只含这一种 3x5 十字反复堆叠, 且 marker 仅在 row 29。生成器会显式保证这些条件。
#
# d) 放置合法性:
#    - period=3 时 30x30 画布最多容纳 9 个精灵: 顶行序列为 25,22,19,16,13,10,7,4,1。故目标总数上限设为 9。
#    - 车道仍固定在 4 个互不重叠的 5 列槽位里, 横向不会碰撞; 标记条仍在 row 29, 与底部精灵之间保留 row 28 背景隔离。
#    - 生成器采用 output-first: 先画合法终态, 再反推输入, 并用 fused solver 做 oracle 校验。
#
# e) 视觉可观察性:
#    - 输入里同色十字会直接粘连成一整条“编织状”图案, 明显区别于原题每层之间有空隙的样子。
#    - 当 count 超过 3, 车道上半部也会被占用, 解题者能从 2-4 组样例中看出“数量范围扩大”与“因此必须紧密堆叠”。
#    - 规则仍由可见结构提示: 红条=来源、绿条=目标、红车道变灰、目标车道得到同数量补充。
#
# f) 可视化整洁度:
#    - 每条车道只有一种 sprite 色, 目标输出也只是在同一 5 列内继续向上叠, 没有不同对象混叠到同一区域。
#    - 保留固定四车道和底部条形标记, 能让人把“哪条车道被转移”与“新堆叠结果”直接对应起来。
#
# g) 变化可推断性:
#    - 颜色变化只有一种结构化映射: 红标车道 -> 灰色, 新增精灵 -> 目标车道原色。没有额外随数值变化的隐式颜色规则。
#    - 生成器专门避免退化到原始数量范围, 因而样本会持续提供 `count > 3` 和 `dst_total > 7` 的证据。


# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_LANE_WIDTH = 5
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5
_SPRITE_PIXELS = 11

# --- CFG: tight vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 3
_BOTTOM_ROW = _BOTTOM_TOP + _SPRITE_H - 1

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 5
_MAX_LANE_TOTAL = 9
_GEN_RETRIES = 400


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the given marker color."""
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


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the fixed 3x5 plus/diamond sprite."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Top rows for a bottom-aligned tight stack, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` tightly packed sprites into one fixed lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _clear_lane(grid: np.ndarray, lane_idx: int, bg: int) -> None:
    c0 = _LANE_STARTS[lane_idx]
    grid[:, c0 : c0 + _LANE_WIDTH] = bg


def _marker_lane_idx(marker_pts: list[tuple[int, int]]) -> int:
    """Map a 1x5 marker bar back to the fixed lane index it labels."""
    _r0, c0, _r1, c1 = bbox_coords(np.array(marker_pts))
    for lane_idx, start in enumerate(_LANE_STARTS):
        if c0 == start and c1 == start + _LANE_WIDTH - 1:
            return lane_idx
    raise AssertionError("marker bar is not aligned to a known lane")


def _read_lane_stack(g: np.ndarray, lane_idx: int, bg: int) -> dict[str, int]:
    """
    Parse one lane by fixed geometry instead of connected components.

    New implicit assumptions checked here:
    - after removing row-29 markers, the lane contains only a single monochrome
      tight stack of the known 3x5 sprite;
    - the stack is contiguous in rows and follows the repeated 3/5/3 profile.
    """
    c0 = _LANE_STARTS[lane_idx]
    lane = g[:, c0 : c0 + _LANE_WIDTH]
    mask = lane != bg
    mask[_MARKER_ROW, :] = False
    coords = np.argwhere(mask)
    if coords.size == 0:
        return {"count": 0, "color": bg}

    colors = {int(lane[r, c]) for r, c in coords}
    assert len(colors) == 1, "lane stack should be monochrome"

    pix = int(coords.shape[0])
    assert pix % _SPRITE_PIXELS == 0, "lane pixel count should be a multiple of one sprite"
    count = pix // _SPRITE_PIXELS

    rows = np.flatnonzero(mask.any(axis=1)).tolist()
    assert len(rows) == count * _SPRITE_H, "tight stack should occupy exactly 3*count rows"
    assert rows == list(range(rows[0], rows[0] + len(rows))), "tight stack rows should be contiguous"
    assert rows[-1] == _BOTTOM_ROW, "all stacks are bottom-aligned in this variant"

    row_profile = [int(mask[r].sum()) for r in rows]
    assert row_profile == [3, 5, 3] * count, "row widths should reveal repeated tight plus sprites"

    return {"count": int(count), "color": int(next(iter(colors)))}


def _feasible_roles(counts: list[int]) -> list[tuple[tuple[int, ...], int, int]]:
    """All (reds, green, output_total) assignments that fit in the tight layout."""
    roles: list[tuple[tuple[int, ...], int, int]] = []
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                total = counts[green] + sum(counts[i] for i in reds)
                if total <= _MAX_LANE_TOTAL:
                    roles.append((reds, green, total))
    return roles


def _choose_visible_role(
    roles: list[tuple[tuple[int, ...], int, int]], rng: np.random.Generator
) -> tuple[tuple[int, ...], int, int] | None:
    """Prefer dense destination totals so the D16 expansion is visually obvious."""
    for threshold in (8, 7):
        cands = [role for role in roles if role[2] >= threshold]
        if cands:
            return cands[int(rng.integers(len(cands)))]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprite counts from red lanes into the green lane in a tightly packed layout.

    Input structure:
    - The scene is still a 30x30 grid with four fixed 5-column lanes, plus 1x5 red/green marker bars on row 29.
    - Every lane contains one monochrome stack of identical 3x5 plus sprites, but now adjacent sprites touch directly with period 3 instead of leaving a blank spacer row.

    Rule:
    - Each red-marked lane keeps the same number of sprites in place but recolors that whole tight stack to grey (5).
    - The green-marked lane receives the total number of source sprites, recolored to the destination lane's own color and stacked tightly from the bottom upward.
    - All marker bars are deleted after the transfer.

    Key insight:
    - Count is still conserved, but connected components no longer separate individual sprites. The repeated 3/5/3 row rhythm inside each fixed lane reveals how many tightly packed pluses are present.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and len(green_markers) == 1, "expected red sources and exactly one green destination"

    red_lanes = sorted({_marker_lane_idx(rm) for rm in red_markers})
    green_lane = _marker_lane_idx(green_markers[0])
    assert green_lane not in red_lanes, "destination lane cannot also be a red source"

    lane_info = [_read_lane_stack(g, lane_idx, bg) for lane_idx in range(_NUM_LANES)]
    dst_gain = sum(lane_info[lane_idx]["count"] for lane_idx in red_lanes)
    dst_total = lane_info[green_lane]["count"] + dst_gain
    assert 1 <= dst_total <= _MAX_LANE_TOTAL, "destination count must fit in the tight stack budget"

    out = g.copy()

    for lane_idx in red_lanes:
        _clear_lane(out, lane_idx, bg)
        _paint_lane(out, lane_idx, _GREY, lane_info[lane_idx]["count"], bg)

    _clear_lane(out, green_lane, bg)
    _paint_lane(out, green_lane, lane_info[green_lane]["color"], dst_total, bg)

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    if max(input_counts) < 4:
        return None

    roles = _feasible_roles(input_counts)
    if not roles:
        return None
    chosen = _choose_visible_role(roles, rng)
    if chosen is None:
        return None

    reds, green, _dst_total = chosen
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _clear_lane(inp, lane_idx, bg)
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    _clear_lane(inp, green, bg)
    _paint_lane(inp, green, lane_colors[green], input_counts[green], bg)

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _LANE_WIDTH] = 2

    c0 = _LANE_STARTS[green]
    inp[_MARKER_ROW, c0 : c0 + _LANE_WIDTH] = 3

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
