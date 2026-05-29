# Combo variant of 9aaea919 (index 36)
# combo_spec: D3/2+D5/1+D7/0
# Name: 重力分界线+侧边标记+紧凑裁剪

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations, product


# Design analysis for the combo variant
# -------------------------------------
# concreteization:
# - 仍使用 30x30 画布、4 条固定 5 列车道、3x5 十字精灵、红源/绿目标/灰结果 的原题核心语义。
# - D5/1 把原题底部 1x5 横条改成“车道左侧 1 列竖条”。例如 lane0 的车道列是 3..7，
#   则其标记竖条位于第 2 列，贯穿车道活动高度；红竖条表示源车道，绿竖条表示目标车道。
# - D3/2 只作用于目标车道：车道内部新增一条 1x5 的水平分界线，颜色不同于背景、标记色和所有车道色。
#   新精灵不再参考“当前最低已有精灵”，而是以 divider_row 正上方的周期槽位作为重力落点。
# - D7/0 要求最终输出不是 30x30，而是在完成重涂、转移、删标记后，对所有非背景像素做最小外接矩形裁剪。
#
# behavior trace:
# - 具体例子：假设目标车道在列 21..25，绿色侧标在第 20 列，分界线在第 21 行。
#   分界线上方可落顶行依次是 17, 13, 9, 5, 1。
# - 若输入目标车道只保留较高的原生精灵，例如顶行集合是 {9, 5}，而两个红源车道总共要转移 2 个精灵，
#   原题会把 max(existing_tops)=9 当成锚点，从 5、1 往上找，甚至和已有结构冲突。
# - 融合规则必须先识别第 20 列的绿色侧标以确定目标车道，再识别 lane 内部分界线，
#   然后把两枚新精灵放到 17 和 13；最后删除侧边标记并把整幅结果裁成只包住精灵与分界线的紧凑矩形。
#
# complexity calibration:
# - 分界线只取离散行 {17, 21, 25}，这样既能产生 4/5/6 个不同容量，也不会让自由度过高。
# - 侧边标记全部放在“车道左侧紧邻一列”，位置规律稳定；目标识别复杂度来自相对位置，而不是任意散布。
# - 普通车道颜色刻意不使用 2 和 3，避免侧边标记与普通精灵同色且横向相邻时发生 4 连通粘连，保持视觉和解析都清晰。
#
# implicit assumptions audit:
# - 原 solver 用 marker bbox 直接当车道列范围；D5/1 后这不成立，必须从 marker_col 推回 marker_col+1..marker_col+5。
# - 原 solver 用目标车道最低已有精灵 `bottom0` 决定补位起点；D3/2 后最低槽位可能为空，必须改为显式识别 divider。
# - 原 solver 默认 `return out` 保持 30x30；D7/0 后这个结尾语义错误，必须在删掉所有侧边标记后再裁剪。
# - 目标车道内部现在同时有“精灵 + 分界线”两类单色组件，所以解析时必须把 1x5 水平条和 sprite-like 组件分开。
#
# legality + cleanliness:
# - 侧边标记位于车道左侧 1 列，不会与车道内部对象重叠；分界线始终画在周期 gap 行，与上下精灵至少隔 1 行背景。
# - 生成器按 divider 容量筛选角色分配，保证目标总精灵数不越界；输入中的目标车道至少保留 1 个原生精灵，便于推断目标颜色。
# - 输出删除所有侧标，但保留分界线；裁剪后的结果只含真正任务对象，视觉上整洁。
#
# observability:
# - 人类从 2-4 个样例里能稳定看到三件事：标记跑到车道左边、目标新增精灵总是贴着分界线上方堆叠、输出边框被裁掉。
# - 生成器刻意让目标车道输入时的最低几个可落槽位为空，这样“divider 决定落点”是可观察变量，而不是无效装饰。
# - D7 裁剪同样可观察：底部两行和左右空边会消失，且高度随最高保留精灵/分界线变化，不会退化成固定尺寸。


def _find_side_markers(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Return thin vertical marker components of the requested color."""
    markers = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if ww == 1 and hh >= 3:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            markers.append((c0, r0, pts))
    markers.sort()
    return [pts for _c0, _r0, pts in markers]


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    m = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
            out.append(pts)
    return out


def _classify_lane_components(
    pts_list: list[list[tuple[int, int]]],
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Split lane components into 1x5 divider bars and sprite-like objects."""
    bars = []
    sprites = []
    for pts in pts_list:
        r0, c0, r1, c1 = bbox_coords(np.array(pts))
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 and ww == 5:
            bars.append(pts)
        else:
            sprites.append(pts)
    return bars, sprites


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    out = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        out.append((rmin, cmin, pts))
    out.sort()
    return out


def _tight_crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(g != bg)
    if nz.size == 0:
        return g[:0, :0].copy()
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _lane_cols_from_marker(marker_pts: list[tuple[int, int]]) -> range:
    _, c0, _, _ = bbox_coords(np.array(marker_pts))
    lane_c0 = c0 + 1
    return range(lane_c0, lane_c0 + 5)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move source-lane sprites into a divider-anchored target lane, then crop the result.

    Input structure:
    - The scene is still a 30x30 four-lane layout with repeated monochrome plus sprites on a period-4 vertical lattice.
    - Thin side markers replace the old bottom bars: red (2) vertical bars sit immediately left of source lanes, and one green (3) vertical bar sits immediately left of the target lane.
    - The target lane also contains a horizontal divider line inside the lane; it marks the gravity floor for newly inserted sprites.

    Rule:
    - Every sprite in a red-marked lane turns grey (5).
    - The same number of sprites is recreated in the nearest green-marked lane, recolored to the target lane color and stacked into the empty lattice slots directly above the divider, then upward by the lane period.
    - All side markers are erased, the divider remains, and the final full-grid result is tight-cropped to the minimal non-background rectangle.

    Key insight: The green side marker tells you which lane receives new sprites, but the divider inside that lane tells you where the stack begins; existing target sprites may float above empty lower slots, so the divider is the only reliable anchor before the final crop.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_side_markers(g, 2)
    green_markers = _find_side_markers(g, 3)
    assert red_markers and green_markers, "expected both red and green side markers"

    green_state = []
    for gm in green_markers:
        dst_cols = _lane_cols_from_marker(gm)
        dst_set = set(gm)
        dst_comps = _single_color_components_in_cols(g, dst_cols, bg)
        dst_comps = [[p for p in pts if p not in dst_set] for pts in dst_comps]
        dst_comps = [pts for pts in dst_comps if pts]
        bars, sprite_like = _classify_lane_components(dst_comps)
        if not bars or not sprite_like:
            continue

        dst_sz = max(len(p) for p in sprite_like)
        dst_sprites = [p for p in sprite_like if len(p) == dst_sz]
        if not dst_sprites:
            continue

        divider = max(bars, key=lambda pts: bbox_coords(np.array(pts))[0])
        divider_row = bbox_coords(np.array(divider))[0]

        dst_sprites_sorted = _sorted_sprites(dst_sprites)
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])

        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        lane_c0 = dst_cols.start
        lane_c1 = dst_cols.stop - 1
        green_state.append(
            {
                "center": (lane_c0 + lane_c1) / 2.0,
                "dst_color": dst_color,
                "period": period,
                "occupied": set(tops),
                "anchor_top": divider_row - period,
            }
        )

    assert green_state, "expected at least one green lane with divider and sprites"

    out = g.copy()

    for rm in red_markers:
        src_cols = _lane_cols_from_marker(rm)
        src_set = set(rm)
        src_comps = _single_color_components_in_cols(g, src_cols, bg)
        src_comps = [[p for p in pts if p not in src_set] for pts in src_comps]
        src_comps = [pts for pts in src_comps if pts]
        _bars, sprite_like = _classify_lane_components(src_comps)
        if not sprite_like:
            continue

        src_sz = max(len(p) for p in sprite_like)
        src_sprites = [p for p in sprite_like if len(p) == src_sz]
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        src_c0 = src_cols.start
        src_c1 = src_cols.stop - 1
        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state, key=lambda st: abs(st["center"] - rc))

        src_sprites_sorted = _sorted_sprites(src_sprites)
        src_ref_r0, _src_ref_c0, src_ref_pts = src_sprites_sorted[0]

        dc_shift = int(round(best["center"] - rc))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
            new_top = best["anchor_top"] - k * best["period"]
            k += 1
            if new_top < 0:
                break
            if new_top + sprite_h > h:
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

    for marker in red_markers + green_markers:
        for r, c in marker:
            out[r, c] = bg

    return _tight_crop_non_bg(out, bg)


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_SPRITE_H, _SPRITE_W = 3, 5
_SIDE_MARKER_OFFSET = 1
_SIDE_MARKER_TOP = 1
_SIDE_MARKER_BOTTOM = 27

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4
_DIVIDER_ROWS = [17, 21, 25]

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5
_LANE_COLOR_POOL = [0, 1, 4, 6, 7, 8, 9]

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _divider_capacity(divider_row: int) -> int:
    return ((divider_row - _PERIOD) // _PERIOD) + 1


def _gravity_tops_bottom_first(divider_row: int, count: int) -> list[int]:
    anchor_top = divider_row - _PERIOD
    return [anchor_top - _PERIOD * k for k in range(count)]


def _paint_lane_positions(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    _paint_lane_positions(grid, lane_idx, color, _lane_tops(count), bg)


def _paint_side_marker(grid: np.ndarray, lane_idx: int, color: int) -> None:
    marker_col = _LANE_STARTS[lane_idx] - _SIDE_MARKER_OFFSET
    grid[_SIDE_MARKER_TOP : _SIDE_MARKER_BOTTOM + 1, marker_col] = color


def _valid_transfer_counts(red_count: int, capacity: int) -> list[tuple[int, tuple[int, ...]]]:
    options = []
    counts_range = range(_MIN_SPRITES, _MAX_SPRITES + 1)
    for green_count in counts_range:
        for red_counts in product(counts_range, repeat=red_count):
            if green_count + sum(red_counts) <= capacity:
                options.append((green_count, red_counts))
    return options


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    lane_palette = [c for c in _LANE_COLOR_POOL if c != bg]
    lane_colors = rng.choice(lane_palette, size=_NUM_LANES, replace=False).tolist()
    divider_palette = [c for c in range(10) if c not in (bg, 2, 3, _GREY) and c not in lane_colors]
    if not divider_palette:
        return None
    divider_color = int(rng.choice(divider_palette))

    green = int(rng.integers(_NUM_LANES))
    red_count = int(rng.choice([1, 2]))
    others = [i for i in range(_NUM_LANES) if i != green]
    red_choices = list(combinations(others, red_count))
    reds = tuple(red_choices[int(rng.integers(len(red_choices)))])
    divider_row = int(rng.choice(_DIVIDER_ROWS))
    transfer_options = _valid_transfer_counts(red_count, _divider_capacity(divider_row))
    if not transfer_options:
        return None

    green_count, red_counts = transfer_options[int(rng.integers(len(transfer_options)))]
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    input_counts[green] = int(green_count)
    for lane_idx, count in zip(reds, red_counts):
        input_counts[lane_idx] = int(count)

    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)
    total_green = input_counts[green] + gain

    full_out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            continue
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(full_out, lane_idx, color, input_counts[lane_idx], bg)

    green_out_tops = _gravity_tops_bottom_first(divider_row, total_green)
    _paint_lane_positions(full_out, green, lane_colors[green], green_out_tops, bg)
    green_c0 = _LANE_STARTS[green]
    full_out[divider_row, green_c0 : green_c0 + _SPRITE_W] = divider_color
    cropped_out = _tight_crop_non_bg(full_out, bg)

    inp = full_out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    inp[divider_row, green_c0 : green_c0 + _SPRITE_W] = divider_color
    for r0 in green_out_tops:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    green_in_tops = green_out_tops[gain:]
    _paint_lane_positions(inp, green, lane_colors[green], green_in_tops, bg)

    _paint_side_marker(inp, green, 3)
    for lane_idx in reds:
        _paint_side_marker(inp, lane_idx, 2)

    if cropped_out.shape == (_H, _W):
        return None
    if green_in_tops and max(green_in_tops) == divider_row - _PERIOD:
        return None
    if not np.array_equal(main(inp), cropped_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
