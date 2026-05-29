# Variant of 9aaea919
# Dimension: D3 — 目标位置与堆叠规则 (dst)
# Change: stacking — 新精灵在目标车道底部已有精灵之上，按周期4(3精灵行+1间隙行)向上堆叠空槽位... → 重力堆叠到指定行

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D3 variant
# ----------------------------------
# concreteization:
# - 保留原题的 30x30 画布、4 条固定车道、3x5 十字精灵、红/绿底部标记条、红车道变灰并向绿车道转移数量。
# - 唯一结构性新增是: 绿色目标车道内部多一条 1x5 的水平分界线，颜色固定为非背景、且不同于标记色与所有车道精灵色。
# - 分界线放在目标车道内部的某个周期栅格行上，充当“重力地板”。新增精灵不再以最底部现有精灵为起点，而是以这条线正上方最近的槽位为起点，按周期 4 向上找空槽。
#
# behavior trace:
# - 原题逻辑: 目标车道若已有精灵顶行集合为 {25, 21}，则 bottom0=max(tops)=25，新精灵从 17、13... 往上补。
# - 本变体: 若分界线在第 25 行，则真正锚点顶行为 21。输入里故意让目标车道只保留更高的槽位，例如 {13, 9}，最低两个槽位 {21, 17} 为空。
# - 这样新精灵应先落到 21，再落到 17；如果仍沿用原题的 bottom0=max(tops)=13，就会错误地从 9 上方继续放，完全看不到“分界线决定起点”的规则。
#
# complexity calibration:
# - 分界线仍只取少数几个离散行 {17, 21, 25}，避免自由度过大。
# - 目标车道现有精灵仍严格落在同一周期栅格上，只是整体悬空若干槽位，视觉上整洁且便于归纳。
# - 不额外引入旋转、镜像、多目标或颜色映射；变化只集中在目标落点。
#
# implicit assumptions audit:
# - 原 solver 的 green_state 里把 max(existing_tops) 当成堆叠起点，这在“最低槽位可能为空”时不成立，必须改为显式识别分界线。
# - 原 generator 默认绿色车道现有精灵已经贴着底部，所以即使用旧 solver 也能做对；本变体必须打破这一点，强制制造“最低可落槽位为空”的输入。
# - 颜色 2/3 仍可作为普通车道精灵色，但分界线颜色不能是 2/3，否则会和标记条的“细横条”特征混淆。
#
# legality + cleanliness:
# - 分界线位于 sprite 周期的“间隙行”，因此与最近的精灵之间始终隔着 1 行背景，不会 4 连通粘连。
# - 目标车道总精灵数受分界线容量约束；生成器只选择能在对应 divider_row 上完整容纳的角色分配。
# - 输出保持分界线，删除的只有红/绿底部标记条，因此视觉上“提示线仍在、标记条消失”很清楚。
#
# observability:
# - 人类从样例中能看到: 只有绿色目标车道内额外出现一条线；输出新增的目标色精灵总是优先填到这条线正上方的低槽位，而不是贴着画布底部或贴着已有精灵。
# - 生成器固定让目标车道输入时的最低槽位为空，因此分界线是真正决定堆叠起点的可观察控制变量，不是无效装饰。


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


def _classify_lane_components(pts_list: list[list[tuple[int, int]]]) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Split lane components into 1x5 horizontal bars and sprite-like components."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from red-marked lanes to green-marked lanes, using a divider-anchored stack.

    Input structure:
    - bg = most common color. Red (2) and green (3) thin bars still mark source and target lanes.
    - The green-marked target lane also contains one extra horizontal divider bar inside the lane.
    - Lane sprites remain repeated identical monochrome plus-shapes on a period-4 vertical lattice.

    Rule:
    - Every red lane's sprites turn grey (5), and that many sprites are recreated in the nearest green lane.
    - Recreated sprites use the source shape recolored to the target lane color, but their placement is anchored by the divider row: fill the empty lattice slots immediately above the divider, then continue upward.
    - Marker bars are erased, while the internal divider line stays.

    Key insight: The target lane's lowest existing sprite is no longer a reliable anchor — the divider bar defines the true gravity floor, so new sprites may need to appear below the pre-existing target sprites.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        dst_cols = range(dst_c0, dst_c1 + 1)
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

        gc = (dst_c0 + dst_c1) / 2.0
        green_state[id(gm)] = {
            "center": gc,
            "dst_color": dst_color,
            "period": period,
            "occupied": set(tops),
            "anchor_top": divider_row - period,
        }

    assert green_state, "expected at least one green lane with a divider and valid sprites"

    out = g.copy()

    for rm in red_markers:
        _, src_c0, _, src_c1 = bbox_coords(np.array(rm))
        src_cols = range(src_c0, src_c1 + 1)
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

        rc = (src_c0 + src_c1) / 2.0
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

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4
_DIVIDER_ROWS = [17, 21, 25]

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return bottom-anchored top-row positions, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _divider_capacity(divider_row: int) -> int:
    """How many 3x5 sprites fit above this divider on the period-4 lattice."""
    return ((divider_row - _PERIOD) // _PERIOD) + 1


def _gravity_tops_bottom_first(divider_row: int, count: int) -> list[int]:
    """Return divider-anchored top rows from lowest slot upward."""
    anchor_top = divider_row - _PERIOD
    return [anchor_top - _PERIOD * k for k in range(count)]


def _paint_lane_positions(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    _paint_lane_positions(grid, lane_idx, color, _lane_tops(count), bg)


def _feasible_roles(counts: list[int]) -> list[tuple[tuple[int, ...], int, list[int]]]:
    """Return feasible (reds, green, divider_rows) assignments."""
    roles = []
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                total = counts[green] + sum(counts[i] for i in reds)
                rows = [row for row in _DIVIDER_ROWS if _divider_capacity(row) >= total]
                if rows:
                    roles.append((reds, green, rows))
    return roles


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    divider_palette = [c for c in range(10) if c not in (bg, 2, 3, _GREY) and c not in lane_colors]
    if not divider_palette:
        return None
    divider_color = int(rng.choice(divider_palette))
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    if not feasible:
        return None

    reds, green, divider_rows = feasible[int(rng.integers(len(feasible)))]
    divider_row = int(rng.choice(divider_rows))
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)
    total_green = input_counts[green] + gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            continue
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, input_counts[lane_idx], bg)

    green_out_tops = _gravity_tops_bottom_first(divider_row, total_green)
    _paint_lane_positions(out, green, lane_colors[green], green_out_tops, bg)
    green_c0 = _LANE_STARTS[green]
    out[divider_row, green_c0:green_c0 + _SPRITE_W] = divider_color

    inp = out.copy()

    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    for r0 in green_out_tops:
        inp[r0:r0 + _SPRITE_H, green_c0:green_c0 + _SPRITE_W] = bg
    green_in_tops = green_out_tops[gain:]
    _paint_lane_positions(inp, green, lane_colors[green], green_in_tops, bg)

    inp[_MARKER_ROW, green_c0:green_c0 + _SPRITE_W] = 3
    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
