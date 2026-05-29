# Combo variant of 9aaea919 (index 8)
# combo_spec: D3/2+D7/0
# Name: 重力堆叠到分界线+紧凑裁剪

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations, product


# Design analysis for the combo variant
# -------------------------------------
# concreteization:
# - 保留原题的 30x30 固定四车道、3x5 十字精灵、底部红/绿 1x5 标记条、红车道变灰并把数量转移到绿车道。
# - D3/2 新增: 绿色目标车道内部再放一条 1x5 水平分界线，位置只取少数离散周期行 {17, 21, 25}。
#   它不是底部标记条，而是留在输出里的“重力地板”；新精灵必须从分界线上方最近的周期槽位开始堆叠。
# - D7/0 新增: 完成整幅 30x30 变换后，再把输出裁成所有非背景像素的最小外接矩形。
#
# behavior trace:
# - 例子: 分界线在 row 25，则最低可落 sprite 顶行为 row 21。若输入绿车道现有顶行为 {13, 9}，说明
#   row 21 与 row 17 这两个低槽位都空着。
# - 原题/旧 solver 会把 max(existing_tops)=13 当作锚点，于是新增精灵会继续放到 9 上方，完全错过下面的空槽。
# - 融合后应先把新增精灵落到 21、17，再保留分界线，并在最后把整张图裁到有效内容；于是输出既能看出
#   “堆叠起点变了”，也能看出“外框被收紧了”。
#
# complexity calibration:
# - 分界线只用少数几个合法行位，降低自由度，同时保证不同样例里会反复出现相同视觉结构。
# - 目标车道仍沿 period=4 的整齐格点放置，且输入时固定把最低若干槽位挖空，让分界线成为真正可观察的控制变量。
# - 裁剪只发生在全部放置完成之后，不改变颜色映射、匹配关系或局部几何，避免叠加过多维度造成混乱。
#
# implicit assumptions audit:
# - 原 solver 假设“目标车道最低现有精灵 == 新精灵堆叠起点”；D3/2 下失效，必须改为显式找分界线。
# - 原 solver 假设返回尺寸与输入相同；D7/0 下失效，但前面的检测与放置仍必须在原始 30x30 坐标系里完成。
# - 由于 green lane 内会同时出现 sprite 和水平条，单纯按单色连通块收集后不能直接都当 sprite，必须先把 1x5 bar 和
#   sprite-like 组件分开。生成器也必须保证分界线颜色不同于背景、2/3 标记色、以及各车道 sprite 色，避免歧义。
#
# legality + cleanliness:
# - 分界线放在周期的“间隙行”，与最近 sprite 至少隔 1 行背景，所以不会 4 连通粘连。
# - 生成器先构造完整终态，再反推出输入；只选择能在给定分界线上方完整容纳总 sprite 数的角色分配。
# - 由于 4 条车道输入时都至少有 1 个 sprite，裁剪后宽度稳定收敛到有效车道包围盒，而高度由最高 sprite 决定，视觉整洁。
#
# observability:
# - 人类能从样例中看到三条一致线索: 绿色车道里存在会保留到输出的内部横线；新增目标色精灵优先贴着这条线向上堆；
#   marker 消失后，整张输出图外圈空白被裁掉。
# - 生成器固定让输入绿车道最低可落槽位为空，因此若错误沿用原题的 bottom-up 规则，输出会与正确输出明显不同，
#   这保证了融合行为不会退化成原题。


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


def _classify_lane_components(
    pts_list: list[list[tuple[int, int]]],
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
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


def _tight_crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(g != bg)
    if nz.size == 0:
        return g[:0, :0].copy()
    r0, c0, r1, c1 = bbox_coords(nz)
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move red-lane sprites into the green lane using a divider-anchored gravity stack, then crop the finished scene.

    Input structure:
    - The input is still a 30x30 four-lane layout with repeated monochrome plus-shape sprites.
    - Red (2) bottom bars mark source lanes and one green (3) bottom bar marks the destination lane.
    - The destination lane also contains one extra internal 1x5 horizontal divider bar that remains after the move.

    Rule:
    - Every sprite in a red-marked lane turns grey (5), and the same number of copies is recreated in the nearest green lane using the destination lane color.
    - New copies do not stack from the canvas bottom; instead they fill the empty period-4 slots immediately above the divider, then continue upward.
    - After all transfers, remove the bottom marker bars and tight-crop to the smallest rectangle containing all non-background pixels.

    Key insight:
    - The divider row, not the lowest existing destination sprite, defines the true gravity floor. Existing green sprites may float above empty lower slots, and the final cropped output exposes both the divider-anchored refill and the removal of the unused border.
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

        divider = max(bars, key=lambda pts: bbox_coords(np.array(pts))[0])
        divider_row = bbox_coords(np.array(divider))[0]

        dst_sz = max(len(p) for p in sprite_like)
        dst_sprites = [p for p in sprite_like if len(p) == dst_sz]
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
            "center": (dst_c0 + dst_c1) / 2.0,
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

    return _tight_crop_non_bg(out, bg)


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
    """Return divider-anchored top rows from the lowest slot upward."""
    anchor_top = divider_row - _PERIOD
    return [anchor_top - _PERIOD * k for k in range(count)]


def _paint_lane_positions(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    _paint_lane_positions(grid, lane_idx, color, _lane_tops(count), bg)


def _count_options(red_count: int, divider_row: int) -> list[tuple[int, tuple[int, ...]]]:
    """All (green_count, red_counts) assignments that fit above the chosen divider."""
    cap = _divider_capacity(divider_row)
    options = []
    for green_count in range(_MIN_SPRITES, _MAX_SPRITES + 1):
        for red_counts in product(range(_MIN_SPRITES, _MAX_SPRITES + 1), repeat=red_count):
            if green_count + sum(red_counts) <= cap:
                options.append((green_count, tuple(int(x) for x in red_counts)))
    return options


def _full_output_grid(
    bg: int,
    lane_colors: list[int],
    divider_color: int,
    input_counts: list[int],
    reds: tuple[int, ...],
    green: int,
    divider_row: int,
) -> tuple[np.ndarray, list[int]]:
    red_set = set(reds)
    gain = sum(input_counts[i] for i in reds)
    total_green = input_counts[green] + gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx == green:
            continue
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, input_counts[lane_idx], bg)

    green_tops = _gravity_tops_bottom_first(divider_row, total_green)
    _paint_lane_positions(out, green, lane_colors[green], green_tops, bg)
    green_c0 = _LANE_STARTS[green]
    out[divider_row, green_c0:green_c0 + _SPRITE_W] = divider_color
    return out, green_tops


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    divider_palette = [c for c in range(10) if c not in (bg, 2, 3, _GREY) and c not in lane_colors]
    if not divider_palette:
        return None
    divider_color = int(rng.choice(divider_palette))

    green = int(rng.integers(_NUM_LANES))
    others = [i for i in range(_NUM_LANES) if i != green]
    red_count = int(rng.integers(1, 3))
    red_choices = list(combinations(others, red_count))
    if not red_choices:
        return None
    reds = tuple(int(x) for x in red_choices[int(rng.integers(len(red_choices)))])

    divider_row = int(rng.choice(_DIVIDER_ROWS))
    count_options = _count_options(red_count, divider_row)
    if not count_options:
        return None
    green_count, red_counts = count_options[int(rng.integers(len(count_options)))]

    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    input_counts[green] = int(green_count)
    for lane_idx, cnt in zip(reds, red_counts):
        input_counts[lane_idx] = int(cnt)

    gain = sum(input_counts[i] for i in reds)

    full_out, green_out_tops = _full_output_grid(
        bg,
        lane_colors,
        divider_color,
        input_counts,
        reds,
        green,
        divider_row,
    )
    cropped_out = _tight_crop_non_bg(full_out, bg)

    inp = full_out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in green_out_tops:
        inp[r0:r0 + _SPRITE_H, green_c0:green_c0 + _SPRITE_W] = bg
    green_in_tops = green_out_tops[gain:]
    _paint_lane_positions(inp, green, lane_colors[green], green_in_tops, bg)

    inp[_MARKER_ROW, green_c0:green_c0 + _SPRITE_W] = 3
    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0:c0 + _SPRITE_W] = 2

    # Fusion observability guard:
    # - D3/2 is only visible when the lowest divider-adjacent slot is absent from the input green lane.
    # - D7/0 is only visible when the final canvas really shrinks.
    if not green_in_tops or max(green_in_tops) >= green_out_tops[0]:
        return None
    if cropped_out.shape == (_H, _W):
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
