# Combo variant of 9aaea919 (index 13)
# combo_spec: D5/3+D14/0
# Name: 三色标记系统+标记变灰保留
#
# Design analysis
# analysis:
#   concrete_form:
#     - 保持原题的 30x30 画布、4 条固定 5 列宽车道、3x5 十字精灵、以及第 29 行的 1x5 横条标记。
#     - 输入中四条车道全部带标记：1-2 条红条(2)=源车道，1 条绿条(3)=目标车道，其余 1-2 条黄条(4)=冻结车道。
#     - 输出中红车道精灵变灰(5)，对应数量的新精灵补到绿车道的空槽里；黄车道精灵保持原样。
#     - 与单变体 D5/3 的关键差别在于：所有底部标记条不再消失，而是统一改成灰色(5)并保留在原位。
#     - 例子：lane0 有 2 个红精灵，lane2 有 1 个绿精灵，lane1/lane3 为黄冻结。
#       原题会做 lane0->lane2，然后删掉红绿条；D5/3 会再删掉黄条；本融合版则做同样转移，
#       但输出底行四个 1x5 条全部保留下来并变成灰色，因此“冻结角色存在”和“标记保留”都能直接看到。
#   complexity_calibration:
#     - 只增加一个新角色“冻结”和一个统一的标记后处理“全部变灰”，几何规则仍是原题的底部向上堆叠。
#     - 为避免黄色条沦为摆设，生成器强制每个样例至少有 1 条黄车道，并且不允许无标记车道。
#     - 为避免灰条与其他对象混淆，背景排除 5，车道精灵颜色也排除 5；灰色仅用于“源车道变灰”和“标记条保留后变灰”。
#   implicit_assumptions_review:
#     - 原版只查红/绿条；融合后若不查黄条，就无法验证“三角色”输入，也无法保证没有无标记车道。
#     - 原版末尾把红/绿条设回背景；融合后这条假设失效，必须统一把红/绿/黄条改成灰色。
#     - 原版默认非红非绿车道无需显式处理；现在黄条是显式“冻结”信号，solver 必须识别但不参与转移。
#     - 车道抽取仍依赖：标记条与底部精灵之间有一行背景间隔；同车道精灵同色同形；绿车道输入里至少有一个原始精灵可推周期。
#     - 由于输出底行会出现灰条，生成器必须继续保证底部精灵最低只到第 27 行，否则灰条可能与精灵连通并破坏组件分离假设。
#   placement_legality:
#     - 纵向周期仍为 4，底部 sprite top 仍为 25，因此 marker row=29 与精灵之间稳定隔开一行。
#     - 绿车道总精灵数仍受 `_MAX_LANE_TOTAL=7` 限制，保证补入后不会越界。
#     - 输出先构造完成态，再反推输入：先保留灰条，再把红车道复原、把绿车道新增的顶部若干精灵挖空、最后把灰条改回角色色。
#   visual_observability:
#     - 输入里四条车道全部带条，解题者能直接比较三种角色：红=捐出并变灰，绿=接收补入，黄=保持不动。
#     - 输出里仍能看到四个灰条，这让“标记没有消失，而是统一变灰”的新规则在 2-3 个样例上都稳定可见。
#     - 变化依赖的是条的位置和颜色类别，而不是颜色编号大小，因此不存在人类不可见的排序信息。
#   visual_cleanliness:
#     - 新信息仍只集中在底部一行，同形同宽的条不会和上方精灵重叠；车道间横向间隔也保持不变。
#     - 灰条与灰化后的红车道精灵虽同色，但垂直上分离明显：灰条始终在最后一行，灰精灵位于各车道内部。
#   color_change_inference:
#     - 结构化颜色变化只有两类：红车道精灵 -> 灰；全部标记条 -> 灰。绿车道补入的精灵继续使用目标车道自身颜色。
#     - 黄色条并不会触发颜色映射，只表明“该车道冻结”；其输出变灰来自统一的条规则，而不是车道内容规则。
#   solver_assumptions_after_change:
#     - `_find_marker_bars` 仍假设标记是独立的细长 4 连通分量；生成器固定使用底行 1x5 横条满足这一点。
#     - `_extract_lane_sprites` 假设 lane 内除 marker 外存在至少一个非 marker 单色组件；生成器令每条车道输入精灵数为 1-3。
#     - 目标状态构建与放置顺序假设“补入槽位按 top row 判重即可”，因为所有 sprite 共用固定高度与 period，生成器持续保证这一晶格。

import numpy as np

from itertools import combinations

from arc_utils import bbox_coords, components_mask, get_bg, paste


_SRC_MARKER = 2
_DST_MARKER = 3
_FROZEN_MARKER = 4
_GREY = 5


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Return thin 4-connected marker components of the requested color."""
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
    mask = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(mask, connectivity=4)]
    out = []
    for pts in comps:
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
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
    lane_center = (lane_c0 + lane_c1) / 2.0
    if not comps:
        return lane_center, []

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return lane_center, []

    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    return lane_center, sprites


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
    Interpret the bottom bars as a three-role legend, then keep those bars visible by turning them grey in the result.

    Input structure:
    - The most common color is background. Each fixed lane has a 1x5 bar on the last row: red (2) marks a source lane, green (3) marks the receiving lane, and yellow (4) marks a frozen lane.
    - Every lane contains repeated monochrome plus sprites positioned on the same vertical lattice, with a blank row between the lowest sprite and the marker row.

    Rule:
    - Sprites from every red-marked lane turn grey (5), and that many copies of the source shape are inserted into the nearest green-marked lane from the bottom upward using the destination lane's own color.
    - Yellow-marked lanes do not donate or receive anything; after all lane logic is complete, every marker bar stays in place but is recolored to grey (5).

    Key insight:
    - The transfer geometry is still the original lane-stacking mechanism; the fused novelty is that the solver must classify three marker roles while also recognizing that marker bars persist as a common grey trace instead of disappearing.
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

        placed = 0
        k = 0
        while placed < len(src_sprites):
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
                out[r, c] = _GREY

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
    """Return the 3x5 plus/diamond sprite with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return sprite top rows from top to bottom for the given stack size."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (red source lanes, green destination lane) assignments."""
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
    # Keep sprite colors disjoint from all semantic marker/trace colors so that
    # the three-role legend is readable from structure instead of ambiguous reuse.
    sprite_palette = [
        c for c in range(10) if c not in (bg, _SRC_MARKER, _DST_MARKER, _FROZEN_MARKER, _GREY)
    ]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [red_count for red_count in (1, 2) if feasible[red_count]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    frozen_set = {lane_idx for lane_idx in range(_NUM_LANES) if lane_idx not in red_set and lane_idx != green}
    if not frozen_set:
        return None
    gain = sum(input_counts[lane_idx] for lane_idx in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        lane_color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, lane_color, output_counts[lane_idx], bg)
        c0 = _LANE_STARTS[lane_idx]
        out[_MARKER_ROW, c0 : c0 + _SPRITE_W] = _GREY

    inp = out.copy()

    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg

    for lane_idx in range(_NUM_LANES):
        c0 = _LANE_STARTS[lane_idx]
        if lane_idx == green:
            marker_color = _DST_MARKER
        elif lane_idx in red_set:
            marker_color = _SRC_MARKER
        else:
            marker_color = _FROZEN_MARKER
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = marker_color

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
