# Combo variant of 9aaea919 (index 34)
# combo_spec: D5/3+D14/0+D17/0
# Name: 三色标记+标记变灰+旁观变灰

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# analysis:
#   a) concrete_form_and_trace:
#     - 保留原题 30x30 画布、4 条固定 5 列车道、3x5 十字精灵、周期 4 的竖向堆叠。
#     - 融合后输入里同时出现四种车道角色：1 条红标源车道、1 条绿标目标车道、1 条黄标冻结车道、1 条无标记旁观车道。
#     - 具体追踪例子：若 lane0=红且有 2 个精灵，lane1=黄且有 3 个精灵，lane2=绿且有 1 个精灵，lane3=无标记且有 2 个精灵，
#       则输出中 lane0 精灵变灰，lane2 在底部向上新增 2 个本车道颜色精灵，lane1 精灵保持原色不动，lane3 精灵变灰。
#       同时 lane0/lane1/lane2 底部条都不删除，而是统一变灰；lane3 仍然没有条。
#     - 与原题相比，变化点不再只是“源变灰+目标增殖”，而是要同时区分冻结与旁观两种都不转移但后果不同的非目标角色。
#   b) complexity_calibration:
#     - 若让全部非目标车道都变灰，黄色冻结角色会失去存在意义；若让所有剩余车道都黄标，则“旁观变灰”无法观察。
#     - 因此生成器固定每个样例都含四种角色各一条车道，让黄冻结与无标记旁观在同一图里形成稳定对照。
#     - 红/绿/黄三种标记仍共用同一 1x5 底边横条，几何复杂度不增加，新增难度只来自角色语义分化。
#   c) implicit_assumptions_review:
#     - 原 solver 默认“非红非绿车道完全不需要分类”；融合后必须区分黄冻结与无标记旁观，否则会把两者错误地统一处理。
#     - 原 D14/0 只要求红绿条改灰；这里黄条也属于显式标记，因此必须一起改灰，而不是保留黄色或直接擦除。
#     - 原 D17/0 的“所有非目标都变灰”若直接照搬，会抹掉黄冻结的独立语义；融合版需要把 D17 限定到无标记旁观车道。
#     - 原代码通过“按标记列提取车道”工作；现在有一条车道没有标记，所以 solver 不能只遍历标记条，必须补上固定车道位置信息。
#   d) placement_legality:
#     - 仍使用 bottom-top=25、period=4、目标总数上限 7，保证目标车道新增精灵不会越界且不会与现有精灵重叠。
#     - 标记条仍位于第 29 行，最底精灵占到第 27 行，中间留出 1 行空隙，任何标记颜色都不会与精灵连通。
#     - 生成器把背景和精灵颜色都排除 {2,3,4,5}，避免标记色和灰色与车道主体混淆，提升融合规则可读性。
#   e) visual_observability:
#     - 每个样例都同时含有红、绿、黄、无标记四种车道，观察者能直接比对：
#       红=变灰且贡献数量，绿=接收且保色，黄=不接收不变色但条变灰，无标记=本车道变灰但不贡献数量。
#     - 底行是否仍保留横条、以及哪一条车道没有横条，都是纯视觉线索，不依赖颜色编号大小等不可视因素。
#   f) visual_cleanliness:
#     - 所有变化仍限制在四个竖向槽位和最底一行标记区，不引入新形状或重叠区域，输出仍是规整的四列堆叠图。
#     - 黄冻结与无标记旁观的唯一区别集中在“是否有黄条、输出是否留本色”，不会在同一区域叠加多层符号。
#   g) color_change_inference:
#     - 颜色变化被结构强约束：只有绿目标保留/接收本色；红源和无标记旁观变灰；黄冻结保持输入颜色；所有显式标记统一变灰。
#     - 这种映射由底部角色条和是否缺少条共同提示，不需要人类猜测任意颜色置换表。
#   h) solver_assumptions_after_change:
#     - `main()` 假设四条车道仍处于固定列范围，并且三类标记都仍是与车道对齐的 1x5 细横条；generator 按此构造。
#     - `main()` 还假设目标车道至少已有 1 个精灵可供推断周期和目标颜色；generator 保持每条车道输入精灵数为 1-3。
#     - 红源车道贡献数量必须能放入目标车道空槽；generator 仅选择满足 `target_count + source_count <= 7` 的角色分配。


_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4

_SRC_MARKER = 2
_DST_MARKER = 3
_FROZEN_MARKER = 4
_GREY = 5

_BG_CHOICES = [0, 1, 6, 7, 8, 9]
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


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


def _lane_sprites_in_cols(
    g: np.ndarray,
    cols: range,
    bg: int,
    ignored: set[tuple[int, int]] | None = None,
) -> list[list[tuple[int, int]]]:
    comps = _single_color_components_in_cols(g, cols, bg)
    if ignored:
        comps = [[p for p in pts if p not in ignored] for pts in comps]
        comps = [pts for pts in comps if pts]
    if not comps:
        return []
    sprite_sz = max(len(pts) for pts in comps)
    return [pts for pts in comps if len(pts) == sprite_sz]


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    items = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        items.append((rmin, cmin, pts))
    items.sort()
    return items


def _lane_bounds_from_marker(marker_pts: list[tuple[int, int]]) -> tuple[int, int]:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    return int(c0), int(c1)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve four lane roles at once: red sources donate, the green target receives, yellow lanes freeze, and the lone unmarked lane is only bleached.

    Input structure:
    - The most common color is the background. Four fixed 5-column lanes hold repeated monochrome plus sprites on a shared vertical lattice.
    - Bottom-row 1x5 bars encode three explicit roles: red (2) marks source lanes, green (3) marks the destination lane, and yellow (4) marks a frozen lane. One lane has no marker.

    Rule:
    - Each red-marked lane contributes its sprite count to the nearest green-marked lane; copied sprites use the source shape but the target lane's color and fill empty slots from the bottom upward.
    - Red source sprites turn grey (5). The unmarked bystander lane also turns grey. The yellow frozen lane keeps its original sprite color and does not transfer anything.
    - Red, green, and yellow marker bars are not erased; they all remain in place but are recolored to grey (5).

    Key insight:
    - The puzzle no longer has a single “all non-target lanes behave alike” rule. You must separate explicit yellow freeze markers from the absence of a marker: both are non-target lanes, but only the unmarked lane is bleached, while only the red lane donates.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, _SRC_MARKER)
    green_markers = _find_marker_bars(g, _DST_MARKER)
    yellow_markers = _find_marker_bars(g, _FROZEN_MARKER)
    assert red_markers and green_markers and yellow_markers, "expected red, green, and yellow marker bars"

    role_by_lane: dict[int, str] = {c0: "bystander" for c0 in _LANE_STARTS}
    marker_by_lane: dict[int, list[tuple[int, int]]] = {}
    for markers, role in (
        (red_markers, "source"),
        (green_markers, "target"),
        (yellow_markers, "frozen"),
    ):
        for marker in markers:
            c0, _c1 = _lane_bounds_from_marker(marker)
            role_by_lane[c0] = role
            marker_by_lane[c0] = marker

    assert list(role_by_lane.values()).count("target") == 1, "expected exactly one target lane"
    assert "bystander" in role_by_lane.values(), "expected one unmarked bystander lane"

    green_state = []
    for gm in green_markers:
        dst_c0, dst_c1 = _lane_bounds_from_marker(gm)
        dst_cols = range(dst_c0, dst_c1 + 1)
        dst_sprites = _lane_sprites_in_cols(g, dst_cols, bg, ignored=set(gm))
        if not dst_sprites:
            continue

        dst_sprites_sorted = _sorted_sprites(dst_sprites)
        dst_color = int(g[dst_sprites_sorted[0][2][0][0], dst_sprites_sorted[0][2][0][1]])
        tops = sorted({rmin for rmin, _cmin, _pts in dst_sprites_sorted})
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(dst_sprites_sorted[0][2]))
            period = (r1 - r0 + 1) + 1

        green_state.append(
            {
                "center": (dst_c0 + dst_c1) / 2.0,
                "dst_color": dst_color,
                "period": period,
                "occupied": set(tops),
                "bottom0": max(tops),
            }
        )

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for lane_idx, c0 in enumerate(_LANE_STARTS):
        role = role_by_lane[c0]
        if role not in {"source", "bystander"}:
            continue
        lane_cols = range(c0, c0 + _SPRITE_W)
        marker = marker_by_lane.get(c0)
        ignored = set(marker) if marker is not None else None
        lane_sprites = _lane_sprites_in_cols(g, lane_cols, bg, ignored=ignored)
        for pts in lane_sprites:
            for r, c in pts:
                out[r, c] = _GREY

    for rm in red_markers:
        src_c0, src_c1 = _lane_bounds_from_marker(rm)
        src_cols = range(src_c0, src_c1 + 1)
        src_sprites = _lane_sprites_in_cols(g, src_cols, bg, ignored=set(rm))
        if not src_sprites:
            continue

        rc = (src_c0 + src_c1) / 2.0
        best = min(green_state, key=lambda st: abs(st["center"] - rc))

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


def _feasible_role_assignments(counts: list[int]) -> list[tuple[int, int, int, int]]:
    feasible = []
    for green in range(_NUM_LANES):
        for red in range(_NUM_LANES):
            if red == green:
                continue
            if counts[green] + counts[red] > _MAX_LANE_TOTAL:
                continue
            leftovers = [i for i in range(_NUM_LANES) if i not in {red, green}]
            frozen, bystander = leftovers
            feasible.append((red, green, frozen, bystander))
            feasible.append((red, green, bystander, frozen))
    return feasible


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _SRC_MARKER, _DST_MARKER, _FROZEN_MARKER, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_role_assignments(input_counts)
    if not feasible:
        return None
    red, green, frozen, bystander = feasible[int(rng.integers(len(feasible)))]

    output_counts = input_counts[:]
    output_counts[green] += input_counts[red]

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = lane_colors[lane_idx]
        if lane_idx in (red, bystander):
            color = _GREY
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    for lane_idx in (red, green, frozen):
        c0 = _LANE_STARTS[lane_idx]
        out[_MARKER_ROW, c0 : c0 + _SPRITE_W] = _GREY

    inp = out.copy()
    for lane_idx in (red, frozen, bystander):
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[: input_counts[red]]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg

    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = _DST_MARKER
    red_c0 = _LANE_STARTS[red]
    inp[_MARKER_ROW, red_c0 : red_c0 + _SPRITE_W] = _SRC_MARKER
    frozen_c0 = _LANE_STARTS[frozen]
    inp[_MARKER_ROW, frozen_c0 : frozen_c0 + _SPRITE_W] = _FROZEN_MARKER

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
