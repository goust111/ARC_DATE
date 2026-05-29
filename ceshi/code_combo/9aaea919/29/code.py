# Combo variant of 9aaea919 (index 29)
# combo_spec: D1/0+D5/2+D14/0
# Name: 颜色匹配+单像素标记+标记变灰

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations, permutations


# Design analysis
# variant_spec:
#   fusion_summary:
#     - 保留原题“四条固定车道 + 十字精灵搬运 + 源车道变灰 + 目标车道自底向上补齐”的主体。
#     - 去掉原来的红/绿 1x5 条形 marker，改成底部两行中的单像素 cue。
#     - cue 的颜色不再直接携带“源/目标”语义；颜色只负责匹配真实车道，角色由 cue 所在的提示行给出。
#     - 处理后 cue 不消失，而是在原地统一变成灰色(5)。
#   a_concretization_and_trace:
#     input_geometry:
#       grid: 30x30
#       lanes:
#         count: 4
#         lane_cols: [3..7], [9..13], [15..19], [21..25]
#         lane_centers: [5, 11, 17, 23]
#       sprites:
#         shape: 3x5 plus/diamond
#         lane_color_rule: 四条车道颜色互异，且都不是 bg/5
#         counts: 每条车道 1-3 个，周期仍为 4
#       fused_cues:
#         source_cues:
#           row: 29
#           shape: 单像素
#           count: 1-2
#         target_cue:
#           row: 28
#           shape: 单像素
#           count: 1
#         placement_rule:
#           - cue 只占用四个“车道底部中心列”之一，因此继承 D5 的“底部单像素”视觉效果
#           - 但 cue 故意不放在其所指向车道的正下方，而是放到别的中心列槽位中
#           - 因此位置只能告诉解题者“这是一枚源 cue 还是目标 cue”，不能直接告诉是哪条车道
#         color_rule:
#           - 每个 cue 的颜色等于某条真实车道当前精灵颜色
#           - 解题者必须把 cue 颜色与四条车道颜色做唯一匹配，才能找到真实源/目标
#       concrete_example:
#         - 假设四条车道颜色从左到右为 [8, 2, 4, 7]
#         - row 29 上出现两个 cue，颜色分别为 7 与 2，但它们落在列 5 和 17
#         - row 28 上出现一个颜色 4 的 cue，落在列 23
#         - 正确解释不是“列 5/17/23 对应的车道有特殊角色”，而是：
#           color 7 的车道与 color 2 的车道是源，color 4 的车道是目标
#         - 之后与原题相同：源车道全部变灰；目标车道得到等量新精灵；三个 cue 在输出里都保留，但统一变成灰色
#     behavior_trace:
#       old:
#         - 直接看 row 29 的红条/绿条就能按位置读出源 lane 与目标 lane
#         - 变换后 marker 完全消失
#       fused:
#         - 先看 cue 行：row 29 表示“源角色”，row 28 表示“目标角色”
#         - 再看 cue 颜色，把它与四条 lane 的精灵颜色做匹配
#         - 因为 cue 槽位与真实车道故意错位，不能只靠 cue 所在列做判断
#         - 输出里 cue 仍保留在原位置，但全部改成灰色
#   b_complexity_calibration:
#     why_not_too_simple:
#       - 如果把 cue 直接放在对应车道正下方，那么单像素 cue 会退化成“缩小版原 marker”，D1 的颜色匹配将失效
#       - 现在必须经过“先认 cue 所属提示行，再按颜色匹配真实车道”的两步推理
#     ambiguity_control:
#       - 车道颜色强制互异，保证每个 cue 颜色只能匹配一条 lane
#       - source cue 限定 1-2 个，target cue 恰好 1 个，角色档位很少
#       - 所有 cue 都只落在四个固定底部中心槽位，视觉上规整，不会散乱
#       - generator 强制每个 cue 槽位与其真实车道错位，避免退化样本
#   c_implicit_assumption_audit_on_original_main:
#     - old `_find_marker_bars(...)` 依赖颜色 2/3 的细条存在；融合后完全失效，必须删除
#     - old `bbox(marker)` 用 marker 的列范围定位 lane；融合后 cue 槽位是“提示托盘”，不再对应真实 lane，不能沿用
#     - old `src_set/dst_set` 从 lane 组件中剔除 marker 像素；融合后 cue 不与真实目标 lane 绑定，只会作为 1 像素杂散组件出现，改为按固定 lane 槽提取最大组件即可
#     - old “nearest green lane” 在原题里是对 marker 位置的补充；融合后目标 lane 必须由 target cue 的颜色唯一确定，不能再靠几何最近邻
#     - old 结尾把 marker 设回 bg；融合后要改成把 cue 设成灰色
#     - new main 的每个关键前提：
#       1. 四条 lane 都有至少一个有效 sprite，且颜色互异
#       2. 只有 row 28/29 的四个中心槽位可能出现 cue
#       3. row 29 有 1-2 个 source cue，row 28 有 1 个 target cue
#       4. cue 颜色都能在 lane 颜色里唯一匹配到真实车道
#       generator 都会显式保证这些条件
#   d_placement_legality:
#     - sprite 仍放在固定 5 列宽车道中，周期仍为 4，底部 top row 仍为 25，不会互相重叠
#     - 最底 sprite 覆盖行 25..27；target cue 放在 row 28，source cue 放在 row 29，因此 cue 与 sprite 物理分离
#     - cue 槽位只有四个中心列 [5,11,17,23]，且总 cue 数最多 3，可做无重叠注入式分配
#     - 目标车道总 sprite 数仍受 `_MAX_LANE_TOTAL` 约束，保证新增 sprite 全部能在网格内自底向上落下
#   e_visual_observability:
#     - 与原题相比，底部不再是大条 marker，而是三枚以内的小点，且输出里这些点不会消失而会变灰
#     - 多组样例中，cue 常常不在其对应车道下方；只有“颜色与某车道一致”这一线索稳定，从而能支撑人类归纳颜色匹配规则
#     - 角色只由 row 28/29 区分，不依赖颜色 id 的大小或任何不可视属性
#   f_visual_cleanliness:
#     - 主体精灵始终在中部四条竖直 lane；cue 只占用最底两行的四个中心点，分区清楚
#     - cue 数量小，且输出统一变灰，画面不会像随机噪声
#   g_change_inferability:
#     - 颜色变化依旧只有两类：源精灵 -> 灰；cue -> 灰
#     - 目标 lane 的新精灵继续保持目标车道原色，因此“哪个颜色是目标 lane 自身颜色”仍能从输入直接看出
#     - cue 颜色种类最多 3 个，并且都能在四个 lane 颜色中找到对应，推断链短而可见
#   post_change_generator_checks:
#     - generator 采用 output-first：先生成真正输出，再反推输入并给 cue 上色
#     - 额外加入“错位 cue 分配”约束，量化上确保颜色匹配不是摆设


_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_LANE_CENTERS = [c0 + 2 for c0 in _LANE_STARTS]
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300
_SOURCE_CUE_ROW = 29
_TARGET_CUE_ROW = 28


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    out = []
    colors = sorted({int(g[r, c]) for r in range(h) for c in cols if int(g[r, c]) != bg})
    for color in colors:
        mask = (g == color) & colmask
        for pts in components_mask(mask, connectivity=4):
            out.append([(int(r), int(c)) for r, c in pts])
    return out


def _lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict | None:
    c0 = _LANE_STARTS[lane_idx]
    comps = _single_color_components_in_cols(g, range(c0, c0 + _SPRITE_W), bg)
    if not comps:
        return None

    sprite_sz = max(len(pts) for pts in comps)
    if sprite_sz <= 1:
        return None
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    if not sprites:
        return None

    sprites_sorted = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        sprites_sorted.append((rmin, cmin, pts))
    sprites_sorted.sort()

    color = int(g[sprites_sorted[0][2][0][0], sprites_sorted[0][2][0][1]])
    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    return {
        "lane_idx": lane_idx,
        "center": float(_LANE_CENTERS[lane_idx]),
        "color": color,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites": [pts for _rmin, _cmin, pts in sprites_sorted],
    }


def _cue_marks(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    source_marks = []
    target_marks = []
    for slot_idx, col in enumerate(_LANE_CENTERS):
        src_val = int(g[_SOURCE_CUE_ROW, col])
        dst_val = int(g[_TARGET_CUE_ROW, col])
        if src_val != bg:
            source_marks.append((slot_idx, src_val))
        if dst_val != bg:
            target_marks.append((slot_idx, dst_val))
    return source_marks, target_marks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites using bottom single-pixel cues whose colors must be matched back to lanes.

    Input structure:
    - The grid has four fixed vertical lanes of repeated monochrome plus sprites on the usual period-4 stack.
    - Role markers are no longer colored bars. Instead, 1-2 source cues appear as single pixels on row 29 and one target cue appears as a single pixel on row 28.
    - Each cue sits in one of the four bottom-center cue slots, but its slot is only a cue location: the cue color, not the slot, identifies the real lane.

    Rule:
    - Match every cue color to the unique lane that currently uses that sprite color.
    - Lanes named by row-29 cues are sources: all of their sprites turn grey (5), and the same number of sprites is added into the lane named by the row-28 cue.
    - New sprites keep the target lane color and stack upward in empty lattice slots. The cue pixels stay visible, but every cue is recolored to grey.

    Key insight:
    - Cue row determines the role (source tray vs target tray), while cue color determines the actual lane; cue slot position is intentionally decoupled from lane identity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    source_marks, target_marks = _cue_marks(g, bg)
    assert 1 <= len(source_marks) <= 2, "expected 1-2 source cue pixels"
    assert len(target_marks) == 1, "expected exactly one target cue pixel"
    assert len({slot for slot, _color in source_marks}) == len(source_marks), "source cue slots must be distinct"

    lane_states = []
    for lane_idx in range(_NUM_LANES):
        st = _lane_state(g, lane_idx, bg)
        assert st is not None, "expected every lane to contain valid sprites"
        lane_states.append(st)

    lanes_by_color = {}
    for st in lane_states:
        assert st["color"] not in lanes_by_color, "lane colors must be unique for cue matching"
        lanes_by_color[st["color"]] = st

    src_colors = [color for _slot, color in source_marks]
    dst_color = target_marks[0][1]
    assert len(src_colors) == len(set(src_colors)), "source cue colors must be unique"
    assert dst_color in lanes_by_color, "target cue color must match a lane"

    dst = lanes_by_color[dst_color]
    src_states = []
    for color in src_colors:
        assert color in lanes_by_color, "source cue color must match a lane"
        src_states.append(lanes_by_color[color])
    assert all(src["lane_idx"] != dst["lane_idx"] for src in src_states), "source and target lanes must differ"

    out = g.copy()

    for src in src_states:
        src_sprites = src["sprites"]
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = _GREY

        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        dc_shift = int(round(dst["center"] - src["center"]))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        n_new = len(src_sprites)
        placed = 0
        k = 0
        while placed < n_new:
            new_top = dst["bottom0"] - k * dst["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in dst["occupied"]:
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
                out[new_top + dr, c] = dst["color"]
            dst["occupied"].add(new_top)
            placed += 1

    for slot_idx, _color in source_marks:
        out[_SOURCE_CUE_ROW, _LANE_CENTERS[slot_idx]] = _GREY
    for slot_idx, _color in target_marks:
        out[_TARGET_CUE_ROW, _LANE_CENTERS[slot_idx]] = _GREY

    return out


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green))
    return grouped


def _assign_cue_slots(role_lanes: list[int], rng: np.random.Generator) -> list[int]:
    candidates = []
    for perm in permutations(range(_NUM_LANES), len(role_lanes)):
        if all(int(slot) != int(lane_idx) for slot, lane_idx in zip(perm, role_lanes)):
            candidates.append(tuple(int(slot) for slot in perm))
    assert candidates, "expected at least one deranged cue-slot assignment"
    choice = candidates[int(rng.integers(len(candidates)))]
    return list(choice)


def _place_cues(
    inp: np.ndarray,
    out: np.ndarray,
    lane_colors: list[int],
    reds: tuple[int, ...],
    green: int,
    rng: np.random.Generator,
) -> None:
    role_lanes = list(reds) + [green]
    slots = _assign_cue_slots(role_lanes, rng)

    for slot_idx, lane_idx in zip(slots[: len(reds)], reds):
        col = _LANE_CENTERS[slot_idx]
        inp[_SOURCE_CUE_ROW, col] = lane_colors[lane_idx]
        out[_SOURCE_CUE_ROW, col] = _GREY

    target_col = _LANE_CENTERS[slots[-1]]
    inp[_TARGET_CUE_ROW, target_col] = lane_colors[green]
    out[_TARGET_CUE_ROW, target_col] = _GREY


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
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
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg

    _place_cues(inp, out, lane_colors, reds, green, rng)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
