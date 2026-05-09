# Variant of 9aaea919
# Dimension: D1 — 匹配条件机制 (match_cond)
# Change: marker_pixels — 红色(2)标记条标识源车道，绿色(3)标记条标识目标车道，通过标记颜色直接指定源和目标... → 颜色匹配替代标记

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis
# variant_spec:
#   source_reference:
#     task: 2c181942
#     borrowed_mechanism: "空间角色不再由固定语义颜色直接给出，而是由'位置角色 + 颜色匹配'联合决定"
#     complexity_target:
#       - 参考题用颜色去匹配 hub 边与外部精灵
#       - 本变体保留原题的固定四车道与堆叠规则，只把“谁是源/谁是目标”的判定换成颜色匹配，避免额外引入旋转/形变复杂度
#   a_concretization_and_trace:
#     input_geometry:
#       grid: 30x30
#       lanes:
#         count: 4
#         lane_cols: [3..7], [9..13], [15..19], [21..25]
#         sprites: 每车道 1-3 个同色十字精灵，颜色两两不同
#       cues:
#         source: 左边缘 col=0 上放 1-2 个 1x1 色块
#         target: 右边缘 col=29 上放 1 个 1x1 色块
#         cue_rows: 从固定候选行中抽取，但与 lane index 无直接对齐语义
#         cue_color_rule: 色块颜色必须等于某个车道当前精灵颜色
#       output:
#         所有边缘色块消失；源车道精灵变灰；目标车道得到相同数量的新精灵，自底向上填充
#     behavior_trace:
#       old:
#         - row 29 上直接看到红条/绿条
#         - solver 先按 marker 位置得到源/目标 lane
#       new:
#         - 先在左右边缘找到色块
#         - 左侧色块的“位置角色”说明这些颜色对应源车道；右侧色块说明其颜色对应目标车道
#         - 再把色块颜色与四个车道的精灵颜色做一一匹配
#       concrete_example:
#         - 四条车道颜色分别为 [8, 2, 4, 7]
#         - 左边缘出现颜色 2 和 7 的小块，右边缘出现颜色 4 的小块
#         - 则 lane(color=2) 与 lane(color=7) 是源，lane(color=4) 是目标
#         - 之后搬运/变灰/堆叠规则与原题完全一致
#   b_complexity_calibration:
#     why_not_too_simple:
#       - 不再能靠颜色 id=2/3 直接解题，必须先识别“左=源、右=目标”，再做颜色到车道的匹配
#       - 同一输入里保留 4 个不同车道颜色，且只有 1-2 个左 cue、1 个右 cue，解题者必须比较颜色对应关系
#     ambiguity_control:
#       - 车道颜色强制互异，防止一个 cue 匹配多个车道
#       - cue 只放在左右边缘，不放底边，视觉上与原 marker bar 明显不同
#       - cue 仅 1x1，数量少，避免和主体图形混杂
#   c_implicit_assumption_audit_on_original_main:
#     - old `_find_marker_bars(...)` 假设颜色 2/3 的细条存在；变体下不成立，必须删除这一路径
#     - old `bbox(marker)` 用 marker 的列范围当 lane 列范围；变体下没有 marker，因此必须改为使用原 spec 的固定 lane 槽位
#     - old 通过 `src_set/dst_set` 从连通块中剔除 marker 像素；变体下 lane 内已无 marker，这个前提消失，lane 组件提取可以简化
#     - old “nearest green lane” 在原题几乎等价于“唯一 green lane”；变体下右侧只允许一个目标 cue，因此应显式唯一化，而不是保留不必要的最近邻分支
#     - old 清除 marker bar 的逻辑假设 marker 在 row 29；变体下要改为清除左右边缘 cue 像素
#   d_placement_legality:
#     - lane 仍使用固定 5 列宽槽位，sprite 周期仍为 4，底部 top row 仍为 25，因此不会互相重叠
#     - cue 只占用 col 0 或 col 29；lane 最靠边的列分别是 3 和 25，左右都留有至少 3 列空白，不会接触主体
#     - 左 cue 行位从候选集不重复抽样；右 cue 行位从剩余候选集中抽样，避免视觉上形成左右一一配对
#     - 目标车道总数继续受 `_MAX_LANE_TOTAL` 约束，保证自底向上的新精灵都在网格内
#   e_visual_observability:
#     - 人类从少量样例可观察到：底部不再有红/绿条，改为左右边缘出现少量彩色点
#     - 输入输出共同支持“左侧对应源、右侧对应目标；颜色与车道颜色相同”这一归纳
#     - 未使用任何颜色 id 排序或不可见数值规则
#   f_visual_cleanliness:
#     - cue 与主体严格分区：主体在中间四条竖直 lane，cue 在最左/最右单列
#     - 输出移除 cue，画面比输入更干净，变换结果清晰
#   g_color_inference:
#     - 颜色变化只发生在原题已有的“源变灰 / 目标保留目标色”路径上
#     - 新增颜色线索只有“边缘 cue 颜色 = 对应车道颜色”，数量最多 3 个，结构性很强
#   post_change_main_assumptions_checked_against_generator:
#     - main 假设左右边缘非背景像素只会是 cue；generator 保证四周其他位置保持背景
#     - main 假设四个 lane 都含至少一个 sprite 且颜色互异；generator 保证 count>=1 且颜色不重复
#     - main 假设右 cue 只有一个、左 cue 有 1-2 个且颜色都能在 lane 中唯一匹配；generator 按该规则构造


_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300
_LEFT_CUE_COL = 0
_RIGHT_CUE_COL = _W - 1
_CUE_ROWS = [4, 10, 16, 22, 28]


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


def _lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict | None:
    c0 = _LANE_STARTS[lane_idx]
    comps = _single_color_components_in_cols(g, range(c0, c0 + _SPRITE_W), bg)
    if not comps:
        return None

    sprite_sz = max(len(p) for p in comps)
    sprites = [p for p in comps if len(p) == sprite_sz]
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
        r0, c0_box, r1, c1_box = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    center = c0 + (_SPRITE_W - 1) / 2.0
    return {
        "lane_idx": lane_idx,
        "center": center,
        "color": color,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites": [pts for _rmin, _cmin, pts in sprites_sorted],
    }


def _edge_cues(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    left = [(int(r), int(g[r, _LEFT_CUE_COL])) for r in range(h) if int(g[r, _LEFT_CUE_COL]) != bg]
    right = [(int(r), int(g[r, w - 1])) for r in range(h) if int(g[r, w - 1]) != bg]
    return left, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from cue-matched source lanes into a cue-matched target lane.

    Input structure:
    - bg is the dominant color. Four fixed vertical lanes contain repeated monochrome plus sprites on a regular vertical lattice.
    - Small edge cues replace the old bottom bars: 1-2 cue pixels on the left edge mark source lane colors, and 1 cue pixel on the right edge marks the target lane color.

    Rule:
    - Match each cue color to the unique lane whose sprites have that color.
    - Every left-cued lane is a source: its sprites turn grey (5), and that many copies of the sprite are added into the right-cued lane.
    - New target sprites use the target lane color and fill empty lattice slots upward from the bottom. All edge cues disappear.

    Key insight:
    - The cue position gives the role (left = source, right = target), while the cue color identifies which lane plays that role.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    left_cues, right_cues = _edge_cues(g, bg)
    left_colors = [color for _row, color in left_cues]
    right_colors = [color for _row, color in right_cues]
    assert 1 <= len(left_colors) <= 2, "expected 1-2 left-edge source cues"
    assert len(right_colors) == 1, "expected exactly one right-edge target cue"
    assert len(left_colors) == len(set(left_colors)), "left cue colors must be unique"

    lane_states = []
    for lane_idx in range(_NUM_LANES):
        st = _lane_state(g, lane_idx, bg)
        assert st is not None, "expected every lane to contain valid sprites"
        lane_states.append(st)

    lanes_by_color = {}
    for st in lane_states:
        assert st["color"] not in lanes_by_color, "lane colors must be unique for cue matching"
        lanes_by_color[st["color"]] = st

    assert right_colors[0] in lanes_by_color, "target cue color must match a lane"
    dst = lanes_by_color[right_colors[0]]
    src_states = []
    for color in left_colors:
        assert color in lanes_by_color, "source cue color must match a lane"
        src_states.append(lanes_by_color[color])
    assert all(st["lane_idx"] != dst["lane_idx"] for st in src_states), "source and target lanes must differ"

    out = g.copy()

    for src in src_states:
        src_sprites = src["sprites"]

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = _GREY

        rc = src["center"]
        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        dc_shift = int(round(dst["center"] - rc))
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

    for r, _color in left_cues:
        out[r, _LEFT_CUE_COL] = bg
    for r, _color in right_cues:
        out[r, _RIGHT_CUE_COL] = bg

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


def _place_cues(inp: np.ndarray, lane_colors: list[int], reds: tuple[int, ...], green: int, bg: int, rng: np.random.Generator) -> None:
    del bg  # cues are written directly; bg kept only for signature symmetry
    left_rows = rng.choice(_CUE_ROWS, size=len(reds), replace=False).tolist()
    remaining_rows = [r for r in _CUE_ROWS if r not in left_rows]
    right_row = int(rng.choice(remaining_rows))

    for row, lane_idx in zip(left_rows, reds):
        inp[int(row), _LEFT_CUE_COL] = lane_colors[lane_idx]
    inp[right_row, _RIGHT_CUE_COL] = lane_colors[green]


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

    _place_cues(inp, lane_colors, reds, green, bg, rng)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
