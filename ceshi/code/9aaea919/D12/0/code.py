# Variant of 9aaea919
# Dimension: D12 — 源-目标基数关系 (meta)
# Change: many:1 — 1-2个红色源车道转移到1个绿色目标车道... → 1:many — 一个源多个目标
#
# Design analysis
# ===============
# variant_spec:
#   concrete_layout:
#     - 保持 30x30 画布、4 条固定车道、3x5 十字精灵、底部向上 period=4 的堆叠方式不变。
#     - 输入里恰好 1 条红色标记车道(源)，恰好 2 条绿色标记车道(目标)，剩余 1 条车道不标记。
#     - 输出里红车道仍灰化；两个绿车道都增长。左侧绿车道拿 ceil(N/2) 个，右侧绿车道拿 floor(N/2) 个，其中 N 是唯一红车道里的精灵数。
#   behavior_trace:
#     - 例子: 红车道有 3 个精灵，左右两个绿车道各已有 2 个和 1 个精灵。
#     - 原题逻辑: 这 3 个精灵会整体并到某一个最近的绿车道，另一个绿车道完全不变。
#     - 本变体逻辑: 红车道 3 个先变灰；左绿车道新增 2 个，右绿车道新增 1 个，且都按各自车道的底部栅格向上补空位。
#     - 差异点: “计数守恒”不再通过单一目标体现，而是通过左右两个目标的同步增长体现。
#   complexity_calibration:
#     - 精确固定为 1 红 + 2 绿，避免出现 1 红 + 1 绿时退化回原题，也避免 3 个以上目标导致人类很难从 2-4 个样例归纳拆分规则。
#     - 生成器强制红车道输入精灵数为 2 或 3，这样两个绿车道都会实际收到精灵；当 N=3 时还能稳定展示“奇数时左侧多一个”。
#   implicit_assumption_audit:
#     - 原 `main()` 的 `assert red_markers and green_markers` 太弱；旧题允许多个红源和单个绿目标，新变体要求正好 1 红 2 绿。
#     - 原 `best = min(green_state.values(), key=...)` 依赖“所有转移都进同一个目标”；在新变体里这会错误地把全部精灵塞进一个绿车道。
#     - 原逻辑逐个红车道更新目标占用集合；在新变体里只有一个红源，因此必须先算出左右分配数，再分别放到两个目标里，避免处理顺序偷走另一个目标应得的份额。
#     - 原 generator 只校验“某一个绿车道总数不超上限”；新 generator 必须分别校验两个绿车道各自的最终容量。
#   placement_legality:
#     - 每条车道仍只在自己的 5 列槽位内作画，精灵之间 vertical period 固定为 4，不会相互重叠。
#     - 两个绿车道在输入中都保留至少 1 个原有精灵，因此 solver 总能从现有目标精灵反推出颜色和 period。
#     - 红车道精灵数限制为 2-3，配合绿车道原始 1-3 个精灵，两个目标最终都不会超过 5 个，远小于 7 个可用槽位上限。
#   visual_observability:
#     - 输入中同时出现两个绿色底部标记条，输出中两个对应车道都增长，这是肉眼可见的核心差异。
#     - 奇数案例会出现左目标比右目标多长出 1 个精灵，且“左/右”只由空间位置决定，不依赖颜色 id 或隐藏排序。
#   cleanliness_and_color_reasoning:
#     - 仍只使用背景色、4 条车道色、红/绿标记、灰色，总颜色复杂度与原题接近。
#     - 所有排序只依据车道的水平位置(center x)，不依据颜色编号，因此规则完全可见。
#   post_change_solver_assumptions_enforced_by_generator:
#     - 恰好 1 个红条、2 个绿条。
#     - 每个标记车道都至少有 1 个可识别的单色十字精灵。
#     - 两个绿目标按从左到右排序后执行 ceil/floor 拆分；生成器和 solver 使用同一约定。

import sys
from itertools import combinations

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_coords, components_mask, get_bg, paste


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
        colors_in = {int(g[r, c]) for r, c in pts}
        if len(colors_in) == 1:
            out.append(pts)
    return out


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    ranked = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        ranked.append((rmin, cmin, pts))
    ranked.sort()
    return ranked


def _lane_state_from_marker(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict:
    """Extract the repeated sprite pattern and stacking metadata for one marked lane."""
    _, c0, _, c1 = bbox_coords(np.array(marker))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker)
    lane_comps = _single_color_components_in_cols(g, lane_cols, bg)
    lane_comps = [[p for p in pts if p not in marker_set] for pts in lane_comps]
    lane_comps = [pts for pts in lane_comps if pts]
    assert lane_comps, "expected marked lane to contain sprites"

    sprite_sizes = [len(pts) for pts in lane_comps if len(pts) != len(marker)]
    assert sprite_sizes, "expected sprite-sized components in marked lane"
    sprite_size = max(sprite_sizes)
    sprites = [pts for pts in lane_comps if len(pts) == sprite_size]
    assert sprites, "expected at least one valid sprite in marked lane"

    sprites_sorted = _sorted_sprites(sprites)
    sample_pt = sprites_sorted[0][2][0]
    lane_color = int(g[sample_pt[0], sample_pt[1]])

    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    return {
        "marker": marker,
        "center": (c0 + c1) / 2.0,
        "color": lane_color,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites_sorted": sprites_sorted,
    }


def _split_transfer(n_sprites: int) -> tuple[int, int]:
    """Left target gets the larger half when the source count is odd."""
    return (n_sprites + 1) // 2, n_sprites // 2


def _place_cloned_sprites(
    out: np.ndarray,
    lane_state: dict,
    src_center: float,
    src_ref_r0: int,
    src_ref_pts: list[tuple[int, int]],
    n_new: int,
    bg: int,
) -> None:
    if n_new <= 0:
        return

    h, w = out.shape
    dc_shift = int(round(lane_state["center"] - src_center))
    src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
    sprite_h = max(dr for dr, _c in src_offsets) + 1

    placed = 0
    k = 0
    while placed < n_new:
        new_top = lane_state["bottom0"] - k * lane_state["period"]
        k += 1
        if new_top < 0 or new_top + sprite_h > h:
            if new_top < 0:
                break
            continue
        if new_top in lane_state["occupied"]:
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
            out[new_top + dr, c] = lane_state["color"]
        lane_state["occupied"].add(new_top)
        placed += 1

    assert placed == n_new, "expected enough room to place all transferred sprites"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split one red-marked source lane across two green-marked target lanes.

    Input structure:
    - The background is the most common color. Thin 1x5 bars on the last row mark one red source lane and two green target lanes.
    - Every lane contains repeated identical monochrome plus sprites aligned to the same vertical lattice.

    Rule:
    - All sprites in the single red lane turn grey.
    - Let N be the number of red-lane sprites. The two green lanes are ordered by horizontal position; the left one receives ceil(N/2) new sprites and the right one receives floor(N/2).
    - Each received sprite copies the source shape, is recolored to that target lane's own color, and is stacked into empty lattice slots from the bottom upward. All marker bars are then erased.

    Key insight:
    - Sprite count is still conserved, but the conserved count is now split spatially across two destinations: the left green lane gets the extra sprite when the source count is odd.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert len(red_markers) == 1, "expected exactly one red source lane"
    assert len(green_markers) == 2, "expected exactly two green target lanes"

    src_state = _lane_state_from_marker(g, red_markers[0], bg)
    green_states = [_lane_state_from_marker(g, gm, bg) for gm in green_markers]
    green_states.sort(key=lambda st: st["center"])

    out = g.copy()

    src_sprites = [pts for _rmin, _cmin, pts in src_state["sprites_sorted"]]
    for pts in src_sprites:
        for r, c in pts:
            out[r, c] = 5

    src_ref_r0, _src_ref_c0, src_ref_pts = src_state["sprites_sorted"][0]
    left_gain, right_gain = _split_transfer(len(src_sprites))
    _place_cloned_sprites(out, green_states[0], src_state["center"], src_ref_r0, src_ref_pts, left_gain, bg)
    _place_cloned_sprites(out, green_states[1], src_state["center"], src_ref_r0, src_ref_pts, right_gain, bg)

    for marker in red_markers + green_markers:
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
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MIN_SOURCE_SPRITES = 2
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int]) -> list[tuple[int, tuple[int, int]]]:
    """Return feasible (red_source, (left_green, right_green)) assignments."""
    roles = []
    for red in range(_NUM_LANES):
        if counts[red] < _MIN_SOURCE_SPRITES:
            continue
        others = [i for i in range(_NUM_LANES) if i != red]
        left_gain, right_gain = _split_transfer(counts[red])
        for greens in combinations(others, 2):
            left_green, right_green = sorted(greens)
            if counts[left_green] + left_gain > _MAX_LANE_TOTAL:
                continue
            if counts[right_green] + right_gain > _MAX_LANE_TOTAL:
                continue
            roles.append((red, (left_green, right_green)))
    return roles


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    if not feasible:
        return None
    red, greens = feasible[int(rng.integers(len(feasible)))]
    left_green, right_green = greens
    left_gain, right_gain = _split_transfer(input_counts[red])

    output_counts = input_counts[:]
    output_counts[left_green] += left_gain
    output_counts[right_green] += right_gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx == red else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    _paint_lane(inp, red, lane_colors[red], input_counts[red], bg)

    for green_lane, gain in ((left_green, left_gain), (right_green, right_gain)):
        c0 = _LANE_STARTS[green_lane]
        for r0 in _lane_tops(output_counts[green_lane])[:gain]:
            inp[r0 : r0 + _SPRITE_H, c0 : c0 + _SPRITE_W] = bg
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 3

    red_c0 = _LANE_STARTS[red]
    inp[_MARKER_ROW, red_c0 : red_c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
