# Variant of 9aaea919
# Dimension: D9 — 车道数量与布局 (meta)
# Change: 固定4条车道，列中心分别在5、11、17、23，间距6列... → 可变车道数量(3-6条)

# Design analysis
# ----------------
# variant_spec:
#   core_change:
#     - 保持 30x30 画布、3x5 十字/菱形精灵、底部向上按 period=4 的堆叠方式不变。
#     - 唯一核心变化是横向车道布局：车道数不再固定为 4，而是在 3..6 之间变化。
#     - 每条车道仍然占用固定的 5 列窗口，红/绿标记条仍是与车道同宽的 1x5 横条。
#   concrete_layout:
#     - 先放下 n 条 5 列车道，占用 5*n 列。
#     - 剩余列数 extra = 30 - 5*n 按“内部间隙优先、再给左右边距”的顺序轮转分配。
#     - 因而 3 条车道时间距明显变大，5 条车道时间距变小，6 条车道时车道会彼此紧贴但不重叠。
#   behavior_trace:
#     - 原逻辑的小例子：固定 4 条车道时，红条在第 1 车道、绿条在第 3 车道，solver 直接把第 1
#       车道中的全部精灵变灰，并把同数量的新精灵补到第 3 车道底部上方的空槽。
#     - 变体后的对应例子：如果只有 3 条车道，start 列可能是 [4, 13, 22]；如果有 6 条车道，
#       start 列可能是 [0, 5, 10, 15, 20, 25]。solver 不再能从固定列中心猜车道，只能从
#       标记条本身定位源/目标车道，然后沿各自的 5 列窗口做同样的转移。
#     - 差异点不在转移规则，而在“先判断当前有几条车道、它们横向如何分布”这一层输入结构分析。
#   complexity_check:
#     - 只开放一个新的自由度：车道数 3..6。
#     - 精灵形状、颜色规则、纵向周期、红/绿角色机制都保持原样，避免多维变化叠加造成歧义。
#     - 车道数变化会直接改变整幅图的横向密度，肉眼可见，不依赖任何不可见的颜色 id 排序。
#   implicit_assumption_audit:
#     - _find_marker_bars:
#       原代码接受任意细长同色连通块；在 6 车道紧贴时，如果两个红条相邻，它们会粘成 1x10，
#       旧写法会把整块误判为一个车道标记。变体中改为只接受精确的 1x5 横条，且生成器禁止
#       “无间隙相邻车道同时为红标记”。
#     - 目标车道状态提取:
#       依赖前提是目标车道至少保留一个原始精灵，用它推断目标颜色和周期。生成器保持每条车道
#       输入中原始精灵数 >= 1，因此绿车道总能被识别。
#     - 新精灵放置:
#       依赖前提是 green_count + transferred_count <= 7，才能保持顶部不越界。生成器继续强制
#       这个上界。
#     - 水平位移:
#       依赖前提是车道窗口宽度恒为 5，源模板只做整数列平移。变体不改变精灵宽度和标记宽度，
#       所以前提仍成立。
#   placement_legality:
#     - 所有车道窗口都是互不重叠的 5 列区间；6 车道时允许相邻贴边，但不会共用列。
#     - 标记条放在最后一行，精灵最底部结束于第 27 行，中间第 28 行留空，保证标记与精灵不连通。
#     - 输出先构造，再反推输入，最后用 main(inp) == out 做 oracle 检查。
#   visual_observability:
#     - 3 条车道样例会有大空隙，6 条车道样例会显得拥挤，4/5 条介于中间；这种差异在 2-3 个样例
#       上足以让解题者看出“车道数量不是常数”。
#     - 颜色变化仍只有“红标记源车道变灰、目标车道补入目标色”这一个结构化规则，可从位置关系中推出。

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """
    Marker bars are exact horizontal 1x5 components of the given color.
    """
    bars = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        if (r1 - r0 + 1, c1 - c0 + 1, len(pts_arr)) == (1, 5, 5):
            pts = [(int(r), int(c)) for r, c in pts_arr]
            bars.append((r0, c0, pts))
    bars.sort()
    return [pts for _r0, _c0, pts in bars]


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


def _lane_sprites_from_marker(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> tuple[list[list[tuple[int, int]]], float] | tuple[None, float]:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return None, (c0 + c1) / 2.0

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return None, (c0 + c1) / 2.0
    sprite_size = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_size]
    return sprites if sprites else None, (c0 + c1) / 2.0


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    tagged = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        tagged.append((rmin, cmin, pts))
    tagged.sort()
    return tagged


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every sprite stack from red-tagged lanes into the green-tagged lane, even when the number of lanes changes.

    Input structure:
    - The background is the most common color. Lanes are not fixed at four columns anymore; instead there are 3-6
      lane windows, each recognized by a 1x5 marker bar aligned with that lane.
    - Inside each lane window, identical monochrome plus/diamond sprites occupy a regular vertical lattice.

    Rule:
    - Every sprite in a red-marked source lane is recolored grey in place.
    - The same number of sprites is recreated inside the green-marked destination lane, using the destination lane's
      own color and filling empty lattice slots from the bottom upward.
    - After the transfer, all red and green marker bars disappear back into the background.

    Key insight:
    - The marker bars define the active lane windows directly, so the transfer rule stays the same even though the
      total lane count and the horizontal spacing between lanes now vary from puzzle to puzzle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = []
    for gm in green_markers:
        dst_sprites, gc = _lane_sprites_from_marker(g, gm, bg)
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

        green_state.append(
            {
                "center": gc,
                "dst_color": dst_color,
                "period": period,
                "occupied": set(tops),
                "bottom0": max(tops),
            }
        )

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        src_sprites, rc = _lane_sprites_from_marker(g, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

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

    for marker_group in (red_markers, green_markers):
        for pts in marker_group:
            for r, c in pts:
                out[r, c] = bg

    return out


# --------------- generator ---------------

# --- CFG: canvas & sprite geometry ---
_H, _W = 30, 30
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: lane-count variant ---
_MIN_LANES = 3
_MAX_LANES = 6

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 400


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 3x5 plus/diamond sprite patch with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_layout(num_lanes: int) -> tuple[list[int], list[int], tuple[int, int]]:
    """
    Return lane starts, interior gaps, and (left_margin, right_margin) for `num_lanes`.

    Extra columns are distributed round-robin across interior gaps first, then the two outer margins,
    so lower lane counts visibly spread out while higher counts compress.
    """
    assert _MIN_LANES <= num_lanes <= _MAX_LANES
    used = num_lanes * _SPRITE_W
    extra = _W - used
    assert extra >= 0

    gaps = [0] * (num_lanes - 1)
    left_margin = 0
    right_margin = 0
    slots: list[tuple[str, int | None]] = [("gap", i) for i in range(num_lanes - 1)] + [
        ("left", None),
        ("right", None),
    ]
    for k in range(extra):
        kind, idx = slots[k % len(slots)]
        if kind == "gap":
            gaps[int(idx)] += 1
        elif kind == "left":
            left_margin += 1
        else:
            right_margin += 1

    starts = [left_margin]
    for gap in gaps:
        starts.append(starts[-1] + _SPRITE_W + gap)
    assert starts[-1] + _SPRITE_W <= _W - right_margin
    return starts, gaps, (left_margin, right_margin)


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_start: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the lane whose window begins at `lane_start`."""
    patch = _sprite_patch(color, bg)
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, lane_start, transparent=bg)


def _touching_neighbor_pairs(starts: list[int]) -> set[tuple[int, int]]:
    """
    Return adjacent lane-index pairs whose 5-wide windows touch without any background column.
    """
    pairs = set()
    for i in range(len(starts) - 1):
        if starts[i + 1] == starts[i] + _SPRITE_W:
            pairs.add((i, i + 1))
    return pairs


def _feasible_roles(
    counts: list[int], starts: list[int]
) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    touching = _touching_neighbor_pairs(starts)
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    num_lanes = len(counts)
    for green in range(num_lanes):
        others = [i for i in range(num_lanes) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) > _MAX_LANE_TOTAL:
                    continue
                if red_count == 2 and tuple(sorted(reds)) in touching:
                    continue
                grouped[red_count].append((reds, green))
    return grouped


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    num_lanes = int(rng.integers(_MIN_LANES, _MAX_LANES + 1))
    lane_starts, _gaps, _margins = _lane_layout(num_lanes)

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=num_lanes, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=num_lanes).tolist()

    feasible = _feasible_roles(input_counts, lane_starts)
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
    for lane_idx, lane_start in enumerate(lane_starts):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_start, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_starts[lane_idx], lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = lane_starts[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = lane_starts[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
