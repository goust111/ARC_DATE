# Variant of 9aaea919
# Dimension: D4 — 颜色变换规则 (transform)
# Change: color_transfer — 转移的精灵重新着色为目标车道的原有精灵颜色；源车道精灵变灰(5)... → 双向颜色交换

"""
design_analysis:
  variant_core:
    preserved_structure:
      - 30x30 画布、4 条固定竖直车道、底部 1x5 标记条、3x5 十字/菱形精灵、周期 4 的竖向堆叠全部保留。
      - 目标车道仍然接收与源车道精灵数量相同的新精灵，新精灵仍然占据目标车道原本向上延伸的空槽位。
    D4_change:
      - 原题里，源车道原位精灵统一变灰，目标车道原有精灵颜色不变，新增精灵使用目标车道颜色。
      - 变体里，标记交互改成一对一颜色交换：
        1. 源车道原位精灵改成目标车道输入时的颜色。
        2. 目标车道原有精灵改成源车道输入时的颜色。
        3. 新增到目标车道顶部空槽的转移精灵仍使用目标车道颜色。
      - 因为“颜色互换”天然要求成对关系，生成器收紧为恰好 1 条红源车道 + 1 条绿目标车道；否则多源对一目标会引入顺序依赖，不再是单纯 D4 改动。
  behavior_trace:
    example:
      input:
        - 源车道颜色 A，有 2 个精灵。
        - 目标车道颜色 B，有 3 个精灵。
      original_output:
        - 源车道 2 个精灵变灰。
        - 目标车道底部原有 3 个精灵仍为 B。
        - 目标车道上方新增 2 个精灵，也为 B。
      variant_output:
        - 源车道 2 个精灵改成 B。
        - 目标车道底部原有 3 个精灵改成 A。
        - 目标车道上方新增 2 个精灵为 B。
      visible_difference:
        - 目标车道输出不再是单色，而是“上层新增 B / 下层原有 A”的双色分层。
        - 源车道不再出现灰色，因此原题与变体在同一输入上的输出必然不同。
  complexity_calibration:
    - 只引入两个参与交换的颜色：源色与目标色；未标记车道完全不参与，避免颜色映射过多导致样例难以归纳。
    - 目标车道的双色结果由“是否原本就在目标车道里”这个空间线索区分，而不是靠颜色 id 或排序规则区分。
    - 保持 4 车道和 1-3 个精灵/车道的原规模，视觉上仍然接近原题，不会退化成过于简单的单对象交换题。
  implicit_assumption_audit:
    original_code_dependencies:
      - 原 solver 读取目标车道颜色一次后，把它只用于新增精灵着色；这依赖“目标车道已有精灵颜色不会变化”。
      - 原 solver 对源车道直接整片写 5；这依赖“源车道输出颜色与目标车道无关”。
      - 原 generator 输出端把整条红车道统一涂灰；这同样依赖旧的单向变灰规则。
    variant_fixes:
      - 新 solver 在放置新增精灵前，先分别重涂源车道已有精灵和目标车道已有精灵，避免把目标旧精灵留成旧色。
      - 新 generator 先构造双色目标车道输出，再反推输入时只擦掉“新增的那一段”，并把目标旧精灵恢复成输入时的目标色。
      - solver 明确断言只有 1 红 + 1 绿标记，生成器也保证这一点，消除多源交换的隐式歧义。
  placement_legality:
    - 沿用原题的 5 列车道槽位和 period=4 竖向晶格；新增精灵仍只会填入该晶格中的空槽。
    - 生成器继续限制目标车道输入计数 + 源车道计数 <= 7，保证最上方新增精灵不会越界。
    - 反推输入时仅擦除输出总堆叠中最上方 gain 个槽位，因为这些正是 solver 会回填的新精灵位置。
  observability_and_cleanliness:
    - 人类可直接看到：红标记车道颜色变成绿标记车道原色，绿标记车道底部原有堆叠变成红标记车道原色。
    - 目标车道双色且按上下分层，避免对象重叠混色；源/目标/未标记区域依旧严格分列，画面整洁。
    - 不使用颜色编号参与匹配、排序或分支；所有判断都由标记条、车道位置和“原有/新增槽位”决定。
"""

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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


def _extract_lane_state(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict:
    """Read one marked lane from the input grid."""
    _, c0, _, c1 = bbox_coords(np.array(marker))
    cols = range(c0, c1 + 1)
    marker_set = set(marker)
    comps = _single_color_components_in_cols(g, cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    assert comps, "expected marked lane to contain sprites"

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker)]
    assert sprite_sizes, "expected non-marker sprite components inside marked lane"
    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    assert sprites, "expected at least one full sprite in marked lane"

    sprites_sorted = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        sprites_sorted.append((rmin, cmin, pts))
    sprites_sorted.sort()

    ref_r0, _ref_c0, ref_pts = sprites_sorted[0]
    color = int(g[ref_pts[0][0], ref_pts[0][1]])
    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(ref_pts))
        period = (r1 - r0 + 1) + 1

    return {
        "marker": marker,
        "center": (c0 + c1) / 2.0,
        "color": color,
        "sprites": sprites,
        "sprites_sorted": sprites_sorted,
        "tops": tops,
        "occupied": set(tops),
        "period": period,
        "bottom0": max(tops),
        "ref_r0": ref_r0,
        "ref_pts": ref_pts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move source-lane sprite copies into the target lane while swapping the pair's colors.

    Input structure:
    - The background is the most common color. One red 1x5 bar marks the source lane and one green 1x5 bar marks the destination lane.
    - Each lane is a fixed 5-column slot containing evenly spaced copies of the same monochrome 3x5 plus sprite.

    Rule:
    - The source lane keeps its sprite count in place but its sprites recolor to the destination lane's input color.
    - The destination lane's pre-existing sprites recolor to the source lane's input color, while equally many transferred copies fill higher empty lattice slots in the destination color.
    - All marker bars are removed and every unmarked lane stays unchanged.

    Key insight: The task still conserves sprite count like the original lane-transfer puzzle, but now the marked pair also performs a two-way color swap, so the solver must distinguish destination sprites that were already present from the newly added stack.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert len(red_markers) == 1 and len(green_markers) == 1, "expected exactly one red and one green marker bar"

    src = _extract_lane_state(g, red_markers[0], bg)
    dst = _extract_lane_state(g, green_markers[0], bg)

    out = g.copy()

    # Bidirectional color swap for the two pre-existing marked stacks.
    for pts in src["sprites"]:
        for r, c in pts:
            out[r, c] = dst["color"]
    for pts in dst["sprites"]:
        for r, c in pts:
            out[r, c] = src["color"]

    # Transfer copies of the source sprite shape into the destination lane.
    dc_shift = int(round(dst["center"] - src["center"]))
    src_offsets = [(r - src["ref_r0"], c + dc_shift) for r, c in src["ref_pts"]]
    sprite_h = max(dr for dr, _c in src_offsets) + 1

    n_new = len(src["sprites"])
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

    assert placed == n_new, "expected every source sprite to transfer into the destination lane"

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

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    """Stamp sprites of one color into specific top-row slots of a lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given lane."""
    _paint_lane_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _feasible_pairs(counts: list[int]) -> list[tuple[int, int]]:
    """Return all feasible (red_source, green_destination) lane pairs."""
    pairs = []
    for red in range(_NUM_LANES):
        for green in range(_NUM_LANES):
            if red == green:
                continue
            if counts[red] + counts[green] <= _MAX_LANE_TOTAL:
                pairs.append((red, green))
    return pairs


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c != bg]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_pairs(input_counts)
    if not feasible:
        return None
    red, green = feasible[int(rng.integers(len(feasible)))]
    gain = input_counts[red]

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx not in (red, green):
            _paint_lane(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    # Source lane output: same count, recolored to the destination color.
    _paint_lane(out, red, lane_colors[green], input_counts[red], bg)

    # Destination lane output: top gained slots use destination color; original slots swap to source color.
    dst_out_tops = _lane_tops(input_counts[green] + gain)
    gained_tops = dst_out_tops[:gain]
    existing_tops = dst_out_tops[gain:]
    _paint_lane_tops(out, green, lane_colors[green], gained_tops, bg)
    _paint_lane_tops(out, green, lane_colors[red], existing_tops, bg)

    # Derive input by restoring the original colors and erasing the transferred copies.
    inp = out.copy()
    _paint_lane(inp, red, lane_colors[red], input_counts[red], bg)
    _paint_lane_tops(inp, green, lane_colors[green], existing_tops, bg)
    green_c0 = _LANE_STARTS[green]
    for r0 in gained_tops:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg

    inp[_MARKER_ROW, _LANE_STARTS[red] : _LANE_STARTS[red] + _SPRITE_W] = 2
    inp[_MARKER_ROW, _LANE_STARTS[green] : _LANE_STARTS[green] + _SPRITE_W] = 3

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
