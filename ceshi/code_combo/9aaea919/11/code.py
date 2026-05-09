# Combo variant of 9aaea919 (index 11)
# combo_spec: D4/2+D12/1
# Name: 双向颜色交换+多源多目标配对

"""
design_analysis:
  concrete_behavior_and_trace:
    - 保留原题的 30x30 画布、4 条固定 5 列车道、底部 1x5 标记条、3x5 十字精灵、period=4 的竖向晶格。
    - 融合后固定为 2 条红源车道 + 2 条绿目标车道；先按横向最近的一对一最优匹配把两组车道配成两对，再在每一对内部执行双向颜色交换。
    - 单对行为追踪：
      1. 输入里红车道颜色为 A、数量为 2；配到的绿车道颜色为 B、数量为 3。
      2. 原题会把红车道变灰，并在绿车道顶部补 2 个 B。
      3. 本融合版改为：红车道原位 2 个精灵改成 B；绿车道原有 3 个精灵改成 A；绿车道顶部新增 2 个精灵仍为 B。
    - 多对行为追踪：
      1. 若红车道在 x=5,11，绿车道在 x=17,23，则局部贪心会让两条红都想投向 x=17。
      2. many:many 规则要求全局一对一，因此必须拆成 (5->17, 11->23)。
      3. 每个目标车道都只和自己配到的源车道交换颜色并接收新增堆叠，避免多源同目标下“已有目标精灵到底换成谁的颜色”的歧义。
  complexity_calibration:
    - 只开放两条红和两条绿，控制匹配复杂度，同时通过“争抢同一目标”的布局让 many:many 差异稳定可见。
    - 四条车道颜色保持全异，因此每对交换后的双色分层肉眼可辨，不会因为颜色复用而弱化规则。
    - 继续限制每条输入车道 1..3 个精灵、每个目标输出总数 <= 6，保证画面紧凑但不拥挤。
    - 生成器强制两条红车道数量不同，这样错误配对会导致两个目标车道新增高度分布不同，视觉上能排除“随便配都一样”的退化。
  implicit_assumption_audit:
    - 原 solver 的 `best = min(...)` 默认每条红都能独立挑最近绿；融合后不成立，必须先做全局匹配，避免两个红共用一个绿。
    - 原 solver 默认目标车道旧精灵颜色不变；融合后不成立，必须在放置新增精灵前先把目标旧精灵整体改成配对源色。
    - 原 solver 默认源车道输出与目标颜色无关且可直接写灰；融合后不成立，源车道颜色来自配对目标的输入色。
    - 顺序依赖也发生变化：若边配对边修改 `occupied` 或边换色边读取颜色，会污染后续决策。因此新 solver 先从输入抽取全部车道状态，再配对，再写输出。
    - 新假设是“每个绿车道只对应一个红车道”，生成器通过固定 2 红 2 绿并使用一对一匹配来保证。
  placement_legality:
    - 每个目标车道只接收一个来源，因此新增精灵仍沿原题的底对齐、向上填充、period=4 空槽放置，不会与别的来源争槽位。
    - 生成器先计算每个绿车道输出总数 = 原数量 + 配对红数量，并限制总数 <= 6，保证最上方 top row 仍在网格内。
    - 反推输入时仅擦除输出中每个目标车道最上面的 gain 个槽位；这些槽位正是 solver 会重新放回的新增精灵。
  observability_and_cleanliness:
    - 人类可以同时看到两层结构线索：哪条红和哪条绿配对，由横向最近的一对一关系决定；配对后每个目标车道都变成“顶部保留目标色 + 底部旧精灵换成源色”的双色分层。
    - 所有变化都绑定在可见结构上：标记条、固定车道位置、已有堆叠与新增堆叠的上下分层。没有使用颜色编号大小、组件遍历顺序等不可视因素。
    - 车道之间留有背景间隔，标记条独占最后一行，输出里不存在重叠覆盖或多种规则挤在同一局部区域导致的脏画面。
"""

import numpy as np

from itertools import permutations

from arc_utils import bbox_coords, components_mask, get_bg, paste


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the requested color."""
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


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    items = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        items.append((rmin, cmin, pts))
    items.sort()
    return items


def _extract_lane_state(g: np.ndarray, marker: list[tuple[int, int]], bg: int) -> dict | None:
    """Collect geometry, color, and sprite-template data for one marked lane."""
    _, c0, _, c1 = bbox_coords(np.array(marker))
    cols = range(c0, c1 + 1)
    marker_set = set(marker)

    comps = _single_color_components_in_cols(g, cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return None

    non_marker_sizes = [len(pts) for pts in comps if len(pts) != len(marker)]
    if not non_marker_sizes:
        return None
    sprite_size = max(non_marker_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_size]
    if not sprites:
        return None

    sprites_sorted = _sorted_sprites(sprites)
    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        r0, _cc0, r1, _cc1 = bbox_coords(np.array(sprites_sorted[0][2]))
        period = (r1 - r0 + 1) + 1

    ref_top, _ref_left, ref_pts = sprites_sorted[0]
    ref_color = int(g[ref_pts[0][0], ref_pts[0][1]])
    sprite_h = max(r for r, _ in ref_pts) - ref_top + 1
    return {
        "marker": marker,
        "center": (c0 + c1) / 2.0,
        "color": ref_color,
        "sprites": [pts for _r0, _c0, pts in sprites_sorted],
        "tops": tops,
        "occupied": set(tops),
        "period": period,
        "bottom0": max(tops),
        "ref_top": ref_top,
        "ref_pts": ref_pts,
        "sprite_h": sprite_h,
        "lane_left": c0,
    }


def _choose_pairing(red_states: list[dict], green_states: list[dict]) -> list[tuple[dict, dict]]:
    """
    Minimum-total-distance one-to-one pairing.

    Ties are broken by the number of crossing assignments, then lexicographically, which
    yields the left-to-right non-crossing pairing on the fixed lane line.
    """
    reds = sorted(red_states, key=lambda st: st["center"])
    greens = sorted(green_states, key=lambda st: st["center"])
    assert len(reds) == len(greens) and reds, "expected equal non-empty red/green marker sets"

    best_key = None
    best_perm = None
    for perm in permutations(range(len(greens))):
        cost = sum(abs(reds[i]["center"] - greens[j]["center"]) for i, j in enumerate(perm))
        crossings = sum(int(perm[i] > perm[j]) for i in range(len(perm)) for j in range(i + 1, len(perm)))
        key = (cost, crossings, tuple(perm))
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm

    assert best_perm is not None
    return [(reds[i], greens[best_perm[i]]) for i in range(len(reds))]


def _place_transfer(out: np.ndarray, src: dict, dst: dict, bg: int) -> None:
    """Recreate every sprite from one source lane into the paired destination lane."""
    dc_shift = int(round(dst["center"] - src["center"]))
    src_offsets = [(r - src["ref_top"], c + dc_shift) for r, c in src["ref_pts"]]

    placed = 0
    needed = len(src["sprites"])
    step = 0
    while placed < needed:
        new_top = dst["bottom0"] - step * dst["period"]
        step += 1
        if new_top < 0:
            break
        if new_top + src["sprite_h"] > out.shape[0]:
            continue
        if new_top in dst["occupied"]:
            continue

        ok = True
        for dr, cc in src_offsets:
            rr = new_top + dr
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                ok = False
                break
            if int(out[rr, cc]) != bg:
                ok = False
                break
        if not ok:
            continue

        for dr, cc in src_offsets:
            out[new_top + dr, cc] = dst["color"]
        dst["occupied"].add(new_top)
        placed += 1

    assert placed == needed, "destination lane lacked enough empty lattice slots"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair red and green lanes globally, then perform per-pair color swaps plus transfers.

    Input structure:
    - The background is the dominant color; the last row contains exactly two red 1x5 bars and two green 1x5 bars.
    - Every marked lane is one of the fixed 5-column slots and contains repeated monochrome 3x5 plus sprites on a shared period-4 vertical lattice.

    Rule:
    - First match the two red lanes to the two green lanes by minimum-total horizontal distance, with left-to-right non-crossing tie-breaking.
    - Inside each matched pair, the source lane keeps its sprite count in place but recolors to the target lane's input color; the target lane's pre-existing sprites recolor to the source lane's input color; and copied source sprites fill higher empty slots in the target lane using the target color.
    - All marker bars are erased and every unmarked lane remains unchanged.

    Key insight:
    - The puzzle fuses a global assignment problem with a local two-way color exchange: you must determine which red lane belongs to which green lane before recoloring, because each target lane's lower stack changes to its matched source color while only the newly added upper stack keeps the target color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert len(red_markers) == 2 and len(green_markers) == 2, "expected exactly two red and two green markers"

    red_states = [_extract_lane_state(g, rm, bg) for rm in red_markers]
    green_states = [_extract_lane_state(g, gm, bg) for gm in green_markers]
    assert all(st is not None for st in red_states + green_states), "every marked lane must contain a valid sprite stack"
    red_states = [st for st in red_states if st is not None]
    green_states = [st for st in green_states if st is not None]

    pairs = _choose_pairing(red_states, green_states)
    out = g.copy()

    for src, dst in pairs:
        for pts in src["sprites"]:
            for r, c in pts:
                out[r, c] = dst["color"]
        for pts in dst["sprites"]:
            for r, c in pts:
                out[r, c] = src["color"]

    for src, dst in pairs:
        _place_transfer(out, src, dst, bg)

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
_MAX_LANE_TOTAL = 6
_GEN_RETRIES = 300

# Layouts chosen so the original per-red greedy rule would send both reds to the same green.
_ROLE_LAYOUTS = [
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((1, 3), (0, 2)),
    ((2, 3), (0, 1)),
]


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return the 3x5 plus/diamond sprite with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Top rows of a bottom-aligned stack, ordered from top to bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    """Stamp same-colored sprites into explicit lattice slots of one lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    _paint_lane_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _lane_center(lane_idx: int) -> float:
    return _LANE_STARTS[lane_idx] + (_SPRITE_W - 1) / 2.0


def _pair_roles(reds: tuple[int, int], greens: tuple[int, int]) -> dict[int, int]:
    red_states = [{"center": _lane_center(idx), "lane": idx} for idx in reds]
    green_states = [{"center": _lane_center(idx), "lane": idx} for idx in greens]
    pairs = _choose_pairing(red_states, green_states)
    return {src["lane"]: dst["lane"] for src, dst in pairs}


def _invert_pair_map(pair_map: dict[int, int]) -> dict[int, int]:
    return {green: red for red, green in pair_map.items()}


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    # Keep marker colors exclusive to the bars so role cues stay visually clean.
    sprite_palette = [c for c in range(10) if c not in (bg, 2, 3, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    reds, greens = _ROLE_LAYOUTS[int(rng.integers(len(_ROLE_LAYOUTS)))]
    pair_map = _pair_roles(reds, greens)
    reverse_pair_map = _invert_pair_map(pair_map)

    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    if input_counts[reds[0]] == input_counts[reds[1]]:
        return None

    output_counts = input_counts[:]
    for red_lane, green_lane in pair_map.items():
        output_counts[green_lane] += input_counts[red_lane]
    if any(output_counts[g] > _MAX_LANE_TOTAL for g in greens):
        return None

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx in reds:
            paired_green = pair_map[lane_idx]
            _paint_lane(out, lane_idx, lane_colors[paired_green], input_counts[lane_idx], bg)
        elif lane_idx in greens:
            paired_red = reverse_pair_map[lane_idx]
            gain = input_counts[paired_red]
            dst_out_tops = _lane_tops(output_counts[lane_idx])
            gained_tops = dst_out_tops[:gain]
            existing_tops = dst_out_tops[gain:]
            _paint_lane_tops(out, lane_idx, lane_colors[lane_idx], gained_tops, bg)
            _paint_lane_tops(out, lane_idx, lane_colors[paired_red], existing_tops, bg)
        else:
            _paint_lane(out, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    for green_lane in greens:
        paired_red = reverse_pair_map[green_lane]
        gain = input_counts[paired_red]
        dst_out_tops = _lane_tops(output_counts[green_lane])
        gained_tops = dst_out_tops[:gain]
        existing_tops = dst_out_tops[gain:]
        _paint_lane_tops(inp, green_lane, lane_colors[green_lane], existing_tops, bg)
        green_c0 = _LANE_STARTS[green_lane]
        for r0 in gained_tops:
            inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2
    for lane_idx in greens:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 3

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
