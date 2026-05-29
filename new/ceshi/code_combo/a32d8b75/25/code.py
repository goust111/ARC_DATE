# Combo variant of a32d8b75 (index 25)
# combo_spec: D1/1+D3/1+D11/2
# Name: 颜色匹配+多色框架+框架色映射

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, neighbors8
from scipy.ndimage import label as _scipy_label

# Design analysis
# a) 具体化与行为追踪
#    - 保留原题的宏观布局: 左/右 legend、6 色分隔列、中央 canvas、底部 5x5 锚点方框与 3x3 旋转时钟都不变。
#    - 融合后的唯一控制信号是 frame 内部的两种颜色 8 和 9。它们同时决定三件事:
#      1. 选哪种 unit variant:
#         8 -> 直接使用原始 unit；
#         9 -> 先把 unit 的两种颜色互换。
#      2. 用哪种铺贴朝向:
#         8 -> 正向；
#         9 -> 水平镜像。
#      3. 如何做颜色映射:
#         对被选中的 variant，取其“首个可见颜色/左上可见色”作为待迁移色 A，
#         把 A 改成当前 frame cell 的颜色（8 或 9），另一种颜色保持不变。
#    - 微型例子:
#      raw unit = [[1,2],[1,2]]
#      frame    = [[8,9],
#                  [9,8]]
#      原题: 四个格都贴同一个 unit（或固定交换后的 unit）。
#      融合后:
#        color 8 -> raw unit, 1->8, no mirror      => [[8,2],[8,2]]
#        color 9 -> swapped unit, 2->9, mirror LR  => [[1,9],[1,9]]
#      差异是单个 frame color 同时改变了“用哪个双色版本”“是否镜像”“哪种颜色被替成框架色”。
# b) 复杂度校准
#    - 只保留两种 frame 语义色 8/9，不再为每个 legend 随机发明新的语义颜色，避免可归纳性崩掉。
#    - 两种语义色在每个 frame 中都强制重复出现，且不是孤立单点；解题者能在同一复合体内看到重复支撑。
#    - unit 仍是完整填满的双色方阵，但 generator 额外要求水平不对称，确保“mirror”不是空操作。
# c) 隐式假设审查
#    - 原 main() 把 frame 压成布尔 mask；融合后这会丢掉全部关键信息，因此 solver 必须保留 frame 的彩色网格。
#    - 原 `_extract_unit_sprite()` 在读出时做固定 a<->b 交换；D11/2 已经推翻这个前提，因此这里改成读取 raw sprite。
#    - 现在 `_tile_for_frame_code()` 假设 frame 只会用 8/9，且 unit 恰好双色；generator 明确保这两条。
#    - 颜色映射使用“当前 variant 的首个可见色”，这要求 unit 的左上可见色稳定；generator 让 unit 全填充且 [0,0] 固定为颜色 A。
#    - color 9 的逻辑依赖 `swap` 与 `mirror` 都产生可观察差异；generator 用 delta 检查排除退化样本。
# d) 放置合法性
#    - 复合体外接尺寸仍只由旋转后的 frame 尺寸与 unit 尺寸决定，所以原 anchor / non-overlap 逻辑可以复用。
#    - generator 先设计每个 legend 的 rotated composite，再用 bbox 级别检查双 legend 不重叠、不越界。
#    - 最后仍用 `main(input) == output` 回验，确保 solver/generator 契约一致。
# e) 视觉可观察性
#    - 输入里 frame 的 8/9 色块可直接对应输出里两类大 tile: 一类带 8、一类带 9，且 9 对应镜像块。
#    - 因为 8/9 同时出现在输入 frame 与输出 tile 内部，解题者能把“框架颜色驱动颜色迁移”直接看出来。
#    - 不使用颜色大小排序、黑色特判等不可见规则；全靠局部可见对象属性。
# f) 可视化整洁度
#    - 8/9 被保留给 frame 语义与输出迁移色；4/7 专门保留给 marker / rotation，不再进入 sprite 或背景。
#    - 输出只混用 sprite 的两色与背景两色，避免和 legend 控制符号串语义。
# g) 变化的可推断性
#    - 每个 legend 只涉及一组 raw unit、两种固定 frame 语义色、两类 tile 行为，变换自由度被压到人类可归纳的范围。
#    - 同一颜色 8/9 在跨样例中语义恒定: 8 总是 raw+recolor+normal，9 总是 swap+recolor+mirror。
# h) main() 的新假设与 generator 对应保证
#    - solver 断言 frame 同时含 8 和 9；generator 强制两色都至少出现多次。
#    - solver 断言 unit 恰好双色；generator 只生成完整双色 sprite。
#    - solver 默认 color 9 的 mirror 和 swap 都会影响输出；generator 用 `variant_delta / mirror_delta / color_delta` 过滤掉不可见样本。

FRAME_RAW_COLOR = 8
FRAME_SWAP_MIRROR_COLOR = 9
FRAME_RULE_COLORS = (FRAME_RAW_COLOR, FRAME_SWAP_MIRROR_COLOR)

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)  # 4..8
FRAME_W_RANGE = (3, 6)  # 3..5
FRAME_ACTIVE_RANGE = (9, 17)
FRAME_FILL_RANGE = (0.50, 0.65)
FRAME_GEN_ATTEMPTS = 100
MIN_RULE_CELLS = 3
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
MARKER_COLOR = 4
ROTATION_COLOR = 7
PALETTE_POOL = [1, 2, 3, 5]  # reserve 4/7 for legend controls and 8/9 for frame semantics
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400

FUSED_RULES = {
    FRAME_RAW_COLOR: (False, False),          # raw variant, no mirror
    FRAME_SWAP_MIRROR_COLOR: (True, True),    # swapped variant, mirror
}
NO_VARIANT_RULES = {
    FRAME_RAW_COLOR: (False, False),
    FRAME_SWAP_MIRROR_COLOR: (False, True),
}
NO_MIRROR_RULES = {
    FRAME_RAW_COLOR: (False, False),
    FRAME_SWAP_MIRROR_COLOR: (True, False),
}


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    _h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    cols6.sort()
    runs: list[tuple[int, int]] = []
    s = e = cols6[0]
    for c in cols6[1:]:
        if c == e + 1:
            e = c
        else:
            runs.append((s, e))
            s = e = c
    runs.append((s, e))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Identify the left canvas separator and an optional right legend separator."""
    _h, w = g.shape
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of an 8x5 marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (
                np.all(sub[0, :] == 6)
                and np.all(sub[4, :] == 6)
                and np.all(sub[:, 0] == 6)
                and np.all(sub[:, 4] == 6)
            ):
                continue
            inner = sub[1:4, 1:4]
            if not np.any((inner != 0) & (inner != 6)):
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            out.append((int(r0), int(c0)))
    out.sort()
    return out


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Read the raw top-most 4-connected non-{0,6} component as the legend sprite."""
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    st = [(rmin, cseed)]
    vis[rmin, cseed] = True
    pts: list[tuple[int, int]] = []
    while st:
        r, c = st.pop()
        pts.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                st.append((rr, cc))
    r0, c0, r1, c1 = bbox_coords(np.asarray(pts))
    spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    """Frame pixels live between the unit sprite and the marker square."""
    r_start = unit_bb[2] + 1
    r_end = border6_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    pts[:, 0] += r_start
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return (r0, c0, r1, c1)


def _position_indices_from_border_square(
    legend: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int] | None:
    inner = legend[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Read the 3x3 clock-hand under the marker square:
      up -> 0, right -> 1, down -> 2, left -> 3.
    """
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    pts = np.argwhere((sub != 0) & (sub != 6))
    assert pts.size > 0, "no rotation marker pixel found"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - center[0]), int(p[1] - center[1])
    if abs(dr) >= abs(dc):
        if dr < 0:
            return 0
        if dr > 0:
            return 2
    else:
        if dc > 0:
            return 1
        if dc < 0:
            return 3
    raise AssertionError("rotation marker pixel at center, direction ambiguous")


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """Convert a 0/1/2 anchor choice into the placed object's top-left coordinate."""
    if anchor_idx <= 0:
        canvas_anchor = 0
        obj_anchor = 0
    elif anchor_idx >= 2:
        canvas_anchor = max(0, size - 1)
        obj_anchor = max(0, place_size - 1)
    else:
        canvas_anchor = (size - 1) // 2 if size > 1 else 0
        obj_anchor = (place_size - 1) // 2 if place_size > 1 else 0
    return canvas_anchor - obj_anchor


def _sprite_colors(sprite: np.ndarray) -> tuple[int, int]:
    cols = sorted({int(x) for x in np.unique(sprite) if int(x) not in (0, 6)})
    assert len(cols) == 2, f"unit sprite must use exactly two colors, got {cols}"
    return cols[0], cols[1]


def _swap_two_colors(sprite: np.ndarray) -> np.ndarray:
    a, b = _sprite_colors(sprite)
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
    return out


def _first_active_color(sprite: np.ndarray) -> int:
    pts = np.argwhere((sprite != 0) & (sprite != 6))
    assert pts.size > 0, "sprite must contain active pixels"
    r0, c0 = pts[0]
    return int(sprite[r0, c0])


def _map_primary_color(sprite: np.ndarray, target_color: int) -> np.ndarray:
    """Map the current variant's first visible color to target_color; keep the other color."""
    out = sprite.copy()
    color_a = _first_active_color(sprite)
    out[sprite == color_a] = target_color
    return out


def _tile_for_frame_code(
    unit_sprite: np.ndarray,
    frame_code: int,
    *,
    rules: dict[int, tuple[bool, bool]] = FUSED_RULES,
    recolor: bool = True,
) -> np.ndarray:
    """Build the local tile selected by one frame code."""
    assert frame_code in rules, f"unexpected frame code: {frame_code}"
    use_swap, use_mirror = rules[frame_code]
    tile = _swap_two_colors(unit_sprite) if use_swap else unit_sprite.copy()
    if recolor:
        tile = _map_primary_color(tile, frame_code)
    if use_mirror:
        tile = np.fliplr(tile)
    return tile


def _build_composite_from_frame(
    unit_sprite: np.ndarray,
    frame_codes: np.ndarray,
    *,
    rules: dict[int, tuple[bool, bool]] = FUSED_RULES,
    recolor: bool = True,
) -> np.ndarray:
    """Expand a semantic 8/9 frame into the final composite sprite."""
    uh, uw = unit_sprite.shape
    fh, fw = frame_codes.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    tile_cache = {
        code: _tile_for_frame_code(unit_sprite, code, rules=rules, recolor=recolor)
        for code in rules
    }
    mask_cache = {
        code: (tile_cache[code] != 0) & (tile_cache[code] != 6)
        for code in tile_cache
    }
    for i in range(fh):
        for j in range(fw):
            code = int(frame_codes[i, j])
            if code == 0 or code == 6:
                continue
            tile = tile_cache[code]
            tile_mask = mask_cache[code]
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][tile_mask] = tile[tile_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode color-instruction frame cells from each legend and paint enlarged composites onto the canvas.

    Input structure:
    - Full-height color-6 columns separate the left legend, the central canvas, and an optional right legend.
    - Each legend still contains four stacked parts: a 2-color unit sprite, a frame template, a 5x5 anchor square, and a 3x3 clock-hand rotation marker.
    - Inside the frame template, the active cells are multi-colored: color 8 and color 9 are both foreground, and each one carries its own assembly rule.

    Rule:
    - Read the unit sprite exactly as drawn in the legend, and rotate the full colored frame layout by the clock-hand direction.
    - For every rotated frame cell:
      color 8 uses the raw unit, recolors its first visible sprite color to 8, and tiles it without mirroring;
      color 9 uses the color-swapped unit, recolors that variant's first visible color to 9, and tiles a left-right mirrored copy.
    - Place the finished composite on the canvas using the anchor encoded by the marker square's 3x3 position code.

    Key insight: the frame is no longer just shape information. Its visible colors are the entire control program: they choose the sprite variant, the local mirror behavior, and the recoloring target all at once.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue
        if marker_c0 + 5 > w:
            continue

        panel = g[:, marker_c0 : marker_c0 + 5]
        unit = _extract_unit_sprite(panel)
        if unit is None:
            continue
        unit_sprite, unit_bb = unit
        _sprite_colors(unit_sprite)

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_active = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_active, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        frame_colors = {int(x) for x in np.unique(frame[frame_active])}
        assert frame_colors <= set(FRAME_RULE_COLORS), f"unexpected frame colors: {frame_colors}"
        assert FRAME_RAW_COLOR in frame_colors and FRAME_SWAP_MIRROR_COLOR in frame_colors, "frame must use both semantic colors"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        rotated_frame = np.rot90(frame, (-dir_k) % 4)
        comp = _build_composite_from_frame(unit_sprite, rotated_frame)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        ch, cw = comp.shape
        for r in range(ch):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(cw):
                cc = left + c
                assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                if out[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val != 0 and val != 6:
                    assert not composite_written[rr, cc], "composites must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


def _allocate_colors(num_legends: int):
    """
    Pick sprite/background colors while keeping legend-control colors isolated.

    We only have four safe palette colors once 4/7/8/9/6 are reserved, so both legends
    share one sprite palette and the striped canvas uses the other disjoint pair.
    """
    if len(PALETTE_POOL) < 4:
        return None
    pool = list(PALETTE_POOL)
    random.shuffle(pool)
    sprite_colors = (pool[0], pool[1])
    bg_colors = (pool[2], pool[3])
    unit_color_sets = [sprite_colors for _ in range(num_legends)]
    return unit_color_sets, bg_colors


def _is_horizontal_asymmetric(sprite: np.ndarray) -> bool:
    return not np.array_equal(sprite, np.fliplr(sprite))


def _make_unit_sprite(n: int, colors: tuple[int, int]) -> np.ndarray:
    """Generate a filled 2-color sprite with stable color-A location and visible mirror effect."""
    color_a, color_b = colors
    total = n * n
    candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) != (0, 0)]
    for _ in range(200):
        sprite = np.full((n, n), color_a, dtype=int)
        max_b = total - 2 if total >= 3 else 1
        num_b = random.randint(1, max_b)
        for r, c in random.sample(candidates, num_b):
            sprite[r, c] = color_b
        sprite[0, 0] = color_a
        if not _is_horizontal_asymmetric(sprite):
            continue
        if len({int(x) for x in np.unique(sprite)}) != 2:
            continue
        return sprite
    raise AssertionError("failed to generate asymmetric unit sprite")


def _frame_mask_ok(mask: np.ndarray) -> bool:
    active = int(mask.sum())
    if not (FRAME_ACTIVE_RANGE[0] <= active <= FRAME_ACTIVE_RANGE[1]):
        return False
    fill = active / float(mask.size)
    if not (FRAME_FILL_RANGE[0] <= fill <= FRAME_FILL_RANGE[1]):
        return False
    _, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    return n8 == 1


def _sample_frame_mask(frame_h: int, frame_w: int) -> np.ndarray | None:
    """Generate an 8-connected frame mask within the original size/density envelope."""
    for _ in range(FRAME_GEN_ATTEMPTS):
        total = frame_h * frame_w
        lo = max(FRAME_ACTIVE_RANGE[0], int(np.ceil(total * FRAME_FILL_RANGE[0])))
        hi = min(FRAME_ACTIVE_RANGE[1], int(np.floor(total * FRAME_FILL_RANGE[1])))
        if lo > hi:
            return None
        target = random.randint(lo, hi)
        mask = np.zeros((frame_h, frame_w), dtype=bool)
        r0 = random.randrange(frame_h)
        c0 = random.randrange(frame_w)
        mask[r0, c0] = True
        while int(mask.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(mask):
                for rr, cc in neighbors8(int(r), int(c)):
                    if 0 <= rr < frame_h and 0 <= cc < frame_w and not mask[rr, cc]:
                        frontier.add((int(rr), int(cc)))
            if not frontier:
                break
            rr, cc = random.choice(list(frontier))
            mask[rr, cc] = True
        if not _frame_mask_ok(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    return None


def _frame_color_schemes(mask: np.ndarray) -> list[np.ndarray]:
    """Candidate subsets that receive color 9; remaining active cells stay color 8."""
    r_idx, c_idx = np.indices(mask.shape)
    schemes = [
        ((r_idx + c_idx) % 2 == 0) & mask,
        ((r_idx + c_idx) % 2 == 1) & mask,
        (r_idx % 2 == 0) & mask,
        (r_idx % 2 == 1) & mask,
        (c_idx % 2 == 0) & mask,
        (c_idx % 2 == 1) & mask,
        (r_idx <= np.median(r_idx[mask])) & mask,
        (c_idx <= np.median(c_idx[mask])) & mask,
    ]
    valid: list[np.ndarray] = []
    active = int(mask.sum())
    for sub in schemes:
        count = int(sub.sum())
        if count < MIN_RULE_CELLS or active - count < MIN_RULE_CELLS:
            continue
        coords = np.argwhere(sub)
        if len({int(r) for r, _ in coords}) < 2:
            continue
        if len({int(c) for _, c in coords}) < 2:
            continue
        valid.append(sub)
    return valid


def _make_frame_codes(frame_h: int, frame_w: int) -> np.ndarray | None:
    """Generate a two-color semantic frame that visibly uses both 8 and 9."""
    for _ in range(FRAME_GEN_ATTEMPTS):
        mask = _sample_frame_mask(frame_h, frame_w)
        if mask is None:
            continue
        schemes = _frame_color_schemes(mask)
        if not schemes:
            continue
        color9_mask = random.choice(schemes)
        codes = np.zeros((frame_h, frame_w), dtype=int)
        codes[mask] = FRAME_RAW_COLOR
        codes[color9_mask] = FRAME_SWAP_MIRROR_COLOR
        return codes
    return None


def _count_delta(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _design_legend(
    leg: dict, unit_colors: tuple[int, int], marker_r0: int, grid_h: int, canvas_w: int
) -> bool:
    """Fill one legend with a sprite, fused semantic frame, and only informative transforms."""
    n = leg["N"]
    sprite = _make_unit_sprite(n, unit_colors)

    avail = marker_r0 - (n + 1)
    combos = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        max_frame_w = min(FRAME_W_RANGE[1] - 1, leg["legend_w"])
        for frame_w in range(FRAME_W_RANGE[0], max_frame_w + 1):
            for rot_k in range(4):
                rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((frame_h, frame_w, rot_k))
    if not combos:
        return False

    random.shuffle(combos)
    for frame_h, frame_w, rot_k in combos:
        frame_codes = _make_frame_codes(frame_h, frame_w)
        if frame_codes is None:
            continue
        rotated_frame = np.rot90(frame_codes, (-rot_k) % 4)

        full_comp = _build_composite_from_frame(sprite, rotated_frame, rules=FUSED_RULES, recolor=True)
        no_variant_comp = _build_composite_from_frame(sprite, rotated_frame, rules=NO_VARIANT_RULES, recolor=True)
        no_mirror_comp = _build_composite_from_frame(sprite, rotated_frame, rules=NO_MIRROR_RULES, recolor=True)
        no_color_comp = _build_composite_from_frame(sprite, rotated_frame, rules=FUSED_RULES, recolor=False)

        variant_delta = _count_delta(full_comp, no_variant_comp)
        mirror_delta = _count_delta(full_comp, no_mirror_comp)
        color_delta = _count_delta(full_comp, no_color_comp)
        if variant_delta == 0 or mirror_delta == 0 or color_delta == 0:
            continue

        leg.update(
            sprite=sprite,
            frame_codes=frame_codes,
            frame_H=frame_h,
            frame_W=frame_w,
            rot_k=rot_k,
            rotated_frame=rotated_frame,
            variant_delta=variant_delta,
            mirror_delta=mirror_delta,
            color_delta=color_delta,
            count8=int(np.sum(rotated_frame == FRAME_RAW_COLOR)),
            count9=int(np.sum(rotated_frame == FRAME_SWAP_MIRROR_COLOR)),
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions while keeping composites in-bounds and non-overlapping."""
    if len(legends) == 1:
        pr, pc = random.randint(0, 2), random.randint(0, 2)
        ch, cw = _comp_size(legends[0])
        legends[0].update(
            pr=pr,
            pc=pc,
            comp_top=_anchor_to_top_left(pr, grid_h, ch),
            comp_left=_anchor_to_top_left(pc, canvas_w, cw),
        )
        return True

    a, b = legends
    a_ch, a_cw = _comp_size(a)
    b_ch, b_cw = _comp_size(b)
    valid_pairs = []
    for apr in range(3):
        for apc in range(3):
            at = _anchor_to_top_left(apr, grid_h, a_ch)
            al = _anchor_to_top_left(apc, canvas_w, a_cw)
            if at < 0 or al < 0 or at + a_ch > grid_h or al + a_cw > canvas_w:
                continue
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_ch)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_cw)
                    if bt < 0 or bl < 0 or bt + b_ch > grid_h or bl + b_cw > canvas_w:
                        continue
                    if at < bt + b_ch and bt < at + a_ch and al < bl + b_cw and bl < al + a_cw:
                        continue
                    valid_pairs.append((apr, apc, at, al, bpr, bpc, bt, bl))
    if not valid_pairs:
        return False
    apr, apc, at, al, bpr, bpc, bt, bl = random.choice(valid_pairs)
    a.update(pr=apr, pc=apc, comp_top=at, comp_left=al)
    b.update(pr=bpr, pc=bpc, comp_top=bt, comp_left=bl)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
    """Build the unchanged striped canvas background."""
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    bw = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
    band_indices = np.arange(dim) // bw % 2
    colors = np.array([bg1, bg2])
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's fused composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite_from_frame(leg["sprite"], leg["rotated_frame"])
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
    """Draw one legend panel with raw sprite, semantic frame, anchor square, and rotation patch."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = MARKER_COLOR

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = ROTATION_COLOR
    grid[cx_r + dr, cx_c + dc] = ROTATION_COLOR

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["sprite"]

    frame_h, frame_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = n + 1 + gap
    frame_col = col_start + random.randint(0, leg["legend_w"] - frame_w)
    frame_region = grid[fr_start : fr_start + frame_h, frame_col : frame_col + frame_w]
    active = leg["frame_codes"] != 0
    frame_region[active] = leg["frame_codes"][active]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    n_hi = UNIT_SIZE_RANGE[1] - (1 if num_legends == 2 else 0)
    for _ in range(num_legends):
        n = random.randint(UNIT_SIZE_RANGE[0], n_hi)
        legends.append({"N": n, "legend_w": n + 2})

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2) = colors
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT

    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0, is_right=True)

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
