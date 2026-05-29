# Combo variant of a32d8b75 (index 3)
# combo_spec: D2/1+D4/2
# Name: 含空洞精灵+反向铺贴

"""
design_analysis:
  concretization:
    - 保留原任务的大框架：输入仍由左 legend、中央 canvas、可选右 legend 组成；每个 legend 仍按 "unit sprite -> frame -> 5x5 anchor square -> 3x3 rotation patch" 排列。
    - 融合后的 unit sprite 同时承担两层信息：它使用恰好 3 种非 {0,6} 颜色，并且内部允许 0 色空洞。solver 读取它时，只对 3 种非零颜色做行优先首次出现顺序的循环置换 a->b->c->a，0 空洞保持不变。
    - frame 仍是单色 8 连通掩膜，但复合图的铺贴槽位反转：不再在 frame 的着色格铺贴，而是在其 bounding box 内的非活跃格铺贴。于是输出里同时出现 "holey tile" 与 "negative frame" 的双重留空。
    - 为了让三色循环顺序对人类可见，generator 固定让首次出现顺序由 (0,0)、(0,1)、(1,0) 三个位置给出；其中 row-major 扫描中在 (1,0) 之前不会出现第三种颜色。
  behavior_trace:
    old_logic:
      - 原题若 legend unit 为 [[2,9],[9,2]]，solver 先做双色交换得到 [[9,2],[2,9]]，再把它铺到 frame 的每个活跃格。
    fused_logic:
      - 现在 legend unit 可能是 [[2,8,0],[5,2,8],[0,5,2]]。按首次出现顺序得到 (2,8,5)，循环后读成 [[8,5,0],[2,8,5],[0,2,8]]。
      - 若旋转后的 frame mask 为
          1 0 1
          0 1 0
        则原题会在 1 的位置铺贴；融合后改为只在 0 的位置铺贴，且每个贴上的 tile 自己还保留 0 洞。
    difference:
      - 一层差异是颜色读取从双色互换变成三色有向循环。
      - 第二层差异是几何铺贴从 frame 正片改成 frame 负片。
      - 第三层差异是 tile 本身不再实心，因此即使落在同一个槽位，输出内部也会透出 canvas 背景。
  complexity_calibration:
    - 只保留 1-2 个 legend，避免对象关系膨胀。
    - 三色循环限定为 3 色且顺序可见，不依赖颜色 id 排序等不可见规则。
    - sprite 洞数控制在少量范围内，并要求 frame 同时含足够多的活跃格与非活跃格，保证 "holey tile + inverse frame" 在 2-3 个样例中就能稳定看出来。
  implicit_assumptions_audit_main:
    - `_extract_unit_sprite` 原本只假设 "顶部组件非空并可 bbox"；融合后新增假设是该组件仍为单个 4 连通块。generator 通过从整块 NxN 中挖洞且保持 4 连通来保证。
    - `_cycle_sprite_colors` 新增假设是恰好存在 3 种非 {0,6} 颜色，而且它们的首次出现顺序唯一可读。generator 固定三处 seed 位置满足这一点。
    - `_extract_frame_bbox` 依赖 unit 与 frame 之间仍有空行分隔；生成器保持原布局，不让二者粘连。
    - 反向铺贴只改动了 "哪些 frame cell 被当成 tile 槽位" 这一处；复合图外框大小、旋转、anchor 解释、越界检查和重叠检查仍沿用原前提。
    - 因为 sprite 含 0 洞，`comp` 中会出现更多保留 canvas 的位置；因此 generator 还需保证 canvas 背景色不与 unit 颜色混淆到不可读，沿用与 unit 色集分离的背景取色策略。
  placement_legality:
    - 输出端仍按完整 frame 外框尺寸计算 composite 的 bbox，因此 anchor 放置和双 legend 非重叠检查可直接沿用。
    - generator 在决定 frame 尺寸、旋转和 anchor 前，先根据旋转后的 frame bbox 与 unit 边长验证 composite 能完整放入 canvas。
    - 双 legend 情况继续枚举 anchor 组合并排除 bbox 重叠，确保两个 composite 不会写到同一非零像素上。
  visual_observability:
    - 三色循环顺序由左上三格显式给出，而不是靠颜色编号。
    - sprite 洞会在每个复制块内部重复出现；frame 负片会在块与块之间形成另一层大尺度空洞，二者尺度不同，肉眼可区分。
    - 生成器拒绝没有 sprite 洞、没有 frame 非活跃位、或融合输出与正向铺贴输出完全相同的样本，避免视觉退化。
  visual_cleanliness:
    - legend 中除了原有四段结构外不新增额外标记层，避免读图负担。
    - canvas 只保留条带背景，sprite 颜色与背景色分离，因此 tile 洞与 frame 留空都能直接透出底色。
  change_inferability:
    - 颜色变化只限 3 色循环，且由固定可见位置给出顺序证据。
    - 几何变化只限 "正片/负片" 两种槽位解释，frame 的 bounding box 与旋转方式不变，便于对照归纳。
    - sprite 洞和 frame 负片会重复出现在多个放大块中，给解题者足够支撑去区分这两层留空来自不同机制。
  new_main_assumptions_to_guarantee:
    - 每个 sprite 必须至少有 1 个 0 洞、恰好 3 种非零颜色、且非零部分保持 4 连通。
    - 每个 frame 必须 8 连通且同时拥有足够多的活跃格与非活跃格，不能退化成全满或近全满。
    - 至少一个 legend 的融合输出必须与 "同 sprite 的正向铺贴" 不同，否则样本会退化成只剩三色循环这一维。
"""

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
FRAME_ACTIVE_RATIO_RANGE = (0.35, 0.68)
MIN_FRAME_ACTIVE = 5
MIN_FRAME_INACTIVE = 4
FRAME_SAMPLE_TRIES = 80
UNIT_HOLE_RATIO_RANGE = (0.08, 0.35)
UNIT_SAMPLE_TRIES = 80
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300
LEGEND_LAYOUT_TRIES = 40
CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    h, w = g.shape
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
    """
    Identify a left separator run (between left legend and canvas), and optionally a
    right separator run (between canvas and a right-side legend panel).
    """
    h, w = g.shape
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


def _ordered_sprite_colors(sprite: np.ndarray) -> list[int]:
    """Visible color order = first appearance order in row-major scan."""
    colors: list[int] = []
    seen: set[int] = set()
    for val in sprite.flat:
        iv = int(val)
        if iv in (0, 6) or iv in seen:
            continue
        seen.add(iv)
        colors.append(iv)
    return colors


def _cycle_sprite_colors(sprite: np.ndarray) -> np.ndarray:
    """Cycle a->b->c->a using the sprite's visible first-appearance order."""
    colors = _ordered_sprite_colors(sprite)
    assert len(colors) == 3, f"unit sprite must use exactly 3 colors, got {colors}"
    out = sprite.copy()
    for i, color in enumerate(colors):
        out[sprite == color] = colors[(i + 1) % 3]
    return out


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    # Unit sprite is the top-most 4-connected component of pixels not in {0,6}.
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
    spr = _cycle_sprite_colors(spr)
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    # Frame pixels live between the unit sprite and the border6 square.
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
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker is in the bottom 3x3 at rows (border6_r0+5 .. border6_r0+7) and cols (1..3),
    drawn with non-{0,6} pixels (typically 7). Use the furthest pixel from the center (1,1)
    to determine direction:
      up -> 0, right -> 1, down -> 2, left -> 3.
    """
    h, w = legend.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
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
    """
    anchor_idx in {0,1,2} selects position on a length:
      0 -> start, 1 -> middle, 2 -> end
    Returns the top-left coordinate so that the corresponding point of the placed object
    aligns with that point on the canvas.
    """
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


def _build_inverse_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile the unit sprite into the zero cells of the frame mask."""
    tile_mask = ~frame_mask
    assert np.any(tile_mask), "frame must contain at least one inactive cell"
    sprite_mask = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not tile_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][sprite_mask] = unit_sprite[sprite_mask]
    return comp


def _build_forward_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Baseline helper used for generator-side anti-degeneracy checks."""
    sprite_mask = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][sprite_mask] = unit_sprite[sprite_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode legend panels into enlarged negative-mask sprites and place them on the canvas.

    Input structure:
    - Full-height color-6 separator columns split the grid into one left legend, one central canvas, and optionally one right legend.
    - Each legend still stacks four ingredients from top to bottom: a bounded unit sprite, a single-color frame mask, a 5x5 anchor square, and a 3x3 rotation mark.
    - The unit sprite now uses exactly three non-{0,6} colors and may contain interior 0 holes; the frame still marks one connected template region inside its bounding box.

    Rule:
    - Read the unit sprite, infer its three visible colors by row-major first appearance, and cycle them a->b->c->a while leaving 0 holes untouched.
    - Rotate the frame layout from the clock-hand, then tile the cycled sprite into the frame's non-active cells rather than its active cells, and place the resulting composite using the anchor encoded by the marker square.

    Key insight:
    - Two different absences matter at once: the sprite's own 0 holes survive inside every copied tile, while the frame color no longer marks where to paste and instead marks the larger holes to skip.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
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

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_inverse_composite(unit_sprite, frame_mask)

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
    """Pick unit colors (3 per legend), 2 background colors, and 1 frame color per legend."""
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets: list[tuple[int, int, int]] = []
    used_unit: set[int] = set()
    for _ in range(num_legends):
        if len(pool) < 3:
            return None
        cols = [pool.pop(), pool.pop(), pool.pop()]
        random.shuffle(cols)
        unit_color_sets.append(tuple(cols))
        used_unit.update(cols)

    bg_cands = [c for c in pool if c not in used_unit]
    if len(bg_cands) < 2:
        return None
    random.shuffle(bg_cands)
    bg_colors = (bg_cands[0], bg_cands[1])

    frame_colors = []
    all_set = set(USABLE_COLORS)
    for uc in unit_color_sets:
        candidates = list(all_set - set(uc))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors


def _mask_is_4_connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    _, n = _scipy_label(mask.astype(int), structure=CONN4)
    return n == 1


def _has_full_bbox(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _sample_unit_mask(n: int) -> np.ndarray | None:
    """Sample an NxN connected mask with at least one hole while keeping the full bbox."""
    total = n * n
    min_holes = 1
    max_holes = max(1, min(total - 3, int(np.floor(total * UNIT_HOLE_RATIO_RANGE[1]))))
    target_min_holes = max(min_holes, int(np.ceil(total * UNIT_HOLE_RATIO_RANGE[0])))
    seed_positions = {(0, 0), (0, 1), (1, 0)}
    preferred = [
        (r, c)
        for r in range(1, n - 1)
        for c in range(1, n - 1)
        if (r, c) not in seed_positions
    ]
    others = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r, c) not in seed_positions and (r, c) not in preferred
    ]
    candidates = preferred + others
    if not candidates:
        return None

    for _ in range(UNIT_SAMPLE_TRIES):
        mask = np.ones((n, n), dtype=bool)
        random.shuffle(candidates)
        hole_goal = random.randint(target_min_holes, max_holes)
        holes = 0
        for r, c in candidates:
            if holes >= hole_goal:
                break
            trial = mask.copy()
            trial[r, c] = False
            if not _has_full_bbox(trial):
                continue
            if not _mask_is_4_connected(trial):
                continue
            mask = trial
            holes += 1
        if holes >= min_holes and _has_full_bbox(mask) and _mask_is_4_connected(mask):
            return mask
    return None


def _make_holey_legend_sprite(n: int, unit_colors: tuple[int, int, int]) -> np.ndarray | None:
    """
    Build an NxN sprite with exactly three visible colors plus at least one 0 hole.

    The row-major first-appearance order is forced by:
      (0,0) -> a, (0,1) -> b, (1,0) -> c.
    """
    mask = _sample_unit_mask(n)
    if mask is None:
        return None

    order = list(unit_colors)
    random.shuffle(order)
    a, b, c = order
    sprite = np.zeros((n, n), dtype=int)
    sprite[0, 0] = a
    sprite[0, 1] = b
    sprite[1, 0] = c

    for cc in range(2, n):
        if mask[0, cc]:
            sprite[0, cc] = random.choice([a, b])
    for rr in range(1, n):
        for cc in range(n):
            if (rr, cc) in {(1, 0)}:
                continue
            if mask[rr, cc]:
                sprite[rr, cc] = random.choice(order)

    for color in order:
        if np.any(sprite == color):
            continue
        candidates = [
            (r, c0)
            for r in range(1, n)
            for c0 in range(n)
            if mask[r, c0] and sprite[r, c0] != color and (r, c0) != (1, 0)
        ]
        if not candidates:
            return None
        rr, cc = random.choice(candidates)
        sprite[rr, cc] = color

    assert np.sum(sprite == 0) >= 1
    assert len(_ordered_sprite_colors(sprite)) == 3
    nonzero_mask = sprite != 0
    assert _mask_is_4_connected(nonzero_mask)
    assert _has_full_bbox(nonzero_mask)
    return sprite


def _sample_frame_mask(frame_h: int, frame_w: int) -> np.ndarray | None:
    """Generate an 8-connected frame mask with enough active and inactive cells for inversion."""
    total = frame_h * frame_w
    min_active = max(MIN_FRAME_ACTIVE, int(np.ceil(total * FRAME_ACTIVE_RATIO_RANGE[0])))
    max_active = min(total - MIN_FRAME_INACTIVE, int(np.floor(total * FRAME_ACTIVE_RATIO_RANGE[1])))
    if min_active > max_active:
        return None

    for _ in range(FRAME_SAMPLE_TRIES):
        frame_mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        active = int(frame_mask.sum())
        inactive = total - active
        if active < min_active or active > max_active:
            continue
        if inactive < MIN_FRAME_INACTIVE:
            continue
        if np.all(frame_mask) or not np.any(~frame_mask):
            continue
        return frame_mask
    return None


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w):
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    n = leg["N"]

    legend_sprite = _make_holey_legend_sprite(n, unit_colors)
    if legend_sprite is None:
        return False
    sprite = _cycle_sprite_colors(legend_sprite)

    avail = marker_r0 - (n + 1)
    combos = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((frame_h, frame_w, rot_k))
    if not combos:
        return False
    frame_h, frame_w, rot_k = random.choice(combos)

    frame_mask = _sample_frame_mask(frame_h, frame_w)
    if frame_mask is None:
        return False
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        unit_holes=int(np.sum(legend_sprite == 0)),
        inverse_slots=int(np.sum(~rot_frame)),
    )
    return True


def _comp_size(leg):
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign anchor positions to legends, avoiding overlap for 2-legend case. Returns False on failure."""
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

    a, b = legends[0], legends[1]
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


def _build_banded_canvas(grid_h, canvas_w, bg1, bg2):
    """Build a canvas with alternating horizontal or vertical color bands."""
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


def _overlay_composites(canvas, legends):
    """Stamp each legend's inverted composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["sprite"], leg["rotated_frame"]
        t, l = leg["comp_top"], leg["comp_left"]
        comp = _build_inverse_composite(sp, rf)
        ch, cw = comp.shape
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _fusion_is_visible(legends) -> bool:
    """Reject samples where reverse tiling or sprite holes would be visually inert."""
    for leg in legends:
        if leg["unit_holes"] <= 0:
            return False
        if leg["inverse_slots"] <= 0:
            return False
        inv = _build_inverse_composite(leg["sprite"], leg["rotated_frame"])
        fwd = _build_forward_composite(leg["sprite"], leg["rotated_frame"])
        if np.any(inv != fwd):
            return True
    return False


def _place_legend(grid, leg, col_start, sep_col, marker_r0, is_right):
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    frame_h, frame_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - frame_w))
    frame_region = grid[fr_start : fr_start + frame_h, mc0 + f_col : mc0 + f_col + frame_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    for _ in range(LEGEND_LAYOUT_TRIES):
        legends = []
        for _ in range(num_legends):
            n = random.randint(*UNIT_SIZE_RANGE)
            legends.append({"N": n, "legend_w": n + 2})

        left_lw = legends[0]["legend_w"]
        left_total = left_lw + 1
        right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
        canvas_w = GRID_W - left_total - right_total
        if canvas_w < MIN_CANVAS_W:
            continue

        colors = _allocate_colors(num_legends)
        if colors is None:
            continue
        unit_color_sets, (bg1, bg2), frame_colors = colors

        layout_ok = True
        for li in range(num_legends):
            if not _design_legend(
                legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w
            ):
                layout_ok = False
                break
        if not layout_ok:
            continue
        if not _assign_anchors(legends, grid_h, canvas_w):
            continue
        if not _fusion_is_visible(legends):
            continue

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
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
