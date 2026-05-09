# Combo variant of a32d8b75 (index 18)
# combo_spec: D4/1+D7/2+D16/0
# Name: 位置旋转+数量编码+九宫格

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    原题流程是：读 unit sprite 并换色 -> 读 frame -> 用底部方向指针给整个 frame/复合精灵做全局 90 度旋转
#    -> 用 5x5 方框内 3x3 单像素给出 start/middle/end 锚点 -> 贴到画布。
#    现在三处同时改变：
#    1. D4/1：frame 的每个活跃格不再贴同朝向 sprite，而是按该格相对 frame 几何中心的方位
#       选择 0/90/180/270 度局部旋转，形成放射状 composite。
#    2. D7/2：底部 3x3 patch 里颜色 7 的“指针”不看朝向，只看中心外额外像素数；1/2/3/4 个分别编码
#       全局旋转 0/90/180/270 度。patch 整体会随机翻转/旋转，使方向失去意义。
#    3. D16/0：5x5 方框内 3x3 单像素不再表示 start/middle/end 锚定，而是直接选定画布九宫格中的一个区域；
#       复合精灵的离散中心对齐到该区域中心。
#    例：若某活跃格位于 frame 中心右侧，则局部贴一个顺时针 90 度 sprite；底部 patch 若有 3 个额外像素，
#       则整个 frame 先按 180 度全局旋转；marker 若在左上格，则最终复合精灵中心落在画布左上三分区的中心。
#    与原题差异是：输出既受“数量编码的全局旋转”控制，又受“frame 内方位”的局部旋转场控制，同时位置不再贴边。
#
# b) 复杂度校准
#    - 三个自由度都保持离散小集合：局部方位 4 档、数量编码旋转 4 档、九宫格位置 9 档。
#    - 生成器强制 unit sprite 的四个 90 度旋转彼此不同，否则 D4 的局部旋转不可见。
#    - 生成器强制 frame 覆盖至少 3 个方位类别，且同时包含纵向/横向类别，否则“放射状”会退化成单向条带。
#    - 生成器优先使用与旧锚点语义不同的九宫格位置，否则 D16 视觉差异会变弱。
#
# c) 隐式假设审查
#    - 原 main 中“所有 frame 活跃格都贴同一个 sprite 朝向”的前提失效，因此改成 `_build_radial_composite()`
#      逐格决定局部旋转。
#    - 原 main 中“底部 patch 的最远像素方向决定旋转”的前提失效，因此改成 `_rotation_from_pointer_count()`
#      并新增假设：中心像素存在、额外像素数在 1..4。生成器显式保证这点。
#    - 原 `_anchor_to_top_left()` 的独立行列锚定前提失效，因此改成 `_grid9_center_to_top_left()`；
#      新前提是 chosen region center 对齐后对象完整落入画布。生成器按最终 bbox 枚举合法位置。
#    - `frame_mask = np.rot90(frame_mask0, (-rot_k) % 4)` 仍然成立，但其含义变为“先整体转动方位场，再按转后的格位
#      触发局部 sprite 旋转”，因此 solver 和 generator 必须共用同一 composite 构造逻辑以避免漂移。
#
# d) 放置合法性
#    - 继续采用 output-first：先定 sprite / frame 尺寸 / 数量编码旋转 / 九宫格候选位置，再挑合法不越界且互不重叠的放置。
#    - 双 legend 时按照最终 composite bbox 做不重叠检测。
#
# e) 视觉可观察性
#    - 人类能在样例中看到：同样的 patch 朝向若额外像素数不同，则全局旋转不同；同样的 frame，不同方位格里的小 sprite
#      方向不一致；角落 marker 不再把物体顶到边界，而是落在对应九宫格中心附近。
#    - patch 的方向被随机扰动，能迫使解题者放弃原题的“看指向”策略。
#
# f) 可视化整洁度
#    - 仍然只有背景画布 + 1~2 个 composite，legend 保持原有紧凑布局。
#    - 不允许 clipping，不允许 composite 互相覆盖，frame 和 sprite 的颜色与背景分离，避免图面混乱。
#
# g) 变化的可推断性
#    - 本融合不新增颜色映射规则，仍只有原题的 sprite 两色互换。
#    - 关键变化都由几何结构显式给出：局部方位、额外像素数量、九宫格位置。没有依赖颜色编号大小或遍历顺序的隐藏规则。


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    _, w = g.shape
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
    _, w = g.shape
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
    """Unit sprite is the top-most 4-connected component of pixels not in {0,6}."""
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
    cols = sorted({int(x) for x in np.unique(spr) if int(x) not in (0, 6)})
    if len(cols) == 2:
        a, b = cols
        spr2 = spr.copy()
        spr2[spr == a] = -1
        spr2[spr == b] = a
        spr2[spr2 == -1] = b
        spr = spr2
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    """Frame pixels live between the unit sprite and the border-6 marker square."""
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


def _rotation_from_pointer_count(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker lives in the 3x3 patch below the bordered square.

    The center pixel must be colored. The count of additional colored pixels,
    not the patch's orientation, encodes the number of clockwise quarter-turns:
      1 -> 0, 2 -> 1, 3 -> 2, 4 -> 3.
    """
    h, w = legend.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    marker = (sub != 0) & (sub != 6)
    total = int(marker.sum())
    assert marker[1, 1], "rotation marker must include the center pixel"
    arm_pixels = total - 1
    assert 1 <= arm_pixels <= 4, f"rotation marker arm length must be 1..4, got {arm_pixels}"
    return arm_pixels - 1


def _cell_rotation_from_position(r: int, c: int, h: int, w: int) -> int:
    """Map a frame-cell position to one of four compass-facing sprite rotations."""
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    dr = float(r) - cr
    dc = float(c) - cc
    if dr == 0.0 and dc == 0.0:
        return 0
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 2
    return 1 if dc > 0 else 3


def _direction_histogram(frame_mask: np.ndarray) -> dict[int, int]:
    hist = {0: 0, 1: 0, 2: 0, 3: 0}
    fh, fw = frame_mask.shape
    for i, j in np.argwhere(frame_mask):
        hist[_cell_rotation_from_position(int(i), int(j), fh, fw)] += 1
    return hist


def _build_radial_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """
    Tile the unit sprite into each active frame cell, rotating the tile according
    to that cell's direction relative to the frame center.
    """
    unit_mask = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    rot_sprites = [np.rot90(unit_sprite, (-k) % 4) for k in range(4)]
    rot_masks = [(spr != 0) & (spr != 6) for spr in rot_sprites]
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            rk = _cell_rotation_from_position(i, j, fh, fw)
            tile = rot_sprites[rk]
            tile_mask = rot_masks[rk]
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][tile_mask] = tile[tile_mask]
    return comp


def _legacy_anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """Original task's independent start/middle/end anchor decoder."""
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


def _third_center(idx: int, size: int) -> int:
    """Return the center cell of the idx-th third (0/1/2) of a discrete line."""
    start = (idx * size) // 3
    end = ((idx + 1) * size) // 3 - 1
    if end < start:
        end = start
    return (start + end) // 2


def _grid9_center_to_top_left(region_idx: int, size: int, place_size: int) -> int:
    """Align the placed object's discrete center to the chosen third's center cell."""
    canvas_center = _third_center(region_idx, size)
    obj_center = (place_size - 1) // 2
    return canvas_center - obj_center


def _composite_bbox(top: int, left: int, ch: int, cw: int) -> tuple[int, int, int, int]:
    return top, left, top + ch, left + cw


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _direction_from_patch_if_original_rule(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """Original-task direction decoder, kept only for generator-side degeneracy checks."""
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    pts = np.argwhere((sub != 0) & (sub != 6))
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - center[0]), int(p[1] - center[1])
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 2
    return 1 if dc > 0 else 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble radial composites from legend panels and place them onto a canvas using fused geometric cues.

    Input structure:
    - Full-height color-6 separator columns split the grid into a canvas and one or two legend panels.
    - Each legend panel contains, from top to bottom, a swapped two-color unit sprite, a frame mask, a 5x5 bordered square whose inner 3x3 picks one of nine canvas regions, and a 3x3 rotation patch.
    - In that rotation patch, the center pixel is always present and the number of extra color-7 pixels encodes the global quarter-turn count; the patch orientation itself is irrelevant.

    Rule:
    - Read the unit sprite, undo the legend-side color swap, decode the global frame rotation from pointer-pixel count, and rotate the frame mask by that amount.
    - Build the composite by stamping the unit sprite into every active frame cell, but rotate each local tile by the cell's direction around the rotated frame center (up/right/down/left).
    - Place the finished composite so its discrete center lands on the center of the selected canvas third-by-third region, then overwrite the canvas background with the composite pixels.

    Key insight:
    - Three spatial codes must be fused: the bottom patch gives a count-based global rotation, the frame geometry gives per-cell local sprite rotation, and the bordered 3x3 marker names an absolute nine-grid destination rather than an edge anchor.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(main_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main_canvas.copy()
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
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        rot_k = _rotation_from_pointer_count(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-rot_k) % 4)
        comp = _build_radial_composite(unit_sprite, frame_mask)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _grid9_center_to_top_left(pr, eff_h, comp.shape[0])
        left = _grid9_center_to_top_left(pc, eff_w, comp.shape[1])

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


# --------------- generator (radial + pointer-count + grid9) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
BASE_ARM_PATH = [(1, 2), (0, 2), (0, 1), (0, 0)]
GEN_RETRIES = 500
SPRITE_RETRIES = 200
FRAME_RETRIES = 200

_GEN_COVERAGE = {
    "num_legends": {1: 0, 2: 0},
    "pos": {(pr, pc): 0 for pr in range(3) for pc in range(3)},
}


def _allocate_colors(num_legends: int):
    """Pick unit colors, background colors, and frame colors."""
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets = []
    used_unit = set()
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        c1, c2 = pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2))
        used_unit.update([c1, c2])

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


def _choose_num_legends() -> int:
    counts = _GEN_COVERAGE["num_legends"]
    min_count = min(counts.values())
    return random.choice([n for n, count in counts.items() if count == min_count])


def _record_coverage(legends: list[dict]) -> None:
    _GEN_COVERAGE["num_legends"][len(legends)] += 1
    for leg in legends:
        _GEN_COVERAGE["pos"][(leg["pr"], leg["pc"])] += 1


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _all_rotations_distinct(sprite: np.ndarray) -> bool:
    return len({np.rot90(sprite, k).tobytes() for k in range(4)}) == 4


def _make_asymmetric_sprite(n: int, c1: int, c2: int) -> np.ndarray | None:
    """Generate a fully filled NxN 2-color sprite with four distinct quarter-turn rotations."""
    area = n * n
    for _ in range(SPRITE_RETRIES):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, area - 1)
        for p in random.sample(range(area), num_c2):
            sprite[p // n, p % n] = c2
        if _all_rotations_distinct(sprite):
            return sprite
    return None


def _frame_has_visible_directionality(frame_mask: np.ndarray) -> bool:
    hist = _direction_histogram(frame_mask)
    used_dirs = sum(v > 0 for v in hist.values())
    has_vertical = hist[0] > 0 or hist[2] > 0
    has_horizontal = hist[1] > 0 or hist[3] > 0
    return used_dirs >= 3 and has_vertical and has_horizontal


def _list_valid_positions(grid_h: int, canvas_w: int, comp_h: int, comp_w: int) -> list[dict]:
    out: list[dict] = []
    for pr in range(3):
        top = _grid9_center_to_top_left(pr, grid_h, comp_h)
        legacy_top = _legacy_anchor_to_top_left(pr, grid_h, comp_h)
        if top < 0 or top + comp_h > grid_h:
            continue
        for pc in range(3):
            left = _grid9_center_to_top_left(pc, canvas_w, comp_w)
            legacy_left = _legacy_anchor_to_top_left(pc, canvas_w, comp_w)
            if left < 0 or left + comp_w > canvas_w:
                continue
            out.append(
                {
                    "pr": pr,
                    "pc": pc,
                    "top": top,
                    "left": left,
                    "differs_from_legacy": (top != legacy_top) or (left != legacy_left),
                }
            )
    return out


def _edge_score(pos: dict) -> int:
    """Prefer non-center nine-grid regions to make the D16 shift more visible."""
    return int(pos["pr"] != 1) + int(pos["pc"] != 1)


def _position_pool_score(valid_positions: list[dict]) -> tuple[int, int, int]:
    diff_positions = [p for p in valid_positions if p["differs_from_legacy"]]
    return (
        len(diff_positions),
        max((_edge_score(p) for p in diff_positions), default=-1),
        len(valid_positions),
    )


def _make_rotation_patch(rot_k: int) -> np.ndarray:
    """
    Build a 3x3 patch with center plus 1..4 additional pixels.

    The prefix length encodes rotation. The patch is then randomly flipped/rotated
    so the count matters but the orientation does not.
    """
    arm_len = rot_k + 1
    patch = np.zeros((3, 3), dtype=int)
    patch[1, 1] = 7
    for rr, cc in BASE_ARM_PATH[:arm_len]:
        patch[rr, cc] = 7
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, random.randint(0, 3))
    return patch


def _design_legend(
    leg: dict, unit_colors: tuple[int, int], frame_color: int, grid_h: int, canvas_w: int
) -> bool:
    """Fill in sprite, frame, rotation, and legal grid9 placements for one legend."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = _make_asymmetric_sprite(n, uc1, uc2)
    if sprite is None:
        return False
    legend_sprite = _color_swap(sprite, uc1, uc2)

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    avail = marker_r0 - (n + 1)
    combos = []
    for f_h in range(*FRAME_H_RANGE):
        if avail < f_h + 1:
            continue
        for f_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                rh, rw = (f_h, f_w) if rot_k % 2 == 0 else (f_w, f_h)
                comp_h = rh * n
                comp_w = rw * n
                valid_positions = _list_valid_positions(grid_h, canvas_w, comp_h, comp_w)
                if not valid_positions:
                    continue
                if not any(pos["differs_from_legacy"] for pos in valid_positions):
                    continue
                combos.append((f_h, f_w, rot_k, valid_positions))
    if not combos:
        return False

    best_score = max(_position_pool_score(valid_positions) for _, _, _, valid_positions in combos)
    best_combos = [combo for combo in combos if _position_pool_score(combo[3]) == best_score]
    random.shuffle(best_combos)

    for frame_h, frame_w, rot_k, valid_positions in best_combos:
        for _ in range(FRAME_RETRIES):
            frame_mask = random_sprite(
                frame_h,
                frame_w,
                density=FRAME_DENSITY,
                connectivity=8,
                color_palette=[1],
                symmetry="not_symmetric",
            ) != 0
            rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
            if not _frame_has_visible_directionality(rot_frame):
                continue
            leg.update(
                sprite=sprite,
                legend_sprite=legend_sprite,
                frame_mask=frame_mask,
                frame_H=frame_h,
                frame_W=frame_w,
                frame_color=frame_color,
                rot_k=rot_k,
                rotated_frame=rot_frame,
                valid_positions=valid_positions,
            )
            return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_positions(legends: list[dict]) -> bool:
    """Choose legal grid9 placements and guarantee the combo's new placement semantics appear."""
    pos_counts = _GEN_COVERAGE["pos"]
    if len(legends) == 1:
        options = [p for p in legends[0]["valid_positions"] if p["differs_from_legacy"]]
        if not options:
            return False
        min_use = min(pos_counts[(p["pr"], p["pc"])] for p in options)
        pool = [p for p in options if pos_counts[(p["pr"], p["pc"])] == min_use]
        best_score = max(_edge_score(p) for p in pool)
        choice = random.choice([p for p in pool if _edge_score(p) == best_score])
        legends[0].update(pr=choice["pr"], pc=choice["pc"], comp_top=choice["top"], comp_left=choice["left"])
        return True

    a, b = legends
    a_h, a_w = _comp_size(a)
    b_h, b_w = _comp_size(b)
    valid_pairs = []
    for pa in a["valid_positions"]:
        box_a = _composite_bbox(pa["top"], pa["left"], a_h, a_w)
        for pb in b["valid_positions"]:
            box_b = _composite_bbox(pb["top"], pb["left"], b_h, b_w)
            if _boxes_overlap(box_a, box_b):
                continue
            if not (pa["differs_from_legacy"] or pb["differs_from_legacy"]):
                continue
            valid_pairs.append((pa, pb))
    if not valid_pairs:
        return False

    preferred_pairs = [pair for pair in valid_pairs if pair[0]["differs_from_legacy"] and pair[1]["differs_from_legacy"]]
    pool = preferred_pairs if preferred_pairs else valid_pairs
    min_total_use = min(pos_counts[(pa["pr"], pa["pc"])] + pos_counts[(pb["pr"], pb["pc"])] for pa, pb in pool)
    pool = [
        pair
        for pair in pool
        if pos_counts[(pair[0]["pr"], pair[0]["pc"])] + pos_counts[(pair[1]["pr"], pair[1]["pc"])] == min_total_use
    ]
    min_peak_use = min(
        max(pos_counts[(pa["pr"], pa["pc"])], pos_counts[(pb["pr"], pb["pc"])]) for pa, pb in pool
    )
    pool = [
        pair
        for pair in pool
        if max(pos_counts[(pair[0]["pr"], pair[0]["pc"])], pos_counts[(pair[1]["pr"], pair[1]["pc"])]) == min_peak_use
    ]
    best_score = max(_edge_score(pa) + _edge_score(pb) for pa, pb in pool)
    pa, pb = random.choice([pair for pair in pool if _edge_score(pair[0]) + _edge_score(pair[1]) == best_score])
    a.update(pr=pa["pr"], pc=pa["pc"], comp_top=pa["top"], comp_left=pa["left"])
    b.update(pr=pb["pr"], pc=pb["pc"], comp_top=pb["top"], comp_left=pb["left"])
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
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


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's radial composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_radial_composite(leg["sprite"], leg["rotated_frame"])
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
    """Draw one legend panel (separator, marker square, count-coded rotation patch, sprite, frame)."""
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
    grid[rr0 : rr0 + 3, mc0 + 1 : mc0 + 4] = _make_rotation_patch(leg["rot_k"])

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    f_h, f_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - f_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - f_w))
    frame_region = grid[fr_start : fr_start + f_h, mc0 + f_col : mc0 + f_col + f_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = _choose_num_legends()

    legends = []
    for _ in range(num_legends):
        n = random.randint(*UNIT_SIZE_RANGE)
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
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], grid_h, canvas_w):
            return None

    if not _assign_positions(legends):
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
            _record_coverage(legends)
            return grid
    except Exception:
        pass
    return None
