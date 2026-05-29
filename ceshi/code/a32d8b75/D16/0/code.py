# Variant of a32d8b75
# Dimension: D16 — 锚点对齐语义 (meta)
# Change: 锚点索引0/1/2分别表示起始/中间/结束，独立应用于行和列方向... → 九宫格绝对定位

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the D16 variant
# ----------------------------------
# a) 具体化与行为追踪
#    原题里，5x5 标记方框内部 3x3 的单个彩色像素，分别给出行/列两个独立锚点：
#    0=起始、1=中间、2=结束。于是 (0,2) 的含义是“复合精灵左上角贴住画布上边，同时右边缘贴住画布右边”。
#    本变体改成更直观的九宫格语义：把逻辑画布按行列各切成 3 段，得到 3x3 区域；marker 的 9 个格位
#    直接指向这 9 个区域的中心，复合精灵的离散中心对齐到该区域中心。
#    例子：若逻辑画布高 20，则三段分别是 [0..5] [6..12] [13..19]，它们的中心行是 2, 9, 16。
#    原规则下 pr=0 会把对象顶到第 0 行；新规则下 pr=0 会把对象中心对齐到第 2 行，所以对象会离上边留出边距。
#    因此真正改变的是“marker -> top-left”的坐标解码环节；sprite 提取、frame 铺贴、旋转、覆写都保持原样。
# b) 复杂度校准
#    仍然只有 9 个位置档位，没有引入额外自由度。视觉上，角落/边缘标记不再贴边，而是落到九宫格区域中心，
#    与原题对比差异明确；同时位置仍然只有 3x3 的离散集合，不会因为连续坐标过多而难以归纳。
# c) 隐式假设审查
#    - 原 main() 的 _anchor_to_top_left 假设 marker 的行列可以分别解释，这在新语义下失效，必须改成区域中心解码。
#    - 原生成器只需保证“按边/中/边锚定后不越界”；新语义下，即使对象整体能放进画布，放到左上九宫格中心时也可能越界，
#      所以生成器必须按新语义枚举合法位置，而不能沿用旧检查。
#    - 原题里 (1,1) 的位置和新语义都接近中心，容易退化成与旧题相同；因此生成器必须主动采样至少一个与旧锚点结果不同的位置。
# d) 放置合法性
#    生成器先设计 sprite/frame/rotation，再枚举九宫格位置。只有当“中心对齐后完整落在逻辑画布内”时该位置才合法。
#    双 legend 时，还要基于最终 top-left 做包围盒不重叠检查。solver 也保留不重叠断言。
# e) 视觉可观察性
#    marker 仍然是可见的 3x3 单像素位置提示，人类可以直接对应到九宫格。为了避免样例退化成旧规则，
#    生成器要求每个样例中至少有一个 legend 的新位置与旧 anchor 位置不同，通常会表现为“同一个角标记不再贴边，而是偏向区域中央”。
# f) 可视化整洁度
#    仍然维持“背景画布 + 1~2 个复合精灵”的层次，避免引入新的叠加元素。对象不允许重叠，不裁切，marker 面板仍待在左右 legend 中。
# g) 颜色变化可推断性
#    本变体不改颜色机制。颜色交换、frame 颜色、背景条纹都沿用原任务设计；规则差异完全由空间位置体现，避免不可见因素参与推理。


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
    """
    Identify a left separator run (between left legend and canvas), and optionally a
    right separator run (between canvas and a right-side legend panel).
    """
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


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Read the 3x3 clock-hand marker below the position square.

    Direction convention:
      up -> 0, right -> 1, down -> 2, left -> 3
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
    """
    Center cell of the idx-th third (0/1/2) of a discrete line of length `size`.
    """
    start = (idx * size) // 3
    end = ((idx + 1) * size) // 3 - 1
    if end < start:
        end = start
    return (start + end) // 2


def _grid9_center_to_top_left(region_idx: int, size: int, place_size: int) -> int:
    """
    New D16 semantics:
    the marker chooses one of three canvas thirds, and the object's discrete center
    aligns to that third's center cell.
    """
    canvas_center = _third_center(region_idx, size)
    obj_center = (place_size - 1) // 2
    return canvas_center - obj_center


def _composite_bbox(top: int, left: int, ch: int, cw: int) -> tuple[int, int, int, int]:
    return top, left, top + ch, left + cw


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


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
    """Prefer positions away from the center cell to make the D16 change visually obvious."""
    return int(pos["pr"] != 1) + int(pos["pc"] != 1)


def _position_pool_score(valid_positions: list[dict]) -> tuple[int, int, int]:
    diff_positions = [p for p in valid_positions if p["differs_from_legacy"]]
    return (
        len(diff_positions),
        max((_edge_score(p) for p in diff_positions), default=-1),
        len(valid_positions),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild enlarged sprites from legend panels and place them on the canvas using nine-grid targets.

    Input structure:
    - Full-height color-6 separator columns split the input into a left legend, a main canvas, and optionally a right legend.
    - Each legend panel contains, from top to bottom, a unit sprite, a frame mask, a 5x5 border-6 square whose inner 3x3 marks one of nine target cells, and a 3x3 rotation clock-hand below it.

    Rule:
    - Read the unit sprite, color-swap it as before, tile it into every active cell of the frame, and rotate the resulting composite by the clock-hand direction.
    - Interpret the 3x3 marker as an absolute nine-grid choice: divide the logical canvas into 3x3 regions, take the chosen region's center, and align the composite's discrete center to that point before drawing it over the background.

    Key insight:
    - The marker no longer means independent start/middle/end anchors on each axis; it directly names one of nine canvas zones, so corner markers land near the centers of corner regions rather than flush against the borders.
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

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)

        unit_mask = (unit_sprite != 0) & (unit_sprite != 6)
        uh, uw = unit_sprite.shape
        fh, fw = frame_mask.shape
        comp = np.zeros((fh * uh, fw * uw), dtype=int)
        for i in range(fh):
            for j in range(fw):
                if not frame_mask[i, j]:
                    continue
                rr0 = i * uh
                cc0 = j * uw
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][unit_mask] = unit_sprite[unit_mask]

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


# --------------- generator (grid9_center_placement) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400


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


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _design_legend(leg: dict, unit_colors: tuple[int, int], frame_color: int, grid_h: int, canvas_w: int) -> bool:
    """Fill in sprite, frame, rotation, and valid nine-grid positions for one legend."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2
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
    frame_h, frame_w, rot_k, valid_positions = random.choice(best_combos)
    frame_mask = random_sprite(
        frame_h,
        frame_w,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
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
        valid_positions=valid_positions,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_positions(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Choose legal nine-grid placements and guarantee at least one differs from the legacy rule."""
    if len(legends) == 1:
        options = [p for p in legends[0]["valid_positions"] if p["differs_from_legacy"]]
        if not options:
            return False
        best_score = max(_edge_score(p) for p in options)
        choice = random.choice([p for p in options if _edge_score(p) == best_score])
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
    """Stamp each legend's tiled composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["sprite"], leg["rotated_frame"]
        n = leg["N"]
        t, l = leg["comp_top"], leg["comp_left"]
        fh, fw = rf.shape

        comp = np.zeros((fh * n, fw * n), dtype=int)
        sprite_mask = (sp != 0) & (sp != 6)
        for i in range(fh):
            for j in range(fw):
                if rf[i, j]:
                    r0, c0 = i * n, j * n
                    comp[r0 : r0 + n, c0 : c0 + n][sprite_mask] = sp[sprite_mask]

        region = output[t : t + fh * n, l : l + fw * n]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
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

    f_h, f_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - f_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - f_w))
    frame_region = grid[fr_start : fr_start + f_h, mc0 + f_col : mc0 + f_col + f_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

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

    if not _assign_positions(legends, grid_h, canvas_w):
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
