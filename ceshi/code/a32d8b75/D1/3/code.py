# Variant of a32d8b75
# Dimension: D1 — 匹配条件：精灵组装方式 (match_cond)
# Change: s1（单元精灵）替换 d1（框架模板）中每个活跃像素，形成分形式的放大复合精灵... → 边缘对齐拼接：单元精灵的边缘图案必须与相邻铺贴块的边缘一致

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis
# spec_variant:
#   core_change:
#     original: 每个 frame 活跃格都放同一个未旋转单元精灵
#     variant: 每个活跃格先尝试放基础朝向；若与已铺相邻块的接触边颜色序列不一致，就改放某个旋转版本
#   concrete_pixel_behavior:
#     sprite:
#       size: 固定 5x5
#       colors: 恰好 2 色，仍保留 legend->solver 的读入后换色机制
#       edges: 四条边都做成彼此不同的回文序列，这样旋转后每个朝向的 top/left edge 都唯一可辨
#       consequence: 同一块的 0° 与 180° 会形成唯一可行的交替拼接；90°/270° 在生成样本里不会满足约束
#     assembly:
#       traversal: 对旋转后的 frame_mask 按 4 邻接组件做 BFS 传播
#       seed_rule: 每个新的 4-连通子块的首格固定使用 0°
#       local_rule: 新格必须匹配已确定邻块的对应接触边（上/下/左/右）
#       visible_effect: 输出复合图案中会出现交替翻转的 5x5 子块，不再是整块重复平铺
#   traced_example:
#     old:
#       frame row: [1,1]
#       placement: [S, S]
#       result: 两块接触边通常不同，但旧规则完全忽略
#     new:
#       first cell: S(0°)
#       second cell: 若其 left edge != 第一块 right edge，则尝试旋转；只有 S(180°) 满足，于是变为 [S, rot180(S)]
#       difference: 旋转是由邻接边颜色序列决定，而不是由 frame 或 marker 直接编码
#   complexity_calibration:
#     why_not_simpler:
#       - 若继续允许 2x2/3x3 单元，二值回文边型种类太少，多个旋转会共享同一条边，解题会退化成任意选旋转
#       - 因此把单元大小收紧到 5x5，使四条边能同时唯一
#     why_not_harder:
#       - 只用局部 4 邻接约束传播，不引入全局回溯
#       - 仍保留原任务的 frame、marker、rotation panel、anchor 放置流程
#   implicit_assumptions_review:
#     solver_changes:
#       - 旧版 comp 构造默认所有 tile 相同；这里不再成立，必须逐格决定旋转
#       - 旧版 frame 只要 8-连通即可；这里边缘匹配只发生在 4 邻接，所以生成器改用 4-连通 frame，保证大量有效接触边
#       - 旧版 unit 任意两色填充即可；这里要求四边序列可区分，所以生成器专门造边缘编码 sprite
#   placement_legality:
#     generator:
#       - 先根据新装配规则生成真实 composite，再做 anchor 放置
#       - 两 legend 情况下仍用 bbox 不重叠约束
#       - frame 尺寸缩小到 2..4，以配合 5x5 tile 后仍能在 30 列画布中完整放下
#   observability:
#     clues:
#       - legend 中的 5x5 单元本身四边不对称
#       - 输出中相邻 5x5 子块共享整条相同边界颜色序列
#       - 同一个 frame 内会重复出现 0°/180° 两种朝向，足够让人从少量样例看出“边缘不对就翻转”
#   cleanliness:
#     - marker 与 rotation panel 保持原样，变化只发生在 composite 内部
#     - 背景颜色仍与单元颜色分离，避免边缘匹配效果被背景噪声掩盖


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
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
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= g.shape[1] - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
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


def _extract_frame_bbox(legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int) -> tuple[int, int, int, int] | None:
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


def _position_indices_from_border_square(legend: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
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


def _rotated_variants(sprite: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(sprite, (-k) % 4).copy() for k in range(4)]


def _edge_signatures(sprite: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    top = tuple(int(x) for x in sprite[0, :])
    right = tuple(int(x) for x in sprite[:, -1])
    bottom = tuple(int(x) for x in sprite[-1, :])
    left = tuple(int(x) for x in sprite[:, 0])
    return top, right, bottom, left


def _candidate_rotations_from_neighbor(
    edges: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    src_k: int,
    dr: int,
    dc: int,
) -> list[int]:
    src_top, src_right, src_bottom, src_left = edges[src_k]
    cands: list[int] = []
    for k, (top, right, bottom, left) in enumerate(edges):
        ok = False
        if dr == -1 and dc == 0:
            ok = bottom == src_top
        elif dr == 1 and dc == 0:
            ok = top == src_bottom
        elif dr == 0 and dc == -1:
            ok = right == src_left
        elif dr == 0 and dc == 1:
            ok = left == src_right
        if ok:
            cands.append(k)
    return cands


def _build_edge_matched_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill each 4-neighbor frame component by propagating edge constraints from a 0° seed.
    """
    variants = _rotated_variants(unit_sprite)
    edges = [_edge_signatures(v) for v in variants]
    fh, fw = frame_mask.shape
    chosen = np.full((fh, fw), -1, dtype=int)

    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j] or chosen[i, j] >= 0:
                continue
            chosen[i, j] = 0
            queue = [(i, j)]
            qpos = 0
            while qpos < len(queue):
                r, c = queue[qpos]
                qpos += 1
                src_k = int(chosen[r, c])
                for nr, nc in neighbors4(r, c):
                    if not (0 <= nr < fh and 0 <= nc < fw and frame_mask[nr, nc]):
                        continue
                    cands = _candidate_rotations_from_neighbor(edges, src_k, nr - r, nc - c)
                    assert cands, "no rotation satisfies local edge-matching constraints"
                    assert len(cands) == 1, "edge-matching is ambiguous; input sprite edges are insufficiently distinctive"
                    cand = cands[0]
                    if chosen[nr, nc] < 0:
                        chosen[nr, nc] = cand
                        queue.append((nr, nc))
                    else:
                        assert int(chosen[nr, nc]) == cand, "edge-matching constraints disagree across neighbors"

    m_unit = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            k = int(chosen[i, j])
            if k < 0:
                continue
            tile = variants[k]
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = tile[m_unit]
    return comp, chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged sprites on a canvas, but choose each tiled copy by edge-matching.

    Input structure:
    - Color-6 separator columns still split legend panel(s) from the output canvas.
    - Each legend still contains a two-color unit sprite, a frame mask, a 5x5 anchor marker square, and a 3x3 rotation clock-hand.

    Rule:
    - Rotate the frame layout according to the clock-hand, then fill each active frame cell with the unit sprite orientation whose touching edges match the already-placed neighbors; if the base orientation fails, a rotated copy is used instead.
    - After the edge-matched composite is built, place it on the canvas using the same 3x3 anchor marker convention as the original task.

    Key insight:
    - The legend sprite is not repeated verbatim anymore: its asymmetric border patterns force local 0°/180° choices so adjacent 5x5 blocks share identical contact-edge color sequences.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
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
        comp, _rotations = _build_edge_matched_composite(unit_sprite, frame_mask)

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


# --------------- generator (edge-matched tiled composite) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 3]
UNIT_SIZE = 5
MIN_CANVAS_W = 10
FRAME_H_RANGE = (2, 5)          # 2..4
FRAME_W_RANGE = (2, 5)          # 2..4
FRAME_DENSITY = 0.65
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400

_EDGE_PATTERNS = [
    (0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 1, 0, 1, 0),
    (0, 1, 1, 1, 0),
]

_INTERIOR_MOTIFS = [
    np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=int),
]


def _allocate_colors(num_legends: int):
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
    out = sprite.copy()
    out[sprite == c1] = -1
    out[sprite == c2] = c1
    out[out == -1] = c2
    return out


def _make_edge_matched_sprite(c1: int, c2: int) -> np.ndarray:
    top_bits, right_bits, bottom_bits, left_bits = random.sample(_EDGE_PATTERNS, 4)
    sprite = np.full((UNIT_SIZE, UNIT_SIZE), c1, dtype=int)
    bit_to_color = lambda seq: np.array([c2 if b else c1 for b in seq], dtype=int)
    sprite[0, :] = bit_to_color(top_bits)
    sprite[-1, :] = bit_to_color(bottom_bits)
    sprite[:, 0] = bit_to_color(left_bits)
    sprite[:, -1] = bit_to_color(right_bits)

    motif = random.choice(_INTERIOR_MOTIFS).copy()
    if random.random() < 0.5:
        motif = np.rot90(motif)
    if random.random() < 0.5:
        motif = np.fliplr(motif)
    sprite[1:4, 1:4] = np.where(motif == 1, c2, c1)

    # Reapply borders after writing the interior motif.
    sprite[0, :] = bit_to_color(top_bits)
    sprite[-1, :] = bit_to_color(bottom_bits)
    sprite[:, 0] = bit_to_color(left_bits)
    sprite[:, -1] = bit_to_color(right_bits)

    variants = _rotated_variants(sprite)
    top_edges = {tuple(int(x) for x in v[0, :]) for v in variants}
    left_edges = {tuple(int(x) for x in v[:, 0]) for v in variants}
    assert len(top_edges) == 4 and len(left_edges) == 4, "sprite edges must uniquely identify rotations"
    return sprite


def _random_frame_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        mask = random_sprite(
            h,
            w,
            density=FRAME_DENSITY,
            connectivity=4,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        active = int(mask.sum())
        if active < 4:
            continue
        if active == 1:
            continue
        horiz = bool(np.any(mask[:, :-1] & mask[:, 1:]))
        vert = bool(np.any(mask[:-1, :] & mask[1:, :]))
        if not (horiz or vert):
            continue
        return mask
    return None


def _design_legend(leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    uc1, uc2 = unit_colors
    sprite = _make_edge_matched_sprite(uc1, uc2)
    legend_sprite = _color_swap(sprite, uc1, uc2)

    combos = []
    avail = marker_r0 - (UNIT_SIZE + 1)
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            if fW > 4:
                continue
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * UNIT_SIZE <= grid_h and rw * UNIT_SIZE <= canvas_w:
                    combos.append((fH, fW, rk))
    if not combos:
        return False

    frame_H, frame_W, rot_k = random.choice(combos)
    frame_mask = _random_frame_mask(frame_H, frame_W)
    if frame_mask is None:
        return False

    rotated_frame = np.rot90(frame_mask, (-rot_k) % 4)
    comp, tile_rots = _build_edge_matched_composite(sprite, rotated_frame)
    if not np.any(tile_rots == 2):
        return False

    leg.update(
        N=UNIT_SIZE,
        legend_w=UNIT_SIZE + 2,
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rotated_frame,
        composite=comp,
        tile_rotations=tile_rots,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    comp = leg["composite"]
    return int(comp.shape[0]), int(comp.shape[1])


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
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


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
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
    output = canvas.copy()
    for leg in legends:
        comp = leg["composite"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    N = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, N - 4)) if not is_right else (sep_col + max(0, N - 3))

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
    grid[1 : 1 + N, mc0 + sp_col : mc0 + sp_col + N] = leg["legend_sprite"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (N + 1)
    gap = random.randint(1, max(1, avail - fH))
    fr_start = N + 1 + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start : fr_start + fH, mc0 + f_col : mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = [{"N": UNIT_SIZE, "legend_w": UNIT_SIZE + 2} for _ in range(num_legends)]
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
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w):
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
