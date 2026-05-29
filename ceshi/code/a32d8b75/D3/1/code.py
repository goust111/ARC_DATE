# Variant of a32d8b75
# Dimension: D3 — 目标对象：框架模板的结构 (dst)
# Change: HxW掩膜（H=4-8, W=3-5），单色绘制，8-连通，活跃像素9-17个，填充率50-65%... → 框架使用多种颜色标记不同铺贴规则
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original_frame: "框架区域里所有非0/6像素都只是布尔活跃位，solver 只看 mask，不看颜色。"
#     new_frame: "框架区域固定使用两种语义色：8 表示正常铺贴，9 表示把单元精灵做水平镜像后再铺贴。"
#   concretization:
#     input:
#       legend_frame: "仍位于 unit sprite 和 marker square 之间，尺寸仍是 H=4-8, W=3-5。"
#       frame_colors:
#         8: "normal tile"
#         9: "mirror-left-right tile"
#       connectivity: "把 8/9 一起看成活跃像素后，整体必须 8-连通。"
#     output:
#       composite: "每个活跃格仍扩成一个 NxN 子块，但子块朝向由该格颜色决定。"
#       rotation: "clock-hand 仍只旋转框架布局，不旋转单元内部；这保持与原题同一条主干。"
#   behavior_trace:
#     micro_example:
#       unit: "[[1,2],[1,1]]"
#       frame_before_rotation: "[[8,9],[9,8]]"
#       old_logic: "四个格都贴 [[1,2],[1,1]]。"
#       new_logic: "8 格贴 [[1,2],[1,1]]；9 格贴 [[2,1],[1,1]]。"
#       difference: "同一复合图里会同时出现正向块和镜像块，且位置由框架颜色直接编码。"
#   complexity_calibration:
#     why_not_more_rules: "只引入两种铺贴行为，避免同时混入多种旋转/翻转导致训练样例不足以归纳。"
#     support_per_example: "每个框架都强制两种颜色都出现且各自重复出现，避免镜像规则只出现一次。"
#   implicit_assumption_audit:
#     - "原 main() 的 `frame_mask0 = (frame != 0) & (frame != 6)` 把所有框架颜色压平成布尔；这在变体里会丢失语义，必须改成保留颜色网格。"
#     - "原生成器只保存 `frame_mask`；现在必须保存 `frame_codes`，否则 solver/generator 会对同一输入给出不同输出。"
#     - "原题允许 unit 恰好左右对称；在变体里那会让镜像行为不可见，所以生成器要显式排除水平对称 unit。"
#     - "原 `_overlay_composites` 对每个活跃格都贴同一个 sprite；现在要按格子颜色选择 sprite 或 fliplr(sprite)。"
#   placement_legality:
#     - "复合图尺寸仍只由旋转后的框架外接尺寸与 N 决定，所以原 anchor / non-overlap 逻辑仍成立。"
#     - "generator 仍采用 output-first，并用新 main() 回验，确保不越界、不重叠。"
#   observability:
#     - "8/9 作为全任务固定语义色，跨样例保持一致，解题者能把输入里的框架颜色与输出里的子块朝向对应起来。"
#     - "镜像不是靠颜色 id 排序等不可见因素触发，而是靠可见的框架着色直接触发。"
#   cleanliness:
#     - "仍只在 legend 中新增一层颜色信息，没有往 canvas 叠加额外标记；画面杂讯基本不增加。"
#   validation_metric:
#     - "统计每个样本的 mirror frame cells 数量，以及与'把所有框架格都当 normal'的旧逻辑相比，输出发生变化的像素数；若后者经常为0，说明退化。"

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, neighbors8
from scipy.ndimage import label as _scipy_label


FRAME_NORMAL_COLOR = 8
FRAME_MIRROR_COLOR = 9
FRAME_RULE_COLORS = (FRAME_NORMAL_COLOR, FRAME_MIRROR_COLOR)

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)  # python range stop is exclusive => 4..8
FRAME_W_RANGE = (3, 6)  # 3..5
FRAME_ACTIVE_RANGE = (9, 17)
FRAME_FILL_RANGE = (0.50, 0.65)
FRAME_GEN_ATTEMPTS = 80
MIN_MIRROR_CELLS = 3
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
COLOR_POOL = [1, 2, 3, 4, 5, 7]  # reserve 8/9 for frame semantics
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
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
    """Identify the left legend separator and an optional right legend separator."""
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
    """List all 5x5 border-6 squares that look like the top of a marker panel."""
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
    """Extract the top-most 4-connected non-{0,6} component and undo the legend color-swap."""
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
        swapped = spr.copy()
        swapped[spr == a] = -1
        swapped[spr == b] = a
        swapped[swapped == -1] = b
        spr = swapped
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int) -> tuple[int, int, int, int] | None:
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


def _position_indices_from_border_square(legend: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
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
    dr, dc = int(p[0] - 1), int(p[1] - 1)
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


def _build_composite_from_frame(unit_sprite: np.ndarray, frame_codes: np.ndarray) -> np.ndarray:
    """Expand a frame of semantic colors into the final composite sprite."""
    unit_mask = (unit_sprite != 0) & (unit_sprite != 6)
    unit_mirror = np.fliplr(unit_sprite)
    mirror_mask = (unit_mirror != 0) & (unit_mirror != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_codes.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            code = int(frame_codes[i, j])
            if code == 0 or code == 6:
                continue
            rr0 = i * uh
            cc0 = j * uw
            if code == FRAME_NORMAL_COLOR:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][unit_mask] = unit_sprite[unit_mask]
            elif code == FRAME_MIRROR_COLOR:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][mirror_mask] = unit_mirror[mirror_mask]
            else:
                raise AssertionError(f"unexpected frame code: {code}")
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode legend panels and paint enlarged composites onto the central canvas.

    Input structure:
    - Full-height color-6 columns separate the left legend, the output canvas, and an optional right legend.
    - Each legend still contains four parts from top to bottom: a 2-color unit sprite, a frame template, a 5x5 anchor square, and a 3x3 clock-hand rotation marker.
    - Inside the frame template, color 8 and color 9 are both active cells, but they do not mean the same tiling action.

    Rule:
    - Read the unit sprite, undo its legend color-swap, and read the frame template between the sprite and marker panel.
    - Rotate the entire frame layout according to the clock-hand.
    - For every rotated frame cell: color 8 places the unit normally, while color 9 places a left-right mirrored copy of that same unit.
    - Anchor the finished composite to the canvas using the marker square's 3x3 position code.

    Key insight:
    - The frame is no longer a plain occupancy mask: its two visible colors encode two different local copy rules, so the solver must preserve frame colors all the way into composite construction.
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
        frame_active = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_active, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        frame_colors = {int(x) for x in np.unique(frame[frame_active])}
        assert frame_colors <= set(FRAME_RULE_COLORS), f"unexpected frame colors: {frame_colors}"
        assert FRAME_NORMAL_COLOR in frame_colors and FRAME_MIRROR_COLOR in frame_colors, "frame must use both rule colors"

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
    """Pick unit colors and two canvas background colors, reserving 8/9 for frame semantics."""
    pool = list(COLOR_POOL)
    random.shuffle(pool)
    unit_color_sets = []
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        unit_color_sets.append((pool.pop(), pool.pop()))
    if len(pool) < 2:
        return None
    bg_colors = (pool.pop(), pool.pop())
    return unit_color_sets, bg_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with its two colors swapped."""
    a, b = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
    return out


def _is_horizontal_asymmetric(sprite: np.ndarray) -> bool:
    return not np.array_equal(sprite, np.fliplr(sprite))


def _make_unit_sprite(n: int, colors: tuple[int, int]) -> np.ndarray:
    """Generate a filled NxN sprite whose horizontal mirror is visually different."""
    c1, c2 = colors
    for _ in range(200):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        if _is_horizontal_asymmetric(sprite):
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
    """Generate an 8-connected binary frame mask inside the original size constraints."""
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
        if _frame_mask_ok(mask):
            if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            return mask
    return None


def _mirror_assignments(mask: np.ndarray) -> list[np.ndarray]:
    """Candidate mirror subsets with repeated support across the frame."""
    r_idx, c_idx = np.indices(mask.shape)
    schemes = [
        ((r_idx + c_idx) % 2 == 0) & mask,
        ((r_idx + c_idx) % 2 == 1) & mask,
        (r_idx % 2 == 0) & mask,
        (r_idx % 2 == 1) & mask,
        (c_idx % 2 == 0) & mask,
        (c_idx % 2 == 1) & mask,
    ]
    valid: list[np.ndarray] = []
    active = int(mask.sum())
    for mm in schemes:
        count = int(mm.sum())
        if count < MIN_MIRROR_CELLS or active - count < MIN_MIRROR_CELLS:
            continue
        coords = np.argwhere(mm)
        if len({int(r) for r, _ in coords}) < 2 or len({int(c) for _, c in coords}) < 2:
            continue
        valid.append(mm)
    return valid


def _make_frame_codes(frame_h: int, frame_w: int) -> np.ndarray | None:
    """Generate a two-color semantic frame with both rule colors repeated."""
    for _ in range(FRAME_GEN_ATTEMPTS):
        mask = _sample_frame_mask(frame_h, frame_w)
        if mask is None:
            continue
        choices = _mirror_assignments(mask)
        if not choices:
            continue
        mirror_mask = random.choice(choices)
        codes = np.zeros((frame_h, frame_w), dtype=int)
        codes[mask] = FRAME_NORMAL_COLOR
        codes[mirror_mask] = FRAME_MIRROR_COLOR
        return codes
    return None


def _design_legend(leg: dict, unit_colors: tuple[int, int], marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    """Fill one legend record with a visible sprite, semantic frame, and rotation."""
    n = leg["N"]
    sprite = _make_unit_sprite(n, unit_colors)
    legend_sprite = _color_swap(sprite, unit_colors[0], unit_colors[1])

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

    random.shuffle(combos)
    for frame_h, frame_w, rot_k in combos:
        frame_codes = _make_frame_codes(frame_h, frame_w)
        if frame_codes is None:
            continue
        rotated_frame = np.rot90(frame_codes, (-rot_k) % 4)
        comp = _build_composite_from_frame(sprite, rotated_frame)
        baseline = _build_composite_from_frame(sprite, np.where(rotated_frame != 0, FRAME_NORMAL_COLOR, 0))
        if np.array_equal(comp, baseline):
            continue
        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame_codes=frame_codes,
            frame_H=frame_h,
            frame_W=frame_w,
            rot_k=rot_k,
            rotated_frame=rotated_frame,
            mirror_cells=int(np.sum(rotated_frame == FRAME_MIRROR_COLOR)),
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions while keeping final composites in-bounds and non-overlapping."""
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
    """Stamp each legend's composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite_from_frame(leg["sprite"], leg["rotated_frame"])
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    """Draw one legend panel with sprite, semantic frame, anchor square, and rotation patch."""
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
    active = leg["frame_codes"] != 0
    frame_region[active] = leg["frame_codes"][active]


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
