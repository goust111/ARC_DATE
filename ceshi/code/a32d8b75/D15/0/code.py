# Variant of a32d8b75
# Dimension: D15 — 处理流程的步骤顺序 (multi_step)
# Change: 五步流程：读取图例 -> 铺贴构建 -> 旋转 -> 确定位置 -> 放置... → 先放置后变换：在画布上原位构建复合精灵

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite

# Design analysis for the D15 variant
# a) Concrete behavior and trace:
#    The legend is still read exactly as before: a two-color unit sprite, a frame mask,
#    a 3x3 anchor marker inside a 5x5 border-6 square, and a clock-hand rotation mark.
#    The changed step order is implemented by introducing a square work region on the
#    canvas. The unrotated enlarged frame is first assigned to this region by anchor,
#    the unit sprite is tiled in-place into active frame cells, and only then is the
#    whole work region rotated. Example: if the unrotated enlarged frame is 12x20 and
#    the marker asks for bottom-right anchoring with a 90-degree turn, the old task
#    would rotate the 12x20 composite to 20x12 and anchor that rotated result. The new
#    task instead anchors a 20x20 work square first, paints the 12x20 content into its
#    upper-left corner, and rotates the whole 20x20 patch. This moves both sprite pixels
#    and the local background inside the patch, making the change visible in output.
# b) Complexity calibration:
#    The 3x3 marker still has only three vertical and three horizontal states, and the
#    clock-hand still has four directions. To keep the new rule visible, the generator
#    only accepts legend settings where either the enlarged frame is non-square or the
#    rotation is a quarter-turn; otherwise the new and old executions can coincide.
# c) Old implicit assumptions that no longer hold:
#    The original solver rotated the frame mask before placement and computed anchor
#    offsets from the rotated composite size. Those lines assumed "placement depends on
#    the final rotated bbox". In this variant that assumption is false: anchor depends on
#    the pre-rotation work region, so size calculations, overlap checks, and writes must
#    all use the square work region instead of the final composite bbox.
# d) Placement legality:
#    Generator placement is done on work-region squares, not on final colored pixels.
#    Two legends are accepted only if their work squares are disjoint and fully inside
#    the canvas. This is stronger than non-overlap of final composites, but it matches
#    the new solver because each legend rotates its whole assigned square in place.
# e) Visual observability:
#    The canvas keeps the original banded background so the local rotation is readable.
#    A quarter-turn changes stripe orientation inside the work square; a non-square frame
#    introduces blank padding that causes the final sprite position inside the square to
#    differ from the old task even for 180-degree turns.
# f) Visual cleanliness:
#    Only one rotated work square is allowed per canvas location, so no patch can rotate
#    through another patch. Frame color is still only a legend cue; it never remains in
#    the output, so the final image is a clean sprite-over-background composition.
# g) Color inferability:
#    Colors are unchanged from the source task: the unit sprite is color-swapped on read,
#    frame color is separate from unit colors, and background colors are separate from
#    unit colors. No sorting or branching depends on numeric color ids.


def _find_border6_square_start(legend: np.ndarray) -> tuple[int, int] | None:
    h, w = legend.shape
    if h < 5 or w < 5:
        return None
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            sub = legend[r0 : r0 + 5, c0 : c0 + 5]
            if np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6):
                return int(r0), int(c0)
    return None


def _find_marker_panel_square(g: np.ndarray, sep_c1: int) -> tuple[int, int] | None:
    """
    Find the 5x5 border-6 position-marker square that belongs to the 8x5 marker panel.

    In some instances the marker panel can appear inside the main canvas as well as in the
    left legend region. Prefer a candidate that lies in the main canvas (col > sep_c1) if any.
    """
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
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
            candidates.append((r0, c0))
    if not candidates:
        return None
    main_cands = [rc for rc in candidates if rc[1] > sep_c1]
    return min(main_cands) if main_cands else min(candidates)


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


def _find_marker_square_with_preferences(
    g: np.ndarray, left_sep: tuple[int, int], right_sep: tuple[int, int] | None
) -> tuple[int, int] | None:
    """
    Find the marker panel's 5x5 border-6 square.

    Preference order:
      1) left legend region (at/before left separator)
      2) right legend region (at/after right separator), if present
      3) anywhere (including inside the output canvas)
    """
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
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
            candidates.append((r0, c0))
    if not candidates:
        return None
    left_end = left_sep[1]
    left_cands = [rc for rc in candidates if rc[1] + 2 <= left_end]
    if left_cands:
        return min(left_cands)
    if right_sep is not None:
        right_start = right_sep[0]
        right_cands = [rc for rc in candidates if rc[1] + 2 >= right_start]
        if right_cands:
            return min(right_cands)
    return min(candidates)


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of an 8x5 marker panel."""
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


def _build_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    m_unit = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = unit_sprite[m_unit]
    return comp


def _work_region_side(comp_shape: tuple[int, int]) -> int:
    return max(int(comp_shape[0]), int(comp_shape[1]))


def _apply_place_then_rotate(
    canvas: np.ndarray,
    unit_sprite: np.ndarray,
    frame_mask: np.ndarray,
    top: int,
    left: int,
    dir_k: int,
) -> np.ndarray:
    out = canvas.copy()
    comp = _build_composite(unit_sprite, frame_mask)
    ch, cw = comp.shape
    side = _work_region_side(comp.shape)

    region = out[top : top + side, left : left + side].copy()
    assert region.shape == (side, side), "work region must stay inside canvas"
    sub = region[:ch, :cw]
    write_mask = (comp != 0) & (comp != 6) & (sub != 6)
    sub[write_mask] = comp[write_mask]
    region[:ch, :cw] = sub
    out[top : top + side, left : left + side] = np.rot90(region, (-dir_k) % 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each legend panel and rebuild its enlarged sprite directly inside a canvas work area.

    Input structure:
    - Full-height color-6 separator columns still split left/right legends from the central canvas.
    - Each legend still contains, from top to bottom, a two-color unit sprite, a frame template,
      a 5x5 border-6 anchor square, and a 3x3 clock-hand rotation marker below it.

    Rule:
    - Read the unit sprite and the unrotated frame mask, compute the anchor first, and reserve a square
      work region on the canvas at that anchor.
    - Tile the decoded unit sprite into every active frame cell in that work region without rotating it first.
    - Rotate the entire occupied work region, including its local background, using the clock-hand direction.

    Key insight:
    - The 3x3 marker no longer points to the final rotated composite bounds; it chooses where the pre-rotation
      work square sits, so the later rotation moves both the sprite and the background inside that square.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    left_sep_c1 = left_sep[1]
    canvas_c0 = left_sep_c1 + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main.shape
    full6_cols = [c for c in range(eff_w) if np.all(main[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main.copy()
    work_region_written = np.zeros(out.shape, dtype=bool)

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
        frame_mask = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos

        comp = _build_composite(unit_sprite, frame_mask)
        side = _work_region_side(comp.shape)
        top = _anchor_to_top_left(pr, eff_h, side)
        left = _anchor_to_top_left(pc, eff_w, side)
        assert 0 <= top and 0 <= left, "work region placed above canvas origin"
        assert top + side <= out.shape[0] and left + side <= out.shape[1], "work region out of canvas bounds"
        assert not np.any(work_region_written[top : top + side, left : left + side]), "work regions must not overlap"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        out = _apply_place_then_rotate(out, unit_sprite, frame_mask, top, left, dir_k)
        work_region_written[top : top + side, left : left + side] = True
    return out


# --------------- generator (refine_8conn_fill_overlap) ---------------

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
GEN_RETRIES = 300


def _allocate_colors(num_legends):
    """Pick unit colors (2 per legend), 2 background colors, and 1 frame color per legend."""
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


def _color_swap(sprite, c1, c2):
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w):
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((N, N), uc1, dtype=int)
    num_c2 = random.randint(1, N * N - 1)
    for p in random.sample(range(N * N), num_c2):
        sprite[p // N, p % N] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (N + 1)
    combos = []
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            comp_h = fH * N
            comp_w = fW * N
            side = max(comp_h, comp_w)
            if side > grid_h or side > canvas_w:
                continue
            for rk in range(4):
                if fH == fW and rk in (0, 2):
                    continue
                combos.append((fH, fW, rk))
    if not combos:
        return False
    frame_H, frame_W, rot_k = random.choice(combos)

    frame_mask = random_sprite(
        frame_H,
        frame_W,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0

    comp_shape = (frame_H * N, frame_W * N)
    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        comp_shape=comp_shape,
        work_side=_work_region_side(comp_shape),
    )
    return True


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign work-region anchors to legends, avoiding overlap for the new in-place rotation."""
    if len(legends) == 1:
        pr, pc = random.randint(0, 2), random.randint(0, 2)
        side = legends[0]["work_side"]
        legends[0].update(
            pr=pr,
            pc=pc,
            work_top=_anchor_to_top_left(pr, grid_h, side),
            work_left=_anchor_to_top_left(pc, canvas_w, side),
        )
        return True

    a, b = legends[0], legends[1]
    a_side = a["work_side"]
    b_side = b["work_side"]
    valid_pairs = []
    for apr in range(3):
        for apc in range(3):
            at = _anchor_to_top_left(apr, grid_h, a_side)
            al = _anchor_to_top_left(apc, canvas_w, a_side)
            if at < 0 or al < 0 or at + a_side > grid_h or al + a_side > canvas_w:
                continue
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_side)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_side)
                    if bt < 0 or bl < 0 or bt + b_side > grid_h or bl + b_side > canvas_w:
                        continue
                    if at < bt + b_side and bt < at + a_side and al < bl + b_side and bl < al + a_side:
                        continue
                    valid_pairs.append((apr, apc, at, al, bpr, bpc, bt, bl))
    if not valid_pairs:
        return False
    apr, apc, at, al, bpr, bpc, bt, bl = random.choice(valid_pairs)
    a.update(pr=apr, pc=apc, work_top=at, work_left=al)
    b.update(pr=bpr, pc=bpc, work_top=bt, work_left=bl)
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
    """Apply the variant rule: place unrotated content into anchored work squares, then rotate each square."""
    output = canvas.copy()
    for leg in legends:
        output = _apply_place_then_rotate(
            output,
            leg["sprite"],
            leg["frame_mask"],
            leg["work_top"],
            leg["work_left"],
            leg["rot_k"],
        )
    return output


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        N = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": N, "legend_w": N + 2})

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


def _place_legend(grid, leg, col_start, sep_col, marker_r0, is_right):
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
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


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
