# Variant of a32d8b75
# Dimension: D3 — 目标对象：框架模板的结构 (dst)
# Change: HxW掩膜（H=4-8, W=3-5），单色绘制，8-连通，活跃像素9-17个，填充率50-65%... → 框架为嵌套层次结构

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label


# Design analysis
# - Reference-task note: the prompt cites `b99e7126` as the fractal-scaling source, but
#   in this workspace only `/ARC-data/.../b99e7126.json` is readable; no solver source is
#   present locally, so the implementation below uses the source-task idea only at the
#   conceptual level and keeps the rest grounded in the original `a32d8b75` code/spec.
# - a) Concrete variant:
#   - Original behavior: `frame_mask[i,j] = 1` means “place one unit sprite block here”.
#   - New behavior: `outer_frame[i,j] = 1` means “place one entire inner-frame block here”,
#     and each active inner-frame cell then receives one unit sprite block.
#   - Small trace:
#     - Original: a 2x3 frame with actives at {(0,0),(0,2),(1,1)} yields three unit blocks.
#     - Variant: the same three outer actives each receive, for example, a 2x2 L-shaped
#       inner frame; each of those inner actives then receives the unit sprite. So one outer
#       active no longer corresponds to one unit, but to a whole cluster of units.
# - b) Complexity calibration:
#   - The visual rule must be obvious from a few samples, so the legend explicitly contains
#     two separated frame components stacked vertically: outer frame first, inner frame
#     second. This avoids ambiguous inference from a single blended mask.
#   - Freedom is limited to 1-2 legends, unit size 2-3, outer frame roughly 2-4 by 2-3,
#     inner frame 2-3 by 2-3. This keeps the fractal visible without exploding canvas size.
# - c) Hidden assumptions from the original solver that no longer hold:
#   - `_extract_frame_bbox(...)` assumed every non-zero pixel between sprite and marker
#     belonged to one contiguous frame. That breaks here because there are now two separate
#     frame components, so the new solver labels 8-connected components and reads the top one
#     as the outer frame and the next one as the inner frame.
#   - The original composite builder assumed one placement scale `frame_cell -> unit`.
#     That must be replaced by two scales: `outer_cell -> inner composite -> unit`.
#   - The original rotation acted on the flat frame layout only. To keep the change local to
#     D3, the new code preserves that convention by rotating only the outer frame layout; the
#     inner frame and the unit sprite remain in their read orientation.
# - d) Placement legality:
#   - Generator works output-first and rejects any legend whose nested composite would not fit
#     its anchor location. For two legends it rejects overlapping bounding boxes.
#   - The legend construction guarantees two frame components are vertically separated by at
#     least one zero row, remain inside a 5-column panel, and stay outside the main canvas.
# - e) Visual observability:
#   - The variant is observable because the output is no longer a single enlarged silhouette.
#     Each macro branch decomposes into repeated sub-branches before the unit texture appears.
#   - Masks are required to have holes / missing cells, so the nested repetition is visible.
# - f) Visual cleanliness:
#   - The legend panel stays narrow and orderly: sprite at top, outer frame, inner frame,
#     then marker square and rotation patch. No elements overlap.
#   - The canvas still uses simple banded backgrounds, so the fractal composite remains easy
#     to isolate by eye.
# - g) Color inference:
#   - No invisible color ordering is used. The frame color is irrelevant to the rule and only
#     marks both frame components as “template parts”. The meaningful transformation is purely
#     structural: one frame now expands into another frame before reaching the unit sprite.
# - Post-change solver assumptions to audit:
#   - Exactly one unit sprite component appears above the frames.
#   - Exactly two 8-connected frame components lie between sprite and marker, ordered from
#     top to bottom as outer then inner.
#   - The nested composite fully fits in the canvas and legends do not overlap on output.
#   - `generate_input()` explicitly constructs inputs satisfying all of these assumptions.


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
    cols = sorted({int(x) for x in np.unique(spr) if int(x) not in (0, 6)})
    if len(cols) == 2:
        a, b = cols
        spr2 = spr.copy()
        spr2[spr == a] = -1
        spr2[spr == b] = a
        spr2[spr2 == -1] = b
        spr = spr2
    return spr, (r0, c0, r1, c1)


def _extract_nested_frames(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], marker_r0: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read two vertically ordered 8-connected frame components: outer first, inner second."""
    r_start = unit_bb[2] + 1
    r_end = marker_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    labeled, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    if n8 < 2:
        return None

    comps: list[tuple[int, int, int, int]] = []
    for idx in range(1, n8 + 1):
        pts = np.argwhere(labeled == idx)
        if pts.size == 0:
            continue
        pts[:, 0] += r_start
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        comps.append((r0, c0, r1, c1))
    comps.sort(key=lambda bb: (bb[0], bb[1], bb[2], bb[3]))
    outer_bb, inner_bb = comps[:2]

    def _crop_mask(bb: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bb
        piece = legend[r0 : r1 + 1, c0 : c1 + 1]
        out = (piece != 0) & (piece != 6)
        _, piece_n8 = _scipy_label(out, structure=np.ones((3, 3), dtype=int))
        assert piece_n8 == 1, "each frame component must be 8-connected"
        return out

    return _crop_mask(outer_bb), _crop_mask(inner_bb)


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
    Read the 3x3 clock-hand marker below the 5x5 anchor square.

    Use the furthest non-{0,6} pixel from the center to determine direction:
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


def _build_inner_composite(unit_sprite: np.ndarray, inner_mask: np.ndarray) -> np.ndarray:
    """Tile the unit sprite into each active cell of the inner frame."""
    m_unit = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    ih, iw = inner_mask.shape
    comp = np.zeros((ih * uh, iw * uw), dtype=int)
    for i in range(ih):
        for j in range(iw):
            if not inner_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = unit_sprite[m_unit]
    return comp


def _build_nested_composite(
    unit_sprite: np.ndarray, outer_mask: np.ndarray, inner_mask: np.ndarray, dir_k: int
) -> np.ndarray:
    """
    Build a three-level composite:
      outer frame cell -> one inner-frame composite -> unit sprites.

    To keep the D3 change local, only the outer frame layout is rotated by the marker,
    matching the original task's “rotate the frame layout, not each unit sprite” rule.
    """
    inner_comp = _build_inner_composite(unit_sprite, inner_mask)
    rot_outer = np.rot90(outer_mask, (-dir_k) % 4)
    sh, sw = inner_comp.shape
    oh, ow = rot_outer.shape
    comp = np.zeros((oh * sh, ow * sw), dtype=int)
    block_mask = inner_comp != 0
    for i in range(oh):
        for j in range(ow):
            if not rot_outer[i, j]:
                continue
            rr0 = i * sh
            cc0 = j * sw
            comp[rr0 : rr0 + sh, cc0 : cc0 + sw][block_mask] = inner_comp[block_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct recursively nested sprite clusters from legend panels and place them on the canvas.

    Input structure:
    - Full-height color-6 separator columns isolate the central canvas from one or two 5-column legend panels.
    - Each legend panel contains, top to bottom, a 2-color unit sprite, an outer frame mask, an inner frame mask, a 5x5 anchor square, and a 3x3 rotation clock-hand.

    Rule:
    - Swap the two unit-sprite colors when reading the legend, tile that unit into every active cell of the inner frame, then tile the resulting inner-frame composite into every active cell of the outer frame.
    - Rotate only the outer frame layout according to the clock-hand direction, preserving the original task's “layout rotation without per-unit rotation” convention.
    - Place the final nested composite on the canvas using the anchor encoded by the marker square.

    Key insight:
    - The frame part of the legend is no longer a single flat mask; it is a template-of-templates, so each active outer-frame cell expands into a whole smaller frame before individual unit sprites appear.
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

        frames = _extract_nested_frames(panel, unit_bb, marker_r0)
        if frames is None:
            continue
        outer_mask0, inner_mask0 = frames

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = _build_nested_composite(unit_sprite, outer_mask0, inner_mask0, dir_k)

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


# --------------- generator (nested_frame_fractal) ---------------

GRID_H_RANGE = (22, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [3, 2]  # weights for [1-legend, 2-legend]
LEGEND_PANEL_W = 5
UNIT_SIZE_RANGE = (2, 3)
OUTER_FRAME_H_RANGE = (2, 4)
OUTER_FRAME_W_RANGE = (2, 3)
INNER_FRAME_H_RANGE = (2, 3)
INNER_FRAME_W_RANGE = (2, 3)
MARKER_PANEL_HEIGHT = 8
BAND_WIDTH_MIN = 4
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400


def _allocate_colors(num_legends: int):
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

    all_set = set(USABLE_COLORS)
    frame_colors = []
    for uc in unit_color_sets:
        frame_colors.append(random.choice(list(all_set - set(uc))))
    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _is_rotation_variant(arr: np.ndarray) -> bool:
    """Reject patterns that look the same under a non-trivial rotation."""
    return not any(np.array_equal(arr, np.rot90(arr, k)) for k in (1, 2, 3))


def _make_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray:
    """Build a fully filled 2-color NxN sprite with visible asymmetry."""
    for _ in range(200):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        if _is_rotation_variant(sprite):
            return sprite
    return sprite


def _generate_frame_mask(h: int, w: int, min_active: int, max_active: int) -> np.ndarray:
    """Generate an 8-connected non-full mask with visible structure."""
    max_active = min(max_active, h * w - 1)
    for _ in range(300):
        mask = random_sprite(
            h,
            w,
            density=0.6,
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=8,
        ) != 0
        active = int(mask.sum())
        if active < min_active or active > max_active:
            continue
        if active == h * w:
            continue
        if _is_rotation_variant(mask):
            return mask
    raise RuntimeError("failed to generate a suitable frame mask")


def _comp_size(leg: dict) -> tuple[int, int]:
    """Return the nested composite size after rotating the outer frame."""
    oh, ow = leg["rotated_outer"].shape
    ih, iw = leg["inner_mask"].shape
    n = leg["N"]
    return oh * ih * n, ow * iw * n


def _design_legend(leg: dict, unit_colors, frame_color: int, marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    """Fill one legend with a sprite plus two stacked frame masks."""
    n = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _make_unit_sprite(n, uc1, uc2)
    legend_sprite = _color_swap(sprite, uc1, uc2)

    combos: list[tuple[int, int, int, int, int]] = []
    for outer_h in range(OUTER_FRAME_H_RANGE[0], OUTER_FRAME_H_RANGE[1] + 1):
        for outer_w in range(OUTER_FRAME_W_RANGE[0], OUTER_FRAME_W_RANGE[1] + 1):
            for inner_h in range(INNER_FRAME_H_RANGE[0], INNER_FRAME_H_RANGE[1] + 1):
                for inner_w in range(INNER_FRAME_W_RANGE[0], INNER_FRAME_W_RANGE[1] + 1):
                    min_rows_used = 1 + n + 1 + outer_h + 1 + inner_h
                    if min_rows_used > marker_r0:
                        continue
                    for rot_k in range(4):
                        rot_outer_h, rot_outer_w = (outer_h, outer_w) if rot_k % 2 == 0 else (outer_w, outer_h)
                        comp_h = rot_outer_h * inner_h * n
                        comp_w = rot_outer_w * inner_w * n
                        if comp_h <= grid_h and comp_w <= canvas_w:
                            combos.append((outer_h, outer_w, inner_h, inner_w, rot_k))
    random.shuffle(combos)
    if not combos:
        return False

    for outer_h, outer_w, inner_h, inner_w, rot_k in combos:
        try:
            outer_mask = _generate_frame_mask(outer_h, outer_w, min_active=3, max_active=6)
            inner_mask = _generate_frame_mask(inner_h, inner_w, min_active=3, max_active=5)
        except RuntimeError:
            continue

        spare = marker_r0 - (1 + n + 1 + outer_h + 1 + inner_h)
        extra1 = random.randint(0, min(2, spare))
        spare -= extra1
        extra2 = random.randint(0, min(2, spare))
        spare -= extra2

        gap1 = 1 + extra1
        gap2 = 1 + extra2
        outer_r0 = 1 + n + gap1
        inner_r0 = outer_r0 + outer_h + gap2
        outer_c0 = random.randint(0, LEGEND_PANEL_W - outer_w)
        inner_c0 = random.randint(0, LEGEND_PANEL_W - inner_w)
        sprite_c0 = random.randint(0, LEGEND_PANEL_W - n)
        rotated_outer = np.rot90(outer_mask, (-rot_k) % 4)

        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            sprite_c0=sprite_c0,
            frame_color=frame_color,
            outer_mask=outer_mask,
            inner_mask=inner_mask,
            outer_r0=outer_r0,
            outer_c0=outer_c0,
            inner_r0=inner_r0,
            inner_c0=inner_c0,
            rot_k=rot_k,
            rotated_outer=rotated_outer,
        )
        return True
    return False


def _build_nested_composite_from_leg(leg: dict) -> np.ndarray:
    """Build the output composite for one legend."""
    return _build_nested_composite(leg["sprite"], leg["outer_mask"], leg["inner_mask"], leg["rot_k"])


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions, avoiding overlap for the 2-legend case."""
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
    """Stamp each legend's nested composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_nested_composite_from_leg(leg)
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int) -> None:
    """Draw one 5-column legend panel plus its separator column."""
    n = leg["N"]
    grid[:, sep_col] = 6

    panel = grid[:, col_start : col_start + LEGEND_PANEL_W]
    panel[:] = 0

    panel[marker_r0 : marker_r0 + 5, :] = 0
    panel[marker_r0, :] = 6
    panel[marker_r0 + 4, :] = 6
    panel[marker_r0 : marker_r0 + 5, 0] = 6
    panel[marker_r0 : marker_r0 + 5, 4] = 6
    panel[marker_r0 + 1 + leg["pr"], 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    panel[rr0 : rr0 + 3, :] = 0
    panel[rr0 : rr0 + 3, 0] = 6
    panel[rr0 : rr0 + 3, 4] = 6
    cx_r, cx_c = rr0 + 1, 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    panel[cx_r, cx_c] = 7
    panel[cx_r + dr, cx_c + dc] = 7

    panel[1 : 1 + n, leg["sprite_c0"] : leg["sprite_c0"] + n] = leg["legend_sprite"]

    outer_region = panel[
        leg["outer_r0"] : leg["outer_r0"] + leg["outer_mask"].shape[0],
        leg["outer_c0"] : leg["outer_c0"] + leg["outer_mask"].shape[1],
    ]
    outer_region[leg["outer_mask"]] = leg["frame_color"]

    inner_region = panel[
        leg["inner_r0"] : leg["inner_r0"] + leg["inner_mask"].shape[0],
        leg["inner_c0"] : leg["inner_c0"] + leg["inner_mask"].shape[1],
    ]
    inner_region[leg["inner_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        legends.append({"N": random.randint(*UNIT_SIZE_RANGE)})

    left_total = LEGEND_PANEL_W + 1
    right_total = (1 + LEGEND_PANEL_W) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w <= 0:
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

    _place_legend(grid, legends[0], col_start=0, sep_col=LEGEND_PANEL_W, marker_r0=marker_r0)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0)

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
