# Combo variant of a32d8b75 (index 15)
# combo_spec: D14/2+D15/0
# Name: 分隔符编码信息+先放后转

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite, scale_sprite

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The fused puzzle keeps the same visible legend ingredients as the source task:
#    a two-color unit sprite, a frame mask, a 5x5 anchor square, and a 3x3 clock-hand.
#    Two things now interact. First, the full-height color-6 separator is a run whose
#    width s is 1..3 and encodes the sprite scale. Second, placement order changes:
#    do not rotate the tiled composite before anchoring. Example: let the unit be 2x2,
#    separator width s=3, and frame size be 2x3. The decoded unit first scales to 6x6,
#    so the unrotated tiled composite is 12x18. Instead of rotating that 12x18 object
#    first and anchoring the rotated 18x12 result, the solver reserves an 18x18 work
#    square on the canvas, paints the 12x18 content into its upper-left corner, and
#    only then rotates the whole 18x18 patch. The separator width changes the size of
#    both the painted content and the work square, so the two dimensions truly couple.
# b) Complexity calibration:
#    The hidden width code now uses all three visible states {1,2,3}, but favors 1 and 2
#    slightly because those are the scales that can also support the dual-legend branch on
#    a 30-column board. Rotation is generated only as a quarter-turn, so the place-then-
#    rotate stage is always readable in the background. This keeps inference discrete and
#    repeated without introducing too many orthogonal freedoms.
# c) Broken assumptions and code impact:
#    - Old logic assumed every separator was one inert column; now separator width must
#      be inferred and verified, and layout math becomes legend+s+canvas(+s+legend).
#    - Old assembly used the raw unit sprite size. Here every frame cell receives the
#      scaled unit, so all fit checks and composite dimensions must use N*s.
#    - Old placement anchored the final rotated bounding box. That is no longer valid:
#      anchors now address the pre-rotation square work region, so overlap checks must
#      track those squares rather than final colored pixels.
#    - Multi-column separators introduce a new hazard: right-side legends must not draw
#      non-6 pixels into separator columns, or the separator run would stop being a run.
#      The generator therefore aligns right marker panels so they touch at most the last
#      separator column, which remains border-6 throughout.
# d) Placement legality:
#    Generator placement is output-first. It picks separator width, scaled composite
#    sizes, and square work regions before assembling the input. Every work square must
#    fit inside the canvas. With two legends, work squares are forced to be disjoint,
#    which is stricter than final sprite non-overlap but exactly matches the new solver.
# e) Visual observability:
#    Humans can see both controls. Thicker separators produce larger repeated units, and
#    quarter-turn rotation visibly rotates the local striped background inside each work
#    square. Comparing a few samples is enough to notice that separator width is not mere
#    spacing, and that anchoring happens before rotation because the rotated background
#    patch stays aligned with the sprite's local square.
# f) Visual cleanliness:
#    The legend remains uncluttered because no extra marker colors are introduced. The
#    output stays readable because each legend owns one disjoint square patch; there is
#    no partial overlap of rotated regions, and frame colors never survive into output.
# g) Inferability of changes:
#    No branch depends on numeric color ordering. Colors keep their original semantic
#    roles, while the new hidden variable is purely geometric: separator width controls
#    scale, and the 3x3 anchor still controls only one square patch location.


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


def _separator_scale(left_sep: tuple[int, int], right_sep: tuple[int, int] | None) -> int:
    """Infer the hidden scale from separator width and verify consistency."""
    scale = left_sep[1] - left_sep[0] + 1
    assert 1 <= scale <= 3, "separator width must encode scale 1..3"
    if right_sep is not None:
        right_scale = right_sep[1] - right_sep[0] + 1
        assert right_scale == scale, "all separator runs must share one width code"
    return scale


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
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """Read up/right/down/left from the 3x3 clock-hand patch below the border square."""
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
    """Map a start/middle/end anchor code to the top-left coordinate of the placed object."""
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
    """Tile the unit sprite into each active frame cell without rotating the frame."""
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
    """Place the unrotated composite into a square work region, then rotate the region."""
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
    Decode each legend panel, infer the separator-coded scale, and rebuild the sprite inside a canvas work square.

    Input structure:
    - Full-height color-6 separator runs split left/right legend panels from the central canvas, and all separator runs in one puzzle share one width from 1 to 3.
    - Each legend panel still contains a two-color unit sprite, a frame template, a 5x5 border-6 anchor square, and a 3x3 clock-hand rotation marker beneath it.

    Rule:
    - Read the separator width as the hidden scale, enlarge the decoded unit sprite by that factor, and tile the scaled unit into every active cell of the unrotated frame.
    - Use the anchor marker to place the resulting unrotated composite into a square work region on the canvas, then rotate that entire square region, including its local background, by the clock-hand direction.

    Key insight:
    - The separator no longer just divides areas: its thickness sets the scale of the repeated unit, and because anchoring happens before rotation, the marker chooses a pre-rotation square patch rather than the final rotated sprite bounds.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    sep_scale = _separator_scale(left_sep, right_sep)
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
        scaled_unit = scale_sprite(unit_sprite, sep_scale)

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

        comp = _build_composite(scaled_unit, frame_mask)
        side = _work_region_side(comp.shape)
        top = _anchor_to_top_left(pr, eff_h, side)
        left = _anchor_to_top_left(pc, eff_w, side)
        assert 0 <= top and 0 <= left, "work region placed above canvas origin"
        assert top + side <= out.shape[0] and left + side <= out.shape[1], "work region out of canvas bounds"
        assert not np.any(work_region_written[top : top + side, left : left + side]), "work regions must not overlap"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        out = _apply_place_then_rotate(out, scaled_unit, frame_mask, top, left, dir_k)
        work_region_written[top : top + side, left : left + side] = True
    return out


# --------------- generator (separator_scale_place_then_rotate) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
# When both branches are feasible, bias toward 2 legends to offset its lower generation success rate.
LEGEND_COUNT_WEIGHTS = [1, 4]
SEPARATOR_WIDTH_CHOICES = [1, 2, 3]
SEPARATOR_WIDTH_WEIGHTS = [2, 2, 1]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (3, 9)  # allow sep_scale=3 to remain physically realizable on a 30-col board
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
QUARTER_TURN_K = [1, 3]
GEN_RETRIES = 500


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


def _legend_has_fit(N: int, sep_scale: int, marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    """Check whether any frame can fit after scaling and square work-region anchoring."""
    avail = marker_r0 - (N + 1)
    scaled_n = N * sep_scale
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            side = max(fH * scaled_n, fW * scaled_n)
            if side <= grid_h and side <= canvas_w:
                return True
    return False


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w, sep_scale):
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((N, N), uc1, dtype=int)
    num_c2 = random.randint(1, N * N - 1)
    for p in random.sample(range(N * N), num_c2):
        sprite[p // N, p % N] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)
    scaled_sprite = scale_sprite(sprite, sep_scale)

    avail = marker_r0 - (N + 1)
    combos = []
    scaled_n = N * sep_scale
    usable_w = leg["frame_panel_col_hi"] - leg["frame_panel_col_lo"] + 1
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], usable_w + 1)):
            side = max(fH * scaled_n, fW * scaled_n)
            if side <= grid_h and side <= canvas_w:
                combos.append((fH, fW))
    if not combos:
        return False
    frame_H, frame_W = random.choice(combos)
    rot_k = random.choice(QUARTER_TURN_K)

    frame_mask = random_sprite(
        frame_H,
        frame_W,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0

    comp_shape = (frame_H * scaled_sprite.shape[0], frame_W * scaled_sprite.shape[1])
    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        scaled_sprite=scaled_sprite,
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        comp_shape=comp_shape,
        work_side=_work_region_side(comp_shape),
        sep_scale=sep_scale,
    )
    return True


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign work-region anchors while keeping square work regions disjoint."""
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


def _overlay_composites(canvas: np.ndarray, legends) -> np.ndarray:
    """Apply the combo rule: scale, place unrotated content into a work square, then rotate."""
    output = canvas.copy()
    for leg in legends:
        output = _apply_place_then_rotate(
            output,
            leg["scaled_sprite"],
            leg["frame_mask"],
            leg["work_top"],
            leg["work_left"],
            leg["rot_k"],
        )
    return output


def _candidate_n_sets(num_legends: int, sep_scale: int, grid_h: int, marker_r0: int) -> list[tuple[int, ...]]:
    """Enumerate unit sizes that leave a feasible canvas and at least one valid work square."""
    out: list[tuple[int, ...]] = []
    for n1 in range(UNIT_SIZE_RANGE[0], UNIT_SIZE_RANGE[1] + 1):
        left_lw = n1 + 2
        if num_legends == 1:
            canvas_w = GRID_W - left_lw - sep_scale
            if canvas_w < MIN_CANVAS_W:
                continue
            if _legend_has_fit(n1, sep_scale, marker_r0, grid_h, canvas_w):
                out.append((n1,))
            continue

        for n2 in range(UNIT_SIZE_RANGE[0], UNIT_SIZE_RANGE[1] + 1):
            right_lw = n2 + 2
            canvas_w = GRID_W - left_lw - sep_scale - sep_scale - right_lw
            if canvas_w < MIN_CANVAS_W:
                continue
            if not _legend_has_fit(n1, sep_scale, marker_r0, grid_h, canvas_w):
                continue
            if not _legend_has_fit(n2, sep_scale, marker_r0, grid_h, canvas_w):
                continue
            out.append((n1, n2))
    return out


def _place_legend(grid, leg, legend_start, sep_start, sep_scale, marker_r0, is_right):
    """Draw one legend panel while preserving every separator column as full-height 6."""
    N = leg["N"]
    grid[:, sep_start : sep_start + sep_scale] = 6

    if not is_right:
        mc0 = legend_start + max(0, N - 4)
    else:
        sep_last = sep_start + sep_scale - 1
        mc0 = sep_last + max(0, N - 3)

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

    sprite_start = legend_start + 1
    grid[1 : 1 + N, sprite_start : sprite_start + N] = leg["legend_sprite"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (N + 1)
    gap = random.randint(1, max(1, avail - fH))
    fr_start = N + 1 + gap
    f_col_min = leg["frame_panel_col_lo"]
    f_col_max = leg["frame_panel_col_hi"] - fW + 1
    f_col = random.randint(f_col_min, f_col_max)
    frame_region = grid[fr_start : fr_start + fH, mc0 + f_col : mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    sep_scale = random.choices(SEPARATOR_WIDTH_CHOICES, weights=SEPARATOR_WIDTH_WEIGHTS)[0]

    feasible = []
    for num_legends in (1, 2):
        n_sets = _candidate_n_sets(num_legends, sep_scale, grid_h, marker_r0)
        if n_sets:
            feasible.append((num_legends, n_sets))
    if not feasible:
        return None

    counts = [x[0] for x in feasible]
    weights = [LEGEND_COUNT_WEIGHTS[n - 1] for n in counts]
    num_legends, n_sets = random.choices(feasible, weights=weights)[0]
    chosen_ns = random.choice(n_sets)
    legends = []
    for idx, N in enumerate(chosen_ns):
        frame_panel_col_lo = 0
        frame_panel_col_hi = 4
        if idx == 0 and N <= 2:
            frame_panel_col_hi = 3
        if idx == 1 and N <= 3:
            frame_panel_col_lo = 1
        legends.append(
            {
                "N": N,
                "legend_w": N + 2,
                "frame_panel_col_lo": frame_panel_col_lo,
                "frame_panel_col_hi": frame_panel_col_hi,
            }
        )

    left_lw = legends[0]["legend_w"]
    right_total = (sep_scale + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_lw - sep_scale - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
            sep_scale,
        ):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_lw + sep_scale
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(
        grid,
        legends[0],
        legend_start=0,
        sep_start=left_lw,
        sep_scale=sep_scale,
        marker_r0=marker_r0,
        is_right=False,
    )
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(
            grid,
            legends[1],
            legend_start=r_sep + sep_scale,
            sep_start=r_sep,
            sep_scale=sep_scale,
            marker_r0=marker_r0,
            is_right=True,
        )

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
