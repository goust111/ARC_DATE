# Combo variant of a32d8b75 (index 13)
# combo_spec: D11/0+D17/0
# Name: 无颜色交换+颜色可重叠
#
# Design analysis for the combo variant
# a) Concrete behavior trace:
#    The spatial grammar is unchanged: each full-height legend panel still contains a 2-color
#    unit sprite, an 8-connected single-color frame mask, a 5x5 anchor square, and a 3x3
#    rotation hand. The fusion changes only the color semantics.
#    Example with a 2x2 sprite:
#      legend sprite = [[2, 8],
#                       [8, 2]]
#      original task tiled the swapped sprite [[8, 2],
#                                             [2, 8]]
#      this combo tiles [[2, 8],
#                        [8, 2]] directly.
#      If the canvas band color under some stamped cells is 8, then those 8-cells are still
#      part of the composite but remain visually unchanged after overlay. So compared with the
#      original task, the combo differs twice on the same placement: the repeated motif keeps
#      the legend colors, and one of those colors can blend into the background.
# b) Complexity calibration:
#    The frame extraction, rotation decoding, anchor decoding, and non-overlap placement all
#    remain intact, so the task still requires multi-step legend interpretation. The variant is
#    not oversimplified because every sample must simultaneously expose two effects:
#    the output disagrees with the original color-swapped interpretation, and at least some
#    placed sprite pixels are camouflaged by equal background colors.
# c) Implicit-assumption review:
#    The original `_extract_unit_sprite()` assumed any 2-color sprite needed a global swap
#    before tiling; that assumption is removed. The rest of the solver only relies on the top
#    non-{0,6} component being the unit sprite and on its nonzero pixels being literal content,
#    which still holds. The overlay loop already treats same-color writes as no-op visuals but
#    valid occupancy, so D17 does not require a new paint rule. `composite_written` must still
#    block legend overlap even when some written pixels are visually hidden.
# d) Placement legality:
#    The generator stays output-first. It chooses sprite/frame/rotation/anchor first, checks
#    that each rotated composite fits fully in the canvas, and rejects 2-legend layouts whose
#    composite bounding boxes intersect. Hidden pixels are counted only after a legal placement
#    is found, so camouflage never relies on clipping or inter-legend overlap.
# e) Visual observability:
#    The unit sprite is fully filled and uses exactly two distinct colors, so interpreting the
#    legend literally versus swapping its colors changes every occupied composite cell. On top of
#    that, the generator forces at least one sprite color to match a canvas band color and
#    requires actual hidden pixels, making the camouflage effect visible across examples.
# f) Visual cleanliness:
#    No extra markers are introduced. The scene remains "legend(s) + separator column(s) +
#    striped canvas". Frame colors stay disjoint from the legend's unit colors so the solver can
#    still isolate the top component cleanly, and hidden pixels arise only inside the final
#    composite placement region rather than through additional clutter.
# g) Change inferability:
#    Only two canvas band colors are used, and each legend contributes at most one hidden color.
#    That keeps the color story small enough for humans to track: the legend sprite is the exact
#    repeated tile, and whichever repeated color matches the background band can disappear.
# h) Quantitative fusion checks:
#    `generate_input()` validates two metrics. First, every legend must contribute hidden pixels
#    (`hidden_count > 0`), proving D17 is active. Second, the final output under this combo must
#    differ from the counterfactual "swap the legend sprite colors before tiling" output, proving
#    D11 is active on the same sample. A helper reports these counts for bulk verification.
# i) Solver assumption audit:
#    `_all_marker_panel_squares()` assumes marker panels are still recognizable by a 5x5 border-6
#    square plus a nonempty 3x3 rotation patch; `_place_legend()` preserves that exactly.
#    `_extract_frame_bbox()` assumes all frame pixels lie strictly between the sprite and marker;
#    the generator keeps zero-only gaps there. `_solve_core()` assumes any full-height/full-width
#    6-runs inside the canvas merely shrink the effective anchor extent; the generated canvas uses
#    only band colors, so those runs do not appear.

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label


GRID_H_RANGE = (20, 30)
GRID_W = 30
UNIT_SIZE_RANGE = (3, 5)
RIGHT_LEGEND_MIN_UNIT = 3
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300
CANVAS_RETRIES = 120
MIN_HIDDEN_PIXELS = 6
_GEN_CALL_COUNT = 0


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


def _swap_two_colors(sprite: np.ndarray) -> np.ndarray:
    """Swap the two non-{0,6} colors in a sprite, if there are exactly two."""
    cols = sorted({int(x) for x in np.unique(sprite) if int(x) not in (0, 6)})
    if len(cols) != 2:
        return sprite.copy()
    a, b = cols
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
    return out


def _extract_unit_sprite(legend: np.ndarray, swap_colors: bool = False) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Extract the top-most 4-connected non-{0,6} component as the unit sprite."""
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
    if swap_colors:
        spr = _swap_two_colors(spr)
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
    Read the 3x3 clock-hand below the marker square.

    The furthest non-{0,6} pixel from the center determines direction:
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


def _compose_tiled_sprite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile the unit sprite into every active frame cell."""
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


def _solve_core(grid: np.ndarray, swap_unit_colors: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    base_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = base_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(base_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(base_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = base_canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue
        if marker_c0 + 5 > w:
            continue

        panel = g[:, marker_c0 : marker_c0 + 5]
        unit = _extract_unit_sprite(panel, swap_colors=swap_unit_colors)
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
        comp = _compose_tiled_sprite(unit_sprite, frame_mask)

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
                if val == 0 or val == 6:
                    continue
                assert not composite_written[rr, cc], "composites must not overlap"
                composite_written[rr, cc] = True
                out[rr, cc] = val
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild enlarged sprites from legend panels and paint them onto the canvas region.

    Input structure:
    - Full-height color-6 separator columns split the grid into a left legend, the central
      canvas, and sometimes a right legend.
    - Each legend panel stacks four cues from top to bottom: a two-color unit sprite, a
      single-color frame mask, a 5x5 anchor square with one inner marker pixel, and a 3x3
      rotation hand beneath that square.

    Rule:
    - Read the unit sprite exactly as shown in the legend, with no color swap, tile it into
      every active frame cell, rotate the resulting composite according to the clock-hand, and
      place it on the canvas using the anchor encoded by the marker square.
    - If a placed sprite pixel has the same color as the underlying canvas cell, it still
      belongs to the composite even though the visible value on that cell does not change.

    Key insight:
    - The legend now gives the final repeated motif directly; the only "hidden" part is that
      some repeated pixels can blend into the striped background, so absence of visible change
      does not mean absence of sprite structure.
    """
    return _solve_core(grid, swap_unit_colors=False)


def _allocate_colors(
    num_legends: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], list[int], list[int]] | None:
    """
    Pick unit colors, two canvas colors, and one frame color per legend.

    At least one background color is reused from each legend's sprite palette so the
    camouflage effect is guaranteed to be possible.
    """
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets: list[tuple[int, int]] = []
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        c1, c2 = pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2))

    hidden_colors = [random.choice(uc) for uc in unit_color_sets]
    if num_legends == 1:
        bg_hidden = hidden_colors[0]
        bg_other_choices = [c for c in USABLE_COLORS if c not in set(unit_color_sets[0])]
        if not bg_other_choices:
            return None
        bg_colors = (bg_hidden, random.choice(bg_other_choices))
    else:
        bg_colors = (hidden_colors[0], hidden_colors[1])

    if random.random() < 0.5:
        bg_colors = (bg_colors[1], bg_colors[0])

    frame_colors: list[int] = []
    all_set = set(USABLE_COLORS)
    for uc in unit_color_sets:
        candidates = list(all_set - set(uc))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors, hidden_colors


def _make_unit_sprite(N: int, c1: int, c2: int, hidden_color: int) -> np.ndarray:
    """
    Generate a full NxN two-color sprite.

    Hidden-color pixels are spread out so background camouflage is visually noticeable.
    """
    min_hidden = max(2, N)
    for _ in range(120):
        sprite = np.full((N, N), c1, dtype=int)
        num_c2 = random.randint(1, N * N - 1)
        for p in random.sample(range(N * N), num_c2):
            sprite[p // N, p % N] = c2
        if np.count_nonzero(sprite == c1) < 2 or np.count_nonzero(sprite == c2) < 2:
            continue
        pts = np.argwhere(sprite == hidden_color)
        if pts.shape[0] < min_hidden:
            continue
        if len(np.unique(pts[:, 0])) < min(2, N):
            continue
        if len(np.unique(pts[:, 1])) < min(2, N):
            continue
        return sprite

    other = c2 if hidden_color == c1 else c1
    rr, cc = np.indices((N, N))
    sprite = np.where((rr + cc) % 2 == 0, hidden_color, other).astype(int)
    if np.all(sprite == hidden_color) or np.all(sprite == other):
        sprite[0, 0] = hidden_color
        sprite[-1, -1] = other
    return sprite


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    hidden_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
) -> bool:
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    uc1, uc2 = unit_colors

    sprite = _make_unit_sprite(N, uc1, uc2, hidden_color)
    swapped_sprite = _swap_two_colors(sprite)

    avail = marker_r0 - (N + 1)
    combos = []
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * N <= grid_h and rw * N <= canvas_w:
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
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)

    leg.update(
        sprite=sprite,
        legend_sprite=sprite.copy(),
        swapped_sprite=swapped_sprite,
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        hidden_color=hidden_color,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    """Return (height, width) of the composite after rotation."""
    rf = leg["rotated_frame"]
    N = leg["N"]
    return rf.shape[0] * N, rf.shape[1] * N


def _build_composite(leg: dict, swapped: bool = False) -> np.ndarray:
    key = "composite_swapped" if swapped else "composite"
    comp = leg.get(key)
    if comp is None:
        sprite = leg["swapped_sprite"] if swapped else leg["sprite"]
        comp = _compose_tiled_sprite(sprite, leg["rotated_frame"])
        leg[key] = comp
    return comp


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions while keeping all composites in bounds and disjoint."""
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


def _build_banded_canvas(
    grid_h: int, canvas_w: int, bg1: int, bg2: int, band_dir: str, band_width: int, offset: int
) -> np.ndarray:
    """Build a canvas with alternating horizontal or vertical bands."""
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    dim = grid_h if band_dir == "h" else canvas_w
    band_indices = ((np.arange(dim) + offset) // band_width) % 2
    colors = np.array([bg1, bg2], dtype=int)
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _hidden_pixel_counts(canvas: np.ndarray, legends: list[dict]) -> list[int]:
    """Count, per legend, pixels whose sprite color matches the underlying canvas color."""
    counts: list[int] = []
    for leg in legends:
        comp = _build_composite(leg, swapped=False)
        ch, cw = comp.shape
        t, l = leg["comp_top"], leg["comp_left"]
        region = canvas[t : t + ch, l : l + cw]
        hidden = (comp != 0) & (comp != 6) & (comp == region)
        counts.append(int(hidden.sum()))
    return counts


def _overlay_composites(canvas: np.ndarray, legends: list[dict], swapped: bool = False) -> np.ndarray:
    """Stamp each legend's tiled composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite(leg, swapped=swapped)
        ch, cw = comp.shape
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (comp != 6) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _build_camouflage_canvas(
    grid_h: int, canvas_w: int, bg1: int, bg2: int, legends: list[dict]
) -> tuple[np.ndarray, list[int]] | None:
    best: tuple[np.ndarray, list[int]] | None = None
    best_score = (-1, -1)
    for _ in range(CANVAS_RETRIES):
        band_dir = random.choice(["h", "v"])
        dim = grid_h if band_dir == "h" else canvas_w
        band_width = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
        offset = random.randint(0, 2 * band_width - 1)
        canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2, band_dir, band_width, offset)
        counts = _hidden_pixel_counts(canvas, legends)
        score = (min(counts), sum(counts))
        if score > best_score:
            best = (canvas, counts)
            best_score = score
        if min(counts) > 0 and sum(counts) >= MIN_HIDDEN_PIXELS:
            return canvas, counts
    if best is None:
        return None
    if best_score[0] <= 0 or best_score[1] < MIN_HIDDEN_PIXELS:
        return None
    return best


def _mode_difference_count(canvas: np.ndarray, legends: list[dict]) -> int:
    """Count output pixels that differ between no-swap and swap-before-tiling interpretations."""
    direct = _overlay_composites(canvas, legends, swapped=False)
    swapped = _overlay_composites(canvas, legends, swapped=True)
    return int(np.count_nonzero(direct != swapped))


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    N = leg["N"]
    grid[:, sep_col] = 6

    # Keep the right-side marker panel strictly inside the legend so the separator column
    # remains a full-height run of 6s that `_split_canvas()` can recover.
    mc0 = col_start + max(0, N - 4)

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


def _gen_attempt(num_legends: int | None = None) -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    if num_legends is None:
        num_legends = random.choice([1, 2])

    legends = []
    for li in range(num_legends):
        if num_legends == 2 and li == 1:
            n_lo = max(UNIT_SIZE_RANGE[0], RIGHT_LEGEND_MIN_UNIT)
            N = random.randint(n_lo, UNIT_SIZE_RANGE[1])
        else:
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
    unit_color_sets, (bg1, bg2), frame_colors, hidden_colors = colors
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT

    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            hidden_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
        ):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    camo = _build_camouflage_canvas(grid_h, canvas_w, bg1, bg2, legends)
    if camo is None:
        return None
    canvas, hidden_counts = camo
    if min(hidden_counts) <= 0 or sum(hidden_counts) < MIN_HIDDEN_PIXELS:
        return None

    if _mode_difference_count(canvas, legends) <= 0:
        return None

    output = _overlay_composites(canvas, legends, swapped=False)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(
            grid,
            legends[1],
            col_start=r_sep + 1,
            sep_col=r_sep,
            marker_r0=marker_r0,
            is_right=True,
        )

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def _fusion_metrics_from_input(grid: np.ndarray) -> dict[str, object]:
    """
    Return quantitative checks for this combo on a concrete input.

    Keys:
      hidden_counts: per-legend counts of same-color sprite/background placements
      hidden_total: total hidden pixel count
      mode_diff_pixels: number of output pixels that would change under the old swap rule
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    hidden_counts: list[int] = []
    legends: list[dict] = []
    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    for marker_r0, marker_c0 in _all_marker_panel_squares(g):
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue
        panel = g[:, marker_c0 : marker_c0 + 5]
        unit = _extract_unit_sprite(panel, swap_colors=False)
        if unit is None:
            continue
        unit_sprite, unit_bb = unit
        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _compose_tiled_sprite(unit_sprite, frame_mask)
        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])
        ch, cw = comp.shape
        region = canvas[top : top + ch, left : left + cw]
        hidden = (comp != 0) & (comp != 6) & (comp == region)
        hidden_counts.append(int(hidden.sum()))
        legends.append(
            {
                "sprite": unit_sprite,
                "swapped_sprite": _swap_two_colors(unit_sprite),
                "rotated_frame": frame_mask,
                "comp_top": top,
                "comp_left": left,
                "composite": comp,
            }
        )

    direct = _overlay_composites(canvas, legends, swapped=False)
    swapped = _overlay_composites(canvas, legends, swapped=True)
    return {
        "hidden_counts": hidden_counts,
        "hidden_total": int(sum(hidden_counts)),
        "mode_diff_pixels": int(np.count_nonzero(direct != swapped)),
    }


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred = 1 if _GEN_CALL_COUNT % 2 == 0 else 2
    _GEN_CALL_COUNT += 1

    primary_retries = GEN_RETRIES // 2
    for target, limit in ((preferred, primary_retries), (3 - preferred, GEN_RETRIES - primary_retries)):
        for _ in range(limit):
            result = _gen_attempt(num_legends=target)
            if result is not None:
                return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
