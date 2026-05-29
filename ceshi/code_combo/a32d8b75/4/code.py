# Combo variant of a32d8b75 (index 4)
# combo_spec: D2/3+D10/1
# Name: 多精灵交替铺贴+精灵叠加混合

"""
Design analysis for the combo variant.

concretization_and_behavior_trace:
  original_pipeline:
    - each legend shows one NxN two-color sprite, one frame mask, one anchor square, one rotation mark
    - after color swap, every active frame cell receives the same sprite
    - when two legends exist, their composites are forced not to overlap
  combo_pipeline:
    - each legend now shows two stacked NxN sprites A/B with the same palette and same size
    - after the frame is rotated, active frame cells use A/B in checkerboard parity on the rotated frame lattice
    - two legend composites are intentionally placed so they overlap; the earlier legend in reading order is the front layer
    - overlap pixels do not keep either sprite color directly: they are rewritten to the front legend's frame color
  small_example:
    - frame active cells = [(0,0),(0,1),(1,0),(1,1)]
    - original enlarged object = S S / S S
    - combo enlarged object from one legend = A B / B A
    - if a second legend overlaps the top-left 2 cells and the left legend is front, those shared pixels become the left legend's frame color
  delta_from_original:
    - extraction changes from one top sprite to two top sprites
    - composite construction changes from uniform tiling to parity-based alternating tiling
    - multi-legend interaction changes from "reject overlap" to "allow overlap and recolor the shared area"

complexity_calibration:
  - both sprites inside one legend share size N and the same two-color palette, so the new freedom is visible geometry rather than a hidden color remap
  - the frame is forced to use both checkerboard parities, otherwise the second sprite could disappear and the task would collapse back toward the base task
  - the generator always produces two legends with partial overlap, so the blend rule is present in every sample instead of appearing as a rare corner case
  - frame colors are kept disjoint from all unit colors and background colors, making the overlap recolor visually obvious

implicit_assumption_audit:
  - original `_extract_unit_sprite` assumed the first non-{0,6} component above the marker was the only unit; this combo breaks that, so the solver now extracts the top two 4-connected square components
  - original `_extract_frame_bbox` assumed the frame started below one sprite; now it must start below the lower of the two sprite boxes
  - original composite writer assumed every active frame cell reused the same sprite; now it must choose by rotated-frame parity
  - original painter asserted that composite pixels never overlap; that assumption is removed and replaced with explicit front-layer blend handling
  - new solver assumption: the visible blend color can be recovered from the legend's frame color. The generator guarantees one solid frame color per legend and keeps it disjoint from sprite colors

placement_legality:
  - both enlarged composites must fit fully inside the canvas for all chosen anchors
  - the two placement masks must overlap by a minimum amount while leaving non-overlapped area from both legends visible
  - the overlap is computed on nonzero composite pixels, not merely bounding boxes, so the shared region is genuine

visual_observability:
  - the stacked pair of sprites is directly visible in each legend, and the output repeats their alternation many times across the enlarged object
  - overlap is readable because the shared area switches to the front legend's frame color, which is shown elsewhere in that same legend
  - front/back order depends only on visible panel order, never on numeric color ids or hidden metadata

visual_cleanliness_and_inference:
  - sprites remain separated by a zero spacer row, so the two source tiles are easy to parse
  - only one blend color can appear per overlap region, preventing noisy multicolor collisions
  - because frame colors are not reused by unit sprites or canvas bands, a human can attribute the shared color patch to the legend/frame interaction rather than to accidental palette overlap

new_solver_assumptions_and_generator_guarantees:
  - solver assumes the first two 4-connected square components above the frame are the unit sprites; generator builds exactly that layout
  - solver assumes each frame mask is 8-connected and monochrome; generator samples and validates accordingly
  - solver assumes the front legend's frame color is the intended overlap color; generator ensures overlap exists and makes the fused output differ from simple overwriting
"""

import random

import numpy as np
from arc_utils import bbox_coords, random_sprite
from scipy.ndimage import label as _scipy_label


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)


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


def _swap_two_colors(sprite: np.ndarray) -> np.ndarray | None:
    cols = sorted({int(x) for x in np.unique(sprite) if int(x) not in (0, 6)})
    if len(cols) != 2:
        return None
    a, b = cols
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
    return out


def _extract_unit_sprites(
    legend: np.ndarray, border6_r0: int
) -> tuple[tuple[np.ndarray, tuple[int, int, int, int]], tuple[np.ndarray, tuple[int, int, int, int]]] | None:
    """Extract the two top-most square sprites stacked above the frame and marker."""
    scan = legend[:border6_r0, :]
    mask = (scan != 0) & (scan != 6)
    labels, n4 = _scipy_label(mask, structure=STRUCT4)
    if n4 < 2:
        return None

    comps: list[tuple[int, int, int, int]] = []
    for idx in range(1, n4 + 1):
        pts = np.argwhere(labels == idx)
        if pts.size == 0:
            continue
        comps.append(bbox_coords(pts))
    comps.sort(key=lambda bb: (bb[0], bb[1], bb[2], bb[3]))
    if len(comps) < 2:
        return None

    result = []
    for bb in comps[:2]:
        r0, c0, r1, c1 = bb
        spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
        if spr.shape[0] != spr.shape[1]:
            return None
        if not np.all((spr != 0) & (spr != 6)):
            return None
        swapped = _swap_two_colors(spr)
        if swapped is None:
            return None
        result.append((swapped, bb))

    if result[0][0].shape != result[1][0].shape:
        return None
    return result[0], result[1]


def _extract_frame_bbox(
    legend: np.ndarray,
    sprite_bbs: list[tuple[int, int, int, int]],
    border6_r0: int,
) -> tuple[int, int, int, int] | None:
    """Frame pixels live below the lower sprite and above the marker square."""
    r_start = max(bb[2] for bb in sprite_bbs) + 1
    r_end = border6_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    pts[:, 0] += r_start
    return bbox_coords(pts)


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


def _build_alternating_composite(frame_mask: np.ndarray, sprite_a: np.ndarray, sprite_b: np.ndarray) -> np.ndarray:
    """Tile two sprites over active frame cells in checkerboard alternation."""
    assert sprite_a.shape == sprite_b.shape
    uh, uw = sprite_a.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    mask_a = (sprite_a != 0) & (sprite_a != 6)
    mask_b = (sprite_b != 0) & (sprite_b != 6)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            if (i + j) % 2 == 0:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][mask_a] = sprite_a[mask_a]
            else:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][mask_b] = sprite_b[mask_b]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode alternating two-sprite legend panels, build enlarged composites, and merge them on the canvas.

    Input structure:
    - Full-height color-6 separator columns isolate the left legend and, optionally, a right legend from the canvas.
    - Each legend panel now contains two stacked NxN unit sprites, one frame mask, one 5x5 anchor square, and one 3x3 clock-hand rotation patch.

    Rule:
    - For every legend, color-swap both unit sprites back, rotate the frame layout from the clock-hand, and tile the two sprites onto active frame cells in checkerboard alternation on the rotated frame lattice.
    - Place each enlarged composite by the anchor marker. If multiple composites cover the same output pixels, the earlier legend in reading order is the front layer and every shared pixel is recolored to that front legend's frame color.

    Key insight:
    - The bottom markers still encode rotation and placement, but the visible output now depends on two extra structural cues: the pair of stacked source sprites determines the checkerboard tiling, and the front legend's frame color reveals where overlap happened.
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

    legend_jobs: list[dict[str, object]] = []
    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue
        if marker_c0 + 5 > w:
            continue

        panel = g[:, marker_c0 : marker_c0 + 5]
        sprites = _extract_unit_sprites(panel, marker_r0)
        if sprites is None:
            continue
        (sprite_a, bb_a), (sprite_b, bb_b) = sprites

        frame_bb = _extract_frame_bbox(panel, [bb_a, bb_b], marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=STRUCT8)
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"
        frame_colors = sorted({int(x) for x in np.unique(frame) if int(x) not in (0, 6)})
        assert len(frame_colors) == 1, "frame must use a single visible color"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_alternating_composite(frame_mask, sprite_a, sprite_b)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        legend_jobs.append(
            {
                "layer_key": (int(marker_r0), int(marker_c0)),
                "top": int(top),
                "left": int(left),
                "comp": comp,
                "blend_color": int(frame_colors[0]),
            }
        )

    out = main_canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)
    for job in sorted(legend_jobs, key=lambda item: item["layer_key"], reverse=True):
        top = int(job["top"])
        left = int(job["left"])
        comp = np.asarray(job["comp"], dtype=int)
        blend_color = int(job["blend_color"])
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
                if composite_written[rr, cc]:
                    out[rr, cc] = blend_color
                else:
                    out[rr, cc] = val
                composite_written[rr, cc] = True
    return out


GRID_H_RANGE = (20, 30)
GRID_W = 30
NUM_LEGENDS = 2
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
SPRITE_STACK_GAP = 1
MIN_SPRITE_DIFF = 2
MIN_FRAME_ACTIVE = 6
MIN_PARITY_CELLS = 2
MIN_OVERLAP_PIXELS = 8
MIN_VISIBLE_PIXELS = 8


def _allocate_colors():
    """Pick disjoint unit, background, and frame colors for the two legends."""
    all_colors = list(USABLE_COLORS)
    for _ in range(200):
        pool = list(all_colors)
        random.shuffle(pool)
        unit_color_sets = []
        used_unit: set[int] = set()
        for _legend_idx in range(NUM_LEGENDS):
            c1, c2 = pool.pop(), pool.pop()
            unit_color_sets.append((c1, c2))
            used_unit.update((c1, c2))

        remaining = [c for c in all_colors if c not in used_unit]
        if len(remaining) < 4:
            continue
        random.shuffle(remaining)
        bg_colors = (remaining[0], remaining[1])
        frame_candidates = [c for c in remaining[2:] if c not in bg_colors]
        if len(frame_candidates) < 2:
            continue
        non_marker_frames = [c for c in frame_candidates if c not in (4, 7)]
        if len(non_marker_frames) >= 2:
            frame_colors = tuple(random.sample(non_marker_frames, 2))
        else:
            frame_colors = tuple(random.sample(frame_candidates, 2))
        return unit_color_sets, bg_colors, frame_colors
    return None


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _random_filled_sprite(n: int, c1: int, c2: int) -> np.ndarray:
    sprite = np.full((n, n), c1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = c2
    return sprite


def _design_sprite_pair(n: int, unit_colors: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    c1, c2 = unit_colors
    min_diff = max(MIN_SPRITE_DIFF, n)
    for _ in range(200):
        sprite_a = _random_filled_sprite(n, c1, c2)
        sprite_b = _random_filled_sprite(n, c1, c2)
        if np.array_equal(sprite_a, sprite_b):
            continue
        if int(np.sum(sprite_a != sprite_b)) < min_diff:
            continue
        return sprite_a, sprite_b
    return None


def _frame_uses_both_parities(mask: np.ndarray) -> bool:
    rr, cc = np.indices(mask.shape)
    even = int(np.sum(mask & (((rr + cc) % 2) == 0)))
    odd = int(np.sum(mask & (((rr + cc) % 2) == 1)))
    return even >= MIN_PARITY_CELLS and odd >= MIN_PARITY_CELLS


def _feasible_unit_sizes(grid_h: int) -> list[int]:
    min_frame_h = FRAME_H_RANGE[0]
    out = []
    for n in range(UNIT_SIZE_RANGE[0], UNIT_SIZE_RANGE[1] + 1):
        after_sprites = 2 * n + SPRITE_STACK_GAP + 1
        marker_r0 = grid_h - MARKER_PANEL_HEIGHT
        avail = marker_r0 - after_sprites
        if avail >= min_frame_h + 1:
            out.append(n)
    return out


def _design_legend(
    leg: dict[str, object],
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
) -> bool:
    """Fill in both sprites, the frame, rotation, and enlarged composite for one legend."""
    n = int(leg["N"])
    sprites = _design_sprite_pair(n, unit_colors)
    if sprites is None:
        return False
    sprite_a, sprite_b = sprites
    uc1, uc2 = unit_colors
    legend_sprite_a = _color_swap(sprite_a, uc1, uc2)
    legend_sprite_b = _color_swap(sprite_b, uc1, uc2)

    after_sprites = 2 * n + SPRITE_STACK_GAP + 1
    avail = marker_r0 - after_sprites
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
    for _ in range(200):
        frame_mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
        if int(frame_mask.sum()) < MIN_FRAME_ACTIVE:
            continue
        if not _frame_uses_both_parities(rot_frame):
            continue
        composite = _build_alternating_composite(rot_frame, sprite_a, sprite_b)
        leg.update(
            sprite_a=sprite_a,
            sprite_b=sprite_b,
            legend_sprite_a=legend_sprite_a,
            legend_sprite_b=legend_sprite_b,
            frame_mask=frame_mask,
            frame_H=frame_h,
            frame_W=frame_w,
            frame_color=frame_color,
            rot_k=rot_k,
            rotated_frame=rot_frame,
            composite=composite,
            comp_mask=(composite != 0),
        )
        return True
    return False


def _comp_size(leg: dict[str, object]) -> tuple[int, int]:
    comp = np.asarray(leg["composite"], dtype=int)
    return int(comp.shape[0]), int(comp.shape[1])


def _placed_mask(mask: np.ndarray, top: int, left: int, canvas_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=bool)
    h, w = mask.shape
    out[top : top + h, left : left + w] = mask
    return out


def _overlap_stats(
    mask_a: np.ndarray,
    top_a: int,
    left_a: int,
    mask_b: np.ndarray,
    top_b: int,
    left_b: int,
    canvas_shape: tuple[int, int],
) -> tuple[int, int, int]:
    placed_a = _placed_mask(mask_a, top_a, left_a, canvas_shape)
    placed_b = _placed_mask(mask_b, top_b, left_b, canvas_shape)
    overlap = int(np.count_nonzero(placed_a & placed_b))
    only_a = int(np.count_nonzero(placed_a & ~placed_b))
    only_b = int(np.count_nonzero(placed_b & ~placed_a))
    return overlap, only_a, only_b


def _assign_anchors(legends: list[dict[str, object]], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions so the two composites overlap visibly but not totally."""
    assert len(legends) == 2
    a, b = legends[0], legends[1]
    valid_pairs = []
    for apr in range(3):
        for apc in range(3):
            a_ch, a_cw = _comp_size(a)
            at = _anchor_to_top_left(apr, grid_h, a_ch)
            al = _anchor_to_top_left(apc, canvas_w, a_cw)
            for bpr in range(3):
                for bpc in range(3):
                    b_ch, b_cw = _comp_size(b)
                    bt = _anchor_to_top_left(bpr, grid_h, b_ch)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_cw)
                    overlap, only_a, only_b = _overlap_stats(
                        np.asarray(a["comp_mask"], dtype=bool),
                        at,
                        al,
                        np.asarray(b["comp_mask"], dtype=bool),
                        bt,
                        bl,
                        (grid_h, canvas_w),
                    )
                    if overlap < MIN_OVERLAP_PIXELS:
                        continue
                    if only_a < MIN_VISIBLE_PIXELS or only_b < MIN_VISIBLE_PIXELS:
                        continue
                    valid_pairs.append((apr, apc, at, al, bpr, bpc, bt, bl, overlap))
    if not valid_pairs:
        return False

    apr, apc, at, al, bpr, bpc, bt, bl, overlap = random.choice(valid_pairs)
    a.update(pr=apr, pc=apc, comp_top=at, comp_left=al, overlap_pixels=overlap)
    b.update(pr=bpr, pc=bpc, comp_top=bt, comp_left=bl, overlap_pixels=overlap)
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


def _marker_c0_for_panel(n: int, col_start: int, sep_col: int, is_right: bool) -> int:
    """Use the same marker-square placement rule in solver-side bookkeeping and drawing."""
    if not is_right:
        return col_start + max(0, n - 4)
    return sep_col + max(0, n - 3)


def _overlay_composites(canvas: np.ndarray, legends: list[dict[str, object]]) -> np.ndarray:
    """Stamp composites onto the canvas, blending overlaps to the front legend's frame color."""
    output = canvas.copy()
    composite_written = np.zeros(output.shape, dtype=bool)
    for leg in sorted(legends, key=lambda item: item["layer_key"], reverse=True):
        comp = np.asarray(leg["composite"], dtype=int)
        blend_color = int(leg["frame_color"])
        top = int(leg["comp_top"])
        left = int(leg["comp_left"])
        ch, cw = comp.shape
        for r in range(ch):
            rr = top + r
            for c in range(cw):
                cc = left + c
                if output[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val == 0 or val == 6:
                    continue
                if composite_written[rr, cc]:
                    output[rr, cc] = blend_color
                else:
                    output[rr, cc] = val
                composite_written[rr, cc] = True
    return output


def _place_legend(
    grid: np.ndarray,
    leg: dict[str, object],
    col_start: int,
    sep_col: int,
    marker_r0: int,
    is_right: bool,
) -> None:
    """Draw one legend panel: two sprites, frame, marker square, and rotation patch."""
    n = int(leg["N"])
    grid[:, sep_col] = 6

    mc0 = _marker_c0_for_panel(n, col_start, sep_col, is_right)

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + int(leg["pr"]), mc0 + 1 + int(leg["pc"])] = 4

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[int(leg["rot_k"])]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sp_col = col_start + 1 - mc0
    spr_c0 = mc0 + sp_col
    grid[1 : 1 + n, spr_c0 : spr_c0 + n] = np.asarray(leg["legend_sprite_a"], dtype=int)
    row_b0 = 1 + n + SPRITE_STACK_GAP
    grid[row_b0 : row_b0 + n, spr_c0 : spr_c0 + n] = np.asarray(leg["legend_sprite_b"], dtype=int)

    frame_h = int(leg["frame_H"])
    frame_w = int(leg["frame_W"])
    after_sprites = 2 * n + SPRITE_STACK_GAP + 1
    avail = marker_r0 - after_sprites
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = after_sprites + gap
    f_col = random.randint(0, max(0, 4 - frame_w))
    frame_region = grid[fr_start : fr_start + frame_h, mc0 + f_col : mc0 + f_col + frame_w]
    frame_region[np.asarray(leg["frame_mask"], dtype=bool)] = int(leg["frame_color"])


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    feasible_sizes = _feasible_unit_sizes(grid_h)
    if not feasible_sizes:
        return None

    legends = []
    for _ in range(NUM_LEGENDS):
        n = random.choice(feasible_sizes)
        legends.append({"N": n, "legend_w": n + 2})

    left_lw = int(legends[0]["legend_w"])
    left_total = left_lw + 1
    right_total = 1 + int(legends[1]["legend_w"])
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors()
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    for li in range(NUM_LEGENDS):
        if not _design_legend(legends[li], unit_color_sets[li], int(frame_colors[li]), marker_r0, grid_h, canvas_w):
            return None

    right_sep = left_total + canvas_w
    legends[0]["layer_key"] = (marker_r0, _marker_c0_for_panel(int(legends[0]["N"]), 0, left_lw, False))
    legends[1]["layer_key"] = (
        marker_r0,
        _marker_c0_for_panel(int(legends[1]["N"]), right_sep + 1, right_sep, True),
    )

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    _place_legend(grid, legends[1], col_start=right_sep + 1, sep_col=right_sep, marker_r0=marker_r0, is_right=True)

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
