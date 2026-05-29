# Combo variant of a32d8b75 (index 27)
# combo_spec: D2/3+D4/1+D15/0
# Name: 交替铺贴+位置旋转+先放后转

"""
Design analysis for combo 27.

a) Concrete behavior and trace:
   - Source task: read one sprite, rotate the frame first, tile one fixed sprite over the
     rotated frame, then anchor the finished composite on the canvas.
   - D2/3 changes the source object: each legend now contains two stacked NxN sprites A/B,
     both color-swapped on read, and active frame cells alternate between them by checkerboard
     parity.
   - D4/1 changes local tiling: each active frame cell chooses a 90-degree tile rotation from
     its compass direction relative to the frame center.
   - D15/0 changes step order: the enlarged object is not rotated before placement. Instead,
     the unrotated enlarged frame is assigned to a square work region on the canvas, filled
     in-place, and only then is the whole square rotated.
   - Small fused example:
       frame active cells = [(0,0),(0,1),(1,0),(1,1)], sprites = A/B, global clock-hand = 90.
       Old rotate-first fused logic would rotate the 2x2 frame first and then decide both
       parity and local direction on that rotated lattice.
       New fused logic decides parity and local direction on the original 2x2 lattice, paints
       those four tiles into the upper-left corner of a square work region, and rotates that
       whole region afterward.
       The local sprite rotations commute with the final quarter-turn, but the work-square
       background and padding do not, so the D15 step-order change is what makes the final
       image differ from a simple rotate-first fusion.

b) Complexity calibration:
   - Alternation uses only two sprites and a single parity rule `(r + c) % 2`.
   - Local direction uses only four compass buckets from the frame center.
   - The generator forces both sprites to have four distinct quarter-turns, forces the frame
     to occupy both checkerboard parities and at least three direction sectors, and rejects
     any sample where the fused composite collapses to an alternating-only or radial-only
     baseline.

c) Implicit assumption audit:
   - Original `_extract_unit_sprite()` assumed one sprite component; fusion replaces this with
     `_extract_unit_sprites()` reading the top two 4-connected components above the frame.
   - Original `_extract_frame_bbox()` assumed the frame starts below one sprite; now it must
     start below the lower of the two sprite boxes.
   - Original `frame_mask = rot90(frame_mask0, ...)` before tiling is no longer valid because
     D15 anchors the pre-rotation work square. The solver must build from the unrotated frame.
   - Original "all active cells stamp the same tile" assumption is invalid twice over: parity
     now selects A/B and direction now selects each tile's quarter-turn.
   - Original placement used the final rotated bbox. Fusion instead anchors a square work
     region with side `max(comp_h, comp_w)` and rotates that region in place.

d) Placement legality:
   - Generator placement is still output-first, but disjointness is checked on work squares,
     not only on the final visible pixels.
   - Each work square must fit fully inside the canvas.
   - Because the solver rotates entire work squares, the generator uses the same stronger
     non-overlap rule.

e) Visual observability:
   - Two stacked legend sprites are directly visible, so alternating source tiles are inferable.
   - Asymmetric sprites and multi-sector frames make the per-cell direction rotations readable.
   - Banded canvas backgrounds make the place-then-rotate step visible because background inside
     the work square rotates together with the tiled object.
   - The generator explicitly rejects cases where the final output matches a rotate-first fused
     baseline on the same input, so D15 is not merely nominal.

f) Visual cleanliness:
   - Legends remain vertically separated by zero rows: sprite A, spacer, sprite B, spacer, frame,
     then marker square and rotation patch.
   - Frame color is still only a legend cue and never appears in the output.
   - Work squares never overlap, so rotated background patches do not visually interfere.

g) Color inferability:
   - Each legend still uses exactly two sprite colors, shared by both A and B, with the same
     visible color-swap convention as the source task.
   - Background colors stay disjoint from unit colors.
   - No branch depends on numeric color ordering; only visible geometry, parity, and position do.

New solver assumptions and generator guarantees:
   - The two top-most non-{0,6} 4-components above the marker are the two square unit sprites.
   - Both sprites share the same size and the same two-color palette.
   - The frame lies strictly below them, is 8-connected, hits both parity classes, and covers
     multiple direction sectors.
   - The anchored work square is fully in-bounds and disjoint from any other work square.
   - At least one legend in each generated sample produces a different result from the natural
     rotate-first fused baseline, so the multi-step fusion is actually exercised.
"""

import random

import numpy as np
from arc_utils import bbox_coords, random_sprite
from scipy.ndimage import label as _scipy_label


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)


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
    """Extract the two top-most unit sprites stacked above the frame and marker."""
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
    drawn with non-{0,6} pixels. Use the furthest pixel from the center (1,1) to determine
    direction: up -> 0, right -> 1, down -> 2, left -> 3.
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


def _build_alternating_composite(frame_mask: np.ndarray, sprite_a: np.ndarray, sprite_b: np.ndarray) -> np.ndarray:
    """Tile the two sprites over active frame cells in checkerboard alternation."""
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


def _build_radial_single_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile one sprite over the frame, rotating each copy by its cell direction."""
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
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][rot_masks[rk]] = rot_sprites[rk][rot_masks[rk]]
    return comp


def _build_fused_composite(frame_mask: np.ndarray, sprite_a: np.ndarray, sprite_b: np.ndarray) -> np.ndarray:
    """
    Build the pre-rotation enlarged composite on the unrotated frame lattice:
    checkerboard sprite choice plus direction-based local sprite rotation.
    """
    assert sprite_a.shape == sprite_b.shape
    uh, uw = sprite_a.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    rot_a = [np.rot90(sprite_a, (-k) % 4) for k in range(4)]
    rot_b = [np.rot90(sprite_b, (-k) % 4) for k in range(4)]
    mask_a = [(spr != 0) & (spr != 6) for spr in rot_a]
    mask_b = [(spr != 0) & (spr != 6) for spr in rot_b]
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            rk = _cell_rotation_from_position(i, j, fh, fw)
            if (i + j) % 2 == 0:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][mask_a[rk]] = rot_a[rk][mask_a[rk]]
            else:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][mask_b[rk]] = rot_b[rk][mask_b[rk]]
    return comp


def _work_region_side(comp_shape: tuple[int, int]) -> int:
    return max(int(comp_shape[0]), int(comp_shape[1]))


def _apply_place_then_rotate(canvas: np.ndarray, comp: np.ndarray, top: int, left: int, dir_k: int) -> np.ndarray:
    """Fill the unrotated composite into a square work region, then rotate the whole region in place."""
    out = canvas.copy()
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


def _overlay_composite(canvas: np.ndarray, comp: np.ndarray, top: int, left: int) -> np.ndarray:
    """Simple overwrite overlay for the rotate-first baseline."""
    out = canvas.copy()
    region = out[top : top + comp.shape[0], left : left + comp.shape[1]]
    write_mask = (comp != 0) & (region != 6)
    region[write_mask] = comp[write_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode dual-sprite legend panels and rebuild each enlarged object directly inside a canvas work square.

    Input structure:
    - Full-height color-6 separators split left/right legend panels from the central canvas.
    - Every legend contains two stacked NxN unit sprites, then one frame template, then a 5x5 border-6 anchor square and a 3x3 clock-hand rotation patch.

    Rule:
    - Read both unit sprites and undo the legend-side color swap on each.
    - Use the unrotated frame template to build an enlarged composite in place: active frame cells alternate between the two sprites by checkerboard parity, and each chosen sprite is locally quarter-turned by that cell's direction inside the frame.
    - Anchor a square work region on the canvas from the marker square, paint the unrotated enlarged composite into the work region, and rotate the whole work region according to the clock-hand.

    Key insight:
    - The legend now encodes three interacting coordinate systems at once: stacked sprites choose the alternating tile family, frame-cell position chooses each tile's local orientation, and the bottom clock-hand rotates the already-painted canvas square rather than a pre-rotated composite.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
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
    work_region_written = np.zeros(out.shape, dtype=bool)

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
        frame_mask = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask, structure=STRUCT8)
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos

        comp = _build_fused_composite(frame_mask, sprite_a, sprite_b)
        side = _work_region_side(comp.shape)
        top = _anchor_to_top_left(pr, eff_h, side)
        left = _anchor_to_top_left(pc, eff_w, side)
        assert 0 <= top and 0 <= left, "work region placed above canvas origin"
        assert top + side <= out.shape[0] and left + side <= out.shape[1], "work region out of canvas bounds"
        assert not np.any(work_region_written[top : top + side, left : left + side]), "work regions must not overlap"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        out = _apply_place_then_rotate(out, comp, top, left, dir_k)
        work_region_written[top : top + side, left : left + side] = True
    return out


# --------------- generator (alternating radial place-then-rotate fusion) ---------------

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
SPRITE_STACK_GAP = 1
SPRITE_RETRIES = 200
FRAME_RETRIES = 200
MIN_SPRITE_DIFF = 2
MIN_FRAME_ACTIVE = 6
_GENERATE_CALL_COUNT = 0


def _allocate_colors(num_legends: int):
    """Pick unit colors (shared by both sprites in a legend), backgrounds, and frame colors."""
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
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _all_rotations_distinct(sprite: np.ndarray) -> bool:
    return len({np.rot90(sprite, k).tobytes() for k in range(4)}) == 4


def _make_asymmetric_sprite(n: int, c1: int, c2: int) -> np.ndarray | None:
    """Generate a fully filled NxN 2-color sprite with four distinct quarter-turns."""
    area = n * n
    for _ in range(SPRITE_RETRIES):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, area - 1)
        for p in random.sample(range(area), num_c2):
            sprite[p // n, p % n] = c2
        if _all_rotations_distinct(sprite):
            return sprite
    return None


def _design_sprite_pair(n: int, unit_colors: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    c1, c2 = unit_colors
    min_diff = max(MIN_SPRITE_DIFF, n)
    for _ in range(SPRITE_RETRIES):
        sprite_a = _make_asymmetric_sprite(n, c1, c2)
        sprite_b = _make_asymmetric_sprite(n, c1, c2)
        if sprite_a is None or sprite_b is None:
            return None
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
    return even > 0 and odd > 0


def _frame_has_visible_directionality(frame_mask: np.ndarray) -> bool:
    hist = _direction_histogram(frame_mask)
    used_dirs = sum(v > 0 for v in hist.values())
    has_vertical = hist[0] > 0 or hist[2] > 0
    has_horizontal = hist[1] > 0 or hist[3] > 0
    return used_dirs >= 3 and has_vertical and has_horizontal


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
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
) -> bool:
    """Fill in both sprites, the frame, and rotation for one legend dict."""
    n = leg["N"]
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
    for f_h in range(*FRAME_H_RANGE):
        if avail < f_h + 1:
            continue
        for f_w in range(*FRAME_W_RANGE):
            comp_shape = (f_h * n, f_w * n)
            side = _work_region_side(comp_shape)
            if side > grid_h or side > canvas_w:
                continue
            for rot_k in range(4):
                combos.append((f_h, f_w, rot_k))
    if not combos:
        return False
    random.shuffle(combos)

    for frame_H, frame_W, rot_k in combos:
        for _ in range(FRAME_RETRIES):
            frame_mask = random_sprite(
                frame_H,
                frame_W,
                density=FRAME_DENSITY,
                connectivity=8,
                color_palette=[1],
                symmetry="not_symmetric",
            ) != 0
            if int(frame_mask.sum()) < MIN_FRAME_ACTIVE:
                continue
            if not _frame_uses_both_parities(frame_mask):
                continue
            if not _frame_has_visible_directionality(frame_mask):
                continue

            fused_comp = _build_fused_composite(frame_mask, sprite_a, sprite_b)
            alternating_only = _build_alternating_composite(frame_mask, sprite_a, sprite_b)
            radial_a = _build_radial_single_composite(sprite_a, frame_mask)
            radial_b = _build_radial_single_composite(sprite_b, frame_mask)
            if np.array_equal(fused_comp, alternating_only):
                continue
            if np.array_equal(fused_comp, radial_a) or np.array_equal(fused_comp, radial_b):
                continue

            comp_shape = fused_comp.shape
            leg.update(
                sprite_a=sprite_a,
                sprite_b=sprite_b,
                legend_sprite_a=legend_sprite_a,
                legend_sprite_b=legend_sprite_b,
                frame_mask=frame_mask,
                frame_H=frame_H,
                frame_W=frame_W,
                frame_color=frame_color,
                rot_k=rot_k,
                fused_comp=fused_comp,
                comp_shape=comp_shape,
                work_side=_work_region_side(comp_shape),
            )
            return True
    return False


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign work-region anchors, avoiding overlap between work squares."""
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


def _overlay_fused_work_regions(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Apply the true fused rule: build on the unrotated work square, then rotate that square."""
    output = canvas.copy()
    for leg in legends:
        output = _apply_place_then_rotate(
            output,
            leg["fused_comp"],
            leg["work_top"],
            leg["work_left"],
            leg["rot_k"],
        )
    return output


def _overlay_rotate_first_fused(canvas: np.ndarray, legends: list[dict], grid_h: int, canvas_w: int) -> np.ndarray:
    """Reference baseline: rotate frame first, then build and place the fused composite normally."""
    output = canvas.copy()
    for leg in legends:
        rotated_frame = np.rot90(leg["frame_mask"], (-leg["rot_k"]) % 4)
        comp = _build_fused_composite(rotated_frame, leg["sprite_a"], leg["sprite_b"])
        top = _anchor_to_top_left(leg["pr"], grid_h, comp.shape[0])
        left = _anchor_to_top_left(leg["pc"], canvas_w, comp.shape[1])
        if top < 0 or left < 0 or top + comp.shape[0] > grid_h or left + comp.shape[1] > canvas_w:
            return output
        output = _overlay_composite(output, comp, top, left)
    return output


def _gen_attempt(num_legends: int | None = None) -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    feasible_sizes = _feasible_unit_sizes(grid_h)
    if not feasible_sizes:
        return None

    if num_legends is None:
        num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]
    legends = []
    for _ in range(num_legends):
        n = random.choice(feasible_sizes)
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
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_fused_work_regions(canvas, legends)
    rotate_first_baseline = _overlay_rotate_first_fused(canvas, legends, grid_h, canvas_w)
    if np.array_equal(output, rotate_first_baseline):
        return None

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


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
    """Draw one legend panel: two sprites, frame, marker square, and rotation patch."""
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
    spr_c0 = mc0 + sp_col
    grid[1 : 1 + n, spr_c0 : spr_c0 + n] = leg["legend_sprite_a"]
    row_b0 = 1 + n + SPRITE_STACK_GAP
    grid[row_b0 : row_b0 + n, spr_c0 : spr_c0 + n] = leg["legend_sprite_b"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    after_sprites = 2 * n + SPRITE_STACK_GAP + 1
    avail = marker_r0 - after_sprites
    gap = random.randint(1, max(1, avail - fH))
    fr_start = after_sprites + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start : fr_start + fH, mc0 + f_col : mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT
    preferred = 1 if (_GENERATE_CALL_COUNT % 2 == 0) else 2
    _GENERATE_CALL_COUNT += 1
    fallback = 2 if preferred == 1 else 1
    primary_budget = (GEN_RETRIES * 3) // 4
    for attempt in range(GEN_RETRIES):
        target_legends = preferred if attempt < primary_budget else fallback
        result = _gen_attempt(num_legends=target_legends)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
