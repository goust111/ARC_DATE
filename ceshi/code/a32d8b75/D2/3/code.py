# Variant of a32d8b75
# Dimension: D2 — 源对象：单元精灵的结构 (src)
# Change: NxN（N=2-5）全填充精灵，恰好2种非{0,6}颜色，读取时进行颜色交换（a<->b）... → 多个不同单元精灵交替铺贴

"""
Design analysis for the D2 variant.

source_task:
  status: none
  note: the variant description does not cite a reference task, so the baseline is the
    original a32d8b75 solver/generator plus the supplied spec.

variant_spec:
  core_change:
    from: each legend panel exposes one NxN fully-filled two-color unit sprite
    to: each legend panel exposes two different NxN fully-filled two-color unit sprites,
      stacked vertically in the legend; when the frame is expanded, active frame cells use
      those sprites in checkerboard alternation on the rotated frame lattice
  preserved_structure:
    - full-height separator columns remain color 6
    - each legend still owns one frame mask, one 5x5 anchor square, one 3x3 rotation patch
    - each sprite is still color-swapped when read from the legend
    - output canvas is still the separator-trimmed middle region
  concrete_pixel_layout:
    legend_panel:
      - sprite A at rows 1..N
      - one all-0 spacer row
      - sprite B at rows (N+2)..(2N+1)
      - one or more all-0 rows
      - frame mask
      - bottom marker square at rows h-8..h-4
      - bottom rotation patch at rows h-3..h-1
    output_composite:
      - rotate the frame mask first, exactly as in the original task
      - then tile sprite A on even-parity active cells and sprite B on odd-parity active cells
      - parity is measured on rotated frame coordinates using (r + c) % 2
  behavior_trace:
    original_small_example:
      legend: one 3x3 sprite S, frame active at [(0,0),(0,1),(1,0),(1,1)]
      output: S S / S S
    variant_small_example:
      legend: top sprite A, second sprite B, same 2x2 active frame
      output: A B / B A
    difference:
      - extraction changes from "top-most component only" to "top two components"
      - composite construction changes from "always stamp one sprite" to
        "choose sprite by cell parity"
  complexity_calibration:
    - both sprites share the same N and the same two-color palette inside one legend, so the
      change stays in structure rather than introducing an extra color-mapping rule
    - generator enforces A != B with a minimum Hamming distance, otherwise the task collapses
      back toward the original one-sprite behavior
    - generator also enforces that the rotated frame has active cells on both parities, so both
      sprites are actually used and the checkerboard rule is visible
  implicit_assumption_audit:
    original_main_dependencies:
      - `_extract_unit_sprite(panel)` assumes exactly one relevant component before the frame
      - `_extract_frame_bbox(..., unit_bb, ...)` assumes frame starts below one sprite
      - composite construction assumes every active frame cell receives the same sprite
    variant_updates:
      - replace single-sprite extraction with two top-most 4-connected components above marker
      - compute frame search start below the lower of the two sprite boxes
      - centralize composite creation in `_build_composite` so solver and generator share the
        same parity rule
  placement_legality:
    - composites keep the same outer size as the original task because both sprites are NxN
    - anchor selection still checks full in-bounds placement
    - for two legends, bounding boxes still must not overlap
  visual_observability:
    - A and B are generated as distinct asymmetric-looking filled patterns using the same palette
    - checkerboard alternation is repeated across several frame cells, so the new rule is visible
      from shape repetition rather than from hidden ids
  visual_cleanliness:
    - the legend remains vertically segmented by 0 rows, avoiding ambiguity between sprite B and
      the frame
    - only two unit colors per legend are used, and the frame color stays separate
  color_inference:
    - no extra color remapping beyond the original per-sprite two-color swap is introduced
    - the solver never relies on ordering by color id; only visible geometry and panel position
      matter
  new_solver_assumptions_and_generator_guarantees:
    - solver assumes the two top-most non-{0,6} 4-components above the marker are the two sprites
    - solver assumes both extracted sprites are square and share the same size
    - solver assumes the frame lives strictly below those sprites and is 8-connected
    - generator guarantees all of the above by construction and re-validates via `main(input)`
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


def _build_composite(frame_mask: np.ndarray, sprite_a: np.ndarray, sprite_b: np.ndarray) -> np.ndarray:
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
            sprite = sprite_a if (i + j) % 2 == 0 else sprite_b
            sprite_mask = mask_a if (i + j) % 2 == 0 else mask_b
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][sprite_mask] = sprite[sprite_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged alternating composites from legend-panel instructions.

    Input structure:
    - Color-6 separators still split the full grid into legend panel(s) and the central canvas.
    - Each legend now shows two stacked NxN unit sprites above one frame template, followed by the same 5x5 anchor square and 3x3 rotation clock-hand.

    Rule:
    - Read both unit sprites from the legend, color-swap each one, rotate the frame layout using the clock-hand, and fill the active frame cells with the two sprites in checkerboard alternation.
    - Place the resulting enlarged composite onto the canvas using the anchor encoded by the marker pixel inside the bordered square.

    Key insight:
    - The bottom markers still decide placement and rotation, but the enlarged object is no longer built from one repeated tile; the solver must notice that neighboring frame cells alternate between the first and second legend sprites.
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

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_composite(frame_mask, sprite_a, sprite_b)

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


# --------------- generator ---------------

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
SPRITE_STACK_GAP = 1
MIN_SPRITE_DIFF = 2
MIN_FRAME_ACTIVE = 6


def _allocate_colors(num_legends):
    """Pick unit colors (shared by the two sprites in each legend), backgrounds, and frame colors."""
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


def _random_filled_sprite(N: int, c1: int, c2: int) -> np.ndarray:
    sprite = np.full((N, N), c1, dtype=int)
    num_c2 = random.randint(1, N * N - 1)
    for p in random.sample(range(N * N), num_c2):
        sprite[p // N, p % N] = c2
    return sprite


def _design_sprite_pair(N: int, unit_colors: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    c1, c2 = unit_colors
    min_diff = max(MIN_SPRITE_DIFF, N)
    for _ in range(200):
        sprite_a = _random_filled_sprite(N, c1, c2)
        sprite_b = _random_filled_sprite(N, c1, c2)
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


def _feasible_unit_sizes(grid_h: int) -> list[int]:
    min_frame_h = FRAME_H_RANGE[0]
    out = []
    for N in range(UNIT_SIZE_RANGE[0], UNIT_SIZE_RANGE[1] + 1):
        after_sprites = 2 * N + SPRITE_STACK_GAP + 1
        marker_r0 = grid_h - MARKER_PANEL_HEIGHT
        avail = marker_r0 - after_sprites
        if avail >= min_frame_h + 1:
            out.append(N)
    return out


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w):
    """Fill in both sprites, the frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    sprites = _design_sprite_pair(N, unit_colors)
    if sprites is None:
        return False
    sprite_a, sprite_b = sprites
    uc1, uc2 = unit_colors
    legend_sprite_a = _color_swap(sprite_a, uc1, uc2)
    legend_sprite_b = _color_swap(sprite_b, uc1, uc2)

    after_sprites = 2 * N + SPRITE_STACK_GAP + 1
    avail = marker_r0 - after_sprites
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

    for _ in range(200):
        frame_mask = random_sprite(
            frame_H,
            frame_W,
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
            rotated_frame=rot_frame,
        )
        return True
    return False


def _comp_size(leg):
    rf = leg["rotated_frame"]
    N = leg["N"]
    return rf.shape[0] * N, rf.shape[1] * N


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign anchor positions to legends, avoiding overlap for 2-legend case. Returns False on failure."""
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
    """Stamp each legend's alternating composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite(leg["rotated_frame"], leg["sprite_a"], leg["sprite_b"])
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    feasible_sizes = _feasible_unit_sizes(grid_h)
    if not feasible_sizes:
        return None

    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]
    legends = []
    for _ in range(num_legends):
        N = random.choice(feasible_sizes)
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
    """Draw one legend panel: two sprites, frame, marker square, and rotation patch."""
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
    spr_c0 = mc0 + sp_col
    grid[1 : 1 + N, spr_c0 : spr_c0 + N] = leg["legend_sprite_a"]
    row_b0 = 1 + N + SPRITE_STACK_GAP
    grid[row_b0 : row_b0 + N, spr_c0 : spr_c0 + N] = leg["legend_sprite_b"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    after_sprites = 2 * N + SPRITE_STACK_GAP + 1
    avail = marker_r0 - after_sprites
    gap = random.randint(1, max(1, avail - fH))
    fr_start = after_sprites + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start : fr_start + fH, mc0 + f_col : mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
