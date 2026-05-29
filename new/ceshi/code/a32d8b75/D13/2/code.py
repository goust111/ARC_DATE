# Variant of a32d8b75
# Dimension: D13 — 输出裁剪与网格形状 (grid_shape)
# Change: 输出为画布区域（去掉图例和分隔列），高度不变 → 输出为复合精灵本身（无背景）

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the D13 variant
# -----------------------------------
# source_task: none
#
# a) Concrete behavior and trace
# - The input keeps the original full layout: left legend, pink separator, central canvas,
#   and the bottom 5x5 marker square + 3x3 rotation hand.
# - The legend still encodes three things that matter for the object itself:
#   1) a small NxN unit sprite,
#   2) an HxW frame mask telling which macro-cells are active,
#   3) a rotation direction read from the clock-hand.
# - The old task built `comp` and then placed it onto the canvas according to the 3x3
#   anchor marker, so the output shape was `(grid_h, canvas_w)`.
# - This variant stops one step earlier. Example:
#   unit size N=2, frame after rotation has shape 3x4. Old logic created a 6x8 composite,
#   then embedded it into a 24x23 canvas crop. New logic returns only the 6x8 composite.
#   The 3x3 anchor marker stays visible in the input, but it no longer changes the output.
#
# b) Complexity calibration
# - Keeping the full old-style input prevents the task from collapsing into "copy the top
#   sprite" because the solver still must recover the color-swapped unit, locate the frame,
#   and apply the rotation.
# - To avoid ambiguity from multiple possible outputs, generation is restricted to one legend.
#   That is a D13-specific adaptation: once placement is removed, returning several unrelated
#   composites would require a new composition rule beyond grid-shape change.
# - Rotation must remain visually relevant, so generation rejects cases where the chosen
#   rotated frame equals the unrotated frame for a non-zero rotation.
#
# c) Old implicit assumptions that must change
# - `_split_canvas` and canvas slicing were previously used to define the output size.
#   In the variant, they are used only to distinguish legend panels from the canvas region.
# - The original `main()` computed `eff_h/eff_w`, anchor offsets, and wrote onto `out`.
#   Those lines assumed the canvas survives into the answer. That assumption is removed.
# - The original code also tolerated 1-2 legends because multiple composites could coexist on
#   one canvas. Without a placement stage, combining multiple legends would be a new rule, so
#   the generator now guarantees exactly one legend and `main()` returns the first legend panel.
#
# d) Placement legality
# - Even though the output ignores placement, the input still carries anchor metadata.
#   The generator therefore continues to choose only anchors that would place the composite
#   fully inside the old canvas. This keeps the legend internally consistent and ensures the
#   input would also be legal under the original interpretation.
#
# e) Visual observability
# - The key visible difference is output shape. To make that difference obvious, generation
#   enforces `comp.shape[0] < grid_h` and `comp.shape[1] < canvas_w`; the answer can never
#   coincide with the old canvas crop size.
# - The canvas background uses unrelated colors and is never copied to the output, so a human
#   can infer that the result isolates the constructed sprite rather than any canvas region.
#
# f) Visual cleanliness
# - Only one composite is encoded, so no overlaps or stacking are needed.
# - The output is a single tight rectangle with internal 0-holes coming from inactive frame
#   cells; there is no background clutter around it.
#
# g) Color-rule clarity
# - Colors follow the original structure: the unit sprite uses exactly two colors, the legend
#   shows their swapped version, and the frame has a third color. No new color mapping is
#   introduced for the variant; only the output crop changes.


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


def _legend_marker_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """Return marker panels that lie outside the canvas region, in reading order."""
    h, w = g.shape
    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    return [(r0, c0) for r0, c0 in _all_marker_panel_squares(g) if not (canvas_c0 <= c0 < canvas_c1)]


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

    # The legend stores the color-swapped version of the unit sprite.
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
    # Frame pixels live between the unit sprite and the border-6 square.
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
    """Tile the recovered unit sprite into each active frame cell."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode one legend panel, build its enlarged sprite, and return that composite only.

    Input structure:
    - The grid still contains a legend panel beside a separator-delimited canvas region.
    - The legend still stacks a color-swapped unit sprite, a frame template, a 5x5 anchor square, and a 3x3 rotation hand.

    Rule:
    - Recover the unit sprite, fill every active frame cell with one copy of that sprite, and rotate the frame layout using the clock-hand direction.
    - Ignore the canvas background and ignore the anchor marker when forming the answer.
    - Output the rotated composite rectangle itself, including any internal 0-cells left by inactive frame positions.

    Key insight:
    - This D13 variant removes the placement stage entirely: the legend still tells you how to construct the object, but the answer is the object alone rather than a canvas crop.
    """
    g = np.asarray(grid, dtype=int)
    marker_squares = _legend_marker_squares(g)
    assert marker_squares, "no legend marker panel found"

    marker_r0, marker_c0 = marker_squares[0]
    assert marker_c0 + 5 <= g.shape[1], "legend panel exceeds grid width"
    panel = g[:, marker_c0 : marker_c0 + 5]

    unit = _extract_unit_sprite(panel)
    assert unit is not None, "no unit sprite found"
    unit_sprite, unit_bb = unit

    frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
    assert frame_bb is not None, "no frame found"
    fr0, fc0, fr1, fc1 = frame_bb
    frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
    frame_mask0 = (frame != 0) & (frame != 6)
    _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
    assert n8 == 1, f"frame must be 8-connected, got {n8} components"

    dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
    frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
    return _build_composite(unit_sprite, frame_mask)


# --------------- generator (D13_comp_only_crop) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)   # stop-exclusive: 4..8
FRAME_W_RANGE = (3, 6)   # stop-exclusive: 3..5
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400
OUTPUT_MARGIN = 1


def _allocate_colors():
    """Pick 2 unit colors, 2 background colors, and 1 frame color."""
    pool = list(USABLE_COLORS)
    random.shuffle(pool)
    uc1, uc2 = pool.pop(), pool.pop()
    bg1, bg2 = pool.pop(), pool.pop()
    frame_candidates = [c for c in USABLE_COLORS if c not in (uc1, uc2)]
    frame_color = random.choice(frame_candidates)
    return (uc1, uc2), (bg1, bg2), frame_color


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _random_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray:
    """Generate a fully filled, visibly two-color unit sprite."""
    for _ in range(80):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        if not np.array_equal(sprite, np.rot90(sprite, 1)):
            return sprite
    raise RuntimeError("failed to sample asymmetric unit sprite")


def _sample_frame_mask(frame_h: int, frame_w: int) -> np.ndarray | None:
    """Sample an 8-connected frame mask with moderate fill and at least one hole."""
    for _ in range(80):
        mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        active = int(mask.sum())
        total = frame_h * frame_w
        if active == 0 or active == total:
            continue
        _, n8 = _scipy_label(mask, structure=np.ones((3, 3)))
        if n8 != 1:
            continue
        return mask
    return None


def _design_legend(leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    """Fill one legend dict. Returns False if no valid visible composite can be sampled."""
    n = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _random_unit_sprite(n, uc1, uc2)
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos: list[tuple[int, int, int]] = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                rot_h, rot_w = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                if rot_h * n >= grid_h:
                    continue
                if rot_w * n >= canvas_w:
                    continue
                if rot_h * n > grid_h - OUTPUT_MARGIN:
                    continue
                if rot_w * n > canvas_w - OUTPUT_MARGIN:
                    continue
                combos.append((frame_h, frame_w, rot_k))
    if not combos:
        return False

    random.shuffle(combos)
    for frame_h, frame_w, rot_k in combos:
        frame_mask = _sample_frame_mask(frame_h, frame_w)
        if frame_mask is None:
            continue
        rotated_frame = np.rot90(frame_mask, (-rot_k) % 4)
        if rot_k != 0 and np.array_equal(rotated_frame, frame_mask):
            continue
        comp = _build_composite(sprite, rotated_frame)
        if comp.shape[0] >= grid_h or comp.shape[1] >= canvas_w:
            continue
        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame_mask=frame_mask,
            frame_H=frame_h,
            frame_W=frame_w,
            frame_color=frame_color,
            rot_k=rot_k,
            rotated_frame=rotated_frame,
            comp=comp,
        )
        return True
    return False


def _assign_anchor(leg: dict, grid_h: int, canvas_w: int) -> bool:
    """Pick a marker-square anchor that would legally place the composite on the old canvas."""
    ch, cw = leg["comp"].shape
    valid: list[tuple[int, int, int, int]] = []
    for pr in range(3):
        for pc in range(3):
            top = _anchor_to_top_left(pr, grid_h, ch)
            left = _anchor_to_top_left(pc, canvas_w, cw)
            if top < 0 or left < 0:
                continue
            if top + ch > grid_h or left + cw > canvas_w:
                continue
            valid.append((pr, pc, top, left))
    if not valid:
        return False
    pr, pc, top, left = random.choice(valid)
    leg.update(pr=pr, pc=pc, comp_top=top, comp_left=left)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
    """Build a canvas with alternating horizontal or vertical color bands."""
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    bw = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
    band_indices = np.arange(dim) // bw % 2
    colors = np.array([bg1, bg2], dtype=int)
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int) -> None:
    """Draw the legend panel while keeping the original spatial organization."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = col_start + max(0, n - 4)

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
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    n = random.randint(*UNIT_SIZE_RANGE)
    legend_w = n + 2
    left_total = legend_w + 1
    canvas_w = GRID_W - left_total
    if canvas_w < MIN_CANVAS_W:
        return None

    unit_colors, (bg1, bg2), frame_color = _allocate_colors()
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    leg = {"N": n, "legend_w": legend_w}

    if not _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w):
        return None
    if not _assign_anchor(leg, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    expected = leg["comp"]

    # Force the D13 change to stay visible: the answer must not match the old canvas crop size.
    if expected.shape == canvas.shape:
        return None

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas
    _place_legend(grid, leg, col_start=0, sep_col=legend_w, marker_r0=marker_r0)

    try:
        if np.array_equal(main(grid), expected):
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
