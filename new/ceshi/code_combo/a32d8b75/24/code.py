# Combo variant of a32d8b75 (index 24)
# combo_spec: D13/2+D15/2
# Name: 纯精灵输出+条件分支

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The input keeps the original legend-plus-canvas scene: a left legend, a full-height
#   separator column of color 6, and a background canvas on the right. Inside the legend,
#   top-to-bottom order is still unit sprite -> frame mask -> 5x5 anchor square -> 3x3
#   rotation hand. The anchor square remains visible but no longer affects the answer.
# - The D13 change removes the final placement crop. The output is only the constructed
#   composite sprite rectangle, with no canvas around it.
# - The D15 change modifies how the clock-hand is applied. Example:
#   unit =
#     1 2
#     1 1
#   frame =
#     1 1
#     0 1
#   rotation = right turn.
#   Old/original path: rotate the 2x2 frame layout first, then stamp the unrotated unit
#   into its active macro-cells.
#   New combo path:
#     odd active-count (3 cells): same as above.
#     even active-count (say a 2x2 full frame with 4 cells): stamp the unit into the
#       original frame first, then rotate the whole enlarged sprite. That rotates every
#       micro-pattern inside the tiles, so the output differs even though the outer size
#       stays the same.
# - Because the output is just the composite block, the branch effect is easier to see:
#   there is no surrounding canvas to hide the difference between "frame rotated" and
#   "whole composite rotated".
#
# b) Complexity calibration
# - The branch cue is only one binary attribute: frame active-count parity. That keeps
#   the rule inferable from a few examples.
# - To avoid collapse, generation uses exactly one legend. Once the background output is
#   removed, multiple legends would require a new composition rule between several
#   composites, which is outside this combo spec.
# - Generation forces quarter-turn rotations (`k in {1,3}`) and asymmetric unit sprites,
#   because a 0/180-degree turn or a rotationally symmetric unit would often erase the
#   difference between the two operation orders.
#
# c) Hidden assumption audit
# - Original main() assumed the output grid was the canvas crop. That no longer holds, so
#   all canvas-writing logic is removed from the solver.
# - Original main() also assumed the rotation marker always acted on the coarse frame.
#   That assumption is now conditional on parity; even-cardinality frames must use a
#   literal post-tiling rotation of the enlarged composite.
# - The original task could support 1-2 legends because several composites could be
#   placed on the same canvas. In this combo, returning "only the composite itself"
#   creates ambiguity if more than one legend exists, so the generator guarantees a single
#   legend and the solver returns exactly one composite.
#
# d) Placement legality
# - Even though the answer ignores placement, the input still contains a valid anchor
#   square. The generator continues to choose anchors that would place the composite fully
#   inside the old canvas. This keeps the legend internally consistent with the original
#   task family and prevents malformed examples.
#
# e) Visual observability
# - Humans can count odd versus even active frame cells directly from the legend.
# - The branch effect is made observable by rejecting samples where the chosen branch and
#   the opposite branch yield the same composite.
# - The output size is forced to be strictly smaller than the old canvas crop, so the D13
#   change is visible as a true object-only answer rather than a disguised canvas slice.
#
# f) Visual cleanliness
# - There is only one legend, one frame, and one returned object. No overlapping objects,
#   no stacking on the canvas, and no extra control glyphs are introduced.
# - Internal zeros remain only where the frame has inactive cells, so the output stays a
#   clean rectangular sprite with meaningful holes rather than visual clutter.
#
# g) Change inferability
# - No new color mapping is introduced. The legend still shows the color-swapped unit
#   sprite exactly as in the source task, and the frame uses a distinct single color.
# - The only two nontrivial deductions are: recover the swapped unit, and decide whether
#   parity sends the rotation to the frame scaffold or to the already-expanded composite.
#
# 9) New main() assumptions and generator guarantees
# - main() assumes there is exactly one usable legend marker outside the canvas; the
#   generator builds exactly one legend panel.
# - main() assumes the frame is a single 8-connected component; the generator rejects any
#   sampled frame that violates this.
# - main() assumes the branch matters on generated examples; the generator rejects samples
#   where odd/even order would coincidentally produce the same composite.
# - main() ignores the anchor square by design, but the generator still fills it with a
#   legal placement code so the input remains structurally faithful to the source task.


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    runs: list[tuple[int, int]] = []
    start = end = cols6[0]
    for c in cols6[1:]:
        if c == end + 1:
            end = c
        else:
            runs.append((start, end))
            start = end = c
    runs.append((start, end))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Locate the left separator and an optional right separator near the far edge."""
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
    """List every 5x5 border-6 square that has a 3x3 rotation patch beneath it."""
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


def _legend_marker_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """Return marker panels that lie outside the separator-delimited canvas."""
    h, w = g.shape
    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    return [(r0, c0) for r0, c0 in _all_marker_panel_squares(g) if not (canvas_c0 <= c0 < canvas_c1)]


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Recover the top-most 4-connected non-{0,6} component and unswap its two colors."""
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    vis[rmin, cseed] = True
    stack = [(rmin, cseed)]
    pts: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        pts.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                stack.append((rr, cc))
    r0, c0, r1, c1 = bbox_coords(np.asarray(pts))
    spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
    cols = sorted({int(x) for x in np.unique(spr) if int(x) not in (0, 6)})
    if len(cols) == 2:
        a, b = cols
        out = spr.copy()
        out[spr == a] = -1
        out[spr == b] = a
        out[out == -1] = b
        spr = out
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    """Frame pixels occupy the rows between the unit sprite and the marker square."""
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
    """Read the anchor marker from the 3x3 interior of the border-6 square."""
    inner = legend[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """Decode up/right/down/left from the 3x3 clock-hand patch under the marker square."""
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
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
    """Map a start/middle/end code onto a legal top-left placement coordinate."""
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


def _tile_unit_into_frame(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Stamp the recovered unit sprite into each active frame cell."""
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


def _rotate_frame_first(frame_mask0: np.ndarray) -> bool:
    """Odd active-cell count means rotate the coarse frame before tiling."""
    return int(np.count_nonzero(frame_mask0)) % 2 == 1


def _original_order_composite(unit_sprite: np.ndarray, frame_mask0: np.ndarray, dir_k: int) -> np.ndarray:
    """The source-task order: rotate frame first, then tile the unrotated unit."""
    frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
    return _tile_unit_into_frame(unit_sprite, frame_mask)


def _build_composite(unit_sprite: np.ndarray, frame_mask0: np.ndarray, dir_k: int) -> np.ndarray:
    """Apply the parity branch and return the final composite sprite."""
    if _rotate_frame_first(frame_mask0):
        return _original_order_composite(unit_sprite, frame_mask0, dir_k)
    base = _tile_unit_into_frame(unit_sprite, frame_mask0)
    return np.rot90(base, (-dir_k) % 4)


def _opposite_branch_composite(unit_sprite: np.ndarray, frame_mask0: np.ndarray, dir_k: int) -> np.ndarray:
    """Return the composite from the branch that was not selected by parity."""
    if _rotate_frame_first(frame_mask0):
        base = _tile_unit_into_frame(unit_sprite, frame_mask0)
        return np.rot90(base, (-dir_k) % 4)
    return _original_order_composite(unit_sprite, frame_mask0, dir_k)


def _legend_diagnostics_from_grid(grid: np.ndarray) -> dict:
    """Expose branch-related facts for testing and anti-degeneracy checks."""
    g = np.asarray(grid, dtype=int)
    marker_squares = _legend_marker_squares(g)
    assert marker_squares, "no legend marker panel found"
    marker_r0, marker_c0 = marker_squares[0]
    panel = g[:, marker_c0 : marker_c0 + 5]

    unit = _extract_unit_sprite(panel)
    assert unit is not None, "no unit sprite found"
    unit_sprite, unit_bb = unit

    frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
    assert frame_bb is not None, "no frame found"
    fr0, fc0, fr1, fc1 = frame_bb
    frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
    frame_mask0 = (frame != 0) & (frame != 6)
    dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)

    chosen = _build_composite(unit_sprite, frame_mask0, dir_k)
    other = _opposite_branch_composite(unit_sprite, frame_mask0, dir_k)
    original = _original_order_composite(unit_sprite, frame_mask0, dir_k)
    return {
        "active_count": int(np.count_nonzero(frame_mask0)),
        "parity": int(np.count_nonzero(frame_mask0)) % 2,
        "rotate_frame_first": _rotate_frame_first(frame_mask0),
        "dir_k": dir_k,
        "chosen_shape": tuple(map(int, chosen.shape)),
        "branch_effect_visible": not np.array_equal(chosen, other),
        "differs_from_original_order": not np.array_equal(chosen, original),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the legend, choose the rotation stage by frame parity, and return only the final composite sprite.

    Input structure:
    - The input still shows the full legend-plus-canvas arrangement from the source task: a legend panel, a color-6 separator, and a background canvas region.
    - Inside the legend, the informative objects are a two-color unit sprite, a single-color frame mask, a 5x5 anchor square, and a 3x3 rotation hand.
    - The frame's active-cell parity is now meaningful, while the anchor square remains present only as leftover metadata from the original family.

    Rule:
    - Recover the unswapped unit sprite, extract the frame mask, and read the quarter-turn direction from the clock-hand.
    - If the frame has an odd number of active cells, rotate the frame layout first and then tile the unrotated unit into that rotated scaffold.
    - If the frame has an even number of active cells, tile the unit into the original frame first and then rotate the entire enlarged composite.
    - Ignore the canvas background and ignore the anchor placement code; output only the resulting composite rectangle.

    Key insight:
    - Two independent changes interact here: the answer is no longer a canvas crop, and the clock-hand does not always act on the same level. You must notice that parity decides whether rotation happens before or after expansion, then return the object itself with no surrounding background.
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
    _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3), dtype=int))
    assert n8 == 1, f"frame must be 8-connected, got {n8} components"

    dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
    return _build_composite(unit_sprite, frame_mask0, dir_k)


GRID_H_RANGE = (20, 30)
GRID_W = 30
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)  # 4..8
FRAME_W_RANGE = (3, 6)  # 3..5
FRAME_DENSITY = 0.5
FRAME_ACTIVE_RANGE = (7, 17)
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ROTATION_CHOICES = [1, 3]
GEN_RETRIES = 600
OUTPUT_MARGIN = 1
_PARITY_CYCLE = 0


def _allocate_colors() -> tuple[tuple[int, int], tuple[int, int], int]:
    """Pick two unit colors, two canvas colors, and one frame color."""
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


def _make_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray | None:
    """Generate a filled two-color unit whose quarter-turn visibly changes it."""
    for _ in range(120):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        rot = np.rot90(sprite, 1)
        if np.array_equal(sprite, rot):
            continue
        if int(np.count_nonzero(sprite != rot)) < max(3, n):
            continue
        return sprite
    return None


def _make_frame_mask(frame_h: int, frame_w: int, desired_parity: int) -> np.ndarray | None:
    """Sample an 8-connected frame mask with the requested active-count parity."""
    lo, hi = FRAME_ACTIVE_RANGE
    for _ in range(200):
        mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        count = int(np.count_nonzero(mask))
        if count < lo or count > hi:
            continue
        if count % 2 != desired_parity:
            continue
        if count == frame_h * frame_w:
            continue
        _, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
        if n8 != 1:
            continue
        return mask
    return None


def _branch_effect_visible(sprite: np.ndarray, frame_mask: np.ndarray, rot_k: int) -> bool:
    """Reject samples where both operation orders accidentally look identical."""
    chosen = _build_composite(sprite, frame_mask, rot_k)
    other = _opposite_branch_composite(sprite, frame_mask, rot_k)
    return not np.array_equal(chosen, other)


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    desired_parity: int,
) -> bool:
    """Fill one legend dict with a visible non-degenerate branch case."""
    n = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _make_unit_sprite(n, uc1, uc2)
    if sprite is None:
        return False
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos: list[tuple[int, int, int]] = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            for rot_k in ROTATION_CHOICES:
                rot_h, rot_w = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                if rot_h * n > grid_h - OUTPUT_MARGIN:
                    continue
                if rot_w * n > canvas_w - OUTPUT_MARGIN:
                    continue
                combos.append((frame_h, frame_w, rot_k))
    random.shuffle(combos)

    for frame_h, frame_w, rot_k in combos:
        frame_mask = _make_frame_mask(frame_h, frame_w, desired_parity)
        if frame_mask is None:
            continue
        comp = _build_composite(sprite, frame_mask, rot_k)
        if comp.shape[0] >= grid_h or comp.shape[1] >= canvas_w:
            continue
        if not _branch_effect_visible(sprite, frame_mask, rot_k):
            continue
        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame_mask=frame_mask,
            frame_H=frame_h,
            frame_W=frame_w,
            frame_color=frame_color,
            rot_k=rot_k,
            desired_parity=desired_parity,
            active_count=int(np.count_nonzero(frame_mask)),
            comp=comp,
        )
        return True
    return False


def _assign_anchor(leg: dict, grid_h: int, canvas_w: int) -> bool:
    """Pick a legal original-task anchor even though the combo output ignores placement."""
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
    """Build a simple striped canvas background that is ignored by the combo output."""
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
    """Draw the single legend panel using the source task's spatial template."""
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


def _gen_attempt(desired_parity: int) -> np.ndarray | None:
    """Output-first construction for one combo-variant example."""
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

    if not _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w, desired_parity):
        return None
    if not _assign_anchor(leg, grid_h, canvas_w):
        return None

    expected = leg["comp"]
    if expected.shape == (grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas
    _place_legend(grid, leg, col_start=0, sep_col=legend_w, marker_r0=marker_r0)

    try:
        if not np.array_equal(main(grid), expected):
            return None
        diag = _legend_diagnostics_from_grid(grid)
        if diag["parity"] != desired_parity:
            return None
        if not diag["branch_effect_visible"]:
            return None
        if desired_parity == 0 and not diag["differs_from_original_order"]:
            return None
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    """Sample until a valid combo-variant instance is built."""
    global _PARITY_CYCLE
    desired_parity = _PARITY_CYCLE % 2
    _PARITY_CYCLE += 1
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(desired_parity)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
