# Combo variant of a32d8b75 (index 9)
# combo_spec: D6/2+D16/2
# Name: 相对偏移+背景标记定位

import itertools
import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
spec_like:
  unchanged_core:
    - Legends still encode a two-color unit sprite, an 8-connected frame mask, and a rotation cue.
    - The solver still color-swaps the unit sprite, tiles it into every active frame cell, rotates the frame layout, and paints the resulting composite onto the canvas.
  fused_change:
    - The old 5x5 anchor square becomes a 7x7 square with a 5x5 interior, like D6/2.
    - Its inner color-4 pixel no longer means start/middle/end. It now maps to a proportional target location on the canvas:
      target_row = round_half_up((marker_row / 4) * (canvas_h - 1))
      target_col = round_half_up((marker_col / 4) * (canvas_w - 1))
    - The final placement anchor does not come from that square. Instead, the canvas contains several color-4 marker pixels, and the solver must select the marker closest to the square-implied target.
    - That chosen canvas marker acts like D16/2's hidden placement anchor: it aligns to the composite's top-left-most painted pixel.
  trace_example:
    - Suppose a legend decodes to inner marker (1, 3) inside the 5x5 square and the canvas size is 20x16.
    - The relative target becomes approximately (5, 11).
    - The canvas may contain hidden markers at (6, 10), (1, 2), and (17, 4). The nearest one is (6, 10).
    - If the built composite's top-left-most painted pixel sits at offset (2, 1) inside the composite, then the final top-left is (6-2, 10-1) = (4, 9).
    - Original task: square alone directly produced one of nine coarse anchor alignments.
    - Fused task: square predicts where to look, but the exact anchor comes from the selected hidden canvas marker, so both signals affect placement.

b) Complexity calibration
  - The 5x5 interior gives 25 relative target codes, which is richer than the old 3x3 but still small enough to infer visually.
  - Hidden markers are kept to a small count: one true marker per legend plus 1-2 decoys total. This prevents clutter while still forcing disambiguation.
  - The generator requires at least one odd relative coordinate per legend, so the visible square often uses quarter-style positions instead of collapsing to the original coarse lattice.
  - Decoys are spaced away from the target neighborhoods, so the intended nearest-marker rule is stable rather than noisy.

c) Implicit assumption audit for the original main()
  - `_all_marker_panel_squares()` originally assumed a 5x5 square with a 3x3 payload. The combo needs a 7x7 square with a 5x5 payload and a lower rotation patch.
  - `_position_indices_from_border_square()` assumed the square directly encoded anchor classes. That assumption is invalid: the square now yields a canvas target, not a placement anchor.
  - `_anchor_to_top_left()` encoded start/middle/end semantics. That logic must be removed from placement.
  - The original code tolerated color 4 in arbitrary roles. The combo cannot: color 4 is now reserved for both legend-side target markers and canvas-side hidden anchor markers, so generation must exclude 4 from unit/frame/background palettes.
  - Pairing legend panels to canvas markers is no longer implicit. The solver now assumes that the best distinct assignment between legend targets and hidden markers is well-defined; the generator enforces that by construction.

d) Placement legality
  - Each legend is designed output-first: build the rotated composite, compute its painted reference offset, then choose a concrete top-left inside bounds.
  - The true canvas marker is derived from that legal placement and is forced to lie near, but not exactly on, the relative target predicted by the legend square.
  - For two legends, bounding boxes may not overlap, and each legend's true marker must be strictly closer to its own target than any other true marker.
  - Decoy markers are placed outside all composite bounding boxes and far enough from every target to avoid stealing the nearest-marker assignment.

e) Visual observability
  - The legend square is visibly larger than the seed task and contains a single off-center color-4 pixel.
  - The canvas now shows multiple isolated color-4 pixels. In the output, only the chosen marker per legend disappears under a composite; the decoys remain.
  - Across samples, the square's relative position predicts which marker is relevant, while the exact marker location explains the final one- or two-pixel adjustment. That gives a human a visible two-stage inference path.

f) Visual cleanliness
  - Marker decoys stay outside placed composites, so leftover color-4 pixels remain readable instead of being half-covered.
  - The marker square occupies the full legend width, which keeps sprite/frame content visually separated from the placement code.
  - Rotation stays in its own centered 3x3 patch below the square, avoiding ambiguity with the 5x5 target marker interior.

g) Change inferability
  - No opaque color remapping is added. Color 4 always means "placement clue", but now clues exist in both the legend and the canvas.
  - The square conveys a coarse proportional search region, and the canvas marker gives the exact anchor. The rule is geometric and visible rather than dependent on hidden color identities.

h) New assumptions introduced by this combo
  - main() assumes each legend square contains exactly one inner marker pixel and each canvas contains at least as many hidden markers as legends.
  - main() assumes the optimal distinct assignment from legend targets to hidden markers is the intended one.
  - generate_input() guarantees those assumptions by reserving color 4, injecting decoys, and enforcing nearest-marker margins.
"""


MARKER_COLOR = 4
MARKER_INNER_SIZE = 5
MARKER_BOX_SIZE = MARKER_INNER_SIZE + 2
ROTATION_PATCH_HEIGHT = 3
MARKER_PANEL_HEIGHT = MARKER_BOX_SIZE + ROTATION_PATCH_HEIGHT

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 3]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
USABLE_COLORS = [1, 2, 3, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 500
DECOY_COUNT_BY_LEGENDS = {1: 1, 2: 2}
TRUE_MARKER_DELTAS = [
    (-2, 0),
    (0, -2),
    (0, 2),
    (2, 0),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
]


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
    right_min_start = w - (MARKER_BOX_SIZE + 3)
    for s, e in reversed(runs[1:]):
        if s >= right_min_start and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 7x7 border-6 squares that look like the top of a marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - MARKER_PANEL_HEIGHT + 1):
        for c0 in range(w - MARKER_BOX_SIZE + 1):
            sub = g[r0 : r0 + MARKER_BOX_SIZE, c0 : c0 + MARKER_BOX_SIZE]
            if not (
                np.all(sub[0, :] == 6)
                and np.all(sub[-1, :] == 6)
                and np.all(sub[:, 0] == 6)
                and np.all(sub[:, -1] == 6)
            ):
                continue
            inner = sub[1:-1, 1:-1]
            pts = np.argwhere((inner != 0) & (inner != 6))
            if pts.shape[0] != 1:
                continue
            rot_band = g[r0 + MARKER_BOX_SIZE : r0 + MARKER_PANEL_HEIGHT, c0 : c0 + MARKER_BOX_SIZE]
            if rot_band.shape != (ROTATION_PATCH_HEIGHT, MARKER_BOX_SIZE):
                continue
            if not (np.all(rot_band[:, 0] == 6) and np.all(rot_band[:, -1] == 6)):
                continue
            rot = rot_band[:, 2:5]
            if rot.shape != (3, 3) or np.count_nonzero((rot != 0) & (rot != 6)) == 0:
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


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    # Frame pixels live between the unit sprite and the border-6 marker square.
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


def _marker_relative_indices(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    sq = g[border6_r0 : border6_r0 + MARKER_BOX_SIZE, border6_c0 : border6_c0 + MARKER_BOX_SIZE]
    inner = sq[1:-1, 1:-1]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.shape[0] != 1:
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def _rotation_from_clockhand(g: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker is in the centered 3x3 patch directly below the 7x7 position square.
    Use the furthest non-{0,6} pixel from the patch center to determine direction:
      up -> 0, right -> 1, down -> 2, left -> 3.
    """
    h, w = g.shape
    r0 = border6_r0 + MARKER_BOX_SIZE
    c0 = border6_c0 + 2
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = g[r0 : r0 + 3, c0 : c0 + 3]
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


def _build_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile the unit sprite into each active frame cell."""
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


def _reference_offset(comp: np.ndarray) -> tuple[int, int]:
    """Use the top-left-most painted pixel as the anchor-alignment reference."""
    pts = np.argwhere((comp != 0) & (comp != 6))
    assert pts.size > 0, "composite must contain painted pixels"
    r = int(pts[:, 0].min())
    c = int(pts[pts[:, 0] == r][:, 1].min())
    return r, c


def _relative_canvas_coord(marker_idx: int, size: int) -> int:
    """Map a 0..4 legend coordinate to a proportional canvas coordinate on one axis."""
    if size <= 1:
        return 0
    denom = MARKER_INNER_SIZE - 1
    return (marker_idx * (size - 1) + denom // 2) // denom


def _hidden_markers(main: np.ndarray) -> list[tuple[int, int]]:
    """Return all hidden canvas marker pixels in stable order."""
    pts = np.argwhere(main == MARKER_COLOR)
    out = [(int(r), int(c)) for r, c in pts]
    out.sort(key=lambda rc: (rc[1], rc[0]))
    return out


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _best_marker_assignment(
    targets: list[tuple[int, int]], hidden: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Assign distinct hidden markers to legend targets by minimum total Manhattan distance."""
    assert len(hidden) >= len(targets), "not enough hidden markers"
    best_key: tuple[int, int, tuple[int, ...]] | None = None
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(len(hidden)), len(targets)):
        dists = [_manhattan(targets[i], hidden[perm[i]]) for i in range(len(targets))]
        key = (sum(dists), max(dists), perm)
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm
    assert best_perm is not None
    return [hidden[i] for i in best_perm]


def _paint_composite(
    out: np.ndarray, comp: np.ndarray, top: int, left: int, composite_written: np.ndarray
) -> None:
    ch, cw = comp.shape
    assert 0 <= top and top + ch <= out.shape[0], "sprite row out of canvas bounds"
    assert 0 <= left and left + cw <= out.shape[1], "sprite col out of canvas bounds"
    for r in range(ch):
        rr = top + r
        for c in range(cw):
            cc = left + c
            if out[rr, cc] == 6:
                continue
            val = int(comp[r, c])
            if val != 0 and val != 6:
                assert not composite_written[rr, cc], "composites must not overlap"
                composite_written[rr, cc] = True
                out[rr, cc] = val


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble enlarged sprites from legend panels and place them using fused relative-target and hidden-anchor cues.

    Input structure:
    - Color-6 separator columns still split the full grid into one or two legend panels and a central canvas.
    - Each legend contains a color-swapped unit sprite, a frame template, a 7x7 bordered square with one inner color-4 marker pixel, and a centered 3x3 rotation patch below.
    - The canvas background contains multiple color-4 marker pixels: one true anchor marker per legend plus 1-2 decoys.

    Rule:
    - Rebuild each composite by color-swapping the unit sprite back, tiling it into the active frame cells, and rotating the frame layout from the clock-hand cue.
    - Decode the 5x5 inner marker position as a proportional target location on the canvas, then choose the distinct hidden canvas marker assignment that best matches those targets.
    - For each legend, align the composite's top-left-most painted pixel to its assigned hidden marker and paint the composite over the canvas background.

    Key insight:
    - The legend square no longer gives the final anchor directly. It only tells you which area of the canvas to search, while the exact placement comes from the nearby hidden marker pixel embedded in the background.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    legend_squares = [rc for rc in marker_squares if not (canvas_c0 <= rc[1] < canvas_c1)]
    legend_squares.sort(key=lambda rc: rc[1])
    assert legend_squares, "no legend marker panel found"

    hidden = _hidden_markers(main_canvas)
    assert len(hidden) >= len(legend_squares), "hidden canvas marker count must cover legend count"

    legend_infos = []
    targets: list[tuple[int, int]] = []
    for marker_r0, marker_c0 in legend_squares:
        panel = g[:, marker_c0 : marker_c0 + MARKER_BOX_SIZE]

        unit = _extract_unit_sprite(panel)
        assert unit is not None, "legend unit sprite missing"
        unit_sprite, unit_bb = unit

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        assert frame_bb is not None, "legend frame missing"
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_composite(unit_sprite, frame_mask)
        ref_r, ref_c = _reference_offset(comp)

        rel = _marker_relative_indices(g, marker_r0, marker_c0)
        assert rel is not None, "relative target marker missing"
        target = (
            _relative_canvas_coord(rel[0], main_canvas.shape[0]),
            _relative_canvas_coord(rel[1], main_canvas.shape[1]),
        )

        legend_infos.append({"comp": comp, "ref": (ref_r, ref_c)})
        targets.append(target)

    assigned_markers = _best_marker_assignment(targets, hidden)

    out = main_canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)
    for info, marker in zip(legend_infos, assigned_markers):
        ref_r, ref_c = info["ref"]
        top = marker[0] - ref_r
        left = marker[1] - ref_c
        _paint_composite(out, info["comp"], top, left, composite_written)
    return out


def _allocate_colors(num_legends: int):
    """Pick unit colors, background colors, and frame colors while reserving color 4."""
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


def _design_legend(leg: dict, unit_colors, frame_color: int, marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    """Fill in sprite, frame, rotation, and composite geometry for one legend dict."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2

    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos = []
    for f_h in range(*FRAME_H_RANGE):
        if avail < f_h + 1:
            continue
        for f_w in range(*FRAME_W_RANGE):
            if f_w > leg["legend_w"]:
                continue
            for rk in range(4):
                rh, rw = (f_h, f_w) if rk % 2 == 0 else (f_w, f_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((f_h, f_w, rk))
    if not combos:
        return False
    frame_h, frame_w, rot_k = random.choice(combos)

    frame_mask = random_sprite(
        frame_h,
        frame_w,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
    comp = _build_composite(sprite, rot_frame)
    ref_r, ref_c = _reference_offset(comp)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        comp=comp,
        comp_h=comp.shape[0],
        comp_w=comp.shape[1],
        ref_r=ref_r,
        ref_c=ref_c,
    )
    return True


def _rects_overlap(a_top: int, a_left: int, a_h: int, a_w: int, b_top: int, b_left: int, b_h: int, b_w: int) -> bool:
    return a_top < b_top + b_h and b_top < a_top + a_h and a_left < b_left + b_w and b_left < a_left + a_w


def _candidate_placements(leg: dict, grid_h: int, canvas_w: int) -> list[dict]:
    """
    Candidate placements where the true hidden marker lies near, but not exactly on,
    the relative target predicted by the legend square.
    """
    candidates_by_rel: dict[tuple[int, int], list[dict]] = {}
    for rel_r in range(MARKER_INNER_SIZE):
        for rel_c in range(MARKER_INNER_SIZE):
            if rel_r % 2 == 0 and rel_c % 2 == 0:
                continue
            target_r = _relative_canvas_coord(rel_r, grid_h)
            target_c = _relative_canvas_coord(rel_c, canvas_w)
            for dr, dc in TRUE_MARKER_DELTAS:
                marker_r = target_r + dr
                marker_c = target_c + dc
                top = marker_r - leg["ref_r"]
                left = marker_c - leg["ref_c"]
                if top < 0 or left < 0:
                    continue
                if top + leg["comp_h"] > grid_h or left + leg["comp_w"] > canvas_w:
                    continue
                candidates_by_rel.setdefault((rel_r, rel_c), []).append(
                    {
                        "rel_r": rel_r,
                        "rel_c": rel_c,
                        "target_r": target_r,
                        "target_c": target_c,
                        "marker_r": marker_r,
                        "marker_c": marker_c,
                        "comp_top": top,
                        "comp_left": left,
                        "true_dist": abs(dr) + abs(dc),
                    }
                )
    candidates = [random.choice(group) for group in candidates_by_rel.values()]
    random.shuffle(candidates)
    return candidates


def _pair_is_consistent(a: dict, b: dict, a_pick: dict, b_pick: dict) -> bool:
    if (a_pick["marker_r"], a_pick["marker_c"]) == (b_pick["marker_r"], b_pick["marker_c"]):
        return False
    if _rects_overlap(
        a_pick["comp_top"],
        a_pick["comp_left"],
        a["comp_h"],
        a["comp_w"],
        b_pick["comp_top"],
        b_pick["comp_left"],
        b["comp_h"],
        b["comp_w"],
    ):
        return False
    a_target = (a_pick["target_r"], a_pick["target_c"])
    b_target = (b_pick["target_r"], b_pick["target_c"])
    a_true = (a_pick["marker_r"], a_pick["marker_c"])
    b_true = (b_pick["marker_r"], b_pick["marker_c"])
    if _manhattan(a_target, a_true) >= _manhattan(a_target, b_true):
        return False
    if _manhattan(b_target, b_true) >= _manhattan(b_target, a_true):
        return False
    return True


def _assign_true_markers(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Choose legal composite placements and derive one true hidden marker per legend."""
    cand_lists = [_candidate_placements(leg, grid_h, canvas_w) for leg in legends]
    if any(not cands for cands in cand_lists):
        return False

    if len(legends) == 1:
        pick = random.choice(cand_lists[0])
        legends[0].update(pick)
        return True

    valid_pairs = []
    a, b = legends
    for a_pick in cand_lists[0]:
        for b_pick in cand_lists[1]:
            if _pair_is_consistent(a, b, a_pick, b_pick):
                valid_pairs.append((a_pick, b_pick))
    if not valid_pairs:
        return False
    a_pick, b_pick = random.choice(valid_pairs)
    a.update(a_pick)
    b.update(b_pick)
    return True


def _cell_in_any_rect(r: int, c: int, rects: list[tuple[int, int, int, int]]) -> bool:
    for top, left, height, width in rects:
        if top <= r < top + height and left <= c < left + width:
            return True
    return False


def _choose_decoys(legends: list[dict], grid_h: int, canvas_w: int) -> list[tuple[int, int]] | None:
    """Pick extra hidden markers that are outside composites and cannot win target matching."""
    need = DECOY_COUNT_BY_LEGENDS[len(legends)]
    rects = [(leg["comp_top"], leg["comp_left"], leg["comp_h"], leg["comp_w"]) for leg in legends]
    true_markers = {(leg["marker_r"], leg["marker_c"]) for leg in legends}
    target_info = [
        (
            (leg["target_r"], leg["target_c"]),
            (leg["marker_r"], leg["marker_c"]),
        )
        for leg in legends
    ]

    candidates = []
    for r in range(grid_h):
        for c in range(canvas_w):
            pos = (r, c)
            if pos in true_markers:
                continue
            if _cell_in_any_rect(r, c, rects):
                continue
            ok = True
            for target, true_marker in target_info:
                if _manhattan(target, pos) <= _manhattan(target, true_marker) + 1:
                    ok = False
                    break
            if ok:
                candidates.append(pos)
    if len(candidates) < need:
        return None
    return random.sample(candidates, need)


def _square_signal_is_active(legends: list[dict], hidden_markers: list[tuple[int, int]]) -> bool:
    """
    Reject samples where a naive "take the left-most markers" strategy matches the
    relative-target assignment, because those samples underuse the legend-side cue.
    """
    hidden_sorted = sorted(hidden_markers, key=lambda rc: (rc[1], rc[0]))
    targets = [(leg["target_r"], leg["target_c"]) for leg in legends]
    assigned = _best_marker_assignment(targets, hidden_sorted)
    naive = hidden_sorted[: len(legends)]
    return assigned != naive


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
    """Stamp each legend's composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + leg["comp_h"], l : l + leg["comp_w"]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int) -> None:
    """Draw one legend panel with a 7x7 relative-target square and centered rotation patch."""
    n = leg["N"]
    legend_w = leg["legend_w"]
    grid[:, sep_col] = 6

    mc0 = col_start

    grid[marker_r0 : marker_r0 + MARKER_BOX_SIZE, mc0 : mc0 + MARKER_BOX_SIZE] = 0
    grid[marker_r0, mc0 : mc0 + MARKER_BOX_SIZE] = 6
    grid[marker_r0 + MARKER_BOX_SIZE - 1, mc0 : mc0 + MARKER_BOX_SIZE] = 6
    grid[marker_r0 : marker_r0 + MARKER_BOX_SIZE, mc0] = 6
    grid[marker_r0 : marker_r0 + MARKER_BOX_SIZE, mc0 + MARKER_BOX_SIZE - 1] = 6
    grid[marker_r0 + 1 + leg["rel_r"], mc0 + 1 + leg["rel_c"]] = MARKER_COLOR

    rr0 = marker_r0 + MARKER_BOX_SIZE
    grid[rr0 : rr0 + ROTATION_PATCH_HEIGHT, mc0 : mc0 + MARKER_BOX_SIZE] = 0
    grid[rr0 : rr0 + ROTATION_PATCH_HEIGHT, mc0] = 6
    grid[rr0 : rr0 + ROTATION_PATCH_HEIGHT, mc0 + MARKER_BOX_SIZE - 1] = 6
    cx_r, cx_c = rr0 + 1, mc0 + MARKER_BOX_SIZE // 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sp_left = col_start + (legend_w - n) // 2
    grid[1 : 1 + n, sp_left : sp_left + n] = leg["legend_sprite"]

    f_h, f_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - f_h))
    fr_start = n + 1 + gap
    f_col = col_start + random.randint(0, legend_w - f_w)
    frame_region = grid[fr_start : fr_start + f_h, f_col : f_col + f_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        n = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": n, "legend_w": max(n + 2, MARKER_BOX_SIZE)})

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
    if marker_r0 <= 2:
        return None

    for i in range(num_legends):
        if not _design_legend(legends[i], unit_color_sets[i], frame_colors[i], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_true_markers(legends, grid_h, canvas_w):
        return None

    decoys = _choose_decoys(legends, grid_h, canvas_w)
    if decoys is None:
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    hidden_markers = [(leg["marker_r"], leg["marker_c"]) for leg in legends] + decoys
    if not _square_signal_is_active(legends, hidden_markers):
        return None
    for r, c in hidden_markers:
        canvas[r, c] = MARKER_COLOR
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0)
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
