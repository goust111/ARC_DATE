# Combo variant of a32d8b75 (index 5)
# combo_spec: D3/1+D5/1
# Name: 多色框架+旋转翻转
#
# Design analysis
# ===============
# a) Concrete fused rule + behavior trace:
#    - Legend layout stays the same: swapped 2-color unit sprite, frame area, 5x5 anchor square,
#      then a 3x3 rotation patch.
#    - The frame now uses two fixed semantic colors:
#        8 -> place the unit sprite as-is
#        9 -> place a left-right mirrored unit sprite
#    - The rotation patch now encodes a second transform layer:
#        endpoint direction -> quarter-turn rotation
#        center color 7/8/9 -> no flip / horizontal flip / vertical flip
#    - Crucial order: first expand the frame into a base composite using local frame semantics,
#      then apply the marker's global rotate+flip to the whole composite.
#    - Micro example:
#        unit = [[1,2],[1,1]]
#        frame = [[8,9],[9,8]]
#        marker = "point right, center color 8"
#        old task: every active frame cell uses [[1,2],[1,1]]; only the old rotation stage acts.
#        fused task: 8-cells use [[1,2],[1,1]], 9-cells use [[2,1],[1,1]]; after those mixed tiles
#        are assembled, the whole composite rotates right and is then flipped horizontally.
#      Difference: one layer of mirroring is local and repeated by frame color, another is global
#      and controlled independently by the marker center.
# b) Complexity calibration:
#    - Both new degrees of freedom are deliberately low-cardinality: frame has only 2 semantic
#      colors, marker center has only 3 flip states.
#    - Every generated frame uses both 8 and 9 multiple times, so the local mirror rule is not a
#      one-off accident.
#    - Every sample contains at least one legend whose global flip is non-zero and visibly changes
#      the post-rotation composite.
# c) Implicit-assumption audit against the original solver:
#    - `frame_mask = (frame != 0) & (frame != 6)` is no longer sufficient; frame colors must be
#      preserved because 8 and 9 carry different local copy rules.
#    - The old single integer rotation decode is insufficient; the marker patch now yields
#      `(rot_k, flip_mode)`.
#    - The old pipeline could tolerate symmetric sprites. Here symmetry can erase either the local
#      mirror effect or the global flip effect, so the generator rejects such cases quantitatively.
#    - Combining D3 and D5 introduces a possible collapse if global transform is applied before
#      local tiling. The solver fixes the order to "local semantic tiling first, global transform
#      second" so the two mirror layers remain independent.
# d) Placement legality:
#    - The base composite size is `(frame_h * N, frame_w * N)` before global transform; only
#      90/270-degree rotation swaps the bbox, and flips do not change size.
#    - Anchor fitting and 2-legend non-overlap are checked on the fully transformed composite.
# e) Visual observability:
#    - Frame semantics are visible in a dedicated structural region, not via hidden color ranking.
#    - Marker-center flip semantics are also structurally pinned to the 3x3 patch, so humans do not
#      have to infer a global color remapping.
#    - The generator rejects cases where either local semantics or global flip produce zero pixel
#      change relative to a simpler baseline.
# f) Visual cleanliness:
#    - No extra objects are added to the canvas; all new information lives inside existing legend
#      subpanels, preserving the original scene decomposition.
#    - Colors 8/9 are reused only in strongly typed regions: frame cells or marker center.
# g) Inference / anti-degeneracy metric:
#    - For each legend, `local_effect_pixels` compares the semantic-frame composite against the
#      "all frame cells are normal" baseline.
#    - For each flipped legend, `global_effect_pixels` compares the final transformed composite
#      against the same rotation without the extra flip.
#    - Generation only accepts legends with `local_effect_pixels > 0`, and samples must contain at
#      least one legend with `global_effect_pixels > 0`.

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, neighbors8
from scipy.ndimage import label as _scipy_label


FRAME_NORMAL_COLOR = 8
FRAME_MIRROR_COLOR = 9
FRAME_RULE_COLORS = (FRAME_NORMAL_COLOR, FRAME_MIRROR_COLOR)

FLIP_NONE = 0
FLIP_H = 1
FLIP_V = 2
FLIP_COLOR_TO_MODE = {7: FLIP_NONE, 8: FLIP_H, 9: FLIP_V}
FLIP_MODE_TO_COLOR = {mode: color for color, mode in FLIP_COLOR_TO_MODE.items()}

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
TWO_LEGEND_UNIT_SIZE_RANGE = (2, 4)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)  # 4..8
FRAME_W_RANGE = (3, 6)  # 3..5
FRAME_ACTIVE_RANGE = (9, 17)
FRAME_FILL_RANGE = (0.50, 0.65)
FRAME_GEN_ATTEMPTS = 48
LEGEND_DESIGN_RETRIES = 36
LEGEND_LAYOUT_RETRIES = 18
MIN_MIRROR_CELLS = 3
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
COLOR_POOL = [1, 2, 3, 4, 5, 7]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300


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
    """Identify the left legend separator and an optional right legend separator."""
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
    """List all 5x5 border-6 squares that look like the top of a marker panel."""
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
    """Extract the top-most 4-connected non-{0,6} component and undo the legend color-swap."""
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    vis[rmin, cseed] = True
    st = [(rmin, cseed)]
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
        swapped = spr.copy()
        swapped[spr == a] = -1
        swapped[spr == b] = a
        swapped[swapped == -1] = b
        spr = swapped
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
    return r0, c0, r1, c1


def _position_indices_from_border_square(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    inner = g[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_flip_from_marker(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int]:
    """Read quarter-turn direction from the endpoint and extra flip state from the center color."""
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    sub = g[r0 : r0 + 3, c0 : c0 + 3]
    assert sub.shape == (3, 3), "rotation marker patch out of bounds"
    center_color = int(sub[1, 1])
    assert center_color in FLIP_COLOR_TO_MODE, "invalid flip marker color"
    pts = np.argwhere((sub != 0) & (sub != 6))
    endpoint = [(int(r), int(c)) for r, c in pts if not (int(r) == 1 and int(c) == 1)]
    assert len(endpoint) == 1, "marker must contain center plus exactly one endpoint"
    rr, cc = endpoint[0]
    dr, dc = rr - 1, cc - 1
    if abs(dr) >= abs(dc):
        rot_k = 0 if dr < 0 else 2
    else:
        rot_k = 1 if dc > 0 else 3
    return rot_k, FLIP_COLOR_TO_MODE[center_color]


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """Convert a 0/1/2 anchor choice into the placed object's top-left coordinate."""
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


def _build_composite_from_frame(unit_sprite: np.ndarray, frame_codes: np.ndarray) -> np.ndarray:
    """Expand a semantic frame into a base composite before any global marker transform."""
    unit_mask = (unit_sprite != 0) & (unit_sprite != 6)
    unit_mirror = np.fliplr(unit_sprite)
    mirror_mask = (unit_mirror != 0) & (unit_mirror != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_codes.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            code = int(frame_codes[i, j])
            if code == 0 or code == 6:
                continue
            rr0 = i * uh
            cc0 = j * uw
            if code == FRAME_NORMAL_COLOR:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][unit_mask] = unit_sprite[unit_mask]
            elif code == FRAME_MIRROR_COLOR:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][mirror_mask] = unit_mirror[mirror_mask]
            else:
                raise AssertionError(f"unexpected frame code: {code}")
    return comp


def _apply_global_transform(arr: np.ndarray, rot_k: int, flip_mode: int) -> np.ndarray:
    """Apply the marker's global transform after local tiling has already happened."""
    out = np.rot90(arr, (-rot_k) % 4)
    if flip_mode == FLIP_H:
        out = np.fliplr(out)
    elif flip_mode == FLIP_V:
        out = np.flipud(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode legend panels whose frame colors control local tiling and whose marker patch controls a later global transform.

    Input structure:
    - Full-height color-6 columns still separate the left legend, the output canvas, and an optional right legend.
    - Each legend contains a swapped 2-color unit sprite, a frame template, a 5x5 anchor square, and a 3x3 marker patch below it.
    - Inside the frame, color 8 means "tile the unit normally" and color 9 means "tile a left-right mirror of the unit".
    - Inside the 3x3 marker patch, the endpoint gives the quarter-turn direction and the center color gives the extra global flip state.

    Rule:
    - Read the unit sprite, undo the legend color swap, and read the semantic frame between the sprite and the anchor square.
    - Expand that frame into a base composite by applying the frame's local copy rule at every active cell.
    - Apply the marker's global transform to the whole base composite: rotate by the endpoint direction, then optionally flip by the center color.
    - Place the transformed composite onto the canvas using the anchor encoded by the bordered square's 3x3 inner position.

    Key insight:
    - Mirroring happens at two independent levels: frame colors choose whether each tile is locally mirrored before assembly, while the marker patch can still mirror the entire assembled object afterward.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = canvas.copy()
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

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_active = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_active, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        frame_colors = {int(x) for x in np.unique(frame[frame_active])}
        assert frame_colors <= set(FRAME_RULE_COLORS), f"unexpected frame colors: {frame_colors}"
        assert FRAME_NORMAL_COLOR in frame_colors and FRAME_MIRROR_COLOR in frame_colors, "frame must use both rule colors"

        rot_k, flip_mode = _rotation_flip_from_marker(g, marker_r0, marker_c0)
        base_comp = _build_composite_from_frame(unit_sprite, frame)
        comp = _apply_global_transform(base_comp, rot_k, flip_mode)

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


def _allocate_colors(num_legends: int):
    """Pick unit colors and striped canvas colors while reserving 8/9 for semantics."""
    pool = list(COLOR_POOL)
    random.shuffle(pool)
    unit_color_sets = []
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        unit_color_sets.append((pool.pop(), pool.pop()))
    if len(pool) < 2:
        return None
    bg_colors = (pool.pop(), pool.pop())
    return unit_color_sets, bg_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    out = sprite.copy()
    out[sprite == c1] = -1
    out[sprite == c2] = c1
    out[out == -1] = c2
    return out


def _same_shape(arr: np.ndarray, other: np.ndarray) -> bool:
    return arr.shape == other.shape and np.array_equal(arr, other)


def _make_asymmetric_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray:
    """Generate a filled NxN sprite whose mirror/rotation variants are usually distinguishable."""
    for _ in range(200):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        if _same_shape(sprite, np.fliplr(sprite)):
            continue
        if _same_shape(sprite, np.flipud(sprite)):
            continue
        if _same_shape(sprite, np.rot90(sprite, 1)):
            continue
        if _same_shape(sprite, np.rot90(sprite, 2)):
            continue
        return sprite
    raise RuntimeError("failed to build asymmetric unit sprite")


def _frame_mask_ok(mask: np.ndarray) -> bool:
    active = int(mask.sum())
    if not (FRAME_ACTIVE_RANGE[0] <= active <= FRAME_ACTIVE_RANGE[1]):
        return False
    fill = active / float(mask.size)
    if not (FRAME_FILL_RANGE[0] <= fill <= FRAME_FILL_RANGE[1]):
        return False
    _, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    return n8 == 1


def _sample_frame_mask(frame_h: int, frame_w: int) -> np.ndarray | None:
    """Generate an 8-connected binary frame mask that still matches the original size constraints."""
    for _ in range(FRAME_GEN_ATTEMPTS):
        total = frame_h * frame_w
        lo = max(FRAME_ACTIVE_RANGE[0], int(np.ceil(total * FRAME_FILL_RANGE[0])))
        hi = min(FRAME_ACTIVE_RANGE[1], int(np.floor(total * FRAME_FILL_RANGE[1])))
        if lo > hi:
            return None
        target = random.randint(lo, hi)
        mask = np.zeros((frame_h, frame_w), dtype=bool)
        r0 = random.randrange(frame_h)
        c0 = random.randrange(frame_w)
        mask[r0, c0] = True
        while int(mask.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(mask):
                for rr, cc in neighbors8(int(r), int(c)):
                    if 0 <= rr < frame_h and 0 <= cc < frame_w and not mask[rr, cc]:
                        frontier.add((int(rr), int(cc)))
            if not frontier:
                break
            rr, cc = random.choice(list(frontier))
            mask[rr, cc] = True
        if not _frame_mask_ok(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    return None


def _mirror_assignments(mask: np.ndarray) -> list[np.ndarray]:
    """Return repeated mirror subsets so color-9 support is obvious inside the frame."""
    r_idx, c_idx = np.indices(mask.shape)
    schemes = [
        ((r_idx + c_idx) % 2 == 0) & mask,
        ((r_idx + c_idx) % 2 == 1) & mask,
        (r_idx % 2 == 0) & mask,
        (r_idx % 2 == 1) & mask,
        (c_idx % 2 == 0) & mask,
        (c_idx % 2 == 1) & mask,
    ]
    valid: list[np.ndarray] = []
    active = int(mask.sum())
    for mm in schemes:
        count = int(mm.sum())
        if count < MIN_MIRROR_CELLS or active - count < MIN_MIRROR_CELLS:
            continue
        coords = np.argwhere(mm)
        if len({int(r) for r, _ in coords}) < 2 or len({int(c) for _, c in coords}) < 2:
            continue
        valid.append(mm)
    return valid


def _make_frame_codes(frame_h: int, frame_w: int) -> np.ndarray | None:
    """Generate a 2-color semantic frame with repeated support for both tile rules."""
    for _ in range(FRAME_GEN_ATTEMPTS):
        mask = _sample_frame_mask(frame_h, frame_w)
        if mask is None:
            continue
        choices = _mirror_assignments(mask)
        if not choices:
            continue
        mirror_mask = random.choice(choices)
        codes = np.zeros((frame_h, frame_w), dtype=int)
        codes[mask] = FRAME_NORMAL_COLOR
        codes[mirror_mask] = FRAME_MIRROR_COLOR
        return codes
    return None


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    required_flip_mode: int | None = None,
) -> bool:
    """Fill one legend record with a visible sprite, semantic frame, and global transform."""
    n = leg["N"]
    c1, c2 = unit_colors
    avail = marker_r0 - (n + 1)
    feasible: list[tuple[int, int, int]] = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                out_h, out_w = (frame_h * n, frame_w * n)
                if rot_k % 2 == 1:
                    out_h, out_w = out_w, out_h
                if out_h <= grid_h and out_w <= canvas_w:
                    feasible.append((frame_h, frame_w, rot_k))
    if not feasible:
        return False

    for _ in range(LEGEND_DESIGN_RETRIES):
        sprite = _make_asymmetric_unit_sprite(n, c1, c2)
        legend_sprite = _color_swap(sprite, c1, c2)
        order = feasible[:]
        random.shuffle(order)
        for frame_h, frame_w, rot_k in order:
            frame_codes = _make_frame_codes(frame_h, frame_w)
            if frame_codes is None:
                continue
            base_comp = _build_composite_from_frame(sprite, frame_codes)
            normal_frame = np.where(frame_codes != 0, FRAME_NORMAL_COLOR, 0)
            local_baseline = _build_composite_from_frame(sprite, normal_frame)
            local_effect_pixels = int(np.sum(base_comp != local_baseline))
            if local_effect_pixels <= 0:
                continue

            rotated_only = _apply_global_transform(base_comp, rot_k, FLIP_NONE)
            valid_modes = [FLIP_NONE]
            if not _same_shape(rotated_only, np.fliplr(rotated_only)):
                valid_modes.append(FLIP_H)
            if not _same_shape(rotated_only, np.flipud(rotated_only)):
                valid_modes.append(FLIP_V)
            if len(valid_modes) == 1:
                continue

            transformed_options = []
            for flip_mode in valid_modes:
                transformed = _apply_global_transform(base_comp, rot_k, flip_mode)
                global_effect_pixels = int(np.sum(transformed != rotated_only)) if flip_mode != FLIP_NONE else 0
                if flip_mode != FLIP_NONE and global_effect_pixels <= 0:
                    continue
                transformed_options.append((flip_mode, transformed, global_effect_pixels))
            if not transformed_options:
                continue

            if required_flip_mode is not None:
                transformed_options = [opt for opt in transformed_options if opt[0] == required_flip_mode]
                if not transformed_options:
                    continue

            if required_flip_mode is None:
                weights = [1 if flip_mode == FLIP_NONE else 4 for flip_mode, _, _ in transformed_options]
                flip_mode, transformed_comp, global_effect_pixels = random.choices(transformed_options, weights=weights)[0]
            else:
                flip_mode, transformed_comp, global_effect_pixels = random.choice(transformed_options)
            leg.update(
                sprite=sprite,
                legend_sprite=legend_sprite,
                frame_codes=frame_codes,
                frame_H=frame_h,
                frame_W=frame_w,
                rot_k=rot_k,
                flip_mode=flip_mode,
                base_comp=base_comp,
                transformed_comp=transformed_comp,
                local_effect_pixels=local_effect_pixels,
                global_effect_pixels=global_effect_pixels,
            )
            return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    return leg["transformed_comp"].shape


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions while keeping transformed composites in-bounds and non-overlapping."""
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

    a, b = legends
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
    """Build the unchanged striped canvas background."""
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
    """Stamp each legend's fully transformed composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["transformed_comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    """Draw one legend panel with sprite, semantic frame, anchor square, and rotation+flip patch."""
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
    grid[cx_r, cx_c] = FLIP_MODE_TO_COLOR[leg["flip_mode"]]
    grid[cx_r + dr, cx_c + dc] = 7

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    frame_h, frame_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - frame_w))
    frame_region = grid[fr_start : fr_start + frame_h, mc0 + f_col : mc0 + f_col + frame_w]
    active = leg["frame_codes"] != 0
    frame_region[active] = leg["frame_codes"][active]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    unit_range = TWO_LEGEND_UNIT_SIZE_RANGE if num_legends == 2 else UNIT_SIZE_RANGE
    for _ in range(LEGEND_LAYOUT_RETRIES):
        legends = []
        for _ in range(num_legends):
            n = random.randint(*unit_range)
            legends.append({"N": n, "legend_w": n + 2})

        left_lw = legends[0]["legend_w"]
        left_total = left_lw + 1
        right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
        canvas_w = GRID_W - left_total - right_total
        if canvas_w < MIN_CANVAS_W:
            continue

        colors = _allocate_colors(num_legends)
        if colors is None:
            continue
        unit_color_sets, (bg1, bg2) = colors
        marker_r0 = grid_h - MARKER_PANEL_HEIGHT

        required_modes = [None] * num_legends
        if num_legends == 1:
            required_modes[0] = random.choice([FLIP_H, FLIP_V])
        else:
            none_idx = random.randrange(2)
            required_modes[none_idx] = FLIP_NONE
            required_modes[1 - none_idx] = random.choice([FLIP_H, FLIP_V])

        design_ok = True
        for li in range(num_legends):
            if not _design_legend(
                legends[li],
                unit_color_sets[li],
                marker_r0,
                grid_h,
                canvas_w,
                required_flip_mode=required_modes[li],
            ):
                design_ok = False
                break
        if not design_ok:
            continue

        if not any(int(leg["flip_mode"]) != FLIP_NONE and int(leg["global_effect_pixels"]) > 0 for leg in legends):
            continue
        if not all(int(leg["local_effect_pixels"]) > 0 for leg in legends):
            continue

        if not _assign_anchors(legends, grid_h, canvas_w):
            continue

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
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
