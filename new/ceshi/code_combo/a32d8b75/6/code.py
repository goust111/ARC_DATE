# Combo variant of a32d8b75 (index 6)
# combo_spec: D3/1+D7/2
# Name: 多色框架+旋转数量编码
#
# Design analysis
# ===============
# 1. Concrete behavior and trace
#    - The original task reads four legend ingredients: a 2-color unit sprite, a frame occupancy mask,
#      a 5x5 anchor square, and a 3x3 directional rotation marker.
#    - This combo changes two links in the chain at once:
#      a) frame cells are no longer just "active/inactive"; color 8 means paste the swapped unit as-is,
#         while color 9 means paste a left-right mirrored copy.
#      b) the 3x3 orange marker no longer encodes rotation by where the hand points; it encodes rotation
#         by how many non-center hand pixels exist: 1/2/3/4 -> 0/90/180/270 degrees.
#    - Micro example:
#         unit = [[1, 2],
#                 [1, 1]]
#         frame = [[8, 9],
#                  [9, 8]]
#         rotation patch has center + 3 extra pixels.
#      Original task:
#         - frame colors are flattened to True, so all four cells paste the same unit.
#         - rotation is decoded from hand direction.
#      Combo task:
#         - 8-cells paste [[1,2],[1,1]], 9-cells paste [[2,1],[1,1]].
#         - center+3 extra pixels always means 180 degrees, regardless of patch orientation.
#      Difference:
#         the final composite mixes normal and mirrored subtiles, and the global rotation must be
#         decoded from count rather than from the furthest hand pixel's direction.
#
# 2. Complexity calibration
#    - Only two frame semantics are used, fixed globally as colors 8 and 9. This is enough to make the
#      local tiling rule visible without exploding the hypothesis space.
#    - The rotation code still has four cases, matching the original control granularity, but the marker
#      orientation is randomized so humans cannot rely on the old directional shortcut.
#    - Each generated frame is required to contain repeated 8-cells and repeated 9-cells, and each unit
#      sprite is required to be horizontally asymmetric. That prevents the mirror rule from becoming
#      visually silent.
#
# 3. Implicit assumption audit against the original main()
#    - `frame_mask0 = (frame != 0) & (frame != 6)` was safe originally; now it would destroy the 8-vs-9
#      distinction. The combo solver must preserve the colored frame grid and only use a boolean mask for
#      connectivity checks.
#    - The old rotation decoder assumed one furthest directional pixel mattered. That assumption is gone.
#      The new solver assumes instead that the marker contains the center pixel plus exactly 1..4 extra
#      pixels. The generator must guarantee this.
#    - The old generator could emit horizontally symmetric unit sprites because every frame cell pasted the
#      same unit. That becomes a degenerate case now, so the combo generator excludes it.
#    - The old `_overlay_composites` stamped a single sprite for every active frame cell. The combo output
#      construction must branch per frame color.
#    - The old generator could let the directional interpretation accidentally match the intended rotation.
#      To make D7/2 genuinely active, the combo generator stores a specific rotation patch and rejects it
#      unless the legacy direction-based decoder would disagree with the intended rotation.
#
# 4. Placement legality
#    - Composite size still depends only on rotated frame shape and unit size, so the original anchor-fit
#      and no-overlap logic remains valid.
#    - Generation stays output-first: choose legends, build the true composite, place onto canvas, then
#      render the legends, and finally validate with the combo solver.
#
# 5. Visual observability and cleanliness
#    - The frame's two semantic colors live in a dedicated legend band, so their meaning is localized and
#      repeated enough to infer.
#    - The orange rotation patch stays in the same 3x3 slot below the anchor square, keeping the legend
#      structure stable. Randomizing its orientation while keeping count fixed makes the new code visible.
#    - No invisible ordering tricks are used; every control signal is encoded by visible geometry or color
#      in a fixed legend location.
#
# 6. Validation metrics used after implementation
#    - Mirror-effect metric: compare the true composite against a baseline that treats every active frame
#      cell as color 8; the pixel difference must be positive.
#    - Rotation-code metric: decode the stored 3x3 patch with the legacy directional rule; it must differ
#      from the intended count-based rotation, proving the sample does not collapse back to the original
#      task.

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, neighbors8
from scipy.ndimage import label as _scipy_label


FRAME_NORMAL_COLOR = 8
FRAME_MIRROR_COLOR = 9
FRAME_RULE_COLORS = (FRAME_NORMAL_COLOR, FRAME_MIRROR_COLOR)

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
SINGLE_LEGEND_UNIT_SIZE_RANGE = (3, 5)
DUAL_LEGEND_UNIT_SIZES = (3,)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)  # 4..8
FRAME_W_RANGE = (3, 6)  # 3..5
FRAME_ACTIVE_RANGE = (9, 17)
FRAME_FILL_RANGE = (0.50, 0.65)
FRAME_GEN_ATTEMPTS = 80
MIN_MIRROR_CELLS = 3
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
COLOR_POOL = [1, 2, 3, 4, 5, 7]  # reserve 8/9 for frame semantics
BASE_ARM_PATH = [(1, 2), (0, 2), (0, 1), (0, 0)]
GEN_RETRIES = 300


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    _h, w = g.shape
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
    """List all 5x5 border-6 squares that look like the top of a marker panel."""
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
    """Extract the top-most 4-connected non-{0,6} component and undo the legend color-swap."""
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
    return (r0, c0, r1, c1)


def _position_indices_from_border_square(
    legend: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int] | None:
    """Read the 3x3 anchor code from the bordered square."""
    inner = legend[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_pointer_count(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker lives in the 3x3 patch below the bordered square.

    The center cell must be colored and the number of additional colored hand pixels
    encodes rotation directly:
      1 arm pixel -> 0, 2 -> 1, 3 -> 2, 4 -> 3.
    """
    h, w = legend.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    marker = (sub != 0) & (sub != 6)
    total = int(marker.sum())
    assert total >= 2, "rotation marker must include center plus at least one arm pixel"
    assert marker[1, 1], "rotation marker must include the center pixel"
    arm_pixels = total - 1
    assert 1 <= arm_pixels <= 4, f"rotation marker arm length must be 1..4, got {arm_pixels}"
    return arm_pixels - 1


def _legacy_rotation_from_patch(patch: np.ndarray) -> int:
    """Decode a 3x3 patch using the original direction-based rule for validation."""
    pts = np.argwhere((patch != 0) & (patch != 6))
    assert pts.size > 0, "no marker pixel found"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - 1), int(p[1] - 1)
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
    """Expand a semantic frame into the final composite sprite."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode legend panels and paint enlarged composites onto the central canvas.

    Input structure:
    - Full-height color-6 columns separate the left legend, the output canvas, and an optional right legend.
    - Each legend still contains four sections from top to bottom: a 2-color unit sprite, a frame template, a 5x5 anchor square, and a 3x3 orange rotation patch.
    - The frame template uses two semantic colors: 8 means normal tiling and 9 means left-right mirrored tiling.
    - In the rotation patch, the center pixel is always present; orientation is irrelevant and only the count of extra orange pixels matters.

    Rule:
    - Read the unit sprite, undo the legend color-swap, and extract the colored frame between the sprite and the marker panel.
    - Decode rotation from marker length: 1/2/3/4 non-center hand pixels mean 0/90/180/270 degrees, then rotate the full frame layout by that amount.
    - Expand the rotated frame cell by cell: color 8 pastes the unit as-is, while color 9 pastes its horizontal mirror.
    - Place the finished composite on the canvas using the anchor position encoded by the marker square's inner 3x3 pixel.

    Key insight:
    - Two independent legend encodings must be fused: frame color controls the local copy transform, while marker pixel count controls the global rotation. Solving the task requires preserving both signals instead of collapsing either one to the original rule.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
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

        rot_k = _rotation_from_pointer_count(g, marker_r0, marker_c0)
        rotated_frame = np.rot90(frame, (-rot_k) % 4)
        comp = _build_composite_from_frame(unit_sprite, rotated_frame)

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
    """Pick unit colors and canvas background colors, reserving 8/9 for frame semantics."""
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
    """Return a copy of sprite with its two colors swapped."""
    a, b = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
    return out


def _is_horizontal_asymmetric(sprite: np.ndarray) -> bool:
    return not np.array_equal(sprite, np.fliplr(sprite))


def _make_unit_sprite(n: int, colors: tuple[int, int]) -> np.ndarray:
    """Generate a filled NxN sprite whose horizontal mirror is visually different."""
    c1, c2 = colors
    for _ in range(200):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        if _is_horizontal_asymmetric(sprite):
            return sprite
    raise AssertionError("failed to generate asymmetric unit sprite")


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
    """Generate an 8-connected binary frame mask inside the original size constraints."""
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
        if _frame_mask_ok(mask):
            if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            return mask
    return None


def _mirror_assignments(mask: np.ndarray) -> list[np.ndarray]:
    """Candidate mirror subsets with repeated support across the frame."""
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
    """Generate a two-color semantic frame with both rule colors repeated."""
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


def _make_rotation_patch(rot_k: int) -> tuple[np.ndarray, int]:
    """
    Build a 3x3 patch with a center pixel plus 1..4 arm pixels.

    The patch is randomly rotated/flipped, but generation rejects any draw whose legacy
    direction-based interpretation matches `rot_k`, so this combo cannot collapse to the
    original directional rule on the same sample.
    """
    arm_len = rot_k + 1
    base = np.zeros((3, 3), dtype=int)
    base[1, 1] = 7
    for rr, cc in BASE_ARM_PATH[:arm_len]:
        base[rr, cc] = 7
    for _ in range(40):
        patch = base.copy()
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        patch = np.rot90(patch, random.randint(0, 3))
        legacy_rot = _legacy_rotation_from_patch(patch)
        if legacy_rot != rot_k:
            return patch, legacy_rot
    raise AssertionError("failed to decorrelate count-coded patch from legacy direction rule")


def _design_legend(leg: dict, unit_colors: tuple[int, int], marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    """Fill one legend record with a visible sprite, semantic frame, and count-coded rotation patch."""
    n = leg["N"]
    sprite = _make_unit_sprite(n, unit_colors)
    legend_sprite = _color_swap(sprite, unit_colors[0], unit_colors[1])

    avail = marker_r0 - (n + 1)
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

    random.shuffle(combos)
    for frame_h, frame_w, rot_k in combos:
        frame_codes = _make_frame_codes(frame_h, frame_w)
        if frame_codes is None:
            continue
        rotated_frame = np.rot90(frame_codes, (-rot_k) % 4)
        comp = _build_composite_from_frame(sprite, rotated_frame)
        baseline = _build_composite_from_frame(sprite, np.where(rotated_frame != 0, FRAME_NORMAL_COLOR, 0))
        if np.array_equal(comp, baseline):
            continue
        rotation_patch, legacy_rot_k = _make_rotation_patch(rot_k)
        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame_codes=frame_codes,
            frame_H=frame_h,
            frame_W=frame_w,
            rot_k=rot_k,
            rotated_frame=rotated_frame,
            rotation_patch=rotation_patch,
            legacy_rot_k=legacy_rot_k,
            mirror_cells=int(np.sum(rotated_frame == FRAME_MIRROR_COLOR)),
            mirror_effect_pixels=int(np.sum(comp != baseline)),
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions while keeping final composites in-bounds and non-overlapping."""
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
    """Stamp each legend's composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite_from_frame(leg["sprite"], leg["rotated_frame"])
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    """Draw one legend panel with sprite, semantic frame, anchor square, and count-coded rotation patch."""
    n = leg["N"]
    grid[:, sep_col] = 6

    # Keep the marker square fully inside the legend strip on both sides.
    # The previous right-legend formula could start the square on the separator
    # column itself, destroying the full-height color-6 split that main() relies on.
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
    grid[rr0 : rr0 + 3, mc0 + 1 : mc0 + 4] = leg["rotation_patch"]

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

    legends = []
    for _ in range(num_legends):
        if num_legends == 1:
            n = random.randint(*SINGLE_LEGEND_UNIT_SIZE_RANGE)
        else:
            n = random.choice(DUAL_LEGEND_UNIT_SIZES)
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
    unit_color_sets, (bg1, bg2) = colors
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT

    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], marker_r0, grid_h, canvas_w):
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


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
