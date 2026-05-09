# Combo variant of a32d8b75 (index 30)
# combo_spec: D15/2+D16/0
# Name: 条件分支+九宫格定位

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis
# a) Concretization and behavior trace
#    The visible legend inventory stays the same: a two-color unit sprite, then a
#    single-color frame mask, then a 5x5 marker square, then a 3x3 clock-hand.
#    The fusion changes two decoding stages at once.
#    Small trace:
#      - Suppose the frame has 6 active cells and the clock-hand points right.
#      - Original task: tile the swapped unit into the unrotated frame, but this was
#        implemented by rotating the frame scaffold first; then decode (0,0) as
#        top-left anchoring, so the composite is flush to the canvas corner.
#      - Fused task: because 6 is even, tile first and rotate the enlarged composite
#        afterward; then decode (0,0) as the center of the top-left ninth, so the
#        same object sits inward from the border.
#      - If instead the frame has 5 active cells, the branch flips: rotate the frame
#        scaffold first and tile afterward, while placement still uses the same
#        top-left ninth center. The internal motif orientation changes, and the final
#        top-left also depends on the composite size because center-alignment is fixed
#        per ninth rather than per edge.
#
# b) Complexity calibration
#    The branch cue has only two states (odd/even active-count parity), and the
#    location cue has only nine states (the 3x3 region centers). That is enough
#    structure to infer from a few ARC examples without introducing continuous
#    coordinates or many special cases. The generator favors two legends, opposite
#    parities, quarter-turn rotations, and size-diverse composites so both factors
#    repeat within a sample set and stay visually distinct.
#
# c) Hidden assumption audit
#    The original shortcut "rotate frame mask before tiling" is no longer always
#    valid, because the even branch requires literal post-tiling rotation. The
#    original anchor decoder is also invalid: marker row/col can no longer be mapped
#    independently to start/middle/end edges and must instead target third-centers on
#    the logical canvas. Because grid9 placement depends on the final composite size,
#    legality checks must use the actual branch-selected composite dimensions rather
#    than only the coarse frame dimensions. Finally, center-ish positions can look too
#    close to the old rule, so generation must actively keep at least one placement
#    different from the legacy anchor semantics.
#
# d) Placement legality
#    The generator is output-first. It builds each branch-selected composite first,
#    enumerates all legal grid9 placements whose full bounding boxes stay inside the
#    canvas, and for two legends rejects overlapping pairs. The solver keeps the
#    non-overlap assertions, so any accepted sample must already satisfy those bounds.
#
# e) Visual observability
#    The frame parity is human-visible because the mask is sparse and countable at the
#    cell level, while the branch effect is amplified by forcing asymmetric unit
#    sprites and quarter-turn rotations only. The grid9 change is human-visible
#    because corner markers no longer pin objects to borders; they aim at the centers
#    of corner thirds instead. The generator rejects legends where either signal would
#    collapse into the old behavior.
#
# f) Visual cleanliness
#    No extra glyphs are added. Legend panels still live outside the canvas and the
#    output still contains at most two composites on a banded background. Bounding-box
#    non-overlap keeps the canvas readable, and opposite parities plus different
#    composite sizes help the two legends feel intentionally different rather than
#    redundant clutter.
#
# g) Change inferability
#    No new color remapping is introduced beyond the original read-time swap of the
#    unit sprite colors. The fused rule is therefore carried entirely by visible shape
#    structure and visible placement, which is appropriate for ARC's low-shot visual
#    inference setting.
#
# 9) New main() assumptions and generator guarantees
#    - main() assumes every frame is a single 8-connected component; the generator
#      enforces that when sampling frame masks.
#    - main() assumes parity is informative; the generator rejects legends whose odd
#      and even execution paths would accidentally yield the same composite.
#    - main() assumes marker positions select legal grid9 placements; the generator
#      enumerates only in-bounds placements under the new center semantics.
#    - main() assumes quarter-turn rotation is requested often enough to expose the
#      branch effect; the generator uses only right/left clock-hands.
#    - main() assumes the grid9 rule is actually exercised; the generator requires at
#      least one chosen placement per sample to differ from the legacy anchor result.
#    - main() assumes two-legend samples do not depend on accidental equality between
#      legends; the generator prefers opposite parities and rejects equal composite
#      sizes when two legends are present.


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    cols6 = [c for c in range(g.shape[1]) if np.all(g[:, c] == 6)]
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
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= g.shape[1] - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
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
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    stack = [(rmin, cseed)]
    vis[rmin, cseed] = True
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
        swapped = spr.copy()
        swapped[spr == a] = -1
        swapped[spr == b] = a
        swapped[swapped == -1] = b
        spr = swapped
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
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
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _position_indices_from_border_square(
    legend: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int] | None:
    inner = legend[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
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


def _tile_unit_into_frame(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    unit_mask = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            r0 = i * uh
            c0 = j * uw
            comp[r0 : r0 + uh, c0 : c0 + uw][unit_mask] = unit_sprite[unit_mask]
    return comp


def _rotate_frame_first(frame_mask0: np.ndarray) -> bool:
    return int(np.count_nonzero(frame_mask0)) % 2 == 1


def _build_composite(unit_sprite: np.ndarray, frame_mask0: np.ndarray, dir_k: int) -> np.ndarray:
    if _rotate_frame_first(frame_mask0):
        return _tile_unit_into_frame(unit_sprite, np.rot90(frame_mask0, (-dir_k) % 4))
    return np.rot90(_tile_unit_into_frame(unit_sprite, frame_mask0), (-dir_k) % 4)


def _other_branch_composite(unit_sprite: np.ndarray, frame_mask0: np.ndarray, dir_k: int) -> np.ndarray:
    if _rotate_frame_first(frame_mask0):
        return np.rot90(_tile_unit_into_frame(unit_sprite, frame_mask0), (-dir_k) % 4)
    return _tile_unit_into_frame(unit_sprite, np.rot90(frame_mask0, (-dir_k) % 4))


def _legacy_anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
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


def _third_center(idx: int, size: int) -> int:
    start = (idx * size) // 3
    end = ((idx + 1) * size) // 3 - 1
    if end < start:
        end = start
    return (start + end) // 2


def _grid9_center_to_top_left(region_idx: int, size: int, place_size: int) -> int:
    return _third_center(region_idx, size) - (place_size - 1) // 2


def _composite_bbox(top: int, left: int, ch: int, cw: int) -> tuple[int, int, int, int]:
    return top, left, top + ch, left + cw


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _list_valid_positions(grid_h: int, canvas_w: int, comp_h: int, comp_w: int) -> list[dict]:
    out: list[dict] = []
    for pr in range(3):
        top = _grid9_center_to_top_left(pr, grid_h, comp_h)
        legacy_top = _legacy_anchor_to_top_left(pr, grid_h, comp_h)
        if top < 0 or top + comp_h > grid_h:
            continue
        for pc in range(3):
            left = _grid9_center_to_top_left(pc, canvas_w, comp_w)
            legacy_left = _legacy_anchor_to_top_left(pc, canvas_w, comp_w)
            if left < 0 or left + comp_w > canvas_w:
                continue
            out.append(
                {
                    "pr": pr,
                    "pc": pc,
                    "top": top,
                    "left": left,
                    "differs_from_legacy": (top != legacy_top) or (left != legacy_left),
                }
            )
    return out


def _edge_score(pos: dict) -> int:
    return int(pos["pr"] != 1) + int(pos["pc"] != 1)


def _position_pool_score(valid_positions: list[dict]) -> tuple[int, int, int, int, int]:
    diff_positions = [p for p in valid_positions if p["differs_from_legacy"]]
    diff_rows = {p["pr"] for p in diff_positions}
    diff_cols = {p["pc"] for p in diff_positions}
    return (
        len(diff_cols),
        len(diff_rows),
        len(diff_positions),
        max((_edge_score(p) for p in diff_positions), default=-1),
        len(valid_positions),
    )


def _branch_difference_score(comp: np.ndarray, alt: np.ndarray) -> int:
    h = max(comp.shape[0], alt.shape[0])
    w = max(comp.shape[1], alt.shape[1])
    a = np.zeros((h, w), dtype=int)
    b = np.zeros((h, w), dtype=int)
    a[: comp.shape[0], : comp.shape[1]] = comp
    b[: alt.shape[0], : alt.shape[1]] = alt
    return int(np.count_nonzero(a != b))


def _legend_records_from_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else g.shape[1]
    canvas = g[:, canvas_c0:canvas_c1]
    records: list[dict] = []
    for marker_r0, marker_c0 in _all_marker_panel_squares(g):
        if canvas_c0 <= marker_c0 < canvas_c1 or marker_c0 + 5 > g.shape[1]:
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
        frame_mask0 = (panel[fr0 : fr1 + 1, fc0 : fc1 + 1] != 0) & (panel[fr0 : fr1 + 1, fc0 : fc1 + 1] != 6)
        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = _build_composite(unit_sprite, frame_mask0, dir_k)
        alt = _other_branch_composite(unit_sprite, frame_mask0, dir_k)
        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        records.append(
            {
                "active_count": int(np.count_nonzero(frame_mask0)),
                "rotate_frame_first": _rotate_frame_first(frame_mask0),
                "dir_k": dir_k,
                "branch_visible": not np.array_equal(comp, alt),
                "branch_difference": _branch_difference_score(comp, alt),
                "comp_shape": tuple(int(x) for x in comp.shape),
                "grid9_top": _grid9_center_to_top_left(pr, canvas.shape[0], comp.shape[0]),
                "grid9_left": _grid9_center_to_top_left(pc, canvas.shape[1], comp.shape[1]),
                "legacy_top": _legacy_anchor_to_top_left(pr, canvas.shape[0], comp.shape[0]),
                "legacy_left": _legacy_anchor_to_top_left(pc, canvas.shape[1], comp.shape[1]),
                "placement_differs": (
                    _grid9_center_to_top_left(pr, canvas.shape[0], comp.shape[0])
                    != _legacy_anchor_to_top_left(pr, canvas.shape[0], comp.shape[0])
                )
                or (
                    _grid9_center_to_top_left(pc, canvas.shape[1], comp.shape[1])
                    != _legacy_anchor_to_top_left(pc, canvas.shape[1], comp.shape[1])
                ),
            }
        )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct enlarged sprites from legend panels and place them on the canvas, with both execution order and placement semantics changed.

    Input structure:
    - Full-height color-6 separator columns split the grid into a left legend, the output canvas, and sometimes a right legend.
    - Every legend panel outside the canvas still contains four stacked cues: a two-color unit sprite, a single-color frame mask, a 5x5 marker square, and a 3x3 clock-hand below it.
    - The frame mask's active-cell parity is meaningful, and the 3x3 marker names one of nine canvas regions rather than edge anchors.

    Rule:
    - Read the unit sprite, swap its two colors as before, read the frame mask, and read the quarter-turn direction from the clock-hand.
    - If the frame has an odd number of active cells, rotate the frame scaffold first and then tile the unrotated unit into the rotated scaffold.
    - If the frame has an even number of active cells, tile the unit into the original scaffold first and rotate the full enlarged composite afterward.
    - Decode the marker square as an absolute 3x3 region choice on the canvas, align the composite's discrete center to that region center, and paint the composite's non-background pixels there.

    Key insight:
    - Two independent-looking legend cues interact: the frame parity decides what gets rotated, while the marker no longer means start/middle/end anchoring and instead fixes the final composite to the center of a chosen ninth, so the same marker lands differently for different composite sizes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape[1] >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else g.shape[1]
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
        if canvas_c0 <= marker_c0 < canvas_c1 or marker_c0 + 5 > g.shape[1]:
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
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = _build_composite(unit_sprite, frame_mask0, dir_k)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _grid9_center_to_top_left(pr, eff_h, comp.shape[0])
        left = _grid9_center_to_top_left(pc, eff_w, comp.shape[1])

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


GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 12]
UNIT_SIZE_RANGE = (2, 4)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (2, 6)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
FRAME_ACTIVE_RANGE = (4, 14)
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ROTATION_CHOICES = [1, 3]
GEN_RETRIES = 600


def _allocate_colors(num_legends: int):
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
    all_colors = set(USABLE_COLORS)
    for uc in unit_color_sets:
        frame_colors.append(random.choice(list(all_colors - set(uc))))
    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    a, b = sorted((c1, c2))
    out = sprite.copy()
    out[sprite == a] = b
    out[sprite == b] = a
    return out


def _make_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray | None:
    for _ in range(160):
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
    lo, hi = FRAME_ACTIVE_RANGE
    for _ in range(36):
        mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        count = int(np.count_nonzero(mask))
        if count < lo or count > hi or count % 2 != desired_parity:
            continue
        _, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
        if n8 != 1:
            continue
        return mask
    return None


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    desired_parity: int,
) -> bool:
    n = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _make_unit_sprite(n, uc1, uc2)
    if sprite is None:
        return False
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    shape_candidates: list[tuple[int, int, int, list[dict]]] = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            comp_h, comp_w = frame_w * n, frame_h * n
            valid_positions = _list_valid_positions(grid_h, canvas_w, comp_h, comp_w)
            if not valid_positions or not any(pos["differs_from_legacy"] for pos in valid_positions):
                continue
            for rot_k in ROTATION_CHOICES:
                shape_candidates.append((frame_h, frame_w, rot_k, valid_positions))
    if not shape_candidates:
        return False
    random.shuffle(shape_candidates)

    candidates: list[dict] = []
    budget = min(18, len(shape_candidates))
    for frame_h, frame_w, rot_k, valid_positions in shape_candidates[:budget]:
        for _ in range(2):
            frame_mask = _make_frame_mask(frame_h, frame_w, desired_parity)
            if frame_mask is None:
                continue
            comp = _build_composite(sprite, frame_mask, rot_k)
            alt = _other_branch_composite(sprite, frame_mask, rot_k)
            if np.array_equal(comp, alt):
                continue
            candidates.append(
                {
                    "frame_mask": frame_mask,
                    "frame_H": frame_h,
                    "frame_W": frame_w,
                    "rot_k": rot_k,
                    "comp": comp,
                    "valid_positions": valid_positions,
                    "branch_difference": _branch_difference_score(comp, alt),
                }
            )
            if len(candidates) >= 8:
                break
        if len(candidates) >= 8:
            break
    if not candidates:
        return False

    best_pos_score = max(_position_pool_score(cand["valid_positions"]) for cand in candidates)
    pos_best = [cand for cand in candidates if _position_pool_score(cand["valid_positions"]) == best_pos_score]
    best_branch = max(cand["branch_difference"] for cand in pos_best)
    choice = random.choice([cand for cand in pos_best if cand["branch_difference"] == best_branch])
    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=choice["frame_mask"],
        frame_H=choice["frame_H"],
        frame_W=choice["frame_W"],
        frame_color=frame_color,
        rot_k=choice["rot_k"],
        desired_parity=desired_parity,
        active_count=int(np.count_nonzero(choice["frame_mask"])),
        comp=choice["comp"],
        valid_positions=choice["valid_positions"],
        branch_difference=choice["branch_difference"],
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    return int(leg["comp"].shape[0]), int(leg["comp"].shape[1])


def _assign_positions(legends: list[dict]) -> bool:
    if len(legends) == 1:
        options = [p for p in legends[0]["valid_positions"] if p["differs_from_legacy"]]
        if not options:
            return False
        choice = random.choice(options)
        legends[0].update(pr=choice["pr"], pc=choice["pc"], comp_top=choice["top"], comp_left=choice["left"])
        return True

    a, b = legends
    a_h, a_w = _comp_size(a)
    b_h, b_w = _comp_size(b)
    valid_pairs: list[tuple[dict, dict]] = []
    for pa in a["valid_positions"]:
        box_a = _composite_bbox(pa["top"], pa["left"], a_h, a_w)
        for pb in b["valid_positions"]:
            box_b = _composite_bbox(pb["top"], pb["left"], b_h, b_w)
            if _boxes_overlap(box_a, box_b):
                continue
            if not (pa["differs_from_legacy"] or pb["differs_from_legacy"]):
                continue
            valid_pairs.append((pa, pb))
    if not valid_pairs:
        return False

    both_diff = [pair for pair in valid_pairs if pair[0]["differs_from_legacy"] and pair[1]["differs_from_legacy"]]
    pool = both_diff if both_diff else valid_pairs
    pa, pb = random.choice(pool)
    a.update(pr=pa["pr"], pc=pa["pc"], comp_top=pa["top"], comp_left=pa["left"])
    b.update(pr=pb["pr"], pc=pb["pc"], comp_top=pb["top"], comp_left=pb["left"])
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
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
    output = canvas.copy()
    for leg in legends:
        comp = leg["comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
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
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - leg["frame_H"]))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - leg["frame_W"]))
    frame_region = grid[fr_start : fr_start + leg["frame_H"], mc0 + f_col : mc0 + f_col + leg["frame_W"]]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        n = random.randint(*UNIT_SIZE_RANGE)
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
    desired_parities = [random.randint(0, 1)] if num_legends == 1 else [0, 1]
    random.shuffle(desired_parities)

    for i in range(num_legends):
        if not _design_legend(
            legends[i],
            unit_color_sets[i],
            frame_colors[i],
            marker_r0,
            grid_h,
            canvas_w,
            desired_parities[i],
        ):
            return None

    if num_legends == 2:
        if {leg["desired_parity"] for leg in legends} != {0, 1}:
            return None
        if _comp_size(legends[0]) == _comp_size(legends[1]):
            return None

    if not _assign_positions(legends):
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
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
