# Combo variant of a32d8b75 (index 28)
# combo_spec: D8/0+D14/1+D18/0
# Name: 随机图例+无分隔+动态标记

import random

import numpy as np
from arc_utils import bbox_coords, random_sprite
from scipy.ndimage import label as _scipy_label


# Design analysis for the fused combo variant
# -------------------------------------------
# a) Concrete behavior trace:
#    - Original task:
#      the solver relied on three fixed facts at once: full-height color-6 separators,
#      legend roles in a fixed vertical order, and a marker panel fixed to the bottom.
#    - Fused variant:
#      separators are now full-height blank 0 columns, the unit sprite and frame can
#      appear in either vertical order, and the dynamic marker bundle starts immediately
#      below whichever of those two components sits lower while its square/rotation
#      subparts may also swap vertical order inside that bundle.
#    - Example:
#      if the frame occupies rows 2..7 and the unit occupies rows 11..13, then the
#      marker square begins at row 14 instead of at `h-8`; if the order flips and the
#      unit is on top while the frame is lower, then the marker begins below the frame.
#      The old code would fail three ways: it would not find a color-6 separator, it
#      would misread the top-most component as the unit, and it would look for the
#      marker at the bottom. The new code instead cuts panels with blank columns,
#      identifies unit/frame by geometry, then derives the marker row from their bboxes.
# b) Complexity calibration:
#    - The combo keeps the same four semantic symbols and the same canvas action, so
#      it does not introduce new hidden mappings.
#    - To make the D8 change visible without creating ambiguity, the generator flips
#      both the relative order of unit/frame and the relative order of
#      position-square/rotation-strip across samples.
#    - To make the D14 change observable, the only full-height all-0 columns in the
#      whole grid are the legend/canvas separators.
#    - To make the D18 change observable, the marker panel is never bottom-locked and
#      always sits directly under the lower of {unit, frame}.
# c) Hidden assumption review:
#    - Any logic depending on full-height color-6 boundaries is invalid and replaced
#      by `_blank_separator_runs()` and `_split_canvas()`.
#    - Any logic depending on "top component = unit" is invalid and replaced by
#      component-shape matching over all non-{0,6} connected components.
#    - Any logic depending on "frame lies between unit and marker" is invalid because
#      the unit may be lower than the frame. The marker row is therefore computed from
#      `max(unit_bottom, frame_bottom) + 1`.
#    - Reading only the 5 marker columns is no longer sufficient after D14, because
#      the blank separator is not part of the marker border; the solver now reads the
#      entire legend block defined by the blank separators.
# d) Placement legality:
#    - Generation stays output-first: choose legends, build the output canvas, then
#      back-project instructions into legends and validate with `main()`.
#    - Rotated composites must fit fully inside the canvas; with 2 legends their
#      composite bounding boxes must not overlap.
#    - Within a legend, unit and frame row spans are disjoint, the marker panel fits
#      below the lower component, and the right-aligned 5-wide marker ensures the
#      legend contains no accidental full-height blank columns.
# e) Visual observability:
#    - A human sees three recurring cues across examples:
#      blank columns, not magenta bars, separate legend from canvas;
#      the frame is not always under the unit;
#      the marker panel floats with the lower component instead of hugging the bottom.
#    - None of these cues depends on numeric color ordering or any invisible metadata.
# f) Visual cleanliness:
#    - The legend still uses one narrow column block per side, so symbols remain easy
#      to scan. The marker square and rotation strip stay adjacent, and the frame is a
#      single-color sparse mask distinct from the filled two-color unit square.
# g) Color-change inferability:
#    - The only color transformation is still the original two-color unit swap.
#      Frame colors remain single-color, and marker/rotation keep their structural
#      roles. No new color permutation rule is introduced.
# h) New solver assumptions introduced by this implementation:
#    - Each legend contains exactly one filled two-color square candidate and exactly
#      one sparse single-color frame candidate; the generator enforces this.
#    - The marker bundle's top row is exactly one row below the lower of {unit, frame};
#      the generator enforces this and keeps one bottom padding row so the bundle is
#      visibly dynamic rather than accidentally matching the original bottom lock.
#    - Each legend column contains at least one nonzero/6 pixel, preventing internal
#      full-height blank columns from being mistaken for separators.


GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 1]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
COMPONENT_GAP_RANGE = (1, 6)
BOTTOM_PAD_MIN = 1
MARKER_PANEL_HEIGHT = 8
MARKER_PANEL_W = 5
SEP_W = 1
CONTENT_COLORS = [1, 2, 3, 5, 8, 9]
MARKER_COLOR = 4
ROTATION_COLOR = 7
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SQUARE_FIRST_PROB = 0.4
GEN_RETRIES = 300


def _blank_separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all 0."""
    _, w = g.shape
    cols0 = [c for c in range(w) if np.all(g[:, c] == 0)]
    if not cols0:
        return []
    runs: list[tuple[int, int]] = []
    s = e = cols0[0]
    for c in cols0[1:]:
        if c == e + 1:
            e = c
        else:
            runs.append((s, e))
            s = e = c
    runs.append((s, e))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Find the blank separator before the canvas and an optional one after it."""
    _, w = g.shape
    runs = _blank_separator_runs(g)
    assert runs, "must have at least one blank separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _legend_panels(
    g: np.ndarray, left_sep: tuple[int, int], right_sep: tuple[int, int] | None
) -> list[np.ndarray]:
    panels: list[np.ndarray] = []
    if left_sep[0] > 0:
        panels.append(g[:, : left_sep[0]])
    if right_sep is not None and right_sep[1] + 1 < g.shape[1]:
        panels.append(g[:, right_sep[1] + 1 :])
    return panels


def _swap_two_colors(sprite: np.ndarray) -> np.ndarray:
    cols = sorted({int(x) for x in np.unique(sprite) if int(x) not in (0, 6)})
    assert len(cols) == 2, "unit sprite must use exactly two colors"
    a, b = cols
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
    return out


def _component_records(panel: np.ndarray) -> list[dict]:
    mask = (panel != 0) & (panel != 6)
    labels, n = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    comps: list[dict] = []
    for idx in range(1, n + 1):
        pts = np.argwhere(labels == idx)
        r0, c0, r1, c1 = bbox_coords(pts)
        local_mask = labels[r0 : r1 + 1, c0 : c1 + 1] == idx
        colors = sorted({int(x) for x in panel[r0 : r1 + 1, c0 : c1 + 1][local_mask]})
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": local_mask.copy(),
                "colors": colors,
                "area": int(local_mask.sum()),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
            }
        )
    return comps


def _find_unit_component(panel: np.ndarray, comps: list[dict]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    candidates: list[dict] = []
    for comp in comps:
        if comp["h"] != comp["w"]:
            continue
        if not (2 <= comp["h"] <= 5):
            continue
        if comp["area"] != comp["h"] * comp["w"]:
            continue
        if len(comp["colors"]) != 2:
            continue
        candidates.append(comp)
    assert len(candidates) == 1, f"expected exactly one unit sprite, got {len(candidates)}"
    r0, c0, r1, c1 = candidates[0]["bbox"]
    sprite = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
    return _swap_two_colors(sprite), candidates[0]["bbox"]


def _find_frame_component(comps: list[dict]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    candidates: list[dict] = []
    for comp in comps:
        h, w = comp["h"], comp["w"]
        if not (4 <= h <= 8 and 3 <= w <= 5):
            continue
        if len(comp["colors"]) != 1:
            continue
        if comp["area"] >= h * w:
            continue
        density = comp["area"] / float(h * w)
        if not (0.35 <= density <= 0.8):
            continue
        candidates.append(comp)
    assert len(candidates) == 1, f"expected exactly one frame, got {len(candidates)}"
    return candidates[0]["mask"].copy(), candidates[0]["bbox"]


def _marker_anchor_row(unit_bb: tuple[int, int, int, int], frame_bb: tuple[int, int, int, int]) -> int:
    return max(unit_bb[2], frame_bb[2]) + 1


def _is_marker_square(panel: np.ndarray, r0: int, c0: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + 4 >= panel.shape[0] or c0 + 4 >= panel.shape[1]:
        return False
    sq = panel[r0 : r0 + 5, c0 : c0 + 5]
    if not (
        np.all(sq[0, :] == 6)
        and np.all(sq[4, :] == 6)
        and np.all(sq[:, 0] == 6)
        and np.all(sq[:, 4] == 6)
    ):
        return False
    inner = sq[1:4, 1:4]
    return np.argwhere((inner != 0) & (inner != 6)).shape[0] == 1


def _is_rotation_patch(panel: np.ndarray, r0: int, c0: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + 2 >= panel.shape[0] or c0 + 4 >= panel.shape[1]:
        return False
    rot_block = panel[r0 : r0 + 3, c0 : c0 + 5]
    if rot_block.shape != (3, 5):
        return False
    if not (np.all(rot_block[:, 0] == 6) and np.all(rot_block[:, 4] == 6)):
        return False
    rot = rot_block[:, 1:4]
    return np.argwhere((rot != 0) & (rot != 6)).shape[0] == 2


def _find_marker_panel(panel: np.ndarray, marker_r0: int) -> tuple[int, int, int]:
    h, w = panel.shape
    assert marker_r0 + 7 < h, "marker panel out of bounds"
    candidates: list[tuple[int, int, int]] = []
    for c0 in range(w - 4):
        if _is_marker_square(panel, marker_r0, c0) and _is_rotation_patch(panel, marker_r0 + 5, c0):
            candidates.append((marker_r0, int(c0), marker_r0 + 5))
        if _is_rotation_patch(panel, marker_r0, c0) and _is_marker_square(panel, marker_r0 + 3, c0):
            candidates.append((marker_r0 + 3, int(c0), marker_r0))
    assert len(candidates) == 1, f"expected exactly one marker panel at row {marker_r0}, got {len(candidates)}"
    return candidates[0]


def _find_position_indices(panel: np.ndarray, square_r0: int, marker_c0: int) -> tuple[int, int]:
    inner = panel[square_r0 + 1 : square_r0 + 4, marker_c0 + 1 : marker_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    assert pts.shape[0] == 1, "marker square must contain exactly one position pixel"
    return int(pts[0, 0]), int(pts[0, 1])


def _find_rotation_dir(panel: np.ndarray, rot_r0: int, marker_c0: int) -> int:
    inner = panel[rot_r0 : rot_r0 + 3, marker_c0 + 1 : marker_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    assert pts.shape[0] == 2, "rotation patch must have exactly two colored pixels"
    pts_set = {tuple(map(int, p)) for p in pts}
    assert (1, 1) in pts_set, "rotation patch must include its center"
    arm = next(p for p in pts_set if p != (1, 1))
    dr, dc = arm[0] - 1, arm[1] - 1
    assert abs(dr) + abs(dc) == 1, "rotation arm must be cardinal"
    return ROTATION_DIRS.index((dr, dc))


def _has_vertical_layout(n: int, frame_h: int, grid_h: int) -> bool:
    for unit_first in [False, True]:
        for gap in range(COMPONENT_GAP_RANGE[0], COMPONENT_GAP_RANGE[1] + 1):
            pair_h = n + frame_h + gap
            if grid_h - (pair_h + MARKER_PANEL_HEIGHT + BOTTOM_PAD_MIN) >= 0:
                return True
    return False


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
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
    Reconstruct enlarged sprites from legends whose boundaries, vertical order, and marker rows are all non-fixed.

    Input structure:
    - Full-height blank 0 columns, not full-height color-6 bars, separate the left legend from the canvas and optionally the canvas from a right legend.
    - Inside each legend, the filled two-color unit square and the sparse single-color frame can appear in either vertical order, while the 5x5 anchor square and 3x5 rotation strip form a dynamic marker bundle immediately below the lower of those two components and may swap order inside that bundle.

    Rule:
    - Split off each legend with the blank separators, identify the unit sprite and frame by geometry, infer the marker bundle row from their bounding boxes, then read the anchor pixel and rotation direction from whichever internal order appears there.
    - Color-swap the unit sprite, tile it into every active frame cell, rotate the composite, and place it onto the canvas using the 3x3 anchor code.

    Key insight:
    - No single absolute row or separator color tells you where instructions live anymore; the legend must be parsed by visual type, and the marker row is a consequence of the unit/frame arrangement rather than a fixed location.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    out = canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    panels = _legend_panels(g, left_sep, right_sep)
    assert panels, "no legend panels found"

    for panel in panels:
        comps = _component_records(panel)
        unit_sprite, unit_bb = _find_unit_component(panel, comps)
        frame_mask0, frame_bb = _find_frame_component(comps)

        marker_anchor_r0 = _marker_anchor_row(unit_bb, frame_bb)
        square_r0, marker_c0, rot_r0 = _find_marker_panel(panel, marker_anchor_r0)
        pr, pc = _find_position_indices(panel, square_r0, marker_c0)
        dir_k = _find_rotation_dir(panel, rot_r0, marker_c0)

        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_composite(unit_sprite, frame_mask)

        top = _anchor_to_top_left(pr, canvas.shape[0], comp.shape[0])
        left = _anchor_to_top_left(pc, canvas.shape[1], comp.shape[1])

        ch, cw = comp.shape
        for r in range(ch):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(cw):
                cc = left + c
                assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                val = int(comp[r, c])
                if val != 0:
                    assert not composite_written[rr, cc], "composites must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


def _allocate_colors(num_legends: int):
    pool = list(CONTENT_COLORS)
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
    all_set = set(CONTENT_COLORS)
    for uc in unit_color_sets:
        frame_colors.append(random.choice(list(all_set - set(uc))))

    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _sample_frame_mask(frame_h: int, frame_w: int) -> np.ndarray:
    for _ in range(40):
        mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        active = int(mask.sum())
        if active == 0 or active == frame_h * frame_w:
            continue
        density = active / float(frame_h * frame_w)
        _, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
        if 0.35 <= density <= 0.8 and n8 == 1:
            return mask
    raise RuntimeError("failed to sample a suitable frame mask")


def _sample_vertical_layout(
    n: int, frame_h: int, grid_h: int, unit_first: bool
) -> tuple[int, int, int]:
    options: list[tuple[int, int, int]] = []
    for gap in range(COMPONENT_GAP_RANGE[0], COMPONENT_GAP_RANGE[1] + 1):
        pair_h = n + frame_h + gap
        max_top = grid_h - (pair_h + MARKER_PANEL_HEIGHT + BOTTOM_PAD_MIN)
        if max_top < 0:
            continue
        for top_pad in range(max_top + 1):
            if unit_first:
                unit_r0 = top_pad
                frame_r0 = unit_r0 + n + gap
            else:
                frame_r0 = top_pad
                unit_r0 = frame_r0 + frame_h + gap
            marker_r0 = max(unit_r0 + n - 1, frame_r0 + frame_h - 1) + 1
            bottom_pad = grid_h - (marker_r0 + MARKER_PANEL_HEIGHT)
            if bottom_pad < BOTTOM_PAD_MIN:
                continue
            options.append((unit_r0, frame_r0, marker_r0))
    assert options, "no valid vertical layout"
    return random.choice(options)


def _design_legend(leg: dict, unit_colors, frame_color: int, grid_h: int, canvas_w: int) -> bool:
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)

    combos: list[tuple[int, int, int]] = []
    for frame_h in range(*FRAME_H_RANGE):
        for frame_w in range(*FRAME_W_RANGE):
            if frame_w > leg["legend_w"]:
                continue
            if not _has_vertical_layout(n, frame_h, grid_h):
                continue
            for rot_k in range(4):
                rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((frame_h, frame_w, rot_k))
    if not combos:
        return False

    frame_h, frame_w, rot_k = random.choice(combos)
    frame_mask = _sample_frame_mask(frame_h, frame_w)
    rotated_frame = np.rot90(frame_mask, (-rot_k) % 4)
    unit_first = random.choice([False, True])
    unit_r0, frame_r0, marker_r0 = _sample_vertical_layout(n, frame_h, grid_h, unit_first)
    square_first = random.random() < SQUARE_FIRST_PROB

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        frame_r0=frame_r0,
        frame_c0=random.randint(0, leg["legend_w"] - frame_w),
        unit_r0=unit_r0,
        unit_c0=0,
        marker_r0=marker_r0,
        marker_c0=leg["legend_w"] - MARKER_PANEL_W,
        square_first=square_first,
        rot_k=rot_k,
        rotated_frame=rotated_frame,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
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
        comp = _build_composite(leg["sprite"], leg["rotated_frame"])
        t, l = leg["comp_top"], leg["comp_left"]
        ch, cw = comp.shape
        region = output[t : t + ch, l : l + cw]
        region[comp != 0] = comp[comp != 0]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int) -> None:
    n = leg["N"]
    ur0, uc0 = leg["unit_r0"], col_start + leg["unit_c0"]
    fr0, fc0 = leg["frame_r0"], col_start + leg["frame_c0"]
    mr0, mc0 = leg["marker_r0"], col_start + leg["marker_c0"]

    grid[ur0 : ur0 + n, uc0 : uc0 + n] = leg["legend_sprite"]

    frame_region = grid[fr0 : fr0 + leg["frame_H"], fc0 : fc0 + leg["frame_W"]]
    frame_region[leg["frame_mask"]] = leg["frame_color"]

    square_r0 = mr0 if leg["square_first"] else mr0 + 3
    rot_r0 = mr0 + 5 if leg["square_first"] else mr0

    grid[square_r0 : square_r0 + 5, mc0 : mc0 + 5] = 0
    grid[square_r0, mc0 : mc0 + 5] = 6
    grid[square_r0 + 4, mc0 : mc0 + 5] = 6
    grid[square_r0 : square_r0 + 5, mc0] = 6
    grid[square_r0 : square_r0 + 5, mc0 + 4] = 6
    grid[square_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = MARKER_COLOR

    grid[rot_r0 : rot_r0 + 3, mc0 : mc0 + 5] = 0
    grid[rot_r0 : rot_r0 + 3, mc0] = 6
    grid[rot_r0 : rot_r0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rot_r0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = ROTATION_COLOR
    grid[cx_r + dr, cx_c + dc] = ROTATION_COLOR


def _gen_attempt() -> np.ndarray | None:
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]
    if num_legends == 2:
        grid_h = random.randint(max(GRID_H_RANGE[0], 24), GRID_H_RANGE[1])
        unit_hi = min(UNIT_SIZE_RANGE[1], 4)
    else:
        grid_h = random.randint(*GRID_H_RANGE)
        unit_hi = UNIT_SIZE_RANGE[1]

    legends = []
    for _ in range(num_legends):
        n = random.randint(UNIT_SIZE_RANGE[0], unit_hi)
        legends.append({"N": n, "legend_w": max(n + 2, MARKER_PANEL_W)})

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + SEP_W
    right_total = (SEP_W + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    for i in range(num_legends):
        if not _design_legend(legends[i], unit_color_sets[i], frame_colors[i], grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0)
    if num_legends == 2:
        _place_legend(grid, legends[1], col_start=canvas_c0 + canvas_w + SEP_W)

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
