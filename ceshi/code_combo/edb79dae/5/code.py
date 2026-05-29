# Combo variant of edb79dae (index 5)
# combo_spec: D4/0+D5/0
# Name: 空洞嵌入+旋转放置

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   preserved backbone:
#   - Keep the original two-layer story: a gray framed panel is the destination region,
#     and outside the frame sit one source sprite per source color plus one explicit
#     two-color legend per source color.
#   - The output is still only the cropped frame region, and the legend still controls
#     the final recolor.
#   fused change concretization:
#   - D4 removes source-colored destination markers. Inside the frame we now show only
#     rectangular bg cavities carved out of a gray tray.
#   - D5 makes placement orientation depend on target position. Here that interaction is
#     strengthened: the cavity size is the bbox of the correctly rotated sprite, not the
#     bbox of the unrotated source sprite.
#   - Example trace:
#     source sprite A outside the frame has base bbox 2x4.
#     If A is placed in the upper-left quadrant, the cavity is 2x4 and the solver pastes
#     A unrotated.
#     If the same A is placed in the upper-right quadrant, the cavity is 4x2, so the
#     solver must first infer "odd-quadrant means swap bbox axes", then rotate A by 90
#     degrees so it fits the 4x2 hole exactly, then recolor it via the legend.
#   behavior difference from the original:
#   - Original task: cavity location directly exposed source color; placement never
#     changed sprite orientation.
#   - This combo: the frame reveals neither source color nor original bbox orientation.
#     The solver must combine two visible constraints:
#       1. the hole size tells which source bbox could fit after undoing the quadrant
#          rotation;
#       2. the hole position tells which rotation to apply before stamping.
#   complexity calibration:
#   - To keep the puzzle observable instead of noisy, the generator uses a small pool of
#     non-square source bboxes with unique unordered size signatures. This avoids
#     ambiguities like simultaneously having 2x4 and 4x2 as different sources.
#   - Every sample covers all four quadrants, and at least one source color repeats
#     across even-vs-odd quadrant parity. That makes the width/height swap visible inside
#     a single sample instead of hiding the rotation rule behind one-off placements.
#   hidden assumptions reviewed against the original solver:
#   - `placeholder_colors = unique(inner)` is invalid because the destination is now bg
#     cavities inside a gray tray.
#   - Matching "placeholder color -> sprite color" is gone; matching becomes
#     `(cavity bbox, quadrant) -> source sprite`.
#   - Clearing the whole interior to bg would destroy the tray, so the new solver must
#     preserve the gray panel and only fill cavity cells.
#   - Rotation is no longer a cosmetic post-step: for odd quadrants it changes bbox
#     dimensions, so the generator must construct cavities from rotated bbox sizes.
#   legality and cleanliness:
#   - Cavities are separated by at least one tray cell, never touch the frame border,
#     and empty slots stay solid tray cells, so bg holes do not merge.
#   - Outside sprites and legends still use non-overlapping placement with reserve +
#     padding, so component parsing remains stable.
#   visual observability and color inference:
#   - Quadrant is visible from geometry only; legend-based recolor stays explicit and is
#     not inferred from hidden color ordering.
#   - Because at least one repeated source appears in both bbox parities, a human can see
#     that the same outside shape fills a wide hole in one place and a tall hole in
#     another place.
#   generator guarantees for the new main:
#   - exactly one frame is present;
#   - all inner bg components are rectangular cavities;
#   - source sprites are connected, touch all four sides of their base bbox, and have
#     four distinct 90-degree rotations;
#   - all four quadrant rotations occur in each sample;
#   - at least one source color is used in both even and odd quadrant parity;
#   - source bbox signatures are unique even after considering 90-degree swapping.


FRAME_COL = 5
GRID_ROWS = 2
GRID_COLS_RANGE = (3, 5)
TOTAL_CAVITIES_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 3)
SIZE_POOL = [(2, 3), (2, 4), (3, 4)]
SPRITE_FILL_MIN = 0.45
SPRITE_FILL_MAX = 0.8
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 400
LAYOUT_RETRIES = 160
MASK_RETRIES = 200


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _rotation_k_for_bbox(
    bbox: tuple[int, int, int, int], frame_h: int, frame_w: int
) -> int:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    mid_r = (frame_h - 1) / 2.0
    mid_c = (frame_w - 1) / 2.0
    assert center_r != mid_r and center_c != mid_c, "cavity center lies on a frame midline"
    if center_r < mid_r and center_c < mid_c:
        return 0
    if center_r < mid_r and center_c > mid_c:
        return 1
    if center_r > mid_r and center_c > mid_c:
        return 2
    return 3


def _rectangular_cavities(crop: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """Return all bg cavities strictly inside the frame; each must be a full rectangle."""
    h, w = crop.shape
    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    cavities: list[tuple[int, int, int, int]] = []
    for pts in components_mask((crop == bg) & inner_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "cavity must stay away from the frame border"
        sub = crop[r0 : r1 + 1, c0 : c1 + 1]
        assert int(len(pts)) == int(sub.size), "cavity must be a solid rectangle"
        assert np.all(sub == bg), "cavity bbox must contain only background"
        cavities.append((int(r0), int(c0), int(r1), int(c1)))

    assert cavities, "no inner cavities found"
    return sorted(cavities)


def _rot_signature(mask: np.ndarray) -> tuple[tuple[tuple[int, int], bytes], ...]:
    return tuple((np.rot90(mask, k).shape, np.rot90(mask, k).tobytes()) for k in range(4))


def _has_four_distinct_rotations(mask: np.ndarray) -> bool:
    return len(set(_rot_signature(mask))) == 4


def _carve_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        choices = []
        if r != tr:
            choices.append((r + (1 if tr > r else -1), c))
        if c != tc:
            choices.append((r, c + (1 if tc > c else -1)))
        r, c = random.choice(choices)
        mask[r, c] = True
    return r, c


def _sprite_mask(h: int, w: int) -> np.ndarray:
    """Generate a connected h x w mask with exact bbox and four distinct rotations."""
    area = h * w
    lo = max(4, int(np.ceil(SPRITE_FILL_MIN * area)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * area)), area - 1)
    hi = max(lo, hi)
    for _ in range(MASK_RETRIES):
        target = random.randint(lo, hi)
        mask = np.zeros((h, w), dtype=bool)
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(anchors)
        cur = anchors[0]
        mask[cur] = True
        for nxt in anchors[1:]:
            cur = _carve_path(mask, cur, nxt)

        while int(mask.sum()) < target:
            frontier = set()
            for r, c in zip(*np.where(mask)):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            assert frontier
            rr, cc = random.choice(sorted(frontier))
            mask[rr, cc] = True

        if _has_four_distinct_rotations(mask):
            return mask
    raise RuntimeError(f"failed to build an asymmetric sprite mask for {(h, w)}")


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _slot_rotation(slot: tuple[int, int], cols: int) -> int:
    rr, cc = slot
    left = cc < (cols / 2.0)
    if rr == 0 and left:
        return 0
    if rr == 0 and not left:
        return 1
    if rr == 1 and not left:
        return 2
    return 3


def _rotated_dims(dims: tuple[int, int], krot: int) -> tuple[int, int]:
    h, w = dims
    return (h, w) if krot % 2 == 0 else (w, h)


def _base_dims_from_cavity_dims(dims: tuple[int, int], krot: int) -> tuple[int, int]:
    h, w = dims
    return (h, w) if krot % 2 == 0 else (w, h)


def _extract_task_parts(
    g: np.ndarray,
) -> tuple[
    int,
    int,
    np.ndarray,
    list[tuple[int, int, int, int]],
    dict[int, int],
    dict[int, np.ndarray],
    dict[tuple[int, int], int],
]:
    bg = get_bg(g)
    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"
    cavities = _rectangular_cavities(crop, bg)

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": [(int(r), int(c)) for r, c in pts],
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(pts)),
                }
            )
            cid += 1

    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            if ci["col"] not in sub_colors or cj["col"] not in sub_colors:
                continue
            if ci["area"] == cj["area"]:
                continue

            if ci["area"] > cj["area"]:
                src, tgt = int(ci["col"]), int(cj["col"])
            else:
                src, tgt = int(cj["col"]), int(ci["col"])

            assert src not in color_map, f"duplicate legend mapping for source color {src}"
            color_map[src] = tgt
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))

    assert color_map, "no legend mappings found"

    possible_base_dims = {
        _base_dims_from_cavity_dims((r1 - r0 + 1, c1 - c0 + 1), _rotation_k_for_bbox((r0, c0, r1, c1), h, w))
        for r0, c0, r1, c1 in cavities
    }

    sprite_masks: dict[int, np.ndarray] = {}
    base_dims_to_src: dict[tuple[int, int], int] = {}
    for src_col in sorted(color_map):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1, c["bbox"][3] - c["bbox"][1] + 1) in possible_base_dims
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        dims = (sh, sw)
        assert dims not in base_dims_to_src, f"ambiguous duplicated source bbox {dims}"
        base_dims_to_src[dims] = int(src_col)

        mask = np.zeros((sh, sw), dtype=bool)
        for r, c in base["pts"]:
            mask[r - sr0, c - sc0] = True
        sprite_masks[src_col] = mask

    assert set(base_dims_to_src) == possible_base_dims, "cavity-derived source sizes do not match outside sprites"
    return bg, frame_col, crop, cavities, color_map, sprite_masks, base_dims_to_src


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill framed cavities with quadrant-rotated sprites, then recolor them using the outside legend.

    Input structure:
    - A gray rectangular frame encloses a solid gray tray whose only non-gray interior regions are
      rectangular bg cavities.
    - Outside the frame sit one source-colored sprite template per source color and one compact
      two-color legend block per source color.

    Rule:
    - Detect the frame, the inner rectangular cavities, the outside sprite templates, and the
      legend-based source->target recolor mapping.
    - For each cavity, determine its quadrant within the frame: upper-left 0 deg, upper-right 90
      deg, lower-right 180 deg, lower-left 270 deg.
    - Undo that quadrant-dependent rotation at the bbox level to identify which outside sprite fits
      the cavity, rotate that sprite back into the cavity's orientation, and recolor it to the
      legend target color while preserving the gray tray and border.

    Key insight: A cavity no longer names its sprite by color; instead, the hole's size and its
    position must be read together. The correct source sprite is the only one whose base bbox,
    after the quadrant-implied rotation, matches the cavity exactly.
    """
    g = np.asarray(grid, dtype=int)
    bg, _frame_col, crop, cavities, color_map, sprite_masks, base_dims_to_src = _extract_task_parts(g)
    h, w = crop.shape

    out = crop.copy()
    for r0, c0, r1, c1 in cavities:
        cavity_bbox = (r0, c0, r1, c1)
        krot = _rotation_k_for_bbox(cavity_bbox, h, w)
        cavity_dims = (r1 - r0 + 1, c1 - c0 + 1)
        base_dims = _base_dims_from_cavity_dims(cavity_dims, krot)
        assert base_dims in base_dims_to_src, f"no source sprite matches cavity dims {cavity_dims} at rotation {krot}"
        src_col = base_dims_to_src[base_dims]
        mask = sprite_masks[src_col]
        assert mask.shape == base_dims, f"sprite bbox mismatch for color {src_col}: {mask.shape} vs {base_dims}"
        rot_mask = np.rot90(mask, krot)
        assert rot_mask.shape == cavity_dims, f"rotated sprite does not fit cavity {cavity_dims}"
        tgt_col = int(color_map[src_col])
        sprite = np.full(cavity_dims, bg, dtype=int)
        sprite[rot_mask] = tgt_col
        paste(out, sprite, r0, c0, transparent=bg)

    return out


def _choose_structure() -> tuple[int, int, int, int] | None:
    options = []
    for cols in range(*GRID_COLS_RANGE):
        slots = GRID_ROWS * cols
        for total in range(TOTAL_CAVITIES_RANGE[0], min(TOTAL_CAVITIES_RANGE[1], slots) + 1):
            empty = slots - total
            if 0 <= empty <= MAX_EMPTY_SLOTS:
                options.append((GRID_ROWS, cols, empty, total))
    return random.choice(options) if options else None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], int]] | None:
    slot_rot = {(r, c): _slot_rotation((r, c), cols) for r in range(rows) for c in range(cols)}
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [slot for slot in slots if slot not in empty]
        if len({r for r, _ in filled}) != rows:
            continue
        if len({c for _, c in filled}) != cols:
            continue
        if {slot_rot[s] for s in filled} != {0, 1, 2, 3}:
            continue
        valid.append(empty)
    if not valid:
        return None
    empty_slots = random.choice(valid)
    filled_slots = [slot for slot in slots if slot not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots, slot_rot


def _assign_cell_colors(
    filled_slots: list[tuple[int, int]],
    src_cols: list[int],
    slot_rot: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int] | None:
    parity_pairs = [
        (a, b)
        for i, a in enumerate(filled_slots)
        for b in filled_slots[i + 1 :]
        if (slot_rot[a] % 2) != (slot_rot[b] % 2)
    ]
    if not parity_pairs:
        return None

    anchor_a, anchor_b = random.choice(parity_pairs)
    anchor_col = random.choice(src_cols)
    cell_colors = {anchor_a: anchor_col, anchor_b: anchor_col}

    remaining_slots = [s for s in filled_slots if s not in (anchor_a, anchor_b)]
    random.shuffle(remaining_slots)
    other_cols = [c for c in src_cols if c != anchor_col]
    if len(remaining_slots) < len(other_cols):
        return None

    for col, slot in zip(other_cols, remaining_slots):
        cell_colors[slot] = col

    for slot in remaining_slots:
        if slot not in cell_colors:
            cell_colors[slot] = random.choice(src_cols)

    assert set(cell_colors) == set(filled_slots)
    assert set(cell_colors.values()) == set(src_cols)
    anchor_parities = {slot_rot[s] % 2 for s, col in cell_colors.items() if col == anchor_col}
    assert len(anchor_parities) >= 2
    return cell_colors


def _layout_cavities(
    rows: int,
    cols: int,
    cell_colors: dict[tuple[int, int], int],
    size_map: dict[int, tuple[int, int]],
    slot_rot: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], int, int]:
    row_heights = []
    for r in range(rows):
        dims = [_rotated_dims(size_map[cell_colors[(r, c)]], slot_rot[(r, c)])[0] for c in range(cols) if (r, c) in cell_colors]
        assert dims
        row_heights.append(max(dims))

    col_widths = []
    for c in range(cols):
        dims = [_rotated_dims(size_map[cell_colors[(r, c)]], slot_rot[(r, c)])[1] for r in range(rows) if (r, c) in cell_colors]
        assert dims
        col_widths.append(max(dims))

    row_offsets = []
    cur = 0
    for rh in row_heights:
        row_offsets.append(cur)
        cur += rh + 1
    content_h = cur - 1

    col_offsets = []
    cur = 0
    for cw in col_widths:
        col_offsets.append(cur)
        cur += cw + 1
    content_w = cur - 1

    placements: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for slot, src in cell_colors.items():
        rr, cc = slot
        ch, cw = _rotated_dims(size_map[src], slot_rot[slot])
        cell_r0 = row_offsets[rr]
        cell_c0 = col_offsets[cc]
        off_r = cell_r0 + (row_heights[rr] - ch) // 2
        off_c = cell_c0 + (col_widths[cc] - cw) // 2
        placements[slot] = (off_r, off_c, ch, cw)

    return placements, content_h, content_w


def _choose_frame_layout(
    content_h: int,
    content_w: int,
    placements: dict[tuple[int, int], tuple[int, int, int, int]],
    slot_rot: dict[tuple[int, int], int],
) -> tuple[int, int, int, int, dict[tuple[int, int], tuple[int, int, int, int]]] | None:
    min_fh = max(MIN_FRAME_H, content_h + 4)
    min_fw = max(MIN_FRAME_W, content_w + 4)
    max_fh = min(MAX_FRAME_DIM, content_h + 4 + FRAME_MARGIN_SLACK)
    max_fw = min(MAX_FRAME_DIM, content_w + 4 + FRAME_MARGIN_SLACK)
    if min_fh > max_fh or min_fw > max_fw:
        return None

    for _ in range(LAYOUT_RETRIES):
        frame_h = random.randint(min_fh, max_fh)
        frame_w = random.randint(min_fw, max_fw)
        inner_h = frame_h - 2
        inner_w = frame_w - 2
        y_slack = inner_h - content_h
        x_slack = inner_w - content_w
        if y_slack < 2 or x_slack < 2:
            continue
        top_margin = random.randint(1, y_slack - 1)
        left_margin = random.randint(1, x_slack - 1)

        actual_bboxes = {}
        ok = True
        for slot, (rr, cc, ch, cw) in placements.items():
            bbox = (1 + top_margin + rr, 1 + left_margin + cc, 1 + top_margin + rr + ch - 1, 1 + left_margin + cc + cw - 1)
            try:
                if _rotation_k_for_bbox(bbox, frame_h, frame_w) != slot_rot[slot]:
                    ok = False
                    break
            except AssertionError:
                ok = False
                break
            actual_bboxes[slot] = bbox
        if ok:
            return frame_h, frame_w, top_margin, left_margin, actual_bboxes
    return None


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _variant_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    _bg, _frame_col, crop, cavities, _color_map, _sprite_masks, base_dims_to_src = _extract_task_parts(g)
    h, w = crop.shape
    rotation_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    swapped_fit = 0
    src_parities: dict[int, set[int]] = {}
    for r0, c0, r1, c1 in cavities:
        bbox = (r0, c0, r1, c1)
        krot = _rotation_k_for_bbox(bbox, h, w)
        rotation_hist[krot] += 1
        cavity_dims = (r1 - r0 + 1, c1 - c0 + 1)
        base_dims = _base_dims_from_cavity_dims(cavity_dims, krot)
        src_col = base_dims_to_src[base_dims]
        src_parities.setdefault(src_col, set()).add(krot % 2)
        if cavity_dims != base_dims:
            swapped_fit += 1
    return {
        "rotation_hist": rotation_hist,
        "swapped_fit_cavities": swapped_fit,
        "sources_spanning_parities": sum(len(v) >= 2 for v in src_parities.values()),
        "total_cavities": len(cavities),
    }


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    rows, cols, empty_count, total = struct

    empty_pick = _choose_empty_slots(rows, cols, empty_count)
    if empty_pick is None:
        return None
    _empty_slots, filled_slots, slot_rot = empty_pick

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(NUM_COLORS_RANGE[0], min(NUM_COLORS_RANGE[1], total, len(SIZE_POOL)))
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    size_choices = random.sample(SIZE_POOL, k)
    size_map = {src: dims for src, dims in zip(src_cols, size_choices)}
    masks = {src: _sprite_mask(*size_map[src]) for src in src_cols}

    cell_colors = _assign_cell_colors(filled_slots, src_cols, slot_rot)
    if cell_colors is None:
        return None

    placements, content_h, content_w = _layout_cavities(rows, cols, cell_colors, size_map, slot_rot)
    frame_layout = _choose_frame_layout(content_h, content_w, placements, slot_rot)
    if frame_layout is None:
        return None
    frame_h, frame_w, _top_margin, _left_margin, actual_bboxes = frame_layout

    input_frame = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    output_frame = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    for slot, src in cell_colors.items():
        r0, c0, r1, c1 = actual_bboxes[slot]
        cavity_h = r1 - r0 + 1
        cavity_w = c1 - c0 + 1
        input_frame[r0 : r1 + 1, c0 : c1 + 1] = bg
        output_frame[r0 : r1 + 1, c0 : c1 + 1] = bg
        rot_mask = np.rot90(masks[src], slot_rot[slot])
        assert rot_mask.shape == (cavity_h, cavity_w)
        paste(output_frame, _colored_sprite(rot_mask, color_to_tgt[src], bg), r0, c0, transparent=bg)

    patch_bbox_area = sum(h0 * w0 for h0, w0 in size_map.values()) + 12 * k + 8 * k
    possible_dims = [
        (h0, w0)
        for h0 in range(*CANVAS_H_RANGE)
        for w0 in range(*CANVAS_W_RANGE)
        if h0 >= frame_h and w0 >= frame_w and h0 * w0 - frame_h * frame_w >= patch_bbox_area
    ]
    if not possible_dims:
        return None

    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src, tgt in zip(src_cols, tgt_cols):
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, tgt), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    if get_bg(grid) != bg:
        return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
        metrics = _variant_metrics(grid)
        if metrics["swapped_fit_cavities"] <= 0:
            return None
        if metrics["sources_spanning_parities"] <= 0:
            return None
        if set(k0 for k0, v0 in metrics["rotation_hist"].items() if v0 > 0) != {0, 1, 2, 3}:
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae combo input")
