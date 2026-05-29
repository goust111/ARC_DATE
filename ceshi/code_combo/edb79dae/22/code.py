# Combo variant of edb79dae (index 22)
# combo_spec: D1/0+D4/0+D5/0
# Name: 形状匹配+空洞嵌入+旋转放置

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# concreteized rule:
#   - Keep the original object families: one gray rectangular frame/tray, one source-colored
#     sprite per source type outside the frame, and one 3x4 two-color legend per source type.
#   - Replace the old interior color placeholders with paired "marker + cavity" slots:
#     inside the gray tray, every occupied slot contains
#       1. a small same-bbox shape marker in the slot's upper-left sub-square,
#       2. a square bg cavity in the slot's lower-right sub-square.
#   - The marker is a copy of the sprite's unrotated shape in a non-semantic marker color.
#     The cavity is where the sprite must be embedded.
#   - The output removes the marker, rotates the matched sprite according to the cavity's
#     frame quadrant, stamps the rotated sprite into the cavity, and recolors it via the legend.
# behavior trace:
#   - Original local step: a solid red 3x3 block inside the frame directly identifies the red
#     outside sprite, which is pasted unrotated at that block and then recolored by the red legend.
#   - Fused local step:
#       * The hole itself is only an empty 3x3 cavity, so it does not reveal the source by color.
#       * A nearby 3x3 marker shape, tucked exactly one cavity-width up-left from the hole,
#         matches one outside sprite by geometry only.
#       * After the solver identifies that source sprite, the hole position decides a quadrant
#         rotation, so the filled hole may no longer look like the guide marker that selected it.
#   - Difference: the solver must chain "read guide shape -> find source template -> rotate by
#     location -> embed into the hole -> recolor by legend", rather than reading color directly.
# complexity calibration:
#   - The tray uses six occupied slots, so the marker/hole pattern repeats enough times to make
#     the pairing relation and the quadrant rule visible from 2-4 examples.
#   - Marker colors come from a tiny palette disjoint from source colors, and at least one marker
#     color is reused across different source shapes. This blocks any fallback to color matching.
#   - Source masks are 3x3, connected, touch all four bbox sides, and are unique up to rotation.
#     This keeps shape matching clean while still making quadrant rotation visually obvious.
# implicit assumption audit against the original solver:
#   - Original assumption: non-bg cells inside the frame are the destination objects themselves.
#     Broken. In this combo, non-bg interior cells are only guide markers; the true destinations
#     are bg cavities.
#   - Original assumption: placeholder color determines source color. Broken. Marker colors are
#     explicitly decoupled from source colors, so legend decoding cannot consult the frame interior.
#   - Original assumption: destination bbox equals the visible placeholder component bbox. Broken.
#     The destination bbox now comes from the rectangular cavity, while the matching clue comes
#     from a separate nearby component.
#   - Original assumption: a matched sprite is pasted without orientation change. Broken. The same
#     matched sprite can produce four different outputs depending on the cavity quadrant.
#   - New combo assumption: each cavity has exactly one marker whose bbox is shifted by (-n, -n),
#     where n is the cavity size. The generator enforces this fixed local geometry.
#   - New combo assumption: every outside sprite shape is unique even after considering its four
#     rotations. The generator enforces this so shape matching stays unambiguous.
# placement legality:
#   - The frame interior is a gray tray, so cavities are isolated bg rectangles and never merge.
#   - Each slot is a 6x6 cell made from a 3x3 marker corner and a 3x3 cavity corner. Adjacent
#     slots can touch only through gray tray cells, so markers and cavities remain separable.
#   - Rotation is safe because every sprite bbox is square, so 90-degree turns still fit the hole.
#   - Outside sprites and legends are still placed with reserve + padding, so the frame, legends,
#     and templates never overlap.
# visual observability:
#   - A human sees three distinct signals at once:
#       1. the frame interior is now a tray with holes rather than colored blocks;
#       2. each hole has a nearby guide shape, making "shape chooses source" visible;
#       3. the same source shape can reappear in different cavity quadrants with different
#          orientations, making the rotation rule legible.
#   - The guide marker is always in the hole's northwest corner, which makes pairing obvious
#     without any hidden ordering or coordinate sorting trick.
# cleanliness:
#   - The frame interior contains only two object families: sparse guide markers and square holes.
#     Markers disappear in the output, so the result stays visually clean instead of stacked.
# color inference:
#   - Final recoloring is still justified only by the explicit outside legends.
#   - Because marker colors are disjoint from source colors and reused across different shapes,
#     interior color changes do not carry hidden semantic meaning.
# solver-generator contract:
#   - main() assumes: one frame, only rectangular bg cavities inside it, one same-size marker
#     exactly up-left of each cavity, unique sprite shapes up to rotation, square sprites, and
#     one legend per source color.
#   - generate_input() constructs samples that satisfy each assumption and rejects any sample where
#     the background would stop being the globally dominant color.


FRAME_COL = 5
SPRITE_N = 3
GRID_ROWS = 2
GRID_COLS = 3
CELL_SIZE = 2 * SPRITE_N
NUM_TYPES_RANGE = (2, 4)
SPRITE_FILL_MIN = 0.35
SPRITE_FILL_MAX = 0.75
FRAME_H_RANGE = (14, 18)
FRAME_W_RANGE = (20, 24)
CANVAS_H_RANGE = (24, 31)
CANVAS_W_RANGE = (24, 31)
PATCH_PADDING = 1
GEN_RETRIES = 400
MASK_RETRIES = 200
LAYOUT_RETRIES = 80


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


def _shape_signature(
    pts: list[tuple[int, int]] | np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1), rel)


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
    h, w = crop.shape
    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    cavities = []
    for pts in components_mask((crop == bg) & inner_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "cavity must stay inside the frame"
        sub = crop[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub == bg), "cavity bbox must contain only background"
        assert int(len(pts)) == int(sub.size), "cavity must be a solid rectangle"
        # The tray outside cavities should stay gray; this keeps the cavity interpretation clean.
        for rr in range(r0 - 1, r1 + 2):
            for cc in range(c0 - 1, c1 + 2):
                if 0 <= rr < h and 0 <= cc < w and not (r0 <= rr <= r1 and c0 <= cc <= c1):
                    if 0 < rr < h - 1 and 0 < cc < w - 1 and crop[rr, cc] == bg:
                        raise AssertionError("unexpected extra bg touching a cavity")
        cavities.append((int(r0), int(c0), int(r1), int(c1)))

    assert cavities, "no inner cavities found"
    return sorted(cavities)


def _extract_task_parts(
    g: np.ndarray,
) -> tuple[int, int, np.ndarray, list[dict[str, object]], dict[int, int], dict[int, np.ndarray]]:
    bg = get_bg(g)
    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    cavities = _rectangular_cavities(crop, bg)

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    marker_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    markers = []
    for col in sorted(marker_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            markers.append(
                {
                    "col": int(col),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "sig": _shape_signature(pts_tuples, (r0, c0, r1, c1)),
                }
            )
    assert len(markers) == len(cavities), "expected one guide marker per cavity"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(pts_tuples)),
                    "sig": _shape_signature(pts_tuples, (r0, c0, r1, c1)),
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
            assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
            if ci["area"] > cj["area"]:
                src, tgt = int(ci["col"]), int(cj["col"])
            else:
                src, tgt = int(cj["col"]), int(ci["col"])
            assert src not in color_map, f"duplicate legend mapping for source color {src}"
            color_map[src] = tgt
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))
    assert color_map, "no legend mappings found"

    sprite_masks: dict[int, np.ndarray] = {}
    shape_to_src: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for src_col in sorted(color_map):
        candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert len(candidates) == 1, f"expected one non-legend sprite for color {src_col}, found {len(candidates)}"
        base = candidates[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        assert sh == sw, f"rotation placement requires square sprites, got {(sh, sw)}"
        mask = np.zeros((sh, sw), dtype=bool)
        for r, c in base["pts"]:
            mask[r - sr0, c - sc0] = True
        sprite_masks[src_col] = mask
        assert base["sig"] not in shape_to_src, "duplicate unrotated sprite shape outside the frame"
        shape_to_src[base["sig"]] = int(src_col)

    markers_by_bbox = {tuple(m["bbox"]): m for m in markers}
    placements = []
    for r0, c0, r1, c1 in cavities:
        ch = r1 - r0 + 1
        cw = c1 - c0 + 1
        expected_marker_bbox = (r0 - ch, c0 - cw, r0 - 1, c0 - 1)
        assert expected_marker_bbox in markers_by_bbox, "missing guide marker for cavity"
        marker = markers_by_bbox[expected_marker_bbox]
        sig = marker["sig"]
        assert sig in shape_to_src, "interior marker shape has no matching external sprite"
        src_col = int(shape_to_src[sig])
        placements.append(
            {
                "marker_col": int(marker["col"]),
                "marker_bbox": expected_marker_bbox,
                "cavity_bbox": (r0, c0, r1, c1),
                "src_col": src_col,
            }
        )
    assert len(placements) == len(markers), "unused or duplicated markers detected"

    return bg, frame_col, crop, placements, color_map, sprite_masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a framed tray by shape-matching guide markers to external sprites, rotating them by cavity quadrant, and recoloring them through the outside legend.

    Input structure:
    - A gray rectangular frame encloses a gray tray whose occupied slots each contain a sparse guide marker in the slot's upper-left corner and a square bg cavity in the slot's lower-right corner.
    - Outside the frame sit one unrotated source-colored sprite template per shape and one compact two-color legend block per source color.

    Rule:
    - Detect every cavity and the same-size marker placed exactly one cavity-width above and left of it, then normalize the marker's unrotated shape.
    - Decode the outside legends to obtain source-color -> target-color mappings, and match each marker to the unique outside sprite with the same unrotated shape signature.
    - Rotate that matched sprite according to the cavity's frame quadrant (upper-left 0, upper-right 90, lower-right 180, lower-left 270), embed it into the cavity, recolor it to the legend target color, and clear the guide marker back to the gray tray.

    Key insight:
    - The nearby marker chooses which sprite family to use before any transformation, while the cavity position independently chooses orientation, so the finished filled hole can intentionally differ from the marker that selected it.
    """
    g = np.asarray(grid, dtype=int)
    bg, frame_col, crop, placements, color_map, sprite_masks = _extract_task_parts(g)
    H, W = crop.shape

    out = np.full(crop.shape, frame_col, dtype=int)

    for item in placements:
        src_col = int(item["src_col"])
        r0, c0, r1, c1 = item["cavity_bbox"]
        mask = sprite_masks[src_col]
        sh, sw = mask.shape
        assert (r1 - r0 + 1, c1 - c0 + 1) == (sh, sw), "cavity size does not match matched sprite"

        krot = _rotation_k_for_bbox((r0, c0, r1, c1), H, W)
        rot_mask = np.rot90(mask, krot)
        tgt_col = int(color_map[src_col])
        sprite = np.full((sh, sw), bg, dtype=int)
        sprite[rot_mask] = tgt_col
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        paste(out, sprite, r0, c0)

    return out


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _rot_signature(mask: np.ndarray) -> tuple[bytes, ...]:
    return tuple(np.rot90(mask, k).tobytes() for k in range(4))


def _has_four_distinct_rotations(mask: np.ndarray) -> bool:
    return len(set(_rot_signature(mask))) == 4


def _sprite_mask(n: int) -> np.ndarray:
    for _ in range(MASK_RETRIES):
        lo = max(n + 1, int(np.ceil(SPRITE_FILL_MIN * n * n)))
        hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
        target = random.randint(lo, max(lo, hi))

        mask = np.zeros((n, n), dtype=bool)
        c = random.randrange(n)
        mask[0, c] = True
        for r in range(1, n):
            c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
            mask[r, c] = True

        for edge_col in (0, n - 1):
            pts = list(zip(*np.where(mask)))
            rr, cc = random.choice(pts)
            c0, c1 = sorted((cc, edge_col))
            mask[rr, c0 : c1 + 1] = True

        if random.random() < 0.5:
            mask = mask.T.copy()
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        while int(mask.sum()) < target:
            dilated = np.zeros_like(mask)
            dilated[1:, :] |= mask[:-1, :]
            dilated[:-1, :] |= mask[1:, :]
            dilated[:, 1:] |= mask[:, :-1]
            dilated[:, :-1] |= mask[:, 1:]
            frontier = list(zip(*np.where(dilated & ~mask)))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        if mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any() and _has_four_distinct_rotations(mask):
            return mask

    raise RuntimeError(f"failed to build an asymmetric {n}x{n} sprite mask")


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _slot_base(
    slot: tuple[int, int], top_margin: int, left_margin: int
) -> tuple[int, int]:
    rr, cc = slot
    return 1 + top_margin + rr * CELL_SIZE, 1 + left_margin + cc * CELL_SIZE


def _marker_bbox(
    slot: tuple[int, int], top_margin: int, left_margin: int
) -> tuple[int, int, int, int]:
    r0, c0 = _slot_base(slot, top_margin, left_margin)
    return (r0, c0, r0 + SPRITE_N - 1, c0 + SPRITE_N - 1)


def _cavity_bbox(
    slot: tuple[int, int], top_margin: int, left_margin: int
) -> tuple[int, int, int, int]:
    r0, c0 = _slot_base(slot, top_margin, left_margin)
    return (r0 + SPRITE_N, c0 + SPRITE_N, r0 + 2 * SPRITE_N - 1, c0 + 2 * SPRITE_N - 1)


def _slot_rotations(
    frame_h: int, frame_w: int, top_margin: int, left_margin: int
) -> dict[tuple[int, int], int]:
    slot_rot = {}
    for rr in range(GRID_ROWS):
        for cc in range(GRID_COLS):
            bbox = _cavity_bbox((rr, cc), top_margin, left_margin)
            slot_rot[(rr, cc)] = _rotation_k_for_bbox(bbox, frame_h, frame_w)
    assert set(slot_rot.values()) == {0, 1, 2, 3}, "layout does not expose all four quadrants"
    return slot_rot


def _choose_frame_layout() -> tuple[int, int, int, int, dict[tuple[int, int], int]] | None:
    content_h = GRID_ROWS * CELL_SIZE
    content_w = GRID_COLS * CELL_SIZE
    for _ in range(LAYOUT_RETRIES):
        frame_h = random.randint(max(FRAME_H_RANGE[0], content_h + 2), FRAME_H_RANGE[1])
        frame_w = random.randint(max(FRAME_W_RANGE[0], content_w + 2), FRAME_W_RANGE[1])
        top_margin = random.randint(0, frame_h - 2 - content_h)
        left_margin = random.randint(0, frame_w - 2 - content_w)
        try:
            slot_rot = _slot_rotations(frame_h, frame_w, top_margin, left_margin)
        except AssertionError:
            continue
        return frame_h, frame_w, top_margin, left_margin, slot_rot
    return None


def _assign_slot_sources(
    slots: list[tuple[int, int]], src_cols: list[int], slot_rot: dict[tuple[int, int], int]
) -> dict[tuple[int, int], int] | None:
    pairs = [
        (a, b)
        for i, a in enumerate(slots)
        for b in slots[i + 1 :]
        if slot_rot[a] != slot_rot[b]
    ]
    if not pairs:
        return None

    anchor_a, anchor_b = random.choice(pairs)
    anchor_col = random.choice(src_cols)
    cell_src = {anchor_a: anchor_col, anchor_b: anchor_col}

    remaining_slots = [s for s in slots if s not in (anchor_a, anchor_b)]
    random.shuffle(remaining_slots)
    other_cols = [c for c in src_cols if c != anchor_col]
    if len(remaining_slots) < len(other_cols):
        return None

    for col, slot in zip(other_cols, remaining_slots):
        cell_src[slot] = col

    for slot in remaining_slots:
        if slot not in cell_src:
            cell_src[slot] = random.choice(src_cols)

    repeated_rotations = {slot_rot[s] for s, col in cell_src.items() if col == anchor_col}
    if len(repeated_rotations) < 2:
        return None
    return cell_src


def _assign_marker_colors(
    slots: list[tuple[int, int]], cell_src: dict[tuple[int, int], int], marker_palette: list[int]
) -> dict[tuple[int, int], int] | None:
    for _ in range(80):
        marker_colors = {slot: random.choice(marker_palette) for slot in slots}
        color_to_srcs: dict[int, set[int]] = {}
        for slot in slots:
            color_to_srcs.setdefault(marker_colors[slot], set()).add(cell_src[slot])
        if max(len(srcs) for srcs in color_to_srcs.values()) >= 2:
            return marker_colors
    return None


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool
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
    _bg, _frame_col, crop, placements, _color_map, sprite_masks = _extract_task_parts(g)
    H, W = crop.shape

    rotation_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    changed_vs_unrotated = 0
    marker_color_to_srcs: dict[int, set[int]] = {}
    for item in placements:
        src_col = int(item["src_col"])
        marker_col = int(item["marker_col"])
        bbox = item["cavity_bbox"]
        krot = _rotation_k_for_bbox(bbox, H, W)
        rotation_hist[krot] += 1
        marker_color_to_srcs.setdefault(marker_col, set()).add(src_col)
        if not np.array_equal(np.rot90(sprite_masks[src_col], krot), sprite_masks[src_col]):
            changed_vs_unrotated += 1

    return {
        "rotation_hist": rotation_hist,
        "changed_vs_unrotated": changed_vs_unrotated,
        "marker_conflict_max": max(len(srcs) for srcs in marker_color_to_srcs.values()),
        "num_cavities": len(placements),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k_hi = min(NUM_TYPES_RANGE[1], GRID_ROWS * GRID_COLS - 1)
    if k_hi < NUM_TYPES_RANGE[0]:
        return None
    k = random.randint(NUM_TYPES_RANGE[0], k_hi)
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    seen_rot_variants: set[bytes] = set()
    masks: dict[int, np.ndarray] = {}
    for src in src_cols:
        for _ in range(80):
            mask = _sprite_mask(SPRITE_N)
            rot_variants = set(_rot_signature(mask))
            if seen_rot_variants.isdisjoint(rot_variants):
                seen_rot_variants.update(rot_variants)
                masks[src] = mask
                break
        else:
            return None

    layout = _choose_frame_layout()
    if layout is None:
        return None
    frame_h, frame_w, top_margin, left_margin, slot_rot = layout

    slots = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    cell_src = _assign_slot_sources(slots, src_cols, slot_rot)
    if cell_src is None:
        return None

    marker_pool = [c for c in usable if c not in src_cols]
    if len(marker_pool) < 2:
        return None
    marker_palette = random.sample(marker_pool, random.randint(2, min(3, len(marker_pool))))
    marker_colors = _assign_marker_colors(slots, cell_src, marker_palette)
    if marker_colors is None:
        return None

    input_frame = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    output_frame = np.full((frame_h, frame_w), FRAME_COL, dtype=int)

    for slot in slots:
        src = cell_src[slot]
        marker_r0, marker_c0, _mr1, _mc1 = _marker_bbox(slot, top_margin, left_margin)
        cav_r0, cav_c0, cav_r1, cav_c1 = _cavity_bbox(slot, top_margin, left_margin)

        marker_patch = _colored_sprite(masks[src], marker_colors[slot], FRAME_COL)
        paste(input_frame, marker_patch, marker_r0, marker_c0, transparent=FRAME_COL)
        input_frame[cav_r0 : cav_r1 + 1, cav_c0 : cav_c1 + 1] = bg

        output_frame[cav_r0 : cav_r1 + 1, cav_c0 : cav_c1 + 1] = bg
        rot_mask = np.rot90(masks[src], slot_rot[slot])
        sprite = _colored_sprite(rot_mask, color_to_tgt[src], bg)
        paste(output_frame, sprite, cav_r0, cav_c0)

    patch_area = k * (SPRITE_N * SPRITE_N + 12 + 12)
    frame_area = frame_h * frame_w
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w >= frame_area + patch_area + frame_area
    ]
    if not possible_dims:
        return None

    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.randrange(0, h - frame_h + 1)
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
        if metrics["changed_vs_unrotated"] < 4:
            return None
        if metrics["marker_conflict_max"] < 2:
            return None
        if set(k for k, v in metrics["rotation_hist"].items() if v > 0) != {0, 1, 2, 3}:
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
