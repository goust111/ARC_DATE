# Combo variant of 58f5dbd5 (index 29)
# combo_spec: D5/0+D7/2+D9/0
# Name: 空心框架+反向掩码保留+紧凑无边框

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


# Design analysis
# ===============
# variant_spec:
#   dst_frames:
#     geometry:
#       - Each destination object is a 5x5 hollow square frame.
#       - Only the outer ring is colored; the centered 3x3 cavity is bg.
#     layout:
#       - Input placement stays on the original RxC lattice with top-left step 6.
#       - Destination region is still axis-separated from the motif region.
#   src_motifs:
#     geometry:
#       - Visible motifs remain connected monochrome sprites whose bbox fits in 3x3.
#       - Matching is still by exact color equality.
#     constraint:
#       - Generated motifs touch all four sides of their own 3x3 bbox so the frame
#         projection leaves evidence on every side of the output object.
#   fused_transform:
#     old_base:
#       - A centered motif punched bg holes inside a solid 5x5 block.
#     D5_effect:
#       - The target is hollow, so the centered 3x3 motif no longer overlaps colored
#         pixels literally; visible action must be interpreted on the surrounding ring.
#     D7_effect:
#       - We invert the mask logic: instead of deleting motif-covered target cells, we
#         keep only the target cells protected by the motif and erase everything else.
#     fused_resolution:
#       - Treat each motif edge/corner cell as projecting outward to aligned frame-border
#         cells, exactly the same visible correspondence that made D5 meaningful.
#       - Then apply the inverse mask on that projected border support: projected border
#         cells stay in the frame color, every other ring cell becomes bg.
#   D9_effect:
#     output:
#       - Packed result has no outer border and no inter-tile gaps.
#       - Tiles are placed flush in their original row/column order, so output shape is
#         simply (R*5) x (C*5).
#
# behavior_trace:
#   concrete_example:
#     - Suppose a frame color matches a 3x3 motif whose top-middle, center-left,
#       center-right, and bottom-middle cells are on.
#     - Base task on a solid block would carve four interior holes in the centered 3x3.
#     - D5 alone would convert those edge positions into four cuts on the frame ring.
#     - This combo keeps exactly those four aligned ring cells instead of cutting them
#       away, producing a sparse border-only cross anchored to the frame perimeter.
#   visible_difference:
#     - The output never looks like a filled tile with holes, because the 3x3 center
#       remains bg.
#     - It also never looks like the D5-only frame-cut variant, because the preserved
#       border support is the projected motif subset rather than its complement.
#
# complexity_calibration:
#   - We keep the original 3-6 destination objects and 6-8 motif slots so the solver
#     still must separate object classes, pair by color, and preserve spatial order.
#   - Destination colors remain unique. This matters more after D9 because adjacent
#     output tiles touch with no separator; color differences keep tile boundaries legible.
#   - Motifs still come from a small family of connected 3x3 masks, so the rule stays
#     learnable from repeated examples instead of exploding into many arbitrary cases.
#
# implicit_assumptions_checked:
#   - Original Step 1 assumed a solid 5x5 monochrome component. That breaks here, so
#     destination detection must explicitly verify "colored ring + bg cavity".
#   - Original Step 3 assumed motif cells overlap colored target pixels directly. That is
#     false for hollow frames; the combo needs a projection from centered motif positions
#     to border positions before the inverse mask is applied.
#   - Original packing assumed a 1-cell border and 1-cell gaps. D9 removes both, so the
#     output geometry and placement step must be recomputed independently of input spacing.
#   - Tight packing would be ambiguous if neighboring tiles shared a color, so the
#     generator must continue assigning distinct colors to all destination frames.
#
# placement_and_generator_guards:
#   - Frames and motifs are still placed in disjoint regions separated by at least one
#     row or column gap, preventing accidental merges during component extraction.
#   - Each destination color gets exactly one matching motif mask; distractor motifs use
#     disjoint colors, so the "smallest same-colored non-frame component" heuristic stays safe.
#   - Motif masks touching all four sides guarantee every generated output tile has
#     preserved pixels on all four sides of the frame, making the D5+D7 fusion observable.
#   - Tight output packing is visually clean because tiles touch only along full 5-cell
#     edges and adjacent tiles always differ in color.
#
# observability_and_non_degeneracy:
#   - A human can infer the combo from a few examples because colored pixels survive only
#     on the frame ring, never in the 3x3 cavity, and their pattern follows the motif shape.
#   - The rule does not collapse to the original task: original outputs change centered
#     interior cells, while this combo preserves projected border cells and erases the rest.
#   - The rule also does not collapse to D5-only carving: D5-only would keep the opposite
#     border subset. Generator-side validation later checks that produced samples realize
#     this distinction in practice.
#
# solver_generator_contract:
#   - main() assumes true destination objects are 5x5 hollow frames with bg interiors.
#   - main() assumes every frame color appears in at least one non-frame component.
#   - generate_input() constructs inputs that satisfy both assumptions and keeps the
#     output-tight-packing case readable through unique frame colors.


FRAME_SIZE = 5
FRAME_STEP = 6
MOTIF_SIZE = 3
MOTIF_STEP = 4

BORDER_MASK = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
BORDER_MASK[0, :] = True
BORDER_MASK[-1, :] = True
BORDER_MASK[:, 0] = True
BORDER_MASK[:, -1] = True


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Local helper: build a boolean mask with True at the given coords (in-bounds only)."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def make_frame_tile(color: int, bg: int) -> np.ndarray:
    """Build one intact 5x5 hollow frame."""
    tile = np.full((FRAME_SIZE, FRAME_SIZE), int(bg), dtype=int)
    tile[BORDER_MASK] = int(color)
    return tile


def is_hollow_frame(patch: np.ndarray, color: int, bg: int) -> bool:
    """Return whether patch is a 5x5 square frame of `color` around a bg cavity."""
    if patch.shape != (FRAME_SIZE, FRAME_SIZE):
        return False
    return bool((patch[BORDER_MASK] == int(color)).all() and (patch[~BORDER_MASK] == int(bg)).all())


def project_motif_to_frame_mask(motif_mask: np.ndarray | None) -> np.ndarray:
    """Project a centered motif's border support onto the 5x5 frame ring."""
    keep = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    if motif_mask is None:
        return keep

    mh, mw = map(int, motif_mask.shape)
    off_r = (FRAME_SIZE - mh) // 2
    off_c = (FRAME_SIZE - mw) // 2

    for rr, cc in np.argwhere(motif_mask):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))

        if tr == 1:
            keep[0, tc] = True
        if tr == FRAME_SIZE - 2:
            keep[FRAME_SIZE - 1, tc] = True
        if tc == 1:
            keep[tr, 0] = True
        if tc == FRAME_SIZE - 2:
            keep[tr, FRAME_SIZE - 1] = True

    return keep


def preserve_frame_overlap(color: int, bg: int, motif_mask: np.ndarray | None) -> np.ndarray:
    """Keep only the frame-border pixels protected by the projected motif mask."""
    tile = np.full((FRAME_SIZE, FRAME_SIZE), int(bg), dtype=int)
    keep = project_motif_to_frame_mask(motif_mask)
    tile[keep] = int(color)
    return tile


def carve_frame_overlap(color: int, bg: int, motif_mask: np.ndarray | None) -> np.ndarray:
    """D5-style reference: erase the projected support from an intact frame."""
    tile = make_frame_tile(int(color), int(bg))
    keep = project_motif_to_frame_mask(motif_mask)
    tile[keep] = int(bg)
    return tile


def preserve_centered_motif(color: int, bg: int, motif_mask: np.ndarray | None) -> np.ndarray:
    """D7-style reference on a solid 5x5 tile: keep only the centered motif cells."""
    tile = np.full((FRAME_SIZE, FRAME_SIZE), int(bg), dtype=int)
    if motif_mask is None:
        return tile

    mh, mw = map(int, motif_mask.shape)
    off_r = (FRAME_SIZE - mh) // 2
    off_c = (FRAME_SIZE - mw) // 2
    for rr, cc in np.argwhere(motif_mask):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))
        tile[tr, tc] = int(color)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve color-matched motif projections on hollow frames and repack them tightly.

    Input structure:
    - bg is the most common color.
    - Several 5x5 monochrome square frames appear in one region; each frame is a colored
      border surrounding a 3x3 bg cavity.
    - Elsewhere, each frame color has a small same-colored motif sprite, with optional
      distractor motifs of other colors.

    Rule:
    - Match every motif to the frame of the same color.
    - Center the motif conceptually inside the 5x5 frame, project its edge/corner support
      outward onto aligned frame-border cells, and keep only those protected border cells
      in the frame color; every other frame pixel becomes bg.
    - Pack all transformed 5x5 tiles by original row/column order with no gaps and no
      outer border, so the output is a flush grid of touching tiles.

    Key insight:
    - Because the target is already hollow, the motif does not act on interior pixels
      directly. The meaningful overlap is the motif's projection onto the surrounding ring,
      and the inverse mask keeps exactly that projected border subset.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Identify destination objects: 5x5 monochrome hollow square frames.
    frames: list[dict[str, int]] = []
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != FRAME_SIZE or bbox_w(b) != FRAME_SIZE:
            continue
        color = int(comp["color"])
        patch = crop_bbox(g, b)
        if is_hollow_frame(patch, int(color), int(bg)):
            r0, c0, _r1, _c1 = map(int, b)
            frames.append({"col": int(color), "r0": int(r0), "c0": int(c0)})

    assert not (not frames)

    # STEP 2) For each frame color, find the smallest same-colored non-frame component.
    motifs: dict[int, np.ndarray] = {}
    frame_colors = {int(f["col"]) for f in frames}
    for fcol in frame_colors:
        candidates: list[tuple[int, int, int, int, np.ndarray]] = []
        comps_col = find_components(
            g,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(fcol)],
            return_format="dict",
        )
        for comp in comps_col:
            b = comp["bbox"]
            if bbox_h(b) == FRAME_SIZE and bbox_w(b) == FRAME_SIZE:
                patch = crop_bbox(g, b)
                if is_hollow_frame(patch, int(fcol), int(bg)):
                    continue

            r0, c0, r1, c1 = map(int, b)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(fcol)] = candidates[0][4]

    # STEP 3) Preserve only the motif-projected support on each 5x5 frame.
    preserved: dict[tuple[int, int], np.ndarray] = {}
    for frame in frames:
        color = int(frame["col"])
        preserved[(int(frame["r0"]), int(frame["c0"]))] = preserve_frame_overlap(
            int(color), int(bg), motifs.get(int(color))
        )

    # STEP 4) Pack transformed tiles into a tight output grid with no gaps or border.
    rs = sorted({int(f["r0"]) for f in frames})
    cs = sorted({int(f["c0"]) for f in frames})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(len(rs) * FRAME_SIZE)
    out_w = int(len(cs) * FRAME_SIZE)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(i * FRAME_SIZE)
        cc = int(j * FRAME_SIZE)
        out[rr : rr + FRAME_SIZE, cc : cc + FRAME_SIZE] = preserved[(int(r0), int(c0))]

    return out


# --------------- generator (combo_d5_d7_d9) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

# Destination frame geometry.
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}
FRAME_COUNTS = tuple(sorted(DST_LAYOUTS))

# Motif slot geometry and shape constraints.
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
}
FILLED_SLOT_RANGE = (6, 8)
MAX_EMPTY_SLOTS = 2

# Region placement and packed output geometry.
OUTER_MARGIN = 1
REGION_GAP = 1

# Generation retry budget.
GENERATION_RETRIES = 300


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid motif masks keyed by filled-cell area."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {area: [] for area in range(min_area, max_area + 1)}
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        m = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(m.sum())
        if area not in masks:
            continue
        if not (m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()):
            continue
        if is_contiguous(m.astype(int), background=0, connectivity=8):
            masks[area].append(m)
    return masks


MASKS_BY_AREA = _make_masks_by_area()


def _can_split(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> bool:
    """Return whether destination and motif regions can be separated by an axis."""
    return (
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ) or (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H
    )


def _placement_schemes(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> list[str]:
    """List feasible orderings for separated destination and motif regions."""
    schemes: list[str] = []
    if dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H and max(dw, mw) + 2 * OUTER_MARGIN <= W:
        schemes.extend(["dst_above", "motif_above"])
    if dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W and max(dh, mh) + 2 * OUTER_MARGIN <= H:
        schemes.extend(["dst_left", "motif_left"])
    return schemes


def _place_vertical(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
    """Place vertically separated regions and return destination then motif origins."""
    if scheme == "dst_above":
        top_h, top_w, bot_h, bot_w = dh, dw, mh, mw
    else:
        top_h, top_w, bot_h, bot_w = mh, mw, dh, dw

    top_r = random.randint(OUTER_MARGIN, H - top_h - bot_h - OUTER_MARGIN - REGION_GAP)
    bot_r = random.randint(top_r + top_h + REGION_GAP, H - bot_h - OUTER_MARGIN)
    top_c = random.randint(OUTER_MARGIN, W - top_w - OUTER_MARGIN)
    bot_c = random.randint(OUTER_MARGIN, W - bot_w - OUTER_MARGIN)
    if scheme == "dst_above":
        return top_r, top_c, bot_r, bot_c
    return bot_r, bot_c, top_r, top_c


def _place_horizontal(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
    """Place horizontally separated regions and return destination then motif origins."""
    if scheme == "dst_left":
        left_h, left_w, right_h, right_w = dh, dw, mh, mw
    else:
        left_h, left_w, right_h, right_w = mh, mw, dh, dw

    left_c = random.randint(OUTER_MARGIN, W - left_w - right_w - OUTER_MARGIN - REGION_GAP)
    right_c = random.randint(left_c + left_w + REGION_GAP, W - right_w - OUTER_MARGIN)
    left_r = random.randint(OUTER_MARGIN, H - left_h - OUTER_MARGIN)
    right_r = random.randint(OUTER_MARGIN, H - right_h - OUTER_MARGIN)
    if scheme == "dst_left":
        return left_r, left_c, right_r, right_c
    return right_r, right_c, left_r, left_c


def _place_regions(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int
) -> tuple[int, int, int, int] | None:
    """Choose a valid separated placement for destination and motif regions."""
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None
    scheme = random.choice(schemes)
    if scheme in {"dst_above", "motif_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int]]:
    """Enumerate layout configurations that fit in the sampled canvas."""
    out: list[tuple[int, int, int, int, int]] = []
    min_filled, max_filled = FILLED_SLOT_RANGE
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            dh = _spanned_size(R, FRAME_SIZE, FRAME_STEP)
            dw = _spanned_size(C, FRAME_SIZE, FRAME_STEP)
            for filled in range(min_filled, max_filled + 1):
                distractors = filled - count
                if not (0 <= distractors <= count):
                    continue
                if distractors > COLOR_COUNT - 1 - count:
                    continue
                for capacity, motif_layouts in MOTIF_LAYOUTS.items():
                    if not (count <= capacity <= 2 * count):
                        continue
                    if not (filled <= capacity and capacity - filled <= MAX_EMPTY_SLOTS):
                        continue
                    for Mr, Mc in motif_layouts:
                        mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                        mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                        if _can_split(H, W, dh, dw, mh, mw):
                            out.append((R, C, Mr, Mc, filled))
    return out


def _configs_for_count(H: int, W: int, count: int) -> list[tuple[int, int, int, int, int]]:
    """Filter feasible layouts down to one destination-count branch."""
    return [cfg for cfg in _configs(H, W) if cfg[0] * cfg[1] == count]


def _choose_canvas_and_config() -> tuple[int, int, tuple[int, int, int, int, int]] | None:
    """Sample canvas and layout while keeping 3/4/6-frame branches well-covered."""
    counts = list(FRAME_COUNTS)
    random.shuffle(counts)
    for count in counts:
        candidates: list[tuple[int, int, tuple[int, int, int, int, int]]] = []
        for H in range(CANVAS_HEIGHT_RANGE[0], CANVAS_HEIGHT_RANGE[1] + 1):
            for W in range(CANVAS_WIDTH_RANGE[0], CANVAS_WIDTH_RANGE[1] + 1):
                for cfg in _configs_for_count(H, W, count):
                    candidates.append((H, W, cfg))
        if candidates:
            return random.choice(candidates)
    return None


def _random_mask() -> np.ndarray:
    """Sample one motif mask from the precomputed masks."""
    min_area, max_area = MOTIF_AREA_RANGE
    area = random.randint(min_area, max_area)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _choose_palette(bg: int, count: int, filled: int) -> tuple[list[int], list[int]]:
    """Pick destination colors followed by distractor colors from non-background colors."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    frame_colors = palette[:count]
    distractor_colors = palette[count : count + (filled - count)]
    return frame_colors, distractor_colors


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, frame_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled frame grid."""
    expected = np.full((R * FRAME_SIZE, C * FRAME_SIZE), bg, dtype=int)
    for i in range(R):
        for j in range(C):
            color = int(frame_colors[i * C + j])
            tile = preserve_frame_overlap(int(color), int(bg), motif_masks[color])
            paste(expected, tile, i * FRAME_SIZE, j * FRAME_SIZE)
    return expected


def _render_d5_reference(
    bg: int, R: int, C: int, frame_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the D5-only transform under the same tight packing."""
    expected = np.full((R * FRAME_SIZE, C * FRAME_SIZE), bg, dtype=int)
    for i in range(R):
        for j in range(C):
            color = int(frame_colors[i * C + j])
            tile = carve_frame_overlap(int(color), int(bg), motif_masks[color])
            paste(expected, tile, i * FRAME_SIZE, j * FRAME_SIZE)
    return expected


def _render_d7_reference(
    bg: int, R: int, C: int, frame_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the D7-only preserve-mask transform under the same tight packing."""
    expected = np.full((R * FRAME_SIZE, C * FRAME_SIZE), bg, dtype=int)
    for i in range(R):
        for j in range(C):
            color = int(frame_colors[i * C + j])
            tile = preserve_centered_motif(int(color), int(bg), motif_masks[color])
            paste(expected, tile, i * FRAME_SIZE, j * FRAME_SIZE)
    return expected


def _is_distinct_combo_output(
    bg: int, R: int, C: int, frame_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> bool:
    """Reject outputs that collapse to an obvious single-variant behavior."""
    combo = _render_expected(bg, R, C, frame_colors, motif_masks)
    if np.array_equal(combo, _render_d5_reference(bg, R, C, frame_colors, motif_masks)):
        return False
    if np.array_equal(combo, _render_d7_reference(bg, R, C, frame_colors, motif_masks)):
        return False
    return True


def _paint_destination_frames(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, frame_colors: list[int], bg: int
) -> None:
    """Paint intact 5x5 hollow destination frames into the input grid."""
    for i in range(R):
        for j in range(C):
            color = int(frame_colors[i * C + j])
            frame = make_frame_tile(int(color), int(bg))
            paste(grid, frame, dst_r + i * FRAME_STEP, dst_c + j * FRAME_STEP)


def _slot_items(
    frame_colors: list[int], distractor_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    """Build the shuffled motif payloads for occupied motif slots."""
    items: list[tuple[int, np.ndarray]] = [(int(color), motif_masks[color]) for color in frame_colors]
    items.extend((int(color), _random_mask()) for color in distractor_colors)
    random.shuffle(items)
    return items


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    frame_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
) -> None:
    """Randomly assign matching and distractor motifs to visible motif slots."""
    slots = [(i, j) for i in range(Mr) for j in range(Mc)]
    random.shuffle(slots)
    slot_items = _slot_items(frame_colors, distractor_colors, motif_masks)
    for (sr, sc), (color, mask) in zip(slots, slot_items):
        _paint_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    frame_colors: list[int],
    distractor_colors: list[int],
    motif_masks: dict[int, np.ndarray],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination frames and motif slots into the input grid."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_frames(inp, dst_r, dst_c, R, C, frame_colors, bg)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, frame_colors, distractor_colors, motif_masks)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one input grid and reject it if the solver oracle disagrees."""
    bg = random.randrange(COLOR_COUNT)

    sample = _choose_canvas_and_config()
    if sample is None:
        return None
    H, W, config = sample
    R, C, Mr, Mc, filled = config

    dst_h = _spanned_size(R, FRAME_SIZE, FRAME_STEP)
    dst_w = _spanned_size(C, FRAME_SIZE, FRAME_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    frame_colors, distractor_colors = _choose_palette(bg, count, filled)
    motif_masks = {color: _random_mask() for color in frame_colors}
    if not _is_distinct_combo_output(bg, R, C, frame_colors, motif_masks):
        return None

    expected = _render_expected(bg, R, C, frame_colors, motif_masks)
    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        frame_colors,
        distractor_colors,
        motif_masks,
        placement,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")
