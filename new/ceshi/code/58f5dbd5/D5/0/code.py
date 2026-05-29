# Variant of 58f5dbd5
# Dimension: D5 — 目标对象：方块结构 (dst)
# Change: 5x5纯色单色实心方块，每个方块颜色不同... → 空心框架方块
#
# Design analysis
# ===============
# variant_spec:
#   dst_frames:
#     geometry: each destination object is a 5x5 monochrome square frame:
#       the outer ring is one non-bg color and the 3x3 interior is bg.
#     layout: unchanged from the base task; frames still sit on a regular RxC grid
#       with step 6 and are separated from the motif region.
#   src_motifs:
#     geometry: unchanged from the base task; each visible motif is a connected
#       3x3 same-colored sprite whose filled cells touch all four sides of its
#       own 3x3 bbox, so its edge profile is readable.
#   transformation:
#     matching: unchanged; each frame is paired with the smallest same-colored
#       component outside the 5x5 destination set.
#     carving rule: the motif is still conceptually centered in the 5x5 target,
#       but only the frame ring can be removed. A motif cell aligned to the top,
#       bottom, left, or right side of the centered 3x3 window deletes the
#       corresponding border cell on that side. Corner motif cells delete one
#       aligned cut on each adjacent side. Frame corners are never deleted.
#     output: unchanged packing rule; carved frames are packed by original
#       row/column order with 1-cell gaps and a 1-cell outer bg border.
#
# behavior_trace:
#   original_local_step:
#     a top-left motif pixel centered in a solid 5x5 block simply turned one
#     interior block cell into bg.
#   variant_local_step:
#     the same top-left motif pixel is now inside the frame cavity, so it would
#     be invisible if copied literally. Instead it projects to two aligned cuts:
#     one on the top edge and one on the left edge of the 5x5 frame.
#   visible_difference:
#     output objects remain hollow squares whose borders are partially snipped;
#     they cannot collapse back into the original "solid tile with punched
#     interior holes" appearance.
#
# complexity_calibration:
#   - Kept the original color-matching mechanic and grid packing, so the change
#     stays focused on D5 instead of adding a second independent rule.
#   - The frame is still 5x5, preserving the original scale and region layout.
#   - Motifs still vary over multiple connected 3x3 masks, so different samples
#     show repeated evidence that border cuts depend on the motif shape.
#   - Frame corners are preserved to avoid total disappearance and to keep the
#     output visually legible as "cut frames" rather than blank tiles.
#
# hidden_assumptions_checked:
#   - Base solver assumed a 5x5 bbox filled entirely with one color; that is no
#     longer true, so destination detection must explicitly verify "ring color +
#     bg interior".
#   - Base carving rewrote arbitrary tile cells to bg; that would be inert here
#     because the centered 3x3 lives inside the hollow cavity. The new carving
#     must instead target border cells derived from motif edge positions.
#   - The motif-selection heuristic "smallest non-destination component of the
#     same color" remains safe because destination frames always have bbox 5x5,
#     while motifs stay inside 3x3 slots.
#
# placement_and_generator_guards:
#   - Frames and motifs are still placed in separated regions chosen by the same
#     axis-split logic, so no objects overlap and frame cavities stay readable.
#   - Every destination color is unique, and distractor motif colors are drawn
#     from disjoint colors, so color matching remains unambiguous.
#   - Motif masks are restricted to 3x3 connected shapes touching all four sides;
#     therefore every matched frame receives cuts on all four edges and the D5
#     change is visually observable in every generated sample.
#
# solver_generator_contract:
#   - main() assumes every true destination object is a 5x5 square frame with a
#     bg 3x3 cavity, and generate_input() constructs exactly that.
#   - main() assumes each destination color has at least one same-colored motif
#     component outside the frame set; generate_input() guarantees one matching
#     motif per destination color plus optional distractors of other colors.

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


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


def carve_frame(color: int, bg: int, motif_mask: np.ndarray | None) -> np.ndarray:
    """Project the centered motif's edge cells onto aligned frame-border cuts."""
    tile = make_frame_tile(int(color), int(bg))
    if motif_mask is None:
        return tile

    mh, mw = map(int, motif_mask.shape)
    off_r = (FRAME_SIZE - mh) // 2
    off_c = (FRAME_SIZE - mw) // 2

    for rr, cc in np.argwhere(motif_mask):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))

        if tr == 1:
            tile[0, tc] = int(bg)
        if tr == FRAME_SIZE - 2:
            tile[FRAME_SIZE - 1, tc] = int(bg)
        if tc == 1:
            tile[tr, 0] = int(bg)
        if tc == FRAME_SIZE - 2:
            tile[tr, FRAME_SIZE - 1] = int(bg)

    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve color-matched motifs into hollow frames and repack the cut frames.

    Input structure:
    - bg is the most common color.
    - Several 5x5 monochrome square frames appear in one region; each frame is a colored border around a 3x3 bg cavity.
    - Elsewhere, each frame color has a small same-colored motif sprite, with optional distractor motifs of other colors.

    Rule:
    - Match every motif to the frame of the same color.
    - Center the motif conceptually inside the 5x5 frame, then use the motif's top/bottom/left/right edge cells to snip aligned border cells out of the frame as bg.
    - Pack all carved frames into an output grid preserving the original row/column order, with 1-cell gaps and a 1-cell bg border.

    Key insight:
    - The motif is still a stencil, but now it cuts the surrounding ring rather than the filled interior: motif edge positions project outward onto the frame's border.
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

    # STEP 3) Carve each 5x5 frame by projecting motif-edge cuts to the border.
    carved: dict[tuple[int, int], np.ndarray] = {}
    for frame in frames:
        color = int(frame["col"])
        carved[(int(frame["r0"]), int(frame["c0"]))] = carve_frame(
            int(color), int(bg), motifs.get(int(color))
        )

    # STEP 4) Pack carved 5x5 frames into a compact output grid.
    rs = sorted({int(f["r0"]) for f in frames})
    cs = sorted({int(f["c0"]) for f in frames})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * FRAME_SIZE + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * FRAME_SIZE + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        i = int(r_index[int(r0)])
        j = int(c_index[int(c0)])
        rr = int(1 + i * FRAME_STEP)
        cc = int(1 + j * FRAME_STEP)
        out[rr : rr + FRAME_SIZE, cc : cc + FRAME_SIZE] = carved[(int(r0), int(c0))]

    return out


# --------------- generator (o5x_d5_frame_variant) ---------------

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
OUTPUT_BORDER = 1

# Generation retry budget.
GENERATION_RETRIES = 300


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 motif masks keyed by filled-cell area."""
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


def _place_regions(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> tuple[int, int, int, int] | None:
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


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int] | None:
    """Sample a valid layout while preserving the original two-stage count weighting."""
    configs = _configs(H, W)
    if not configs:
        return None
    counts = sorted({R * C for R, C, _Mr, _Mc, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


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
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int, R: int, C: int, frame_colors: list[int], motif_masks: dict[int, np.ndarray]
) -> np.ndarray:
    """Render the oracle output corresponding to the sampled frame grid."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, FRAME_SIZE, FRAME_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, FRAME_SIZE, FRAME_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(frame_colors[i * C + j])
            tile = carve_frame(int(color), int(bg), motif_masks[color])
            paste(expected, tile, OUTPUT_BORDER + i * FRAME_STEP, OUTPUT_BORDER + j * FRAME_STEP)
    return expected


def _paint_destination_frames(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, frame_colors: list[int], bg: int
) -> None:
    """Paint the intact 5x5 hollow destination frames into the input grid."""
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
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D5 input")
