# Combo variant of 58f5dbd5 (index 3)
# combo_spec: D5/0+D7/0
# Name: 空心框架+加法填充
#
# Design analysis
# ===============
# a) Pixel concretization and behavior trace:
#    - Base task input: several solid 5x5 monochrome blocks plus several small
#      same-colored motif sprites in a separate region.
#    - This combo changes the destination template first: every target becomes a
#      5x5 hollow square frame, meaning a colored 1-pixel ring around a 3x3 bg
#      cavity.
#    - It then flips the stencil direction: instead of deleting motif cells from
#      a filled tile, we center the motif inside the frame cavity and color those
#      cells in. For a concrete example, a plus-shaped 3x3 motif used to create
#      bg holes inside a solid square; now the same motif paints a plus into an
#      otherwise empty 3x3 center while the outer ring stays intact.
#    - The visible difference is therefore twofold and coupled: the target is no
#      longer a solid tile, and the motif no longer removes color. Outputs are
#      frame+interior-pattern composites rather than carved solids or snipped
#      frames.
#
# b) Complexity calibration:
#    - The original color-matching problem remains: each destination color is
#      unique, motifs are matched only by visible same-color correspondence, and
#      distractor motifs may still appear.
#    - Keeping 3-6 targets, 6-8 motif slots, connected 3x3 motifs, and compact
#      repacking preserves the original reasoning load.
#    - The frame cavity creates a clear stage for the additive behavior, making
#      the combo visually distinct from both the original task and the pure D5
#      "border-cutting" variant.
#
# c) Implicit-assumption audit:
#    - Base destination detection relied on "5x5 patch is all one color"; this
#      breaks immediately once targets are hollow, so detection must check ring
#      color plus bg interior explicitly.
#    - Base rendering initialized each target with `np.full((5, 5), color)`;
#      that no longer represents the input or output template. We must instead
#      start from an intact frame tile.
#    - The motif matching heuristic "smallest same-colored non-destination
#      component" still works only if generator guarantees one frame per target
#      color and keeps motifs smaller than 5x5.
#    - Packing by original row/column order is unaffected by the fusion.
#
# d) Placement legality:
#    - Destination frames stay on the original 5x5 size / 6-step lattice.
#    - Motifs stay inside 3x3 slots on a 4-step lattice.
#    - Generator still separates destination and motif regions along one axis,
#      with explicit margins and gaps, so objects cannot overlap or clip.
#
# e) Visual observability:
#    - In 2-4 examples, a human can see that the small sprite of a given color
#      reappears centered inside the matching empty frame of the same color.
#    - No hidden dependence on color ids, parity, ordering by numeric value, or
#      other non-visual features is introduced.
#    - Motifs are constrained to touch all four sides of their 3x3 bbox, so the
#      filled interior pattern is legible and not trivially mistaken for noise.
#
# f) Visual cleanliness:
#    - The input still has two clean regions: large 5x5 frame targets and small
#      3x3 motif sprites.
#    - The output remains a tidy packed grid with 1-cell gaps and a 1-cell outer
#      border, so the learned rule is not obscured by clutter.
#
# g) Inferability and solver-generator contract:
#    - Colors are preserved, not remapped: a motif fills the frame of the same
#      visible color.
#    - `main()` assumes true targets are exactly hollow 5x5 frames and that each
#      target color has at least one same-colored non-frame component elsewhere.
#      `generate_input()` constructs exactly those conditions.
#    - `main()` also assumes motifs fit within the centered 3x3 cavity; the
#      generator enforces motif bboxes <= 3x3 and uses the same centered offset
#      when building the oracle output.

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, is_contiguous, paste


FRAME_SIZE = 5
FRAME_STEP = 6
MOTIF_SIZE = 3
MOTIF_STEP = 4
OUTPUT_BORDER = 1
OUTER_MARGIN = 1
REGION_GAP = 1
COLOR_COUNT = 10
GENERATION_RETRIES = 300

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


def fill_frame(color: int, bg: int, motif_mask: np.ndarray | None) -> np.ndarray:
    """Center a motif inside the frame cavity and paint those cells in the frame color."""
    tile = make_frame_tile(int(color), int(bg))
    if motif_mask is None:
        return tile

    mh, mw = map(int, motif_mask.shape)
    off_r = (FRAME_SIZE - mh) // 2
    off_c = (FRAME_SIZE - mw) // 2

    for rr, cc in np.argwhere(motif_mask):
        tr = int(off_r + int(rr))
        tc = int(off_c + int(cc))
        if 0 <= tr < FRAME_SIZE and 0 <= tc < FRAME_SIZE:
            tile[tr, tc] = int(color)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill color-matched motifs into hollow frames and repack the finished tiles.

    Input structure:
    - The most common color is background.
    - One region contains several 5x5 monochrome square frames: colored outer rings with 3x3 background cavities.
    - A separate region contains small monochrome motif sprites, including one motif for each frame color and optional distractors.

    Rule:
    - Match each motif to the frame that has the same color.
    - Conceptually center the motif inside that frame's 3x3 cavity and paint the motif cells with the frame color.
    - Pack the filled 5x5 tiles into a compact output grid that preserves the original frame row/column order, using 1-cell gaps and a 1-cell background border.

    Key insight:
    - The motif is now an additive stencil applied to an empty-centered frame: it tells you where color should appear inside the hollow middle, not where color should be removed.
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

    assert frames

    # STEP 2) For each frame color, find the smallest same-colored non-frame component.
    motifs: dict[int, np.ndarray] = {}
    frame_colors = {int(frame["col"]) for frame in frames}
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
            r0, c0, r1, c1 = map(int, b)
            patch = crop_bbox(g, b)
            if is_hollow_frame(patch, int(fcol), int(bg)):
                continue

            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
            m = mask_from_coords(rel_coords, (h, w))
            candidates.append((int(comp["area"]), int(h * w), int(r0), int(c0), m))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
            motifs[int(fcol)] = candidates[0][4]

    # STEP 3) Fill each 5x5 frame by centering its motif inside the cavity.
    filled: dict[tuple[int, int], np.ndarray] = {}
    for frame in frames:
        color = int(frame["col"])
        filled[(int(frame["r0"]), int(frame["c0"]))] = fill_frame(
            int(color), int(bg), motifs.get(int(color))
        )

    # STEP 4) Pack filled 5x5 tiles into a compact output grid.
    rs = sorted({int(frame["r0"]) for frame in frames})
    cs = sorted({int(frame["c0"]) for frame in frames})
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
        out[rr : rr + FRAME_SIZE, cc : cc + FRAME_SIZE] = filled[(int(r0), int(c0))]

    return out


# --------------- generator (combo D5/0 + D7/0) ---------------

# Canvas and palette bounds.
CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)

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
    """Pick frame colors followed by distractor colors from non-background colors."""
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
            tile = fill_frame(int(color), int(bg), motif_masks[color])
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")
