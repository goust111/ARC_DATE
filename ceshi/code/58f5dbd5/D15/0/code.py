# Variant of 58f5dbd5
# Dimension: D15 — 多步处理 (multi_step)
# Change: 单步处理：所有方块同时独立雕刻，无顺序依赖... → 迭代雕刻（多轮应用）

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import (
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    is_contiguous,
    paste,
)


# Design analysis for the D15 variant
# -----------------------------------
# spec:
#   core_change:
#     old: each block color had one effective motif, so carving was a one-shot stencil.
#     new: each block color has an ordered sequence of 2-3 motifs; every round carves the
#       same 5x5 tile again, with later rounds rotated by an additional quarter-turn.
#   concretization:
#     - Input still has a destination region of solid 5x5 blocks and a separate motif region.
#     - A block color now appears on multiple 3x3 motifs instead of exactly one.
#     - The visible order is spatial: for one color, sort its motifs by top-to-bottom then
#       left-to-right position in the motif region.
#     - Round 1 uses the motif as drawn, round 2 uses the same motif after a 90 degree
#       clockwise rotation, round 3 after a 180 degree rotation.
#   behavior_trace:
#     - Original path: color X -> choose one smallest motif -> center it once -> holes.
#     - Variant path: color X -> collect every non-block component of color X ->
#       order them by position -> carve each one in turn with step-dependent rotation.
#     - Difference: the same input can contain several same-colored motifs, and the output
#       depends on their order, not just their union as raw shapes.
#   complexity_calibration:
#     - Sequence length is restricted to 2 or 3, which is enough to show multi-step logic
#       without making correspondence ambiguous.
#     - Motifs stay in 3x3 slots and blocks stay 5x5, preserving the visual grammar of the
#       original task while making the new rule visible through repeated same-color sprites.
#     - The generator uses asymmetric motifs and rejects samples whose final output matches
#       the simpler "no per-round rotation" baseline, so D15 remains observable.
#   implicit_assumptions_review:
#     - The original solver assumed at most one useful motif per color and encoded that by
#       keeping only the smallest same-colored component. That would silently discard later
#       rounds here, so the variant must keep every motif component for a block color.
#     - The original carve step assumed one static mask. Here the loop over motifs must be
#       monotone: later rounds only turn more cells into background and never recolor holes.
#     - Sorting by color id would be invisible; sorting by motif position is visible and is
#       therefore the generator's supported ordering contract.
#   placement_legality:
#     - Blocks and motif slots still live in disjoint regions separated by one axis, so
#       components never overlap and the solver can recover them with connected-components.
#     - Slot spacing remains 4 for 3x3 motifs, preserving a one-cell gap between components.
#   visual_observability:
#     - Repeated same-colored motifs are obvious in the input.
#     - Their ordinal effect is readable because later rounds rotate by a fixed pattern
#       (0, 90, 180 degrees), and generator masks are chosen to be asymmetric enough that
#       these rotations visibly change the carved result.
#   cleanliness:
#     - No extra markers are added inside blocks, so the scene remains close to the original:
#       a clean block region and a clean motif region.
#   color_reasoning:
#     - Colors are still only used for matching motif groups to blocks, not for sequencing.
# new_main_assumptions_and_generator_guarantees:
#   - Every destination block color has at least 2 motifs in the motif region.
#   - Same-colored motifs do not touch because they are placed in separate 3x3 slots.
#   - The top-left order of same-colored motifs is the intended sequence order.
#   - At least one block in every generated sample has a final carved tile that differs from
#     the simpler baseline that would carve all motifs without per-round rotation.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given coords."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _is_solid_block(comp: dict, grid: np.ndarray) -> bool:
    """Return whether a component is a solid 5x5 monochrome block."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    col = int(comp["color"])
    return bool((crop_bbox(grid, b) == col).all())


def _component_mask(comp: dict) -> tuple[int, int, np.ndarray]:
    """Return top-left coordinates and a tight boolean mask for one component."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
    return r0, c0, mask_from_coords(rel_coords, (h, w))


def _rotated_mask(mask: np.ndarray, step_idx: int) -> np.ndarray:
    """Rotate the step mask by 0, 90, 180, 270 degrees clockwise over successive rounds."""
    turns = int(step_idx) % 4
    return np.rot90(mask, -turns)


def _carve_tile(
    bg: int, color: int, motif_sequence: list[np.ndarray], *, rotate_steps: bool
) -> np.ndarray:
    """Apply all motifs of one color to a fresh 5x5 tile."""
    tile = np.full((5, 5), int(color), dtype=int)
    for step_idx, raw_mask in enumerate(motif_sequence):
        mm = _rotated_mask(raw_mask, step_idx) if rotate_steps else raw_mask
        mh, mw = map(int, mm.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mm):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Iteratively carve each solid block with an ordered sequence of same-colored motifs, then pack the finished blocks.

    Input structure:
    - The dominant color is the background.
    - Several solid 5x5 monochrome blocks form the destination set.
    - Elsewhere, each block color appears on multiple small same-colored motif sprites; their positions encode order.

    Rule:
    - For one block color, collect every non-block component of that same color and sort them in reading order.
    - Carve the first motif as drawn, the second after a 90 degree clockwise turn, the third after 180 degrees, and continue cycling if needed.
    - After all rounds are applied to every 5x5 block, pack the carved blocks into a compact grid with the original row/column order, 1-cell gaps, and a 1-cell border.

    Key insight:
    - Repeated motifs of one color are a stencil sequence rather than duplicate choices, and the round number changes each motif's orientation before it removes more cells.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"
    )

    blocks: list[dict[str, int]] = []
    for comp in comps:
        if _is_solid_block(comp, g):
            r0, c0, _r1, _c1 = map(int, comp["bbox"])
            blocks.append({"col": int(comp["color"]), "r0": int(r0), "c0": int(c0)})

    assert blocks

    block_colors = {int(b["col"]) for b in blocks}
    motif_sequences: dict[int, list[tuple[int, int, int, int, np.ndarray]]] = {
        color: [] for color in block_colors
    }
    for comp in comps:
        color = int(comp["color"])
        if color not in block_colors:
            continue
        if _is_solid_block(comp, g):
            continue
        r0, c0, m = _component_mask(comp)
        b = comp["bbox"]
        motif_sequences[color].append(
            (int(r0), int(c0), int(comp["area"]), int(bbox_h(b) * bbox_w(b)), m)
        )

    ordered_masks: dict[int, list[np.ndarray]] = {}
    for color, items in motif_sequences.items():
        items.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        ordered_masks[color] = [item[4] for item in items]

    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        color = int(b["col"])
        motif_seq = ordered_masks.get(color, [])
        carved[(int(b["r0"]), int(b["c0"]))] = _carve_tile(
            bg, color, motif_seq, rotate_steps=True
        )

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for b in blocks:
        r0 = int(b["r0"])
        c0 = int(b["c0"])
        rr = int(1 + r_index[r0] * step)
        cc = int(1 + c_index[c0] * step)
        out[rr : rr + 5, cc : cc + 5] = carved[(r0, c0)]

    return out


# --------------- generator (o5x) ---------------

CANVAS_HEIGHT_RANGE = (18, 22)
CANVAS_WIDTH_RANGE = (18, 24)
COLOR_COUNT = 10

DST_BLOCK_SIZE = 5
DST_BLOCK_STEP = 6
DST_LAYOUTS = {
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}

MOTIF_SIZE = 3
MOTIF_STEP = 4
MOTIF_AREA_RANGE = (4, 8)
MOTIF_LAYOUTS = {
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
}
ROUNDS_BY_COUNT = {
    3: (2, 3),
    4: (2,),
}
MAX_DISTRACTORS = 2
MAX_EMPTY_SLOTS = 2

OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1

GENERATION_RETRIES = 400


def _spanned_size(count: int, item_size: int, step: int) -> int:
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 connected masks that visibly occupy the full slot."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
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
ALL_MASKS = [m.copy() for area in sorted(MASKS_BY_AREA) for m in MASKS_BY_AREA[area]]
STEP_MASKS = {
    step: [m.copy() for m in ALL_MASKS if not np.array_equal(m, _rotated_mask(m, step))]
    for step in range(1, 4)
}


def _can_split(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> bool:
    return (
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H
        and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ) or (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W
        and max(dh, mh) + 2 * OUTER_MARGIN <= H
    )


def _placement_schemes(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int
) -> list[str]:
    schemes: list[str] = []
    if (
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H
        and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ):
        schemes.extend(["dst_above", "motif_above"])
    if (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W
        and max(dh, mh) + 2 * OUTER_MARGIN <= H
    ):
        schemes.extend(["dst_left", "motif_left"])
    return schemes


def _place_vertical(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int, scheme: str
) -> tuple[int, int, int, int]:
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
    if scheme == "dst_left":
        left_h, left_w, right_h, right_w = dh, dw, mh, mw
    else:
        left_h, left_w, right_h, right_w = mh, mw, dh, dw

    left_c = random.randint(
        OUTER_MARGIN, W - left_w - right_w - OUTER_MARGIN - REGION_GAP
    )
    right_c = random.randint(left_c + left_w + REGION_GAP, W - right_w - OUTER_MARGIN)
    left_r = random.randint(OUTER_MARGIN, H - left_h - OUTER_MARGIN)
    right_r = random.randint(OUTER_MARGIN, H - right_h - OUTER_MARGIN)
    if scheme == "dst_left":
        return left_r, left_c, right_r, right_c
    return right_r, right_c, left_r, left_c


def _place_regions(
    H: int, W: int, dh: int, dw: int, mh: int, mw: int
) -> tuple[int, int, int, int] | None:
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None
    scheme = random.choice(schemes)
    if scheme in {"dst_above", "motif_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int, int]]:
    """Feasible layouts: (R, C, Mr, Mc, rounds, filled_slots)."""
    out: list[tuple[int, int, int, int, int, int]] = []
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for rounds in ROUNDS_BY_COUNT[count]:
                matching = count * rounds
                for capacity, motif_layouts in MOTIF_LAYOUTS.items():
                    if matching > capacity:
                        continue
                    min_filled = max(matching, capacity - MAX_EMPTY_SLOTS)
                    max_filled = min(capacity, matching + MAX_DISTRACTORS)
                    if min_filled > max_filled:
                        continue
                    for filled in range(min_filled, max_filled + 1):
                        distractors = filled - matching
                        if distractors > COLOR_COUNT - 1 - count:
                            continue
                        for Mr, Mc in motif_layouts:
                            mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                            mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                            if _can_split(H, W, dh, dw, mh, mw):
                                out.append((R, C, Mr, Mc, rounds, filled))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int, int] | None:
    configs = _configs(H, W)
    if not configs:
        return None
    counts = sorted({R * C for R, C, _Mr, _Mc, _rounds, _filled in configs})
    count = random.choice(counts)
    return random.choice([cfg for cfg in configs if cfg[0] * cfg[1] == count])


def _random_mask() -> np.ndarray:
    area = random.randint(*MOTIF_AREA_RANGE)
    return random.choice(MASKS_BY_AREA[area]).copy()


def _choose_palette(bg: int, count: int, distractors: int) -> tuple[list[int], list[int]]:
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + distractors]
    return block_colors, distractor_colors


def _sequence_has_step_effect(sequence: list[np.ndarray]) -> bool:
    variant = _carve_tile(0, 1, sequence, rotate_steps=True)
    baseline = _carve_tile(0, 1, sequence, rotate_steps=False)
    return not np.array_equal(variant, baseline)


def _sample_sequence(rounds: int) -> list[np.ndarray] | None:
    for _ in range(200):
        seq = [_random_mask()]
        for step in range(1, rounds):
            seq.append(random.choice(STEP_MASKS[step]).copy())
        if _sequence_has_step_effect(seq):
            return seq
    return None


def _sample_sequences(
    block_colors: list[int], rounds: int
) -> dict[int, list[np.ndarray]] | None:
    out: dict[int, list[np.ndarray]] = {}
    for color in block_colors:
        seq = _sample_sequence(rounds)
        if seq is None:
            return None
        out[int(color)] = seq
    return out


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
    *,
    rotate_steps: bool,
) -> np.ndarray:
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _carve_tile(
                bg, color, motif_sequences[color], rotate_steps=rotate_steps
            )
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _paint_destination_blocks(
    grid: np.ndarray, dst_r: int, dst_c: int, R: int, C: int, block_colors: list[int]
) -> None:
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), color, dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _choose_slot_payloads(
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
) -> dict[int, tuple[int, np.ndarray]]:
    capacity = Mr * Mc
    matching = sum(len(motif_sequences[color]) for color in block_colors)
    filled = matching + len(distractor_colors)

    occupied_slots = random.sample(range(capacity), filled)
    available = set(occupied_slots)
    payloads: dict[int, tuple[int, np.ndarray]] = {}

    color_order = block_colors.copy()
    random.shuffle(color_order)
    for color in color_order:
        rounds = len(motif_sequences[color])
        chosen = sorted(random.sample(list(available), rounds))
        for slot_idx, mask in zip(chosen, motif_sequences[color]):
            payloads[int(slot_idx)] = (int(color), mask)
            available.remove(slot_idx)

    rest = list(available)
    random.shuffle(rest)
    for slot_idx, color in zip(rest, distractor_colors):
        payloads[int(slot_idx)] = (int(color), _random_mask())
    return payloads


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
) -> None:
    payloads = _choose_slot_payloads(
        Mr, Mc, block_colors, distractor_colors, motif_sequences
    )
    for slot_idx, (color, mask) in payloads.items():
        sr = slot_idx // Mc
        sc = slot_idx % Mc
        _paint_motif(grid, motif_r + sr * MOTIF_STEP, motif_c + sc * MOTIF_STEP, color, mask)


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, motif_sequences
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, Mr, Mc, rounds, filled = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    distractor_count = filled - count * rounds
    block_colors, distractor_colors = _choose_palette(bg, count, distractor_count)
    motif_sequences = _sample_sequences(block_colors, rounds)
    if motif_sequences is None:
        return None

    expected = _render_expected(
        bg, R, C, block_colors, motif_sequences, rotate_steps=True
    )
    baseline = _render_expected(
        bg, R, C, block_colors, motif_sequences, rotate_steps=False
    )
    if np.array_equal(expected, baseline):
        return None

    inp = _render_input(
        H,
        W,
        bg,
        R,
        C,
        Mr,
        Mc,
        block_colors,
        distractor_colors,
        motif_sequences,
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 D15 input")
