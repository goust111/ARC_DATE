# Combo variant of 58f5dbd5 (index 6)
# combo_spec: D15/0+D19/0
# Name: 迭代雕刻+多对一映射

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


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - The scene still has two axis-separated regions: a block region made of solid 5x5
#      monochrome squares, and a motif region made of disconnected 3x3-slot sprites.
#    - Fusion rule:
#      each block color appears on 2-3 motif sprites, not one. Their visible reading order
#      defines a carving sequence, and every motif in that sequence is centered onto the same
#      5x5 block in turn. Since carving is monotone, the final pattern is the cumulative union
#      of all those same-colored stencils.
#    - Example trace for one color X:
#      original task: block X + one motif m -> center m once -> one hole pattern.
#      D19-only reading: block X has several matching motifs, so all of them must be used.
#      fused path here: block X + motifs m1,m2,m3 -> carve m1, then m2, then m3 onto the
#      already holed tile. Difference: repeated same-colored motifs are all operative inputs,
#      and later steps must contribute additional holes instead of being ignored.
#
# b) Complexity calibration:
#    - Keeping 3-6 blocks from the base task is too dense once every color needs 2-3 motifs
#      while regions remain separated inside an 18-22 by 18-24 canvas.
#    - The minimal stable range is 3-4 blocks. Count 4 uses exactly 2 motifs per color; count
#      3 uses 2 or 3 motifs per color. This preserves the original visual grammar while making
#      the fusion conspicuous through repeated same-colored sprites.
#    - Motifs remain 3x3 connected masks with area 4-8, so humans still read them as the same
#      kind of object as in the base task.
#
# c) Hidden assumptions audit against the original main():
#    - Original Step 2 kept only one smallest same-colored motif. That silently breaks both
#      D15 and D19, because later motifs would be discarded.
#    - Original carving assumed one static mask. Here we need a list of masks per color and
#      must apply all of them in reading order to the same tile.
#    - Order cannot depend on invisible properties such as color ids. We therefore use the
#      top-to-bottom, then left-to-right order of motif positions, which is visible in input.
#    - Repeated carving is monotone: cells only change from block color to background, so
#      sequential application is safe and deterministic.
#
# d) Placement legality:
#    - Blocks and motif slots are still placed in disjoint rectangular regions separated by at
#      least one axis. Slot spacing stays 4 for 3x3 motifs, so components never touch.
#    - Generator uses only layouts whose total region spans fit inside the sampled canvas.
#      Empty motif slots are allowed only in small numbers and never overlap objects.
#
# e) Visual observability and cleanliness:
#    - Every block color repeats 2-3 times in the motif region, so the multi-to-one relation is
#      obvious by color matching alone.
#    - The generator rejects sequences where any later motif is redundant, so humans can see
#      that multiple same-colored motifs all matter to the carved result.
#    - No same-color distractors are introduced. This keeps the motif region readable and makes
#      the solver's "all same-colored small components belong to that block color" assumption
#      exact on generated inputs.
#
# f) Inference support and validation metrics:
#    - Metric 1: each generated block color must have 2 or 3 matching motifs.
#    - Metric 2: the fused output must differ from a single-motif baseline.
#    - Metric 3: every motif in a sequence must contribute at least one unique hole cell. This
#      prevents silent regression to original 1:1 behavior where extra motifs are decorative.
#
# g) New solver assumptions and generator guarantees:
#    - Assumption: every non-5x5 component sharing a destination color is an intended motif.
#    - Guarantee: destination colors are unique, only matching motifs use those colors, motifs
#      are disconnected by slot spacing, and per-color motif order is the visible reading order.


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Build a boolean mask with True at the given in-bounds coordinates."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _is_solid_block(comp: dict, grid: np.ndarray) -> bool:
    """Return whether the component is a solid 5x5 monochrome block."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    color = int(comp["color"])
    return bool((crop_bbox(grid, b) == color).all())


def _component_mask(comp: dict) -> tuple[int, int, np.ndarray]:
    """Return top-left coordinates and tight boolean mask for one component."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
    return int(r0), int(c0), mask_from_coords(rel_coords, (h, w))


def _carve_tile(bg: int, color: int, motif_sequence: list[np.ndarray]) -> np.ndarray:
    """Apply every motif in the visible sequence to one fresh 5x5 block tile."""
    tile = np.full((5, 5), int(color), dtype=int)
    for mask in motif_sequence:
        mh, mw = map(int, mask.shape)
        off_r = (5 - mh) // 2
        off_c = (5 - mw) // 2
        for rr, cc in np.argwhere(mask):
            tr = int(off_r + int(rr))
            tc = int(off_c + int(cc))
            if 0 <= tr < 5 and 0 <= tc < 5:
                tile[tr, tc] = int(bg)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve each solid block with its ordered multi-motif stencil sequence, then pack the finished blocks.

    Input structure:
    - The dominant color is the background.
    - Several solid 5x5 monochrome blocks form the destination region.
    - Elsewhere, each block color appears on 2-3 smaller same-colored motifs, each isolated in a 3x3 slot.

    Rule:
    - For one block color, collect every non-block component of that color and sort them by reading order.
    - Center each motif directly in the 5x5 block and carve it in sequence; later motifs continue carving the already-holed block.
    - After all rounds are applied to every block, pack the carved 5x5 blocks into a compact grid that preserves the original row/column order, with 1-cell gaps and a 1-cell border.

    Key insight:
    - Repeated motifs of one color are all operative stencils for the same block, not distractors or alternative choices.
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
    motifs_by_color: dict[int, list[tuple[int, int, int, int, np.ndarray]]] = {
        color: [] for color in block_colors
    }
    for comp in comps:
        color = int(comp["color"])
        if color not in block_colors or _is_solid_block(comp, g):
            continue
        r0, c0, mask = _component_mask(comp)
        b = comp["bbox"]
        motifs_by_color[color].append(
            (int(r0), int(c0), int(comp["area"]), int(bbox_h(b) * bbox_w(b)), mask)
        )

    ordered_masks: dict[int, list[np.ndarray]] = {}
    for color, items in motifs_by_color.items():
        items.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        ordered_masks[color] = [mask for _r0, _c0, _area, _bbox_area, mask in items]

    carved: dict[tuple[int, int], np.ndarray] = {}
    for b in blocks:
        color = int(b["col"])
        carved[(int(b["r0"]), int(b["c0"]))] = _carve_tile(bg, color, ordered_masks.get(color, []))

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


# --------------- generator (combo variant) ---------------

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
MAX_EMPTY_SLOTS = 1
MIN_FINAL_HOLE_AREA = 6
MAX_FINAL_HOLE_AREA = 8

OUTER_MARGIN = 1
REGION_GAP = 1
OUTPUT_BORDER = 1

GENERATION_RETRIES = 500
SEQUENCE_RETRIES = 400


def _spanned_size(count: int, item_size: int, step: int) -> int:
    return (count - 1) * step + item_size


def _make_masks_by_area() -> dict[int, list[np.ndarray]]:
    """Precompute valid 3x3 connected masks that visibly occupy the full slot."""
    min_area, max_area = MOTIF_AREA_RANGE
    masks: dict[int, list[np.ndarray]] = {
        area: [] for area in range(min_area, max_area + 1)
    }
    for bits in itertools.product([False, True], repeat=MOTIF_SIZE * MOTIF_SIZE):
        mask = np.array(bits, dtype=bool).reshape(MOTIF_SIZE, MOTIF_SIZE)
        area = int(mask.sum())
        if area not in masks:
            continue
        if not (
            mask[0].any()
            and mask[-1].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        ):
            continue
        if is_contiguous(mask.astype(int), background=0, connectivity=8):
            masks[area].append(mask)
    return masks


MASKS_BY_AREA = _make_masks_by_area()
ALL_MASKS = [m.copy() for area in sorted(MASKS_BY_AREA) for m in MASKS_BY_AREA[area]]


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


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int]]:
    """Feasible layouts: (R, C, Mr, Mc, rounds)."""
    out: list[tuple[int, int, int, int, int]] = []
    for count, dst_layouts in DST_LAYOUTS.items():
        for R, C in dst_layouts:
            dh = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            dw = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
            for rounds in ROUNDS_BY_COUNT[count]:
                matching = count * rounds
                for capacity, motif_layouts in MOTIF_LAYOUTS.items():
                    if not (matching <= capacity <= matching + MAX_EMPTY_SLOTS):
                        continue
                    for Mr, Mc in motif_layouts:
                        mh = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
                        mw = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
                        if _can_split(H, W, dh, dw, mh, mw):
                            out.append((R, C, Mr, Mc, rounds))
    return out


def _choose_config(H: int, W: int) -> tuple[int, int, int, int, int] | None:
    configs = _configs(H, W)
    if not configs:
        return None
    counts = sorted({R * C for R, C, _Mr, _Mc, _rounds in configs})
    count = random.choice(counts)
    count_configs = [cfg for cfg in configs if cfg[0] * cfg[1] == count]
    rounds = random.choice(sorted({cfg[4] for cfg in count_configs}))
    return random.choice([cfg for cfg in count_configs if cfg[4] == rounds])


def _choose_palette(bg: int, count: int) -> list[int]:
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    return palette[:count]


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _union_mask(masks: list[np.ndarray]) -> np.ndarray:
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask
    return union


def _sequence_is_valid(sequence: list[np.ndarray]) -> bool:
    """Keep only sequences where every later motif visibly changes the carved block."""
    union = _union_mask(sequence)
    hole_area = int(union.sum())
    if not (MIN_FINAL_HOLE_AREA <= hole_area <= MAX_FINAL_HOLE_AREA):
        return False

    if np.array_equal(union, sequence[0]):
        return False

    signatures = {tuple(int(v) for v in mask.flatten()) for mask in sequence}
    if len(signatures) != len(sequence):
        return False

    for idx, mask in enumerate(sequence):
        others = np.zeros_like(mask, dtype=bool)
        for j, other in enumerate(sequence):
            if j != idx:
                others |= other
        if not np.any(mask & ~others):
            return False
    return True


def _sample_sequence(rounds: int) -> list[np.ndarray] | None:
    for _ in range(SEQUENCE_RETRIES):
        seq = [_random_mask() for _ in range(rounds)]
        if _sequence_is_valid(seq):
            return seq
    return None


def _random_mask() -> np.ndarray:
    return random.choice(ALL_MASKS).copy()


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


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
) -> np.ndarray:
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _carve_tile(bg, color, motif_sequences[color])
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
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _choose_slot_payloads(
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
) -> dict[int, tuple[int, np.ndarray]]:
    capacity = Mr * Mc
    matching = sum(len(motif_sequences[color]) for color in block_colors)
    occupied_slots = random.sample(range(capacity), matching)
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
    return payloads


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
) -> None:
    payloads = _choose_slot_payloads(Mr, Mc, block_colors, motif_sequences)
    for slot_idx, (color, mask) in payloads.items():
        sr = slot_idx // Mc
        sc = slot_idx % Mc
        _paint_motif(
            grid,
            motif_r + sr * MOTIF_STEP,
            motif_c + sc * MOTIF_STEP,
            color,
            mask,
        )


def _render_input(
    H: int,
    W: int,
    bg: int,
    R: int,
    C: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), bg, dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(inp, motif_r, motif_c, Mr, Mc, block_colors, motif_sequences)
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*CANVAS_HEIGHT_RANGE)
    W = random.randint(*CANVAS_WIDTH_RANGE)
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W)
    if config is None:
        return None
    R, C, Mr, Mc, rounds = config

    dst_h = _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    dst_w = _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    motif_h = _spanned_size(Mr, MOTIF_SIZE, MOTIF_STEP)
    motif_w = _spanned_size(Mc, MOTIF_SIZE, MOTIF_STEP)
    placement = _place_regions(H, W, dst_h, dst_w, motif_h, motif_w)
    if placement is None:
        return None

    count = R * C
    block_colors = _choose_palette(bg, count)
    motif_sequences = _sample_sequences(block_colors, rounds)
    if motif_sequences is None:
        return None

    expected = _render_expected(bg, R, C, block_colors, motif_sequences)

    first_only_sequences = {
        color: [motif_sequences[color][0]] for color in block_colors
    }
    single_motif_baseline = _render_expected(bg, R, C, block_colors, first_only_sequences)

    if np.array_equal(expected, single_motif_baseline):
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
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")
