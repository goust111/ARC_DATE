# Combo variant of 58f5dbd5 (index 13)
# combo_spec: D3/1+D7/2+D15/0
# Name: 可旋转motif+反向掩码+迭代应用

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The seed task had one same-colored motif per 5x5 block, used directly as a hole stencil.
#    Here each block color appears on 2 or 3 small motifs in the source region. Each visible motif
#    is a rotated or flipped copy of one canonical 3x3 prototype. The solver must first normalize
#    every motif back to its prototype orientation, then apply reverse masking repeatedly:
#    start from a solid 5x5 block, keep only the centered cells covered by motif 1, then keep only
#    the cells that also survive motif 2, and so on. The final colored pixels are exactly the
#    intersection of all centered canonical motifs. Example: if a color has canonical masks A and B,
#    the old task would carve holes shaped like one raw visible motif; this combo keeps only A∩B.
# b) Complexity calibration:
#    Canonicalization is constrained to a fixed library of asymmetric full-3x3 prototypes, so humans
#    can infer "same underlying shape in different poses" from repeated evidence across colors.
#    Multi-step behavior is limited to 2-3 motifs per block color, enough to make the intersection
#    visibly stricter without flooding the grid with too many repeated objects.
# c) Implicit-assumption audit of the original main():
#    - "One useful motif per color" no longer holds; the solver must keep every non-block component
#      of each block color.
#    - "Visible motif pose is authoritative" no longer holds; each source motif must be mapped back
#      to its prototype orientation before use.
#    - "Applying a motif means recolor covered cells to bg" no longer holds; now covered cells are
#      the only cells that survive, so sequential application is an AND over centered motif masks.
#    - Packing by block row/column order still holds unchanged.
# d) Placement legality:
#    Destination blocks remain a disjoint 5x5 grid and motifs remain in a disjoint 3x3-slot region
#    with one-cell gaps, so connected-components extraction stays reliable. The generator only uses
#    layouts that fit within the original canvas bounds.
# e) Visual observability:
#    Every matching color repeats across multiple motif slots, so multi-step usage is obvious. The
#    generator forces non-identity rotations/flips for matching motifs and rejects samples where the
#    final preserved intersection would equal the naive "use visible masks directly" baseline.
# f) Visual cleanliness:
#    The input still has exactly two clean regions: solid blocks and motif sprites. No markers are
#    drawn on top of blocks, and same-colored motifs remain separated by slot gaps.
# g) Change inferability:
#    Color is only a grouping key, not an arithmetic rule. The preserved output shape is small and
#    centered, making the reverse-mask interpretation visible. Repeated same-colored motifs make it
#    clear that all of them matter, not just one.
# h) Fusion nuance:
#    Under reverse masking, sequential application is commutative at the pixel level: applying motifs
#    in any order yields the same intersection. The solver still recovers them in reading order so the
#    procedure is well defined, while the generator makes the D15 dimension observable through strict
#    shrinkage after combining multiple same-colored motifs.
# i) New solver assumptions and generator guarantees:
#    - Every matching motif belongs to the fixed prototype library up to a dihedral transform.
#    - Every block color has 2 or 3 motifs, and each sequence shrinks relative to any single motif.
#    - At least one block in every sample differs from the raw-visible baseline, the single-step
#      baseline, and the original subtractive-carving intuition.


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


def _variant_entries(mask: np.ndarray) -> list[tuple[np.ndarray, int, bool, bool]]:
    """Return unique dihedral variants as (variant, rot_k, flip, is_identity)."""
    out: list[tuple[np.ndarray, int, bool, bool]] = []
    seen: set[bytes] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for flip in (False, True):
            cand = np.fliplr(rot) if flip else rot
            key = cand.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((cand.copy(), int(k), bool(flip), bool(k == 0 and not flip)))
    return out


STANDARD_MOTIFS: list[np.ndarray] = [
    np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
]

STANDARD_MOTIF_BY_VARIANT: dict[bytes, np.ndarray] = {}
MOTIF_INDEX_BY_VARIANT: dict[bytes, int] = {}
NON_ID_VARIANTS_BY_INDEX: dict[int, list[np.ndarray]] = {}
for motif_idx, motif in enumerate(STANDARD_MOTIFS):
    variants = _variant_entries(motif)
    assert len(variants) == 8
    non_id: list[np.ndarray] = []
    for variant, _k, _flip, is_identity in variants:
        key = variant.tobytes()
        assert key not in STANDARD_MOTIF_BY_VARIANT
        STANDARD_MOTIF_BY_VARIANT[key] = motif.copy()
        MOTIF_INDEX_BY_VARIANT[key] = int(motif_idx)
        if not is_identity:
            non_id.append(variant.copy())
    NON_ID_VARIANTS_BY_INDEX[int(motif_idx)] = non_id


def _canonicalize_motif(mask: np.ndarray) -> np.ndarray:
    """Map a displayed 3x3 motif back to its standard prototype orientation."""
    key = np.asarray(mask, dtype=bool).tobytes()
    if key not in STANDARD_MOTIF_BY_VARIANT:
        raise ValueError("motif is not a supported rotated/flipped prototype")
    return STANDARD_MOTIF_BY_VARIANT[key].copy()


def _is_solid_block(comp: dict, grid: np.ndarray) -> bool:
    """Return whether a component is a solid 5x5 monochrome block."""
    b = comp["bbox"]
    if bbox_h(b) != 5 or bbox_w(b) != 5:
        return False
    color = int(comp["color"])
    return bool((crop_bbox(grid, b) == int(color)).all())


def _component_mask(comp: dict) -> tuple[int, int, np.ndarray]:
    """Return top-left coordinates and a tight mask for one component."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
    return int(r0), int(c0), mask_from_coords(rel_coords, (h, w))


def _centered_in_block(mask: np.ndarray) -> np.ndarray:
    """Embed a motif mask into the centered 3x3 window of a 5x5 tile."""
    mh, mw = map(int, mask.shape)
    off_r = (5 - mh) // 2
    off_c = (5 - mw) // 2
    out = np.zeros((5, 5), dtype=bool)
    out[off_r : off_r + mh, off_c : off_c + mw] = np.asarray(mask, dtype=bool)
    return out


def _preserve_tile(bg: int, color: int, motif_sequence: list[np.ndarray]) -> np.ndarray:
    """Keep only cells covered by every centered motif in the sequence."""
    if not motif_sequence:
        return np.full((5, 5), int(color), dtype=int)

    keep = np.ones((5, 5), dtype=bool)
    for mask in motif_sequence:
        keep &= _centered_in_block(mask)

    tile = np.full((5, 5), int(bg), dtype=int)
    tile[keep] = int(color)
    return tile


def _extract_scene(
    grid: np.ndarray,
) -> tuple[int, list[dict[str, int]], dict[int, list[np.ndarray]]]:
    """Recover block objects and ordered visible motif masks grouped by block color."""
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
    motif_items: dict[int, list[tuple[int, int, int, int, np.ndarray]]] = {
        color: [] for color in block_colors
    }
    for comp in comps:
        color = int(comp["color"])
        if color not in block_colors:
            continue
        if _is_solid_block(comp, g):
            continue
        r0, c0, mask = _component_mask(comp)
        b = comp["bbox"]
        motif_items[color].append(
            (int(r0), int(c0), int(comp["area"]), int(bbox_h(b) * bbox_w(b)), mask)
        )

    ordered_visible: dict[int, list[np.ndarray]] = {}
    for color, items in motif_items.items():
        items.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        ordered_visible[int(color)] = [item[4] for item in items]

    return bg, blocks, ordered_visible


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize repeated transformed motifs, keep only their shared protected cells inside each block, and pack the results.

    Input structure:
    - The most common color is the background.
    - Several solid 5x5 monochrome blocks appear in one region, each with a distinct non-bg color.
    - Elsewhere, every block color appears on multiple small same-colored 3x3 motifs, and each visible motif may be rotated or mirror-flipped.

    Rule:
    - For one block color, collect all non-block motifs of that color in reading order and convert each visible pose back to its canonical prototype orientation.
    - Center every canonical motif inside the 5x5 block and apply reverse masking repeatedly: after each step, only cells covered by that motif remain colored.
    - Equivalently, the finished tile keeps exactly the intersection of all centered canonical motifs; then pack all finished 5x5 tiles with 1-cell gaps and a 1-cell bg border, preserving block row/column order.

    Key insight:
    - The source motifs are transformed clues, and repeated same-colored motifs act as successive protection masks: a pixel survives only if every normalized motif covers it.
    """
    bg, blocks, ordered_visible = _extract_scene(grid)

    canonical_sequences: dict[int, list[np.ndarray]] = {}
    for color, visible_seq in ordered_visible.items():
        assert visible_seq
        canonical_sequences[int(color)] = [_canonicalize_motif(mask) for mask in visible_seq]

    preserved: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        color = int(block["col"])
        preserved[(int(block["r0"]), int(block["c0"]))] = _preserve_tile(
            bg, color, canonical_sequences.get(color, [])
        )

    rs = sorted({int(b["r0"]) for b in blocks})
    cs = sorted({int(b["c0"]) for b in blocks})
    r_index = {int(r): int(i) for i, r in enumerate(rs)}
    c_index = {int(c): int(j) for j, c in enumerate(cs)}

    out_h = int(1 + len(rs) * 5 + (len(rs) - 1) + 1)
    out_w = int(1 + len(cs) * 5 + (len(cs) - 1) + 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    step = 6
    for block in blocks:
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        rr = int(1 + r_index[r0] * step)
        cc = int(1 + c_index[c0] * step)
        out[rr : rr + 5, cc : cc + 5] = preserved[(r0, c0)]

    return out


# --------------- generator (combo) ---------------

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

GENERATION_RETRIES = 500


def _spanned_size(count: int, item_size: int, step: int) -> int:
    """Return the span for count fixed-size items placed at a constant step."""
    return (count - 1) * step + item_size


def _can_split(H: int, W: int, dh: int, dw: int, mh: int, mw: int) -> bool:
    """Return whether destination and motif regions can be separated by an axis."""
    return (
        dh + mh + 2 * OUTER_MARGIN + REGION_GAP <= H
        and max(dw, mw) + 2 * OUTER_MARGIN <= W
    ) or (
        dw + mw + 2 * OUTER_MARGIN + REGION_GAP <= W
        and max(dh, mh) + 2 * OUTER_MARGIN <= H
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
    """Choose a valid separated placement for destination and motif regions."""
    schemes = _placement_schemes(H, W, dh, dw, mh, mw)
    if not schemes:
        return None
    scheme = random.choice(schemes)
    if scheme in {"dst_above", "motif_above"}:
        return _place_vertical(H, W, dh, dw, mh, mw, scheme)
    return _place_horizontal(H, W, dh, dw, mh, mw, scheme)


def _configs(H: int, W: int) -> list[tuple[int, int, int, int, int, int]]:
    """Feasible layouts as (R, C, Mr, Mc, rounds, filled_slots)."""
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


def _choose_config(
    H: int, W: int, count: int | None = None, rounds: int | None = None
) -> tuple[int, int, int, int, int, int] | None:
    """Sample a valid layout, optionally restricted to one (block-count, rounds) branch."""
    configs = _configs(H, W)
    if count is not None:
        configs = [cfg for cfg in configs if cfg[0] * cfg[1] == count]
    if rounds is not None:
        configs = [cfg for cfg in configs if cfg[4] == rounds]
    if not configs:
        return None
    return random.choice(configs)


BRANCH_PAIRS = sorted(
    {
        (count, rounds)
        for count, round_options in ROUNDS_BY_COUNT.items()
        for rounds in round_options
    }
)
CANVAS_DIMS_BY_BRANCH_PAIR: dict[tuple[int, int], list[tuple[int, int]]] = {}
for branch_pair in BRANCH_PAIRS:
    count, rounds = branch_pair
    dims: list[tuple[int, int]] = []
    for H in range(CANVAS_HEIGHT_RANGE[0], CANVAS_HEIGHT_RANGE[1] + 1):
        for W in range(CANVAS_WIDTH_RANGE[0], CANVAS_WIDTH_RANGE[1] + 1):
            if _choose_config(H, W, count=count, rounds=rounds) is not None:
                dims.append((int(H), int(W)))
    assert dims
    CANVAS_DIMS_BY_BRANCH_PAIR[branch_pair] = dims


def _intersect_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Intersect a list of equal-shaped boolean masks."""
    out = np.ones_like(masks[0], dtype=bool)
    for mask in masks:
        out &= np.asarray(mask, dtype=bool)
    return out


VALID_CANONICAL_INDEX_SEQUENCES_BY_ROUNDS: dict[int, list[tuple[int, ...]]] = {}
for rounds in sorted({rounds for values in ROUNDS_BY_COUNT.values() for rounds in values}):
    valid: list[tuple[int, ...]] = []
    for idxs in itertools.product(range(len(STANDARD_MOTIFS)), repeat=rounds):
        masks = [STANDARD_MOTIFS[idx] for idx in idxs]
        inter = _intersect_masks(masks)
        area = int(inter.sum())
        if len(set(idxs)) < 2:
            continue
        if not (1 <= area <= 4):
            continue
        if any(np.array_equal(inter, mask) for mask in masks):
            continue
        valid.append(tuple(int(idx) for idx in idxs))
    assert valid
    VALID_CANONICAL_INDEX_SEQUENCES_BY_ROUNDS[int(rounds)] = valid


def _choose_palette(bg: int, count: int, distractors: int) -> tuple[list[int], list[int]]:
    """Pick block colors followed by distractor colors from the non-bg palette."""
    palette = [c for c in range(COLOR_COUNT) if c != bg]
    random.shuffle(palette)
    block_colors = palette[:count]
    distractor_colors = palette[count : count + distractors]
    return block_colors, distractor_colors


def _choose_sequence_indices(count: int, rounds: int) -> list[tuple[int, ...]]:
    """Sample canonical prototype index sequences with some visible variety."""
    pool = VALID_CANONICAL_INDEX_SEQUENCES_BY_ROUNDS[int(rounds)]
    seqs = [random.choice(pool) for _ in range(count)]
    if count >= 2 and len(set(seqs)) == 1 and len(pool) > 1:
        alternatives = [seq for seq in pool if seq != seqs[0]]
        seqs[-1] = random.choice(alternatives)
    return seqs


def _sample_display_sequence(canonical_idxs: tuple[int, ...]) -> list[np.ndarray] | None:
    """Pick non-identity visible poses whose raw intersection differs from the canonical one."""
    canonical_masks = [STANDARD_MOTIFS[idx].copy() for idx in canonical_idxs]
    canonical_tile = _preserve_tile(0, 1, canonical_masks)
    for _ in range(200):
        displayed = [
            random.choice(NON_ID_VARIANTS_BY_INDEX[int(idx)]).copy()
            for idx in canonical_idxs
        ]
        raw_tile = _preserve_tile(0, 1, displayed)
        if not np.array_equal(raw_tile, canonical_tile):
            return displayed
    return None


def _sample_sequences(
    block_colors: list[int], rounds: int
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]] | None:
    """Assign canonical sequences plus visible transformed sequences to each block color."""
    canonical_sequences: dict[int, list[np.ndarray]] = {}
    display_sequences: dict[int, list[np.ndarray]] = {}
    for color, idxs in zip(block_colors, _choose_sequence_indices(len(block_colors), rounds)):
        canonical_sequences[int(color)] = [STANDARD_MOTIFS[idx].copy() for idx in idxs]
        display = _sample_display_sequence(idxs)
        if display is None:
            return None
        display_sequences[int(color)] = display
    return canonical_sequences, display_sequences


def _paint_motif(grid: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    """Paint one 3x3 motif through its mask into an input grid."""
    patch = np.full((MOTIF_SIZE, MOTIF_SIZE), int(color), dtype=int)
    paste(grid, patch, r0, c0, mask=mask)


def _render_expected(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
) -> np.ndarray:
    """Render packed preserved tiles from one motif-sequence assignment."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = _preserve_tile(bg, color, motif_sequences[color])
            paste(
                expected,
                tile,
                OUTPUT_BORDER + i * DST_BLOCK_STEP,
                OUTPUT_BORDER + j * DST_BLOCK_STEP,
            )
    return expected


def _render_first_only(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
) -> np.ndarray:
    """Render the baseline that keeps only the first canonical motif of each color."""
    first_only = {
        int(color): [motif_sequences[int(color)][0].copy()] for color in block_colors
    }
    return _render_expected(bg, R, C, block_colors, first_only)


def _render_hole_baseline(
    bg: int,
    R: int,
    C: int,
    block_colors: list[int],
    motif_sequences: dict[int, list[np.ndarray]],
) -> np.ndarray:
    """Render the old subtractive-carving intuition using only the first canonical motif."""
    out_h = 2 * OUTPUT_BORDER + _spanned_size(R, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    out_w = 2 * OUTPUT_BORDER + _spanned_size(C, DST_BLOCK_SIZE, DST_BLOCK_STEP)
    expected = np.full((out_h, out_w), bg, dtype=int)

    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            tile = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            tile[_centered_in_block(motif_sequences[color][0])] = int(bg)
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
    """Paint the solid 5x5 destination blocks into the input grid."""
    for i in range(R):
        for j in range(C):
            color = int(block_colors[i * C + j])
            block = np.full((DST_BLOCK_SIZE, DST_BLOCK_SIZE), int(color), dtype=int)
            paste(grid, block, dst_r + i * DST_BLOCK_STEP, dst_c + j * DST_BLOCK_STEP)


def _choose_slot_payloads(
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    display_sequences: dict[int, list[np.ndarray]],
) -> dict[int, tuple[int, np.ndarray]]:
    """Assign matching sequence motifs and distractors to occupied slots."""
    capacity = Mr * Mc
    matching = sum(len(display_sequences[int(color)]) for color in block_colors)
    filled = matching + len(distractor_colors)

    occupied_slots = random.sample(range(capacity), filled)
    available = set(occupied_slots)
    payloads: dict[int, tuple[int, np.ndarray]] = {}

    color_order = block_colors.copy()
    random.shuffle(color_order)
    for color in color_order:
        seq = display_sequences[int(color)]
        chosen = sorted(random.sample(list(available), len(seq)))
        for slot_idx, mask in zip(chosen, seq):
            payloads[int(slot_idx)] = (int(color), mask)
            available.remove(slot_idx)

    rest = list(available)
    random.shuffle(rest)
    for slot_idx, color in zip(rest, distractor_colors):
        motif_idx = random.randrange(len(STANDARD_MOTIFS))
        mask = random.choice(NON_ID_VARIANTS_BY_INDEX[int(motif_idx)]).copy()
        payloads[int(slot_idx)] = (int(color), mask)
    return payloads


def _paint_motif_slots(
    grid: np.ndarray,
    motif_r: int,
    motif_c: int,
    Mr: int,
    Mc: int,
    block_colors: list[int],
    distractor_colors: list[int],
    display_sequences: dict[int, list[np.ndarray]],
) -> None:
    """Paint motif slots using transformed matching motifs plus distractors."""
    payloads = _choose_slot_payloads(
        Mr, Mc, block_colors, distractor_colors, display_sequences
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
    display_sequences: dict[int, list[np.ndarray]],
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Render the sampled destination region plus transformed motif region."""
    dst_r, dst_c, motif_r, motif_c = placement
    inp = np.full((H, W), int(bg), dtype=int)
    _paint_destination_blocks(inp, dst_r, dst_c, R, C, block_colors)
    _paint_motif_slots(
        inp, motif_r, motif_c, Mr, Mc, block_colors, distractor_colors, display_sequences
    )
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Sample one valid input grid and reject degenerate cases."""
    count, rounds = random.choice(BRANCH_PAIRS)
    H, W = random.choice(CANVAS_DIMS_BY_BRANCH_PAIR[(count, rounds)])
    bg = random.randrange(COLOR_COUNT)

    config = _choose_config(H, W, count=count, rounds=rounds)
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
    sampled = _sample_sequences(block_colors, rounds)
    if sampled is None:
        return None
    canonical_sequences, display_sequences = sampled

    expected = _render_expected(bg, R, C, block_colors, canonical_sequences)
    raw_visible = _render_expected(bg, R, C, block_colors, display_sequences)
    first_only = _render_first_only(bg, R, C, block_colors, canonical_sequences)
    hole_baseline = _render_hole_baseline(bg, R, C, block_colors, canonical_sequences)
    if np.array_equal(expected, raw_visible):
        return None
    if np.array_equal(expected, first_only):
        return None
    if np.array_equal(expected, hole_baseline):
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
        display_sequences,
        placement,
    )

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate one valid combo-variant input grid."""
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58f5dbd5 combo input")
