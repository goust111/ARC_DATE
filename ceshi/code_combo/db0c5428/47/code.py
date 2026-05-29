# Combo variant of db0c5428 (index 47)
# combo_spec: D5/3+D10/2
# Name: 颜色数编码规则+角像素编码中心

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis for the fused D5/3 + D10/2 variant
# a) Concretization and behavior trace:
#    - The visible object is still one 9x9 motif on teal background: eight filled 3x3 blocks
#      surrounding an empty 3x3 center, and the output still grows this motif into the same
#      15x15 five-by-five block layout by copying each visible block to its opposite outer slot.
#    - The fusion changes only how the missing center is decoded.
#      * First layer (D5/3): read the four inward-facing corner pixels:
#          TL block (2,2), TR (2,0), BL (0,2), BR (0,0).
#        They form a 2x2 symbol [[a,b],[c,d]].
#      * Second layer (D10/2): count the number of distinct non-background colors in the motif.
#        If there are exactly 2 foreground colors, decode by nearest-neighbor enlargement:
#            a a b
#            a a b
#            c c d
#        If there are exactly 3 foreground colors, decode by "coded corners + majority cross":
#            a M b
#            M M M
#            c M d
#        where M is the motif's majority foreground color.
#    - Concrete trace on one sample:
#      original task: the center's edge-middle cells came from the four edge blocks facing the hole.
#      fused task: edge blocks no longer explain the center at all; only corner code pixels do,
#      and the foreground color count selects which 3x3 stencil interprets the same 2x2 code.
# b) Complexity calibration:
#    - The generator uses only two mode values, 2 colors and 3 colors, because that is enough to
#      make the meta-rule visible without introducing a hard-to-learn lookup table.
#    - The two decoders are intentionally far apart visually: one produces quadrant expansion, the
#      other produces a majority-colored cross with only four coded corners. This keeps the fusion
#      observable in a few examples rather than hiding the change in a single cell.
#    - Three-color samples force all three colors to appear in the corner symbol itself, so the
#      count-based branch is supported locally inside the center rather than only by a global tally.
# c) Implicit-assumption review of main():
#    - The original center helper assumed the eight inner-boundary sample locations were the whole
#      message. That assumption is broken: the solver must now ignore the edge blocks for center
#      decoding and instead count motif colors plus read only the four corner code pixels.
#    - The bbox extraction and block-grid parsing logic still hold because the motif remains one
#      contiguous 9x9 foreground bbox and the center is still an empty 3x3 hole in the input.
#    - The outer-copy loop still holds verbatim because block geometry and block destinations do
#      not change in this fusion.
# d) Placement legality:
#    - The motif is still placed with a full 3-cell block of padding on every side, so the 15x15
#      expanded layout always fits inside the canvas.
#    - Every visible block is fully non-background, so the motif bbox is exactly 9x9 and the
#      solver's bbox-based block partition is stable.
# e) Visual observability:
#    - Humans can see the four code pixels directly: they are the corner cells of the four corner
#      blocks that point into the hole.
#    - The center output visibly reuses those colors, but two-color and three-color examples show
#      different fill stencils. Because the only global difference is foreground color count, the
#      branch condition is inferable from a small set of examples.
#    - Edge-facing pixels near the hole are deliberately made unhelpful for the original rule, so
#      the new center cannot be explained away as ordinary inner-boundary sampling.
# f) Visual cleanliness:
#    - Only one semantic layer occupies the center: the decoded symbol. No overlaps or secondary
#      markers are added anywhere else.
#    - Blocks are mostly majority-colored with a few accents, which keeps the corner code pixels
#      easy to spot and keeps the global color count easy to read by eye.
# g) Color-change inferability:
#    - No hidden remapping by color id is used. Colors are only counted, then reused directly in
#      the decoded center.
#    - The generator rejects samples where the fused center accidentally matches either the original
#      boundary-sampling center or the alternate decoder for the same 2x2 symbol, preventing silent
#      collapse of the count-controlled branch.
# h) Quantitative anti-degeneration metric used during validation:
#    - For each generated sample we record:
#        * `legacy_equal`: whether the fused center equals the original task's center patch.
#        * `alternate_equal`: whether the fused center equals the other count-mode decoder applied
#          to the same 2x2 corner symbol.
#        * `mode`: the actual foreground color count (2 or 3).
#      Valid generation requires both equality flags to stay false.
# i) New solver assumptions enforced by the generator:
#    - Inputs use exactly 2 or exactly 3 non-background colors.
#    - The four inward-facing corner cells are the intended 2x2 symbol and remain visible.
#    - In three-color samples, the majority-colored cross differs from the edge-facing pixels, so
#      the old boundary rule cannot accidentally reproduce the fused output.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
BG = 8

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
FOREGROUND_COUNT_CHOICES = (2, 3)
MAX_RETRIES = 400

ALL_BLOCK_SLOTS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)

CORNER_CODE_POS = {
    (0, 0): (2, 2),
    (0, 2): (2, 0),
    (2, 0): (0, 2),
    (2, 2): (0, 0),
}

EDGE_HOLE_POS = {
    (0, 1): (2, 1),
    (1, 0): (1, 2),
    (1, 2): (1, 0),
    (2, 1): (0, 1),
}

EXTRA_GUARANTEE_POS = {
    (0, 0): (0, 1),
    (0, 1): (0, 0),
    (0, 2): (1, 2),
    (1, 0): (0, 0),
    (1, 2): (0, 2),
    (2, 0): (1, 0),
    (2, 1): (2, 2),
    (2, 2): (2, 1),
}

_INNER_BOUNDARY_ROWS = np.array([BLOCK - 1, BLOCK + 1, BLOCK + BLOCK], dtype=int)
_INNER_BOUNDARY_COLS = np.array([BLOCK - 1, BLOCK + 1, BLOCK + BLOCK], dtype=int)


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _foreground_colors(g: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(g[g != int(bg)]))


def _decode_two_color(symbol: np.ndarray) -> np.ndarray:
    idx = np.array([0, 0, 1], dtype=int)
    return symbol[np.ix_(idx, idx)].copy()


def _decode_three_color(symbol: np.ndarray, majority: int) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), int(majority), dtype=int)
    patch[0, 0] = int(symbol[0, 0])
    patch[0, 2] = int(symbol[0, 1])
    patch[2, 0] = int(symbol[1, 0])
    patch[2, 2] = int(symbol[1, 1])
    return patch


def _decode_symbol(symbol: np.ndarray, majority: int, color_count: int) -> np.ndarray:
    if color_count == 2:
        return _decode_two_color(symbol)
    if color_count == 3:
        return _decode_three_color(symbol, majority)
    raise AssertionError("expected exactly 2 or 3 foreground colors")


def _corner_symbol(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    return np.array(
        [
            [g[int(r0) + 2, int(c0) + 2], g[int(r0) + 2, int(c0) + 2 * block]],
            [g[int(r0) + 2 * block, int(c0) + 2], g[int(r0) + 2 * block, int(c0) + 2 * block]],
        ],
        dtype=int,
    )


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    symbol = _corner_symbol(g, r0, c0, block)
    majority = _major_foreground_color(g, bg)
    color_count = len(_foreground_colors(g, bg))
    return _decode_symbol(symbol, majority, color_count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the visible 3x3 block motif and decode its missing center from corners plus color count.

    Input structure:
    - One 9x9 foreground motif sits on a teal background.
    - The motif is a 3x3 array of 3x3 blocks whose middle block is empty.
    - The four corner blocks contain one inward-facing code pixel each; together they form a 2x2
      symbol, while the total number of foreground colors in the motif is either 2 or 3.

    Rule:
    - Keep the original 9x9 motif in place.
    - Copy each of the eight visible 3x3 blocks to the opposite outer slot of the implied 5x5
      block grid, without rotating the copied block.
    - Fill the center from the 2x2 corner symbol, but choose the decoding stencil by color count:
      2 colors means nearest-neighbor enlargement, and 3 colors means the symbol occupies only the
      four center-block corners while the remaining five cells become the motif's majority color.

    Key insight:
    - The same four corner code pixels support two different reconstructions. The solver must first
      recognize that the foreground color count selects the decoding mode, then apply that mode to
      the 2x2 symbol before performing the unchanged outer block expansion.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0, "motif must be a square 3x3 block grid"
    block = h // 3

    out = g.copy()
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = crop_bbox(
                g,
                (
                    int(r0) + br * block,
                    int(c0) + bc * block,
                    int(r0) + (br + 1) * block - 1,
                    int(c0) + (bc + 1) * block - 1,
                ),
            )
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, src, dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


def _legacy_center_patch_from_motif(motif: np.ndarray, bg: int) -> np.ndarray:
    majority = _major_foreground_color(motif, bg)
    patch = motif[np.ix_(_INNER_BOUNDARY_ROWS, _INNER_BOUNDARY_COLS)].copy()
    patch[1, 1] = majority
    return patch


def _sample_symbol(colors: list[int], majority: int, color_count: int) -> np.ndarray:
    while True:
        vals = [random.choice(colors) for _ in range(4)]
        symbol = np.array(vals, dtype=int).reshape(2, 2)
        if color_count == 2:
            if len(set(vals)) < 2:
                continue
            if vals.count(majority) == 0 or vals.count(majority) == 4:
                continue
            if sum(v != majority for v in vals) < 2:
                continue
        else:
            if len(set(vals)) < 3:
                continue
            if not all(color in vals for color in colors):
                continue
        return symbol


def _make_block(
    majority: int,
    accent_colors: list[int],
    forced: dict[tuple[int, int], int],
    random_accent_range: tuple[int, int] = (1, 3),
) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), int(majority), dtype=int)
    for (r, c), color in forced.items():
        block[r, c] = int(color)

    if accent_colors:
        available = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if (r, c) not in forced]
        random.shuffle(available)
        lo, hi = random_accent_range
        extra_count = min(len(available), random.randint(lo, hi))
        for r, c in available[:extra_count]:
            block[r, c] = int(random.choice(accent_colors))
    return block


def _assemble_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _expanded_output(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _forced_color_maps(
    colors: list[int],
    majority: int,
    symbol: np.ndarray,
    color_count: int,
) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    forced = {slot: {} for slot in ALL_BLOCK_SLOTS}

    symbol_slots = {
        (0, 0): int(symbol[0, 0]),
        (0, 2): int(symbol[0, 1]),
        (2, 0): int(symbol[1, 0]),
        (2, 2): int(symbol[1, 1]),
    }
    for slot, code_color in symbol_slots.items():
        forced[slot][CORNER_CODE_POS[slot]] = int(code_color)

    if color_count == 2:
        edge_color = random.choice(colors)
    else:
        edge_color = random.choice([c for c in colors if c != majority])
    for slot, pos in EDGE_HOLE_POS.items():
        forced[slot][pos] = int(edge_color)

    guarantee_slots = list(ALL_BLOCK_SLOTS)
    random.shuffle(guarantee_slots)
    idx = 0
    for color in colors[1:]:
        for _ in range(2):
            slot = guarantee_slots[idx % len(guarantee_slots)]
            idx += 1
            forced[slot][EXTRA_GUARANTEE_POS[slot]] = int(color)
    return forced


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    motif = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    symbol = _corner_symbol(g, int(r0), int(c0), BLOCK)
    color_count = len(_foreground_colors(motif, bg))
    majority = _major_foreground_color(motif, bg)
    fused = _decode_symbol(symbol, majority, color_count)
    legacy = _legacy_center_patch_from_motif(motif, bg)
    alternate = _decode_three_color(symbol, majority) if color_count == 2 else _decode_two_color(symbol)
    return {
        "mode": int(color_count),
        "legacy_equal": int(np.array_equal(fused, legacy)),
        "alternate_equal": int(np.array_equal(fused, alternate)),
        "center_vs_legacy_hamming": int(np.count_nonzero(fused != legacy)),
        "center_vs_alternate_hamming": int(np.count_nonzero(fused != alternate)),
        "symbol_distinct_colors": int(len(set(int(v) for v in symbol.flatten()))),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.choice(FOREGROUND_COUNT_CHOICES)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = int(colors[0])
    symbol = _sample_symbol(colors, majority, color_count)
    center = _decode_symbol(symbol, majority, color_count)
    alternate = _decode_three_color(symbol, majority) if color_count == 2 else _decode_two_color(symbol)
    if np.array_equal(center, alternate):
        return None

    forced_maps = _forced_color_maps(colors, majority, symbol, color_count)
    accent_colors = [int(c) for c in colors if int(c) != majority]
    blocks = {
        slot: _make_block(majority, accent_colors, forced_maps[slot])
        for slot in ALL_BLOCK_SLOTS
    }
    motif = _assemble_motif(blocks)

    if len(_foreground_colors(motif, BG)) != color_count:
        return None
    if _major_foreground_color(motif, BG) != majority:
        return None

    legacy = _legacy_center_patch_from_motif(motif, BG)
    if np.array_equal(center, legacy):
        return None

    expanded = _expanded_output(motif, center)
    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    metrics = variant_metrics(inp)
    if metrics["legacy_equal"] or metrics["alternate_equal"]:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
