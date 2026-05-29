# Variant of db0c5428
# Dimension: D5 — 中心块的重建机制 (transform)
# Change: 中心块的8个非中心像素取自8个相邻块的朝向像素，中心像素取多数前景色
# ... → 中心块模式由四个角块朝向中心的角像素编码成 2x2 符号，再解码为 3x3 图案

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis for the D5 variant
# a) Concretization and behavior trace:
#    - The visible object is still one 9x9 motif made of eight 3x3 blocks around an empty
#      3x3 center, and the output still expands it to a 15x15 five-by-five block layout.
#    - The changed part is only the missing center block. Instead of reading all eight
#      hole-facing pixels, the solver now reads exactly four pixels:
#      TL corner block's bottom-right cell, TR's bottom-left, BL's top-right, BR's top-left.
#      Those four cells form a 2x2 symbol. That symbol is then enlarged to 3x3 with a fixed
#      nearest-neighbor rule, yielding:
#          a a b
#          a a b
#          c c d
#      for symbol [[a, b], [c, d]].
#    - Concrete trace on one motif:
#      original logic sampled rows/cols [2,4,6] around the hole and could use edge blocks.
#      new logic ignores edge-facing samples entirely, reads only the four corner code pixels,
#      builds a 2x2 code, and expands it. The difference is that the center is now controlled
#      by corner symbols rather than by the whole inner boundary.
# b) Complexity calibration:
#    - The code alphabet is still just ARC colors already present in the motif.
#    - There is only one decoding rule across all samples: 2x2 symbol -> 3x3 nearest-neighbor
#      enlargement. This avoids too many branches while still creating visibly new behavior.
#    - The generator forces at least two non-majority code cells so the center is not a trivial
#      solid square.
# c) Implicit-assumption review of main():
#    - The original _center_patch assumed the eight boundary sample locations mattered and that
#      the center pixel should be overwritten by the majority color. That assumption is invalid
#      now, so the helper is replaced completely.
#    - The bbox/3x3-block parsing logic in main() still holds because the motif remains a single
#      contiguous 9x9 foreground bbox with a 3x3 empty middle.
#    - The outer-copy loop still holds because block placement and output geometry are unchanged.
# d) Placement legality:
#    - The motif is still pasted with at least one block of padding on every side needed by the
#      15x15 expansion, so no copied block can go out of bounds.
#    - All eight visible blocks are fully foreground-colored, so the motif bbox is exactly 9x9
#      and the solver's bbox extraction is stable.
# e) Visual observability:
#    - The four inward-facing corner pixels directly reappear as the four quadrants of the filled
#      center block after enlargement, which is visible in a single sample without needing hidden
#      metadata such as color ids or ordering.
#    - Edge-facing pixels near the hole are forced to the majority color so the new center is
#      visibly not explained by the original boundary-sampling rule.
# f) Visual cleanliness:
#    - Only one layer occupies the center: the decoded symbol. The surrounding eight blocks are
#      unchanged except for outward copies, so there is no ambiguous overlap.
#    - Blocks are mostly majority-colored with a few accents, keeping the encoded corner pixels
#      legible instead of dissolving into noise.
# g) Color-change inferability:
#    - No hidden color remapping is used. The center reuses the exact colors seen in the four
#      corner code pixels.
#    - The generator rejects samples where the new center accidentally matches the old boundary
#      reconstruction, preventing silent regression to the base task.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _decode_symbol(symbol: np.ndarray) -> np.ndarray:
    """Expand a 2x2 symbol into a 3x3 patch by nearest-neighbor scaling."""
    idx = np.array([0, 0, 1], dtype=int)
    return symbol[np.ix_(idx, idx)].copy()


def _corner_symbol(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    return np.array(
        [
            [g[int(r0) + 2, int(c0) + 2], g[int(r0) + 2, int(c0) + 2 * block]],
            [g[int(r0) + 2 * block, int(c0) + 2], g[int(r0) + 2 * block, int(c0) + 2 * block]],
        ],
        dtype=int,
    )


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    return _decode_symbol(_corner_symbol(g, r0, c0, block))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 block arrangement around its empty center.

    Input structure:
    - One 9x9 foreground motif sits on a teal background.
    - The motif is a 3x3 array of 3x3 blocks, and the middle block is entirely empty.
    - The four corner blocks each carry one special corner pixel pointing toward the hole;
      together those four pixels form a compact symbol for the missing center.

    Rule:
    - Preserve the original 9x9 motif in place.
    - Copy each of the eight non-center blocks to the opposite outer slot of the implied
      5x5 block grid, without rotating any copied block.
    - Rebuild the center block by reading the four inward-facing corner pixels, arranging
      them as a 2x2 symbol, and enlarging that symbol to 3x3 with nearest-neighbor scaling.

    Key insight:
    - The empty center is no longer described by the full inner boundary. It is encoded
      only by the four corner blocks, so the task is to decode a tiny corner symbol and
      place the enlarged result back into the middle before performing the outer expansion.
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

    paste(out, _center_patch(g, int(r0), int(c0), block), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (o5x) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_PAD = 3

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

# -- CFG: generation --
MAX_RETRIES = 300

_CORNER_CODE_POS = {
    (0, 0): (2, 2),
    (0, 2): (2, 0),
    (2, 0): (0, 2),
    (2, 2): (0, 0),
}

_EDGE_HOLE_POS = {
    (0, 1): (2, 1),
    (1, 0): (1, 2),
    (1, 2): (1, 0),
    (2, 1): (0, 1),
}


def _sample_symbol(colors: list[int], majority: int) -> np.ndarray:
    """Choose a visible 2x2 code with at least two distinct colors and two
    non-majority cells so the decoded center is not trivial."""
    while True:
        vals = [random.choice(colors) for _ in range(4)]
        if len(set(vals)) < 2:
            continue
        if sum(v != majority for v in vals) < 2:
            continue
        return np.array(vals, dtype=int).reshape(2, 2)


def _make_block(majority: int, colors: list[int], forced: dict[tuple[int, int], int]) -> np.ndarray:
    """Create a 3x3 block that is mostly majority color, with forced cells preserved."""
    block = np.full((BLOCK, BLOCK), majority, dtype=int)
    for (r, c), color in forced.items():
        block[r, c] = color

    accent_colors = [c for c in colors if c != majority]
    available = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if (r, c) not in forced]
    random.shuffle(available)
    accent_count = min(len(available), random.randint(2, 4))
    for r, c in available[:accent_count]:
        block[r, c] = random.choice(accent_colors)
    return block


def _assemble_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _gen_center_patch(symbol: np.ndarray) -> np.ndarray:
    return _decode_symbol(symbol)


def _old_style_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = np.array([BLOCK - 1, BLOCK + 1, BLOCK + BLOCK], dtype=int)
    cols = np.array([BLOCK - 1, BLOCK + 1, BLOCK + BLOCK], dtype=int)
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


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


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    symbol = _sample_symbol(colors, majority)

    blocks: dict[tuple[int, int], np.ndarray] = {}
    symbol_slots = {
        (0, 0): int(symbol[0, 0]),
        (0, 2): int(symbol[0, 1]),
        (2, 0): int(symbol[1, 0]),
        (2, 2): int(symbol[1, 1]),
    }

    for slot, code_color in symbol_slots.items():
        forced = {_CORNER_CODE_POS[slot]: code_color}
        blocks[slot] = _make_block(majority, colors, forced)

    for slot, hole_color in _EDGE_HOLE_POS.items():
        forced = {hole_color: majority}
        blocks[slot] = _make_block(majority, colors, forced)

    motif = _assemble_motif(blocks)
    center = _gen_center_patch(symbol)

    old_majority = _major_foreground_color(motif, BG)
    if np.array_equal(center, _old_style_center_patch(motif, old_majority)):
        return None

    expanded = _expanded_output(motif, center)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 input")
