# Variant of db0c5428
# Dimension: D7 — 输出网格的整体形状 (grid_shape)
# Change: 输出与输入同尺寸（same），外扩块在网格范围内原地放置，超出部分被截断... → 输出仅外扩部分（去除原始图案）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D7 variant
# ----------------------------------
# a) Concrete behavior and trace
# - Input stays identical to the original task: one 9x9 motif made of a 3x3 grid
#   of 3x3 blocks, with the middle 3x3 block missing.
# - The original solver produced a same-size canvas containing a 15x15 expanded
#   arrangement around that motif. In block coordinates the filled locations were:
#     X . X . X
#     . X X X .
#     X X X X X
#     . X X X .
#     X . X . X
# - The D7 variant keeps the copy/reconstruction logic but crops away the original
#   8 visible input blocks. The output is therefore the 15x15 "new content only"
#   region with occupancy:
#     X . X . X
#     . . . . .
#     X . X . X
#     . . . . .
#     X . X . X
#   where the center X is the reconstructed 3x3 patch and the other 8 X slots are
#   the opposite-side copies of input blocks.
# - Small trace: if the input top-left corner block b00 originally also appears in
#   the output at outer slot (4,4), the old task kept both the input copy of b00
#   and that new outer copy. The variant keeps only the outer copy; the original
#   b00 position becomes background because it belonged to the removed source motif.
#
# b) Complexity calibration
# - The deduction burden is still the same core idea: infer the 5x5 block lattice,
#   mirror each surrounding block to the opposite outer slot, and reconstruct the
#   center from the eight boundary pixels.
# - Cropping is visually obvious because the output shape changes from N x N to a
#   fixed 15x15 and exactly eight 3x3 blocks disappear from the middle ring.
# - No extra branch logic is added, so the task is not simplified into a trivial
#   crop-only puzzle; the solver still needs the original reconstruction rule.
#
# c) Hidden assumptions audit for the original main()
# - `out = g.copy()` assumed the answer lives on the original canvas. This is false
#   now; we must create a fresh 15x15 background canvas.
# - `expanded_r0/c0 = r0/c0 - block` assumed output coordinates are expressed in
#   the input canvas. In the variant, output coordinates are local to the cropped
#   15x15 region, so destination slots must be computed from block indices alone.
# - Pasting the reconstructed center back at `(r0 + block, c0 + block)` assumed the
#   center stays inside the original motif. Now it belongs at the middle slot of
#   the cropped output, namely `(2*block, 2*block)`.
# - Motif detection via bbox and center reconstruction from the hole boundary still
#   remain valid, so those earlier lines are intentionally unchanged.
#
# d) Placement legality
# - Input generation keeps the original placement rule `r0,c0 >= 3`, even though
#   the cropped output no longer needs margin. This avoids introducing an unrelated
#   placement variation and preserves the original motif appearance.
# - Since the output is a standalone 15x15 canvas, no clipping or overlap issues
#   remain during solving. Generator verification checks exact equality with main().
#
# e) Visual observability
# - Humans can see the change from a few examples: outputs are all 15x15, contain a
#   sparse 3x3 checkerboard of blocks, and omit the original middle-ring blocks.
# - The removed blocks are not chosen by color id or any hidden ordering; they are
#   precisely the blocks that were already present in the input motif.
#
# f) Visual cleanliness
# - The output has only nine 3x3 blocks separated by background gaps, so regions do
#   not overlap and the reconstructed center remains easy to distinguish.
#
# g) Color inference
# - Colors are unchanged from the original design. The variant does not introduce a
#   new color mapping, so the original structural cues remain sufficient.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)

    hole_r = int(r0) + block
    hole_c = int(c0) + block
    offsets = [-1, 1, 3]
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[hole_r + offsets[r], hole_c + offsets[c]])
    return patch


def _outer_only_output(motif: np.ndarray, center: np.ndarray, bg: int, block: int) -> np.ndarray:
    out = np.full((block * 5, block * 5), int(bg), dtype=int)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            paste(out, src, (4 - 2 * br) * block, (4 - 2 * bc) * block)
    paste(out, center, 2 * block, 2 * block)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center and return only the newly created outer pattern.

    Input structure:
    - A teal background contains one 9x9 motif arranged as a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is absent, leaving a teal hole at the motif center.
    - The eight pixels touching that hole encode the border of the missing center block.

    Rule:
    - Reconstruct the center 3x3 block from the hole boundary; its middle cell is the
      motif's majority foreground color.
    - Copy each of the eight visible input blocks to the opposite outer slot of the
      implied 5x5 block lattice, without rotating the copied block.
    - Output only those newly generated blocks: the eight outer copies plus the rebuilt
      center block. Do not include the original eight input blocks in the answer.

    Key insight: The input still describes a larger symmetric 5x5 block arrangement, but
    the answer is the cropped set of blocks that were missing from the input rather than
    the input motif itself.
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

    motif = crop_bbox(g, motif_bbox)
    center = _center_patch(g, int(r0), int(c0), block, bg)
    return _outer_only_output(motif, center, bg, block)


# --------------- generator (o5x, D7 crop variant) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
OUTPUT_SIZE = BLOCK * 5
MOTIF_PAD = 3

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

# -- CFG: generation --
MAX_RETRIES = 300

# Diagonal-symmetric orbits for the corner seed block.
DIAG_ORBITS = (
    ((0, 0),),
    ((1, 1),),
    ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)

# Left-right symmetric orbits for the edge seed block.
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)

_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_orbits(orbits, forced_minor, target_count):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits) if all(coord not in forced_minor for coord in orbit)
    ]
    choices = []
    for mask in range(1 << len(usable)):
        selected = set()
        total = 0
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                selected.add(idx)
                total += len(orbits[idx])
        if total == target_count:
            choices.append(selected)
    assert choices, "unreachable symmetric majority count"
    return random.choice(choices)


def _seed_block(orbits, forced_minor, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_minor, target_count)
    minor_orbits = []
    for idx, orbit in enumerate(orbits):
        if idx in major_idxs:
            for r, c in orbit:
                block[r, c] = majority
        else:
            minor_orbits.append((block, orbit))
    return block, minor_orbits


def _assign_minor_colors(minor_orbits, colors):
    random.shuffle(minor_orbits)
    guaranteed = list(colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))
    ]
    for (block, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            block[r, c] = color


def _make_blocks(majority, minor_colors, corner_major, edge_major):
    b00, corner_minors = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
    b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    return b00, b01


def _assemble_motif(b00, b01):
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): np.fliplr(b00),
        (1, 0): b01.T,
        (1, 2): np.rot90(b01, -1),
        (2, 0): np.flipud(b00),
        (2, 1): np.flipud(b01),
        (2, 2): np.rot90(b00, 2),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _gen_center_patch(motif, majority):
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    possible_totals = [
        total
        for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
        if 18 - total >= len(minor_colors)
    ]
    majority_total = random.choice(possible_totals)
    pairs = [
        (corner, majority_total - corner)
        for corner in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1)
        if EDGE_MAJOR_MIN <= majority_total - corner <= EDGE_MAJOR_MAX
    ]
    corner_major, edge_major = random.choice(pairs)

    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    expected = _outer_only_output(motif, center, BG, BLOCK)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D7 input")
