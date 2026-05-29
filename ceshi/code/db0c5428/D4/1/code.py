# Variant of db0c5428
# Dimension: D4 — 块复制时的几何变换 (transform)
# Change: 复制到对侧外部槽位时不旋转不翻转，保持原始方向... → 复制时沿水平轴翻转

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior and trace:
#    - The input still contains a single 9x9 motif made of eight visible 3x3
#      blocks around an empty 3x3 hole.
#    - The output still fills the hole from the eight inner-boundary pixels and
#      still writes the eight visible blocks into the opposite outer slots of an
#      implied 5x5 block lattice.
#    - The only rule change is the copied block orientation at those outer slots:
#      instead of pasting the source block unchanged, paste flipud(source).
#    - Small trace on one block:
#        source = [[1,2,3],[4,5,6],[7,8,9]]
#        original outer copy = [[1,2,3],[4,5,6],[7,8,9]]
#        variant outer copy  = [[7,8,9],[4,5,6],[1,2,3]]
#      The center reconstruction is unchanged.
# b) Complexity calibration:
#    - The source task 7ed72f31 uses a direct visible reflection operation with
#      no hidden bookkeeping; this variant should stay at that level.
#    - To avoid degenerate samples where flipud has no visual effect, generation
#      rejects seed blocks that are vertically symmetric.
# c) Implicit-assumption audit for main():
#    - bbox(g != bg) assumes exactly one isolated foreground motif; generator
#      places only one motif on a blank background.
#    - The 9x9 bbox and block=3 assumption remain valid; generator still builds
#      a 3x3 grid of 3x3 blocks.
#    - Outer copies are taken from the original input grid before any writes to
#      the output, so transformed pastes cannot contaminate later reads.
#    - Outer destination slots remain disjoint and fit inside the canvas because
#      the motif is padded by one full block on every side.
# d) Placement legality:
#    - Grid size, motif placement, and expanded 15x15 footprint are unchanged,
#      so the existing placement constraints are still sufficient.
# e/f) Visual observability and cleanliness:
#    - The motif keeps the same clean D4 layout; only the outer ring changes.
#    - Requiring both seed blocks to differ from their vertical flips makes the
#      new transform visible in the outer ring on nearly every sample.
# g) Color reasoning:
#    - No color remapping is introduced. Colors keep their original identities;
#      only geometry changes, so inference stays local and human-readable.


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


def _outer_transform(block: np.ndarray) -> np.ndarray:
    """Variant D4 transform: reflect a copied block across the horizontal axis."""
    return np.flipud(np.asarray(block, dtype=int))


def _variant_difference_score(motif: np.ndarray) -> int:
    """Count how many copied outer pixels would change vs the original task."""
    score = 0
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            score += int(np.count_nonzero(src != _outer_transform(src)))
    return score


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 block layout with flipped outer copies.

    Input structure:
    - A single 9x9 foreground motif sits on a teal background.
    - The motif is a 3x3 arrangement of 3x3 blocks whose middle block is empty.
    - The eight pixels facing the hole encode the missing center block border.

    Rule:
    - Keep the original motif where it is.
    - Rebuild the missing center block from those eight inner-boundary pixels,
      using the motif's majority foreground color at the center.
    - Copy each non-center 3x3 block to the opposite outer slot of the implied
      5x5 block grid, but flip the copied block top-to-bottom before pasting.

    Key insight: The task still uses the same block lattice and center decoding
    as the base puzzle; the only changed signal is that outer copies are mirror
    reflections across a horizontal axis rather than unchanged duplicates.
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
            paste(out, _outer_transform(src), dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (o5x, D4 horizontal-flip variant) ---------------

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

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

# -- CFG: generation --
MAX_RETRIES = 300
MIN_SEED_FLIP_DIFF = 2
MIN_VARIANT_SCORE = 16

# -- Symmetry orbits for 3x3 blocks --
DIAG_ORBITS = (
    ((0, 0),),
    ((1, 1),),
    ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)

_CENTER_OFFSETS = np.array([-1, 1, 3])


def _flipud_distance(block: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(block, dtype=int) != np.flipud(block)))


def _major_orbits(orbits, forced_minor, target_count):
    """Pick an orbit subset whose total size is target_count."""
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
    """Build one 3x3 seed block and return its still-uncolored minor orbits."""
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
    """Fill every minor orbit and ensure each chosen minor color appears."""
    random.shuffle(minor_orbits)
    guaranteed = list(colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))]
    for (block, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            block[r, c] = color


def _make_blocks(majority, minor_colors, corner_major, edge_major):
    """Build the two seed blocks and reject vertical-flip-degenerate cases."""
    b00, corner_minors = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
    b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    if _flipud_distance(b00) < MIN_SEED_FLIP_DIFF:
        return None
    if _flipud_distance(b01) < MIN_SEED_FLIP_DIFF:
        return None
    return b00, b01


def _assemble_motif(b00, b01):
    """Assemble the input 9x9 motif; only the outer-copy rule changes in this variant."""
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
    """Decode the missing center block from the eight inner-boundary pixels."""
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _expanded_output(motif, center):
    """Build the 15x15 output region with flipud-applied outer copies."""
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            paste(expanded, _outer_transform(src), (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build input, verify with the oracle, or reject."""
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    possible_totals = [
        total for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1) if 18 - total >= len(minor_colors)
    ]
    majority_total = random.choice(possible_totals)
    pairs = [
        (corner, majority_total - corner)
        for corner in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1)
        if EDGE_MAJOR_MIN <= majority_total - corner <= EDGE_MAJOR_MAX
    ]
    corner_major, edge_major = random.choice(pairs)

    blocks = _make_blocks(majority, minor_colors, corner_major, edge_major)
    if blocks is None:
        return None
    b00, b01 = blocks
    motif = _assemble_motif(b00, b01)
    if _variant_difference_score(motif) < MIN_VARIANT_SCORE:
        return None

    center = _gen_center_patch(motif, majority)
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
    raise AssertionError("failed to generate a valid db0c5428 D4 variant input")
