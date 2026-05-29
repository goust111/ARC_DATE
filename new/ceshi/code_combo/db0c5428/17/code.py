# Combo variant of db0c5428 (index 17)
# combo_spec: D4/1+D7/3
# Name: 水平翻转复制+分帧输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Input stays the same as the base task: one 9x9 motif built from eight
#      visible 3x3 blocks around an empty 3x3 center.
#    - The fused output has two side-by-side frames:
#        left  = repaired 9x9 motif only, with no extra transforms
#        right = full 15x15 expansion, where every outer copied block is
#                mirrored across the horizontal axis before pasting
#      with one solid divider column between them.
#    - Small trace on a single source block:
#        source           = [[1,2,3],[4,5,6],[7,8,9]]
#        left-frame usage = stays [[1,2,3],[4,5,6],[7,8,9]] inside the repaired motif
#        right outer copy =        [[7,8,9],[4,5,6],[1,2,3]]
#      The center patch is still reconstructed from the eight inner-boundary
#      pixels, so the fusion combines "same center-decoding" with "different
#      presentation + transformed outer expansion".
# b) Complexity calibration:
#    - The task still has only two stable outputs to infer from each input:
#      a repaired local motif and a larger expansion. This keeps the framing
#      change legible instead of introducing many panels or placements.
#    - The only geometric change inside the right frame is one fixed transform
#      family: flipud on every outer copy. There is no per-block branching.
#    - Generator rejects motifs whose blocks are too close to vertical-flip
#      symmetry, so the D4 change stays visible in the right frame.
# c) Implicit-assumption audit for main():
#    - `bbox(g != bg)` still assumes exactly one isolated motif on a plain
#      background; generator preserves that.
#    - The bbox is still 9x9 and decomposes into 3x3 blocks, so `block = h // 3`
#      remains valid.
#    - The base-task assumption `out = g.copy()` is no longer valid because the
#      output shape is now fixed by the framed layout, not by the input canvas.
#    - Outer copies must be read from the untouched input motif, not from a
#      partially written expanded canvas, otherwise flipped writes could pollute
#      later reads.
#    - The divider color must differ from the background; the generator also
#      keeps it out of the motif palette so the separator is visually explicit.
# d) Placement legality:
#    - Input placement remains the original safe regime: a 9x9 motif inside a
#      16..19 square grid with at least one full 3x3 block of margin.
#    - Output no longer depends on input coordinates. The left frame is centered
#      vertically in a 15-row canvas, the right frame exactly occupies 15x15,
#      and the divider is a dedicated column, so there is no overlap or clipping.
# e) Visual observability:
#    - Humans can compare the two frames directly: the left frame shows the
#      repaired original motif, while the right frame shows the larger pattern.
#    - Because the left frame preserves block orientation and the right frame
#      flips only the outer copies, two or three examples are enough to isolate
#      the per-frame rule difference.
# f) Visual cleanliness:
#    - Frames do not overlap; each occupies a dedicated rectangle separated by a
#      single full-height column.
#    - The left frame contains only the 9x9 repaired motif with teal padding,
#      keeping it visually distinct from the denser 15x15 right frame.
# g) Change inferability:
#    - No color remapping is introduced. Color identities stay stable, and the
#      new logic is carried entirely by frame placement and vertical reflection.
#    - Repeated transformed outer copies provide enough structure for a human to
#      infer the flip without relying on hidden numeric properties of colors.


DIVIDER_COLOR = 0


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
    return np.flipud(np.asarray(block, dtype=int))


def _filled_motif(motif: np.ndarray, center: np.ndarray, block: int) -> np.ndarray:
    out = np.asarray(motif, dtype=int).copy()
    paste(out, center, block, block)
    return out


def _expanded_output(motif: np.ndarray, center: np.ndarray, bg: int, block: int) -> np.ndarray:
    expanded = np.full((block * 5, block * 5), bg, dtype=int)
    output_motif = _filled_motif(motif, center, block)
    paste(expanded, output_motif, block, block)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block:(br + 1) * block, bc * block:(bc + 1) * block]
            paste(expanded, _outer_transform(src), (4 - 2 * br) * block, (4 - 2 * bc) * block)
    return expanded


def _framed_output(motif_with_center: np.ndarray, expanded: np.ndarray, bg: int) -> np.ndarray:
    frame_h = int(expanded.shape[0])
    left_h, left_w = map(int, motif_with_center.shape)
    right_w = int(expanded.shape[1])
    out = np.full((frame_h, left_w + 1 + right_w), bg, dtype=int)
    out[:, left_w] = DIVIDER_COLOR
    paste(out, motif_with_center, (frame_h - left_h) // 2, 0)
    paste(out, expanded, 0, left_w + 1)
    return out


def _flipud_distance(block: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(block, dtype=int) != np.flipud(block)))


def _variant_difference_score(motif: np.ndarray) -> int:
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
    Reconstruct the missing center, then show the repaired motif beside a flipped-copy expansion.

    Input structure:
    - A square teal canvas contains exactly one 9x9 motif built from a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is empty teal, while the other eight blocks remain visible.
    - The eight pixels that face the hole encode the border cells of the missing center block.

    Rule:
    - Recover the center block from those eight inner-boundary pixels and place the motif's majority foreground color at its center.
    - Produce a left frame containing only the repaired 9x9 motif, with its original block orientations unchanged.
    - Produce a right frame containing the full 5x5 block expansion, where each visible block is copied to the opposite outer slot and each such outer copy is flipped top-to-bottom before pasting; separate the two frames with one solid divider column.

    Key insight: The puzzle now asks for two synchronized outputs from the same motif: the local repair on the left and the larger extrapolation on the right, with the right frame alone applying the vertical reflection rule to the outer copies.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert DIVIDER_COLOR != bg, "divider color must differ from background"

    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0, "motif must be a square 3x3 block grid"
    block = h // 3

    motif = crop_bbox(g, motif_bbox)
    center = _center_patch(g, int(r0), int(c0), block, bg)
    motif_with_center = _filled_motif(motif, center, block)
    expanded = _expanded_output(motif, center, bg, block)
    return _framed_output(motif_with_center, expanded, bg)


# --------------- generator (combo: D4/1 + D7/3) ---------------

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


def _major_orbits(orbits, forced_minor, target_count):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits)
        if all(coord not in forced_minor for coord in orbit)
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
    assignments = guaranteed + [random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))]
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
    if _flipud_distance(b00) < MIN_SEED_FLIP_DIFF:
        return None
    if _flipud_distance(b01) < MIN_SEED_FLIP_DIFF:
        return None
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
        total for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
        if 18 - total >= len(minor_colors)
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
    motif_with_center = _filled_motif(motif, center, BLOCK)
    expanded = _expanded_output(motif, center, BG, BLOCK)
    expected = _framed_output(motif_with_center, expanded, BG)

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
    raise AssertionError("failed to generate a valid db0c5428 combo input")
