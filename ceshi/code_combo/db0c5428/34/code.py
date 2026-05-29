# Combo variant of db0c5428 (index 34)
# combo_spec: D4/1+D13/2
# Name: 无间隙紧邻放置+翻转复制

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - The input stays a single 9x9 motif made of eight visible 3x3 blocks around
#      an empty 3x3 center hole. The hole is still reconstructed from the eight
#      inner-boundary pixels exactly as in the base task.
#    - The output no longer uses the sparse checkerboard-like outer ring. Instead
#      the eight copied blocks are packed so they touch the original motif by an
#      edge, yielding a connected 15x15 silhouette whose block-row occupancies are
#      3 / 3 / 5 / 3 / 3.
#    - Point-symmetric copying is preserved at the block level, but each pasted
#      outer copy is vertically flipped before placement.
#    - Small trace with one source block:
#        source        = [[1,2,3],[4,5,6],[7,8,9]]
#        flipud(source)= [[7,8,9],[4,5,6],[1,2,3]]
#      If that source is the top-middle input block, the combo pastes
#      flipud(source) directly below the bottom-middle original block, so the
#      shared border is between [7,8,9] and the bottom block's last row. Under
#      the base task there would be no touching border there; under D4-only there
#      would still be a gap-pattern layout. The combo makes the flip visible at
#      the contact seam itself.
# b) Complexity calibration:
#    - The new layout adds one geometric idea only: compress the sparse outer ring
#      into a connected band around the motif. The copy mapping and center-decoding
#      logic stay local and repetitive, so humans can infer the rule from a few
#      examples without extra bookkeeping.
#    - To keep the flip visually meaningful, generation rejects seed blocks that
#      are too close to their own vertical flips and also rejects motifs whose
#      touching boundaries would look almost the same with or without the flip.
# c) Implicit-assumption audit for main():
#    - `bbox(g != bg)` still assumes exactly one isolated motif on a blank canvas;
#      the generator keeps that invariant.
#    - The bbox is still 9x9 and divisible into 3x3 blocks, so the block-size
#      inference remains valid.
#    - The center hole is still at the same coordinates, so `_center_patch(...)`
#      does not need a new decoding rule.
#    - Outer copies must be read from the original input rather than from `out`,
#      because the no-gap layout places new copies immediately next to the motif
#      and later writes must not contaminate future source reads.
#    - The solver now assumes the output footprint is the connected compressed
#      layout described by `_OUTER_SPECS`; the generator constructs exactly that
#      footprint and leaves one full 3-pixel block of canvas margin so it fits.
# d) Placement legality:
#    - The combo output still fits inside the same 15x15 bbox as the base task:
#      rows [r0-3, r0+11] and cols [c0-3, c0+11].
#    - The generator therefore keeps the same padding constraint `r0,c0 >= 3`
#      and `r0+12,c0+12 <= N`; this is sufficient even after compressing gaps.
#    - All destination slots in `_OUTER_SPECS` are disjoint, so no pasted blocks
#      overlap each other or the original 9x9 motif.
# e) Visual observability:
#    - The human-visible cue is the seam between a pasted outer block and the
#      neighboring original block. Because the copies are now edge-adjacent, the
#      top/bottom row reversal created by `flipud` can be seen directly where the
#      two blocks touch, rather than only by comparing distant copies.
#    - The generator enforces both a global flip-difference score and a boundary
#      continuity delta so the combo does not collapse into a nearly-base sample.
# f) Visual cleanliness:
#    - The output remains a single clean connected figure with no overlaps and no
#      extra markers. The compressed layout is symmetric and easy to parse: three
#      compact rows above/below and a single five-block middle row.
#    - No color remapping is introduced, so the viewer only has to parse spatial
#      structure and reflection, not hidden palette logic.
# g) Change inferability:
#    - Colors retain their original roles. The only nontrivial change is spatial:
#      where the outer blocks sit and how they are mirrored. Repeated seams across
#      the top, bottom, left, and right provide enough evidence to infer both the
#      no-gap placement and the vertical flip from a small number of examples.


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
    """Reflect a copied block across the horizontal axis."""
    return np.flipud(np.asarray(block, dtype=int))


_OUTER_SPECS = (
    ((0, 0), (3, 2), (2, 2), "below"),
    ((0, 1), (3, 1), (2, 1), "below"),
    ((0, 2), (3, 0), (2, 0), "below"),
    ((1, 0), (1, 3), (1, 2), "right"),
    ((1, 2), (1, -1), (1, 0), "left"),
    ((2, 0), (-1, 2), (0, 2), "above"),
    ((2, 1), (-1, 1), (0, 1), "above"),
    ((2, 2), (-1, 0), (0, 0), "above"),
)


def _iter_outer_specs():
    for spec in _OUTER_SPECS:
        yield spec


def _extract_block(motif: np.ndarray, br: int, bc: int) -> np.ndarray:
    return motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]


def _edge_match_count(outer_block: np.ndarray, adjacent_block: np.ndarray, side: str) -> int:
    if side == "above":
        return int(np.count_nonzero(outer_block[-1, :] == adjacent_block[0, :]))
    if side == "below":
        return int(np.count_nonzero(outer_block[0, :] == adjacent_block[-1, :]))
    if side == "left":
        return int(np.count_nonzero(outer_block[:, -1] == adjacent_block[:, 0]))
    if side == "right":
        return int(np.count_nonzero(outer_block[:, 0] == adjacent_block[:, -1]))
    raise ValueError(f"unknown side: {side}")


def _flipud_distance(block: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(block, dtype=int) != np.flipud(block)))


def _variant_difference_score(motif: np.ndarray) -> int:
    """Count changed pixels between copied blocks with and without the combo flip."""
    score = 0
    for (src_br, src_bc), _dst, _adj, _side in _iter_outer_specs():
        src = _extract_block(motif, src_br, src_bc)
        score += int(np.count_nonzero(src != _outer_transform(src)))
    return score


def _boundary_continuity_delta(motif: np.ndarray) -> int:
    """
    Quantify the combo-specific seam signal:
    how much the number of matching edge pixels changes when the pasted copy is
    flipped versus left unchanged.
    """
    delta = 0
    for (src_br, src_bc), _dst, (adj_br, adj_bc), side in _iter_outer_specs():
        src = _extract_block(motif, src_br, src_bc)
        adj = _extract_block(motif, adj_br, adj_bc)
        flipped = _outer_transform(src)
        delta += abs(_edge_match_count(flipped, adj, side) - _edge_match_count(src, adj, side))
    return int(delta)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a connected 15x15 layout with flipped outer copies.

    Input structure:
    - A single 9x9 foreground motif sits on a teal background.
    - The motif is a 3x3 arrangement of 3x3 blocks whose middle block is empty.
    - The eight pixels facing the hole still encode the missing center block border.

    Rule:
    - Keep the original motif in place and reconstruct the missing center block
      from the eight inner-boundary pixels, using the majority foreground color
      at the center of that 3x3 patch.
    - Copy each non-center block to its point-symmetric outer location, but pack
      those outer copies so they touch the original motif with no blank block gap.
    - Before each outer paste, flip the copied 3x3 block top-to-bottom.

    Key insight: The hole decoding is unchanged, but the outer ring is no longer
    a sparse checkerboard. The copied blocks are compressed until they share edges
    with the motif, so the vertical flip becomes readable from the seam patterns
    where the new copies meet the original blocks.
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
    for (src_br, src_bc), (dst_br, dst_bc), _adj, _side in _iter_outer_specs():
        src = crop_bbox(
            g,
            (
                int(r0) + src_br * block,
                int(c0) + src_bc * block,
                int(r0) + (src_br + 1) * block - 1,
                int(c0) + (src_bc + 1) * block - 1,
            ),
        )
        paste(out, _outer_transform(src), int(r0) + dst_br * block, int(c0) + dst_bc * block)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (o5x, D4/1 + D13/2 combo) ---------------

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
MAX_RETRIES = 400
MIN_SEED_FLIP_DIFF = 2
MIN_VARIANT_SCORE = 16
MIN_BOUNDARY_DELTA = 6

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
    """Assemble the 9x9 input motif; only the outer-output layout changes."""
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
    """Build the connected 15x15 combo output region."""
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for (src_br, src_bc), (dst_br, dst_bc), _adj, _side in _iter_outer_specs():
        src = _extract_block(motif, src_br, src_bc)
        paste(expanded, _outer_transform(src), (dst_br + 1) * BLOCK, (dst_bc + 1) * BLOCK)
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
    if _boundary_continuity_delta(motif) < MIN_BOUNDARY_DELTA:
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
    raise AssertionError("failed to generate a valid db0c5428 combo input")
