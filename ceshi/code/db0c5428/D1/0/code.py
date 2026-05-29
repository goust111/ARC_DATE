# Variant of db0c5428
# Dimension: D1 — 块到外部槽位的映射规则 (match_cond)
# Change: 每个非中心3x3块被复制到以中心为对称点的对侧外部槽位，即block(r,c)映射到5x5网格的(4-2r, 4-2c)... → 旋转90度映射而非点对称映射

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D1 variant
# ----------------------------------
# variant_spec:
#   concrete_behavior:
#     - Input still contains one 9x9 motif made of eight visible 3x3 blocks around
#       an empty 3x3 center hole on background color 8.
#     - The center 3x3 block is reconstructed exactly as in the original task:
#       the 8 pixels touching the hole become the border of the missing block,
#       and the middle pixel becomes the majority foreground color.
#     - The only rule change is the destination of each non-center block in the
#       surrounding 5x5 block layout. Instead of sending block (br,bc) to the
#       point-symmetric slot (4-2*br, 4-2*bc), we interpret (br,bc) relative to
#       the motif center and rotate that direction 90 degrees clockwise, then
#       stretch it to the outer ring. This gives the slot (2*bc, 4-2*br).
#     - Concrete trace: the top-left source block used to appear in the bottom-
#       right outer corner; now it appears in the top-right outer corner. The
#       left-middle source block used to appear at outer right-middle; now it
#       appears at outer top-middle.
#   complexity_calibration:
#     - The original generator uses highly symmetric seed blocks. That is mostly
#       fine here, but some rare seeds would make the clockwise layout too close
#       to the old point-symmetric one. The generator therefore rejects motifs
#       whose new outer ring differs too little from the old ring.
#     - We also reject corner/edge seeds that are accidentally invariant under
#       the specific transforms that separate the two mappings.
#   implicit_assumption_audit:
#     - `_center_patch(...)` depends only on the hole boundary, not on the outer
#       copy rule, so it can stay unchanged.
#     - `bbox(...)` and the inferred `block = h // 3` still assume a single 9x9
#       foreground motif with a square bounding box. The generator keeps that.
#     - The original `dst_r = 4 - 2*br`, `dst_c = 4 - 2*bc` hard-coded a point
#       reflection. In the variant that assumption is invalid and must be replaced
#       by the clockwise-rotation mapping.
#     - `paste(out, src, ...)` is still safe because every destination is a distinct
#       outer slot of the 5x5 block frame; no overlaps are introduced.
#   placement_legality:
#     - The 15x15 expanded output size is unchanged, so the original placement
#       margin of 3 cells around the 9x9 motif still guarantees no clipping when
#       the outer ring is added.
#   observability_and_cleanliness:
#     - The new mapping is visible only through spatial placement; colors are not
#       remapped and no hidden numeric ordering is used.
#     - By enforcing block patterns that are not too transform-invariant and by
#       measuring the pixel difference between the new outer ring and the original
#       point-symmetric outer ring, samples remain visually legible and the D1
#       change is recoverable from a few examples.


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


def _outer_slot_clockwise(br: int, bc: int) -> tuple[int, int]:
    """Map a 3x3 motif block to the 5x5 outer ring via a block-level CW rotation."""
    return 2 * int(bc), 4 - 2 * int(br)


def _outer_slot_point_symmetric(br: int, bc: int) -> tuple[int, int]:
    """Original task mapping, kept only for generator-side visibility checks."""
    return 4 - 2 * int(br), 4 - 2 * int(bc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a larger 5x5 block frame around its missing center.

    Input structure:
    - One 9x9 motif sits on teal background and is made from a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is absent, leaving a teal hole.
    - The 8 colored pixels that face this hole encode the border of the missing block.

    Rule:
    - Preserve the original 9x9 motif in place.
    - Rebuild the center 3x3 block from the hole boundary pixels, using the motif's
      majority foreground color for the center cell.
    - Copy each of the eight visible 3x3 blocks to the outer ring of the implied 5x5
      block grid, but send it to the slot obtained by rotating its block position
      90 degrees clockwise around the motif center.

    Key insight:
    The motif is still the middle of a larger block arrangement, but the outer copies
    are organized by a quarter-turn relation at block level rather than by point
    symmetry across the center.
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
            dst_br, dst_bc = _outer_slot_clockwise(br, bc)
            dst_r = expanded_r0 + dst_br * block
            dst_c = expanded_c0 + dst_bc * block
            paste(out, src, dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
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

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

# -- CFG: generation --
MAX_RETRIES = 300
VARIANT_DIFF_MIN = 12

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


def _expanded_output(motif, center, slot_fn):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            dst_br, dst_bc = slot_fn(br, bc)
            paste(expanded, src, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _variant_signal(motif, center) -> int:
    cw = _expanded_output(motif, center, _outer_slot_clockwise)
    old = _expanded_output(motif, center, _outer_slot_point_symmetric)
    return int(np.count_nonzero(cw != old))


def _blocks_show_rotation_signal(b00, b01) -> bool:
    return (
        np.count_nonzero(b00 != np.flipud(b00)) >= 2
        and np.count_nonzero(b00 != np.rot90(b00, 2)) >= 2
        and np.count_nonzero(b01 != b01.T) >= 2
        and np.count_nonzero(b01 != np.rot90(b01, -1)) >= 2
    )


def _gen_attempt() -> np.ndarray | None:
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

    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    if not _blocks_show_rotation_signal(b00, b01):
        return None

    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    if _variant_signal(motif, center) < VARIANT_DIFF_MIN:
        return None
    expanded = _expanded_output(motif, center, _outer_slot_clockwise)

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
    raise AssertionError("failed to generate a valid db0c5428 D1 variant input")
