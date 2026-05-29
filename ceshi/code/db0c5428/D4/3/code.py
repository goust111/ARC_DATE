# Variant of db0c5428
# Dimension: D4 — 块复制时的几何变换 (transform)
# Change: 复制到对侧外部槽位时不旋转不翻转，保持原始方向... → 角落块旋转、边缘块不旋转

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior:
#    The input is still one 9x9 motif made of eight visible 3x3 blocks around a
#    missing 3x3 center. The output is still a 15x15 block layout centered on the
#    original motif. The only rule change is in the eight outer copies:
#    - edge-source blocks: copied straight across with no transform
#    - corner-source blocks: copied straight across after a fixed 90-degree CW turn
#    Example trace at block level:
#    - original rule: top-left corner block b00 -> outer bottom-right slot as b00
#    - new rule:      top-left corner block b00 -> outer bottom-right slot as rot90CW(b00)
#    The center reconstruction still uses the 8 pixels immediately facing the hole.
# b) Complexity calibration:
#    The branch space is intentionally small: only two block types exist in every
#    sample (corner vs edge), and all four corners share one rotation rule. This is
#    enough to make the change visible without introducing many unrelated cases.
# c) Hidden assumptions checked against the old main():
#    - locating the single 9x9 motif by foreground bbox still holds
#    - block size is still 3, so center-patch reconstruction is unchanged
#    - the old outer-copy loop assumed raw pasting was always correct; that is the
#      only solver line whose semantic assumption changed
#    - center filling happens inside the original motif, so the new outer rotations
#      do not interfere with it
# d) Placement legality:
#    The generator still places the 9x9 motif with a 3-cell margin, so the 15x15
#    expanded region remains fully in-bounds and non-overlapping.
# e) Visual observability:
#    If a corner block were invariant under a 90-degree turn, the variant could
#    collapse back to the original task on that sample. The generator therefore
#    rejects motifs whose four corner instances would look unchanged after the new
#    corner-copy rotation.
# f) Visual cleanliness:
#    No extra objects are added. The same sparse 5x5 block scaffold is used, so
#    the changed behavior is isolated to the four outer corner copies.
# g) Color inference:
#    Colors keep the original role: one majority foreground color plus 1-3 minors.
#    No ordering or transformation depends on color ids; position type alone
#    determines whether a copied block rotates.


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


def _is_corner_block(br: int, bc: int) -> bool:
    return br != 1 and bc != 1


def _outer_copy_transform(block: np.ndarray, br: int, bc: int) -> np.ndarray:
    if _is_corner_block(br, bc):
        return np.rot90(block, -1)
    return block


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a larger 5x5 block frame with conditional rotation.

    Input structure:
    - A single 9x9 motif sits on a uniform background.
    - The motif is a 3x3 arrangement of 3x3 blocks with the middle block removed.
    - The eight pixels facing the hole encode the missing center block's border.

    Rule:
    - Preserve the original 9x9 motif where it is.
    - Rebuild the missing center block from the inner boundary pixels, using the
      motif's majority foreground color at the center.
    - Copy each visible block to the opposite outer slot of the implied 5x5 block
      layout; edge blocks keep their orientation, while corner blocks are rotated
      90 degrees clockwise before being pasted.

    Key insight:
    - The input already tells you two independent things: the center block comes
      from the ring around the hole, and the outer expansion depends on whether a
      source block is on a corner or on an edge of the 3x3 motif.
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
            paste(out, _outer_copy_transform(src, br, bc), dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (conditional-corner-rotation) ---------------

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


def _corner_rotation_visible(motif: np.ndarray) -> bool:
    for br, bc in ((0, 0), (0, 2), (2, 0), (2, 2)):
        blk = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
        if np.array_equal(blk, np.rot90(blk, -1)):
            return False
    return True


def _expanded_output(motif, center):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            paste(expanded, _outer_copy_transform(src, br, bc), (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


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

    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    motif = _assemble_motif(b00, b01)
    if not _corner_rotation_visible(motif):
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
    raise AssertionError("failed to generate a valid db0c5428 input")
