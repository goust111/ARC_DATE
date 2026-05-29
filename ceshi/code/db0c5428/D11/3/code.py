# Variant of db0c5428
# Dimension: D11 — 中心空洞的形状 (src)
# Change: 中心空洞恰好是一个3x3块，与其他块大小相同，位于3x3阵列正中... → 多个空洞（2-3个块缺失）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis for the D11 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The visible object is still a 9x9 motif made from a 3x3 array of 3x3 blocks.
#    - The original task always blanked only the center block. This variant always
#      blanks the center block and also blanks 1 or 2 extra non-center blocks, so
#      the input contains 2 or 3 all-background 3x3 holes.
#    - The full hidden motif still obeys the same global 180/D4 block symmetry as
#      the source task. Therefore every non-center block equals the 180-degree
#      rotation of its point-opposite partner.
#    - Execution trace on a small example:
#      old: only (1,1) is bg -> read the 8 pixels around that hole -> write center.
#      new: (1,1) and e.g. (0,0) are bg -> first rebuild (0,0) from rot180((2,2));
#           only after that are all 8 center-adjacent clue pixels present again, so
#           the center patch can be reconstructed exactly as in the source task.
#      difference: center filling can no longer happen first, because extra holes may
#      delete some of the clue pixels that the old `_center_patch` sampled.
# b) Complexity calibration
#    - Extra holes are limited to 1-2 non-center blocks. This makes the change
#      visually obvious while keeping the inference stable and learnable.
#    - At most one block is removed from each opposite pair. Otherwise both members
#      of a 180-symmetric pair could be missing and the reconstruction would become
#      ambiguous.
# c) Hidden assumptions in the original `main()`
#    - `_center_patch(...)` assumed every center-facing pixel already exists. This is
#      false once an adjacent edge or corner block may also be missing.
#    - The loop that copies outer blocks assumed every non-center source block is
#      already present in the input. This also becomes false in the variant.
#    - The bbox-based localization remains safe only if each outer block row and
#      column retains at least one visible block. Choosing at most two extra holes,
#      never both members of one opposite pair, preserves that property.
# d) Placement legality
#    - The motif footprint stays 9x9 and the expanded output stays 15x15, so the
#      original placement margins remain valid.
#    - Input holes are full 3x3 background blocks, so they never partially overlap
#      neighboring blocks and the solver can test them blockwise.
# e) Visual observability
#    - Humans can see 2-3 empty 3x3 slots in the input instead of exactly one.
#    - The additional holes are not arbitrary: each non-center hole has a visible
#      opposite partner whose 180-degree-rotated pattern matches the missing block.
#      After those are restored, the center is filled by the same local border rule.
# f) Visual cleanliness
#    - The grid still contains just one compact motif on a plain background.
#    - No extra markers or overlays are added; the only difference is that more
#      whole blocks are absent, so the object remains easy to parse.
# g) Color inference
#    - No sorting or branching uses color ids.
#    - The center pixel remains the majority visible foreground color, and the
#      generator validates every sample against the solver so no hidden color-based
#      ambiguity is introduced.
# h) Generator guarantee for new solver assumptions
#    - The center block is always one of the holes.
#    - Extra holes are chosen from distinct opposite pairs, so every non-center hole
#      has a visible opposite donor block.
#    - Samples are accepted only if the solver reproduces the intended output and
#      the visible majority color still matches the hidden center color.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_PAD = 3

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8
MAX_RETRIES = 300

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
NONCENTER_OPPOSITE_PAIRS = (
    ((0, 0), (2, 2)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 0)),
    ((1, 0), (1, 2)),
)


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _block_bbox(r0: int, c0: int, block: int, br: int, bc: int) -> tuple[int, int, int, int]:
    return (
        int(r0) + br * block,
        int(c0) + bc * block,
        int(r0) + (br + 1) * block - 1,
        int(c0) + (bc + 1) * block - 1,
    )


def _extract_block(g: np.ndarray, r0: int, c0: int, block: int, br: int, bc: int) -> np.ndarray:
    return crop_bbox(g, _block_bbox(r0, c0, block, br, bc))


def _is_bg_block(block_arr: np.ndarray, bg: int) -> bool:
    return bool(np.all(block_arr == int(bg)))


def _restore_noncenter_holes(motif: np.ndarray, block: int, bg: int) -> list[tuple[int, int]]:
    holes = []
    for br in range(3):
        for bc in range(3):
            blk = motif[br * block:(br + 1) * block, bc * block:(bc + 1) * block]
            if _is_bg_block(blk, bg):
                holes.append((br, bc))

    assert (1, 1) in holes, "variant expects the center block to be missing"
    for br, bc in holes:
        if (br, bc) == (1, 1):
            continue
        obr, obc = 2 - br, 2 - bc
        donor = motif[obr * block:(obr + 1) * block, obc * block:(obc + 1) * block]
        assert not _is_bg_block(donor, bg), "non-center hole lacks a visible opposite donor"
        paste(motif, np.rot90(donor, 2), br * block, bc * block)
    return holes


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 block motif when several blocks are missing.

    Input structure:
    - Teal background surrounds one 9x9 foreground motif.
    - The motif is a 3x3 array of 3x3 blocks with 2 or 3 empty teal blocks.
    - The center block is always one of the holes, and each extra hole has a
      visible block at the point-opposite position of the motif.

    Rule:
    - Rebuild every non-center hole by rotating its opposite block by 180 degrees.
    - After the surrounding blocks have been restored, reconstruct the center
      block from the eight pixels facing the center hole; its middle pixel is the
      motif's majority foreground color.
    - Place the repaired 9x9 motif back in place, then copy each non-center block
      to the opposite outer slot of the implied 5x5 block grid without rotating it.

    Key insight: The extra missing blocks are recoverable from whole-block
    point symmetry, but the center block still depends on local border pixels, so
    the reconstruction must happen in two stages.
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

    motif = crop_bbox(g, motif_bbox).copy()
    _restore_noncenter_holes(motif, block, bg)
    center = _center_patch(motif, 0, 0, block, bg)
    paste(motif, center, block, block)

    out = g.copy()
    paste(out, motif, int(r0), int(c0))
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block:(br + 1) * block, bc * block:(bc + 1) * block]
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, src, dst_r, dst_c)
    return out


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


def _expanded_output(full_motif):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    paste(expanded, full_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = full_motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _choose_holes() -> list[tuple[int, int]]:
    extra_count = random.choice([1, 2])
    pairs = random.sample(list(NONCENTER_OPPOSITE_PAIRS), extra_count)
    holes = [(1, 1)]
    for a, b in pairs:
        holes.append(random.choice([a, b]))
    return holes


def _apply_holes(motif: np.ndarray, holes: list[tuple[int, int]]) -> np.ndarray:
    out = motif.copy()
    for br, bc in holes:
        out[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK] = BG
    return out


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
    center = _gen_center_patch(motif, majority)
    full_motif = motif.copy()
    paste(full_motif, center, BLOCK, BLOCK)

    holes = _choose_holes()
    input_motif = _apply_holes(full_motif, holes)
    if _major_foreground_color(input_motif, BG) != majority:
        return None

    expanded = _expanded_output(full_motif)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, input_motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D11 variant input")
