# Combo variant of db0c5428 (index 45)
# combo_spec: D3/2+D5/0+D11/1
# Name: 简化中心+无空洞+环形排列

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve()
_SOLVER_DIR = _ROOT.parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for combo D3/2 + D5/0 + D11/1
# ------------------------------------------------
# a) Concretization + behavior trace:
#    - Original input: a 9x9 motif made of eight visible 3x3 blocks around a missing
#      3x3 center; output fills that hole, then copies the 8 non-center blocks into
#      the opposite slots of a sparse 5x5 block lattice.
#    - Combo input: the motif is now a complete 3x3 block array with no hole. Because
#      D11/1 removes the missing-center repair step, D5/0 must be realized as an explicit
#      output rewrite: the center block is simplified to a solid majority-color 3x3 patch,
#      regardless of its detailed input texture.
#    - Combo output: before doing any outer copy, the solver replaces the center block by
#      that solid majority patch. The 8 non-center blocks are then copied outward into an
#      8-position ring, not into the old sparse 5x5 grid. The D1 rule from the base task
#      is preserved implicitly: each source block still goes to the point-opposite
#      destination direction, but those destination slots now sit on a 17x17 ring frame.
#      Example: source NW -> outer SE, source N -> outer S.
#
# b) Complexity calibration:
#    - The change is not allowed to collapse into "just copy everything outward somehow".
#      Keeping opposite-direction mapping preserves a learnable correspondence, while the
#      ring geometry makes the layout visibly distinct from the source task.
#    - The center block is simplified to one solid color in the output, but the eight
#      surrounding blocks retain asymmetric internal patterns, so humans can still
#      visually track which block moved where.
#    - Only one ring radius and one gap width are used. More layout modes would be hard
#      to infer from 2-4 examples and would introduce unnecessary ambiguity.
#
# c) Hidden-assumption audit of the original main():
#    - `_center_patch(...)` assumed a 3x3 hole existed and that inner-boundary pixels
#      encoded the answer. D11/1 breaks that assumption completely, so hole-based repair
#      is removed; D5/0 is instead implemented as a majority-solid rewrite of the existing
#      center block.
#    - The `(4 - 2 * br, 4 - 2 * bc)` destination formula assumed the output lived on a
#      5x5 block grid. D3/2 invalidates that geometry and must be replaced by explicit
#      ring coordinates.
#    - `bbox(g != bg)` still works because the motif remains one contiguous 9x9 object.
#    - The block size assumption remains 3x3; the ring layout is defined for that size.
#
# d) Placement legality:
#    - The ring envelope is 17x17: 3x3 motif blocks centered inside a frame with a
#      1-pixel moat between the motif and the outer ring blocks.
#    - Therefore the generator uses square canvases of size 17..19 and places the motif
#      with a 4-pixel margin on all sides, guaranteeing the full ring fits in-bounds.
#    - Because the ring is pasted with transparent background, no copied block overlaps
#      another copied block or an unrelated object.
#
# e) Visual observability:
#    - "No hole" is immediately visible because the center is an actual filled 3x3 block.
#    - The ring layout is also immediately visible because the outer copies are centered
#      on the four cardinal directions and four diagonals, with a 1-pixel gap from the
#      motif; this does not resemble the old sparse lattice.
#    - The center being a solid majority-color square is human-visible and does not rely
#      on any hidden numeric ordering of colors.
#
# f) Visual cleanliness:
#    - The output contains exactly one object family: the original central motif plus one
#      clean outer ring of 8 copies. No separators, markers, or overlapping layers are
#      introduced.
#    - The 1-pixel moat keeps the ring legible and prevents the result from looking like
#      an accidental 15x15 blob.
#
# g) Change inferability / color discipline:
#    - No color remapping is introduced. Colors keep their identity between input and
#      output, so the task stays about structure and placement rather than hidden color
#      codes.
#    - The center-majority condition is stable because the 8 outer blocks are still built
#      with majority-dominant patterns, and the center contributes 9 more majority cells.
#
# 9) New solver assumptions guaranteed by generate_input():
#    - The input always contains exactly one complete 9x9 motif on a background canvas.
#    - The motif always has 3x3 blocks and at least a 4-cell margin so the 17x17 ring
#      frame can be pasted without clipping.
#    - The generator already tends to build a center close to the majority color, but the
#      solver itself now enforces the D5/0 simplification so the fusion does not depend on
#      that input detail.
#    - A degeneration check rejects samples whose output would be too close to the legacy
#      5x5-lattice behavior on the same input.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
RING_GAP = 1
MOTIF_OFFSET = BLOCK + RING_GAP
RING_SIZE = MOTIF_SIZE + 2 * MOTIF_OFFSET
GRID_SIZE_MIN, GRID_SIZE_MAX = 17, 19
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8
MAX_RETRIES = 300
MIN_VARIANT_DIFF = 24

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

_RING_COORD = {-1: 0, 0: 7, 1: 14}


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _extract_block(g: np.ndarray, r0: int, c0: int, br: int, bc: int, block: int = BLOCK) -> np.ndarray:
    return crop_bbox(
        g,
        (
            int(r0) + br * block,
            int(c0) + bc * block,
            int(r0) + (br + 1) * block - 1,
            int(c0) + (bc + 1) * block - 1,
        ),
    )


def _ring_slot_from_source(br: int, bc: int) -> tuple[int, int]:
    assert (br, bc) != (1, 1), "center block is not copied into the ring"
    dr = 1 - int(br)
    dc = 1 - int(bc)
    return _RING_COORD[dr], _RING_COORD[dc]


def _ring_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    patch = np.full((RING_SIZE, RING_SIZE), bg, dtype=int)
    paste(patch, motif, MOTIF_OFFSET, MOTIF_OFFSET)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            dst_r, dst_c = _ring_slot_from_source(br, bc)
            paste(patch, src, dst_r, dst_c)
    return patch


def _solid_center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    major = _major_foreground_color(motif, bg)
    return np.full((BLOCK, BLOCK), major, dtype=int)


def _legacy_center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
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


def _legacy_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0
    block = h // 3

    out = g.copy()
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = _extract_block(g, int(r0), int(c0), br, bc, block)
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, src, dst_r, dst_c)

    paste(out, _legacy_center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a complete 3x3 block motif by simplifying its center and copying the outer ring.

    Input structure:
    - One 9x9 foreground motif sits on a plain background canvas.
    - The motif is a full 3x3 arrangement of 3x3 blocks with no missing center block.
    - The center block is already present, so the input is a complete contiguous object.

    Rule:
    - Rewrite the center 3x3 block to a solid patch of the motif's majority foreground color.
    - Keep that simplified 9x9 motif exactly where it is.
    - Copy each of the eight non-center 3x3 blocks to the opposite direction around the motif.
    - Place those copied blocks on a ring layout: corners go to diagonal ring slots and edge blocks
      go to cardinal ring slots, leaving a 1-pixel moat between the ring and the central motif.

    Key insight: D11/1 removes the missing-hole repair, so D5/0 has to show up as a direct
    simplification of the already-present center block. After that simplification, the task becomes
    an outward ring expansion that preserves the old opposite-direction correspondence while changing
    the destination geometry from a sparse grid to a ring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == MOTIF_SIZE and w == MOTIF_SIZE, "expected one complete 9x9 motif"

    frame_r0 = int(r0) - MOTIF_OFFSET
    frame_c0 = int(c0) - MOTIF_OFFSET
    motif = crop_bbox(g, motif_bbox)
    motif = motif.copy()
    motif[BLOCK:2 * BLOCK, BLOCK:2 * BLOCK] = _solid_center_patch(motif, bg)
    out = g.copy()
    paste(out, _ring_patch(motif, bg), frame_r0, frame_c0, transparent=bg)
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


def _assemble_motif(b00: np.ndarray, b01: np.ndarray, majority: int) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): np.fliplr(b00),
        (1, 0): b01.T,
        (1, 1): np.full((BLOCK, BLOCK), majority, dtype=int),
        (1, 2): np.rot90(b01, -1),
        (2, 0): np.flipud(b00),
        (2, 1): np.flipud(b01),
        (2, 2): np.rot90(b00, 2),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _variant_signal(inp: np.ndarray) -> int:
    return int(np.count_nonzero(main(inp) != _legacy_output(inp)))


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
    motif = _assemble_motif(b00, b01, majority)
    ring = _ring_patch(motif, BG)

    r0 = random.randint(MOTIF_OFFSET, n - MOTIF_SIZE - MOTIF_OFFSET)
    c0 = random.randint(MOTIF_OFFSET, n - MOTIF_SIZE - MOTIF_OFFSET)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, ring, r0 - MOTIF_OFFSET, c0 - MOTIF_OFFSET, transparent=BG)

    if not np.array_equal(main(inp), out):
        return None
    if _variant_signal(inp) < MIN_VARIANT_DIFF:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid combo db0c5428 input")
