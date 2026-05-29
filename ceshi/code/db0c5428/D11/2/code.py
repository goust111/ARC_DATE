# Variant of db0c5428
# Dimension: D11 — 中心空洞的形状 (src)
# Change: 中心空洞恰好是一个3x3块，与其他块大小相同，位于3x3阵列正中... → 空洞不在中心而在角落

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D11 variant
# a) Concrete behavior and execution trace
#    - The completed object is still a 9x9 motif made from a 3x3 array of 3x3 blocks.
#    - Unlike the source task, the middle 3x3 block is already present. One corner
#      block is fully missing and is painted with the background color.
#    - The finished motif still obeys the same block-level D4 pattern used by the
#      original generator:
#        corners come from one diagonal-symmetric seed block,
#        edges come from one LR-symmetric seed block,
#        center is the original task's reconstructed center patch.
#    - Small trace:
#      original: a hole at (1,1) was filled by sampling the eight pixels around the
#      hole, then the surrounding eight blocks were copied to the outer ring.
#      variant: suppose the hole is at (0,0). Local boundary sampling no longer
#      applies. Instead, the solver must find the diagonally opposite corner (2,2),
#      rotate that 3x3 block by 180 degrees to recover (0,0), paste it back, then
#      perform the same point-symmetric outer copy step. The difference is that the
#      reconstruction source is non-local and asymmetric with respect to the hole.
# b) Complexity calibration
#    - The missing position varies across the four corners, so the rule is visibly
#      about "corner holes" rather than one fixed absolute location.
#    - Corner blocks are not uniform: the seed generator forces at least one
#      non-majority pixel in the center-facing corner, so the 180-degree recovery is
#      visually observable.
#    - The center block is kept non-background, which prevents the puzzle from being
#      confused with the original center-hole task.
# c) Review of original main() assumptions
#    - The original skip condition `if (br, bc) == (1, 1)` assumed the only missing
#      block was the center. That is no longer enough: the input now contains a
#      missing corner, but the output still needs the outer copy corresponding to
#      that corner after reconstruction.
#    - `_center_patch(...)` assumed the answer was encoded by the eight cells around
#      the center hole. In the variant those cells are already occupied, so the same
#      logic would overwrite valid content and leave the missing corner unresolved.
#    - The bbox-based motif localization still works because removing any one corner
#      leaves other foreground blocks on the same outer rows and columns, so the
#      foreground bbox remains 9x9.
# d) Placement legality
#    - The motif is still placed with a margin of one block on each side, so the
#      15x15 expanded output always fits.
#    - Only one 3x3 corner block is blanked, so the solver always sees a unique hole.
# e) Visual observability
#    - Across examples, the center stays filled while one of the four corners is the
#      only all-background 3x3 block. The repaired output restores that corner and
#      also produces its point-symmetric outer copy, making the new rule visible.
# f) Visual cleanliness
#    - There is still exactly one compact motif and one expanded output region.
#      No extra markers or overlapping layers are introduced.
# g) Color inference
#    - No rule depends on color ids. Colors only serve as visible texture inside the
#      symmetric blocks, and the reconstruction rule depends purely on position and
#      orientation.
# 9) New solver assumptions and generator guarantees
#    - Assumption: exactly one corner block is all background.
#      Guarantee: the generator blanks exactly one randomly chosen corner.
#    - Assumption: the opposite corner is present and determines the missing corner
#      by a 180-degree rotation.
#      Guarantee: the completed motif is built from the original D4 template, so
#      every corner pair satisfies this relation.
#    - Assumption: the center block is already present.
#      Guarantee: the generator fills the center before erasing a corner.


def _extract_block(g: np.ndarray, r0: int, c0: int, block: int, br: int, bc: int) -> np.ndarray:
    return crop_bbox(
        g,
        (
            int(r0) + br * block,
            int(c0) + bc * block,
            int(r0) + (br + 1) * block - 1,
            int(c0) + (bc + 1) * block - 1,
        ),
    )


def _missing_corner(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> tuple[int, int]:
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    missing = [
        (br, bc)
        for br, bc in corners
        if np.all(_extract_block(g, r0, c0, block, br, bc) == int(bg))
    ]
    assert len(missing) == 1, "expected exactly one missing corner block"
    return missing[0]


def _restore_corner(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> tuple[tuple[int, int], np.ndarray]:
    br, bc = _missing_corner(g, r0, c0, block, bg)
    src = _extract_block(g, r0, c0, block, 2 - br, 2 - bc)
    return (br, bc), np.rot90(src, 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a corner-missing 3x3 block motif and reflect it into a 5x5 block layout.

    Input structure:
    - A teal background contains one 9x9 motif made from nine 3x3 block slots.
    - The middle slot is already filled, but exactly one corner slot is a blank teal
      3x3 hole.
    - The remaining visible blocks still follow the same global block symmetries as
      the finished motif.

    Rule:
    - Find which corner block is missing.
    - Rebuild that block by taking the diagonally opposite corner block and rotating
      it by 180 degrees.
    - Insert the repaired 3x3 motif, then copy every non-center block to its
      point-reflected outer slot in the surrounding 5x5 block grid.

    Key insight:
    - Once the empty slot moves away from the center, the answer is no longer encoded
      by the hole's immediate neighbors; it is determined by the opposite corner of
      the latent D4-symmetric motif.
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
    (hole_br, hole_bc), repaired = _restore_corner(g, int(r0), int(c0), block, bg)
    paste(motif, repaired, hole_br * block, hole_bc * block)

    out = g.copy()
    paste(out, motif, int(r0), int(c0))

    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, src, dst_r, dst_c)
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
CORNER_POSITIONS = ((0, 0), (0, 2), (2, 0), (2, 2))

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
        idx
        for idx, orbit in enumerate(orbits)
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


def _completed_motif(majority, minor_colors, corner_major, edge_major):
    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    paste(motif, center, BLOCK, BLOCK)
    return motif


def _mask_corner(motif, hole_corner):
    masked = motif.copy()
    br, bc = hole_corner
    masked[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK] = BG
    return masked


def _expanded_output(motif):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    paste(expanded, motif, BLOCK, BLOCK)
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

    full_motif = _completed_motif(majority, minor_colors, corner_major, edge_major)
    hole_corner = random.choice(CORNER_POSITIONS)
    input_motif = _mask_corner(full_motif, hole_corner)
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
    raise AssertionError("failed to generate a valid db0c5428 D11 input")
