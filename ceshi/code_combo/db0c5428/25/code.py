# Combo variant of db0c5428 (index 25)
# combo_spec: D1/1+D11/2
# Name: 角落空洞+镜像映射

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a_concretization:
#   - The completed object is still a 9x9 motif made from nine 3x3 block slots,
#     but the center is already filled and exactly one corner slot is blank teal.
#   - The finished motif has exactly one mirror axis, never both. That axis drives
#     both changed parts of the task:
#       * reconstruction: the missing corner is copied from the block across that
#         same axis, using flipud for a horizontal axis or fliplr for a vertical axis;
#       * expansion: every visible block is copied into the 5x5 outer frame by the
#         same single-axis mirror map instead of point reflection.
#   - Concrete trace: if the motif is horizontally symmetric and the hole is at
#     block (0,0), the original D11 corner-hole rule would use block (2,2) rotated
#     180 degrees, and the original outer-copy rule would send block (0,0) to outer
#     slot (4,4). In this combo, block (0,0) is rebuilt from block (2,0) via flipud,
#     and after repair that block copies to outer slot (4,1). Both "where to look"
#     and "where to paste" now depend on the same inferred axis.
# b_complexity:
#   - The key hidden variable is kept small: axis in {horizontal, vertical}. Corner
#     position still ranges over four values, but every sample repeats the same axis
#     evidence across multiple blocks, so humans can infer it from a few examples.
#   - Blocks are generated with enough asymmetry that the mirror-axis choice and the
#     corner-repair source differ visibly from the source task's diagonal logic.
# c_hidden_assumptions_review:
#   - Original `_center_patch(...)` assumed the blank 3x3 area was at the center and
#     encoded by adjacent pixels. That premise is false here because the center block
#     is already filled and the only hole is a corner.
#   - Original point-reflection destinations `(4-2*br, 4-2*bc)` assumed both row and
#     column indices invert together. The combo keeps only one mirrored coordinate, so
#     destination selection must become axis-aware.
#   - Inferring the axis from `motif == flipud(motif)` or `motif == fliplr(motif)` no
#     longer works directly because one corner is erased and breaks exact equality.
#     The solver instead checks the surviving block-pair relations and self-symmetric
#     middle strip required by each axis.
#   - Reconstructing the hole now depends on a new assumption: exactly one corner
#     block is fully background, and the block mirrored across the true axis exists.
# d_placement_legality:
#   - The expanded output is still a 15x15 block frame, so the original one-block
#     margin around the 9x9 motif remains sufficient.
#   - Only one corner is blanked, and it stays inside the 9x9 motif bbox because the
#     other blocks still occupy the top/bottom rows and left/right columns.
# e_visual_observability:
#   - The input visibly differs from the source task: the center is present and a
#     corner is empty.
#   - The unique mirror axis is recoverable from repeated structure in the remaining
#     blocks, and the output reinforces it because the outer copies form a one-axis
#     mirrored frame rather than a point-symmetric one.
#   - The generator rejects samples where axis-based corner repair would accidentally
#     match the old diagonal-rotation repair, or where the outer expansion would
#     coincide with the old point-symmetric layout. That keeps the fusion observable.
# f_cleanliness:
#   - There is still one compact motif on a blank background and one expanded answer.
#     No extra markers, counters, or overlays are added.
# g_color_inference:
#   - No rule depends on numeric color ids. Colors only provide texture so that the
#     axis relation and the changed reconstruction source are visually inspectable.
# h_new_solver_assumptions_and_generator_guarantees:
#   - Assumption: exactly one axis remains consistent with the incomplete motif.
#     Guarantee: the generator builds motifs with one true axis and explicit
#     asymmetries that invalidate the other axis even after any single corner is blanked.
#   - Assumption: the axis-based repair differs from the original diagonal repair.
#     Guarantee: generation rejects motifs where those two recovered corner blocks match.


HORIZONTAL_AXIS = "horizontal"
VERTICAL_AXIS = "vertical"
CORNER_POSITIONS = ((0, 0), (0, 2), (2, 0), (2, 2))


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


def _block_from_motif(motif: np.ndarray, block: int, br: int, bc: int) -> np.ndarray:
    return motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]


def _is_bg_block(block_arr: np.ndarray, bg: int) -> bool:
    return bool(np.all(block_arr == int(bg)))


def _missing_corner(motif: np.ndarray, block: int, bg: int) -> tuple[int, int]:
    missing = [
        (br, bc)
        for br, bc in CORNER_POSITIONS
        if _is_bg_block(_block_from_motif(motif, block, br, bc), bg)
    ]
    assert len(missing) == 1, "expected exactly one missing corner block"
    return missing[0]


def _axis_consistent(motif: np.ndarray, block: int, bg: int, axis: str) -> bool:
    evidence = 0
    if axis == HORIZONTAL_AXIS:
        for bc in range(3):
            top = _block_from_motif(motif, block, 0, bc)
            bottom = _block_from_motif(motif, block, 2, bc)
            top_bg = _is_bg_block(top, bg)
            bottom_bg = _is_bg_block(bottom, bg)
            if top_bg != bottom_bg:
                if bc == 1:
                    return False
                evidence += 1
            elif not top_bg:
                if not np.array_equal(top, np.flipud(bottom)):
                    return False
                evidence += 1
        for bc in range(3):
            mid = _block_from_motif(motif, block, 1, bc)
            if _is_bg_block(mid, bg) or not np.array_equal(mid, np.flipud(mid)):
                return False
            evidence += 1
        return evidence > 0

    for br in range(3):
        left = _block_from_motif(motif, block, br, 0)
        right = _block_from_motif(motif, block, br, 2)
        left_bg = _is_bg_block(left, bg)
        right_bg = _is_bg_block(right, bg)
        if left_bg != right_bg:
            if br == 1:
                return False
            evidence += 1
        elif not left_bg:
            if not np.array_equal(left, np.fliplr(right)):
                return False
            evidence += 1
    for br in range(3):
        mid = _block_from_motif(motif, block, br, 1)
        if _is_bg_block(mid, bg) or not np.array_equal(mid, np.fliplr(mid)):
            return False
        evidence += 1
    return evidence > 0


def _infer_axis(motif: np.ndarray, block: int, bg: int) -> str:
    horizontal = _axis_consistent(motif, block, bg, HORIZONTAL_AXIS)
    vertical = _axis_consistent(motif, block, bg, VERTICAL_AXIS)
    assert horizontal ^ vertical, "motif must imply exactly one mirror axis"
    return HORIZONTAL_AXIS if horizontal else VERTICAL_AXIS


def _restore_corner(motif: np.ndarray, block: int, bg: int, axis: str) -> tuple[tuple[int, int], np.ndarray]:
    br, bc = _missing_corner(motif, block, bg)
    if axis == HORIZONTAL_AXIS:
        src = _block_from_motif(motif, block, 2 - br, bc)
        repaired = np.flipud(src)
    else:
        src = _block_from_motif(motif, block, br, 2 - bc)
        repaired = np.fliplr(src)
    assert not _is_bg_block(src, bg), "mirror source block must exist"
    return (br, bc), repaired


def _outer_slot(br: int, bc: int, axis: str) -> tuple[int, int]:
    if axis == HORIZONTAL_AXIS:
        if br == 0:
            return 4, 1 + bc
        if br == 2:
            return 0, 1 + bc
        assert br == 1 and bc in (0, 2)
        return 2, 0 if bc == 0 else 4

    if bc == 0:
        return 1 + br, 4
    if bc == 2:
        return 1 + br, 0
    assert bc == 1 and br in (0, 2)
    return (0 if br == 0 else 4), 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a corner-missing block motif and expand it by a single mirror axis.

    Input structure:
    - A teal background contains one 9x9 motif made from nine 3x3 block slots.
    - The center block is already filled, but exactly one corner block is a blank
      teal 3x3 hole.
    - The completed motif would have exactly one mirror axis, and the visible
      blocks still preserve enough of that structure to identify the axis.

    Rule:
    - Locate the missing corner block.
    - Infer whether the motif is horizontally or vertically mirror-symmetric.
    - Rebuild the missing corner from the block across that same axis using the
      matching pixel flip, then copy every non-center block to the outer 5x5
      frame according to the same single-axis mirror mapping.

    Key insight:
    The corner repair and the outer expansion are coupled: once the center is no
    longer the hole, "the opposite block" is defined by the motif's lone mirror
    axis rather than by diagonal point symmetry.
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
    axis = _infer_axis(motif, block, bg)
    (hole_br, hole_bc), repaired = _restore_corner(motif, block, bg, axis)
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
            dst_br, dst_bc = _outer_slot(br, bc, axis)
            paste(out, src, expanded_r0 + dst_br * block, expanded_c0 + dst_bc * block)
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
MAX_RETRIES = 500
MAJORITY_MIN, MAJORITY_MAX = 5, 7

GENERAL_ORBITS = tuple(((r, c),) for r in range(BLOCK) for c in range(BLOCK))
UD_ORBITS = (
    ((0, 0), (2, 0)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 2)),
    ((1, 0),),
    ((1, 1),),
    ((1, 2),),
)
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)


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
    assert choices, "unreachable majority count"
    return random.choice(choices)


def _assign_minor_colors(orbits_left, palette):
    assert palette, "need at least one minor color"
    random.shuffle(orbits_left)
    guaranteed = list(palette[: min(len(palette), len(orbits_left))])
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(palette) for _ in range(len(orbits_left) - len(guaranteed))
    ]
    return assignments


def _make_block(
    orbits,
    majority: int,
    minor_colors: list[int],
    forced_minor,
    *,
    forbid_ud: bool = False,
    forbid_lr: bool = False,
) -> np.ndarray:
    for _ in range(200):
        target_major = random.randint(MAJORITY_MIN, MAJORITY_MAX)
        block_arr = np.full((BLOCK, BLOCK), -1, dtype=int)
        major_idxs = _major_orbits(orbits, forced_minor, target_major)
        leftover = []
        for idx, orbit in enumerate(orbits):
            if idx in major_idxs:
                for r, c in orbit:
                    block_arr[r, c] = majority
            else:
                leftover.append(orbit)

        for orbit, color in zip(leftover, _assign_minor_colors(leftover, minor_colors)):
            for r, c in orbit:
                block_arr[r, c] = color

        if forbid_ud and np.array_equal(block_arr, np.flipud(block_arr)):
            continue
        if forbid_lr and np.array_equal(block_arr, np.fliplr(block_arr)):
            continue
        return block_arr
    raise AssertionError("failed to build asymmetric block")


def _assemble_horizontal_motif(majority: int, minor_colors: list[int]) -> np.ndarray:
    tl = _make_block(GENERAL_ORBITS, majority, minor_colors, [(2, 2)], forbid_ud=True, forbid_lr=True)
    tm = _make_block(GENERAL_ORBITS, majority, minor_colors, [(2, 1)], forbid_ud=True, forbid_lr=True)
    tr = _make_block(GENERAL_ORBITS, majority, minor_colors, [(2, 0)], forbid_ud=True, forbid_lr=True)
    ml = _make_block(UD_ORBITS, majority, minor_colors, [(1, 2)], forbid_lr=True)
    mr = _make_block(UD_ORBITS, majority, minor_colors, [(1, 0)], forbid_lr=True)

    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): tl,
        (0, 1): tm,
        (0, 2): tr,
        (1, 0): ml,
        (1, 2): mr,
        (2, 0): np.flipud(tl),
        (2, 1): np.flipud(tm),
        (2, 2): np.flipud(tr),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _assemble_vertical_motif(majority: int, minor_colors: list[int]) -> np.ndarray:
    tl = _make_block(GENERAL_ORBITS, majority, minor_colors, [(2, 2)], forbid_ud=True, forbid_lr=True)
    ml = _make_block(GENERAL_ORBITS, majority, minor_colors, [(1, 2)], forbid_ud=True, forbid_lr=True)
    bl = _make_block(GENERAL_ORBITS, majority, minor_colors, [(0, 2)], forbid_ud=True, forbid_lr=True)
    tm = _make_block(LR_ORBITS, majority, minor_colors, [(2, 1)], forbid_ud=True)
    bm = _make_block(LR_ORBITS, majority, minor_colors, [(0, 1)], forbid_ud=True)

    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): tl,
        (0, 1): tm,
        (0, 2): np.fliplr(tl),
        (1, 0): ml,
        (1, 2): np.fliplr(ml),
        (2, 0): bl,
        (2, 1): bm,
        (2, 2): np.fliplr(bl),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    main = _major_foreground_color(motif, bg)
    patch = np.full((BLOCK, BLOCK), main, dtype=int)
    hole_r = BLOCK
    hole_c = BLOCK
    offsets = (-1, 1, 3)
    for r in range(BLOCK):
        for c in range(BLOCK):
            if (r, c) == (1, 1):
                continue
            patch[r, c] = int(motif[hole_r + offsets[r], hole_c + offsets[c]])
    return patch


def _completed_motif(axis: str, majority: int, minor_colors: list[int]) -> np.ndarray:
    for _ in range(200):
        motif = (
            _assemble_horizontal_motif(majority, minor_colors)
            if axis == HORIZONTAL_AXIS
            else _assemble_vertical_motif(majority, minor_colors)
        )
        center = _center_patch(motif, BG)
        paste(motif, center, BLOCK, BLOCK)
        if _infer_axis(motif, BLOCK, BG) != axis:
            continue
        return motif
    raise AssertionError("failed to build motif with a unique axis")


def _mask_corner(motif: np.ndarray, hole_corner: tuple[int, int]) -> np.ndarray:
    masked = motif.copy()
    br, bc = hole_corner
    masked[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK] = BG
    return masked


def _expanded_output(motif: np.ndarray, axis: str) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    paste(expanded, motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            dst_br, dst_bc = _outer_slot(br, bc, axis)
            paste(expanded, src, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _point_expanded_output(motif: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    paste(expanded, motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _axis_repair_block(motif: np.ndarray, hole_corner: tuple[int, int], axis: str) -> np.ndarray:
    br, bc = hole_corner
    if axis == HORIZONTAL_AXIS:
        return np.flipud(_block_from_motif(motif, BLOCK, 2 - br, bc))
    return np.fliplr(_block_from_motif(motif, BLOCK, br, 2 - bc))


def _diagonal_repair_block(motif: np.ndarray, hole_corner: tuple[int, int]) -> np.ndarray:
    br, bc = hole_corner
    return np.rot90(_block_from_motif(motif, BLOCK, 2 - br, 2 - bc), 2)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:] or [random.choice([c for c in FOREGROUND_COLORS if c != majority])]
    axis = random.choice([HORIZONTAL_AXIS, VERTICAL_AXIS])

    full_motif = _completed_motif(axis, majority, minor_colors)
    hole_corner = random.choice(CORNER_POSITIONS)
    axis_repair = _axis_repair_block(full_motif, hole_corner, axis)
    diagonal_repair = _diagonal_repair_block(full_motif, hole_corner)
    if np.array_equal(axis_repair, diagonal_repair):
        return None

    expanded = _expanded_output(full_motif, axis)
    if np.array_equal(expanded, _point_expanded_output(full_motif)):
        return None

    input_motif = _mask_corner(full_motif, hole_corner)
    if _infer_axis(input_motif, BLOCK, BG) != axis:
        return None

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
    raise AssertionError("failed to generate a valid db0c5428 combo input")
