# Combo variant of db0c5428 (index 16)
# combo_spec: D2/3+D11/1
# Name: 十字形阵列+无空洞

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior trace
# - Original task: the 9x9 motif is a full 3x3 block array except that the
#   center 3x3 block is a hole. Solver copies the eight visible blocks outward
#   into a 5x5 block frame, then reconstructs the missing center block from the
#   pixels facing the hole.
# - Fused task: the motif becomes a cross of five occupied 3x3 blocks inside
#   the same 9x9 envelope. The four corners are empty background, while the
#   center is now a real block with content. There is therefore nothing to
#   reconstruct. The only action is outward extension: top/bottom/left/right
#   arm blocks are copied to the opposite outer slot, turning the middle 3x3
#   cross into a 5x5 plus sign.
# - Visual difference: original input had 8 occupied blocks with an empty
#   center; fused input has 5 occupied blocks with empty corners and a filled
#   center. Output changes from "filled 3x3 center plus diagonally populated
#   outer ring" to "clean 9-block plus sign".
#
# b) Complexity calibration
# - The rule is simpler than the original because center reconstruction is gone,
#   but not too simple: the human must notice both the changed occupancy mask
#   (cross instead of ring) and the opposite-side copying of asymmetric arm
#   blocks.
# - To keep samples readable and non-ambiguous, generator uses dense 3x3 arm
#   blocks with no background holes, and enforces that the four arm blocks are
#   pairwise distinct. This makes the copy direction visible in each example.
#
# c) Hidden assumptions broken by fusion
# - Original `main()` assumed every non-center block exists. That fails because
#   fused inputs have empty corners.
# - Original `main()` assumed the center must be rebuilt from surrounding
#   boundary pixels. That fails because the center already contains content.
# - Original `_center_patch()` and the unconditional center paste therefore have
#   to be removed. Solver must instead identify occupied sub-blocks inside the
#   3x3 block lattice and only outward-copy the four arm blocks.
#
# d) Placement legality
# - Input motif still occupies a 9x9 bbox, so the original padding logic of at
#   least one block of margin on each side remains sufficient for the 15x15
#   output footprint. Generator places the motif with `r0,c0 >= 3` and
#   `r0+12,c0+12 <= N`.
#
# e/f/g) Observability, neatness, inferability
# - Center block is always filled, corners are always blank, so the fusion is
#   visually obvious relative to the base task.
# - Arm blocks are copied without rotation and are pairwise distinct, giving the
#   solver three repeated cues in the output: the middle cross stays put, each
#   outer extension matches the opposite inner arm exactly, and the whole output
#   is a tidy plus sign with no overlapping layers.
# ---------------------------------------------------------------------------


BLOCK = 3
MOTIF_BLOCKS = 3
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
EXPANDED_BLOCKS = 5
EXPANDED_SIZE = BLOCK * EXPANDED_BLOCKS

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = BLOCK
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
MAX_RETRIES = 300

CROSS_POSITIONS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
ARM_POSITIONS = ((0, 1), (1, 0), (1, 2), (2, 1))
CROSS_MASK = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)


def _block_slice(br: int, bc: int) -> tuple[int, int, int, int]:
    r0 = int(br) * BLOCK
    c0 = int(bc) * BLOCK
    return r0, c0, r0 + BLOCK - 1, c0 + BLOCK - 1


def _extract_block(grid: np.ndarray, r0: int, c0: int, br: int, bc: int) -> np.ndarray:
    rr0, cc0, rr1, cc1 = _block_slice(br, bc)
    return crop_bbox(grid, (int(r0) + rr0, int(c0) + cc0, int(r0) + rr1, int(c0) + cc1))


def _occupancy_and_blocks(grid: np.ndarray, r0: int, c0: int, bg: int) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    occupied = np.zeros((MOTIF_BLOCKS, MOTIF_BLOCKS), dtype=bool)
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            block = _extract_block(grid, r0, c0, br, bc)
            blocks[(br, bc)] = block
            occupied[br, bc] = bool(np.any(block != bg))
    return occupied, blocks


def _plus_output(grid: np.ndarray, r0: int, c0: int, blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    out = grid.copy()
    expanded_r0 = int(r0) - BLOCK
    expanded_c0 = int(c0) - BLOCK
    for br, bc in ARM_POSITIONS:
        dst_r = expanded_r0 + (4 - 2 * br) * BLOCK
        dst_c = expanded_c0 + (4 - 2 * bc) * BLOCK
        paste(out, blocks[(br, bc)], dst_r, dst_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a cross-shaped 3x3-block motif into a longer plus sign.

    Input structure:
    - Teal background surrounds one 9x9 motif aligned to a 3x3 lattice of 3x3 blocks.
    - Only the center row and center column blocks contain color, forming a cross
      of five occupied blocks.
    - The four corner blocks of that 3x3 lattice are pure background; unlike the
      original task, the middle block is already filled.

    Rule:
    - Keep the existing five-block cross unchanged.
    - Copy each arm block to the opposite outer slot of the implied 5x5 block
      lattice, without rotating it.
    - Do not reconstruct or alter the center block; the transformation is purely
      an outward extension of the cross.

    Key insight:
    The foreground bbox is still a 9x9 block-aligned envelope, but its occupancy
    pattern is a plus sign rather than a ring. The missing-corner pattern tells
    you that only the four arm blocks should propagate outward.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == MOTIF_SIZE and w == MOTIF_SIZE, "motif must occupy a 9x9 envelope"

    occupied, blocks = _occupancy_and_blocks(g, int(r0), int(c0), bg)
    assert np.array_equal(occupied, CROSS_MASK), "expected a filled-center cross motif"
    return _plus_output(g, int(r0), int(c0), blocks)


def _random_block(colors: list[int], majority: int, *, min_majority: int, max_majority: int) -> np.ndarray:
    count = random.randint(min_majority, max_majority)
    flat = [majority] * count
    accents = [c for c in colors if c != majority]
    while len(flat) < BLOCK * BLOCK:
        flat.append(random.choice(accents))
    random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(BLOCK, BLOCK)


def _arm_variants(seed: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    return {
        (0, 1): seed.copy(),
        (1, 0): np.rot90(seed, 1),
        (1, 2): np.rot90(seed, -1),
        (2, 1): np.flipud(seed),
    }


def _signatures(blocks: dict[tuple[int, int], np.ndarray], positions: tuple[tuple[int, int], ...]) -> list[tuple[int, ...]]:
    return [tuple(map(int, blocks[pos].flatten())) for pos in positions]


def _make_cross_blocks(colors: list[int]) -> dict[tuple[int, int], np.ndarray]:
    majority = colors[0]
    for _ in range(MAX_RETRIES):
        arm_seed = _random_block(colors, majority, min_majority=4, max_majority=6)
        arm_blocks = _arm_variants(arm_seed)
        arm_sigs = _signatures(arm_blocks, ARM_POSITIONS)
        if len(set(arm_sigs)) != 4:
            continue

        center = _random_block(colors, majority, min_majority=4, max_majority=7)
        center_sig = tuple(map(int, center.flatten()))
        if center_sig in set(arm_sigs):
            continue

        blocks = dict(arm_blocks)
        blocks[(1, 1)] = center
        return blocks
    raise AssertionError("failed to create distinctive cross blocks")


def _assemble_input_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for pos in CROSS_POSITIONS:
        br, bc = pos
        paste(motif, blocks[pos], br * BLOCK, bc * BLOCK)
    return motif


def _expected_output_region(motif: np.ndarray, blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    paste(expanded, motif, BLOCK, BLOCK)
    for br, bc in ARM_POSITIONS:
        dst_r = (4 - 2 * br) * BLOCK
        dst_c = (4 - 2 * bc) * BLOCK
        paste(expanded, blocks[(br, bc)], dst_r, dst_c)
    return expanded


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    blocks = _make_cross_blocks(colors)
    motif = _assemble_input_motif(blocks)
    expanded = _expected_output_region(motif, blocks)

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
