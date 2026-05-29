# Variant of db0c5428
# Dimension: D6 — 图案的整体对称性 (match_cond)
# Change: 输入的9x9图案具有完全的双轴对称性（LR和UD），相对块互为镜像翻转... → 四重旋转对称（C4）

from __future__ import annotations

import random
import sys

import numpy as np

_SOLVERS_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVERS_DIR not in sys.path:
    sys.path.append(_SOLVERS_DIR)

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D6 variant
# ----------------------------------
# concrete_behavior:
# - The 9x9 motif is still a 3x3 array of 3x3 blocks with a teal center hole.
# - The change is at block level: the eight visible blocks now form a pure C4 orbit.
#   If the top-left corner block is A and the top edge block is B, then:
#   top-right=rot90cw(A), right=rot90cw(B), bottom-right=rot180(A),
#   bottom=rot180(B), bottom-left=rot270(A), left=rot270(B).
# - The center reconstruction rule stays local: the eight pixels facing the hole
#   still become the center block's rim, with the center cell set to the motif's
#   majority color.
# - The output extension changes together with the symmetry: instead of copying a
#   block to the opposite outer slot, each block is copied outward in the same
#   direction. This makes the 5x5 block layout a larger C4-symmetric ring around
#   the reconstructed center.
#
# behavior_trace:
# - Original mapping for block (0,0): source top-left corner -> outer bottom-right.
# - Variant mapping for block (0,0): source top-left corner -> outer top-left.
# - Original neighboring corner relation: block(0,2)=flipLR(block(0,0)).
# - Variant neighboring corner relation: block(0,2)=rot90cw(block(0,0)).
# - So a visibly asymmetric seed block now changes orientation by quarter-turns
#   around the hole, and the outer ring continues those same orientations.
#
# complexity_calibration:
# - If seed blocks were solid colors or rotation-invariant, the C4 change would be
#   invisible. The generator therefore forces each seed block to be non-invariant
#   under 90/180/270-degree rotation and also rejects motifs that accidentally gain
#   LR or UD mirror symmetry.
# - Only two seed blocks are still used, so the task remains close to the original
#   in object count and reconstruction complexity.
#
# implicit_assumption_audit:
# - main() still assumes a single square foreground motif whose side length is a
#   multiple of 3; the generator preserves that exactly.
# - _center_patch() only assumes the missing center is surrounded by eight visible
#   boundary-facing pixels; this remains true under C4.
# - The old outer-slot formula depended on point-symmetric expansion. It is the
#   only solver line that becomes logically wrong under the variant, so it is
#   replaced by a same-direction expansion formula.
#
# placement_legality:
# - Grid size remains 16..19 and the 9x9 motif still keeps at least a 3-cell pad.
#   That guarantees the expanded 15x15 output fits after shifting by one block.
#
# visual_observability:
# - The generated motif is guaranteed to have C4 symmetry but not LR or UD mirror
#   symmetry, so the rotational rule is visible from the input alone.
# - The output ring uses the same directional slots, making the contrast against
#   the original "copy to the opposite side" behavior easy to see.
#
# cleanliness_and_colors:
# - No new object types or overlaps are introduced; the layout remains the same
#   sparse 5x5 block pattern as the original task.
# - Colors are chosen from 2..4 foreground colors, with a majority color and a few
#   repeated minor colors. Color ids are never used as hidden ordering keys.
#
# new_main_assumptions_and_generator_guarantees:
# - main() assumes the outer ring cells around the motif are empty background and
#   safe to fill by same-direction expansion; generator placement guarantees that.
# - main() does not check C4 explicitly, so generator enforces it and rejects any
#   accidental D4/bilateral degeneracy.


def _rot90cw(block: np.ndarray, k: int = 1) -> np.ndarray:
    return np.rot90(block, -int(k))


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a rotationally symmetric 3x3 block motif into a larger 5x5 block frame.

    Input structure:
    - A teal background contains one 9x9 foreground motif.
    - The motif is a 3x3 array of 3x3 blocks with the middle block missing.
    - The eight visible blocks form a C4 pattern: moving around the hole turns the
      seed blocks by quarter-rotations rather than reflecting them.

    Rule:
    - Leave the original 9x9 motif where it is.
    - Rebuild the missing center block from the eight pixels facing the hole; the
      center cell is the motif's majority foreground color.
    - Copy each visible 3x3 block outward to the outer ring slot in the same
      direction from the center, without changing that block again.

    Key insight: The visible motif is the inner ring of a larger C4-symmetric
    block arrangement, so the solver extends the ring directionally and uses the
    hole boundary to recover the missing middle block.
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
            dst_r = expanded_r0 + (2 * br) * block
            dst_c = expanded_c0 + (2 * bc) * block
            paste(out, src, dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator ---------------

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
BLOCK_RETRIES = 200

_CENTER_OFFSETS = np.array([-1, 1, 3])
_ALL_CELLS = [(r, c) for r in range(BLOCK) for c in range(BLOCK)]


def _is_rotation_distinct(block: np.ndarray) -> bool:
    return all(not np.array_equal(block, _rot90cw(block, k)) for k in (1, 2, 3))


def _same_rotation_family(a: np.ndarray, b: np.ndarray) -> bool:
    return any(np.array_equal(_rot90cw(a, k), b) for k in range(4))


def _major_positions(forced_minor: set[tuple[int, int]], target_count: int) -> set[tuple[int, int]]:
    usable = [coord for coord in _ALL_CELLS if coord not in forced_minor]
    assert target_count <= len(usable)
    return set(random.sample(usable, target_count))


def _seed_block(forced_minor: list[tuple[int, int]], majority: int, target_count: int):
    forced_minor_set = set(forced_minor)
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_positions = _major_positions(forced_minor_set, target_count)
    minor_cells = []
    for r, c in _ALL_CELLS:
        if (r, c) in major_positions:
            block[r, c] = majority
        else:
            minor_cells.append((block, (r, c)))
    return block, minor_cells


def _assign_minor_colors(minor_cells, colors):
    random.shuffle(minor_cells)
    guaranteed = list(colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(colors) for _ in range(len(minor_cells) - len(guaranteed))
    ]
    for (block, (r, c)), color in zip(minor_cells, assignments):
        block[r, c] = color


def _assemble_motif(b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): _rot90cw(b00, 1),
        (1, 2): _rot90cw(b01, 1),
        (2, 2): _rot90cw(b00, 2),
        (2, 1): _rot90cw(b01, 2),
        (2, 0): _rot90cw(b00, 3),
        (1, 0): _rot90cw(b01, 3),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _expanded_output(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (2 * br) * BLOCK, (2 * bc) * BLOCK)
    return expanded


def _make_blocks(majority: int, minor_colors: list[int], corner_major: int, edge_major: int):
    for _ in range(BLOCK_RETRIES):
        b00, corner_minors = _seed_block([(2, 2)], majority, corner_major)
        b01, edge_minors = _seed_block([(2, 1)], majority, edge_major)
        minor_cells = corner_minors + edge_minors
        if len(minor_cells) < len(minor_colors):
            continue
        _assign_minor_colors(minor_cells, minor_colors)
        if not _is_rotation_distinct(b00):
            continue
        if not _is_rotation_distinct(b01):
            continue
        if _same_rotation_family(b00, b01):
            continue
        motif = _assemble_motif(b00, b01)
        if np.array_equal(motif, np.fliplr(motif)):
            continue
        if np.array_equal(motif, np.flipud(motif)):
            continue
        return b00, b01
    raise AssertionError("failed to build non-degenerate C4 seed blocks")


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

    try:
        b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    except AssertionError:
        return None
    motif = _assemble_motif(b00, b01)
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
    raise AssertionError("failed to generate a valid db0c5428 D6 input")
