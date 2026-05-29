# Combo variant of db0c5428 (index 22)
# combo_spec: D1/0+D4/3+D6/1
# Name: 旋转映射+条件旋转+C4对称

from __future__ import annotations

import random
import sys

import numpy as np

_SOLVERS_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVERS_DIR not in sys.path:
    sys.path.append(_SOLVERS_DIR)

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Input remains one 9x9 motif: eight visible 3x3 blocks around a missing
#      3x3 center hole on teal background.
#    - D6/1 changes the motif relation to pure C4 rotation. If the top-left
#      corner seed is A and the top edge seed is B, then:
#        top-right=rot90cw(A), right=rot90cw(B), bottom-right=rot180(A),
#        bottom=rot180(B), bottom-left=rot270(A), left=rot270(B).
#    - D1/0 changes the outer-slot mapping. Block (br,bc) no longer goes to the
#      point-reflected slot (4-2*br, 4-2*bc); it goes to the slot obtained by a
#      block-level 90-degree clockwise turn: (2*bc, 4-2*br).
#    - D4/3 then changes the copied block itself: source corners rotate another
#      90 degrees clockwise before pasting, while source edges keep orientation.
#    - Trace on one corner: input top-left A -> outer top-right rot90cw(A).
#      Trace on one edge: input top edge B -> outer right edge B.
#      This creates the intended three-layer rotation stack:
#        input orientation from C4, slot direction from D1, copy transform from D4.
#
# b) Complexity calibration:
#    - The branch space is intentionally small and repeated: only two seed block
#      families exist (corner seed A, edge seed B), each reused four times.
#    - Corners all share the same "extra quarter-turn" rule; edges all share the
#      same "no extra turn" rule. This is enough to make the combo visible without
#      exploding into many unrelated cases.
#    - A useful emergent property is that corners end up continuing the natural
#      C4 outer corner orbit, while edges do not. That contrast makes the fusion
#      easier to infer from 2-4 examples than three unrelated changes would be.
#
# c) Implicit assumption audit against main():
#    - `bbox(g != bg)` still assumes one isolated foreground motif; generator keeps
#      a single 9x9 motif on background.
#    - `h == w and h % 3 == 0` still assumes a square 3x3 block grid; generator
#      still builds exactly that structure.
#    - `_center_patch(...)` only depends on the eight pixels facing the hole and
#      on a majority foreground color; C4 and outer-copy changes do not break it.
#    - The original destination formula encoded point symmetry and is invalid here;
#      it must be replaced by the clockwise slot mapping.
#    - The original raw `paste(out, src, ...)` assumption is also invalid here,
#      because copied corners now need an extra rotation.
#    - No new overlap assumptions are introduced: the eight destinations remain
#      eight distinct outer slots of the same 5x5 scaffold.
#
# d) Placement legality:
#    - The output footprint is still a 15x15 expanded frame around the 9x9 motif.
#      The original 3-cell placement pad is therefore still sufficient.
#    - Generator places the motif at `(r0,c0)` with `r0,c0 >= 3` and with enough
#      room on the bottom/right so the 15x15 output never clips.
#
# e) Visual observability:
#    - To prevent degeneration, generator rejects blocks that are invariant under
#      quarter-turns and rejects motifs whose combo output is too close to simpler
#      ablations: no-corner-rotation, point-symmetric mapping, or same-direction
#      C4 extension.
#    - The rejection metric is purely pixel-visible. No rule depends on numeric
#      ordering of color ids or any hidden metadata.
#
# f) Visual cleanliness:
#    - The layout stays the same sparse 5x5 block frame, so nothing overlaps and
#      the changed behavior is isolated to orientation/placement of known blocks.
#    - Only 2-4 foreground colors are used and the same two seed families support
#      all eight visible blocks, keeping the pattern legible.
#
# g) New main() assumptions and generator guarantees:
#    - main() assumes the input really is a C4-arranged 3x3 block motif, but it
#      does not verify that explicitly; generator enforces it.
#    - main() assumes corner-vs-edge type is recoverable from block position; in
#      this task that is guaranteed by the fixed 3x3 motif layout.


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


def _outer_slot_clockwise(br: int, bc: int) -> tuple[int, int]:
    return 2 * int(bc), 4 - 2 * int(br)


def _outer_slot_point_symmetric(br: int, bc: int) -> tuple[int, int]:
    return 4 - 2 * int(br), 4 - 2 * int(bc)


def _outer_slot_same_direction(br: int, bc: int) -> tuple[int, int]:
    return 2 * int(br), 2 * int(bc)


def _is_corner_block(br: int, bc: int) -> bool:
    return br != 1 and bc != 1


def _identity_transform(block: np.ndarray, br: int, bc: int) -> np.ndarray:
    del br, bc
    return block


def _combo_copy_transform(block: np.ndarray, br: int, bc: int) -> np.ndarray:
    if _is_corner_block(br, bc):
        return _rot90cw(block, 1)
    return block


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a C4-symmetric 3x3 block motif into a rotated outer frame around its hole.

    Input structure:
    - A teal background contains one 9x9 motif made from a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is missing, leaving a background-colored hole.
    - The eight visible blocks around that hole form a quarter-turn orbit rather
      than a mirror-symmetric one.

    Rule:
    - Keep the original 9x9 motif in place.
    - Rebuild the center block from the eight pixels facing the hole, with the
      center cell set to the motif's majority foreground color.
    - Copy each visible block to the outer ring slot obtained by rotating its
      block position 90 degrees clockwise; when the source block is a corner,
      rotate the copied 3x3 block 90 degrees clockwise again before pasting.

    Key insight:
    The solver must combine three rotation layers at once: the input motif already
    rotates around the hole, the outer destination is chosen by another quarter
    turn at block level, and only corner copies receive one more quarter-turn.
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
            paste(out, _combo_copy_transform(src, br, bc), dst_r, dst_c)

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
BLOCK_RETRIES = 250
CORNER_ROTATION_DIFF_MIN = 8
MAPPING_DIFF_MIN = 16
EDGE_PHASE_DIFF_MIN = 8
ORIGINAL_DIFF_MIN = 24

_CENTER_OFFSETS = np.array([-1, 1, 3])
_ALL_CELLS = [(r, c) for r in range(BLOCK) for c in range(BLOCK)]


def _is_rotation_distinct(block: np.ndarray) -> bool:
    return all(not np.array_equal(block, _rot90cw(block, k)) for k in (1, 2, 3))


def _same_rotation_family(a: np.ndarray, b: np.ndarray) -> bool:
    return any(np.array_equal(_rot90cw(a, k), b) for k in range(4))


def _rotation_gap(block: np.ndarray) -> int:
    return int(np.count_nonzero(block != _rot90cw(block, 1)))


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


def _expanded_output(
    motif: np.ndarray,
    center: np.ndarray,
    slot_fn,
    transform_fn,
) -> np.ndarray:
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
            paste(
                expanded,
                transform_fn(src, br, bc),
                dst_br * BLOCK,
                dst_bc * BLOCK,
            )
    return expanded


def _variant_signal(motif: np.ndarray, center: np.ndarray) -> dict[str, int]:
    combo = _expanded_output(motif, center, _outer_slot_clockwise, _combo_copy_transform)
    no_corner_turn = _expanded_output(motif, center, _outer_slot_clockwise, _identity_transform)
    point_mapped = _expanded_output(motif, center, _outer_slot_point_symmetric, _combo_copy_transform)
    c4_natural = _expanded_output(motif, center, _outer_slot_same_direction, _identity_transform)
    original = _expanded_output(motif, center, _outer_slot_point_symmetric, _identity_transform)
    return {
        "corner_rotation_diff": int(np.count_nonzero(combo != no_corner_turn)),
        "mapping_diff": int(np.count_nonzero(combo != point_mapped)),
        "edge_phase_diff": int(np.count_nonzero(combo != c4_natural)),
        "original_diff": int(np.count_nonzero(combo != original)),
    }


def _make_blocks(majority: int, minor_colors: list[int], corner_major: int, edge_major: int):
    for _ in range(BLOCK_RETRIES):
        b00, corner_minors = _seed_block([(2, 2)], majority, corner_major)
        b01, edge_minors = _seed_block([(2, 1)], majority, edge_major)
        minor_cells = corner_minors + edge_minors
        if len(minor_cells) < len(minor_colors):
            continue
        _assign_minor_colors(minor_cells, minor_colors)
        if _rotation_gap(b00) < 2 or _rotation_gap(b01) < 2:
            continue
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
        center = _gen_center_patch(motif, majority)
        signal = _variant_signal(motif, center)
        if signal["corner_rotation_diff"] < CORNER_ROTATION_DIFF_MIN:
            continue
        if signal["mapping_diff"] < MAPPING_DIFF_MIN:
            continue
        if signal["edge_phase_diff"] < EDGE_PHASE_DIFF_MIN:
            continue
        if signal["original_diff"] < ORIGINAL_DIFF_MIN:
            continue
        return b00, b01
    raise AssertionError("failed to build a visible combo motif")


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
    expanded = _expanded_output(motif, center, _outer_slot_clockwise, _combo_copy_transform)

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
