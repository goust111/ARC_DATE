# Combo variant of db0c5428 (index 9)
# combo_spec: D4/3+D6/1
# Name: 条件旋转+四重旋转对称

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
#    The visible input is still one 9x9 motif made of eight 3x3 blocks around a
#    missing 3x3 center. The fused change has two linked parts:
#    - D6/1 changes the block-to-block relation inside the motif from mirror-based
#      D4 to pure C4 rotation: if corner seed is A and edge seed is B, then
#      top-right=rot90cw(A), right=rot90cw(B), bottom-right=rot180(A), and so on.
#    - D4/3 then changes how the outer ring is copied: expansion is same-direction
#      as in the C4 variant, but only corner sources get one extra 90-degree CW
#      turn at copy time while edge sources are pasted unchanged.
#    Example trace: top-left corner A goes to the outer top-left slot as rot90cw(A);
#    top edge B goes to the outer top edge slot as B. Because the inner corner ring
#    already follows quarter-turn symmetry, the outer corner ring becomes the same
#    C4 cycle but phase-shifted by one turn relative to the inner corners.
# b) Complexity calibration:
#    There are still only two seed blocks per sample (one corner family, one edge
#    family), so the task does not explode into many case distinctions. The only
#    branch is the same one humans can see directly: corner positions rotate again,
#    edge positions do not.
# c) Hidden assumptions audit against main():
#    - bbox(g != bg) still finds exactly one contiguous 9x9 motif; generator keeps
#      all other cells at background.
#    - h == w and h % 3 == 0 still holds; the generator always builds a 3x3 block
#      arrangement with block size 3.
#    - _center_patch() assumes the hole is surrounded by eight visible boundary
#      pixels; C4 changes block orientation but not that local geometry.
#    - The outer-copy loop can no longer use the original opposite-side destination
#      formula; C4 requires same-direction expansion instead.
#    - The copied block itself can no longer always be pasted raw; corner sources
#      need an extra quarter-turn while edge sources do not.
# d) Placement legality:
#    The motif is still placed with a 3-cell pad inside a 16..19 square canvas, so
#    the implied 15x15 expanded region always fits fully in-bounds with no overlap.
# e) Visual observability:
#    If a seed block were invariant under 90-degree rotation, the C4 relation or the
#    extra corner rotation could become invisible. The generator therefore enforces
#    both seed blocks to be rotation-distinct and also rejects accidental LR/UD
#    mirror symmetry of the whole motif.
# f) Visual cleanliness:
#    No new markers or overlapping layers are introduced. The output remains the
#    same sparse 5x5 scaffold, with the only visible novelty being that the outer
#    corners advance one quarter-turn while the outer edges do not.
# g) Change inferability:
#    Colors keep the original role of majority-plus-minors; no logic depends on
#    numeric color ids. The rule is inferable from geometry alone: quarter-turn
#    relationships around the hole reveal C4, and comparing inner vs outer corners
#    reveals the extra conditional rotation.
# new main() assumptions and generator guarantees:
#    - main() assumes corner slots are identifiable purely by block coordinates
#      (br != 1 and bc != 1); generator always uses the standard 3x3 ring layout.
#    - main() assumes same-direction outer cells are empty background; motif padding
#      guarantees those slots are available.
#    - main() does not explicitly test C4 or non-degeneracy; generator enforces a
#      non-mirror C4 motif and visible corner rotation effect before accepting input.


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


def _is_corner_block(br: int, bc: int) -> bool:
    return br != 1 and bc != 1


def _outer_copy_transform(block: np.ndarray, br: int, bc: int) -> np.ndarray:
    if _is_corner_block(br, bc):
        return _rot90cw(block, 1)
    return block


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a C4-symmetric 3x3 block motif into a 5x5 frame with conditional corner turns.

    Input structure:
    - One 9x9 motif appears on a uniform background.
    - The motif is a 3x3 arrangement of 3x3 blocks with the center block missing.
    - The eight visible blocks form a quarter-turn cycle around the hole, and the
      eight pixels facing that hole encode the border of the missing center block.

    Rule:
    - Keep the original 9x9 motif in place and rebuild the missing center block
      from the inner-boundary pixels, using the motif's majority foreground color
      for the center cell.
    - Copy each visible block outward to the outer slot in the same direction from
      the center.
    - During that outward copy, rotate corner-source blocks 90 degrees clockwise,
      but leave edge-source blocks unchanged.

    Key insight:
    - The input already shows a rotational block pattern around the hole, so the
      outer ring should continue in the same directions; the only extra twist is
      that corners advance by one more quarter-turn when they are copied outward.
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
            paste(out, _outer_copy_transform(src, br, bc), dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (combo: C4 + conditional corner rotation) ---------------

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
            paste(expanded, _outer_copy_transform(src, br, bc), (2 * br) * BLOCK, (2 * bc) * BLOCK)
    return expanded


def _combo_behavior_visible(motif: np.ndarray) -> bool:
    d6_only = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    combo = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            dst_r = (2 * br) * BLOCK
            dst_c = (2 * bc) * BLOCK
            paste(d6_only, src, dst_r, dst_c)
            paste(combo, _outer_copy_transform(src, br, bc), dst_r, dst_c)
            if _is_corner_block(br, bc):
                if np.array_equal(src, _outer_copy_transform(src, br, bc)):
                    return False
            else:
                if not np.array_equal(src, _outer_copy_transform(src, br, bc)):
                    return False
    return not np.array_equal(d6_only, combo)


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
        if not _combo_behavior_visible(motif):
            continue
        return b00, b01
    raise AssertionError("failed to build non-degenerate combo seed blocks")


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
    raise AssertionError("failed to generate a valid db0c5428 combo input")
