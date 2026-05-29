# Combo variant of db0c5428 (index 15)
# combo_spec: D1/0+D3/2
# Name: 旋转映射+环形排列

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# fused_rule:
#   - Keep the original source object: one 9x9 motif made from eight visible 3x3
#     blocks around a missing 3x3 center block.
#   - Keep the original center reconstruction: the 8 pixels facing the hole still
#     rebuild the border of the missing center block, and the center cell is still
#     the motif's majority foreground color.
#   - Replace the old rectangular 5x5 outer frame with an octagonal ring-like
#     arrangement. The motif sits in the middle of a 17x17 local patch, and the
#     eight external copies occupy the 8 compass directions around it:
#       N=(0,7), NE=(1,13), E=(7,14), SE=(13,13),
#       S=(14,7), SW=(13,1), W=(7,0), NW=(1,1)
#     in patch coordinates. This is visibly not a block-grid expansion because the
#     edge blocks sit farther out on cardinal axes while diagonal blocks sit on the
#     diagonals of an octagonal ring.
#
# concrete_behavior_trace:
#   - Original task:
#     source block (0,0) [NW of the hole] was copied to the opposite outer corner
#     of a 5x5 block grid; source block (0,1) [N] was copied to the bottom-middle.
#   - D1-only change:
#     source direction is rotated 90 degrees clockwise before choosing the external
#     slot, so NW -> NE and N -> E.
#   - D3-only change:
#     those external destinations are no longer interpreted as cells of a 5x5 grid,
#     but as 8 positions on a ring around the motif.
#   - Fused combo:
#     NW source block now lands in the ring's NE position; N lands at the ring's E
#     position; W lands at N; S lands at W. The block content itself is not rotated.
#     This makes the solver reason in two stages that reinforce each other:
#     first rotate the block-level direction, then read that direction on a ring.
#
# complexity_calibration:
#   - The ring uses one fixed octagonal layout instead of many possible radii or
#     offsets. That keeps the destination rule learnable from 2-4 examples.
#   - The original motif generator already creates asymmetry inside seed blocks, but
#     some seeds are still too close to the old point-symmetric answer. The combo
#     generator therefore rejects samples whose ring+CW output is too similar to:
#       1. ring+point-symmetric mapping  (tests D1 visibility)
#       2. grid+CW mapping               (tests D3 visibility)
#   - Additional block-level asymmetry checks prevent accidental invariance under
#     the transforms that separate the variants.
#
# implicit_assumption_audit:
#   - `_center_patch(...)` remains valid because center reconstruction depends only
#     on the inner boundary around the hole, not on the external copy layout.
#   - `bbox(...)` and `block = h // 3` still assume one square 3x3 block motif. The
#     generator preserves that assumption exactly.
#   - The original `expanded_r0 = r0 - block` and `expanded_c0 = c0 - block` assumed
#     a 15x15 rectangular frame whose destinations were multiples of `block`. That
#     assumption is broken by the ring layout and is replaced by fixed per-direction
#     offsets measured in raw pixels.
#   - The original `(4 - 2*br, 4 - 2*bc)` destination formula encoded point symmetry.
#     In the combo, source direction `(br-1, bc-1)` must first rotate to `(bc-1, 1-br)`
#     and only then be mapped to one of the 8 ring offsets.
#   - Every ring position is unique and disjoint from the central motif, so `paste`
#     remains safe with no overlaps.
#
# placement_legality:
#   - The ring patch is 17x17, with the motif anchored at offset `(4,4)` inside it.
#     Therefore the input motif must be placed with margin 4 on every side.
#   - Grid sizes are tightened from 16-19 to 17-19. Size 16 cannot contain the full
#     17x17 output patch, while size 17 allows the exact minimal placement at `(4,4)`.
#   - Because the ring offsets are fixed and all lie inside the 17x17 patch, pasting
#     the patch at `(r0-4, c0-4)` is always legal under the generator's placement rule.
#
# observability_and_cleanliness:
#   - The ring positions are separated from the motif by background gaps, so the
#     external copies read as distinct "orbiting" tiles rather than a crowded halo.
#   - No ordering or mapping depends on numeric color ids. The only cues are visible
#     source positions, visible rotation of those positions, and visible ring slots.
#   - Because the motif is D4-related but not fully invariant, examples show repeated
#     evidence that CW direction, not point symmetry, determines where each block goes.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
PATCH_SIZE = 17
MOTIF_PATCH_OFFSET = 4
GRID_PATCH_OFFSET = 1

BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

GRID_SIZE_MIN, GRID_SIZE_MAX = 17, 19
MOTIF_PAD = MOTIF_PATCH_OFFSET

MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

MAX_RETRIES = 400
ROTATION_DIFF_MIN = 12
LAYOUT_DIFF_MIN = 16

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

_RING_OFFSETS = {
    (-1, -1): (-3, -3),
    (-1, 0): (-4, 3),
    (-1, 1): (-3, 9),
    (0, 1): (3, 10),
    (1, 1): (9, 9),
    (1, 0): (10, 3),
    (1, -1): (9, -3),
    (0, -1): (3, -4),
}


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


def _rot90_cw(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), int(-dr)


def _point_reflect(dr: int, dc: int) -> tuple[int, int]:
    return int(-dr), int(-dc)


def _ring_offset_for_direction(dr: int, dc: int) -> tuple[int, int]:
    return _RING_OFFSETS[(int(dr), int(dc))]


def _ring_destination_offset(br: int, bc: int, dir_map) -> tuple[int, int]:
    src_dir = (int(br) - 1, int(bc) - 1)
    dst_dir = dir_map(*src_dir)
    return _ring_offset_for_direction(*dst_dir)


def _grid_slot_clockwise(br: int, bc: int) -> tuple[int, int]:
    return 2 * int(bc), 4 - 2 * int(br)


def _extract_block(g: np.ndarray, r0: int, c0: int, block: int, br: int, bc: int) -> np.ndarray:
    return crop_bbox(
        g,
        (
            int(r0) + int(br) * block,
            int(c0) + int(bc) * block,
            int(r0) + (int(br) + 1) * block - 1,
            int(c0) + (int(bc) + 1) * block - 1,
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center block, then place the surrounding blocks on an octagonal ring.

    Input structure:
    - One 9x9 motif sits on a uniform background.
    - The motif is a 3x3 arrangement of 3x3 blocks, with the middle block missing.
    - The eight visible pixels facing that hole encode the border of the missing center block.

    Rule:
    - Keep the original motif in place and reconstruct its center block from the inner-boundary cues.
    - For each visible source block, look at its direction relative to the missing center, rotate that
      direction 90 degrees clockwise, and use the rotated direction to choose an outer ring position.
    - Copy block contents unchanged into that ring slot; only the destination position changes.

    Key insight:
    The output is no longer a rectangular 5x5 block frame. The eight copies form a ring around the motif,
    so the correct mapping is best understood as "rotate the source direction, then read that direction on
    an octagonal ring" rather than "mirror to the opposite side."
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
    assert block == BLOCK, "this combo variant expects 3x3 blocks"

    out = g.copy()
    center = _center_patch(g, int(r0), int(c0), block, bg)
    paste(out, center, int(r0) + block, int(c0) + block)

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = _extract_block(g, int(r0), int(c0), block, br, bc)
            dr, dc = _ring_destination_offset(br, bc, _rot90_cw)
            paste(out, src, int(r0) + dr, int(c0) + dc)
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


def _assemble_motif(b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
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


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _render_ring_patch(motif: np.ndarray, center: np.ndarray, dir_map) -> np.ndarray:
    patch = np.full((PATCH_SIZE, PATCH_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(patch, output_motif, MOTIF_PATCH_OFFSET, MOTIF_PATCH_OFFSET)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            dr, dc = _ring_destination_offset(br, bc, dir_map)
            paste(patch, src, MOTIF_PATCH_OFFSET + dr, MOTIF_PATCH_OFFSET + dc)
    return patch


def _render_grid_patch(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    patch = np.full((PATCH_SIZE, PATCH_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(patch, output_motif, MOTIF_PATCH_OFFSET, MOTIF_PATCH_OFFSET)
    expanded_r0 = GRID_PATCH_OFFSET
    expanded_c0 = GRID_PATCH_OFFSET
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            dst_br, dst_bc = _grid_slot_clockwise(br, bc)
            paste(patch, src, expanded_r0 + dst_br * BLOCK, expanded_c0 + dst_bc * BLOCK)
    return patch


def _combo_metrics(motif: np.ndarray, center: np.ndarray) -> tuple[int, int]:
    combo = _render_ring_patch(motif, center, _rot90_cw)
    ring_point = _render_ring_patch(motif, center, _point_reflect)
    grid_cw = _render_grid_patch(motif, center)
    rotation_diff = int(np.count_nonzero(combo != ring_point))
    layout_diff = int(np.count_nonzero(combo != grid_cw))
    return rotation_diff, layout_diff


def _blocks_show_rotation_signal(b00: np.ndarray, b01: np.ndarray) -> bool:
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

    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    if not _blocks_show_rotation_signal(b00, b01):
        return None

    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    rotation_diff, layout_diff = _combo_metrics(motif, center)
    if rotation_diff < ROTATION_DIFF_MIN or layout_diff < LAYOUT_DIFF_MIN:
        return None

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, _render_ring_patch(motif, center, _rot90_cw), r0 - MOTIF_PATCH_OFFSET, c0 - MOTIF_PATCH_OFFSET)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
