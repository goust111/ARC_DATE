# Combo variant of db0c5428 (index 30)
# combo_spec: D1/2+D8/1
# Name: 按距离映射+非正方形块

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo D1/2 + D8/1 variant
#
# a) Concrete trace and behavior change
#    - The visible motif is no longer 9x9; it is a 9x6 rectangle made from a 3x3
#      arrangement of 3x2 blocks, with the middle 3x2 block missing.
#    - Center reconstruction follows the rectangular-hole sampling from D8/1:
#      rows use offsets [-1, +1, +3], cols use offsets [-1, +2], so the missing
#      3x2 patch is read from the six hole-facing pixels around the gap.
#    - The D1/2 change is applied at block-placement level. We measure block
#      distance in the rectangular lattice, so one vertical block step counts as
#      3 rows while one horizontal block step counts as 2 cols.
#      source left-middle  relative offset ( 0, -2) -> nearest tier
#      source top-middle   relative offset (-3,  0) -> middle tier
#      source top-left     relative offset (-3, -2) -> farthest tier
#    - The outer 5x5 lattice already contains three rectangular-distance tiers:
#      left/right midpoints have squared distance 16, top/bottom midpoints 36,
#      corners 52. Those match the rank order of source distances 4, 9, 13.
#    - Therefore each source block is projected outward in the same direction:
#      (0,1)->top midpoint, (1,0)->left midpoint, corners->same corner. This is
#      not the original opposite-side copy; e.g. top-left now goes to outer
#      top-left instead of outer bottom-right.
#
# b) Complexity calibration
#    - The fusion creates exactly three distance tiers, not an open-ended metric.
#      That is enough to make the rectangular geometry visible while still being
#      learnable from 2-4 examples.
#    - Each tier has repeated support: 4 corners, 2 vertical edge blocks, 2
#      horizontal edge blocks. The solver does not need color-based tie-breaking.
#
# c) Hidden-assumption audit against the original code
#    - Original `h == w and h % 3 == 0` assumed square blocks and a square motif.
#      That breaks here; the new invariant is exact motif size (9, 6).
#    - Original D4 assembly/transposes are invalid for 3x2 blocks because a 90°
#      turn would become 2x3. The generator uses only LR/UD/180 relations.
#    - Original outer-slot formula `(4 - 2*br, 4 - 2*bc)` assumed point symmetry.
#      That assumption is replaced with same-direction outward projection whose
#      tier is determined by rectangular distance from the center.
#    - Original center patch assumed a 3x3 square with a unique middle pixel.
#      Here the missing center is 3x2, so every cell is sampled from neighbors.
#
# d) Placement legality
#    - Motif size is 9x6 and expanded output size is 15x10.
#    - Input placement uses `r0 >= 3`, `c0 >= 2`, `r0 + 12 <= N`, `c0 + 8 <= N`,
#      so pasting the expanded region at `(r0-3, c0-2)` is always in bounds.
#    - The chosen outer slots are the standard eight non-overlapping positions of
#      the 5x5 frame, so copies never collide with one another or the motif.
#
# e) Visual observability
#    - The human-visible cue is geometric, not numeric color ordering:
#      left/right neighbors are visibly closer to the hole than top/bottom
#      neighbors because blocks are 3x2, and the output preserves that ranking in
#      the outer frame.
#    - The generator keeps corner, top, and side seed blocks asymmetric enough
#      that same-direction mapping differs visibly from the old opposite mapping.
#
# f) Cleanliness
#    - There is still exactly one motif and one clean expanded frame.
#    - No markers or overlapping annotations are introduced; only block placement
#      and the rectangular center reconstruction change.
#
# g) Anti-degeneration metric
#    - `variant_gap` counts pixel differences between this combo's outer-frame
#      output and the rectangular point-symmetric D8-only output on the same motif.
#      Samples with too small a gap are rejected so the generator does not drift
#      back toward the original copy rule.
#
# h) New solver assumptions guaranteed by the generator
#    - The foreground bbox is exactly one 9x6 motif on background color 8.
#    - All eight visible blocks are fully defined and occupy the expected 3x2
#      slots, so the six center samples always exist and are non-background.
#    - Distinct opposite-side blocks are usually not identical under the mirrors
#      that would hide the D1 change; the generator's gap check enforces this.
# ---------------------------------------------------------------------------


BLOCK_H = 3
BLOCK_W = 2
MOTIF_H = BLOCK_H * 3
MOTIF_W = BLOCK_W * 3
EXPANDED_H = BLOCK_H * 5
EXPANDED_W = BLOCK_W * 5

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD_R = BLOCK_H
MOTIF_PAD_C = BLOCK_W

BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

CORNER_MAJOR_CHOICES = (4, 5)
TOP_MAJOR_CHOICES = (2, 4)
SIDE_MAJOR_CHOICES = (3, 4, 5)

MAX_RETRIES = 400
MIN_VARIANT_GAP = 12

SOURCE_POSITIONS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)

CORNER_ORBITS = tuple(((r, c),) for r in range(BLOCK_H) for c in range(BLOCK_W))
LR_ORBITS = (
    ((0, 0), (0, 1)),
    ((1, 0), (1, 1)),
    ((2, 0), (2, 1)),
)
UD_ORBITS = (
    ((0, 0), (2, 0)),
    ((0, 1), (2, 1)),
    ((1, 0),),
    ((1, 1),),
)


def _center_patch(g: np.ndarray, r0: int, c0: int) -> np.ndarray:
    hole_r = int(r0) + BLOCK_H
    hole_c = int(c0) + BLOCK_W
    row_offsets = (-1, 1, 3)
    col_offsets = (-1, 2)
    rows = [hole_r + dr for dr in row_offsets]
    cols = [hole_c + dc for dc in col_offsets]
    return g[np.ix_(rows, cols)].copy()


def _rect_sqdist(dr_blocks: int, dc_blocks: int, step_r: int, step_c: int) -> int:
    return (abs(int(dr_blocks)) * step_r) ** 2 + (abs(int(dc_blocks)) * step_c) ** 2


def _outer_slot_by_rect_distance(br: int, bc: int) -> tuple[int, int]:
    """
    Project a source block outward in its own direction, using rectangular
    row/col scales so horizontal neighbors, vertical neighbors, and corners form
    three distinct distance tiers.
    """
    dr = int(br) - 1
    dc = int(bc) - 1
    sq = _rect_sqdist(dr, dc, BLOCK_H, BLOCK_W)

    if sq == BLOCK_W ** 2:
        assert dr == 0 and dc != 0
        return 2, 0 if dc < 0 else 4
    if sq == BLOCK_H ** 2:
        assert dc == 0 and dr != 0
        return (0 if dr < 0 else 4), 2

    assert sq == BLOCK_H ** 2 + BLOCK_W ** 2 and dr != 0 and dc != 0
    return (0 if dr < 0 else 4), (0 if dc < 0 else 4)


def _outer_slot_point_symmetric(br: int, bc: int) -> tuple[int, int]:
    return 4 - 2 * int(br), 4 - 2 * int(bc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a rectangular 3x3 block motif into a 5x5 frame using distance-ranked slots.

    Input structure:
    - A square teal background contains one 9x6 foreground motif.
    - The motif is a 3x3 arrangement of 3x2 rectangular blocks, with the middle
      3x2 block missing.
    - The six pixels that face the rectangular hole encode the missing center block.

    Rule:
    - Keep the original 9x6 motif in place and rebuild the missing 3x2 center
      directly from those six hole-facing samples.
    - Copy each visible 3x2 block to the outer 5x5 frame without rotating it.
    - Choose the destination by rectangular distance from the center: left/right
      neighbors go to the closer side midpoints, top/bottom neighbors go to the
      farther top/bottom midpoints, and corners go to the farthest corner slots,
      always staying in the same outward direction as the source block.

    Key insight:
    The block lattice is rectangular rather than square, so "one step away" is
    not isotropic: horizontal neighbors are physically closer than vertical ones.
    The output copies preserve that three-tier distance order instead of using the
    original task's opposite-side point symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert (h, w) == (MOTIF_H, MOTIF_W), "motif must be a 9x6 rectangular 3x3 block grid"

    out = g.copy()
    expanded_r0 = int(r0) - BLOCK_H
    expanded_c0 = int(c0) - BLOCK_W

    for br, bc in SOURCE_POSITIONS:
        src = crop_bbox(
            g,
            (
                int(r0) + br * BLOCK_H,
                int(c0) + bc * BLOCK_W,
                int(r0) + (br + 1) * BLOCK_H - 1,
                int(c0) + (bc + 1) * BLOCK_W - 1,
            ),
        )
        dst_br, dst_bc = _outer_slot_by_rect_distance(br, bc)
        paste(out, src, expanded_r0 + dst_br * BLOCK_H, expanded_c0 + dst_bc * BLOCK_W)

    paste(out, _center_patch(g, int(r0), int(c0)), int(r0) + BLOCK_H, int(c0) + BLOCK_W)
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
    assert choices, "unreachable majority count for chosen symmetry"
    return random.choice(choices)


def _seed_block(orbits, forced_minor, majority, target_count):
    block = np.full((BLOCK_H, BLOCK_W), -1, dtype=int)
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


def _make_blocks(majority, minor_colors, corner_major, top_major, side_major):
    # Forced minor cells are exactly the six cells sampled into the center patch.
    b00, corner_minors = _seed_block(CORNER_ORBITS, [(2, 1)], majority, corner_major)
    b01, top_minors = _seed_block(LR_ORBITS, [(2, 0)], majority, top_major)
    b10, side_minors = _seed_block(UD_ORBITS, [(1, 1)], majority, side_major)
    minor_orbits = corner_minors + top_minors + side_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1) and np.all(b10 != -1)
    return b00, b01, b10


def _assemble_motif(b00, b01, b10):
    motif = np.full((MOTIF_H, MOTIF_W), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): np.fliplr(b00),
        (1, 0): b10,
        (1, 2): np.fliplr(b10),
        (2, 0): np.flipud(b00),
        (2, 1): np.flipud(b01),
        (2, 2): np.flipud(np.fliplr(b00)),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK_H, bc * BLOCK_W)
    return motif


def _expanded_output(motif, center, slot_fn):
    expanded = np.full((EXPANDED_H, EXPANDED_W), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK_H, BLOCK_W)
    paste(expanded, output_motif, BLOCK_H, BLOCK_W)
    for br, bc in SOURCE_POSITIONS:
        src = motif[
            br * BLOCK_H:(br + 1) * BLOCK_H,
            bc * BLOCK_W:(bc + 1) * BLOCK_W,
        ]
        dst_br, dst_bc = slot_fn(br, bc)
        paste(expanded, src, dst_br * BLOCK_H, dst_bc * BLOCK_W)
    return expanded


def _variant_gap(expanded_new: np.ndarray, expanded_old: np.ndarray) -> int:
    return int(np.count_nonzero(expanded_new != expanded_old))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    corner_major = random.choice(CORNER_MAJOR_CHOICES)
    top_major = random.choice(TOP_MAJOR_CHOICES)
    side_major = random.choice(SIDE_MAJOR_CHOICES)

    b00, b01, b10 = _make_blocks(majority, minor_colors, corner_major, top_major, side_major)
    motif = _assemble_motif(b00, b01, b10)
    center = _center_patch(motif, 0, 0)
    expanded = _expanded_output(motif, center, _outer_slot_by_rect_distance)
    original_like = _expanded_output(motif, center, _outer_slot_point_symmetric)
    if _variant_gap(expanded, original_like) < MIN_VARIANT_GAP:
        return None

    r0 = random.randint(MOTIF_PAD_R, n - MOTIF_H - MOTIF_PAD_R)
    c0 = random.randint(MOTIF_PAD_C, n - MOTIF_W - MOTIF_PAD_C)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK_H, c0 - BLOCK_W)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
