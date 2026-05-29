# Combo variant of db0c5428 (index 31)
# combo_spec: D5/2+D11/2
# Name: 少数色中心+角落空洞

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve()
_SOLVER_DIR = _ROOT.parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and execution trace
#    - The finished object is still a 9x9 motif built from a 3x3 array of 3x3 blocks,
#      and the output is still the 15x15 point-symmetric 5x5 block expansion.
#    - D11/2 moves the missing block from the middle to one corner. The center block
#      is now present in the input, while exactly one corner block is all background.
#    - D5/2 still changes the "special center pixel" rule, but now it applies to the
#      center cell of the missing corner block rather than to the old middle hole.
#      The rest of the corner block is recovered from the diagonally opposite corner
#      by a 180-degree rotation.
#    - Small trace for a top-left hole:
#      1. Restore the top-left 3x3 block by rotating the bottom-right block.
#      2. Read the three hole-facing pixels from the visible neighbors:
#         top-middle block's middle-left cell, left-middle block's top-middle cell,
#         and the middle block's top-left cell.
#      3. Among those three colors, the one appearing once is written into the center
#         of the repaired corner block.
#      Original task used 8 samples around a center hole and filled the middle pixel
#      with a global majority color. This combo instead uses 3 corner-hole samples
#      and inverts the count logic from majority to minority.
# b) Complexity calibration
#    - The missing corner varies across the four corners, so the source change is
#      clearly visible across examples.
#    - The generator forces the two edge-side samples to match and the diagonal sample
#      to differ, producing a unique 2-vs-1 minority outcome instead of ambiguous
#      1-1-1 ties.
#    - The rotated opposite corner still determines 8 of the 9 repaired pixels, so the
#      task remains coherent rather than becoming a free-form recoloring puzzle.
# c) Hidden-assumption review of main()
#    - bbox-based localization still assumes one isolated 9x9 motif on a uniform
#      background. Erasing one corner does not shrink the bbox because the same outer
#      rows and cols are still occupied by the other visible blocks.
#    - The original _center_patch logic assumed the missing block was the middle block
#      and that all 8 neighbors existed. That fails here because only three blocks face
#      the corner hole.
#    - The original expansion loop skipped only the center because the center was the
#      missing block. Here the center is already present, so the solver must first
#      repair the missing corner inside a local motif copy and then expand from that
#      repaired motif.
#    - The minority-color rule assumes the three samples have a unique least-frequent
#      color. The generator guarantees a 2-1 count split.
# d) Placement legality
#    - The motif still sits with at least one full 3x3 block of padding on all sides,
#      so the 15x15 expanded output fits inside the 16..19 canvas.
#    - Only one corner block is blanked, so there is a unique hole and no overlap.
# e) Visual observability
#    - A human can see one all-background corner hole, the intact center block, and
#      the restored output corner. Across examples, the restored block matches the
#      opposite corner except for its center cell, whose color is explained by the
#      3-sample local minority.
#    - Because the generator fixes the edge-pair color and the diagonal color to be
#      different, the D5/2 inversion signal is active in every sample.
# f) Visual cleanliness
#    - There is still only one compact motif and one expanded result region on a plain
#      teal background. No auxiliary markers or overlapping objects are introduced.
# g) Color inference
#    - No rule depends on numeric color ids. The minority choice is based only on
#      visible multiplicities among three structural sample positions.
# 9) New solver assumptions and generator guarantees
#    - Assumption: exactly one corner block is all background.
#      Guarantee: generate_input() blanks exactly one random corner after the motif
#      has been fully constructed.
#    - Assumption: the opposite corner restores the missing block's non-center cells.
#      Guarantee: the motif is assembled from the original D4 corner/edge template.
#    - Assumption: the three hole-facing samples always have a unique minority color.
#      Guarantee: the edge-facing pair is forced to the majority color, while the
#      diagonal center-block sample is forced to a different corner-facing accent.
#    - Assumption: the fused rule differs from the D11-only baseline.
#      Guarantee: the rotated opposite corner's center is forced to majority, but the
#      combo-repaired center is forced to the different minority accent color.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


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
    corners = ((0, 0), (0, 2), (2, 0), (2, 2))
    missing = [
        (br, bc)
        for br, bc in corners
        if np.all(_extract_block(g, r0, c0, block, br, bc) == int(bg))
    ]
    assert len(missing) == 1, "expected exactly one missing corner block"
    return missing[0]


def _minority_color(colors: list[int]) -> int:
    counts = Counter(map(int, colors))
    min_count = min(counts.values())
    minority = [color for color, count in counts.items() if count == min_count]
    assert len(minority) == 1, "expected a unique minority color"
    return int(minority[0])


def _hole_facing_samples(g: np.ndarray, r0: int, c0: int, block: int, hole_br: int, hole_bc: int) -> list[int]:
    edge_bc = 1
    edge_br = 1
    h_local_r = 1
    h_local_c = 0 if hole_bc == 0 else block - 1
    v_local_r = 0 if hole_br == 0 else block - 1
    v_local_c = 1
    d_local_r = 0 if hole_br == 0 else block - 1
    d_local_c = 0 if hole_bc == 0 else block - 1

    samples = [
        int(g[int(r0) + hole_br * block + h_local_r, int(c0) + edge_bc * block + h_local_c]),
        int(g[int(r0) + edge_br * block + v_local_r, int(c0) + hole_bc * block + v_local_c]),
        int(g[int(r0) + edge_br * block + d_local_r, int(c0) + edge_bc * block + d_local_c]),
    ]
    return samples


def _restore_corner(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> tuple[tuple[int, int], np.ndarray]:
    hole_br, hole_bc = _missing_corner(g, r0, c0, block, bg)
    repaired = np.rot90(_extract_block(g, r0, c0, block, 2 - hole_br, 2 - hole_bc), 2)
    repaired[block // 2, block // 2] = _minority_color(
        _hole_facing_samples(g, r0, c0, block, hole_br, hole_bc)
    )
    return (hole_br, hole_bc), repaired


def _center_patch_from_motif(motif: np.ndarray, majority: int) -> np.ndarray:
    patch = motif[np.ix_(BLOCK + _CENTER_OFFSETS, BLOCK + _CENTER_OFFSETS)].copy()
    patch[1, 1] = majority
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a corner-missing block motif and expand it into the surrounding 5x5 layout.

    Input structure:
    - One 9x9 motif sits on a teal background and is partitioned into nine 3x3 block slots.
    - The middle slot is already filled, but exactly one corner slot is a blank 3x3 hole.
    - The visible blocks still follow the original D4 corner/edge template, and three
      visible pixels face the corner hole from its neighboring blocks.

    Rule:
    - Rebuild the missing corner block from the diagonally opposite corner by a 180-degree
      rotation.
    - Replace the repaired block's center cell with the minority color among the three
      hole-facing sample pixels from the adjacent horizontal block, adjacent vertical
      block, and the diagonal middle block.
    - Insert the repaired motif back into place, then copy every non-center block to its
      point-reflected outer slot in the implied 5x5 block grid.

    Key insight:
    - Moving the hole to a corner destroys the original 8-neighbor center reconstruction.
      The block shape now comes from opposite-corner symmetry, while the only non-symmetric
      decision is the repaired block's center color, chosen by a 3-sample minority vote
      around the corner hole.
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
    assert block == 3, "this task uses 3x3 blocks"

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


# --------------- generator (o5x combo) ---------------

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
_CENTER_OFFSETS = np.array([-1, 1, 3], dtype=int)


def _orbit_indices_containing(orbits, coords) -> set[int]:
    coord_set = set(coords)
    found = set()
    for idx, orbit in enumerate(orbits):
        if any(coord in coord_set for coord in orbit):
            found.add(idx)
    return found


def _major_orbit_options(orbits, forced_major, forced_minor, target_count) -> list[set[int]]:
    forced_major_idxs = _orbit_indices_containing(orbits, forced_major)
    forced_minor_idxs = _orbit_indices_containing(orbits, forced_minor)
    assert not (forced_major_idxs & forced_minor_idxs), "orbit cannot be both major and minor"

    base_count = sum(len(orbits[idx]) for idx in forced_major_idxs)
    usable = [
        idx
        for idx in range(len(orbits))
        if idx not in forced_major_idxs and idx not in forced_minor_idxs
    ]
    choices = []
    for mask in range(1 << len(usable)):
        selected = set(forced_major_idxs)
        total = base_count
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                selected.add(idx)
                total += len(orbits[idx])
        if total == target_count:
            choices.append(selected)
    return choices


def _seed_block(orbits, forced_major, forced_minor, majority, target_count):
    choices = _major_orbit_options(orbits, forced_major, forced_minor, target_count)
    assert choices, "unreachable symmetric majority count"
    major_idxs = random.choice(choices)
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
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


def _feasible_pairs(minor_color_count: int) -> list[tuple[int, int]]:
    pairs = []
    for corner_major in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1):
        corner_choices = _major_orbit_options(
            DIAG_ORBITS,
            forced_major=[(1, 1)],
            forced_minor=[(2, 2)],
            target_count=corner_major,
        )
        if not corner_choices:
            continue
        for edge_major in range(EDGE_MAJOR_MIN, EDGE_MAJOR_MAX + 1):
            total = corner_major + edge_major
            if not (MAJORITY_TOTAL_MIN <= total <= MAJORITY_TOTAL_MAX):
                continue
            edge_choices = _major_orbit_options(
                LR_ORBITS,
                forced_major=[(1, 0), (1, 2)],
                forced_minor=[(2, 1)],
                target_count=edge_major,
            )
            if not edge_choices:
                continue
            enough_minor_orbits = any(
                (len(DIAG_ORBITS) - len(csel)) + (len(LR_ORBITS) - len(esel)) >= minor_color_count
                for csel in corner_choices
                for esel in edge_choices
            )
            if enough_minor_orbits:
                pairs.append((corner_major, edge_major))
    assert pairs, "no feasible majority-count pairs"
    return pairs


def _make_blocks(majority, minor_colors, corner_major, edge_major):
    b00, corner_minors = _seed_block(
        DIAG_ORBITS,
        forced_major=[(1, 1)],
        forced_minor=[(2, 2)],
        majority=majority,
        target_count=corner_major,
    )
    b01, edge_minors = _seed_block(
        LR_ORBITS,
        forced_major=[(1, 0), (1, 2)],
        forced_minor=[(2, 1)],
        majority=majority,
        target_count=edge_major,
    )
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    assert int(b00[2, 2]) != int(majority)
    assert int(b00[1, 1]) == int(majority)
    assert int(b01[1, 0]) == int(majority) and int(b01[1, 2]) == int(majority)
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


def _completed_motif(majority, minor_colors, corner_major, edge_major):
    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    motif = _assemble_motif(b00, b01)
    center = _center_patch_from_motif(motif, majority)
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


def _combo_repaired_block(motif: np.ndarray, hole_corner: tuple[int, int]) -> np.ndarray:
    br, bc = hole_corner
    repaired = np.rot90(motif[(2 - br) * BLOCK : (3 - br) * BLOCK, (2 - bc) * BLOCK : (3 - bc) * BLOCK], 2)
    repaired = repaired.copy()
    repaired[1, 1] = _minority_color(_hole_facing_samples(motif, 0, 0, BLOCK, br, bc))
    return repaired


def _combo_signal_ok(full_motif: np.ndarray, hole_corner: tuple[int, int]) -> bool:
    br, bc = hole_corner
    samples = _hole_facing_samples(full_motif, 0, 0, BLOCK, br, bc)
    minority = _minority_color(samples)
    rotated_center = int(full_motif[(2 - br) * BLOCK + 1, (2 - bc) * BLOCK + 1])
    return minority != rotated_center


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    corner_major, edge_major = random.choice(_feasible_pairs(len(minor_colors)))
    full_motif = _completed_motif(majority, minor_colors, corner_major, edge_major)
    hole_corner = random.choice(CORNER_POSITIONS)
    if not _combo_signal_ok(full_motif, hole_corner):
        return None

    output_motif = full_motif.copy()
    paste(output_motif, _combo_repaired_block(full_motif, hole_corner), hole_corner[0] * BLOCK, hole_corner[1] * BLOCK)
    input_motif = _mask_corner(output_motif, hole_corner)
    expanded = _expanded_output(output_motif)

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
