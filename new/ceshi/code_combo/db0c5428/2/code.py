# Combo variant of db0c5428 (index 2)
# combo_spec: D1/0+D5/1
# Name: 旋转映射+对称性重建中心

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


# Design analysis for combo D1/0 + D5/1
# --------------------------------------
# a) Concrete behavior and trace:
#    - The input remains a single 9x9 motif made from eight visible 3x3 blocks
#      surrounding an empty 3x3 center hole on teal background.
#    - Change 1 (D1/0): the outer-ring copy no longer uses point symmetry. If a
#      source block is at motif block coordinate (br,bc), its outer-slot block
#      coordinate becomes (2*bc, 4-2*br), which is a 90-degree clockwise rotation
#      of the source direction around the motif center.
#    - Change 2 (D5/1): the missing center no longer reads the eight inward-facing
#      pixels around the hole. Instead each missing non-center cell is reflected
#      across both the LR and UD symmetry axes of the motif, so center(0,0) reads
#      from the visible bottom-right corner of the motif, center(0,1) from the
#      bottom-middle block's middle column, center(1,0) from the right-middle
#      block's middle row, and so on.
#    - Concrete trace:
#      original top-left source block b00 used to copy to the outer bottom-right
#      slot; now it copies to the outer top-right slot.
#      original center(0,0) used to read b00[2,2]; now it reads b22[2,2], which
#      equals b00[0,0] under the motif's D4 construction.
#      So both the outer layout and the center content differ from the base task.
# b) Complexity calibration:
#    - Because the motif itself is symmetric, some seeds could accidentally hide
#      the clockwise remapping or make the symmetry-repaired center match the old
#      inward-facing center. The generator rejects those cases.
#    - To keep the combo visually inferable, generation enforces:
#      1. all eight non-central center pixels differ from the legacy center rule,
#      2. the clockwise outer ring differs noticeably from the old point-symmetric
#         outer ring, and
#      3. the full combo output differs from both single-variant partial outputs.
# c) Hidden-assumption audit of main():
#    - `bbox(g != bg)` still assumes one isolated non-background motif; the
#      generator keeps exactly one 9x9 object on background.
#    - `h == w and h % 3 == 0` still assumes a square 3x3 block grid; generation
#      still builds exactly that.
#    - `block == 3` is now more important than before because both center-repair
#      index tables and the generator's symmetry orbits are written specifically
#      for 3x3 blocks.
#    - The old implicit assumption that inward-facing pixels encode the center is
#      no longer valid; `_center_patch(...)` must read from opposite-side symmetric
#      locations instead.
#    - The old implicit assumption that outer copies are point-symmetric is also
#      invalid; destination slots must use the clockwise mapping helper.
#    - `paste(...)` remains safe because the eight clockwise destinations are still
#      distinct positions on the 5x5 frame, and the central 9x9 motif stays in the
#      same place.
# d) Placement legality:
#    - The expanded output is still 15x15, so the original 3-cell padding around
#      the 9x9 motif remains sufficient to prevent clipping inside 16..19 grids.
# e) Visual observability:
#    - A human can see the outer placement rule from repeated corner/edge blocks:
#      top-left now reappears at outer top-right rather than outer bottom-right.
#    - A human can also see the center rule because the filled center matches the
#      motif's global LR/UD continuation instead of the pixels touching the hole.
#      The generator forces these two candidate centers to disagree at all eight
#      non-central cells.
# f) Visual cleanliness:
#    - The scene still contains one clean motif on a plain background and one clean
#      expanded 5x5 layout in the output. No extra markers or overlapping objects
#      are introduced.
# g) Color inference:
#    - No color-id ordering is used. Colors are sampled directly. The only special
#      color-frequency rule is the majority foreground color used for the center
#      fixed point, matching the original task family.
# 9) New assumptions guaranteed by generate_input():
#    - Corner seed b00 is diagonal-symmetric with b00[0,0] forced major and
#      b00[2,2] forced minor; edge seed b01 is LR-symmetric with b01[0,1] forced
#      major and b01[2,1] forced minor. This guarantees the symmetry center and
#      the legacy inward-facing center differ in both corner and edge channels.
#    - The generator rejects motifs whose clockwise ring is too similar to the old
#      point-symmetric ring, so the D1 change remains visible.
#    - The generator also rejects motifs where the full combo output matches either
#      partial single-variant output too closely, so the produced samples actually
#      exercise the fused rule rather than a degenerate subset of it.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _outer_slot_clockwise(br: int, bc: int) -> tuple[int, int]:
    """Map motif block (br,bc) to the outer 5x5 frame via a CW quarter-turn."""
    return 2 * int(bc), 4 - 2 * int(br)


def _outer_slot_point_symmetric(br: int, bc: int) -> tuple[int, int]:
    """Original task mapping, kept for generator-side signal checks."""
    return 4 - 2 * int(br), 4 - 2 * int(bc)


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this combo variant uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)

    src_rows = [int(r0) + 3 * block - 1, int(r0) + block + 1, int(r0)]
    src_cols = [int(c0) + 3 * block - 1, int(c0) + block + 1, int(c0)]
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[src_rows[r], src_cols[c]])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 frame while restoring its missing center.

    Input structure:
    - A single 9x9 motif sits on teal background as a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is absent, leaving a teal square hole.
    - The eight visible blocks form a globally symmetric ring around that hole.

    Rule:
    - Keep the original 9x9 motif in place.
    - Reconstruct the missing center block by reflecting each non-central cell
      across the motif's horizontal and vertical symmetry axes to the opposite
      visible side; fill the exact center with the motif's majority foreground color.
    - Copy each visible 3x3 block to the outer ring of the implied 5x5 block grid,
      but place it in the slot obtained by rotating its block position 90 degrees
      clockwise around the motif center.

    Key insight:
    The completed output combines two different global relations: the center is
    recovered from LR/UD symmetry through the hole, while the surrounding outer
    ring is organized by a block-level quarter-turn rather than by point symmetry.
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
            paste(out, src, expanded_r0 + dst_br * block, expanded_c0 + dst_bc * block)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (combo o5x) ---------------

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
MAX_RETRIES = 400
MIN_CENTER_DIFF = 8
MIN_RING_DIFF = 12
MIN_COMBO_DIFF = 20

# -- Symmetry orbits for 3x3 seed blocks --
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


def _orbit_indices_containing(orbits, coords):
    coord_set = set(coords)
    found = set()
    for idx, orbit in enumerate(orbits):
        if any(coord in coord_set for coord in orbit):
            found.add(idx)
    return found


def _major_orbits(orbits, forced_major, forced_minor, target_count):
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
    assert choices, "unreachable symmetric majority count"
    return random.choice(choices)


def _seed_block(orbits, forced_major, forced_minor, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_major, forced_minor, target_count)
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
    b00, corner_minors = _seed_block(
        DIAG_ORBITS,
        forced_major=[(0, 0)],
        forced_minor=[(2, 2)],
        majority=majority,
        target_count=corner_major,
    )
    b01, edge_minors = _seed_block(
        LR_ORBITS,
        forced_major=[(0, 1)],
        forced_minor=[(2, 1)],
        majority=majority,
        target_count=edge_major,
    )
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    assert int(b00[0, 0]) == int(majority)
    assert int(b00[2, 2]) != int(majority)
    assert int(b01[0, 1]) == int(majority)
    assert int(b01[2, 1]) != int(majority)
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


def _legacy_center_patch_from_motif(motif: np.ndarray, majority: int) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), majority, dtype=int)
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    for r in range(BLOCK):
        for c in range(BLOCK):
            if r == BLOCK // 2 and c == BLOCK // 2:
                continue
            patch[r, c] = int(motif[rows[r], cols[c]])
    return patch


def _symmetry_center_patch_from_motif(motif: np.ndarray, majority: int) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), majority, dtype=int)
    src_rows = [MOTIF_SIZE - 1, BLOCK + 1, 0]
    src_cols = [MOTIF_SIZE - 1, BLOCK + 1, 0]
    for r in range(BLOCK):
        for c in range(BLOCK):
            if r == BLOCK // 2 and c == BLOCK // 2:
                continue
            patch[r, c] = int(motif[src_rows[r], src_cols[c]])
    return patch


def _expanded_output(motif, center, slot_fn):
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
            paste(expanded, src, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _blocks_show_rotation_signal(b00, b01) -> bool:
    return (
        np.count_nonzero(b00 != np.flipud(b00)) >= 2
        and np.count_nonzero(b00 != np.rot90(b00, 2)) >= 2
        and np.count_nonzero(b01 != b01.T) >= 2
        and np.count_nonzero(b01 != np.rot90(b01, -1)) >= 2
    )


def _combo_metrics(motif: np.ndarray, majority: int) -> dict[str, int]:
    legacy_center = _legacy_center_patch_from_motif(motif, majority)
    symmetry_center = _symmetry_center_patch_from_motif(motif, majority)

    combo_out = _expanded_output(motif, symmetry_center, _outer_slot_clockwise)
    old_out = _expanded_output(motif, legacy_center, _outer_slot_point_symmetric)
    d1_only_out = _expanded_output(motif, legacy_center, _outer_slot_clockwise)
    d5_only_out = _expanded_output(motif, symmetry_center, _outer_slot_point_symmetric)

    mask = np.ones((BLOCK, BLOCK), dtype=bool)
    mask[BLOCK // 2, BLOCK // 2] = False
    center_diff = int(np.count_nonzero(symmetry_center[mask] != legacy_center[mask]))
    ring_diff = int(np.count_nonzero(combo_out != d5_only_out))
    combo_diff = int(np.count_nonzero(combo_out != old_out))
    diff_vs_d1_only = int(np.count_nonzero(combo_out != d1_only_out))
    diff_vs_d5_only = int(np.count_nonzero(combo_out != d5_only_out))
    return {
        "center_diff": center_diff,
        "ring_diff": ring_diff,
        "combo_diff": combo_diff,
        "diff_vs_d1_only": diff_vs_d1_only,
        "diff_vs_d5_only": diff_vs_d5_only,
    }


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
    if not _blocks_show_rotation_signal(b00, b01):
        return None

    motif = _assemble_motif(b00, b01)
    metrics = _combo_metrics(motif, majority)
    if metrics["center_diff"] < MIN_CENTER_DIFF:
        return None
    if metrics["ring_diff"] < MIN_RING_DIFF:
        return None
    if metrics["combo_diff"] < MIN_COMBO_DIFF:
        return None
    if metrics["diff_vs_d1_only"] < MIN_CENTER_DIFF:
        return None
    if metrics["diff_vs_d5_only"] < MIN_RING_DIFF:
        return None

    center = _symmetry_center_patch_from_motif(motif, majority)
    expanded = _expanded_output(motif, center, _outer_slot_clockwise)

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
