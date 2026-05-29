# Combo variant of db0c5428 (index 11)
# combo_spec: D5/1+D15/0
# Name: 对称性重建中心+块内对称

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve()
_SOLVER_DIR = _ROOT.parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D5/1 + D15/0 combo
# a) Concretization + trace:
#    The visible object is still a single 9x9 ring of eight 3x3 blocks around a
#    blank 3x3 hole, and the output still expands it to the same 15x15 5x5 block
#    layout. The fusion changes two linked layers:
#    - D5/1 changes center repair. Example with majority color M:
#      corner seed b00 can be [[M,a,M],[b,c,b],[d,e,d]] (LR-symmetric, not UD),
#      top-edge seed b01 can be [[f,M,f],[g,h,g],[i,j,i]] (LR-symmetric, not UD).
#      Original db0c5428 would set center(0,0)=b00[2,2]=d and center(0,1)=b01[2,1]=j
#      from the pixels facing the hole. The fused rule instead uses opposite-side
#      global symmetry: center(0,0)=bottom-right[2,2]=b00[0,0]=M and
#      center(0,1)=bottom-middle[2,1]=b01[0,1]=M.
#    - D15/0 changes the block vocabulary. Every visible 3x3 block must itself have
#      an axis symmetry. Here the generator makes seed blocks LR-symmetric; after
#      transpose/rotation, the left/right edge blocks become UD-symmetric, so the
#      ring visibly mixes local LR and UD symmetry while the whole motif keeps its
#      global LR+UD symmetry.
#    The recovered center now inherits both global and local regularities, so the
#    finished center block is itself axis-symmetric instead of being just an
#    unconstrained 3x3 patch.
# b) Complexity calibration:
#    The task is still a single-object geometric completion problem, but the center
#    can no longer be guessed from the inward-facing pixels alone. The generator
#    forces b00[0,0] != b00[2,2] and b01[0,1] != b01[2,1], which makes all eight
#    non-central cells of the D5 center disagree with the original rule. At the
#    same time, each visible block has obvious internal symmetry, providing repeated
#    support for the extra D15 cue instead of hiding it in one accidental block.
# c) Hidden-assumption audit of main():
#    - bbox/get_bg still assume one isolated non-background motif on a plain canvas.
#    - The expansion loop still assumes a square 3x3 block grid with equal block
#      size; the generator keeps block==3 exactly.
#    - The new center repair assumes global LR and UD symmetry of the ring, because
#      it reads the hole from the opposite visible bands instead of the inward
#      boundary samples.
#    - No new matching or ordering logic is introduced, so the only fragile part is
#      the center reconstruction source map. The generator guarantees those source
#      positions are populated by the eight visible blocks and are globally aligned.
# d) Placement legality:
#    The motif is still placed with a 3-cell margin inside a 16..19 square grid, so
#    the 15x15 expanded output always fits. There is only one object, so no overlap
#    or competition between placements can occur.
# e) Visual observability:
#    Across examples, a human can see two stable facts: each visible 3x3 block has
#    its own mirror axis, and the missing center matches the opposite blocks rather
#    than the inward-facing boundary pixels. Because the generator enforces an
#    8-pixel disagreement with the legacy rule, this difference is not subtle.
# f) Visual cleanliness:
#    The scene remains one compact motif on a teal background. No markers, numbers,
#    or extra objects are added, so the nested symmetry signal is concentrated in
#    the block patterns themselves and stays easy to read.
# g) Change inferability:
#    Colors are sampled directly from the normal ARC palette without any hidden
#    ordering. The only color-frequency cue is the majority foreground color for the
#    center fixed point, while all other center cells are justified structurally by
#    opposite-side symmetry and validated by block-internal symmetry.
# 9) New main() assumptions guaranteed by generate_input():
#    - The 9x9 ring is globally symmetric under both left-right and up-down flips.
#    - Every visible 3x3 block has at least one axis symmetry, and the sample set
#      contains both strict LR-symmetric and strict UD-symmetric visible blocks.
#    - The D5 center reconstructed from opposite blocks is itself axis-symmetric.
#    - The D5 center differs from the original inward-boundary center in all eight
#      non-central cells, so the combo never degenerates back to the base task.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _block_axis_kind(block: np.ndarray) -> str:
    lr = np.array_equal(block, np.fliplr(block))
    ud = np.array_equal(block, np.flipud(block))
    if lr and ud:
        return "both"
    if lr:
        return "lr"
    if ud:
        return "ud"
    return "none"


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this combo variant uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)

    # Repair the hole by reflecting across both global axes of the 9x9 ring.
    src_rows = [int(r0) + 3 * block - 1, int(r0) + block + 1, int(r0)]
    src_cols = [int(c0) + 3 * block - 1, int(c0) + block + 1, int(c0)]
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[src_rows[r], src_cols[c]])
    return patch


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a symmetric 3x3-block ring into a 5x5 layout and restore its center.

    Input structure:
    - A teal canvas contains one 9x9 motif made from eight visible 3x3 blocks.
    - Those blocks form a 3x3 arrangement with the middle 3x3 block missing.
    - The full ring is symmetric across both the left-right and up-down axes, and
      each visible block also has its own mirror symmetry.

    Rule:
    - Keep the original 9x9 motif in place.
    - Copy each visible 3x3 block to the point-symmetric outer slot of the implied
      5x5 block layout, without rotating the copied block.
    - Rebuild the missing center block by reading each non-central cell from the
      opposite side of the ring under global LR+UD symmetry; set the center cell to
      the motif's majority foreground color.

    Key insight: The center is recovered from a nested symmetry pattern. The whole
    ring tells you which opposite block supplies each missing pixel, and the fact
    that every 3x3 block is itself mirror-symmetric provides an extra consistency
    check on the repaired center.
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
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, src, dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (o5x combo D5/1 + D15/0) ---------------

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
MIN_VARIANT_DIFF = 8

# Every seed block is constructed as LR-symmetric. After transpose/rotation the
# side blocks become UD-symmetric, so the full motif visibly mixes both axis types.
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)

# Original task sampled the center from these inner-boundary coordinates.
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
    if not choices:
        return None
    return random.choice(choices)


def _seed_block(orbits, forced_major, forced_minor, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_major, forced_minor, target_count)
    if major_idxs is None:
        return None

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
    corner_seed = _seed_block(
        LR_ORBITS,
        forced_major=[(0, 0)],
        forced_minor=[(2, 2)],
        majority=majority,
        target_count=corner_major,
    )
    edge_seed = _seed_block(
        LR_ORBITS,
        forced_major=[(0, 1)],
        forced_minor=[(2, 1)],
        majority=majority,
        target_count=edge_major,
    )
    if corner_seed is None or edge_seed is None:
        return None

    b00, corner_minors = corner_seed
    b01, edge_minors = edge_seed
    minor_orbits = corner_minors + edge_minors
    if len(minor_orbits) < len(minor_colors):
        return None

    _assign_minor_colors(minor_orbits, minor_colors)
    if not (np.all(b00 != -1) and np.all(b01 != -1)):
        return None
    if _block_axis_kind(b00) != "lr" or _block_axis_kind(b01) != "lr":
        return None
    if int(b00[0, 0]) == int(b00[2, 2]):
        return None
    if int(b01[0, 1]) == int(b01[2, 1]):
        return None
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


def _visible_block_axis_kinds(motif):
    kinds = []
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            block = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            kinds.append(_block_axis_kind(block))
    return kinds


def _motif_has_global_axes(motif):
    return np.array_equal(motif, np.fliplr(motif)) and np.array_equal(motif, np.flipud(motif))


def _gen_center_patch(motif, majority):
    return _symmetry_center_patch_from_motif(motif, majority)


def _expanded_output(motif, center):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _variant_diff(motif, majority) -> int:
    new_center = _symmetry_center_patch_from_motif(motif, majority)
    old_center = _legacy_center_patch_from_motif(motif, majority)
    mask = np.ones((BLOCK, BLOCK), dtype=bool)
    mask[BLOCK // 2, BLOCK // 2] = False
    return int(np.count_nonzero(new_center[mask] != old_center[mask]))


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

    seeds = _make_blocks(majority, minor_colors, corner_major, edge_major)
    if seeds is None:
        return None
    b00, b01 = seeds
    motif = _assemble_motif(b00, b01)

    if not _motif_has_global_axes(motif):
        return None
    axis_kinds = _visible_block_axis_kinds(motif)
    if any(kind == "none" for kind in axis_kinds):
        return None
    if "lr" not in axis_kinds or "ud" not in axis_kinds:
        return None

    center = _gen_center_patch(motif, majority)
    if _block_axis_kind(center) == "none":
        return None
    if _variant_diff(motif, majority) < MIN_VARIANT_DIFF:
        return None

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
