# Variant of db0c5428
# Dimension: D5 — 中心块的重建机制 (transform)
# Change: 中心块的8个非中心像素取自8个相邻块的朝向像素（facing pixel），中心像素取图案的多数前景色
#   → 中心块改为通过围绕空洞的 LR/UD 全局对称恢复：每个非中心像素从跨越空洞后的对侧对称位置取值。

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


# Design analysis for the D5 variant
# a) Concretization + trace:
#    The input still contains one 9x9 motif made of eight visible 3x3 blocks around
#    a missing 3x3 center. The output still expands this motif to a 15x15 5x5 block
#    layout. The only behavioral change is how the missing center is reconstructed.
#    Original: center(0,0) came from the top-left block's facing corner b00[2,2].
#    Variant: center(0,0) comes from the bottom-right block's opposite corner
#    b22[2,2], because the missing block is repaired by reflecting across the empty
#    gap's horizontal and vertical symmetry lines. Likewise center(0,1) comes from
#    the bottom middle block, center(1,0) from the right middle block, etc.
# b) Complexity calibration:
#    The outer 5x5 copy rule is unchanged, so the task remains a single-object
#    geometric reasoning problem. To make the D5 change visible, generation forces
#    the old inward-facing samples and the new opposite-side samples to disagree in
#    both the corner channel and the edge channel; this yields 8 visible center-pixel
#    differences relative to the legacy rule, not just a subtle corner case.
# c) Hidden-assumption audit of main():
#    - bbox/get_bg still assume exactly one isolated non-bg motif on a bg canvas.
#    - The expansion loop assumes the motif is a 3x3 grid of equal-size blocks; that
#      remains valid and unchanged.
#    - The new center repair assumes block==3, because each center row/col maps to
#      one of three known source bands: bottom edge, middle band, or top edge.
#    - The solver no longer assumes inner-boundary pixels carry the answer; it reads
#      from opposite blocks instead, so any generator relying on the old boundary
#      encoding would now be invalid.
# d) Placement legality:
#    The motif is still placed with a 3-cell margin, so the 15x15 expanded output
#    fits entirely inside the 16..19 square canvas without overlap or clipping.
# e) Visual observability:
#    Samples remain easy to parse: a single symmetric ring around a missing center.
#    Because generation enforces b00[0,0] != b00[2,2] and b01[0,1] != b01[2,1],
#    the new center differs from the legacy center in all 8 non-central pixels.
# f) Visual cleanliness:
#    No extra markers are introduced. The grid still contains one motif on a plain
#    teal background, and the output contains the same clean 5x5 block pattern.
# g) Color inference:
#    No hidden color-id ordering is used. Colors are sampled directly, and the only
#    color-frequency rule kept from the original task is the majority fill for the
#    center fixed point, which symmetry alone cannot determine.
# 9) New main() assumptions guaranteed by generate_input():
#    - The visible ring obeys the same D4 block relations as the original task, so
#      the opposite-side symmetry readout is consistent and human-observable.
#    - The center fixed point is set to the majority foreground color.
#    - The generator rejects degenerate motifs where the new center would collapse to
#      the old center, so the D5 variant signal is always active.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this variant uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)

    # Reflect each missing cell across the hole to the opposite visible band.
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
    Expand a 3x3 block motif into a 5x5 block frame and repair its missing center.

    Input structure:
    - One 9x9 non-background motif sits on a teal canvas.
    - The motif is a 3x3 arrangement of 3x3 blocks with the middle block removed.
    - The eight visible blocks form a symmetric ring around the empty hole.

    Rule:
    - Leave the original 9x9 motif in place.
    - Copy each visible block to the point-symmetric outer slot of the implied 5x5
      block layout, without rotating the copied block.
    - Reconstruct the center 3x3 block by reflecting each missing non-center cell
      across the hole to the opposite known side; keep the center cell as the motif's
      majority foreground color.

    Key insight: The hole is not encoded by the pixels facing inward. Instead, the
    surrounding ring continues through the gap by global left-right and up-down
    symmetry, so each missing pixel is read from its opposite-side counterpart.
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


# --------------- generator (o5x-D5 symmetry repair) ---------------

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
MIN_VARIANT_DIFF = 8

# -- Symmetry orbits for 3x3 blocks --
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

    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    motif = _assemble_motif(b00, b01)
    if _variant_diff(motif, majority) < MIN_VARIANT_DIFF:
        return None

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
    raise AssertionError("failed to generate a valid db0c5428 D5-variant input")
