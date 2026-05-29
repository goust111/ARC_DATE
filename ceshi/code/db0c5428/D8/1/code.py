# Variant of db0c5428
# Dimension: D8 — 块大小与网格尺寸的关系 (meta)
# Change: 每个块固定为3x3，图案为9x9，嵌入在16-19大小的正方形网格中... → 块为非正方形矩形（如3x2）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the D8 variant
#
# variant_spec:
#   block_shape: fixed 3x2 rectangle (height=3, width=2)
#   motif_shape: 9x6 foreground bbox = 3x3 arrangement of 3x2 blocks
#   expanded_shape: 15x10 foreground bbox = 5x5 block lattice with the same
#     occupied slots pattern as the original task
#   outer_grid: still square, side length 16..19
#   preserved_rule:
#     - the visible motif is the middle of a larger 5x5 block arrangement
#     - the 8 non-center blocks are copied to the point-symmetric outer slots
#       without rotation
#   changed_rule:
#     - block height and width must be handled independently
#     - 90-degree rotation / transpose can no longer be used because a 3x2 block
#       would become 2x3 and break the lattice
#     - motif symmetry is therefore reduced from full D4 to the rectangular
#       analogue: left-right mirror, up-down mirror, and 180-degree rotation
#   center_reconstruction:
#     - the missing 3x2 center is reconstructed from six hole-facing samples:
#       rows offsets = [-1, +1, +3], col offsets = [-1, +2]
#     - these coordinates are the six immediate diagonal/lateral samples around
#       the rectangular hole that still form a 3x2 array:
#         top row    <- NW.bottom-right, NE.bottom-left
#         middle row <- W.middle-right,  E.middle-left
#         bottom row <- SW.top-right,   SE.top-left
#   small_trace:
#     - original logic used one scalar `block`, assumed square blocks, and copied
#       with `dst_r = ... * block`, `dst_c = ... * block`; that is harmless only
#       when row and column block sizes are identical
#     - in this variant the correct destinations are
#         `dst_r = expanded_r0 + (4 - 2*br) * block_h`
#         `dst_c = expanded_c0 + (4 - 2*bc) * block_w`
#       so horizontal and vertical spacing remain aligned for 3x2 blocks
#     - original center fill also assumed a 3x3 patch with a unique center cell;
#       that assumption fails for width=2, so the new center patch is entirely
#       sampled from surrounding pixels instead of injecting one majority cell
#   complexity_check:
#     - the rectangular 9x6 motif is visually distinct from the original 9x9
#     - corner blocks are intentionally asymmetric, so left-right / up-down
#       reflections are visible and the task does not collapse into trivial
#       monochrome rectangles
#     - top-middle and middle-left seeds use different internal symmetries
#       (LR vs UD), giving repeated evidence for the axis-separated logic
#   legality_check:
#     - input placement uses r0 in [3, N-12], c0 in [2, N-8], guaranteeing that
#       the 15x10 expanded output fits in an N x N square without overlap issues
#     - there is exactly one foreground motif, so bbox detection is unambiguous
#   generator_guarantees_for_main:
#     - motif bbox is exactly 9x6
#     - all 8 visible blocks are fully foreground-colored; only the center block
#       is background, so every sampled center pixel is defined
#     - the hole-facing cells copied into the center are non-background and
#       usually include non-majority colors, making the rectangular behavior
#       visually observable in a few samples
# ---------------------------------------------------------------------------


BLOCK_H = 3
BLOCK_W = 2
MOTIF_H = BLOCK_H * 3
MOTIF_W = BLOCK_W * 3
EXPANDED_H = BLOCK_H * 5
EXPANDED_W = BLOCK_W * 5


def _center_patch(g: np.ndarray, r0: int, c0: int) -> np.ndarray:
    hole_r = int(r0) + BLOCK_H
    hole_c = int(c0) + BLOCK_W
    row_offsets = (-1, 1, 3)
    col_offsets = (-1, 2)
    rows = [hole_r + dr for dr in row_offsets]
    cols = [hole_c + dc for dc in col_offsets]
    return g[np.ix_(rows, cols)].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a rectangular 3x3 block motif into its larger 5x5 counterpart.

    Input structure:
    - A square teal background contains one 9x6 foreground motif.
    - The motif is a 3x3 arrangement of 3x2 rectangular blocks.
    - The middle 3x2 block is missing, and the pixels around that rectangular
      hole encode the six cells needed to rebuild it.

    Rule:
    - Keep the original 9x6 motif in place.
    - Copy each visible 3x2 block to the point-symmetric outer slot of the
      implied 5x5 block lattice, preserving its orientation.
    - Reconstruct the missing 3x2 center from the six hole-facing samples:
      diagonal samples make the top and bottom rows, side samples make the
      middle row.

    Key insight: Once the motif is recognized as a rectangular block lattice,
    the expansion is still the same outer mirroring pattern as the original
    task, but every row/column offset must use separate block height and width.
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

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = crop_bbox(
                g,
                (
                    int(r0) + br * BLOCK_H,
                    int(c0) + bc * BLOCK_W,
                    int(r0) + (br + 1) * BLOCK_H - 1,
                    int(c0) + (bc + 1) * BLOCK_W - 1,
                ),
            )
            dst_r = expanded_r0 + (4 - 2 * br) * BLOCK_H
            dst_c = expanded_c0 + (4 - 2 * bc) * BLOCK_W
            paste(out, src, dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0)), int(r0) + BLOCK_H, int(c0) + BLOCK_W)
    return out


# --------------- generator ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD_R = BLOCK_H
MOTIF_PAD_C = BLOCK_W

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

# -- CFG: majority counts per seed block --
CORNER_MAJOR_CHOICES = (4, 5)
TOP_MAJOR_CHOICES = (2, 4)
SIDE_MAJOR_CHOICES = (3, 4, 5)

# -- CFG: generation --
MAX_RETRIES = 300

# Corner seed is intentionally asymmetric; each cell is its own orbit.
CORNER_ORBITS = tuple(((r, c),) for r in range(BLOCK_H) for c in range(BLOCK_W))

# Top-middle block must survive left-right mirroring inside a 3x2 rectangle.
LR_ORBITS = (
    ((0, 0), (0, 1)),
    ((1, 0), (1, 1)),
    ((2, 0), (2, 1)),
)

# Middle-left block must survive up-down mirroring inside a 3x2 rectangle.
UD_ORBITS = (
    ((0, 0), (2, 0)),
    ((0, 1), (2, 1)),
    ((1, 0),),
    ((1, 1),),
)


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
    # Forced minor cells are exactly the ones sampled into the center patch.
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


def _expanded_output(motif, center):
    expanded = np.full((EXPANDED_H, EXPANDED_W), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK_H, BLOCK_W)
    paste(expanded, output_motif, BLOCK_H, BLOCK_W)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[
                br * BLOCK_H:(br + 1) * BLOCK_H,
                bc * BLOCK_W:(bc + 1) * BLOCK_W,
            ]
            paste(expanded, src, (4 - 2 * br) * BLOCK_H, (4 - 2 * bc) * BLOCK_W)
    return expanded


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
    expanded = _expanded_output(motif, center)

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
    raise AssertionError("failed to generate a valid db0c5428 D8 input")
