# Combo variant of db0c5428 (index 39)
# combo_spec: D7/1+D9/0
# Name: 图案居中+紧凑裁剪

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the fused variant:
# a) Concrete behavior:
#    - Original task: a 9x9 motif can sit at many offsets inside a 16-19 square;
#      the solver writes the reconstructed 15x15 expansion back onto that same canvas.
#    - Fused task: the input motif is always centered in the canvas, so the empty
#      margin is symmetric on all four sides. The solver still reconstructs the
#      15x15 expansion in block space, but then tight-crops the result to the
#      non-background bbox instead of preserving the outer input canvas.
#    - Example: on a 19x19 input the motif top-left is fixed at (5,5). The full
#      expansion occupies rows/cols 2..16 in the original canvas, but the output
#      discards that border and returns only the 15x15 expanded region.
# b) Complexity calibration:
#    - Centering removes offset search, but the crop adds an observable output-size
#      change. Using only 17x17 and 19x19 canvases preserves two distinct centered
#      margin sizes (4 or 5) so examples still show repeated evidence.
# c) Implicit assumptions audit:
#    - `bbox(g != bg)` must still isolate exactly one 9x9 motif. Generator keeps
#      a single centered motif on a uniform bg canvas.
#    - `h == w` and `h % 3 == 0` still hold. Generator still uses a 3x3 array of
#      3x3 blocks, so bbox size is exactly 9x9.
#    - Center reconstruction reads the 8 pixels facing the hole. Generator keeps
#      the same D4 motif structure, so those boundary pixels remain valid.
#    - Returning `g.copy()` no longer makes sense because D7/1 deletes the input
#      border. The solver therefore renders into a fresh 15x15 expansion and then
#      crops to the minimal non-bg bbox.
# d) Placement legality:
#    - Centered placement uses `r0 = c0 = (n - 9) // 2` with `n in {17, 19}`.
#      This guarantees equal margins, no overlap, and the 9x9 motif remains fully
#      inside the input grid.
# e/f/g) Visual observability and neatness:
#    - Humans can now see two coupled changes directly: the input motif is centered,
#      and the output is a borderless 15x15 expansion. No hidden color ordering or
#      numeric-id tricks are used; all variation remains positional and symmetric.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    majority = _major_foreground_color(g, bg)
    patch = np.full((block, block), majority, dtype=int)

    hole_r = int(r0) + block
    hole_c = int(c0) + block
    offsets = (-1, 1, 3)
    for r in range(block):
        for c in range(block):
            if (r, c) == (block // 2, block // 2):
                continue
            patch[r, c] = int(g[hole_r + offsets[r], hole_c + offsets[c]])
    return patch


def _expanded_region(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    expanded = np.full((block * 5, block * 5), int(bg), dtype=int)
    filled_motif = crop_bbox(
        g,
        (int(r0), int(c0), int(r0) + block * 3 - 1, int(c0) + block * 3 - 1),
    )
    paste(filled_motif, _center_patch(g, int(r0), int(c0), block, bg), block, block)
    paste(expanded, filled_motif, block, block)

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
            dst_r = (4 - 2 * br) * block
            dst_c = (4 - 2 * bc) * block
            paste(expanded, src, dst_r, dst_c)
    return expanded


def _tight_crop_non_bg(g: np.ndarray, bg: int) -> np.ndarray:
    box = bbox(g != int(bg))
    assert box is not None, "expanded output should not be empty"
    return crop_bbox(g, box)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center block, expand the motif outward, then return only
    the occupied part of the expansion.

    Input structure:
    - A square bg canvas contains exactly one 9x9 motif made from a 3x3 array of
      3x3 blocks, with the middle block left as bg.
    - The motif is centered in the input grid, so the empty border is balanced on
      all four sides.
    - The eight pixels facing the hole still encode the missing center block.

    Rule:
    - Recover the center 3x3 block from those inner-boundary pixels, using the
      majority foreground color for the center cell.
    - Keep the original surrounding eight blocks in the middle of a 5x5 block
      arrangement and copy each one to its point-symmetric outer slot.
    - After the full expansion is built, crop away every bg-only border and
      return the minimal rectangle containing the result.

    Key insight:
    The centered placement is only a presentation constraint; the real structure
    is still a 3x3 block motif that extends to a 5x5 block pattern. Once that
    larger pattern is rendered, tight cropping reveals the complete expansion.
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

    expanded = _expanded_region(g, int(r0), int(c0), block, bg)
    return _tight_crop_non_bg(expanded, bg)


# --------------- generator (output-first, centered motif) ---------------

# -- CFG: canvas & structure --
GRID_SIZES = (17, 19)                        # centered 9x9 motif requires odd side
BLOCK = 3                                    # block side length
MOTIF_SIZE = BLOCK * 3                       # 9x9 motif (3x3 block grid)
EXPANDED_SIZE = BLOCK * 5                    # 15x15 full expansion

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
_CENTER_OFFSETS = np.array([-1, 1, 3], dtype=int)


def _major_orbits(orbits, forced_minor, target_count):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits) if all(coord not in forced_minor for coord in orbit)
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
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _expanded_output(motif, center):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    filled_motif = motif.copy()
    paste(filled_motif, center, BLOCK, BLOCK)
    paste(expanded, filled_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
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
    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    expanded = _expanded_output(motif, center)

    r0 = (n - MOTIF_SIZE) // 2
    c0 = (n - MOTIF_SIZE) // 2
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)

    # Generator guarantees the new solver assumptions:
    # - exactly one foreground bbox of size 9x9
    # - centered placement with equal margins
    # - cropped output differs from the original same-size behavior
    if np.array_equal(main(inp), expanded):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
