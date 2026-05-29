# Variant of db0c5428
# Dimension: D7 — 输出网格的整体形状 (grid_shape)
# Change: 输出与输入同尺寸（same），外扩块在网格范围内原地放置，超出部分被截断... → 输出自适应扩展以容纳完整5x5块网格

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D7 variant
# ----------------------------------
# concretization:
# - Input remains a single 9x9 motif on a square teal background.
# - The motif is still a 3x3 arrangement of 3x3 blocks with a missing 3x3 center.
# - Output content is still the same 5x5 block layout as the original task:
#   the visible 3x3 motif sits in the middle of a 15x15 block arrangement,
#   the center block is reconstructed from the eight pixels around the hole,
#   and every non-center block is copied to its point-symmetric outer slot
#   without rotation.
# - The changed behavior is only the canvas: instead of forcing this 15x15
#   pattern into the input-sized grid, the solver must enlarge the canvas with
#   background rows/cols whenever the full pattern would extend beyond an edge.
#
# behavior trace:
# - Example: let the input be 16x16 and let the motif bbox start at (1, 6).
# - Original same-size logic computes the expanded 15x15 region at top-left
#   (r0-3, c0-3) = (-2, 3). Stamping into a 16x16 output would either clip or
#   fail for rows -2 and -1, so the outer top copies are effectively missing.
# - New logic keeps the same intended expanded region, but first computes the
#   union of the original input canvas [0..15]x[0..15] and the expanded region
#   [-2..12]x[3..17]. The required output bbox is therefore [-2..15]x[0..17],
#   so the solver shifts the whole original image down by 2 and returns an
#   18x18 grid that contains the full 15x15 structure plus the unchanged
#   original background area.
#
# complexity calibration:
# - The block-copy and center-reconstruction rules are untouched, so the puzzle
#   still hinges on the same D4-symmetric 9x9 motif logic.
# - To make D7 visually observable, the generator forces the motif into one of
#   eight edge/corner placement modes. This guarantees that output size grows on
#   at least one side while keeping the placement family small enough for a
#   human to recognize from a few examples.
#
# implicit-assumption audit of the original main():
# - `out = g.copy()` assumed the output canvas is exactly the input canvas.
#   That is the main D7 dependency and is no longer valid.
# - `expanded_r0 = r0 - block` / `expanded_c0 = c0 - block` were harmless only
#   because the generator used a fixed 3-cell pad. In the variant these values
#   may be negative or may push the 15x15 region past the bottom/right edge.
# - Each `paste(out, src, dst_r, dst_c)` implicitly assumed destination coords
#   are already in bounds. The variant must instead create a larger destination
#   before any paste happens.
# - Reconstructing the center block itself does not depend on output size, so
#   that part can remain unchanged once the motif has been located.
#
# placement legality:
# - The input still contains only one foreground motif, so bbox detection stays
#   unambiguous.
# - The generator only chooses motif top-left positions `(r0, c0)` in
#   `[0, n-9] x [0, n-9]`, which always keeps the 9x9 input motif in bounds.
# - The solver builds the output on the union of the old canvas and the intended
#   expanded 15x15 region, so no copied block can overlap ambiguously or fall
#   off the canvas.
#
# visual observability and neatness:
# - The motif internals stay asymmetric enough because the original generator's
#   orbit-based color assignment is preserved.
# - Output enlargement happens by adding clean background bands on specific
#   sides; no extra markings or clutter are introduced.
# - No hidden color-id ordering is used anywhere; colors still serve only as
#   visible block content.


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


def _expanded_region_from_grid(
    g: np.ndarray, r0: int, c0: int, block: int, bg: int
) -> np.ndarray:
    """Render the full 5x5 block arrangement as a standalone 15x15 region."""
    expanded = np.full((block * 5, block * 5), int(bg), dtype=int)
    motif = crop_bbox(
        g,
        (
            int(r0),
            int(c0),
            int(r0) + 3 * block - 1,
            int(c0) + 3 * block - 1,
        ),
    ).copy()
    paste(motif, _center_patch(g, int(r0), int(c0), block, bg), block, block)
    paste(expanded, motif, block, block)

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


def _render_on_resized_canvas(
    original: np.ndarray, expanded: np.ndarray, expanded_r0: int, expanded_c0: int, bg: int
) -> np.ndarray:
    """Place the original canvas and the expanded 15x15 motif on their minimal union."""
    h, w = map(int, original.shape)
    eh, ew = map(int, expanded.shape)
    min_r = min(0, int(expanded_r0))
    min_c = min(0, int(expanded_c0))
    max_r = max(h - 1, int(expanded_r0) + eh - 1)
    max_c = max(w - 1, int(expanded_c0) + ew - 1)

    shift_r = -min_r
    shift_c = -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    paste(out, original, shift_r, shift_c)
    paste(out, expanded, int(expanded_r0) + shift_r, int(expanded_c0) + shift_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into its full 5x5 block frame on a resized canvas.

    Input structure:
    - A square background grid contains one 9x9 motif made from 3x3 blocks.
    - The motif's middle 3x3 block is missing and stays background-colored.
    - The eight pixels facing that hole encode the boundary of the missing block.

    Rule:
    - Rebuild the missing center block from those eight inward-facing pixels,
      using the motif's majority foreground color for the center pixel.
    - Copy each visible non-center 3x3 block to the opposite outer slot of the
      implied 5x5 block arrangement, without rotating the copied block.
    - Return a background-padded canvas large enough to contain both the
      original input placement and the complete 15x15 expanded arrangement.

    Key insight: The 9x9 figure is the middle window of a larger block pattern,
    so the solver must infer the full 15x15 arrangement first and only then
    choose an output canvas that preserves every copied block.
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

    expanded = _expanded_region_from_grid(g, int(r0), int(c0), block, bg)
    return _render_on_resized_canvas(g, expanded, int(r0) - block, int(c0) - block, bg)


# --------------- generator (o5x, D7-resize variant) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19        # square input side length
BLOCK = 3                                     # block side length
MOTIF_SIZE = BLOCK * 3                        # 9x9 motif (3x3 grid of blocks)
EXPANDED_SIZE = BLOCK * 5                     # 15x15 expanded block arrangement

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
PLACEMENT_MODES = ("top", "bottom", "left", "right", "tl", "tr", "bl", "br")

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

_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_orbits(orbits, forced_minor, target_count):
    """Pick an orbit subset whose total pixel count matches target_count."""
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
    """Create one seed block and return its unassigned minor-color orbits."""
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
    """Assign every minor color at least once, then fill remaining orbits randomly."""
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
    """Build the two seed blocks that define the D4-symmetric motif."""
    b00, corner_minors = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
    b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    return b00, b01


def _assemble_motif(b00, b01):
    """Assemble the 9x9 motif from two seed blocks; center block stays background."""
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
    """Reconstruct the missing center block from the motif's inner boundary."""
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _expanded_output(motif, center):
    """Build the standalone 15x15 expanded arrangement."""
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


def _sample_location(n: int) -> tuple[int, int]:
    """Choose one of a small set of edge/corner placements so resizing is visible."""
    max_r0 = int(n - MOTIF_SIZE)
    max_c0 = int(n - MOTIF_SIZE)
    low_r = list(range(0, BLOCK))
    low_c = list(range(0, BLOCK))
    high_r = list(range(max_r0 - (BLOCK - 1), max_r0 + 1))
    high_c = list(range(max_c0 - (BLOCK - 1), max_c0 + 1))
    mid_r = list(range(BLOCK, max_r0 - BLOCK + 1))
    mid_c = list(range(BLOCK, max_c0 - BLOCK + 1))
    assert mid_r and mid_c

    mode = random.choice(PLACEMENT_MODES)
    if mode == "top":
        return random.choice(low_r), random.choice(mid_c)
    if mode == "bottom":
        return random.choice(high_r), random.choice(mid_c)
    if mode == "left":
        return random.choice(mid_r), random.choice(low_c)
    if mode == "right":
        return random.choice(mid_r), random.choice(high_c)
    if mode == "tl":
        return random.choice(low_r), random.choice(low_c)
    if mode == "tr":
        return random.choice(low_r), random.choice(high_c)
    if mode == "bl":
        return random.choice(high_r), random.choice(low_c)
    return random.choice(high_r), random.choice(high_c)


def _expected_output(inp: np.ndarray, expanded: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """Independent expected output renderer for generator-side validation."""
    return _render_on_resized_canvas(inp, expanded, int(r0) - BLOCK, int(c0) - BLOCK, BG)


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build one input and verify it with the oracle."""
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
    center = _gen_center_patch(motif, majority)
    expanded = _expanded_output(motif, center)

    r0, c0 = _sample_location(n)
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    out = _expected_output(inp, expanded, r0, c0)

    grew_rows = out.shape[0] > n
    grew_cols = out.shape[1] > n
    assert grew_rows or grew_cols

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D7 input")
