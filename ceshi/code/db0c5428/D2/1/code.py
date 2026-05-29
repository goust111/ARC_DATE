# Variant of db0c5428
# Dimension: D2 — 源图案的块阵列结构 (src)
# Change: 9x9图案由3x3个3x3块组成，中心块缺失（填充背景色），块间无间隔... → 块间带1像素分隔线

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D2 variant
# ----------------------------------
# source_task: none
#
# a) Concrete behavior and trace
#    - Input no longer contains a solid 9x9 foreground square. Instead it contains
#      an 11x11 motif: three 3x3 blocks per axis, with 1-pixel separator rows and
#      columns at indices 3 and 7. The center 3x3 block is still missing and uses
#      the background color. The separator has its own dedicated color, distinct
#      from both background and block-internal foreground colors.
#    - Output keeps the original "copy each non-center block to its point-symmetric
#      outer slot" rule, but the separator lattice must also expand. Therefore the
#      output region is 19x19: five possible block slots per axis with separator
#      rows/cols at indices 3, 7, 11, 15. Empty logical slots stay background.
#    - Small trace at the top-left corner:
#      original task: a source block starting at row 0 occupies rows 0..2, and its
#      copied partner is pasted at rows 12..14 in the 15x15 output.
#      this variant: the same block occupies rows 0..2 in the 11x11 motif, but the
#      next block band starts at row 4 because row 3 is a separator. The copied
#      partner must therefore land at rows 16..18 in the 19x19 output, leaving
#      row 15 as the continued separator line. The rule change is not just "bigger
#      canvas"; it changes the block pitch from 3 to 4.
#
# b) Complexity calibration
#    - The separator color is a single global cue with repeated support: exactly
#      two full rows and two full columns in every input motif, and four rows/cols
#      in the expanded output. This is visually obvious in 2-4 samples.
#    - Block contents keep the original asymmetry constraints, so the copied outer
#      blocks remain distinguishable and the task does not collapse into a trivial
#      uniform-grid fill.
#    - The separator is not allowed to also appear inside blocks, avoiding ambiguity
#      between "line color" and "foreground pattern color".
#
# c) Original main() assumptions that break under D2
#    - `h % 3 == 0` assumed no separators. The new motif bbox is 11x11, so direct
#      division by 3 is invalid.
#    - `block = h // 3` would misread 11 as block size 3 only by accident and would
#      ignore the separator rows/cols entirely.
#    - Cropping source blocks with `br * block` / `bc * block` assumed contiguous
#      blocks. In the variant, every source block start is `br * (block + 1)`.
#    - Center reconstruction offsets `[-1, 1, 3]` assumed the hole touches adjacent
#      blocks directly. With separators, the inward-facing samples move to
#      `[-2, 1, 4]` relative to the center block start.
#    - Majority-color detection on `g != bg` becomes wrong because the separator
#      color would dominate the counts. The solver must exclude the separator.
#
# d) Placement legality
#    - Input motif size is 11x11 and the expanded output region is 19x19, so the
#      generator must place the motif with a margin of one block pitch (4 pixels)
#      on each side. This guarantees the 19x19 output patch fits without clipping.
#    - The generator still uses output-first verification: build a valid motif,
#      build the exact expanded output, place both on a blank canvas, and keep the
#      sample only if `main(input) == output`.
#
# e) Visual observability
#    - Humans can infer the separator rule because the input visibly contains a
#      colored grid skeleton, and the output visibly extends that skeleton outward
#      to the new copied blocks. The change is structural, not hidden in color ids.
#
# f) Visual cleanliness
#    - Separator rows/cols occupy only the 1-pixel gutters between 3x3 blocks, so
#      they never overwrite block interiors. Empty logical slots in the 5x5 layout
#      remain clean 3x3 background holes bounded by the separator lattice.
#
# g) Color inference
#    - No color remapping is introduced. The only new color role is the separator,
#      which is identified by its repeated full-row/full-column geometry rather than
#      by palette ordering.
#
# h) Variant metric to check non-degeneration
#    - A valid generated sample must have motif bbox size 11x11, exactly one
#      separator color forming 2 full rows + 2 full cols in the input motif, and a
#      19x19 expanded region with 4 separator rows + 4 separator cols in the output.
#      These metrics are incompatible with the original no-gap geometry.
#
# i) New solver assumptions and generator guarantees
#    - The solver assumes there is exactly one foreground motif and exactly one
#      non-background color that forms the two full separator rows and columns.
#      The generator enforces both.
#    - The solver assumes separator width is 1 and block size is 3. The generator
#      fixes that geometry explicitly.


BLOCK = 3
SEP = 1
PITCH = BLOCK + SEP
MOTIF_BLOCKS = 3
EXPANDED_BLOCKS = 5


def _motif_size(block_count: int) -> int:
    return block_count * BLOCK + (block_count - 1) * SEP


MOTIF_SIZE = _motif_size(MOTIF_BLOCKS)          # 11
EXPANDED_SIZE = _motif_size(EXPANDED_BLOCKS)    # 19


def _block_starts(block_count: int) -> list[int]:
    return [i * PITCH for i in range(block_count)]


def _separator_positions(block_count: int) -> list[int]:
    return [BLOCK + i * PITCH for i in range(block_count - 1)]


def _separator_grid(block_count: int, bg: int, sep_color: int) -> np.ndarray:
    size = _motif_size(block_count)
    grid = np.full((size, size), bg, dtype=int)
    for pos in _separator_positions(block_count):
        grid[pos, :] = sep_color
        grid[:, pos] = sep_color
    return grid


def _extract_block(motif: np.ndarray, br: int, bc: int, block: int = BLOCK) -> np.ndarray:
    r0 = br * (block + SEP)
    c0 = bc * (block + SEP)
    return motif[r0 : r0 + block, c0 : c0 + block].copy()


def _find_separator_structure(motif: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = motif.shape
    assert h == w, "motif must be square"
    candidates: list[tuple[int, int]] = []
    for raw_color in np.unique(motif):
        color = int(raw_color)
        if color == int(bg):
            continue
        row_hits = [r for r in range(h) if np.all(motif[r, :] == color)]
        col_hits = [c for c in range(w) if np.all(motif[:, c] == color)]
        if len(row_hits) != 2 or len(col_hits) != 2:
            continue
        row_gap = row_hits[1] - row_hits[0]
        col_gap = col_hits[1] - col_hits[0]
        if row_gap != col_gap or row_gap < 2:
            continue
        block = row_gap - SEP
        if h != block * 3 + SEP * 2 or w != block * 3 + SEP * 2:
            continue
        expected = [block, 2 * block + SEP]
        if row_hits == expected and col_hits == expected:
            candidates.append((color, block))
    assert len(candidates) == 1, "expected a unique separator color"
    return candidates[0]


def _major_foreground_color(g: np.ndarray, bg: int, sep_color: int) -> int:
    mask = (g != int(bg)) & (g != int(sep_color))
    vals, counts = np.unique(g[mask], return_counts=True)
    assert vals.size, "input must contain non-separator foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(motif: np.ndarray, block: int, bg: int, sep_color: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    main = _major_foreground_color(motif, bg, sep_color)
    starts = _block_starts(MOTIF_BLOCKS)
    rows = [starts[0] + block - 1, starts[1] + block // 2, starts[2]]
    cols = [starts[0] + block - 1, starts[1] + block // 2, starts[2]]
    patch = motif[np.ix_(rows, cols)].copy()
    patch[block // 2, block // 2] = main
    return patch


def _expanded_output(
    motif: np.ndarray,
    center: np.ndarray,
    block: int,
    bg: int,
    sep_color: int,
) -> np.ndarray:
    expanded = _separator_grid(EXPANDED_BLOCKS, bg, sep_color)
    output_motif = motif.copy()
    paste(output_motif, center, PITCH, PITCH)
    paste(expanded, output_motif, PITCH, PITCH)
    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                continue
            src = _extract_block(motif, br, bc, block)
            paste(expanded, src, (4 - 2 * br) * PITCH, (4 - 2 * bc) * PITCH)
    return expanded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a separated 3x3 block motif into the corresponding separated 5x5 layout.

    Input structure:
    - One foreground motif sits on a uniform background, but the motif is now 11x11.
    - It is a 3x3 arrangement of 3x3 blocks with 1-pixel separator rows and columns.
    - The middle 3x3 block is background, while the separator color forms the full
      gutters that outline the missing center.

    Rule:
    - Detect the separator color and treat the remaining colored 3x3 islands as the
      actual source blocks.
    - Rebuild the missing center block from the eight inward-facing pixels around the
      hole; the center cell of that block is the majority non-separator color.
    - Place the eight visible source blocks into the same point-symmetric outer slots
      as the original task, and extend the separator rows/columns across the whole
      enlarged 5x5 lattice.

    Key insight:
    - The separator lines are part of the geometry, not decoration: they reveal the
      block pitch. Once the 11x11 motif is parsed as 3x3 blocks plus gutters, the
      copying rule is the same block-level pattern as in the original task.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    motif = crop_bbox(g, motif_bbox)
    sep_color, block = _find_separator_structure(motif, bg)
    assert block == BLOCK, "expected 3x3 blocks separated by 1-pixel gutters"

    center = _center_patch(motif, block, bg, sep_color)
    expanded = _expanded_output(motif, center, block, bg, sep_color)

    out = g.copy()
    paste(out, expanded, int(r0) - PITCH, int(c0) - PITCH)
    return out


# --------------- generator (o5x) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 19, 23
MOTIF_PAD = PITCH

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


def _assemble_motif(b00: np.ndarray, b01: np.ndarray, sep_color: int) -> np.ndarray:
    motif = _separator_grid(MOTIF_BLOCKS, BG, sep_color)
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
        paste(motif, blk, br * PITCH, bc * PITCH)
    return motif


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    starts = _block_starts(MOTIF_BLOCKS)
    rows = [starts[0] + BLOCK - 1, starts[1] + BLOCK // 2, starts[2]]
    cols = [starts[0] + BLOCK - 1, starts[1] + BLOCK // 2, starts[2]]
    patch = motif[np.ix_(rows, cols)].copy()
    patch[BLOCK // 2, BLOCK // 2] = majority
    return patch


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]
    separator_choices = [c for c in FOREGROUND_COLORS if c not in colors]
    separator = random.choice(separator_choices)

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
    motif = _assemble_motif(b00, b01, separator)
    center = _gen_center_patch(motif, majority)
    expanded = _expanded_output(motif, center, BLOCK, BG, separator)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - PITCH, c0 - PITCH)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D2 input")
