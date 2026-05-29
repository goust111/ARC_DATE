# Combo variant of db0c5428 (index 5)
# combo_spec: D2/1+D13/0
# Name: 带分隔线的块阵列+分隔线填充空白槽位

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for combo D2/1 + D13/0
# ---------------------------------------
# source_task: db0c5428
#
# a) Concrete behavior and trace
#    - The input adopts D2/1 geometry: the original 9x9 motif becomes an 11x11
#      motif made from 3x3 blocks separated by 1-pixel gutters. Separator rows
#      and columns are at local indices 3 and 7, and the center 3x3 slot remains
#      missing/background.
#    - The original output would expand this to a 5x5 logical block layout. Under
#      D2/1 alone, that output would be 19x19 with separator rows/cols continued,
#      but the 8 unused 3x3 logical slots would still stay background.
#    - D13/0 changes exactly those 8 unused 3x3 slots: instead of background, they
#      are filled with the same separator color. Example trace:
#        input top row of logical slots: [block][sep][block][sep][block]
#        D2/1-only output top row:      [copy ][sep][empty][sep][copy ]
#        combo output top row:          [copy ][sep][ sep ][sep][copy ]
#      The solver must therefore infer both the block pitch and that the separator
#      color extends through the whole expanded grid frame, not just the 1-pixel
#      gutter lines.
#
# b) Complexity calibration
#    - The separator color has repeated geometric support in every sample: two full
#      rows and two full columns in the input motif, then four separator rows and
#      four separator columns in the output, plus eight entire 3x3 empty slots.
#    - This is visually stronger than plain D2/1, because the output no longer has
#      isolated background holes inside the 19x19 expansion. The grid frame becomes
#      an obvious object in its own right.
#    - The eight real source blocks still carry asymmetric content constraints, so
#      the task does not collapse into "fill everything with the separator color".
#
# c) Implicit assumptions from the original solver that break
#    - `h % 3 == 0` is invalid because the input motif bbox is now 11x11, not 9x9.
#    - `block = h // 3` ignores gutter pixels; block starts must use pitch 4.
#    - Source cropping via `br * block` / `bc * block` is wrong; starts are
#      `br * (BLOCK + SEP)` and `bc * (BLOCK + SEP)`.
#    - The original center sampling offsets `[-1, 1, 3]` assumed the hole touched
#      neighboring blocks directly. With 1-pixel gutters, the inward-facing samples
#      shift to the three block-aligned rows/cols `[2, 5, 8]`.
#    - Majority-color detection may be dominated by the separator color unless the
#      separator is explicitly excluded.
#    - After D13/0, the output can no longer be initialized as background with only
#      separator lines drawn. The 8 blank logical slots must also be prefilled with
#      separator color before the copied blocks are pasted.
#
# d) Placement legality
#    - Input motif size is 11x11 and the expanded output region is 19x19, so the
#      generator places the input motif with a 4-pixel margin on each side. Then
#      pasting the 19x19 output at `(r0 - 4, c0 - 4)` is always in-bounds.
#    - Only one motif is placed on the canvas, so there is no overlap ambiguity.
#    - The separator color is chosen distinct from background and from all block
#      colors, so the solver's separator parsing remains unambiguous.
#
# e) Visual observability
#    - In 2-4 examples a human can see three stable cues:
#      1. input has a colored lattice between the 3x3 blocks;
#      2. output extends that lattice outward to a 5x5 logical array;
#      3. the places where no copied block lands are not background holes anymore,
#         but solid separator-colored 3x3 pads.
#    - No step depends on raw color ids; the cue is geometric repetition.
#
# f) Visual cleanliness
#    - Block interiors never mix with separator pixels because the separator stays
#      confined to full 1-pixel lines and to entire empty 3x3 slots.
#    - The separator color is forbidden inside blocks, so the frame and the copied
#      motifs remain easy to distinguish.
#
# g) Non-degeneration metric
#    - Relative to D2/1-only output, this combo must change exactly 8 logical empty
#      slots, i.e. `8 * 3 * 3 = 72` pixels per sample, from background to separator
#      color. This fixed delta is strong evidence that D13/0 is active rather than
#      latent.
#
# h) New solver assumptions and generator guarantees
#    - The solver assumes exactly one non-background color forms the two full
#      separator rows and two full separator columns in the input motif. The
#      generator enforces that and keeps separator color out of block interiors.
#    - The solver assumes block size 3 and separator width 1. The generator fixes
#      those values explicitly.
#    - The solver assumes the only blank logical position inside the 3x3 input motif
#      is the center slot. The generator always constructs the motif that way.


BLOCK = 3
SEP = 1
PITCH = BLOCK + SEP
MOTIF_BLOCKS = 3
EXPANDED_BLOCKS = 5


def _motif_size(block_count: int) -> int:
    return block_count * BLOCK + (block_count - 1) * SEP


MOTIF_SIZE = _motif_size(MOTIF_BLOCKS)
EXPANDED_SIZE = _motif_size(EXPANDED_BLOCKS)

GRID_SIZE_MIN, GRID_SIZE_MAX = 19, 23
MOTIF_PAD = PITCH

BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

MAX_RETRIES = 300
EMPTY_SLOT_DELTA_PIXELS = 8 * BLOCK * BLOCK

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

BLANK_SLOTS = (
    (0, 1),
    (0, 3),
    (1, 0),
    (1, 4),
    (3, 0),
    (3, 4),
    (4, 1),
    (4, 3),
)


def _block_start(index: int) -> int:
    return index * PITCH


def _block_starts(block_count: int) -> list[int]:
    return [_block_start(i) for i in range(block_count)]


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
    r0 = _block_start(br)
    c0 = _block_start(bc)
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
    assert block == BLOCK, "this task uses 3x3 blocks"
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
    *,
    fill_blank_slots: bool,
) -> np.ndarray:
    expanded = _separator_grid(EXPANDED_BLOCKS, bg, sep_color)
    if fill_blank_slots:
        for br, bc in BLANK_SLOTS:
            r0 = _block_start(br)
            c0 = _block_start(bc)
            expanded[r0 : r0 + block, c0 : c0 + block] = sep_color

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
        paste(motif, blk, _block_start(br), _block_start(bc))
    return motif


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    starts = _block_starts(MOTIF_BLOCKS)
    rows = [starts[0] + BLOCK - 1, starts[1] + BLOCK // 2, starts[2]]
    cols = [starts[0] + BLOCK - 1, starts[1] + BLOCK // 2, starts[2]]
    patch = motif[np.ix_(rows, cols)].copy()
    patch[BLOCK // 2, BLOCK // 2] = majority
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a separated 3x3 block motif into a separator-filled 5x5 lattice.

    Input structure:
    - One motif sits on a uniform background, and the motif is 11x11 rather than 9x9.
    - It consists of nine logical 3x3 slots arranged in a 3x3 array, with 1-pixel
      separator rows and columns between neighboring slots.
    - Eight slots contain colored 3x3 blocks, the middle slot is background, and
      the separator color outlines the missing center.

    Rule:
    - Detect the separator color and parse the visible pattern as 3x3 blocks on a
      pitch of four pixels.
    - Reconstruct the missing center block from the eight inward-facing pixels
      around the hole, using the majority non-separator color for the center cell.
    - Copy each visible block to its point-symmetric outer slot in the implied 5x5
      layout, keep the separator lattice continuous, and fill the otherwise empty
      3x3 logical slots with the separator color instead of background.

    Key insight:
    - The separator is structural twice over: first it reveals how to parse the
      input into blocks plus gutters, and then it acts as the material that fills
      the unused expanded slots. Solving the task requires extending the whole
      separator framework, not just copying blocks.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, _, _ = motif_bbox
    motif = crop_bbox(g, motif_bbox)
    sep_color, block = _find_separator_structure(motif, bg)
    assert block == BLOCK, "expected 3x3 blocks separated by 1-pixel gutters"

    center = _center_patch(motif, block, bg, sep_color)
    expanded = _expanded_output(
        motif,
        center,
        block,
        bg,
        sep_color,
        fill_blank_slots=True,
    )

    out = g.copy()
    paste(out, expanded, int(r0) - PITCH, int(c0) - PITCH)
    return out


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
    expanded = _expanded_output(
        motif,
        center,
        BLOCK,
        BG,
        separator,
        fill_blank_slots=True,
    )
    baseline = _expanded_output(
        motif,
        center,
        BLOCK,
        BG,
        separator,
        fill_blank_slots=False,
    )
    assert np.count_nonzero(expanded != baseline) == EMPTY_SLOT_DELTA_PIXELS

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
    raise AssertionError("failed to generate a valid db0c5428 combo input")
