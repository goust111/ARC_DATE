# Combo variant of db0c5428 (index 23)
# combo_spec: D2/1+D7/0+D13/0
# Name: 分隔线阵列+自适应扩展+分隔线填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Input uses the D2 geometry: an 11x11 motif formed by a 3x3 array of 3x3
#      blocks with 1-pixel separator rows/cols at indices 3 and 7. The center
#      3x3 block is missing and stays background-colored in the input.
#    - Output uses the same point-symmetric block copying rule as the base task,
#      but on the separated lattice. The full logical 5x5 block layout therefore
#      occupies 19x19 pixels because the pitch is now 4 instead of 3.
#    - D7 changes the canvas policy: if the 19x19 separated expansion would stick
#      out of the input canvas, the output grows to the minimal union of the old
#      canvas and the intended 19x19 region instead of clipping anything away.
#    - D13 changes how empty logical slots are rendered inside that 19x19 region:
#      the unoccupied 3x3 slots are not background. They are filled with the same
#      separator color, so the separator lattice visually expands into a complete
#      grid framework.
#    - Example trace: let the 11x11 motif be placed at input top-left (1, 6) in
#      a 20x20 grid. The full separated expansion must start at (1-4, 6-4)=(-3,2)
#      and end at (15,20). Under the original same-size assumption the top copies
#      would be lost. In this combo variant the whole input is shifted down by 3
#      onto a 23x20 output, and the empty slot at expanded block position (0, 1)
#      becomes a 3x3 separator-colored patch instead of background.
#
# b) Complexity calibration
#    - The separator role is repeated many times: every input has exactly two full
#      separator rows and two full separator columns, and every output has the same
#      color on the continued lattice plus on the empty 3x3 slots.
#    - D7 is forced to be visible by placing motifs near edges/corners only. This
#      prevents the resized-canvas behavior from degenerating into the D2-only case.
#    - D13 is forced to be visible because the separator color is chosen outside
#      the block palette, so solid separator-colored empty slots cannot be mistaken
#      for copied blocks or for plain background.
#
# c) Implicit-assumption audit of the original main()
#    - `bbox(g != bg)` still works, but the detected bbox is now 11x11 because the
#      separator color is part of the motif. The old `h % 3 == 0` assumption fails.
#    - Cropping with contiguous `br * block` starts is invalid once 1-pixel gutters
#      exist; every logical block start is `br * PITCH`.
#    - Majority-color detection must exclude the separator color or it may dominate.
#    - The old center-pixel sampling offsets `[-1, 1, 3]` assumed touching blocks.
#      With separators, the inward-facing samples are at global rows/cols [2, 5, 8]
#      relative to the 11x11 motif, with the middle sample overwritten afterward.
#    - `out = g.copy()` and direct pasting are invalid under D7 because the 19x19
#      expansion may start at negative coordinates or extend past the input bounds.
#    - D13 means the expanded renderer can no longer initialize empty slots with
#      background. The 19x19 region must begin as separator color and then have the
#      actual block content pasted into the occupied slots.
#
# d) Placement legality
#    - Input motif size is fixed at 11x11, so any sampled `(r0, c0)` in
#      `[0, n-11] x [0, n-11]` is in-bounds.
#    - The generator restricts placements to edge/corner bands of width 4 so the
#      output must grow on at least one side, but the input motif itself never clips.
#    - Output is rendered on the exact union of the original canvas and the intended
#      19x19 expansion, so no copied block or separator-filled slot is lost.
#
# e) Visual observability
#    - A human can see three coupled changes directly from a few examples:
#      separator rows/cols exist already in the input; those same separators keep
#      extending in the output; and the empty block slots become separator-colored
#      squares rather than background holes.
#    - The rule does not depend on color-id ordering. The separator is inferred from
#      its geometry: full repeated rows/columns between 3x3 blocks.
#
# f) Visual cleanliness
#    - Block interiors, separator lines, separator-filled empty slots, and outer
#      background bands occupy disjoint spatial roles. No cell has to encode two
#      meanings at once.
#    - Using one dedicated separator color keeps the enlarged framework legible even
#      after D13 turns the empty logical slots into solid colored patches.
#
# g) Change inferability
#    - No hidden color remapping is introduced. Only one extra semantic color role
#      exists: separator/filler. Because it appears as full gutters in the input and
#      as both gutters and empty slots in the output, the mapping remains learnable.
#
# h) Non-degeneration metric
#    - Useful checks are: input motif bbox must be 11x11; output must grow on at
#      least one side beyond the input; and at least one empty 3x3 slot in the 5x5
#      layout must be entirely separator-colored. If any of these fail systematically,
#      the combo has collapsed toward a simpler variant.
#
# i) New solver assumptions and generator guarantees
#    - The solver assumes exactly one separator color forms the two full motif rows
#      and two full motif columns at the expected positions; the generator enforces
#      this by keeping the separator out of block interiors.
#    - The solver assumes block size 3 and separator width 1. The generator fixes
#      both geometrically.
#    - The solver assumes there is exactly one motif on a uniform background and that
#      the separator-filled 19x19 expansion is the intended standalone pattern before
#      canvas resizing; generator-side construction follows that same geometry.


BLOCK = 3
SEP = 1
PITCH = BLOCK + SEP
MOTIF_BLOCKS = 3
EXPANDED_BLOCKS = 5


def _motif_size(block_count: int) -> int:
    return block_count * BLOCK + (block_count - 1) * SEP


MOTIF_SIZE = _motif_size(MOTIF_BLOCKS)
EXPANDED_SIZE = _motif_size(EXPANDED_BLOCKS)


def _block_starts(block_count: int) -> list[int]:
    return [i * PITCH for i in range(block_count)]


def _separator_positions(block_count: int) -> list[int]:
    return [BLOCK + i * PITCH for i in range(block_count - 1)]


def _extract_block(motif: np.ndarray, br: int, bc: int, block: int = BLOCK) -> np.ndarray:
    r0 = br * PITCH
    c0 = bc * PITCH
    return motif[r0 : r0 + block, c0 : c0 + block].copy()


def _separator_frame(block_count: int, bg: int, sep_color: int) -> np.ndarray:
    size = _motif_size(block_count)
    grid = np.full((size, size), int(bg), dtype=int)
    for pos in _separator_positions(block_count):
        grid[pos, :] = int(sep_color)
        grid[:, pos] = int(sep_color)
    return grid


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


def _major_foreground_color(motif: np.ndarray, bg: int, sep_color: int) -> int:
    mask = (motif != int(bg)) & (motif != int(sep_color))
    vals, counts = np.unique(motif[mask], return_counts=True)
    assert vals.size, "input must contain non-separator foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(motif: np.ndarray, block: int, bg: int, sep_color: int) -> np.ndarray:
    assert block == BLOCK, "this task uses 3x3 blocks"
    starts = _block_starts(MOTIF_BLOCKS)
    rows = [starts[0] + block - 1, starts[1] + block // 2, starts[2]]
    cols = [starts[0] + block - 1, starts[1] + block // 2, starts[2]]
    patch = motif[np.ix_(rows, cols)].copy()
    patch[block // 2, block // 2] = _major_foreground_color(motif, bg, sep_color)
    return patch


def _expanded_output(
    motif: np.ndarray,
    center: np.ndarray,
    block: int,
    sep_color: int,
) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), int(sep_color), dtype=int)
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


def _render_on_resized_canvas(
    original: np.ndarray, expanded: np.ndarray, expanded_r0: int, expanded_c0: int, bg: int
) -> np.ndarray:
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
    Expand a separated 3x3 block motif into a full separator-filled 5x5 lattice.

    Input structure:
    - One motif sits on a uniform background, and its bbox is 11x11 rather than 9x9.
    - The motif is a 3x3 array of 3x3 blocks with 1-pixel separator rows and columns.
    - The middle 3x3 block is missing and remains background in the input, while the
      separator color outlines that hole and reveals the block pitch.

    Rule:
    - Detect the separator color and parse the motif as 3x3 blocks spaced by 1-pixel
      gutters instead of as one contiguous foreground square.
    - Reconstruct the missing center block from the eight inward-facing pixels around
      the hole, using the majority non-separator color for its center pixel.
    - Copy each visible non-center block to the opposite outer slot of the implied 5x5
      lattice without rotation, fill the remaining empty lattice slots with separator
      color, and return the minimal background-padded canvas that contains the entire
      19x19 separated expansion.

    Key insight:
    - The separator lines are structural coordinates, not decoration. Once the input
      is read as an 11x11 window into a separated 5x5 lattice, the solver must extend
      that same lattice outward and use the separator color to occupy every slot that
      stays logically empty.
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
    expanded = _expanded_output(motif, center, block, sep_color)
    return _render_on_resized_canvas(g, expanded, int(r0) - PITCH, int(c0) - PITCH, bg)


# --------------- generator (o5x, combo variant) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 19, 23
MOTIF_PAD = PITCH
PLACEMENT_MODES = ("top", "bottom", "left", "right", "tl", "tr", "bl", "br")

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

EMPTY_SLOTS = (
    (0, 1), (0, 3),
    (1, 0), (1, 4),
    (3, 0), (3, 4),
    (4, 1), (4, 3),
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
    motif = _separator_frame(MOTIF_BLOCKS, BG, sep_color)
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


def _sample_location(n: int) -> tuple[int, int]:
    max_r0 = int(n - MOTIF_SIZE)
    max_c0 = int(n - MOTIF_SIZE)
    low_r = list(range(0, PITCH))
    low_c = list(range(0, PITCH))
    high_r = list(range(max_r0 - (PITCH - 1), max_r0 + 1))
    high_c = list(range(max_c0 - (PITCH - 1), max_c0 + 1))
    mid_r = list(range(PITCH, max_r0 - PITCH + 1))
    mid_c = list(range(PITCH, max_c0 - PITCH + 1))
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


def _empty_slot_is_separator(expanded: np.ndarray, slot: tuple[int, int], sep_color: int) -> bool:
    br, bc = slot
    r0 = br * PITCH
    c0 = bc * PITCH
    return bool(np.all(expanded[r0 : r0 + BLOCK, c0 : c0 + BLOCK] == int(sep_color)))


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
    expanded = _expanded_output(motif, center, BLOCK, separator)

    r0, c0 = _sample_location(n)
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    out = _render_on_resized_canvas(inp, expanded, int(r0) - PITCH, int(c0) - PITCH, BG)

    assert out.shape[0] > n or out.shape[1] > n
    assert all(_empty_slot_is_separator(expanded, slot, separator) for slot in EMPTY_SLOTS)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
