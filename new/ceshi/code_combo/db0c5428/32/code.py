# Combo variant of db0c5428 (index 32)
# combo_spec: D7/0+D14/1
# Name: 迭代外扩+自适应输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for combo D7/0 + D14/1
# ---------------------------------------
# a) concretization and behavior trace:
#    The input remains one 9x9 motif made from a 3x3 array of 3x3 blocks, with the
#    middle block missing. As in the base task, the 8 pixels facing the hole encode
#    the missing center block, and the 8 visible blocks are directional templates.
#    The fused rule has two linked changes:
#    - D14/1: expand twice, not once. The original motif is the radius-1 ring around
#      the hole, the first copy round creates the radius-2 ring (5x5 blocks), and a
#      second copy round creates a radius-3 ring (7x7 blocks, 21x21 pixels).
#    - D7/0: the output canvas is no longer tied to the input canvas. Instead, return
#      the minimal background-padded union of the original input grid and the full
#      21x21 iterative expansion.
#    Example: let the input be 17x17 and place the motif at (r0, c0) = (1, 7). The
#    original one-step task wants a 15x15 region at (-2, 4). The fused task instead
#    wants a 21x21 region at (-5, 1). The output therefore cannot stay 17x17: the
#    minimal union spans rows [-5..16] and cols [0..21], so the solver must shift the
#    old canvas down by 5 and return a 22x22 grid containing the complete 7x7 pattern.
#
# b) complexity calibration:
#    The variant stays human-solvable because there is only one fixed iterative depth:
#    exactly two outward rounds. The generator keeps the original asymmetric block
#    internals, so humans can visually track that the same directional blocks appear in
#    both the first outer ring and the second outer ring. To make D7 visible as well,
#    motif placement is biased to edge and corner regimes, so output sizes vary across
#    samples instead of collapsing to a constant 21x21 canvas.
#
# c) implicit-assumption audit of the original main():
#    - `out = g.copy()` is invalid after D7 because the destination may need extra rows
#      and columns on any side.
#    - A single expansion offset `(r0 - block, c0 - block)` is invalid after D14/1;
#      the true top-left of the full result is `(r0 - 2*block, c0 - 2*block)`.
#    - The old loop assumes one copy radius only. The fused rule needs two radii while
#      keeping the original motif centered inside the larger 7x7 layout.
#    - Center reconstruction still assumes a 9x9 motif with 3x3 blocks and fixed inner
#      boundary offsets. The generator preserves those assumptions exactly.
#    - `bbox(g != bg)` still assumes a single foreground object. The generator ensures
#      that the only non-background structure is the motif.
#
# d) placement legality:
#    The input motif is always fully inside the input grid. Because the solver renders
#    the 21x21 expansion on a separate canvas first and only then unions it with the
#    original input canvas, no pasted block can fall out of bounds. Within the 7x7
#    arrangement, radius-2 targets and radius-3 targets occupy disjoint block slots, so
#    there is no ambiguous overlap between iteration rounds.
#
# e) visual observability:
#    Across a few examples, a human can infer both changes: the output grows larger than
#    the input, and the copied directional blocks do not stop at the 5x5 shell. The
#    second shell repeats the same corner/edge identities one more step outward, which
#    is visible directly from block geometry rather than from hidden color conventions.
#
# f) visual cleanliness:
#    The output is a sparse concentric 7x7 block layout with a filled 3x3 center patch.
#    Each occupied block slot has a single source and there is no stacking of multiple
#    patterns in the same place. The resized canvas adds only clean background margins.
#
# g) inferability of changes:
#    No new color remapping is introduced. Colors only define block content and the
#    majority center pixel, exactly as in the base task. The fused change is therefore
#    purely spatial: recognize the second outward propagation and compute the minimal
#    output canvas that preserves the whole propagated pattern.


BLOCK = 3
MOTIF_BLOCKS = 3
OUTPUT_BLOCKS = 7
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
EXPANDED_SIZE = BLOCK * OUTPUT_BLOCKS
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MAX_RETRIES = 300
PLACEMENT_MODES = ("top", "bottom", "left", "right", "tl", "tr", "bl", "br")
_CENTER_OFFSETS = np.array([-1, 1, 3])

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


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE), "motif must be 9x9"
    main = _major_foreground_color(motif, bg)
    patch = motif[np.ix_(BLOCK + _CENTER_OFFSETS, BLOCK + _CENTER_OFFSETS)].copy()
    patch[1, 1] = main
    return patch


def _directional_blocks(motif: np.ndarray):
    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                continue
            dr = br - 1
            dc = bc - 1
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            yield dr, dc, src


def _expanded_output(
    motif: np.ndarray, center: np.ndarray, bg: int, *, radii: tuple[int, ...] = (2, 3)
) -> np.ndarray:
    max_radius = max(radii, default=1)
    output_blocks = 2 * max_radius + 1
    expanded = np.full((output_blocks * BLOCK, output_blocks * BLOCK), int(bg), dtype=int)

    motif_with_center = motif.copy()
    paste(motif_with_center, center, BLOCK, BLOCK)
    paste(expanded, motif_with_center, (max_radius - 1) * BLOCK, (max_radius - 1) * BLOCK)

    center_idx = max_radius
    for dr, dc, src in _directional_blocks(motif):
        for radius in radii:
            paste(expanded, src, (center_idx - radius * dr) * BLOCK, (center_idx - radius * dc) * BLOCK)
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


def _assemble_motif(b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
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


def _sample_location(n: int) -> tuple[int, int]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a missing-center 3x3 block motif through two outer rings on a resized canvas.

    Input structure:
    - A single 9x9 motif appears on a uniform background.
    - The motif is a 3x3 arrangement of 3x3 blocks whose middle block is empty.
    - The eight pixels bordering that hole encode the boundary of the missing block.

    Rule:
    - Reconstruct the missing 3x3 center block from those eight inward-facing pixels,
      with the motif's majority foreground color placed at the patch center.
    - Treat each visible block as a directional template and copy it to the opposite
      side twice: first to complete the 5x5 ring, then one more step outward to form
      a 7x7 block layout, always without rotating the copied block.
    - Return the smallest background-filled canvas that contains both the original
      input placement and the full 21x21 iterative expansion.

    Key insight:
    - The visible 9x9 figure is only the middle window of a larger directional pattern.
      Solving requires noticing that propagation continues for a second round and then
      computing whatever canvas enlargement is needed to keep that entire 7x7 result.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h == MOTIF_SIZE, "motif must be a 9x9 square"

    motif = crop_bbox(g, motif_bbox)
    center = _center_patch(motif, bg)
    expanded = _expanded_output(motif, center, bg, radii=(2, 3))
    return _render_on_resized_canvas(g, expanded, int(r0) - 2 * BLOCK, int(c0) - 2 * BLOCK, bg)


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
    center = _center_patch(motif, BG)
    expanded_7 = _expanded_output(motif, center, BG, radii=(2, 3))
    expanded_5 = _expanded_output(motif, center, BG, radii=(2,))

    r0, c0 = _sample_location(n)
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)

    out = _render_on_resized_canvas(inp, expanded_7, r0 - 2 * BLOCK, c0 - 2 * BLOCK, BG)
    baseline = _render_on_resized_canvas(inp, expanded_5, r0 - BLOCK, c0 - BLOCK, BG)

    # Guard against accidental degeneration: every sample must require both
    # the second expansion round and a resized output canvas.
    if out.shape[0] <= n and out.shape[1] <= n:
        return None
    if np.array_equal(out, baseline):
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
