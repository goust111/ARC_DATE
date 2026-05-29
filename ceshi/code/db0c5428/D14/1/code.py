# Variant of db0c5428
# Dimension: D14 — 多步处理与操作顺序 (multi_step)
# Change: 单步处理：识别块结构 -> 复制到外侧 -> 重建中心 -> 输出... → 迭代外扩（先3x3到5x5，再5x5到7x7）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D14 variant
# a) Concrete behavior and trace:
#    The input stays close to the original task: one 9x9 motif made from a 3x3 array
#    of 3x3 blocks, with the middle 3x3 block missing. The eight visible blocks still
#    encode a directional ring around that hole, and the 8 inner-boundary pixels still
#    reconstruct the missing center block.
#    The change is only in output growth. In the original task, a top-edge source block
#    at block-offset (-1, 0) from the hole is copied once to the opposite slot at
#    offset (+2, 0), giving a 5x5 block layout. In this variant, that same directional
#    block is copied twice: first to (+2, 0), then again to (+3, 0). The final output
#    is therefore a 7x7 block layout whose occupied positions form three directional
#    rings: radius 1 (original motif), radius 2 (round one), radius 3 (round two).
# b) Complexity calibration:
#    The extra step is visually obvious because every sample gains a new outer ring of
#    8 copied blocks. The block patterns remain asymmetric enough that a human can track
#    which direction each copy follows, so the new recursion is visible rather than a
#    barely-different crop-size change. The rule still has one fixed depth (two rounds),
#    keeping the inference burden bounded.
# c) Review of old assumptions against the new rule:
#    - `bbox(g != bg)` still assumes a single foreground motif. Generator preserves that.
#    - `h == w and h % 3 == 0` still holds because the input motif is still 9x9.
#    - `block == 3` still matters because center reconstruction samples fixed offsets
#      around the hole. Generator still uses 3x3 blocks only.
#    - The original one-pass destination formula is no longer sufficient; a second pass
#      must not overwrite the center motif or assume that one expansion ends the task.
#      The new solver computes both outer radii explicitly from the same 8 directions.
#    - Pasting now starts at `(r0 - 2*block, c0 - 2*block)` instead of one-block margin.
#      Generator must therefore reserve two block-widths of clearance around the motif.
# d) Placement legality:
#    The generator places the 9x9 motif so that the full 21x21 (7x7 blocks) output fits
#    inside the canvas with at least one-cell outer padding. No copied ring overlaps any
#    other because radius-2 targets live at block offsets in {-2,0,2} and radius-3
#    targets live at {-3,0,3}; these sets are disjoint from the embedded 3x3 motif.
# e) Visual observability:
#    Comparing input/output pairs, a solver must notice that the first outward ring is
#    not the end state. The extra 8 blocks on the outermost ring repeat the same corner
#    and edge identities one step farther out, which is visible in just a few examples.
# f) Visual cleanliness:
#    The 7x7 layout is sparse but orderly: each round occupies a distinct ring, so there
#    is no ambiguous stacking. The center reconstruction remains a compact 3x3 patch.
# g) Color inferability:
#    No new color mapping is introduced. Colors keep their original role: one majority
#    color fills the center pixel, while the other 8 center pixels are read directly from
#    the inner boundary. The multi-step change is purely spatial and therefore human-
#    inferable from geometry rather than hidden color-id logic.


BLOCK = 3
MOTIF_BLOCKS = 3
OUTPUT_BLOCKS = 7
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
EXPANDED_SIZE = BLOCK * OUTPUT_BLOCKS
_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE), "motif must be 9x9"
    main = _major_foreground_color(motif, bg)
    patch = np.full((BLOCK, BLOCK), main, dtype=int)
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch[:, :] = motif[np.ix_(rows, cols)]
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


def _expanded_output(motif: np.ndarray, center: np.ndarray, bg: int) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), bg, dtype=int)

    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, 2 * BLOCK, 2 * BLOCK)

    center_idx = OUTPUT_BLOCKS // 2
    for dr, dc, src in _directional_blocks(motif):
        paste(expanded, src, (center_idx - 2 * dr) * BLOCK, (center_idx - 2 * dc) * BLOCK)
        paste(expanded, src, (center_idx - 3 * dr) * BLOCK, (center_idx - 3 * dc) * BLOCK)
    return expanded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a missing-center 3x3 block motif through two outward reflection rounds.

    Input structure:
    - A single 9x9 foreground motif sits on a uniform background.
    - The motif is a 3x3 array of 3x3 blocks with its middle block removed.
    - The eight pixels facing the hole still encode the missing center block.

    Rule:
    - Rebuild the center block from those inner-boundary pixels, keeping the majority
      foreground color in the patch center.
    - Keep the original 3x3 block motif as the middle ring of the answer.
    - Copy each of the eight directional source blocks to the opposite side twice:
      once to form the original 5x5 expansion, then once more to extend that pattern
      into a 7x7 block layout, always without rotating the copied block.

    Key insight:
    - The 8 visible blocks are directional templates around the hole. After the center
      is reconstructed, those same directions keep propagating outward ring by ring,
      so solving requires recognizing the second expansion step instead of stopping at
      the first 5x5 shell.
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
    expanded = _expanded_output(motif, center, bg)

    out = g.copy()
    paste(out, expanded, int(r0) - 2 * BLOCK, int(c0) - 2 * BLOCK)
    return out


# --------------- generator (o5x -> two_step_iterative_ring) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 24, 27       # square grid side length
OUTPUT_PAD = 1                              # min distance from final 21x21 region to edge

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

# Diagonal-symmetric orbits (b00 == b00.T)
DIAG_ORBITS = (
    ((0, 0),), ((1, 1),), ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)

# LR-symmetric orbits (b01 == fliplr(b01))
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),), ((1, 1),), ((2, 1),),
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
    expanded = _expanded_output(motif, center, BG)

    min_pos = 2 * BLOCK + OUTPUT_PAD
    max_pos = n - EXPANDED_SIZE - OUTPUT_PAD + 2 * BLOCK
    if min_pos > max_pos:
        return None
    r0 = random.randint(min_pos, max_pos)
    c0 = random.randint(min_pos, max_pos)

    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - 2 * BLOCK, c0 - 2 * BLOCK)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D14 input")
