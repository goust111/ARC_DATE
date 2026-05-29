# Combo variant of db0c5428 (index 8)
# combo_spec: D4/2+D14/2
# Name: 颜色取反复制+全局颜色替换后处理

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The base task expands the 9x9 hollow motif to a 15x15 5x5-block lattice and
#    reconstructs the missing center from the eight pixels facing the hole. This
#    combo keeps that geometry, but changes the color logic in two ordered steps.
#    First, each copied outer block swaps the motif's majority color with the
#    color that appears on the rebuilt center block's edge midpoints. Second, once
#    the whole pattern exists, every pixel of the rebuilt center block's corner
#    color is globally recolored to that same edge-midpoint color.
#    Example trace with center [[7,4,7],[4,1,4],[7,4,7]]:
#    - Base task outer copy of a source block [[1,1,4],[1,7,1],[4,1,1]] stays the
#      same, then the task ends.
#    - Combo step 1 copies it outward as [[4,4,1],[4,7,4],[1,4,4]] because
#      1<->4 is the local swap and 7 is untouched there.
#    - Combo step 2 recolors 7->4 everywhere, so the final outer copy becomes
#      [[4,4,1],[4,4,4],[1,4,4]], while the original motif's own 7-pixels also
#      become 4. This preserves a visible difference between central blocks and
#      outer copies, so the two stages do not collapse into one.
# b) Complexity calibration:
#    The generator uses exactly three foreground colors:
#    - majority color M
#    - swap target / center-edge color E
#    - global recolor source / center-corner color C
#    This keeps the color logic small enough to infer from 2-4 ARC examples.
#    The outer-copy rule repeatedly shows M<->E across all eight copied blocks,
#    while the post-process repeatedly removes C everywhere. That gives strong,
#    repeated evidence for both steps without introducing many free colors.
# c) Hidden assumptions audit for main():
#    - `_major_foreground_color(...)` assumes a unique dominant foreground color.
#      The generator preserves that by keeping majority counts high.
#    - `_center_patch(...)` still relies on block size 3 and on the inner-boundary
#      sampling pattern from the original task. The combo does not alter this.
#    - The original copy loop assumed pasted outer blocks were final. That no
#      longer holds: copied blocks must first pass through a local swap map.
#    - The new local swap rule assumes the color on `center[0,1]` is distinct from
#      the majority color. The generator locks that orbit to a separate color.
#    - The new global recolor rule assumes `center[0,0]` and `center[0,1]` are
#      distinct non-background colors. The generator enforces that too.
#    - The order matters: if recolor were applied before the outer-copy swap, the
#      inferred swap pair could change or collapse. `main()` therefore reconstructs
#      the center first, uses it to define both mappings, performs swapped copying,
#      then runs the recolor pass last.
# d) Placement legality:
#    The motif remains 9x9 and the expanded result remains 15x15, so the original
#    padding argument still works: placing the motif with `MOTIF_PAD = 3` leaves
#    room for the full expanded output. The added color steps do not change shape
#    occupancy, so they cannot create overlap or out-of-bounds writes.
# e) Visual observability:
#    A human can see two separate regularities. First, outer copies are not merely
#    duplicates of their opposite source blocks: majority-colored pixels and the
#    edge-role color exchange only in the outer ring. Second, after expansion, one
#    third color disappears globally and merges into the edge-role color. Because
#    these effects repeat across the whole pattern, the composite rule is visible.
# f) Visual cleanliness:
#    No extra markers or overlaid objects are introduced. The same clean 5x5 block
#    lattice is preserved, with color changes aligned to block boundaries and to
#    center-derived roles, so the output remains easy to parse.
# g) Change inferability:
#    The swap pair and recolor pair are not arbitrary numeric color ids. Both are
#    anchored to structural roles around the missing center:
#    - local outer-copy swap: majority color <-> center edge-midpoint color
#    - global recolor: center corner color -> center edge-midpoint color
#    The shared target color ties the two stages together while still leaving the
#    outer ring visibly different from the center after composition.
# h) Anti-degeneration checks built into generation:
#    - The recolor source color C is reserved for the diagonal-to-hole orbit only,
#      so the global recolor is always meaningful and observable.
#    - The majority total is capped below 16, guaranteeing at least one additional
#      free minor orbit beyond the two locked structural colors; this prevents the
#      motif from reducing to a too-thin edge/corner cue.
#    - Because local swap is M<->E but global recolor is C->E, applying the combo
#      never matches the plain D14-only behavior on generated samples.
# i) New solver assumptions checked against generation:
#    Every loop in `main()` still handles exactly one motif and one expanded
#    lattice. The generator guarantees:
#    - exactly one foreground motif exists
#    - block size is 3
#    - majority, center-edge, and center-corner colors are pairwise distinct
#    - the local swap pair is inferable from the input itself via center roles
#    - the global recolor pass always changes at least the four diagonal-to-hole
#      positions and their descendants in the finished output.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _apply_color_map(block: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = block.copy()
    for src, dst in color_map.items():
        out[block == int(src)] = int(dst)
    return out


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


def _outer_swap_map(majority: int, center: np.ndarray) -> dict[int, int]:
    swap_color = int(center[0, 1])
    if int(majority) == swap_color:
        return {}
    return {int(majority): swap_color, swap_color: int(majority)}


def _center_recolor_pair(center: np.ndarray) -> tuple[int, int]:
    return int(center[0, 0]), int(center[0, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the hollow 3x3-block motif with swapped outer copies, then recolor once.

    Input structure:
    - One 9x9 motif sits on a uniform background and is made from eight visible
      3x3 blocks arranged around a missing middle block.
    - The motif still has the original D4-style block relationships, and the
      eight pixels immediately facing the hole encode the missing 3x3 center.
    - One foreground color is globally dominant, while the rebuilt center reveals
      two more role colors: its corner color and its edge-midpoint color.

    Rule:
    - Reconstruct the missing center block from the hole boundary; its middle cell
      is the motif's dominant foreground color.
    - Copy every non-center 3x3 block to the point-symmetric outer slot of the
      implied 5x5 block lattice, but in those outer copies swap the dominant
      color with the rebuilt center's edge-midpoint color.
    - After the center is filled and all outer copies are placed, replace every
      pixel of the rebuilt center's corner color with the rebuilt center's
      edge-midpoint color across the entire output.

    Key insight: The puzzle is no longer just geometric expansion. The missing
    center provides two role-defined colors that drive two ordered color stages:
    a local swap only on the outer ring, then a whole-grid recolor. The final
    output is understood only by keeping those stages separate.
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

    center = _center_patch(g, int(r0), int(c0), block, bg)
    copy_map = _outer_swap_map(_major_foreground_color(g, bg), center)
    recolor_src, recolor_dst = _center_recolor_pair(center)

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
            paste(out, _apply_color_map(src, copy_map), dst_r, dst_c)

    paste(out, center, int(r0) + block, int(c0) + block)

    if recolor_src != recolor_dst:
        out[out == recolor_src] = recolor_dst
    return out


# --------------- generator (o5x) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_PAD = 3

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 3, 3

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 15
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

_CENTER_OFFSETS = np.array([-1, 1, 3])


def _orbit_index(orbits, coord):
    for idx, orbit in enumerate(orbits):
        if coord in orbit:
            return idx
    raise AssertionError(f"coordinate {coord} not found in any orbit")


def _major_orbits(orbits, blocked_idxs, target_count):
    blocked_idxs = set(blocked_idxs)
    usable = [idx for idx in range(len(orbits)) if idx not in blocked_idxs]
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


def _seed_block(orbits, locked_orbit_colors, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, locked_orbit_colors.keys(), target_count)
    free_orbits = []
    for idx, orbit in enumerate(orbits):
        if idx in locked_orbit_colors:
            color = int(locked_orbit_colors[idx])
            for r, c in orbit:
                block[r, c] = color
        elif idx in major_idxs:
            for r, c in orbit:
                block[r, c] = int(majority)
        else:
            free_orbits.append((block, orbit))
    return block, free_orbits


def _assign_minor_colors(free_orbits, required_colors, palette):
    assert len(free_orbits) >= len(required_colors)
    random.shuffle(free_orbits)
    guaranteed = list(required_colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(palette) for _ in range(len(free_orbits) - len(guaranteed))
    ]
    for (block, orbit), color in zip(free_orbits, assignments):
        for r, c in orbit:
            block[r, c] = int(color)


def _make_blocks(majority, swap_color, recolor_src, corner_major, edge_major):
    corner_locked = {_orbit_index(DIAG_ORBITS, (2, 2)): recolor_src}
    edge_locked = {_orbit_index(LR_ORBITS, (2, 1)): swap_color}

    b00, corner_free = _seed_block(DIAG_ORBITS, corner_locked, majority, corner_major)
    b01, edge_free = _seed_block(LR_ORBITS, edge_locked, majority, edge_major)
    free_orbits = corner_free + edge_free

    # All remaining minor structure uses the swap color so the local M<->E outer
    # swap has repeated support beyond the single hole-facing orbit.
    _assign_minor_colors(free_orbits, [swap_color], [swap_color])
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
    patch[1, 1] = int(majority)
    return patch


def _expanded_output(motif, center, copy_map):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(
                expanded,
                _apply_color_map(src, copy_map),
                (4 - 2 * br) * BLOCK,
                (4 - 2 * bc) * BLOCK,
            )
    return expanded


def _recolored_output(expanded, center):
    out = expanded.copy()
    recolor_src, recolor_dst = _center_recolor_pair(center)
    if recolor_src != recolor_dst:
        out[out == recolor_src] = recolor_dst
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    swap_color = colors[1]
    recolor_src = colors[2]

    possible_totals = [
        total
        for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
        if 18 - total >= 3
    ]
    majority_total = random.choice(possible_totals)
    pairs = [
        (corner, majority_total - corner)
        for corner in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1)
        if EDGE_MAJOR_MIN <= majority_total - corner <= EDGE_MAJOR_MAX
    ]
    corner_major, edge_major = random.choice(pairs)

    b00, b01 = _make_blocks(
        majority,
        swap_color,
        recolor_src,
        corner_major,
        edge_major,
    )
    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    if len({int(majority), int(center[0, 0]), int(center[0, 1])}) != 3:
        return None

    copy_map = _outer_swap_map(int(majority), center)
    expanded = _expanded_output(motif, center, copy_map)
    recolored = _recolored_output(expanded, center)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, recolored, r0 - BLOCK, c0 - BLOCK)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
