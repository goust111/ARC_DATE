# Variant of db0c5428
# Dimension: D14 — 多步处理与操作顺序 (multi_step)
# Change: 单步处理：识别块结构 -> 复制到外侧 -> 重建中心 -> 输出... → 外扩后再执行全局颜色替换

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D14 variant
# a) Concrete behavior:
#    The input is still the original 9x9 motif with a 3x3 hole. First we rebuild
#    the center block from the 8 visible boundary pixels and expand the motif to a
#    15x15 5x5-block layout exactly as in db0c5428. Then we do one more pass over
#    the finished output: the color used on the reconstructed center block's four
#    corners is globally replaced by the color used on that center block's four
#    edge midpoints.
#    Small trace: if the rebuilt center is [[6,4,6],[4,1,4],[6,4,6]], the base
#    task stops there. This variant performs an extra recolor 6->4 on the whole
#    expanded result, so every 6 in the final pattern becomes 4.
# b) Complexity calibration:
#    The recolor source and target are role-based and repeated many times:
#    source color appears on the four diagonal pixels around the hole, on the four
#    corners of the reconstructed center, and on the four extreme corners of the
#    expanded pattern before recolor. The target color appears on the hole-facing
#    cardinal pixels and on the center cross. That makes the added step visible.
# c) Solver assumptions to revisit:
#    Original copying logic assumes "copy block content verbatim" is the final
#    state. That is no longer true; after block pasting and center filling, the
#    solver must run a whole-grid recolor. The center patch must therefore be
#    computed before the recolor mapping is known, because the mapping is defined
#    by two roles inside the reconstructed center block.
# d) Placement legality:
#    The generator still places one 9x9 motif with at least 3 cells of margin so
#    the 15x15 expanded result fits. Recolor is global but color-only, so it does
#    not introduce overlap or out-of-bounds risks.
# e) Visual observability:
#    Across examples, the extra step is visible because one secondary color
#    disappears everywhere in the output while another secondary color grows in all
#    corresponding positions. The rule is tied to visible roles around the hole,
#    not to numeric color ids.
# f) Visual cleanliness:
#    No new markers are added. The output remains the same clean 5x5 block layout,
#    with only a uniform recolor pass on top.
# g) Color-change inferability:
#    The recolor uses only two structurally signposted colors: diagonal-around-hole
#    color -> cardinal-around-hole color. The generator guarantees these colors are
#    distinct and uses only 3-4 foreground colors total so the mapping is learnable.
# h) Generator anti-degeneration:
#    The recolor source color is reserved for the diagonal-around-hole orbit only,
#    preventing samples that accidentally collapse back into the original task.
# i) New solver assumptions checked against generation:
#    `main()` assumes the center-corner color and center-edge color can be read
#    from a valid 3x3 reconstructed patch. The generator enforces block size 3,
#    keeps the original D4/LR symmetry, and forces those two role colors to be
#    distinct non-background colors.


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


def _center_recolor_pair(center: np.ndarray) -> tuple[int, int]:
    # D14 step: use two role-defined colors from the reconstructed center.
    return int(center[0, 0]), int(center[0, 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the hollow 3x3-block motif, rebuild its center, then recolor once.

    Input structure:
    - A single 9x9 motif sits on a uniform background.
    - The motif is a 3x3 arrangement of 3x3 blocks with the middle block missing.
    - The eight visible pixels around that hole encode the missing 3x3 center.

    Rule:
    - Keep the visible 9x9 motif in place and copy each non-center block to the
      point-symmetric outer slot of the implied 5x5 block grid.
    - Reconstruct the center block from the hole boundary; its middle cell takes
      the motif's dominant foreground color.
    - After the whole expanded pattern exists, replace every pixel of the rebuilt
      center's corner color with the rebuilt center's edge-midpoint color.

    Key insight: The input still describes a larger symmetric block pattern, but
    the final output is no longer just "expand and fill". You must infer the
    center first, then read two structural colors from that center and use them as
    a global recolor mapping over the completed result.
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

    center = _center_patch(g, int(r0), int(c0), block, bg)
    paste(out, center, int(r0) + block, int(c0) + block)

    recolor_src, recolor_dst = _center_recolor_pair(center)
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
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 3, 4

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
    """Build a BLOCK x BLOCK array with some singleton orbits color-locked."""
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
    if not free_orbits:
        assert not required_colors
        return
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


def _make_blocks(majority, recolor_src, recolor_dst, extra_colors, corner_major, edge_major):
    """Build seed blocks while reserving the recolor source for diagonal-to-hole cells
    and the recolor target for cardinal-to-hole cells."""
    corner_locked = {_orbit_index(DIAG_ORBITS, (2, 2)): recolor_src}
    edge_locked = {_orbit_index(LR_ORBITS, (2, 1)): recolor_dst}

    b00, corner_free = _seed_block(DIAG_ORBITS, corner_locked, majority, corner_major)
    b01, edge_free = _seed_block(LR_ORBITS, edge_locked, majority, edge_major)
    free_orbits = corner_free + edge_free

    # Keep recolor_src unique to its structural role. Other minor orbits can use
    # recolor_dst or any extra color so the added recolor step stays readable.
    palette = [recolor_dst, *extra_colors]
    _assign_minor_colors(free_orbits, extra_colors, palette)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    return b00, b01


def _assemble_motif(b00, b01):
    """Assemble the 9x9 motif from two seed blocks via D4 symmetry; center block is bg."""
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
    """Reconstruct the center block from the 8 inner-boundary pixels of the motif."""
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = int(majority)
    return patch


def _expanded_output(motif, center):
    """Build the 15x15 expanded region before the D14 recolor pass."""
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


def _recolored_output(expanded, center):
    out = expanded.copy()
    recolor_src, recolor_dst = _center_recolor_pair(center)
    if recolor_src != recolor_dst:
        out[out == recolor_src] = recolor_dst
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build motif, verify via oracle, return input or None."""
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    recolor_dst = colors[1]
    recolor_src = colors[2]
    extra_colors = colors[3:]

    possible_totals = [
        total
        for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
        if 18 - total >= 2 + len(extra_colors)
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
        recolor_src,
        recolor_dst,
        extra_colors,
        corner_major,
        edge_major,
    )
    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    if int(center[0, 0]) == int(center[0, 1]):
        return None
    expanded = _expanded_output(motif, center)
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
    raise AssertionError("failed to generate a valid db0c5428 input")
