# Combo variant of db0c5428 (index 27)
# combo_spec: D3/3+D7/3
# Name: 堆叠独立区域+分帧输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis
# ---------------
# concretization and behavior trace:
# - Input stays identical to the source task: one 9x9 motif made of a 3x3 array
#   of 3x3 blocks, with the middle 3x3 block missing on teal background.
# - Original task behavior had two linked actions inside one same-size canvas:
#   (1) rebuild the missing center block inside the motif, and
#   (2) copy each visible block to the opposite slot of a local 5x5 block frame.
# - This combo keeps action (1) but splits action (2) across a framed output:
#   the left frame shows only the repaired 9x9 motif, while the right frame shows
#   the eight non-center source blocks stacked in row-major order, skipping the
#   hole: (0,0)(0,1)(0,2)(1,0)(1,2)(2,0)(2,1)(2,2).
# - Example trace for source block (0,0):
#   original solver copied it to the opposite outer corner of the 5x5 layout;
#   D3/3 alone would move it into the first slot of a detached strip;
#   the fused rule keeps that detached-strip destination, but now the strip lives
#   inside the right frame of a side-by-side presentation, while the repaired
#   motif lives in the left frame.
#
# complexity calibration:
# - The core inference remains nontrivial because center reconstruction still
#   depends on the 8 inner-boundary pixels and the majority-color center cell.
# - The new layout freedom is intentionally limited to a single visible stack
#   mode: the detached region is always a horizontal 3x24 strip. This avoids an
#   extra hidden branch while still making the D3 change obvious.
# - The D7 frame separation is also single-mode: one 15x9 left panel, one 15x24
#   right panel, and one solid divider column. A solver must infer both content
#   transformation and global reframing.
#
# hidden-assumption audit of the original main():
# - `out = g.copy()` no longer holds because the output is no longer tied to the
#   input canvas size. The combo always emits a dedicated framed canvas.
# - `expanded_r0 = r0 - block` and `expanded_c0 = c0 - block` assumed all new
#   content remained in the motif neighborhood. That breaks under both D3/3 and
#   D7/3 because the copied blocks move into a detached frame.
# - `dst_r = expanded_r0 + (4 - 2*br) * block` and the analogous column formula
#   encoded the old point-symmetric 5x5 destinations. Those destinations are
#   replaced by fixed strip slots indexed by source block order.
# - `bbox(g != bg)` still works because the input still contains exactly one
#   isolated motif. The generator preserves that.
# - Center reconstruction still assumes 3x3 blocks and eight visible boundary
#   cue pixels. The generator keeps the original D4-symmetric motif recipe.
#
# placement legality:
# - Input placement remains the original safe square-canvas setup: a 9x9 motif is
#   placed with at least 3 cells of background around it, so bbox detection is
#   unambiguous and no clipping occurs.
# - Output is synthesized on a dedicated 15x34 canvas, so there is no risk of
#   overlap between left motif frame, divider, and right stack frame.
# - The right stack occupies a single 3x24 band centered vertically inside its
#   panel, so every copied block fits exactly once with no collisions.
#
# observability, cleanliness, and inferability:
# - Across examples, the output always has two visually separate regions:
#   a repaired local motif on the left and an ordered mapping strip on the right.
# - The right frame is clearly not the old 5x5 expansion because it contains only
#   one horizontal row of eight 3x3 blocks; the source-to-destination order is
#   readable directly from block positions, not from color ids.
# - The divider uses color 0, which is excluded from motif generation, so humans
#   can reliably parse the framed layout from a few examples.


DIVIDER_COLOR = 0
BLOCK = 3
MOTIF_BLOCKS = 3
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
FRAME_H = BLOCK * 5
LEFT_FRAME_W = MOTIF_SIZE
STACK_ORDER = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
RIGHT_FRAME_W = len(STACK_ORDER) * BLOCK


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch_from_motif(motif: np.ndarray, bg: int, block: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    main = _major_foreground_color(motif, bg)
    patch = np.full((block, block), main, dtype=int)
    offsets = (-1, 1, 3)
    hole_r = block
    hole_c = block
    for r in range(block):
        for c in range(block):
            if (r, c) == (block // 2, block // 2):
                continue
            patch[r, c] = int(motif[hole_r + offsets[r], hole_c + offsets[c]])
    return patch


def _filled_motif(motif: np.ndarray, center: np.ndarray, block: int) -> np.ndarray:
    out = motif.copy()
    paste(out, center, block, block)
    return out


def _stack_strip(motif: np.ndarray, bg: int, block: int) -> np.ndarray:
    strip = np.full((block, len(STACK_ORDER) * block), bg, dtype=int)
    for idx, (br, bc) in enumerate(STACK_ORDER):
        src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
        paste(strip, src, 0, idx * block)
    return strip


def _framed_output(motif_with_center: np.ndarray, strip: np.ndarray, bg: int) -> np.ndarray:
    out = np.full((FRAME_H, LEFT_FRAME_W + 1 + RIGHT_FRAME_W), bg, dtype=int)
    out[:, LEFT_FRAME_W] = DIVIDER_COLOR
    paste(out, motif_with_center, (FRAME_H - MOTIF_SIZE) // 2, 0)
    paste(out, strip, (FRAME_H - BLOCK) // 2, LEFT_FRAME_W + 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden center block, then present the repaired motif and detached block mapping as two frames.

    Input structure:
    - A square teal canvas contains exactly one 9x9 motif built from a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is missing, so only the eight surrounding blocks are visible.
    - The eight pixels facing the hole encode the border cells of the missing center block.

    Rule:
    - Recover the center block from those inner-boundary pixels and set its middle cell to the motif's majority foreground color.
    - Build a left frame containing only the repaired 9x9 motif.
    - Build a right frame containing the eight non-center source blocks copied without rotation into a detached horizontal strip in row-major order, skipping the center hole.

    Key insight: The hidden-center logic is unchanged, but the outer-copy logic is no longer a local 5x5 expansion around the motif; instead the task separates "repair the source object" and "export its surrounding blocks" into two clearly framed regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert DIVIDER_COLOR != bg, "divider color must differ from background"

    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    motif = crop_bbox(g, motif_bbox)
    h, w = map(int, motif.shape)
    assert h == w and h % 3 == 0, "motif must be a square 3x3 block grid"
    block = h // 3

    center = _center_patch_from_motif(motif, bg, block)
    motif_with_center = _filled_motif(motif, center, block)
    strip = _stack_strip(motif, bg, block)
    return _framed_output(motif_with_center, strip, bg)


# --------------- generator (combo variant) ---------------

# -- CFG: input canvas --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3

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


def _expected_output(motif: np.ndarray) -> np.ndarray:
    center = _center_patch_from_motif(motif, BG, BLOCK)
    motif_with_center = _filled_motif(motif, center, BLOCK)
    strip = _stack_strip(motif, BG, BLOCK)
    return _framed_output(motif_with_center, strip, BG)


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
    expected = _expected_output(motif)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
