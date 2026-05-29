# Variant of db0c5428
# Dimension: D3 — 外部槽位的网格布局 (dst)
# Change: 5x5块网格中原始3x3占据(1,1)-(3,3)，新块填充到对称外侧的8个位置，其余12个外部位置保持背景色... → 堆叠到独立的输出区域

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis
# ----------------
# concretization:
# - Input keeps the original object: one 9x9 motif made from a 3x3 grid of 3x3 blocks,
#   with the middle block missing and encoded by the 8 visible inner-boundary pixels.
# - Output still fills the center block exactly as in the source task.
# - The only destination-layout change is that the 8 non-center source blocks are no longer
#   copied into a surrounding 5x5 ring. They are copied, without rotation, into a detached
#   strip: a single 3x24 row on wide canvases or a single 24x3 column on tall canvases.
# - The strip order is fixed and visible: row-major over the source 3x3 block grid, skipping
#   the hole, i.e. (0,0)(0,1)(0,2)(1,0)(1,2)(2,0)(2,1)(2,2).
#
# behavior trace:
# - Original logic for source block (0,0): place it into the opposite corner of the 5x5
#   block frame at block coordinates (4,4); repeated for all 8 blocks.
# - Variant logic for the same source block: make it the first item of the detached strip;
#   source block (0,1) becomes the second item; ... ; source block (2,2) becomes the eighth.
# - The center block logic does not move: the 8 boundary pixels still rebuild the missing 3x3
#   block in the motif itself.
#
# complexity calibration:
# - The variant uses only two layout modes because more destination modes would be hard to infer
#   from 2-4 examples. Wide grids always use a row strip; tall grids always use a column strip.
# - Internal block asymmetry from the original generator is preserved, so the ordered strip is
#   visually readable and not reducible to a line of identical solid-color blocks.
#
# hidden-assumption audit for the original main():
# - `expanded_r0 = r0 - block` / `expanded_c0 = c0 - block` assumed the answer lived in one
#   local 15x15 neighborhood around the motif. That is false here and must be removed.
# - `dst_r = expanded_r0 + (4 - 2*br) * block` and the analogous `dst_c` encoded the old
#   point-symmetric outer-ring slots. Those formulas are exactly the D3 behavior being changed.
# - The block extraction itself is still valid because source blocks are unchanged.
# - `paste(out, center_patch, r0+block, c0+block)` remains valid because center reconstruction
#   is not part of the D3 change.
#
# placement legality:
# - Wide case: H in [16,19], W in [28,31]. The motif is placed with `r0 >= 4` so a detached
#   3-row strip at the top cannot touch it, and with `c0 <= W-24` so the whole strip fits.
# - Tall case: H in [28,31], W in [16,19]. The motif is placed with `c0 >= 4` so a detached
#   3-column strip at the left cannot touch it, and with `r0 <= H-24` so the whole strip fits.
# - Because only one axis is long enough for 24 cells, the row/column choice is unambiguous.
#
# observability and cleanliness:
# - The detached strip occupies a clean reserved band with at least one background row/column
#   separating it from the motif, so the two regions do not visually merge.
# - The aspect ratio itself is a visible cue for whether the detached area is a row or a column.
# - No color-id-based ordering or recoloring is introduced; only visible spatial order matters.


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


def _use_horizontal_stack(shape: tuple[int, int], block: int) -> bool:
    stack_extent = len(STACK_ORDER) * block
    h, w = map(int, shape)
    if w >= stack_extent and h < stack_extent:
        return True
    if h >= stack_extent and w < stack_extent:
        return False
    return w > h


def _stack_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        strip = np.full((block, len(STACK_ORDER) * block), bg, dtype=int)
    else:
        strip = np.full((len(STACK_ORDER) * block, block), bg, dtype=int)

    for idx, (br, bc) in enumerate(STACK_ORDER):
        src = crop_bbox(
            g,
            (
                int(r0) + br * block,
                int(c0) + bc * block,
                int(r0) + (br + 1) * block - 1,
                int(c0) + (bc + 1) * block - 1,
            ),
        )
        if horizontal:
            paste(strip, src, 0, idx * block)
        else:
            paste(strip, src, idx * block, 0)
    return strip


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center block, then move the eight surrounding blocks into a detached strip.

    Input structure:
    - One 9x9 foreground motif sits on a background canvas.
    - The motif is a 3x3 arrangement of 3x3 blocks with the center block missing.
    - The 8 pixels facing the hole encode the border of that missing center block.

    Rule:
    - Keep the original motif in place and reconstruct its center block from the visible boundary cues.
    - Copy each of the eight non-center 3x3 blocks without rotation into a separate stacking area.
    - On wide grids the stack is a horizontal row; on tall grids it is a vertical column.

    Key insight: The source object is still the same symmetric 3x3-block motif, but the output no longer
    completes a local 5x5 frame; instead it exports the surrounding blocks into a detached ordered strip.
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
    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)

    horizontal = _use_horizontal_stack(g.shape, block)
    strip = _stack_patch(g, int(r0), int(c0), block, bg, horizontal)
    if horizontal:
        paste(out, strip, 0, int(c0))
    else:
        paste(out, strip, int(r0), 0)
    return out


# --------------- generator (o5x variant) ---------------

# -- CFG: canvas & structure --
WIDE_H_MIN, WIDE_H_MAX = 16, 19
WIDE_W_MIN, WIDE_W_MAX = 28, 31
TALL_H_MIN, TALL_H_MAX = 28, 31
TALL_W_MIN, TALL_W_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
STACK_BLOCKS = len(STACK_ORDER)
STACK_SIZE = BLOCK * STACK_BLOCKS
STACK_GAP = 1

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

_CENTER_OFFSETS = np.array([-1, 1, 3])


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


def _gen_center_patch(motif, majority):
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _strip_from_motif(motif: np.ndarray, horizontal: bool) -> np.ndarray:
    return _stack_patch(motif, 0, 0, BLOCK, BG, horizontal)


def _stacked_output(shape: tuple[int, int], motif: np.ndarray, center: np.ndarray, r0: int, c0: int, horizontal: bool) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(out, output_motif, r0, c0)
    strip = _strip_from_motif(motif, horizontal)
    if horizontal:
        paste(out, strip, 0, c0)
    else:
        paste(out, strip, r0, 0)
    return out


def _sample_canvas(horizontal: bool) -> tuple[int, int, int, int]:
    if horizontal:
        h = random.randint(WIDE_H_MIN, WIDE_H_MAX)
        w = random.randint(WIDE_W_MIN, WIDE_W_MAX)
        r0 = random.randint(BLOCK + STACK_GAP, h - MOTIF_SIZE)
        c0 = random.randint(0, w - STACK_SIZE)
    else:
        h = random.randint(TALL_H_MIN, TALL_H_MAX)
        w = random.randint(TALL_W_MIN, TALL_W_MAX)
        r0 = random.randint(0, h - STACK_SIZE)
        c0 = random.randint(BLOCK + STACK_GAP, w - MOTIF_SIZE)
    return h, w, r0, c0


def _gen_attempt() -> np.ndarray | None:
    horizontal = bool(random.getrandbits(1))
    h, w, r0, c0 = _sample_canvas(horizontal)

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

    inp = np.full((h, w), BG, dtype=int)
    paste(inp, motif, r0, c0)
    out = _stacked_output((h, w), motif, center, r0, c0, horizontal)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D3-variant input")
