# Variant of db0c5428
# Dimension: D4 — 块复制时的几何变换 (transform)
# Change: 复制到对侧外部槽位时不旋转不翻转，保持原始方向... → 复制时颜色取反（使用颜色映射表）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior:
#    The original task copies each visible 3x3 block to its opposite outer slot
#    with identical pixels. This variant keeps the same block-to-slot geometry and
#    the same center reconstruction, but every copied outer block swaps the two
#    visible foreground colors. Example: if a source block uses colors {2, 7},
#    then its outer copy keeps the same 3x3 shape while every 2 becomes 7 and
#    every 7 becomes 2. The middle 3x3 hole is still reconstructed from the eight
#    inner-boundary pixels and keeps the majority color at its center.
# b) Complexity calibration:
#    To make the color mapping unambiguous from only a few ARC examples, the
#    generator is tightened to exactly two foreground colors. With three or four
#    colors, a pure output-only color remapping would be much harder to infer.
#    Keeping two colors makes the mapping repeated eight times per output and
#    visually obvious without adding unrelated structural changes.
# c) Hidden assumptions audit for main():
#    - `_major_foreground_color(...)` still assumes one foreground color is the
#      global majority. The generator preserves that.
#    - `_center_patch(...)` still reads the same eight inner-boundary pixels and
#      does not depend on the outer-copy rule, so it remains valid unchanged.
#    - The original outer-copy loop assumed "copy pixels verbatim". That is the
#      only solver behavior that must change here; now the copied patch must pass
#      through a color map before being pasted.
#    - The new helper `_infer_swap_map(...)` assumes exactly two foreground colors
#      are visible in the motif. The generator guarantees that and ensures both
#      colors actually occur.
# d) Placement legality:
#    The motif is still 9x9 and the expanded output still occupies a 15x15 block
#    region. Keeping `MOTIF_PAD = 3` preserves the original proof that placing the
#    9x9 motif with that padding leaves room for the full 15x15 output.
# e) Visual observability:
#    The new rule changes all 72 pixels of the eight outer copied blocks relative
#    to the original "copy verbatim" behavior, because every outer block uses only
#    the two swappable foreground colors. This makes the D4 change easy to notice.
# f) Visual cleanliness:
#    No extra markers or overlapping layers are introduced. The output keeps the
#    original clean 5x5 block lattice; only the colors of the outer copies differ.
# g) Color-change inferability:
#    The mapping is purely pairwise and symmetric: the two visible non-background
#    colors exchange roles. This uses repeated support across all eight copied
#    blocks and avoids any dependency on numeric color ids.
# h) New solver assumptions review:
#    Every loop in `main()` still only manipulates one motif and one expanded
#    lattice. The only additional assumption is the two-color foreground set, and
#    `generate_input()` enforces it by construction.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _swap_map_from_colors(colors) -> dict[int, int]:
    colors = sorted(int(x) for x in colors)
    assert len(colors) == 2, "variant expects exactly two foreground colors"
    a, b = colors
    return {a: b, b: a}


def _infer_swap_map(g: np.ndarray, bg: int) -> dict[int, int]:
    return _swap_map_from_colors(np.unique(g[g != int(bg)]))


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 block layout with color-swapped outer copies.

    Input structure:
    - Teal background surrounds one 9x9 motif made of eight visible 3x3 blocks.
    - The middle 3x3 block is missing and left as background.
    - The visible motif uses exactly two foreground colors, and the pixels facing
      the hole encode the border of the missing center block.

    Rule:
    - Keep the original 9x9 motif in place and rebuild its missing center block.
    - Copy each non-center 3x3 block to the opposite outer slot of the implied
      5x5 block lattice, preserving its geometry.
    - Before pasting each outer copy, swap the motif's two foreground colors.

    Key insight: The spatial expansion is the same as the base task, but the
    repeated outer copies are recolored by the motif's own two-color swap map.
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
    color_map = _infer_swap_map(crop_bbox(g, motif_bbox), bg)

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
            paste(out, _apply_color_map(src, color_map), dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
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
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 2

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


def _expanded_output(motif, center, color_map):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, _apply_color_map(src, color_map), (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]
    color_map = _swap_map_from_colors(colors)

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
    expanded = _expanded_output(motif, center, color_map)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 input")
