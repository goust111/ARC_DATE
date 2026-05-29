# Combo variant of db0c5428 (index 24)
# combo_spec: D5/3+D15/2
# Name: 角像素编码中心+朝向标记像素

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis for the fused variant
# a) Concretization and behavior trace:
#    - The visible object is still one 9x9 motif on background, built from eight 3x3 blocks
#      around an empty 3x3 center, and the output is still the same 15x15 expansion frame.
#    - The original task used the eight inner-boundary pixels to rebuild the center and pasted
#      outward copies without rotating them. Here both of those assumptions change together.
#    - Each visible block now contains exactly one unique marker-color corner pixel. Its corner
#      position gives the block orientation and therefore the clockwise quarter-turn count used
#      for the outward copy.
#    - In each corner block, the center code pixel is not a fixed hole-facing corner anymore.
#      Instead, it is the corner immediately clockwise from the marker corner. Example:
#      if a corner block's marker sits at top-right, its code corner is bottom-right.
#    - The four extracted corner-code colors form a 2x2 symbol in corner-block order
#      (top-left, top-right / bottom-left, bottom-right), and that symbol is enlarged to
#      the 3x3 center patch by nearest-neighbor scaling:
#          a a b
#          a a b
#          c c d
#      for symbol [[a, b], [c, d]].
#    - So the solver must first identify the marker color, then for each corner block separate
#      "this corner is the orientation marker" from "the next clockwise corner is the center code".
# b) Complexity calibration:
#    - There is one global marker color and exactly four orientation states, each reused twice
#      across the eight visible blocks. That keeps the rotation rule learnable inside one sample.
#    - Center decoding uses only four colors from four corner blocks and one fixed enlargement
#      rule, so the symbolic part stays compact.
#    - The generator forces at least two distinct symbol colors and at least two non-majority
#      symbol cells, which prevents the filled center from collapsing into a trivial solid patch.
# c) Implicit-assumption review of main():
#    - `bbox(g != bg)` still works because the input is still a single 9x9 non-background motif.
#    - `block == 3` still holds; the variant changes control logic, not geometry.
#    - The original `_center_patch()` assumption "read the 8 cells around the hole" is invalid.
#      The fused solver must instead discover the marker color and use marker-relative code corners.
#    - The original outer loop assumed `paste(src, ...)` with no transform. That is invalid too:
#      outward copies now require marker-driven rotation.
#    - New solver assumptions:
#      1. One color appears exactly once in each visible block and only on corners.
#      2. In each corner block, the marker's clockwise-adjacent corner is an ordinary body color.
#      3. The corner-code extraction order is by corner-block location, not by color rank.
# d) Placement legality:
#    - The motif still sits with at least one full 3x3 block of padding on every side, so the
#      entire 15x15 expanded pattern fits without clipping.
#    - There is still exactly one motif object, so bbox-based extraction is unambiguous.
# e) Visual observability:
#    - The unique marker color is visibly repeated once per block and occupies all four corners
#      across the sample, so a human can infer that corner position drives outward rotation.
#    - The filled center reuses colors already visible in the corner blocks, and the source corner
#      is always the one immediately clockwise from the marker, giving a local geometric clue
#      rather than a hidden color-id lookup.
# f) Visual cleanliness:
#    - The marker has a single role in each raw block: it is the only pixel of its color there.
#      No other block pixels use that color, so the control channel stays legible.
#    - The center contains only the enlarged 2x2 symbol, with no overlaid boundary logic.
# g) Color-change inferability:
#    - No hidden remapping is used. Center colors are copied directly from visible corner pixels.
#    - The generator rejects cases where the fused center accidentally matches either the original
#      boundary-based center or the naive fixed-corner D5 center, so the combo stays behaviorally
#      distinct instead of silently regressing.
# h) Behavior metrics for non-degeneracy:
#    - We measure `rotation_profile(grid)` to confirm the sample uses all four rotation states
#      with histogram [2, 2, 2, 2].
#    - We measure center differences against two weaker decoders:
#      1. original inner-boundary reconstruction,
#      2. fixed hole-facing-corner reconstruction.
#      Requiring positive difference against both prevents either half of the combo from becoming
#      ignorable.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
MAX_RETRIES = 400

BG = 8
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
BODY_COLOR_COUNT_MIN, BODY_COLOR_COUNT_MAX = 2, 4

VISIBLE_BLOCKS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
CORNER_SLOTS = ((0, 0), (0, 2), (2, 0), (2, 2))
EDGE_SLOTS = ((0, 1), (1, 0), (1, 2), (2, 1))
CORNERS = ((0, 0), (0, 2), (2, 2), (2, 0))
CORNER_TO_CW = {(0, 0): 0, (0, 2): 1, (2, 2): 2, (2, 0): 3}
CW_NEXT_CORNER = {
    (0, 0): (0, 2),
    (0, 2): (2, 2),
    (2, 2): (2, 0),
    (2, 0): (0, 0),
}
CENTER_OFFSETS = np.array([-1, 1, 3], dtype=int)


def _block_slice(r0: int, c0: int, br: int, bc: int) -> tuple[int, int, int, int]:
    return (
        int(r0) + br * BLOCK,
        int(c0) + bc * BLOCK,
        int(r0) + (br + 1) * BLOCK - 1,
        int(c0) + (bc + 1) * BLOCK - 1,
    )


def _extract_motif(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h == MOTIF_SIZE, "motif must be a 9x9 3x3-block arrangement"
    return int(r0), int(c0), BLOCK


def _iter_visible_blocks(g: np.ndarray, r0: int, c0: int) -> list[np.ndarray]:
    return [crop_bbox(g, _block_slice(r0, c0, br, bc)) for br, bc in VISIBLE_BLOCKS]


def _marker_color(g: np.ndarray, r0: int, c0: int, bg: int) -> int:
    blocks = _iter_visible_blocks(g, r0, c0)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    candidates: list[int] = []
    for color in colors:
        ok = True
        for block in blocks:
            hits = np.argwhere(block == int(color))
            if hits.shape[0] != 1:
                ok = False
                break
            rc = tuple(int(x) for x in hits[0])
            if rc not in CORNERS:
                ok = False
                break
        if ok:
            candidates.append(int(color))
    assert len(candidates) == 1, "expected one unique marker color"
    return int(candidates[0])


def _marker_corner(block: np.ndarray, marker: int) -> tuple[int, int]:
    hits = np.argwhere(block == int(marker))
    assert hits.shape[0] == 1, "each block must contain the marker color exactly once"
    rc = tuple(int(x) for x in hits[0])
    assert rc in CORNERS, "marker color must sit on a corner"
    return rc


def _rotation_steps_cw(block: np.ndarray, marker: int) -> int:
    return int(CORNER_TO_CW[_marker_corner(block, marker)])


def _rotate_cw(block: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(block, -int(steps))


def _decode_symbol(symbol: np.ndarray) -> np.ndarray:
    idx = np.array([0, 0, 1], dtype=int)
    return symbol[np.ix_(idx, idx)].copy()


def _code_corner_from_marker(marker_corner: tuple[int, int]) -> tuple[int, int]:
    return CW_NEXT_CORNER[marker_corner]


def _corner_symbol(g: np.ndarray, r0: int, c0: int, marker: int) -> np.ndarray:
    symbol = np.zeros((2, 2), dtype=int)
    for br, bc in CORNER_SLOTS:
        block = crop_bbox(g, _block_slice(r0, c0, br, bc))
        mr, mc = _marker_corner(block, marker)
        code_r, code_c = _code_corner_from_marker((mr, mc))
        symbol[br // 2, bc // 2] = int(block[code_r, code_c])
    return symbol


def _naive_hole_symbol(g: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.array(
        [
            [g[int(r0) + 2, int(c0) + 2], g[int(r0) + 2, int(c0) + 2 * BLOCK]],
            [g[int(r0) + 2 * BLOCK, int(c0) + 2], g[int(r0) + 2 * BLOCK, int(c0) + 2 * BLOCK]],
        ],
        dtype=int,
    )


def _major_body_color(g: np.ndarray, bg: int, marker: int) -> int:
    vals, counts = np.unique(g[(g != int(bg)) & (g != int(marker))], return_counts=True)
    assert vals.size, "input must contain body colors"
    return int(vals[int(np.argmax(counts))])


def _old_boundary_center(g: np.ndarray, r0: int, c0: int, bg: int, marker: int) -> np.ndarray:
    main = _major_body_color(g, bg, marker)
    patch = np.full((BLOCK, BLOCK), main, dtype=int)
    hole_r = int(r0) + BLOCK
    hole_c = int(c0) + BLOCK
    for rr in range(BLOCK):
        for cc in range(BLOCK):
            if rr == 1 and cc == 1:
                continue
            patch[rr, cc] = int(g[hole_r + CENTER_OFFSETS[rr], hole_c + CENTER_OFFSETS[cc]])
    return patch


def _center_patch(g: np.ndarray, r0: int, c0: int, marker: int) -> np.ndarray:
    return _decode_symbol(_corner_symbol(g, r0, c0, marker))


def rotation_profile(grid: np.ndarray) -> list[int]:
    """Clockwise quarter-turns required by the eight visible blocks."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    r0, c0, _block = _extract_motif(g, bg)
    marker = _marker_color(g, r0, c0, bg)
    return [_rotation_steps_cw(block, marker) for block in _iter_visible_blocks(g, r0, c0)]


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether both fused mechanisms are active on a generated sample."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    r0, c0, _block = _extract_motif(g, bg)
    marker = _marker_color(g, r0, c0, bg)
    center = _center_patch(g, r0, c0, marker)
    naive_corner_center = _decode_symbol(_naive_hole_symbol(g, r0, c0))
    old_boundary_center = _old_boundary_center(g, r0, c0, bg, marker)
    profile = rotation_profile(g)
    motif = g[int(r0) : int(r0) + MOTIF_SIZE, int(c0) : int(c0) + MOTIF_SIZE].copy()
    rotated = _expanded_output(motif, center, marker)
    no_rotation = _expanded_no_rotation(motif, center)
    return {
        "center_vs_naive_corner_diff": int(np.sum(center != naive_corner_center)),
        "center_vs_old_boundary_diff": int(np.sum(center != old_boundary_center)),
        "nonzero_rotations": int(sum(step != 0 for step in profile)),
        "outer_vs_no_rotation_diff": int(np.sum(rotated != no_rotation)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a marked 3x3-block motif into a 5x5 block layout and decode its center.

    Input structure:
    - One 9x9 motif sits on teal background, formed by eight visible 3x3 blocks around an empty 3x3 middle block.
    - Every visible block contains exactly one special marker-color corner pixel, and that corner gives the block's orientation.
    - In each corner block, the corner immediately clockwise from the marker is the data corner for the missing center, so the marker and code occupy neighboring corners.

    Rule:
    - Keep the original 9x9 motif in place.
    - Rebuild the center by reading one marker-guided corner from each of the four corner blocks, arranging those four colors into a 2x2 symbol, and enlarging that symbol to 3x3 with nearest-neighbor scaling.
    - Copy each visible block into the opposite outer slot of the implied 5x5 block grid, rotating the whole 3x3 block by the clockwise quarter-turn count encoded by its marker corner.

    Key insight:
    - The same corner marker controls two decisions: it tells you how the outward copy rotates, and it tells you which adjacent corner in a corner block is real center data. The missing center cannot be solved by reading fixed hole-facing corners anymore.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    r0, c0, block = _extract_motif(g, bg)
    marker = _marker_color(g, r0, c0, bg)

    out = g.copy()
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = crop_bbox(g, _block_slice(int(r0), int(c0), br, bc))
            rotated = _rotate_cw(src, _rotation_steps_cw(src, marker))
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, rotated, dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), marker), int(r0) + block, int(c0) + block)
    return out


def _sample_symbol(colors: list[int], majority: int) -> np.ndarray:
    while True:
        vals = [random.choice(colors) for _ in range(4)]
        if len(set(vals)) < 2:
            continue
        if sum(v != majority for v in vals) < 2:
            continue
        return np.array(vals, dtype=int).reshape(2, 2)


def _random_block(
    marker_corner: tuple[int, int],
    marker_color: int,
    majority: int,
    accent_colors: list[int],
    forced_cells: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    forced_cells = dict(forced_cells or {})
    block = np.full((BLOCK, BLOCK), int(majority), dtype=int)
    block[marker_corner] = int(marker_color)
    for coord, color in forced_cells.items():
        assert coord != marker_corner, "forced cell overlaps marker"
        assert int(color) != int(marker_color), "marker color is reserved"
        block[coord] = int(color)

    free_cells = [
        (r, c)
        for r in range(BLOCK)
        for c in range(BLOCK)
        if (r, c) != marker_corner and (r, c) not in forced_cells
    ]
    random.shuffle(free_cells)
    extra_count = random.randint(1, min(3, len(free_cells)))
    extra_palette = list(accent_colors)
    for coord in free_cells[:extra_count]:
        block[coord] = int(random.choice(extra_palette))
    return block


def _assemble_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _expanded_output(motif: np.ndarray, center: np.ndarray, marker_color: int) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br, bc in VISIBLE_BLOCKS:
        src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        rotated = _rotate_cw(src, _rotation_steps_cw(src, marker_color))
        paste(expanded, rotated, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _expanded_no_rotation(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br, bc in VISIBLE_BLOCKS:
        src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _marker_unique_in_motif(motif: np.ndarray, marker_color: int) -> bool:
    colors = [int(c) for c in np.unique(motif) if int(c) != BG]
    candidates: list[int] = []
    for color in colors:
        ok = True
        for br, bc in VISIBLE_BLOCKS:
            block = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            hits = np.argwhere(block == int(color))
            if hits.shape[0] != 1:
                ok = False
                break
            if tuple(int(x) for x in hits[0]) not in CORNERS:
                ok = False
                break
        if ok:
            candidates.append(int(color))
    return candidates == [int(marker_color)]


def _body_majority_ok(motif: np.ndarray, majority: int, marker_color: int) -> bool:
    vals, counts = np.unique(motif[(motif != BG) & (motif != int(marker_color))], return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    if not freq or int(majority) not in freq:
        return False
    if len(freq) == 1:
        return True
    return freq[int(majority)] > max(v for k, v in freq.items() if k != int(majority))


def _center_from_motif(motif: np.ndarray, marker_color: int) -> np.ndarray:
    grid = np.asarray(motif, dtype=int)
    symbol = np.zeros((2, 2), dtype=int)
    for br, bc in CORNER_SLOTS:
        block = grid[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        mr, mc = _marker_corner(block, marker_color)
        cr, cc = _code_corner_from_marker((mr, mc))
        symbol[br // 2, bc // 2] = int(block[cr, cc])
    return _decode_symbol(symbol)


def _naive_hole_center_from_motif(motif: np.ndarray) -> np.ndarray:
    symbol = np.array(
        [
            [motif[2, 2], motif[2, 6]],
            [motif[6, 2], motif[6, 6]],
        ],
        dtype=int,
    )
    return _decode_symbol(symbol)


def _old_boundary_center_from_motif(motif: np.ndarray, majority: int) -> np.ndarray:
    patch = motif[np.ix_(BLOCK + CENTER_OFFSETS, BLOCK + CENTER_OFFSETS)].copy()
    patch[1, 1] = int(majority)
    return patch


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)

    marker_color = random.choice(PALETTE)
    body_palette = [c for c in PALETTE if c != marker_color]
    body_count = random.randint(BODY_COLOR_COUNT_MIN, BODY_COLOR_COUNT_MAX)
    body_colors = random.sample(body_palette, body_count)
    majority = int(body_colors[0])
    accent_colors = [int(c) for c in body_colors[1:]]
    symbol = _sample_symbol([int(c) for c in body_colors], majority)

    corner_markers = random.sample(list(CORNERS), len(CORNERS))
    edge_markers = random.sample(list(CORNERS), len(CORNERS))

    blocks: dict[tuple[int, int], np.ndarray] = {}
    for slot, marker_corner in zip(CORNER_SLOTS, corner_markers):
        code_corner = _code_corner_from_marker(marker_corner)
        code_color = int(symbol[slot[0] // 2, slot[1] // 2])
        forced = {code_corner: code_color}
        blocks[slot] = _random_block(marker_corner, marker_color, majority, accent_colors, forced)

    for slot, marker_corner in zip(EDGE_SLOTS, edge_markers):
        blocks[slot] = _random_block(marker_corner, marker_color, majority, accent_colors)

    motif = _assemble_motif(blocks)
    if not _marker_unique_in_motif(motif, marker_color):
        return None
    if not _body_majority_ok(motif, majority, marker_color):
        return None

    center = _center_from_motif(motif, marker_color)
    naive_hole_center = _naive_hole_center_from_motif(motif)
    old_boundary_center = _old_boundary_center_from_motif(motif, majority)
    if np.array_equal(center, naive_hole_center):
        return None
    if np.array_equal(center, old_boundary_center):
        return None

    expanded = _expanded_output(motif, center, marker_color)
    if np.array_equal(expanded, _expanded_no_rotation(motif, center)):
        return None

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    profile = rotation_profile(inp)
    if sorted(profile) != [0, 0, 1, 1, 2, 2, 3, 3]:
        return None
    metrics = behavior_metrics(inp)
    if metrics["center_vs_naive_corner_diff"] <= 0:
        return None
    if metrics["center_vs_old_boundary_diff"] <= 0:
        return None
    if metrics["nonzero_rotations"] < 6:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
