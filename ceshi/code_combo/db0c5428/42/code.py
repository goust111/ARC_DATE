# Combo variant of db0c5428 (index 42)
# combo_spec: D1/3+D12/0
# Name: 颜色匹配映射+特殊标记色空洞

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The visible object is still the 9x9 middle of a 5x5 block lattice, but the
#      center 3x3 is no longer background. It is a solid special hole color 0.
#    - The input also contains eight isolated outer marker pixels, one centered in
#      each outer 3x3 slot. Their colors are a permutation of the eight source
#      blocks' dominant colors.
#    - Example trace:
#      original task: block(0,0) always copies to outer slot(4,4) by symmetry.
#      combo task: if block(0,0) is mostly color 6 and the marker color 6 is at
#      outer slot(0,2), that block must go to slot(0,2) instead. Meanwhile the
#      central 3x3 block of color 0 is not an object to copy; it is a "missing
#      area" marker that gets replaced by the reconstructed center patch.
# b) Complexity calibration:
#    - To avoid ambiguous color lookup, each of the eight source blocks receives a
#      distinct dominant color. The outer markers reuse exactly those eight colors.
#    - The hole marker uses a ninth color (0) that never appears in any source
#      block or outer marker, so the two marker systems are visually separable.
# c) Hidden assumptions broken from the original solver:
#    - `bbox(g != bg)` is no longer usable because outer markers expand the global
#      non-background bbox to the full 15x15 frame. The solver must isolate the
#      largest connected component, which is the 9x9 motif.
#    - The original "center is background" assumption is invalid. The solver must
#      now tolerate a filled center block, recognize it as a uniform special color,
#      and overwrite it with the reconstructed patch.
#    - The original opposite-slot formula is invalid because the outer copy rule is
#      now driven by color equality between source-block major colors and marker
#      pixels.
#    - The original motif-wide majority color is unstable once all eight source
#      blocks use distinct majors. The reconstructed center therefore uses the mode
#      of the eight boundary samples around the hole.
# d) Placement legality:
#    - The motif still sits with a margin of one block, so the full 15x15 expanded
#      region stays inside the NxN canvas.
#    - Outer markers are placed at the centers of the outer slots, which keeps them
#      disconnected from the motif component and prevents overlap with pasted blocks.
#    - The center hole is a full 3x3 block, so it never collides with source data;
#      it simply occupies the block that used to be empty.
# e) Visual observability:
#    - Across examples, the outer marker permutation changes, so the user can infer
#      that outer placement depends on color matching rather than point symmetry.
#    - The central special-color square is visually different from both background
#      and outer markers: it is a full 3x3 region, not a single pixel.
# f) Visual cleanliness:
#    - There are only two annotation forms: a solid 3x3 center hole and eight
#      centered single-pixel outer tags. They occupy disjoint regions and remain
#      easy to parse by eye.
# g) Color-inference guardrails:
#    - No numeric ordering on colors is used anywhere. Matching depends only on
#      visible equality. The special hole color is distinguished by location and
#      shape, not by the fact that its id is 0.
# h) Anti-degeneration metric:
#    - The generator enforces at least 4 disagreements with the original
#      opposite-slot mapping, measured as the number of outer slots whose marker
#      color does not match the source block that would have gone there originally.
#    - It also always uses a full 3x3 special-color center hole, so the D12 change
#      is present in every sample instead of collapsing back to a background hole.
# i) New solver assumptions guaranteed by the generator:
#    - The largest connected non-background component is the 9x9 motif.
#    - The center block of that motif is uniformly colored with the special hole
#      color, distinct from background and from every source-block major color.
#    - Each outer slot contains exactly one centered marker pixel whose color
#      matches exactly one source block's major color.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
BG = 8
HOLE_COLOR = 0
SOURCE_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_RETRIES = 300
MIN_ANTI_ORIGINAL_MISMATCH = 4

SOURCE_POSITIONS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)
OUTER_SLOTS = (
    (0, 0), (0, 2), (0, 4),
    (2, 0),         (2, 4),
    (4, 0), (4, 2), (4, 4),
)
ORTHOGONAL_POSITIONS = {(0, 1), (1, 0), (1, 2), (2, 1)}
INNER_SAMPLE_COORD = {
    (0, 0): (2, 2),
    (0, 1): (2, 1),
    (0, 2): (2, 0),
    (1, 0): (1, 2),
    (1, 2): (1, 0),
    (2, 0): (0, 2),
    (2, 1): (0, 1),
    (2, 2): (0, 0),
}
CENTER_OFFSETS = np.array([-1, 1, 3], dtype=int)


def _major_block_color(block: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(block[block != int(bg)], return_counts=True)
    assert vals.size, "block must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _largest_component_bbox(g: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    mask = g != int(bg)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best_bbox = None
    best_size = -1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                size += 1
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if size > best_size:
                best_size = size
                best_bbox = (r0, c0, r1, c1)
    return best_bbox


def _center_hole_color(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> int:
    center = crop_bbox(
        g,
        (int(r0) + block, int(c0) + block, int(r0) + 2 * block - 1, int(c0) + 2 * block - 1),
    )
    vals = np.unique(center)
    assert vals.size == 1, "center hole marker must be uniform"
    hole = int(vals[0])
    assert hole != int(bg), "center hole marker must differ from background"
    return hole


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    assert block == 3, "this variant uses 3x3 blocks"
    hole_r = int(r0) + block
    hole_c = int(c0) + block
    rows = hole_r + CENTER_OFFSETS
    cols = hole_c + CENTER_OFFSETS
    patch = g[np.ix_(rows, cols)].copy()
    ring_vals = [int(patch[r, c]) for r in range(block) for c in range(block) if (r, c) != (1, 1)]
    patch[1, 1] = Counter(ring_vals).most_common(1)[0][0]
    return patch


def _slot_marker_color(
    g: np.ndarray,
    expanded_r0: int,
    expanded_c0: int,
    slot_br: int,
    slot_bc: int,
    block: int,
    bg: int,
    hole_color: int,
) -> int:
    sr = expanded_r0 + slot_br * block
    sc = expanded_c0 + slot_bc * block
    marker = int(g[sr + block // 2, sc + block // 2])
    assert marker != int(bg), f"outer slot {(slot_br, slot_bc)} is missing its marker"
    assert marker != int(hole_color), "outer marker color must differ from center hole color"
    return marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a tagged 3x3-block motif into a 5x5 block layout with two marker roles.

    Input structure:
    - Teal background contains one 9x9 motif made of eight 3x3 source blocks.
    - The middle 3x3 block is filled with one special color that marks the hole;
      it is not background and it is not one of the outer marker colors.
    - Each of the eight outer 3x3 slots contains a single centered marker pixel,
      and the pixels around the central hole still encode the missing center patch.

    Rule:
    - Keep the original 9x9 motif in place, except replace the special-color hole
      with the reconstructed center block.
    - Reconstruct that center block from the eight boundary pixels facing the hole;
      the center cell takes the modal boundary color.
    - For each outer slot, read its centered marker color and copy in the source
      block whose dominant color matches that marker.

    Key insight: two non-background markers coexist but mean different things. The
    solid central square marks "missing area to rebuild", while the isolated outer
    pixels act as color labels that choose which source block belongs in each slot.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    motif_bbox = _largest_component_bbox(g, bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0, "motif must be a square 3x3 block grid"
    block = h // 3
    assert block == 3, "this variant expects a 9x9 motif"

    hole_color = _center_hole_color(g, int(r0), int(c0), block, bg)
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block
    out = g.copy()

    blocks_by_color: dict[int, np.ndarray] = {}
    for br, bc in SOURCE_POSITIONS:
        src = crop_bbox(
            g,
            (
                int(r0) + br * block,
                int(c0) + bc * block,
                int(r0) + (br + 1) * block - 1,
                int(c0) + (bc + 1) * block - 1,
            ),
        )
        major = _major_block_color(src, bg)
        assert major != hole_color, "source-block major color must differ from hole marker"
        if major in blocks_by_color:
            assert np.array_equal(blocks_by_color[major], src), (
                "reused marker colors require identical source blocks"
            )
        else:
            blocks_by_color[major] = src

    for slot_br, slot_bc in OUTER_SLOTS:
        marker = _slot_marker_color(
            g,
            expanded_r0,
            expanded_c0,
            slot_br,
            slot_bc,
            block,
            bg,
            hole_color,
        )
        assert marker in blocks_by_color, f"no source block has major color {marker}"
        paste(out, blocks_by_color[marker], expanded_r0 + slot_br * block, expanded_c0 + slot_bc * block)

    paste(out, _center_patch(g, int(r0), int(c0), block), int(r0) + block, int(c0) + block)
    return out


def _random_block(main_color: int, ring_color: int, sample_coord: tuple[int, int]) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), main_color, dtype=int)
    if ring_color != main_color:
        block[sample_coord] = ring_color
    remaining = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if (r, c) != sample_coord]
    extra_coords = random.sample(remaining, 2)
    accent_pool = [color for color in SOURCE_COLORS if color != main_color]
    extra_colors = random.sample(accent_pool, 2)
    for (rr, cc), color in zip(extra_coords, extra_colors):
        block[rr, cc] = color
    return block


def _assemble_input_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), HOLE_COLOR, dtype=int)
    for pos, block in blocks.items():
        br, bc = pos
        paste(motif, block, br * BLOCK, bc * BLOCK)
    return motif


def _old_mapping_mismatch_count(
    source_colors: dict[tuple[int, int], int],
    marker_colors: dict[tuple[int, int], int],
) -> int:
    mismatches = 0
    for br, bc in SOURCE_POSITIONS:
        old_slot = (4 - 2 * br, 4 - 2 * bc)
        if marker_colors[old_slot] != source_colors[(br, bc)]:
            mismatches += 1
    return mismatches


def _expanded_output(
    motif: np.ndarray,
    center: np.ndarray,
    blocks_by_color: dict[int, np.ndarray],
    marker_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for slot_br, slot_bc in OUTER_SLOTS:
        paste(
            expanded,
            blocks_by_color[marker_colors[(slot_br, slot_bc)]],
            slot_br * BLOCK,
            slot_bc * BLOCK,
        )
    return expanded


def _gen_attempt() -> tuple[np.ndarray, int] | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    source_color_list = random.sample(SOURCE_COLORS, len(SOURCE_POSITIONS))
    source_colors = {pos: color for pos, color in zip(SOURCE_POSITIONS, source_color_list)}

    marker_perm = random.sample(source_color_list, len(source_color_list))
    marker_colors = {slot: color for slot, color in zip(OUTER_SLOTS, marker_perm)}
    mismatch_count = _old_mapping_mismatch_count(source_colors, marker_colors)
    if mismatch_count < MIN_ANTI_ORIGINAL_MISMATCH:
        return None

    center_color = random.choice(source_color_list)
    ring_colors: dict[tuple[int, int], int] = {}
    for pos in SOURCE_POSITIONS:
        if pos in ORTHOGONAL_POSITIONS:
            ring_colors[pos] = center_color
        else:
            candidates = [color for color in SOURCE_COLORS if color != center_color]
            ring_colors[pos] = random.choice(candidates)

    blocks = {
        pos: _random_block(source_colors[pos], ring_colors[pos], INNER_SAMPLE_COORD[pos])
        for pos in SOURCE_POSITIONS
    }
    motif = _assemble_input_motif(blocks)
    center = _center_patch(motif, 0, 0, BLOCK)
    if int(center[1, 1]) != center_color:
        return None

    blocks_by_color = {source_colors[pos]: blocks[pos] for pos in SOURCE_POSITIONS}
    expanded = _expanded_output(motif, center, blocks_by_color, marker_colors)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    for slot_br, slot_bc in OUTER_SLOTS:
        sr = r0 - BLOCK + slot_br * BLOCK
        sc = c0 - BLOCK + slot_bc * BLOCK
        inp[sr + 1, sc + 1] = marker_colors[(slot_br, slot_bc)]

    if np.count_nonzero(inp[r0 + BLOCK:r0 + 2 * BLOCK, c0 + BLOCK:c0 + 2 * BLOCK] == HOLE_COLOR) != 9:
        return None
    if np.array_equal(main(inp), out):
        return inp, mismatch_count
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _mismatch_count = result
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo variant input")
