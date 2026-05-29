# Variant of db0c5428
# Dimension: D1 — 块到外部槽位的映射规则 (match_cond)
# Change: 每个非中心3x3块被复制到以中心为对称点的对侧外部槽位，即block(r,c)映射到5x5网格的(4-2r, 4-2c)... → 颜色匹配决定映射目标

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior:
#    - The visible motif is still a 9x9 ring of eight 3x3 blocks around a missing
#      3x3 center. The output still expands to an implied 5x5 block lattice and
#      still reconstructs the center patch from the pixels facing the hole.
#    - The changed part is the outer-copy rule. Input now contains one colored
#      marker pixel at the center of each of the eight outer 3x3 slots. A source
#      block no longer goes to its point-symmetric opposite slot. Instead, the
#      solver reads the block's major color and copies that block into whichever
#      outer slot carries the same color marker.
#    - Small trace:
#      original: block(0,0) always copied to slot(4,4), block(0,1) always to
#      slot(4,2), purely by coordinates.
#      variant: if block(0,0) is mostly color 6 and the marker with color 6 sits
#      at outer slot(0,2), the block goes to slot(0,2) regardless of geometry.
#      The reasoning path changes from spatial reflection to color matching.
# b) Complexity calibration:
#    - To avoid ambiguous matching, every source block gets a distinct major color.
#      The eight outer markers are a permutation of those same eight colors.
#      This keeps the rule easy to perceive while making the target layout depend
#      entirely on the marker permutation.
# c) Hidden assumptions in the original main():
#    - `bbox(g != bg)` no longer works because marker pixels enlarge the overall
#      bbox to the full 15x15 implied region. The new solver must isolate the
#      largest connected non-bg component, which is the 9x9 motif.
#    - The original outer-slot destination formula assumed a fixed spatial map.
#      That assumption is removed; slot choice is now read from marker colors.
#    - The original center pixel used the motif-wide majority color. With eight
#      distinct block-major colors that would be unstable, so the new center uses
#      the modal color among the eight sampled boundary pixels around the hole.
# d) Placement legality:
#    - The motif still sits with a 3-cell margin so the full 15x15 expanded region
#      remains in-bounds. Outer markers are placed at the exact centers of those
#      eight outer slots, never touching the motif component.
# e) Visual observability:
#    - The input visibly shows eight dominant-color blocks plus eight isolated
#      outer markers. Across examples, the marker permutation changes, so the user
#      can see that outer placement follows color equality rather than reflection.
# f) Visual cleanliness:
#    - Each outer slot starts with exactly one centered marker pixel. There is no
#      overlap among markers, motif, or reconstructed center.
# g) Color-inference guardrails:
#    - No ordering by numeric color id is used. Only visible equality matters:
#      a marker with the same color as a block's dominant color selects that block.
# h) Anti-degeneration metric:
#    - The generator measures how many outer slots disagree with the original
#      opposite-slot mapping on the same input. It enforces at least 4 mismatches,
#      so the generated samples do not collapse back to the original task.
# i) New solver assumptions to satisfy:
#    - The largest non-bg connected component is the 9x9 motif.
#    - If two blocks share a major color, they must be identical because the solver
#      uses color -> block lookup. The generator uses unique major colors, which is
#      stronger than required and guarantees this property.
#    - Each outer slot center contains a non-bg marker whose color matches some
#      source block's major color.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
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
) -> int:
    sr = expanded_r0 + slot_br * block
    sc = expanded_c0 + slot_bc * block
    marker = int(g[sr + block // 2, sc + block // 2])
    assert marker != int(bg), f"outer slot {(slot_br, slot_bc)} is missing its marker"
    return marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 block motif using color-tagged outer slots.

    Input structure:
    - Teal background contains one 9x9 motif made of eight 3x3 blocks around an
      empty 3x3 center.
    - The pixels facing the hole still encode the border of the missing center.
    - Each of the eight outer 3x3 slots contains a single centered marker pixel
      whose color names which source block belongs there.

    Rule:
    - Keep the original 9x9 motif in place.
    - Reconstruct the center block from the eight sampled boundary pixels; the
      center cell takes the most frequent of those eight colors.
    - For each outer slot, read its marker color and copy in the surrounding 3x3
      source block whose major color matches that marker.

    Key insight: the outer layout is no longer determined by geometric symmetry.
    The marker colors create a color-to-block lookup, so destination slots must be
    solved by matching dominant block colors to those markers.
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

    out = g.copy()
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block

    blocks_by_color: dict[int, np.ndarray] = {}
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
            major = _major_block_color(src, bg)
            if major in blocks_by_color:
                assert np.array_equal(blocks_by_color[major], src), (
                    "reused marker colors require identical source blocks"
                )
            else:
                blocks_by_color[major] = src

    for slot_br, slot_bc in OUTER_SLOTS:
        marker = _slot_marker_color(g, expanded_r0, expanded_c0, slot_br, slot_bc, block, bg)
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
    accent_pool = [color for color in FOREGROUND_COLORS if color != main_color]
    extra_colors = random.sample(accent_pool, 2)
    for (rr, cc), color in zip(extra_coords, extra_colors):
        block[rr, cc] = color
    return block


def _assemble_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for pos, block in blocks.items():
        br, bc = pos
        paste(motif, block, br * BLOCK, bc * BLOCK)
    return motif


def _marker_mismatch_count(
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
        paste(expanded, blocks_by_color[marker_colors[(slot_br, slot_bc)]], slot_br * BLOCK, slot_bc * BLOCK)
    return expanded


def _gen_attempt() -> tuple[np.ndarray, int] | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    source_color_list = random.sample(FOREGROUND_COLORS, len(SOURCE_POSITIONS))
    source_colors = {pos: color for pos, color in zip(SOURCE_POSITIONS, source_color_list)}

    marker_perm = random.sample(source_color_list, len(source_color_list))
    marker_colors = {slot: color for slot, color in zip(OUTER_SLOTS, marker_perm)}
    mismatch_count = _marker_mismatch_count(source_colors, marker_colors)
    if mismatch_count < MIN_ANTI_ORIGINAL_MISMATCH:
        return None

    center_color = random.choice(source_color_list)
    ring_colors: dict[tuple[int, int], int] = {}
    for pos in SOURCE_POSITIONS:
        if pos in ORTHOGONAL_POSITIONS:
            ring_colors[pos] = center_color
        else:
            candidates = [color for color in FOREGROUND_COLORS if color != center_color]
            ring_colors[pos] = random.choice(candidates)

    blocks = {
        pos: _random_block(source_colors[pos], ring_colors[pos], INNER_SAMPLE_COORD[pos])
        for pos in SOURCE_POSITIONS
    }
    motif = _assemble_motif(blocks)
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

    if np.array_equal(main(inp), out):
        return inp, mismatch_count
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _mismatch_count = result
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D1 variant input")
