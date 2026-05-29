# Combo variant of db0c5428 (index 4)
# combo_spec: D1/3+D10/1
# Name: 颜色匹配映射+独特颜色块

from __future__ import annotations

from collections import Counter
import sys
import random

import numpy as np

try:
    from arc_utils import crop_bbox, get_bg, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/arc2_solver_select")
    from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The base scene is still a 9x9 motif made of eight visible 3x3 blocks around
#      a blank 3x3 hole, and the output still fills the hole plus eight outer slots
#      in the implied 5x5 block lattice.
#    - D1/3 changes the outer-copy rule: outer slots are no longer determined by
#      point symmetry. Each outer slot contains one centered marker pixel, and the
#      marker color chooses the source block whose dominant color matches it.
#    - D10/1 changes the block coloring: every visible block uses exactly two colors
#      and every block has its own unordered two-color signature. The dominant color
#      is the identity that the marker refers to, while the secondary color makes the
#      block visually unique even before it is copied.
#    - Small trace: in the original task, source block (0,0) always goes to slot
#      (4,4). Here, if that block is mostly color 6 with signature {6,3}, and the
#      outer marker colored 6 sits at slot (0,2), then the whole 3x3 block is copied
#      to slot (0,2). Another block may also contain color 6 as an accent, but it
#      cannot win the match because only the dominant color counts.
# b) Complexity calibration
#    - To remove matching ambiguity, the generator gives the eight source blocks
#      eight distinct dominant colors. That keeps the color->block lookup one-to-one.
#    - To make the D10 change visible, each block uses exactly two colors and the
#      generator enforces eight distinct unordered pairs. The scene therefore looks
#      different from both the original task and the pure D1 variant.
#    - Four edge blocks share one repeated "hub" accent color on the pixels facing
#      the hole. This supplies a stable center-pixel rule without reintroducing a
#      motif-wide majority color.
# c) Hidden assumptions broken by the fusion
#    - `bbox(g != bg)` is no longer valid because the isolated marker pixels expand
#      the overall bbox to the full 15x15 scaffold. The solver must isolate the
#      largest connected non-background component, which is the 9x9 motif ring.
#    - The original destination formula `(4-2*br, 4-2*bc)` is no longer correct:
#      destination slots are selected by marker colors, not by geometry.
#    - The original center pixel used the motif-wide majority foreground color. With
#      eight unique dominant block colors, that premise is unstable, so the fused
#      solver uses the modal color among the eight hole-border samples instead.
#    - The new lookup loop assumes each dominant color identifies at most one source
#      block. The generator guarantees this by construction.
# d) Placement legality
#    - Grid size remains 16..19, the motif remains 9x9, and its origin keeps a
#      3-cell margin. That leaves room for the whole 15x15 expanded region.
#    - Each outer slot starts as background except for one centered marker pixel,
#      so markers never touch the motif and never overlap each other.
# e) Visual observability
#    - Across examples, the marker permutation changes, so a human can see that outer
#      placement follows color equality rather than reflection.
#    - Within each example, every block has a distinct two-color signature, so the
#      color system is carrying both "who am I?" and "where do I go?" information.
# f) Visual cleanliness
#    - Markers are single pixels placed at slot centers, and every visible block uses
#      only two colors with a simple 6-vs-3 dominant split. This keeps the lattice
#      legible instead of turning into a noisy multicolor texture.
# g) Color-rule inferability
#    - No numeric ordering of color ids is used anywhere. Matching only depends on
#      visible equality of marker color and dominant block color.
#    - The repeated hub color on the four orthogonal hole-border pixels creates a
#      visible reason for the repaired center pixel, so the center rule is still
#      inferable from a few examples.
# h) Anti-degeneration checks
#    - The generator enforces `pair_signature_count == 8` so the D10 behavior is
#      always present.
#    - It also enforces at least four mismatches against the original point-symmetric
#      outer mapping so the D1 behavior does real work and samples do not collapse
#      back to the source task.
# i) New solver assumptions that the generator must satisfy
#    - The largest connected non-bg component is the motif ring.
#    - Every outer slot center contains exactly one non-bg marker.
#    - Each marker color matches exactly one source block's dominant color.
#    - The eight hole-border colors have a unique mode; the generator achieves that
#      by giving all four edge-facing samples the same hub color.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_RETRIES = 400
MIN_ANTI_ORIGINAL_MISMATCH = 4

SOURCE_POSITIONS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)
CORNER_POSITIONS = ((0, 0), (0, 2), (2, 0), (2, 2))
EDGE_POSITIONS = ((0, 1), (1, 0), (1, 2), (2, 1))
OUTER_SLOTS = (
    (0, 0), (0, 2), (0, 4),
    (2, 0),         (2, 4),
    (4, 0), (4, 2), (4, 4),
)

CENTER_OFFSETS = np.array([-1, 1, 3], dtype=int)

ACCENT_COORDS = {
    (0, 0): ((2, 2), (0, 1), (1, 0)),
    (0, 1): ((2, 1), (0, 0), (0, 2)),
    (0, 2): ((2, 0), (0, 1), (1, 2)),
    (1, 0): ((1, 2), (0, 0), (2, 0)),
    (1, 2): ((1, 0), (0, 2), (2, 2)),
    (2, 0): ((0, 2), (1, 0), (2, 1)),
    (2, 1): ((0, 1), (2, 0), (2, 2)),
    (2, 2): ((0, 0), (1, 2), (2, 1)),
}


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
    assert block == 3, "this combo uses 3x3 blocks"
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
    Expand a color-coded 3x3 block motif into a 5x5 scaffold around its hole.

    Input structure:
    - A teal background contains one 9x9 motif made from eight visible 3x3 blocks
      surrounding an empty 3x3 center.
    - Each visible block uses its own two-color signature, with one color clearly
      dominating that block.
    - Each of the eight outer 3x3 slots contains a single centered marker pixel,
      and the pixels facing the hole still encode the border of the missing center.

    Rule:
    - Keep the original 9x9 motif where it is.
    - Rebuild the center block from the eight sampled pixels around the hole; the
      center cell becomes the most frequent of those eight sampled colors.
    - For every outer slot, read its marker color and copy in the source block whose
      dominant color matches that marker, without rotating the copied block.

    Key insight: the same color system serves two roles at once. Each block's local
    palette makes it visually distinct, but only its dominant color participates in
    the marker-based outer mapping, so the solver must separate block identity from
    block accent before placing the outer copies.
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
    assert block == 3, "this combo expects a 9x9 motif"

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
            assert major not in blocks_by_color, "dominant block colors must be unique"
            blocks_by_color[major] = src

    for slot_br, slot_bc in OUTER_SLOTS:
        marker = _slot_marker_color(g, expanded_r0, expanded_c0, slot_br, slot_bc, block, bg)
        assert marker in blocks_by_color, f"no source block has dominant color {marker}"
        paste(out, blocks_by_color[marker], expanded_r0 + slot_br * block, expanded_c0 + slot_bc * block)

    paste(out, _center_patch(g, int(r0), int(c0), block), int(r0) + block, int(c0) + block)
    return out


def _build_block(major: int, accent: int, pos: tuple[int, int]) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), major, dtype=int)
    for r, c in ACCENT_COORDS[pos]:
        block[r, c] = accent
    return block


def _assemble_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for (br, bc), block in blocks.items():
        paste(motif, block, br * BLOCK, bc * BLOCK)
    return motif


def _expanded_output(
    motif: np.ndarray,
    center: np.ndarray,
    blocks_by_major: dict[int, np.ndarray],
    marker_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for slot_br, slot_bc in OUTER_SLOTS:
        paste(expanded, blocks_by_major[marker_colors[(slot_br, slot_bc)]], slot_br * BLOCK, slot_bc * BLOCK)
    return expanded


def _block_pair(block: np.ndarray) -> tuple[int, int]:
    vals = sorted(int(v) for v in np.unique(block))
    assert len(vals) == 2, "each visible block must use exactly two colors"
    return vals[0], vals[1]


def _pair_signature_count(motif: np.ndarray) -> int:
    pairs = set()
    for br, bc in SOURCE_POSITIONS:
        block = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
        pairs.add(_block_pair(block))
    return len(pairs)


def _marker_mismatch_count(
    source_majors: dict[tuple[int, int], int],
    marker_colors: dict[tuple[int, int], int],
) -> int:
    mismatches = 0
    for br, bc in SOURCE_POSITIONS:
        old_slot = (4 - 2 * br, 4 - 2 * bc)
        if marker_colors[old_slot] != source_majors[(br, bc)]:
            mismatches += 1
    return mismatches


def _choose_accents(source_majors: dict[tuple[int, int], int]) -> tuple[dict[tuple[int, int], int], int]:
    hub_corner = random.choice(CORNER_POSITIONS)
    hub_color = source_majors[hub_corner]
    accents: dict[tuple[int, int], int] = {}

    for pos in EDGE_POSITIONS:
        accents[pos] = hub_color

    other_corners = random.sample([pos for pos in CORNER_POSITIONS if pos != hub_corner], 3)
    accents[hub_corner] = source_majors[other_corners[0]]
    accents[other_corners[0]] = source_majors[other_corners[1]]
    accents[other_corners[1]] = source_majors[other_corners[2]]
    accents[other_corners[2]] = source_majors[other_corners[0]]
    return accents, hub_color


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    major_list = random.sample(FOREGROUND_COLORS, len(SOURCE_POSITIONS))
    source_majors = {pos: color for pos, color in zip(SOURCE_POSITIONS, major_list)}
    accents, hub_color = _choose_accents(source_majors)

    blocks = {
        pos: _build_block(source_majors[pos], accents[pos], pos)
        for pos in SOURCE_POSITIONS
    }
    motif = _assemble_motif(blocks)
    if _pair_signature_count(motif) != len(SOURCE_POSITIONS):
        return None

    marker_perm = random.sample(major_list, len(major_list))
    marker_colors = {slot: color for slot, color in zip(OUTER_SLOTS, marker_perm)}
    mismatch_count = _marker_mismatch_count(source_majors, marker_colors)
    if mismatch_count < MIN_ANTI_ORIGINAL_MISMATCH:
        return None

    center = _center_patch(motif, 0, 0, BLOCK)
    if int(center[1, 1]) != hub_color:
        return None

    blocks_by_major = {source_majors[pos]: blocks[pos] for pos in SOURCE_POSITIONS}
    expanded = _expanded_output(motif, center, blocks_by_major, marker_colors)

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
        return inp, {
            "pair_signature_count": _pair_signature_count(motif),
            "mismatch_count": mismatch_count,
            "center_color": hub_color,
        }
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _metrics = result
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
