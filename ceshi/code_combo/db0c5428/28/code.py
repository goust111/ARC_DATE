# Combo variant of db0c5428 (index 28)
# combo_spec: D4/2+D10/1
# Name: 颜色取反复制+独特颜色块

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Base task: a 9x9 motif contains eight visible 3x3 blocks around a missing
#      3x3 center. The output keeps that motif, reconstructs the center from the
#      eight pixels facing the hole, and copies each visible block to the opposite
#      outer slot of a 5x5 block scaffold.
#    - D10/1 contribution: each visible block now uses its own unordered two-color
#      pair. Example: top-left may use {2,7}, top-middle {2,5}, right-middle {1,9}.
#      These local palettes act as block identities.
#    - D4/2 contribution in this fused setting: outer copies are not recolored by a
#      single motif-wide map. Instead, every copied block swaps its own two colors.
#      So a source block using {2,7} keeps the same 3x3 geometry outside, but every
#      2 becomes 7 and every 7 becomes 2, while a different block using {1,9} swaps
#      1<->9 independently.
#    - Relative to the original logic, the changed step is exactly the outer-copy
#      paste: geometry is unchanged, but the pasted pixels are the local color inverse
#      of the source block. Center reconstruction stays unchanged.
# b) Complexity calibration
#    - A global many-color remapping would be too hard to infer from 2-4 ARC pairs.
#      This design keeps each block strictly two-colored, so the inversion is a
#      simple local swap rather than an unconstrained permutation.
#    - To make block identities readable, the generator uses exactly five foreground
#      colors, which yields ten unordered pairs. Eight distinct pairs are sampled so
#      every visible block gets a unique signature with repeated support in output.
#    - Both colors are forced to appear inside every visible block, preventing the
#      swap from degenerating into a no-op.
# c) Hidden-assumption audit of main()
#    - `_major_foreground_color(...)` still assumes one color is globally dominant;
#      the generator guarantees a designated majority color remains the motif-wide
#      winner and uses it for the center pixel.
#    - `_center_patch(...)` still assumes 3x3 blocks and the original ring sampling
#      around the hole. The fused variant does not disturb that geometry.
#    - The original outer-copy loop assumed verbatim copying. That assumption is
#      replaced with a stricter one: each visible block must expose exactly two colors
#      so the solver can infer a local swap map from the block itself.
#    - The solver does not need block palettes to be unique, but human inference does.
#      The generator therefore enforces eight distinct unordered pairs.
# d) Placement legality
#    - Grid size stays 16..19, motif size stays 9, expanded size stays 15, and motif
#      origin stays padded by one block. Therefore the 5x5 scaffold fits exactly as
#      in the base task, with no overlap and no out-of-bounds pastes.
# e) Visual observability
#    - A human sees the same source block shape reappear outside with the same local
#      geometry but with its two colors exchanged. Because each source block has a
#      different palette, the viewer can match each source-target pair and observe
#      that the color inversion is local to that block, not global to the whole motif.
# f) Visual cleanliness
#    - No extra markers, overlays, or mixed-object regions are introduced. The scene
#      remains one clean 5x5 scaffold with a repaired middle and eight recolored
#      outer copies.
# g) Change inferability
#    - The only color rule is pairwise exchange inside each source block. There is no
#      dependence on numeric color ids or hidden ordering. The repeated block lattice
#      gives enough structure to infer this from a few examples.
# h) New solver assumptions review
#    - Every visible 3x3 block must be fully colored and must contain exactly two
#      distinct non-background colors.
#    - The generator guarantees those assumptions via fixed two-color masks, unique
#      pair assignment, and a verification check that the designated majority really
#      is the motif-wide majority color.


BLOCK_POSITIONS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _block_colors(block: np.ndarray) -> tuple[int, int]:
    vals = sorted(int(v) for v in np.unique(block))
    assert len(vals) == 2, "each visible block must use exactly two colors"
    return vals[0], vals[1]


def _swap_block_colors(block: np.ndarray) -> np.ndarray:
    a, b = _block_colors(block)
    out = block.copy()
    out[block == a] = b
    out[block == b] = a
    return out


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    hole_r = int(r0) + block
    hole_c = int(c0) + block
    rows = hole_r + _CENTER_OFFSETS
    cols = hole_c + _CENTER_OFFSETS
    patch = g[np.ix_(rows, cols)].copy()
    patch[1, 1] = _major_foreground_color(g, bg)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 scaffold with per-block color-inverted copies.

    Input structure:
    - A teal background contains one 9x9 motif made from eight visible 3x3 blocks
      around a missing 3x3 center hole.
    - Each visible block is fully filled with exactly two foreground colors, and
      every block uses a different two-color combination from the others.
    - The eight pixels facing the hole still encode the border of the missing
      center block.

    Rule:
    - Keep the original 9x9 motif in place and reconstruct its missing center.
    - Copy every non-center 3x3 block to the point-symmetric outer slot of the
      implied 5x5 block layout without rotating the copied geometry.
    - Before pasting an outer copy, swap the two colors used inside that source
      block, independently for each block.

    Key insight: The spatial expansion is unchanged, but the color rule is now
    local rather than global. Each visible block carries its own two-color identity,
    and its opposite outer copy preserves shape while inverting just that pair.
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
            paste(out, _swap_block_colors(src), dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (combo) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_PAD = 3

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
PALETTE_SIZE = 5

# -- CFG: generation --
MAX_RETRIES = 300

# These masks keep the original corner/edge geometry readable while ensuring both
# colors appear in every block. The preferred color is assigned to the larger side.
CORNER_MASKS = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int),
)
EDGE_MASKS = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=int),
)


def _colorize(mask: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
    a, b = pair
    return np.where(mask == 1, a, b).astype(int)


def _order_pair_for_mask(mask: np.ndarray, pair: tuple[int, int], preferred: int | None) -> tuple[int, int]:
    if preferred is None or preferred not in pair:
        return pair if random.random() < 0.5 else (pair[1], pair[0])
    other = pair[0] if pair[1] == preferred else pair[1]
    ones = int(np.sum(mask == 1))
    zeros = int(mask.size - ones)
    if ones >= zeros:
        return preferred, other
    return other, preferred


def _sample_unique_pairs(majority: int, accents: list[int]) -> list[tuple[int, int]]:
    pairs = [(majority, accent) for accent in accents]
    pairs.extend(random.sample(list(itertools.combinations(accents, 2)), 4))
    random.shuffle(pairs)
    return pairs


def _build_masks() -> dict[tuple[int, int], np.ndarray]:
    corner = random.choice(CORNER_MASKS)
    edge = random.choice(EDGE_MASKS)
    return {
        (0, 0): corner,
        (0, 1): edge,
        (0, 2): np.fliplr(corner),
        (1, 0): edge.T,
        (1, 2): np.rot90(edge, -1),
        (2, 0): np.flipud(corner),
        (2, 1): np.flipud(edge),
        (2, 2): np.rot90(corner, 2),
    }


def _assemble_motif(majority: int, accents: list[int]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    masks = _build_masks()
    for pos, pair in zip(BLOCK_POSITIONS, _sample_unique_pairs(majority, accents)):
        mask = masks[pos]
        ordered_pair = _order_pair_for_mask(mask, pair, majority if majority in pair else None)
        paste(motif, _colorize(mask, ordered_pair), pos[0] * BLOCK, pos[1] * BLOCK)
    return motif


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = int(majority)
    return patch


def _expanded_output(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, _swap_block_colors(src), (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _block_pair(block: np.ndarray) -> tuple[int, int]:
    return _block_colors(block)


def _pair_signature_count(motif: np.ndarray) -> int:
    pairs = set()
    for br, bc in BLOCK_POSITIONS:
        block = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        pairs.add(_block_pair(block))
    return len(pairs)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    colors = random.sample(FOREGROUND_COLORS, PALETTE_SIZE)
    majority = colors[0]
    accents = colors[1:]

    motif = _assemble_motif(majority, accents)
    if _pair_signature_count(motif) != 8:
        return None
    if _major_foreground_color(motif, BG) != majority:
        return None

    center = _gen_center_patch(motif, majority)
    expanded = _expanded_output(motif, center)

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
    raise AssertionError("failed to generate a valid db0c5428 combo input")
