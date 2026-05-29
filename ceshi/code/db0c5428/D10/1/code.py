# Variant of db0c5428
# Dimension: D10 — 前景色数量与角色分配 (meta)
# Change: 使用2-4种非背景前景色，不同颜色在块内形成对称图案，无显式角色区分... → 每个块使用独特颜色组合

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D10 variant
# ----------------------------------
# source_task: None
# a) Concrete behavior
#    - Input still contains one 9x9 motif made of eight visible 3x3 blocks around an
#      empty 3x3 center hole, placed on a teal background.
#    - The output still expands that motif to a 5x5 block layout by copying each
#      visible block to its point-symmetric outer slot with no rotation.
#    - The D10 change is carried by color allocation: each of the eight visible
#      blocks uses its own two-color combination, so color pairs become visible
#      position signatures. The corner/edge geometries are unchanged from the
#      original task family, but their fills are recolored per block.
#    - Center reconstruction intentionally stays as close to the original task as
#      possible: the ring pixels around the hole still fill the border of the missing
#      center block, and the middle pixel is still the motif-wide majority foreground
#      color. The D10 change is therefore isolated to how block colors are allocated.
# b) Complexity calibration
#    - Using exactly five foreground colors gives ten unordered two-color pairs; the
#      generator picks eight distinct pairs, one per block. Four of those pairs include
#      a designated majority color, so the original center-pixel rule remains valid.
#    - Each block still uses only two colors and a small symmetric structure, so the
#      scene stays readable rather than becoming a noisy many-color texture.
# c) Implicit-assumption audit of main()
#    - `_major_foreground_color(...)` still matters because the middle cell of the
#      repaired center patch keeps the original majority-color behavior.
#    - `_center_patch(...)` still assumes a 3x3 block size and that the eight sampled
#      pixels around the hole exist. The generator keeps the original 3x3 block layout
#      and hole placement, so that assumption remains safe.
#    - The block-copy loop in `main()` only depends on geometry, not on color roles,
#      so it remains unchanged.
# d) Placement legality
#    - Grid size remains 16..19, motif size remains 9, expanded output remains 15,
#      and motif origin stays at least one block away from the border. Therefore the
#      copied outer blocks never overlap or leave the canvas.
# e) Visual observability
#    - A solver can now match outer copies to source blocks by their distinctive
#      two-color combinations, while the center-repair logic stays familiar.
# f) Visual cleanliness
#    - No new objects are introduced; only block-internal coloring changes. The
#      output remains the same clean 5x5 block scaffold as the original task.
# g) Color-rule inferability
#    - The generator never uses color ids for ordering or arithmetic. It only samples
#      visible color subsets and visible unordered color pairs. The designated majority
#      color is visible because it occupies the dominant mask cells in four blocks.


_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


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
    Expand a 3x3 block motif into a 5x5 block scaffold around its empty center.

    Input structure:
    - A teal background contains one 9x9 motif made from a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is missing, leaving a teal hole.
    - Each of the eight visible blocks has its own two-color signature, and the
      pixels facing the hole reveal the border of the missing center block.

    Rule:
    - Leave the visible 9x9 motif where it is.
    - Copy every non-center 3x3 block to the point-symmetric outer slot of the
      implied 5x5 block layout, without rotating that copied block.
    - Rebuild the missing center block from the eight pixels surrounding the hole;
      the center pixel is the motif's majority foreground color.

    Key insight:
    The geometry and center repair stay close to the original family task, but each
    visible block now advertises its identity through a unique two-color signature.
    The solver must still reconstruct the hole, while also reading position cues from
    repeated local palettes rather than from globally shared block colors.
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

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (D10 variant) ---------------

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

# These masks preserve the original task family's corner/edge symmetry, while
# keeping both colors visible in every block.
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


def _sample_unique_pairs(majority: int, accents: list[int]) -> list[tuple[int, int]]:
    chosen = [(majority, accent) for accent in accents]
    chosen.extend(random.sample(list(itertools.combinations(accents, 2)), 4))
    random.shuffle(chosen)
    return chosen


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
        if majority in pair:
            accent = pair[0] if pair[1] == majority else pair[1]
            ordered_pair = (majority, accent)
        else:
            ordered_pair = pair if random.random() < 0.5 else (pair[1], pair[0])
        paste(motif, _colorize(masks[pos], ordered_pair), pos[0] * BLOCK, pos[1] * BLOCK)
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
            src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _block_pair(block: np.ndarray) -> tuple[int, int]:
    vals = sorted(int(v) for v in np.unique(block))
    assert len(vals) == 2, "each visible block must use exactly two colors"
    return vals[0], vals[1]


def _pair_signature_count(motif: np.ndarray) -> int:
    pairs = set()
    for br, bc in BLOCK_POSITIONS:
        block = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
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
    raise AssertionError("failed to generate a valid db0c5428 D10 variant input")
