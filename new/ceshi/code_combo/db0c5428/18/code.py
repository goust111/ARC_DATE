# Combo variant of db0c5428 (index 18)
# combo_spec: D4/0+D6/2
# Name: 无对称约束+180度旋转复制

from __future__ import annotations

# Design analysis
# ---------------
# a) Concrete behavior and trace
#    - The visible scene stays the same type: one 9x9 motif made of eight visible
#      3x3 blocks around a missing 3x3 center, on bg=8.
#    - Original task: the eight visible blocks are generated from two symmetric
#      seeds, and each outer copy is pasted unchanged into the point-symmetric
#      slot of the surrounding 5x5 block layout.
#    - Fused task: the center is still reconstructed from the eight inward-facing
#      pixels, but the eight visible blocks are now independent 3x3 patterns with
#      no enforced LR/UD/diagonal relationship, and every outer copy is rotated
#      180 degrees before pasting.
#    - Concrete trace for block b00 at motif position (0,0):
#      original output writes b00 unchanged into the outer slot (4,4);
#      fused output writes rot180(b00) there instead. Because b22 is no longer
#      constrained to equal rot180(b00), the learner cannot rely on any global
#      symmetry shortcut and must infer the placement+rotation rule directly.
# b) Complexity calibration
#    - Removing symmetry can make the scene noisy, so the generator keeps the
#      original block size, grid size, hole structure, and center-rebuild rule.
#    - To keep the D4 change visible, every generated 3x3 block is forced to
#      differ from its own 180-degree rotation. This avoids fake examples where
#      rotating the copied block would look identical.
#    - To keep the D6 change visible, the generator rejects motifs that satisfy
#      any of the seed task's old block-to-block symmetry equations.
# c) Implicit assumption audit against main()
#    - `bbox(g != bg)` assumes a single foreground motif. Generator guarantee:
#      only one 9x9 motif is placed and there are no stray foreground pixels.
#    - `h == w and h % 3 == 0` assumes a square 3x3 block lattice. Generator
#      guarantee: motif size remains exactly 9x9 with BLOCK=3.
#    - `_center_patch()` assumes the middle block is empty and the eight sampled
#      inner-boundary pixels exist at the same offsets as the seed task. The
#      combo changes neither hole placement nor the sampling geometry.
#    - The outer-copy loop now depends on orientation because it uses rot180(src).
#      Generator guarantee: each src block is rot180-asymmetric, so this branch
#      has an observable effect on accepted samples.
# d) Placement legality
#    - The motif is still padded by one full block on every side, so the 15x15
#      expanded output always fits inside the 16-19 square canvas.
#    - There is only one object, so no overlaps occur between independent blocks;
#      all pastes are to fixed non-overlapping 3x3 slots.
# e) Visual observability
#    - A solver can compare the central 3x3 ring with the new outer ring and see
#      that each external block comes from the point-opposite source but flipped
#      both horizontally and vertically.
#    - Because the inner eight blocks are all different and do not mirror each
#      other, multiple examples expose that block identity comes from position,
#      not from symmetric pairing.
# f) Visual cleanliness
#    - The canvas still contains one compact motif on uniform background.
#    - Independence is expressed only through the cell patterns inside existing
#      3x3 blocks, so there are no extra markers, overlays, or collisions.
# g) Change inferability
#    - No hidden use of numeric color ids is introduced. Colors are only visible
#      cell values. The two key clues are geometric: block position and rot180.
# 9) New solver assumptions and generator support
#    - `main()` assumes the global majority foreground color is well-defined,
#      because that value becomes the center pixel. Generator support: every
#      visible block contains 5-7 majority cells, so the majority color strictly
#      dominates the 72 foreground cells of the motif.
#    - `main()` assumes the source blocks read from the input are untouched by
#      earlier writes. Generator support: input contains only the 9x9 motif, and
#      solver reads from `g` while writing into a separate `out`.
#    - For the fused behavior to be non-degenerate, generator support also
#      enforces three metrics: all 8 blocks are distinct, none equals its own
#      rot180, and the legacy symmetry match count is zero.

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "solvers" / "arc_utils.py"
        if cand.exists():
            sys.path.append(str(cand.parent))
            return
    raise ImportError("could not locate solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import bbox, crop_bbox, get_bg, paste


def _rot180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 2)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the observed 3x3 block ring into a larger layout with rotated outer copies.

    Input structure:
    - A teal background contains one 9x9 motif arranged as a 3x3 grid of 3x3 blocks.
    - The middle 3x3 block is missing and stays teal in the input.
    - The eight visible surrounding blocks need not obey any global symmetry; they are
      read only by their grid positions.

    Rule:
    - Keep the original 9x9 motif in place.
    - Rebuild the missing center block from the eight inward-facing cells around the
      hole; its center pixel is the motif's majority foreground color.
    - Copy each visible 3x3 block to the opposite outer slot of the implied 5x5 block
      layout, rotating that copied block by 180 degrees before pasting.

    Key insight:
    The input already shows the middle ring of a larger block pattern. The correct
    output comes from position-based opposite-slot mapping plus a uniform rot180
    transform on every outward duplicate, not from any symmetry among the source blocks.
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
            paste(out, _rot180(src), dst_r, dst_c)

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
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

# -- CFG: generation --
MAX_RETRIES = 400
BLOCK_MAJOR_MIN, BLOCK_MAJOR_MAX = 5, 7
MIN_ROTATION_VISIBILITY = 24
MIN_CENTER_NON_MAJOR = 3
MIN_CENTER_DISTINCT = 2

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
ROT180_PAIRS = (
    ((0, 0), (2, 2)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 0)),
    ((1, 0), (1, 2)),
)
_CENTER_OFFSETS = np.array([-1, 1, 3])


def _sample_block(majority: int, minor_colors: list[int], required_minor: int | None) -> np.ndarray:
    """Construct one 3x3 block with a strict majority color and visible rot180 asymmetry."""
    block = np.full((BLOCK, BLOCK), majority, dtype=int)
    target_major = random.randint(BLOCK_MAJOR_MIN, BLOCK_MAJOR_MAX)
    non_major_total = BLOCK * BLOCK - target_major

    pair = random.choice(ROT180_PAIRS)
    minor_pos, protected_major_pos = random.sample(pair, 2)
    available = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if (r, c) not in {minor_pos, protected_major_pos}]
    non_major_positions = [minor_pos] + random.sample(available, non_major_total - 1)

    forced_color = required_minor if required_minor is not None else random.choice(minor_colors)
    block[minor_pos] = int(forced_color)
    for pos in non_major_positions[1:]:
        block[pos] = int(random.choice(minor_colors))

    assert not np.array_equal(block, _rot180(block))
    return block


def _assemble_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = int(majority)
    return patch


def _expanded_output(motif: np.ndarray, center: np.ndarray, *, rotate_outer: bool) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            patch = _rot180(src) if rotate_outer else src
            paste(expanded, patch, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _rotation_visibility_score(motif: np.ndarray, center: np.ndarray) -> int:
    legacy = _expanded_output(motif, center, rotate_outer=False)
    variant = _expanded_output(motif, center, rotate_outer=True)
    return int(np.count_nonzero(legacy != variant))


def _legacy_symmetry_match_count(blocks: dict[tuple[int, int], np.ndarray]) -> int:
    checks = [
        np.array_equal(blocks[(0, 2)], np.fliplr(blocks[(0, 0)])),
        np.array_equal(blocks[(1, 0)], blocks[(0, 1)].T),
        np.array_equal(blocks[(1, 2)], np.rot90(blocks[(0, 1)], -1)),
        np.array_equal(blocks[(2, 0)], np.flipud(blocks[(0, 0)])),
        np.array_equal(blocks[(2, 1)], np.flipud(blocks[(0, 1)])),
        np.array_equal(blocks[(2, 2)], _rot180(blocks[(0, 0)])),
    ]
    return int(sum(checks))


def _distinct_block_count(blocks: dict[tuple[int, int], np.ndarray]) -> int:
    return len({tuple(map(int, blk.flatten())) for blk in blocks.values()})


def _center_boundary_metrics(center: np.ndarray, majority: int) -> tuple[int, int]:
    boundary = center.copy()
    boundary[1, 1] = int(majority)
    non_major = int(np.count_nonzero(boundary != int(majority)))
    distinct = len(set(map(int, center.flatten())))
    return non_major, distinct


def _build_independent_blocks(majority: int, minor_colors: list[int]) -> dict[tuple[int, int], np.ndarray] | None:
    required_minors = list(minor_colors)
    random.shuffle(required_minors)
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for idx, pos in enumerate(BLOCK_POSITIONS):
        forced_minor = required_minors[idx] if idx < len(required_minors) else None
        blocks[pos] = _sample_block(majority, minor_colors, forced_minor)

    if _distinct_block_count(blocks) != len(BLOCK_POSITIONS):
        return None
    if _legacy_symmetry_match_count(blocks) != 0:
        return None
    return blocks


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify that a generated sample really exhibits the combo behavior."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None
    r0, c0, r1, c1 = motif_bbox
    block = (int(r1 - r0 + 1)) // 3
    assert block == BLOCK

    blocks = {
        (br, bc): crop_bbox(
            g,
            (
                int(r0) + br * block,
                int(c0) + bc * block,
                int(r0) + (br + 1) * block - 1,
                int(c0) + (bc + 1) * block - 1,
            ),
        )
        for br, bc in BLOCK_POSITIONS
    }
    majority = _major_foreground_color(g, bg)
    motif = crop_bbox(g, motif_bbox)
    center = _gen_center_patch(motif, majority)
    center_non_major, center_distinct = _center_boundary_metrics(center, majority)
    return {
        "rotation_visibility": _rotation_visibility_score(motif, center),
        "legacy_symmetry_matches": _legacy_symmetry_match_count(blocks),
        "distinct_blocks": _distinct_block_count(blocks),
        "center_non_major": center_non_major,
        "center_distinct": center_distinct,
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    blocks = _build_independent_blocks(majority, minor_colors)
    if blocks is None:
        return None

    motif = _assemble_motif(blocks)
    center = _gen_center_patch(motif, majority)
    center_non_major, center_distinct = _center_boundary_metrics(center, majority)
    if center_non_major < MIN_CENTER_NON_MAJOR or center_distinct < MIN_CENTER_DISTINCT:
        return None

    rotation_visibility = _rotation_visibility_score(motif, center)
    if rotation_visibility < MIN_ROTATION_VISIBILITY:
        return None

    expanded = _expanded_output(motif, center, rotate_outer=True)

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
    raise AssertionError("failed to generate a valid db0c5428 D4/0+D6/2 input")
