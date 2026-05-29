# Combo variant of db0c5428 (index 6)
# combo_spec: D2/2+D8/0
# Name: 可变块大小+块大小可变但保持正方形

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concrete_spec:
#   - Input still contains one square foreground motif on a uniform background.
#   - The motif is still a touching 3x3 array of square blocks with the center
#     block removed, but block side length now varies across examples:
#       * block=2 -> motif is 6x6, expanded output region is 10x10
#       * block=3 -> motif is 9x9, expanded output region is 15x15
#       * block=4 -> motif is 12x12, expanded output region is 20x20
#   - This fuses D2/2 and D8/0 by making block size inferable from two aligned
#     clues: the motif footprint itself and the compatible canvas-size range.
# behavior_trace:
#   - Original task hard-coded block=3 and rebuilt the center by sampling eight
#     pixels around the 3x3 hole plus one majority-color center pixel.
#   - In the combo variant, a 4x4 example should behave as follows:
#       1. infer block=4 from a 12x12 foreground bbox,
#       2. copy each visible 4x4 block to the point-symmetric outer slot of the
#          implied 5x5 block lattice,
#       3. rebuild the missing 4x4 center so that its top row comes from the
#          inward row of the top edge block, its bottom row from the bottom edge
#          block, its left/right columns from the side blocks, and its four
#          corners from the inward corners of the diagonal blocks,
#       4. fill any remaining interior cells with the majority foreground color.
#   - For block=3 this generic rule collapses exactly to the original
#     8-boundary-pixels-plus-center reconstruction; for block=2 it yields a full
#     2x2 border-only patch, making the size change visually obvious.
# complexity_calibration:
#   - Block size only has three discrete values, so the hidden parameter is
#     inferable rather than unconstrained.
#   - Canvas sizes are also split by block size, reinforcing D8/0 without
#     creating an opaque numeric rule: larger motifs visibly require larger
#     grids, and repeated samples show that relation.
# hidden_assumptions_review:
#   - main() still assumes one foreground motif whose bbox is a square multiple
#     of 3; generator preserves that by placing only one motif on background.
#   - Original _center_patch assumed fixed offsets valid only for 3x3 blocks.
#     That assumption is removed; center reconstruction now reads whole inward
#     faces, which remains valid for all square block sizes in {2,3,4}.
#   - The block-copy loop assumes the expanded 5x5 block pattern fits around the
#     motif. Generator enforces r0,c0 in [block, n-4*block], so the output patch
#     at (r0-block, c0-block) is always in bounds.
# placement_legality:
#   - For each allowed block size b, grid sizes satisfy n >= 5b + 2, leaving at
#     least one-cell canvas margin beyond the 5b x 5b expanded region.
#   - Motif placement uses the same b on all sides, so copied outer blocks never
#     overlap outside the intended 5x5 lattice and never clip the canvas.
# visual_observability:
#   - The motif footprint changes from 6x6 to 9x9 to 12x12, so block size is
#     directly visible even before solving.
#   - The reconstructed center also changes qualitatively: 2x2 has no interior,
#     3x3 has a single interior majority cell, and 4x4 has a 2x2 majority core.
# visual_tidy:
#   - No extra markers or overlays are added; all copied blocks stay on the same
#     clean 5x5 block lattice as the original task.
#   - Symmetry is preserved in the generator so the visual organization remains
#     crisp rather than noisy despite the size variation.
# inferability:
#   - Colors are copied literally and never remapped by id.
#   - To keep the center-reconstruction rule observable, the generator forces at
#     least one inward corner orbit and one inward edge orbit to use non-majority
#     colors, so the filled center is not visually indistinguishable from a flat
#     majority block.


BLOCK_CHOICES = (2, 3, 4)
GRID_SIZE_BY_BLOCK = {
    2: (12, 16),
    3: (16, 19),
    4: (22, 26),
}
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN = 2
COLOR_COUNT_MAX_BY_BLOCK = {
    2: 3,
    3: 4,
    4: 4,
}
MAJORITY_FRAC_MIN = 0.56
MAJORITY_FRAC_MAX = 0.89
MAX_RETRIES = 500


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _infer_block_from_bbox(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0, "motif must be a square 3x3 block grid"
    block = h // 3
    assert block in BLOCK_CHOICES, "block side must be 2, 3, or 4"
    return int(r0), int(c0), int(block)


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)

    hole_r = int(r0) + block
    hole_c = int(c0) + block

    patch[0, :] = g[hole_r - 1, hole_c : hole_c + block]
    patch[-1, :] = g[hole_r + block, hole_c : hole_c + block]
    patch[:, 0] = g[hole_r : hole_r + block, hole_c - 1]
    patch[:, -1] = g[hole_r : hole_r + block, hole_c + block]

    patch[0, 0] = int(g[hole_r - 1, hole_c - 1])
    patch[0, -1] = int(g[hole_r - 1, hole_c + block])
    patch[-1, 0] = int(g[hole_r + block, hole_c - 1])
    patch[-1, -1] = int(g[hole_r + block, hole_c + block])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a missing-center 3x3 block motif whose square block size varies.

    Input structure:
    - One foreground motif lies on a uniform background.
    - The motif is a 3x3 array of touching square blocks with the middle block
      erased to background.
    - The block side length is not fixed: different examples use 2x2, 3x3, or
      4x4 blocks, so the solver must infer the size from the visible motif.

    Rule:
    - Keep the original motif where it is.
    - Copy each of the eight visible blocks to the point-symmetric outer slot of
      the implied 5x5 block lattice, without rotating the copied block.
    - Rebuild the missing center block by taking its border from the inward
      faces of the surrounding blocks; any remaining interior cells use the
      motif's majority foreground color.

    Key insight: The task logic is still block-based, but the latent block size
    now changes across examples. Once that square block size is inferred from
    the 3x3 motif footprint, the same expansion and inward-face reconstruction
    apply uniformly to 2x2, 3x3, and 4x4 cases.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    r0, c0, block = _infer_block_from_bbox(g, bg)

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


def _diag_orbits(block: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(block):
        for c in range(block):
            if (r, c) in seen:
                continue
            orbit = tuple(sorted({(r, c), (c, r)}))
            seen.update(orbit)
            orbits.append(orbit)
    return tuple(orbits)


def _lr_orbits(block: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(block):
        for c in range(block):
            if (r, c) in seen:
                continue
            orbit = tuple(sorted({(r, c), (r, block - 1 - c)}))
            seen.update(orbit)
            orbits.append(orbit)
    return tuple(orbits)


def _inward_edge_probe(block: int) -> tuple[int, int]:
    if block % 2 == 1:
        return block - 1, block // 2
    return block - 1, block // 2 - 1


def _major_selections(
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    forced_minor: list[tuple[int, int]],
) -> list[tuple[set[int], int, int]]:
    forced_minor_set = set(forced_minor)
    usable = [
        idx
        for idx, orbit in enumerate(orbits)
        if all(coord not in forced_minor_set for coord in orbit)
    ]
    selections: list[tuple[set[int], int, int]] = []
    for mask in range(1 << len(usable)):
        selected: set[int] = set()
        total = 0
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                selected.add(idx)
                total += len(orbits[idx])
        minor_orbits = len(orbits) - len(selected)
        selections.append((selected, total, minor_orbits))
    return selections


def _seed_block(
    block: int,
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    majority: int,
    major_idxs: set[int],
) -> tuple[np.ndarray, list[tuple[np.ndarray, tuple[tuple[int, int], ...]]]]:
    arr = np.full((block, block), -1, dtype=int)
    minor_orbits: list[tuple[np.ndarray, tuple[tuple[int, int], ...]]] = []
    for idx, orbit in enumerate(orbits):
        if idx in major_idxs:
            for r, c in orbit:
                arr[r, c] = majority
        else:
            minor_orbits.append((arr, orbit))
    return arr, minor_orbits


def _assign_minor_colors(
    minor_orbits: list[tuple[np.ndarray, tuple[tuple[int, int], ...]]],
    colors: list[int],
) -> None:
    assert colors, "need at least one minor color"
    random.shuffle(minor_orbits)
    guaranteed = list(colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))
    ]
    for (block_arr, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            block_arr[r, c] = color


def _make_blocks(block: int, majority: int, minor_colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    diag_orbits = _diag_orbits(block)
    lr_orbits = _lr_orbits(block)
    corner_forced = [(block - 1, block - 1)]
    edge_forced = [_inward_edge_probe(block)]

    corner_selections = [
        selection
        for selection in _major_selections(diag_orbits, corner_forced)
        if selection[1] > 0
    ]
    edge_selections = [
        selection
        for selection in _major_selections(lr_orbits, edge_forced)
        if selection[1] > 0
    ]

    total_cells = 2 * block * block
    min_major = int(np.ceil(MAJORITY_FRAC_MIN * total_cells))
    max_major = int(np.floor(MAJORITY_FRAC_MAX * total_cells))
    feasible_pairs = [
        (corner_sel, edge_sel)
        for corner_sel in corner_selections
        for edge_sel in edge_selections
        if min_major <= corner_sel[1] + edge_sel[1] <= max_major
        and corner_sel[2] + edge_sel[2] >= len(minor_colors)
    ]
    assert feasible_pairs, "unreachable majority/minor-orbit configuration"
    corner_sel, edge_sel = random.choice(feasible_pairs)

    b00, corner_minors = _seed_block(block, diag_orbits, majority, corner_sel[0])
    b01, edge_minors = _seed_block(block, lr_orbits, majority, edge_sel[0])
    minor_orbits = corner_minors + edge_minors
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    return b00, b01


def _assemble_motif(block: int, b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
    motif = np.full((block * 3, block * 3), BG, dtype=int)
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
        paste(motif, blk, br * block, bc * block)
    return motif


def _expanded_output(motif: np.ndarray, center: np.ndarray, block: int) -> np.ndarray:
    expanded = np.full((block * 5, block * 5), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, block, block)
    paste(expanded, output_motif, block, block)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            paste(expanded, src, (4 - 2 * br) * block, (4 - 2 * bc) * block)
    return expanded


def _gen_attempt() -> np.ndarray | None:
    block = random.choice(BLOCK_CHOICES)
    n_min, n_max = GRID_SIZE_BY_BLOCK[block]
    n = random.randint(n_min, n_max)
    majority = random.choice(FOREGROUND_COLORS)

    diag_orbits = _diag_orbits(block)
    lr_orbits = _lr_orbits(block)
    max_minor_orbits = len(diag_orbits) + len(lr_orbits) - 1
    color_count_max = min(COLOR_COUNT_MAX_BY_BLOCK[block], max_minor_orbits + 1)
    color_count = random.randint(COLOR_COUNT_MIN, color_count_max)
    minor_colors = random.sample([c for c in FOREGROUND_COLORS if c != majority], color_count - 1)

    b00, b01 = _make_blocks(block, majority, minor_colors)
    motif = _assemble_motif(block, b00, b01)
    center = _center_patch(motif, 0, 0, block, BG)
    expanded = _expanded_output(motif, center, block)

    r0 = random.randint(block, n - 4 * block)
    c0 = random.randint(block, n - 4 * block)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - block, c0 - block)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
