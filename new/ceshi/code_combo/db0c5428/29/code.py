# Combo variant of db0c5428 (index 29)
# combo_spec: D2/2+D9/1
# Name: 可变块大小+图案紧贴角落

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for combo D2/2 + D9/1
# --------------------------------------
# a) Concrete behavior trace
#    - The object is still one hollow 3x3 block motif, but each block is now
#      either 2x2 or 4x4 instead of fixed 3x3.
#    - The whole motif is pasted flush against one canvas corner: TL / TR / BL / BR.
#    - Small trace with block=2 at the top-left corner:
#        input motif occupies rows 0..5 and cols 0..5.
#        Old full-margin rule would try to place outer copies on the 5x5 lattice
#        with top-left at (-2, -2), so five of the eight outer slots would lie
#        outside the grid. Only the slots at lattice coordinates (1,3), (3,1),
#        (3,3) remain fully visible.
#        The center block is still reconstructed from the inward-facing borders
#        of the surrounding blocks, but the solver must infer block=2 first.
#    - Compared with the source task, two stages change together:
#        1. infer block size from the motif bbox (6x6 -> 2, 12x12 -> 4),
#        2. copy only those opposite outer slots that still fit on the canvas.
# b) Complexity calibration
#    - Block size is limited to exactly two choices, 2 or 4, so inference stays
#      tractable and is supported by repeated full-block structure.
#    - Corner placement is limited to four corner modes. Every sample therefore
#      loses exactly two outward directions, making the truncation visually clear
#      instead of being a rare partial edge case.
#    - Corner placement always leaves exactly three visible outer copies, which
#      differs sharply from the original eight-copy halo.
# c) Hidden assumptions audited from the old main()
#    - `block == 3` inside `_center_patch` no longer holds; center recovery must
#      work for any allowed block size.
#    - `expanded_r0 = r0 - block` and `expanded_c0 = c0 - block` used to imply
#      every destination paste was in bounds because the generator left margins.
#      With corner placement this is false: several outer slots are intentionally
#      out of bounds and must be skipped.
#    - Reading the motif via one bbox still works because there is still exactly
#      one connected foreground motif with one square center hole.
# d) Placement legality
#    - For block b, the input motif footprint is 3b x 3b and the grid side is
#      sampled from [4b .. 4b + k]. This guarantees the motif fits in a corner
#      and that the three opposite-side outer slots fit fully.
#    - The generator never uses edge-but-not-corner placement, so the truncation
#      pattern is unambiguous and no pasted block is only partially clipped.
# e) Visual observability
#    - Across examples, humans can see two obvious sources of variation:
#      motif footprints switch between 6x6 and 12x12, and the motif touches a
#      corner so the output grows only away from that corner.
#    - No invisible ordering or color-id arithmetic is introduced; colors are
#      copied directly and the missing center is still explained by local border
#      evidence around the hole.
# f) Visual cleanliness
#    - There is only one motif and one reconstructed region. Outer blocks never
#      overlap because they stay on the same 5x5 block lattice as the source task.
#    - Corner placement removes off-canvas copies entirely, so the visible output
#      is a clean corner-attached partial halo rather than a cluttered overlap.
# g) Color / rule inferability
#    - The majority color still dominates the motif and fills the interior of the
#      repaired center block. Non-major colors sit on repeated symmetry orbits,
#      which gives visible support for both the block-copying rule and the border
#      reconstruction rule.
# New solver assumptions introduced by the combo
#    - Assumption: the motif bbox side length is divisible by 3 and yields block
#      size 2 or 4. Guarantee: generator builds only those two footprints.
#    - Assumption: the motif touches exactly two perpendicular canvas edges.
#      Guarantee: generator samples one of the four true corners only.
#    - Assumption: any visible outer copy occupies a whole block-sized slot.
#      Guarantee: corner placement plus integer block alignment ensures each slot
#      is either fully in bounds or fully absent.


BLOCK_CHOICES = (2, 4)
CORNERS = ("tl", "tr", "bl", "br")
GRID_SIZE_BY_BLOCK = {
    2: (8, 12),
    4: (16, 20),
}
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN = 2
COLOR_COUNT_MAX_BY_BLOCK = {
    2: 3,
    4: 4,
}
MAJORITY_FRAC_MIN = 0.56
MAJORITY_FRAC_MAX = 0.89
MAX_RETRIES = 500


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    block = motif.shape[0] // 3
    main = _major_foreground_color(motif, bg)
    patch = np.full((block, block), main, dtype=int)

    hole_r = block
    hole_c = block

    patch[0, :] = motif[hole_r - 1, hole_c : hole_c + block]
    patch[-1, :] = motif[hole_r + block, hole_c : hole_c + block]
    patch[:, 0] = motif[hole_r : hole_r + block, hole_c - 1]
    patch[:, -1] = motif[hole_r : hole_r + block, hole_c + block]

    patch[0, 0] = int(motif[hole_r - 1, hole_c - 1])
    patch[0, -1] = int(motif[hole_r - 1, hole_c + block])
    patch[-1, 0] = int(motif[hole_r + block, hole_c - 1])
    patch[-1, -1] = int(motif[hole_r + block, hole_c + block])
    return patch


def _slot_fits(shape: tuple[int, int], r0: int, c0: int, block: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r0) and 0 <= int(c0) and int(r0) + int(block) <= h and int(c0) + int(block) <= w


def _place_outer_blocks(
    out: np.ndarray,
    motif: np.ndarray,
    r0: int,
    c0: int,
    block: int,
) -> None:
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            if not _slot_fits(out.shape, dst_r, dst_c, block):
                continue
            src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            paste(out, src, dst_r, dst_c)


def _apply_rule(g: np.ndarray, motif: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    out = g.copy()
    _place_outer_blocks(out, motif, r0, c0, block)
    paste(out, _center_patch(motif, bg), int(r0) + block, int(c0) + block)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a corner-attached hollow motif whose block size must be inferred.

    Input structure:
    - One foreground motif sits on a uniform background and touches a canvas corner.
    - The motif is a 3x3 array of touching square blocks with the center block
      erased to background.
    - The block side length is not fixed: each example uses either 2x2 or 4x4
      blocks, so the solver must infer the size from the motif footprint.

    Rule:
    - Keep the original motif where it is and rebuild its missing center block
      from the inward-facing borders of the surrounding blocks.
    - For each visible non-center block, copy it to the point-symmetric outer
      slot of the implied 5x5 block lattice without rotating it.
    - If that outer slot would fall outside the canvas because the motif is
      glued to a corner, skip that copy entirely.

    Key insight: the task now combines hidden scale with boundary truncation.
    First recover the block size from the 3x3 motif bbox, then apply the usual
    opposite-slot expansion only to block positions that remain fully in bounds.
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
    assert block in BLOCK_CHOICES, "block side must be 2 or 4"

    motif = crop_bbox(g, motif_bbox)
    assert np.all(motif[block : 2 * block, block : 2 * block] == bg), "center block must be empty"
    return _apply_rule(g, motif, int(r0), int(c0), block, bg)


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
    edge_forced = [(block - 1, block // 2 - 1)]

    corner_selections = [
        selection for selection in _major_selections(diag_orbits, corner_forced) if selection[1] > 0
    ]
    edge_selections = [
        selection for selection in _major_selections(lr_orbits, edge_forced) if selection[1] > 0
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
    motif_size = block * 3
    motif = np.full((motif_size, motif_size), BG, dtype=int)
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


def _corner_origin(n: int, block: int, corner: str) -> tuple[int, int]:
    motif_size = 3 * block
    r0 = 0 if "t" in corner else int(n) - motif_size
    c0 = 0 if "l" in corner else int(n) - motif_size
    return int(r0), int(c0)


def _visible_outer_slot_count(shape: tuple[int, int], r0: int, c0: int, block: int) -> int:
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block
    count = 0
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            if _slot_fits(shape, dst_r, dst_c, block):
                count += 1
    return count


def _expected_output(inp: np.ndarray, motif: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    return _apply_rule(inp, motif, r0, c0, block, BG)


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

    corner = random.choice(CORNERS)
    r0, c0 = _corner_origin(n, block, corner)
    assert _visible_outer_slot_count((n, n), r0, c0, block) == 3

    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    expected = _expected_output(inp, motif, r0, c0, block)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
