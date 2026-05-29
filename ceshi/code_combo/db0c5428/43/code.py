# Combo variant of db0c5428 (index 43)
# combo_spec: D2/2+D4/1+D11/1
# Name: 可变块大小+无空洞+翻转复制

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for combo D2/2 + D4/1 + D11/1
# ------------------------------------------------
# a) Concrete behavior and trace:
#    - The base task shows a 3x3 array of touching blocks with the middle block
#      missing, then reconstructs that hole and copies the eight visible blocks
#      to the opposite outer slots of an implied 5x5 lattice.
#    - This combo removes the hole entirely, so the input is now a complete 3x3
#      block array. It also changes the block size from fixed 3 to either 2 or
#      4, and changes the outer-copy orientation from identity to flipud.
#    - Small trace with block=2:
#        source top-left block     [[1, 2],      copied outer block     [[3, 4],
#                                  [3, 4]]                             [1, 2]]
#      The original task would paste [[1,2],[3,4]] unchanged and would also
#      need to infer the center block. The combo task keeps the already-filled
#      center block untouched and only performs the flipped outer expansion.
#    - Small trace with block=4 is identical at the block-logic level: infer
#      block size from the 12x12 motif bbox, then paste flipud(block) into each
#      point-symmetric outer slot.
# b) Complexity calibration:
#    - The task is intentionally simpler than the source because center recovery
#      disappears, but it still has two hidden parameters that must be inferred
#      from examples: block size in {2, 4} and the fact that the copied blocks
#      are flipped vertically.
#    - To keep the flip visible, generation rejects samples whose copied blocks
#      are too close to vertical-flip symmetry; otherwise the combo would look
#      like plain copying and collapse toward the original rule family.
# c) Implicit-assumption audit of main():
#    - bbox(g != bg) still assumes one isolated foreground motif. The generator
#      keeps a single motif on a blank background-majority canvas.
#    - The original solver assumed the center block was background and rebuilt
#      it. That assumption is no longer valid, so the combo solver removes all
#      center-patch logic and treats the whole 3x3 motif as already complete.
#    - The loop that crops blocks still assumes a square 3x3 block lattice with
#      no gaps. The generator preserves that, but the solver now also asserts
#      the inferred block side is one of the allowed sizes {2, 4}.
#    - The write loop assumes outer destinations are disjoint and in bounds.
#      Placement is sampled so that the expanded 5b x 5b output always fits.
# d) Placement legality:
#    - For block side b, the motif occupies 3b x 3b and the output occupies
#      5b x 5b, anchored one block up-left from the motif. Sampling
#      r0, c0 in [b, n - 4b] guarantees both the input motif and the output
#      footprint remain inside the n x n canvas for both b=2 and b=4.
# e) Visual observability:
#    - Across examples, solvers can visibly notice that the input motif is
#      sometimes 6x6 and sometimes 12x12, so the hidden block size is grounded
#      by the motif footprint itself.
#    - The copied outer blocks visibly invert top-vs-bottom row order, because
#      seed blocks are generated to differ from their own flipud images.
# f) Visual cleanliness:
#    - The central 3x3 motif remains compact and D4-structured, and the outer
#      copies occupy the same clean ring positions as the base task. No region
#      contains overlapping symbols, and the empty diagonal gaps of the 5x5
#      lattice remain background, making the expansion easy to parse.
# g) Change inferability:
#    - Colors are never remapped. One majority color dominates repeated block
#      orbits, while 1-3 minor colors repeat across symmetric positions. That
#      keeps the only real inference burden on geometry: identify the block size
#      and notice that the outer copies are flipped rather than unchanged.
# h) Post-change assumption review:
#    - main() now assumes the center block already contains foreground pixels and
#      should be preserved. The generator always fills the center block with the
#      majority foreground color, so no hidden reconstruction step remains.
#    - main() also assumes transformed reads come from the original input, not
#      from previously pasted output. The implementation crops from g and writes
#      into a separate out copy, so later pastes cannot contaminate reads.


BLOCK_CHOICES = (2, 4)
GRID_SIZE_BY_BLOCK = {
    2: (12, 16),
    4: (22, 26),
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
MIN_SEED_FLIP_DIFF_BY_BLOCK = {
    2: 2,
    4: 4,
}
MIN_VARIANT_SCORE_BY_BLOCK = {
    2: 12,
    4: 24,
}


def _outer_transform(block: np.ndarray) -> np.ndarray:
    """Reflect a copied block across the horizontal axis."""
    return np.flipud(np.asarray(block, dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a complete 3x3 block motif, inferring its block size, with flipped outer copies.

    Input structure:
    - One foreground motif sits on a uniform background.
    - The motif is a complete 3x3 array of touching square blocks; unlike the
      source task, the center block is already filled and is not special-cased.
    - The block side length is not fixed: each example uses either 2x2 or 4x4
      blocks, so the solver must infer the size from the motif bbox.

    Rule:
    - Keep the original 3x3 motif in place.
    - For each of the eight non-center blocks, copy it to the point-symmetric
      outer slot of the implied 5x5 block lattice.
    - Before each outer paste, flip the copied block top-to-bottom.

    Key insight: The task has become pure block-level expansion. Once the
    solver reads the motif footprint to recover the hidden block size, the only
    nontrivial step is recognizing that every outer duplicate is a vertical
    reflection of its source block rather than a direct copy.
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
            paste(out, _outer_transform(src), dst_r, dst_c)
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
    if not colors:
        return
    random.shuffle(minor_orbits)
    guaranteed = list(colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))
    ]
    for (block_arr, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            block_arr[r, c] = color


def _flipud_distance(block: np.ndarray) -> int:
    arr = np.asarray(block, dtype=int)
    return int(np.count_nonzero(arr != np.flipud(arr)))


def _make_blocks(block: int, majority: int, minor_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    diag_orbits = _diag_orbits(block)
    lr_orbits = _lr_orbits(block)
    corner_forced = [(block - 1, block - 1)]
    edge_forced = [(block - 1, max(0, block // 2 - 1))]

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
    if not feasible_pairs:
        return None

    corner_sel, edge_sel = random.choice(feasible_pairs)
    b00, corner_minors = _seed_block(block, diag_orbits, majority, corner_sel[0])
    b01, edge_minors = _seed_block(block, lr_orbits, majority, edge_sel[0])
    minor_orbits = corner_minors + edge_minors
    _assign_minor_colors(minor_orbits, minor_colors)
    if np.any(b00 == -1) or np.any(b01 == -1):
        return None
    if _flipud_distance(b00) < MIN_SEED_FLIP_DIFF_BY_BLOCK[block]:
        return None
    if _flipud_distance(b01) < MIN_SEED_FLIP_DIFF_BY_BLOCK[block]:
        return None
    return b00, b01


def _assemble_motif(block: int, b00: np.ndarray, b01: np.ndarray, center: np.ndarray) -> np.ndarray:
    motif_size = block * 3
    motif = np.full((motif_size, motif_size), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): np.fliplr(b00),
        (1, 0): b01.T,
        (1, 1): center,
        (1, 2): np.rot90(b01, -1),
        (2, 0): np.flipud(b00),
        (2, 1): np.flipud(b01),
        (2, 2): np.rot90(b00, 2),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * block, bc * block)
    return motif


def _expanded_output(motif: np.ndarray, block: int) -> np.ndarray:
    expanded_size = block * 5
    expanded = np.full((expanded_size, expanded_size), BG, dtype=int)
    paste(expanded, motif, block, block)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            paste(expanded, _outer_transform(src), (4 - 2 * br) * block, (4 - 2 * bc) * block)
    return expanded


def _variant_difference_score(motif: np.ndarray, block: int) -> int:
    score = 0
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            score += int(np.count_nonzero(src != _outer_transform(src)))
    return score


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

    blocks = _make_blocks(block, majority, minor_colors)
    if blocks is None:
        return None
    b00, b01 = blocks
    center = np.full((block, block), majority, dtype=int)
    motif = _assemble_motif(block, b00, b01, center)
    if _variant_difference_score(motif, block) < MIN_VARIANT_SCORE_BY_BLOCK[block]:
        return None

    expanded = _expanded_output(motif, block)
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
