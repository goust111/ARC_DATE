# Variant of db0c5428
# Dimension: D12 — 背景色与空洞的区分方式 (meta)
# Change: 空洞区域填充与网格背景相同的颜色8，空洞通过其在图案内部的位置和被非背景像素包围来识别... → 空洞内包含部分正确像素

from __future__ import annotations

import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox, get_bg, paste


# Design analysis for the D12 variant
# -----------------------------------
# variant_spec:
#   unchanged_structure:
#     - The useful object is still one 9x9 motif made from a 3x3 arrangement of 3x3 blocks.
#     - The eight non-center blocks still define the same D4-symmetric frame.
#     - The output is still a 15x15 block layout embedded in the same canvas.
#   core_change:
#     - The center 3x3 region is no longer an empty teal hole.
#     - Instead it is a corrupted preview of the true center block: some cells are already
#       correct, while the remaining cells are wrong and must be repaired.
#   behavior_trace:
#     - Original task: center block in the input is all background, so filling the whole
#       3x3 center never risks overwriting good evidence.
#     - New task: the input center may already contain correct values, so the solver must
#       treat the whole center as an editable repair region and reconstruct the target from
#       the surrounding eight inward-facing boundary pixels rather than trusting any center
#       cell individually.
#     - Concrete example: if the true center is
#           6 9 6
#           9 5 9
#           6 9 6
#       the input may show
#           6 9 6
#           8 1 9
#           6 9 6
#       where the ring cells are already correct but the middle-left and center cells are
#       corrupted. The output must restore the full true center.
#   complexity_calibration:
#     - Corruption masks are chosen from a small library of 4 recurring patterns. This keeps
#       the "partially repaired" idea visible across samples without exploding into many
#       unrelated subcases.
#     - Wrong pixels are drawn only from the background plus existing motif colors, so the
#       center looks damaged rather than introducing unexplained new colors.
#   main_assumption_review:
#     - bbox(g != bg): still valid, because the outer eight blocks remain foreground and fix
#       the motif's 9x9 extent even if some center cells revert to background.
#     - majority-color inference from the whole motif: no longer safe, because corrupted
#       center pixels can skew counts. The solver must infer the majority from the outer
#       eight blocks only.
#     - unconditional overwrite of the center with the reconstructed patch: now intentional.
#       Some center cells are already right, but replacing the whole 3x3 block is safe
#       because reconstruction comes from the reliable outer ring.
#     - copying source blocks from the input: still safe because only the center block is
#       corrupted and the copy loop continues to skip that block.
#   placement_legality:
#     - The same padding constraints as the original task guarantee that the 15x15 expanded
#       output fits after placing the 9x9 motif.
#     - Corruption is confined to the existing center block, so no new overlaps are created.
#   visual_observability:
#     - Humans can still see the original expansion rule from the outer eight blocks.
#     - The D12 change is visible because the middle block contains a mix of already-correct
#       cells and obviously wrong cells, so the center is read as "damaged content" rather
#       than a blank hole.
#   neatness_and_color_inference:
#     - At least one center cell stays correct and at least one becomes wrong.
#     - When there are multiple wrong cells, generation forces a mix of background and
#       non-background corruption, making the repair behavior visible without creating a
#       cluttered rainbow patch.


GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_PAD = 3

BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

MAX_RETRIES = 300

DIAG_ORBITS = (
    ((0, 0),),
    ((1, 1),),
    ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)

_CENTER_OFFSETS = np.array([-1, 1, 3], dtype=int)

CORRECT_CENTER_MASKS = {
    "ring": np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=bool,
    ),
    "checker": np.array(
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        dtype=bool,
    ),
    "cross": np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=bool,
    ),
    "corners": np.array(
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
        dtype=bool,
    ),
}


def _major_foreground_color_from_outer_blocks(
    g: np.ndarray, r0: int, c0: int, block: int, bg: int
) -> int:
    motif = g[int(r0) : int(r0) + 3 * block, int(c0) : int(c0) + 3 * block]
    mask = np.ones((3 * block, 3 * block), dtype=bool)
    mask[block : 2 * block, block : 2 * block] = False
    vals, counts = np.unique(motif[mask & (motif != int(bg))], return_counts=True)
    assert vals.size, "input must contain foreground colors outside the repaired center"
    return int(vals[int(np.argmax(counts))])


def _reconstruct_center_patch(
    g: np.ndarray, r0: int, c0: int, block: int, bg: int
) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    majority = _major_foreground_color_from_outer_blocks(g, r0, c0, block, bg)
    rows = int(r0) + block + _CENTER_OFFSETS
    cols = int(c0) + block + _CENTER_OFFSETS
    patch = g[np.ix_(rows, cols)].copy()
    patch[block // 2, block // 2] = majority
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the partially filled center of a 3x3-block motif, then expand it into a 5x5 block layout.

    Input structure:
    - Teal background surrounds one 9x9 motif made from eight visible 3x3 blocks around a central 3x3 slot.
    - The outer eight blocks still form the same symmetric frame as in the original task.
    - The middle slot is not blank anymore: some pixels already match the hidden center block, while the remaining pixels are corrupted placeholders.

    Rule:
    - Reconstruct the true center block from the eight inward-facing boundary pixels of the surrounding blocks; the center cell of that block is the majority foreground color measured from the outer eight blocks only.
    - Replace the whole middle 3x3 slot by this reconstructed patch, even if some input cells there were already correct.
    - Keep the repaired 9x9 motif in place and copy each non-center block to the opposite outer slot of the implied 5x5 block grid, without rotating the copied block.

    Key insight:
    - The center 3x3 area is damaged evidence, not trustworthy structure. The reliable information lives in the surrounding symmetric ring, which both determines the true center patch and supplies the tiles for the outer expansion.
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
    center = _reconstruct_center_patch(g, int(r0), int(c0), block, bg)
    paste(out, center, int(r0) + block, int(c0) + block)

    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = g[
                int(r0) + br * block : int(r0) + (br + 1) * block,
                int(c0) + bc * block : int(c0) + (bc + 1) * block,
            ]
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, src, dst_r, dst_c)
    return out


def _major_orbits(orbits, forced_minor, target_count):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits) if all(coord not in forced_minor for coord in orbit)
    ]
    choices = []
    for mask in range(1 << len(usable)):
        selected = set()
        total = 0
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                selected.add(idx)
                total += len(orbits[idx])
        if total == target_count:
            choices.append(selected)
    assert choices, "unreachable symmetric majority count"
    return random.choice(choices)


def _seed_block(orbits, forced_minor, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_minor, target_count)
    minor_orbits = []
    for idx, orbit in enumerate(orbits):
        if idx in major_idxs:
            for r, c in orbit:
                block[r, c] = majority
        else:
            minor_orbits.append((block, orbit))
    return block, minor_orbits


def _assign_minor_colors(minor_orbits, colors):
    random.shuffle(minor_orbits)
    guaranteed = list(colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))]
    for (block, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            block[r, c] = color


def _make_blocks(majority, minor_colors, corner_major, edge_major):
    b00, corner_minors = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
    b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    return b00, b01


def _assemble_motif(b00, b01):
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
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
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _gen_center_patch(motif, majority):
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _pick_wrong_foreground(target: int, motif_colors: list[int]) -> int:
    candidates = [color for color in motif_colors if color != int(target)]
    assert candidates, "need at least one alternate motif color for corruption"
    return int(random.choice(candidates))


def _corrupt_center_patch(true_center: np.ndarray, motif_colors: list[int]):
    mask_name = random.choice(sorted(CORRECT_CENTER_MASKS))
    keep_mask = CORRECT_CENTER_MASKS[mask_name]
    corrupted = true_center.copy()
    wrong_positions = [tuple(rc) for rc in np.argwhere(~keep_mask)]
    assert wrong_positions, "each corruption mask must leave something to repair"

    if len(wrong_positions) == 1:
        r, c = wrong_positions[0]
        target = int(true_center[r, c])
        corrupted[r, c] = random.choice([BG, _pick_wrong_foreground(target, motif_colors)])
    else:
        bg_r, bg_c = wrong_positions[0]
        bg_target = int(true_center[bg_r, bg_c])
        corrupted[bg_r, bg_c] = BG if bg_target != BG else _pick_wrong_foreground(bg_target, motif_colors)

        fg_r, fg_c = wrong_positions[1]
        corrupted[fg_r, fg_c] = _pick_wrong_foreground(int(true_center[fg_r, fg_c]), motif_colors)

        for r, c in wrong_positions[2:]:
            target = int(true_center[r, c])
            choices = [BG, _pick_wrong_foreground(target, motif_colors)]
            corrupted[r, c] = int(random.choice(choices))

    assert np.any(corrupted == true_center), "variant needs some already-correct center cells"
    assert np.any(corrupted != true_center), "variant needs some incorrect center cells"
    return corrupted, mask_name


def _expanded_output(motif, center):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None
    r0, c0, r1, c1 = motif_bbox
    block = (int(r1 - r0 + 1)) // 3
    center = g[int(r0) + block : int(r0) + 2 * block, int(c0) + block : int(c0) + 2 * block]
    target = _reconstruct_center_patch(g, int(r0), int(c0), block, bg)
    correct = int(np.sum(center == target))
    wrong = int(np.sum(center != target))
    wrong_bg = int(np.sum((center != target) & (center == bg)))
    wrong_fg = int(np.sum((center != target) & (center != bg)))
    return {
        "correct_cells": correct,
        "wrong_cells": wrong,
        "wrong_bg_cells": wrong_bg,
        "wrong_fg_cells": wrong_fg,
    }


def _gen_attempt(return_meta: bool = False):
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    possible_totals = [
        total
        for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
        if 18 - total >= len(minor_colors)
    ]
    majority_total = random.choice(possible_totals)
    pairs = [
        (corner, majority_total - corner)
        for corner in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1)
        if EDGE_MAJOR_MIN <= majority_total - corner <= EDGE_MAJOR_MAX
    ]
    corner_major, edge_major = random.choice(pairs)

    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    motif = _assemble_motif(b00, b01)
    true_center = _gen_center_patch(motif, majority)
    corrupted_center, mask_name = _corrupt_center_patch(true_center, colors)
    expanded = _expanded_output(motif, true_center)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    motif_with_corrupted_center = motif.copy()
    paste(motif_with_corrupted_center, corrupted_center, BLOCK, BLOCK)
    paste(inp, motif_with_corrupted_center, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    if not np.array_equal(main(inp), out):
        return None if not return_meta else (None, None)

    if return_meta:
        meta = variant_metrics(inp)
        meta["mask_name"] = mask_name
        return inp, meta
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D12 variant input")
