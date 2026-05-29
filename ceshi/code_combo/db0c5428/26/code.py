# Combo variant of db0c5428 (index 26)
# combo_spec: D12/2+D14/0
# Name: 部分正确空洞+先重建再外扩

from __future__ import annotations

import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The base task shows one 9x9 motif built from a 3x3 arrangement of 3x3 blocks.
#    Its center block is no longer a clean hole: under D12/2 the input middle 3x3
#    already contains a mixture of correct and incorrect cells. Under D14/0 we must
#    repair that center first, then treat the repaired 3x3 block array as the source
#    for expansion.
#    Concrete trace:
#    - Suppose the true center should be
#          6 9 6
#          9 5 9
#          6 9 6
#    - The input may instead show
#          6 9 6
#          8 1 9
#          6 9 6
#      so two cells are corrupted.
#    - Original db0c5428 would only use the eight surrounding non-center blocks to
#      fill the sparse outer ring; the center repair affects only the middle 3x3.
#    - This combo first restores the true center, then uses that repaired center as
#      an additional expansion source: the eight old outer-ring targets still receive
#      the eight directional source blocks, and the eight boundary gap slots that used
#      to remain background are now all filled with the repaired center block.
#    - Therefore a bad center reconstruction now propagates to eight extra 3x3 copies
#      in the output rather than being confined to the middle block.
# b) Complexity calibration:
#    The corruption masks come from a small recurring library, so "some cells are
#    already correct, some need repair" is visible without exploding into many cases.
#    The D14 step is also tightly controlled: the new expansion always repeats the
#    repaired center into the same 8 formerly empty boundary slots. This makes the
#    multi-step dependency obvious while keeping the rule learnable from few examples.
# c) Implicit-assumption audit:
#    - `bbox(g != bg)` still works because the outer eight blocks remain intact and
#      determine the 9x9 motif extent even if some center cells are background.
#    - Inferring the majority color from the whole motif is no longer safe because the
#      damaged center can skew counts. We must infer the center pixel's color from the
#      outer eight blocks only.
#    - Reading the center directly from the input is unsafe; the solver must overwrite
#      the whole center block with a reconstruction from reliable boundary evidence.
#    - Original copy logic skipped the center because it was absent. That assumption is
#      exactly what D14/0 breaks. After reconstruction, the center becomes a valid
#      source block and must affect outer placement.
#    - The new gap-fill loop assumes the output layout still uses the original 5x5
#      frame and that those 8 boundary-interstitial slots are empty before D14.
#      Generator preserves that geometry.
# d) Placement legality:
#    The generator keeps the original padding constraints: the 9x9 motif is placed at
#    least one block-width away from every canvas edge, so the full 15x15 output fits.
#    Corruption is confined to the existing center block, and the extra D14 copies land
#    only in the 8 predefined background slots of the 5x5 layout, so no overlaps arise.
# e) Visual observability:
#    Across 2-4 examples a human can see both parts of the fusion:
#    the center is partially repaired rather than blank, and the finished center then
#    reappears repeatedly around the outside. The added center copies make the step
#    order visible because the outer result now depends on correct repair.
# f) Visual cleanliness:
#    The output remains a regular 15x15 block layout. The new content occupies the same
#    8 empty boundary slots in every sample, so the picture stays structured instead of
#    noisy or overlapping.
# g) Inferability / anti-degeneration:
#    No hidden color-id rule is introduced. Colors still come from visible motif roles.
#    To ensure the combo does not collapse back to the original behavior, every sample
#    has at least one wrong center cell in the input, and the D14 step always adds
#    8 non-background center copies to slots that were background in the base task.


BLOCK = 3
MOTIF_BLOCKS = 3
OUTPUT_BLOCKS = 5
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
EXPANDED_SIZE = BLOCK * OUTPUT_BLOCKS
MOTIF_PAD = BLOCK

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
_CENTER_GAP_SLOTS = (
    (0, 1),
    (0, 3),
    (1, 0),
    (1, 4),
    (3, 0),
    (3, 4),
    (4, 1),
    (4, 3),
)

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
    assert block == BLOCK, "this task uses fixed 3x3 blocks"
    majority = _major_foreground_color_from_outer_blocks(g, r0, c0, block, bg)
    rows = int(r0) + block + _CENTER_OFFSETS
    cols = int(c0) + block + _CENTER_OFFSETS
    patch = g[np.ix_(rows, cols)].copy()
    patch[block // 2, block // 2] = majority
    return patch


def _extract_motif(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    return g[int(r0) : int(r0) + 3 * block, int(c0) : int(c0) + 3 * block].copy()


def _expanded_output(repaired_motif: np.ndarray, center: np.ndarray, bg: int) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), int(bg), dtype=int)
    paste(expanded, repaired_motif, BLOCK, BLOCK)

    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                continue
            src = repaired_motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)

    for br, bc in _CENTER_GAP_SLOTS:
        paste(expanded, center, br * BLOCK, bc * BLOCK)
    return expanded


def _original_style_output(repaired_motif: np.ndarray, bg: int) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), int(bg), dtype=int)
    paste(expanded, repaired_motif, BLOCK, BLOCK)
    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                continue
            src = repaired_motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a partially corrupted center block, then expand the repaired 3x3 block array.

    Input structure:
    - A uniform teal background contains one 9x9 motif made from a 3x3 arrangement of 3x3 blocks.
    - The eight non-center blocks still form the same D4-symmetric frame as in the base task.
    - The middle 3x3 block is a damaged preview of the true center: some cells are already correct, while others are wrong.

    Rule:
    - Reconstruct the full center block from the eight inward-facing boundary pixels of the surrounding blocks; the center pixel of that block is the majority foreground color measured from the outer eight blocks only.
    - Replace the entire middle 3x3 block with that reconstruction before any outward copying.
    - Paste the repaired 9x9 motif back into the middle of the output, copy each non-center source block to its original opposite outer slot, and fill the 8 remaining boundary gap slots of the 5x5 frame with copies of the repaired center block.

    Key insight:
    - The center area is editable evidence, not trusted input. You must first recover the hidden center from the reliable outer ring, and only then use that repaired center as an active source whose pattern repeats into the outer expansion.
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
    assert block == BLOCK, "this combo expects 3x3 blocks"

    center = _reconstruct_center_patch(g, int(r0), int(c0), block, bg)
    repaired_motif = _extract_motif(g, int(r0), int(c0), block)
    paste(repaired_motif, center, block, block)

    out = g.copy()
    expanded = _expanded_output(repaired_motif, center, bg)
    paste(out, expanded, int(r0) - block, int(c0) - block)
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
    assignments = guaranteed + [
        random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))
    ]
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


def variant_metrics(grid: np.ndarray) -> dict[str, int | str]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None
    r0, c0, r1, c1 = motif_bbox
    block = (int(r1 - r0 + 1)) // 3
    center = g[int(r0) + block : int(r0) + 2 * block, int(c0) + block : int(c0) + 2 * block]
    target = _reconstruct_center_patch(g, int(r0), int(c0), block, bg)
    repaired_motif = _extract_motif(g, int(r0), int(c0), block)
    paste(repaired_motif, target, block, block)
    combo_out = _expanded_output(repaired_motif, target, bg)
    base_out = _original_style_output(repaired_motif, bg)
    return {
        "correct_cells": int(np.sum(center == target)),
        "wrong_cells": int(np.sum(center != target)),
        "wrong_bg_cells": int(np.sum((center != target) & (center == bg))),
        "wrong_fg_cells": int(np.sum((center != target) & (center != bg))),
        "d14_added_cells": int(np.sum(combo_out != base_out)),
        "center_palette_size": int(np.unique(target).size),
    }


def _gen_attempt(return_meta: bool = False):
    n = random.randint(16, 19)
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
    repaired_motif = motif.copy()
    paste(repaired_motif, true_center, BLOCK, BLOCK)
    expanded = _expanded_output(repaired_motif, true_center, BG)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    motif_with_corrupted_center = motif.copy()
    paste(motif_with_corrupted_center, corrupted_center, BLOCK, BLOCK)
    paste(inp, motif_with_corrupted_center, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    if not np.array_equal(main(inp), out):
        return (None, None) if return_meta else None

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
    raise AssertionError("failed to generate a valid db0c5428 combo input")
