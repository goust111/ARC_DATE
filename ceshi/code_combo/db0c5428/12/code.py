# Combo variant of db0c5428 (index 12)
# combo_spec: D7/0+D14/0
# Name: 自适应扩展输出+先重建中心再外扩

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, get_bg, paste


# Design analysis for combo D7/0 + D14/0
# ----------------------------------------
# concretization:
# - Input keeps the original object family: one 9x9 motif on a teal background,
#   arranged as 3x3 blocks of size 3, with the middle 3x3 block missing.
# - D7/0 changes the canvas contract: the output is no longer forced to stay on
#   the input canvas. Instead we return the minimal background-padded canvas
#   containing both the original input placement and the fully expanded motif.
# - D14/0 changes the processing order. In the original task, outer copies are
#   produced from the visible 8 blocks first, and only afterwards is the center
#   reconstructed. So the four side-center outer slots of the 5x5 frame are
#   determined by edge-block copies only.
# - In the combo, the center is reconstructed first. The completed middle row
#   and middle column can then participate in the outward completion, so the
#   four side-center outer slots at block coordinates (0,2), (2,0), (2,4),
#   (4,2) are overwritten by copies of the reconstructed center block. This is
#   the visible D14 effect: those four outer positions change identity.
#
# behavior trace:
# - Example: input size 16x16, motif bbox top-left at (1, 6).
# - Original task: center is still empty during the copy loop, so only the 8
#   visible non-center blocks determine the outer ring. The intended 15x15
#   region starts at (-2, 3), but same-size output clips the top extension.
# - D7-only variant: resize the canvas to the union of the input canvas and the
#   15x15 expanded region. Output becomes 18x18, and the four side-center outer
#   slots still show the copied edge blocks from the original one-step logic.
# - Combo variant: first rebuild the center patch, then expand. The 18x18 resized
#   output now differs from D7-only on those four side-center 3x3 slots because
#   the center block is copied there after reconstruction and overwrites the edge
#   copies that used to occupy them.
#
# complexity calibration:
# - The puzzle still hinges on the original visible cues: the eight boundary
#   pixels around the hole determine the missing center block, and the whole
#   9x9 motif remains D4-structured.
# - The new effect is strong but not noisy: exactly four stable axis slots switch
#   from edge-block content to center-block content, so a human can spot
#   "rebuild center first" across a few examples without hidden metadata.
# - To keep D7 visible, the generator places the motif near at least one edge so
#   the minimal output canvas must grow on one or more sides.
#
# implicit-assumption audit of the original main():
# - `out = g.copy()` assumed output shape == input shape. D7/0 breaks that.
# - The original copy loop skipped `(1, 1)` because the center was absent while
#   expansion ran. D14/0 breaks that sequencing assumption: the center exists by
#   step 2 and must contribute to the outer completion.
# - `expanded_r0 = r0 - block` and `expanded_c0 = c0 - block` are still the
#   correct anchor for the 15x15 expanded region, but only if we allocate a
#   larger destination canvas before pasting.
# - `_center_patch(...)` assumes the 8 inward-facing pixels exist and are
#   non-background. The generator keeps that true by reusing the original motif
#   construction, where every surrounding block is fully foreground-filled.
#
# placement legality:
# - Input generation keeps the 9x9 motif fully inside the input grid.
# - The expanded region is always exactly 15x15 and is rendered on the minimal
#   union canvas of the original input and the expanded placement, so nothing is
#   clipped and no destination paste goes out of bounds.
# - The four new center-arm copies intentionally target the same side-center
#   slots that edge copies reach in the original logic. The overwrite order is
#   deterministic: center goes last, so no ambiguity remains.
#
# visual observability and neatness:
# - The added center-arm copies are whole 3x3 overwrites, not tiny markers, so
#   the D14 change is obvious even when colors are repeated.
# - Those four slots are axis-aligned and isolated by background, which keeps the
#   enlarged 5x5 arrangement easy to parse rather than cluttered.
# - No rule depends on color ids or hidden ordering; colors only define visible
#   block content, exactly as in the source task.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MAX_RETRIES = 300
PLACEMENT_MODES = ("top", "bottom", "left", "right", "tl", "tr", "bl", "br")
MIN_COMBO_DELTA_CELLS = 16
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

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
_CENTER_ARM_SLOTS = ((0, 2), (2, 0), (2, 4), (4, 2))


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert int(block) == BLOCK, "this task uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)
    hole_r = int(r0) + block
    hole_c = int(c0) + block
    for r in range(block):
        for c in range(block):
            if (r, c) == (block // 2, block // 2):
                continue
            patch[r, c] = int(g[hole_r + int(_CENTER_OFFSETS[r]), hole_c + int(_CENTER_OFFSETS[c])])
    return patch


def _extract_motif(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    return g[int(r0) : int(r0) + 3 * block, int(c0) : int(c0) + 3 * block].copy()


def _expanded_output_from_completed_motif(
    motif: np.ndarray, center: np.ndarray, bg: int
) -> np.ndarray:
    """Render the combo's full 15x15 expanded region."""
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), int(bg), dtype=int)
    completed = motif.copy()
    paste(completed, center, BLOCK, BLOCK)
    paste(expanded, completed, BLOCK, BLOCK)

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = completed[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)

    for dst_br, dst_bc in _CENTER_ARM_SLOTS:
        paste(expanded, center, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _render_on_resized_canvas(
    original: np.ndarray, expanded: np.ndarray, expanded_r0: int, expanded_c0: int, bg: int
) -> np.ndarray:
    """Place the original canvas and expanded motif on their minimal union."""
    h, w = map(int, original.shape)
    eh, ew = map(int, expanded.shape)
    min_r = min(0, int(expanded_r0))
    min_c = min(0, int(expanded_c0))
    max_r = max(h - 1, int(expanded_r0) + eh - 1)
    max_c = max(w - 1, int(expanded_c0) + ew - 1)

    shift_r = -min_r
    shift_c = -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    paste(out, original, shift_r, shift_c)
    paste(out, expanded, int(expanded_r0) + shift_r, int(expanded_c0) + shift_c)
    return out


def _expanded_region_from_grid(
    g: np.ndarray, r0: int, c0: int, block: int, bg: int
) -> np.ndarray:
    motif = _extract_motif(g, int(r0), int(c0), block)
    center = _center_patch(g, int(r0), int(c0), block, bg)
    return _expanded_output_from_completed_motif(motif, center, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing middle block, then expand the completed motif on a resized canvas.

    Input structure:
    - A square background grid contains one 9x9 motif made from 3x3 blocks.
    - The motif's center 3x3 block is missing, but the eight inward-facing pixels
      around that hole encode its boundary.
    - The visible 8 blocks still follow the original D4-style arrangement.

    Rule:
    - First reconstruct the missing center block from the hole boundary, using the
      motif's majority foreground color for the center pixel.
    - Then treat the now-complete 3x3 block array as the source for outward growth:
      keep the completed motif in the middle, copy the 8 non-center blocks to their
      opposite outer slots, and then copy the reconstructed center block to the four
      side-center outer slots of the 5x5 block frame, overwriting the edge copies
      that would occupy those slots in the original one-step process.
    - Return the minimal background-padded canvas that contains both the original
      input placement and this full 15x15 expanded arrangement.

    Key insight: The center block is no longer a final cosmetic fill. It must be
    recovered before expansion, because that recovered block changes which block
    ends up in the four side-center outer slots of the enlarged 5x5 frame.
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

    expanded = _expanded_region_from_grid(g, int(r0), int(c0), block, bg)
    return _render_on_resized_canvas(g, expanded, int(r0) - block, int(c0) - block, bg)


def _major_orbits(orbits, forced_minor, target_count):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits)
        if all(coord not in forced_minor for coord in orbit)
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


def _assemble_motif(b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
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


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = int(majority)
    return patch


def _sample_location(n: int) -> tuple[int, int]:
    max_r0 = int(n - MOTIF_SIZE)
    max_c0 = int(n - MOTIF_SIZE)
    low_r = list(range(0, BLOCK))
    low_c = list(range(0, BLOCK))
    high_r = list(range(max_r0 - (BLOCK - 1), max_r0 + 1))
    high_c = list(range(max_c0 - (BLOCK - 1), max_c0 + 1))
    mid_r = list(range(BLOCK, max_r0 - BLOCK + 1))
    mid_c = list(range(BLOCK, max_c0 - BLOCK + 1))
    assert mid_r and mid_c

    mode = random.choice(PLACEMENT_MODES)
    if mode == "top":
        return random.choice(low_r), random.choice(mid_c)
    if mode == "bottom":
        return random.choice(high_r), random.choice(mid_c)
    if mode == "left":
        return random.choice(mid_r), random.choice(low_c)
    if mode == "right":
        return random.choice(mid_r), random.choice(high_c)
    if mode == "tl":
        return random.choice(low_r), random.choice(low_c)
    if mode == "tr":
        return random.choice(low_r), random.choice(high_c)
    if mode == "bl":
        return random.choice(high_r), random.choice(low_c)
    return random.choice(high_r), random.choice(high_c)


def _expanded_output_d7_baseline(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    completed = motif.copy()
    paste(completed, center, BLOCK, BLOCK)
    paste(expanded, completed, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = completed[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _combo_delta_cells(combo_out: np.ndarray, d7_out: np.ndarray) -> int:
    assert combo_out.shape == d7_out.shape
    return int(np.count_nonzero(combo_out != d7_out))


def _center_arm_pixels(expanded: np.ndarray) -> int:
    total = 0
    for br, bc in _CENTER_ARM_SLOTS:
        block = expanded[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        total += int(np.count_nonzero(block != BG))
    return total


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    possible_totals = [
        total for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
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
    center = _gen_center_patch(motif, majority)
    combo_expanded = _expanded_output_from_completed_motif(motif, center, BG)
    d7_expanded = _expanded_output_d7_baseline(motif, center)

    r0, c0 = _sample_location(n)
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)

    combo_out = _render_on_resized_canvas(inp, combo_expanded, int(r0) - BLOCK, int(c0) - BLOCK, BG)
    d7_out = _render_on_resized_canvas(inp, d7_expanded, int(r0) - BLOCK, int(c0) - BLOCK, BG)

    if combo_out.shape == inp.shape:
        return None
    if _combo_delta_cells(combo_out, d7_out) < MIN_COMBO_DELTA_CELLS:
        return None
    if _center_arm_pixels(combo_expanded) != 4 * BLOCK * BLOCK:
        return None
    if np.array_equal(main(inp), combo_out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo D7/0+D14/0 input")
