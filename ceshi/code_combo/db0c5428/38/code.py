# Combo variant of db0c5428 (index 38)
# combo_spec: D3/1+D6/0
# Name: 单方向扩展+单轴对称
#
# Design analysis for this combo variant
# a) Concrete behavior and trace
#    - The input is still one 9x9 motif made of 3x3 blocks, with the center block empty.
#    - D6/0 removes the original D4-style redundancy: the motif now has exactly one
#      mirror axis, either LR or UD, but never both.
#    - D3/1 then uses that detected axis as the direction of growth:
#      * LR-only motif -> output is a 3x5 block band. The three right-column blocks are
#        copied into a new outer left column, and the three left-column blocks are copied
#        into a new outer right column.
#      * UD-only motif -> output is a 5x3 block band. The three bottom-row blocks are
#        copied into a new outer top row, and the three top-row blocks are copied into a
#        new outer bottom row.
#    - The center block is still reconstructed from the eight inner-boundary pixels around
#      the hole; its center pixel is the motif's majority foreground color.
#    - Small trace: on an LR-only motif, the original solver would also add top and bottom
#      outer rows because it always used point-opposite placement in a full 5x5 frame.
#      The combo solver instead notices strict LR symmetry and leaves those outer rows as bg.
#
# b) Complexity calibration
#    - There are only two branches (horizontal vs vertical extension), so the rule remains
#      learnable from 2-4 examples.
#    - The generator keeps the axis visually obvious by rejecting accidental double symmetry,
#      and each branch repeats the same copy behavior across three blocks, giving enough local
#      evidence for a human to infer the direction rule.
#
# c) Hidden-assumption audit
#    - bbox extraction and 9x9 sizing still hold: the input still contains one isolated motif.
#    - Center reconstruction still holds because the hole and its eight boundary pixels stay in
#      the same coordinates.
#    - The original double loop that mapped every block to a point-reflected outer slot no longer
#      holds. It assumed full two-axis expansion, which would overfill the output in this combo.
#    - The new solver adds one assumption: the motif has exactly one mirror axis. The generator
#      enforces this with explicit symmetry tests.
#
# d) Placement legality
#    - The motif is still placed with at least one block of padding on every side, so the largest
#      possible 15x15 envelope remains in-bounds.
#    - Actual combo outputs are sparser than that envelope: only the left/right outer columns or
#      only the top/bottom outer rows are filled, and these slots never overlap the original motif.
#
# e) Visual observability
#    - Inputs visibly differ between the two branches because one mirror axis is present and the
#      other is absent.
#    - Outputs also differ at a glance: LR examples become 3x5, UD examples become 5x3. This is
#      enough for a human to connect the axis in the input to the extension direction in the output.
#
# f) Visual cleanliness
#    - No overlays are used. Every copied block lands in empty background space.
#    - The original 9x9 motif stays centered in the same 15x15 frame, so the sparse outer copies
#      are easy to segment from the source motif.
#
# g) Change inferability
#    - Colors are preserved exactly; there is no color remapping keyed on hidden ids.
#    - The only color-dependent rule is the center pixel of the reconstructed block taking the
#      majority foreground color, which is already inferable from the motif itself.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox, crop_bbox, get_bg, paste


# --------------- solver helpers ---------------

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


def _detect_axis(motif: np.ndarray) -> str:
    lr = np.array_equal(motif, np.fliplr(motif))
    ud = np.array_equal(motif, np.flipud(motif))
    assert lr ^ ud, "motif must have exactly one mirror axis"
    return "lr" if lr else "ud"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center block, then extend the motif only along its lone mirror axis.

    Input structure:
    - A teal background contains one 9x9 motif arranged as a 3x3 grid of 3x3 blocks.
    - The middle block is empty teal, and the eight pixels touching that hole encode
      the border of the missing center block.
    - The full 9x9 motif has exactly one global mirror axis: either LR or UD symmetry.

    Rule:
    - Determine whether the visible motif is LR-symmetric or UD-symmetric.
    - Reconstruct the center block from the inner-boundary pixels, setting its center
      to the motif's majority foreground color.
    - Keep the original motif in place and extend it only in the direction of that
      symmetry axis by copying boundary blocks into the opposite outer slots.

    Key insight: The missing block is still solved locally, but the growth direction is
    global: the motif's single surviving mirror axis tells you whether to expand sideways
    or vertically, and all other outer slots stay background.
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

    motif = crop_bbox(g, motif_bbox)
    axis = _detect_axis(motif)
    out = g.copy()
    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)

    if axis == "lr":
        for br in range(3):
            src_left = crop_bbox(
                g,
                (
                    int(r0) + br * block,
                    int(c0),
                    int(r0) + (br + 1) * block - 1,
                    int(c0) + block - 1,
                ),
            )
            src_right = crop_bbox(
                g,
                (
                    int(r0) + br * block,
                    int(c0) + 2 * block,
                    int(r0) + (br + 1) * block - 1,
                    int(c0) + 3 * block - 1,
                ),
            )
            paste(out, src_right, int(r0) + br * block, int(c0) - block)
            paste(out, src_left, int(r0) + br * block, int(c0) + 3 * block)
    else:
        for bc in range(3):
            src_top = crop_bbox(
                g,
                (
                    int(r0),
                    int(c0) + bc * block,
                    int(r0) + block - 1,
                    int(c0) + (bc + 1) * block - 1,
                ),
            )
            src_bottom = crop_bbox(
                g,
                (
                    int(r0) + 2 * block,
                    int(c0) + bc * block,
                    int(r0) + 3 * block - 1,
                    int(c0) + (bc + 1) * block - 1,
                ),
            )
            paste(out, src_bottom, int(r0) - block, int(c0) + bc * block)
            paste(out, src_top, int(r0) + 3 * block, int(c0) + bc * block)

    return out


# --------------- generator (output-first) ---------------

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

# -- CFG: majority density --
ASYM_MAJOR_MIN, ASYM_MAJOR_MAX = 5, 7
SYM_MAJOR_MIN, SYM_MAJOR_MAX = 5, 7

# -- CFG: generation --
MAX_RETRIES = 400
MOTIF_RETRIES = 120

SINGLETON_ORBITS = tuple(((r, c),) for r in range(BLOCK) for c in range(BLOCK))
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)
UD_ORBITS = (
    ((0, 0), (2, 0)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 2)),
    ((1, 0),),
    ((1, 1),),
    ((1, 2),),
)
_CENTER_OFFSETS = np.array([-1, 1, 3])


def _reachable_major_counts(orbits, forced_minor):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits)
        if all(coord not in forced_minor for coord in orbit)
    ]
    totals = set()
    for mask in range(1 << len(usable)):
        total = 0
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                total += len(orbits[idx])
        totals.add(total)
    return sorted(totals)


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


def _seed_block(orbits, forced_minor, majority, major_min, major_max):
    valid_counts = [
        count
        for count in _reachable_major_counts(orbits, forced_minor)
        if major_min <= count <= major_max
    ]
    assert valid_counts, "no valid majority count for block family"
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_minor, random.choice(valid_counts))
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


def _gen_center_patch(motif, majority):
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _assemble_lr_motif(b00, b01, b10, b20, b21):
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): np.fliplr(b00),
        (1, 0): b10,
        (1, 2): np.fliplr(b10),
        (2, 0): b20,
        (2, 1): b21,
        (2, 2): np.fliplr(b20),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _assemble_ud_motif(b00, b01, b02, b10, b12):
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): b02,
        (1, 0): b10,
        (1, 2): b12,
        (2, 0): np.flipud(b00),
        (2, 1): np.flipud(b01),
        (2, 2): np.flipud(b02),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _build_lr_motif(majority, minor_colors):
    for _ in range(MOTIF_RETRIES):
        b00, minors00 = _seed_block(SINGLETON_ORBITS, [(2, 2)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b01, minors01 = _seed_block(LR_ORBITS, [(2, 1)], majority, SYM_MAJOR_MIN, SYM_MAJOR_MAX)
        b10, minors10 = _seed_block(SINGLETON_ORBITS, [(1, 2)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b20, minors20 = _seed_block(SINGLETON_ORBITS, [(0, 2)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b21, minors21 = _seed_block(LR_ORBITS, [(0, 1)], majority, SYM_MAJOR_MIN, SYM_MAJOR_MAX)

        minor_orbits = minors00 + minors01 + minors10 + minors20 + minors21
        if len(minor_orbits) < len(minor_colors):
            continue
        _assign_minor_colors(minor_orbits, minor_colors)
        motif = _assemble_lr_motif(b00, b01, b10, b20, b21)
        if np.array_equal(motif, np.fliplr(motif)) and not np.array_equal(motif, np.flipud(motif)):
            return motif
    raise AssertionError("failed to build a strict LR-only motif")


def _build_ud_motif(majority, minor_colors):
    for _ in range(MOTIF_RETRIES):
        b00, minors00 = _seed_block(SINGLETON_ORBITS, [(2, 2)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b01, minors01 = _seed_block(SINGLETON_ORBITS, [(2, 1)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b02, minors02 = _seed_block(SINGLETON_ORBITS, [(2, 0)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b10, minors10 = _seed_block(UD_ORBITS, [(1, 2)], majority, SYM_MAJOR_MIN, SYM_MAJOR_MAX)
        b12, minors12 = _seed_block(UD_ORBITS, [(1, 0)], majority, SYM_MAJOR_MIN, SYM_MAJOR_MAX)

        minor_orbits = minors00 + minors01 + minors02 + minors10 + minors12
        if len(minor_orbits) < len(minor_colors):
            continue
        _assign_minor_colors(minor_orbits, minor_colors)
        motif = _assemble_ud_motif(b00, b01, b02, b10, b12)
        if np.array_equal(motif, np.flipud(motif)) and not np.array_equal(motif, np.fliplr(motif)):
            return motif
    raise AssertionError("failed to build a strict UD-only motif")


def _expanded_output(motif, center, axis):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)

    if axis == "lr":
        for br in range(3):
            left = motif[br * BLOCK:(br + 1) * BLOCK, 0:BLOCK]
            right = motif[br * BLOCK:(br + 1) * BLOCK, 2 * BLOCK:3 * BLOCK]
            paste(expanded, right, BLOCK + br * BLOCK, 0)
            paste(expanded, left, BLOCK + br * BLOCK, 4 * BLOCK)
    else:
        for bc in range(3):
            top = motif[0:BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            bottom = motif[2 * BLOCK:3 * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            paste(expanded, bottom, 0, BLOCK + bc * BLOCK)
            paste(expanded, top, 4 * BLOCK, BLOCK + bc * BLOCK)

    return expanded


def _full_expanded_output(motif, center):
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


def _outer_block_count(expanded, bg):
    count = 0
    for br in range(5):
        for bc in range(5):
            if 1 <= br <= 3 and 1 <= bc <= 3:
                continue
            block = expanded[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            if np.any(block != bg):
                count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    axis = random.choice(("lr", "ud"))
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    motif = (
        _build_lr_motif(majority, minor_colors)
        if axis == "lr"
        else _build_ud_motif(majority, minor_colors)
    )
    center = _gen_center_patch(motif, majority)
    expanded = _expanded_output(motif, center, axis)
    full_expanded = _full_expanded_output(motif, center)

    assert _detect_axis(motif) == axis
    assert _outer_block_count(expanded, BG) == 6
    assert not np.array_equal(expanded, full_expanded), "combo output degenerated to original behavior"

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
