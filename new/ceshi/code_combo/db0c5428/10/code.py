# Combo variant of db0c5428 (index 10)
# combo_spec: D5/3+D11/0
# Name: 角像素编码中心+十字形空洞

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste

# Design analysis for the combo variant
# a) Concretization and behavior trace:
#    - The visible object is still one 9x9 motif made from eight 3x3 blocks around
#      a nominal center block, and the output is still the 15x15 five-by-five block
#      expansion centered on that motif.
#    - Two coupled changes happen at once.
#      1) D5/3 changes the center reconstruction source:
#         instead of using the eight hole-facing boundary pixels, the solver reads
#         only the four corner-block pixels that point diagonally toward the hole:
#         motif-local (2,2), (2,6), (6,2), (6,6).
#         They form a 2x2 symbol [[a,b],[c,d]], which is enlarged to
#             a a b
#             a a b
#             c c d
#         by nearest-neighbor decoding.
#      2) D11/0 changes the missing region shape:
#         the whole 3x3 center is erased, and the center cell of each adjacent edge
#         block is also erased, producing a 13-cell cross at motif-local positions
#         rows 3..5, cols 3..5 plus (1,4), (4,1), (4,7), (7,4).
#    - Concrete trace on one sample:
#      original task:
#        visible edge-facing pixels at (2,4), (4,2), (4,6), (6,4) helped fill the
#        center, and edge blocks were copied directly because they were complete.
#      fused task:
#        those visible edge-facing pixels no longer define the center at all; the
#        center comes only from the four diagonal corner code pixels.
#        After decoding the 3x3 center, the four erased arm cells are restored from
#        the decoded center's cardinal cells:
#          top arm <- center[0,1], left arm <- center[1,0],
#          right arm <- center[1,2], bottom arm <- center[2,1].
#        Only then can the eight surrounding blocks be copied outward.
#    - Main behavioral difference:
#      the solver must heal a larger cross-shaped gap, but it still receives only a
#      four-pixel code. The center decode and arm repair are therefore one linked
#      process rather than two independent subrules.
# b) Complexity calibration:
#    - The code alphabet is just the ordinary ARC foreground colors already visible
#      in the motif; there is no hidden palette arithmetic.
#    - The decoding rule is fixed across all samples: 2x2 symbol -> 3x3 nearest-
#      neighbor center. The arm-repair rule is also fixed: each missing arm cell
#      copies the adjacent decoded center cell.
#    - The generator forces at least two non-majority arm values and at least two
#      distinct arm colors, so the larger cross repair is visibly controlled by the
#      decoded center instead of collapsing to a trivial majority-colored fill.
# c) Implicit-assumption review of main():
#    - bbox(g != bg) still works because the cross erosion stays strictly inside the
#      same 9x9 motif bbox; the four extra missing pixels do not shrink the bbox.
#    - h == w and h % 3 == 0 still hold because block size and motif layout are
#      unchanged from the source task.
#    - Cropping blocks directly from the raw input is no longer valid for the four
#      edge blocks, because each has a missing center cell. The solver must rebuild
#      a healed 9x9 motif first and only then perform outward copying.
#    - The original boundary-sampling _center_patch assumption is invalid. Edge-
#      facing pixels remain visible, but in this combo they are decoys with respect
#      to center reconstruction; only corner code pixels matter.
#    - The arm-repair step introduces a new dependency: the solver assumes the four
#      erased arm cells should equal the four cardinal cells of the decoded center.
#      The generator explicitly constructs completed motifs with that property.
# d) Placement legality:
#    - The generator still places the 9x9 motif with padding >= 3 on every side, so
#      the 15x15 expanded output fits entirely inside the canvas.
#    - The carved cross removes cells only inside the motif, so there is no new
#      overlap risk and no new out-of-bounds case.
# e) Visual observability:
#    - A human sees a cross-shaped tear instead of a simple empty square, and the
#      output visibly repairs both the center and the four punctured edge-block
#      centers.
#    - The center colors can be traced directly back to the four inward-facing
#      corner-code pixels. Because the edge-facing pixels near the hole are forced to
#      a different pattern, multiple examples reveal that the center cannot be coming
#      from the original eight-boundary rule.
# f) Visual cleanliness:
#    - There is still only one motif on a uniform background, and the extra missing
#      arm cells are sparse, symmetric punctures rather than dense clutter.
#    - The repaired cross arms touch the decoded center, so the larger missing region
#      reads as one coherent object instead of unrelated missing bits.
# g) Color-change inferability:
#    - No hidden remapping is used. The filled center and the repaired arm cells reuse
#      exactly the colors already present in the four visible corner code pixels.
#    - The generator rejects samples where the decoded center accidentally matches the
#      original boundary-based center, preventing silent regression to the source task.
# h) Post-edit assumption audit:
#    - _corner_symbol assumes the four corner code pixels are visible. The generator
#      never erases corner blocks, so this is guaranteed.
#    - _restore_cross_arms assumes block size 3 and that the missing arm cells are
#      exactly the four edge-block centers. The generator carves precisely those four
#      cells and no others.
#    - _expanded_output assumes the healed motif is complete before copying. main()
#      now enforces that by reconstructing the motif first.


def _decode_symbol(symbol: np.ndarray) -> np.ndarray:
    """Expand a 2x2 symbol into a 3x3 patch by nearest-neighbor scaling."""
    idx = np.array([0, 0, 1], dtype=int)
    return symbol[np.ix_(idx, idx)].copy()


def _corner_symbol(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    return np.array(
        [
            [g[int(r0) + 2, int(c0) + 2], g[int(r0) + 2, int(c0) + 2 * block]],
            [g[int(r0) + 2 * block, int(c0) + 2], g[int(r0) + 2 * block, int(c0) + 2 * block]],
        ],
        dtype=int,
    )


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    return _decode_symbol(_corner_symbol(g, r0, c0, block))


def _restore_cross_arms(motif: np.ndarray, center: np.ndarray, block: int) -> None:
    mid = block // 2
    motif[mid, block + mid] = int(center[0, mid])
    motif[block + mid, mid] = int(center[mid, 0])
    motif[block + mid, 2 * block + mid] = int(center[mid, 2])
    motif[2 * block + mid, block + mid] = int(center[2, mid])


def _healed_motif(g: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    motif = crop_bbox(
        g,
        (
            int(r0),
            int(c0),
            int(r0) + 3 * block - 1,
            int(c0) + 3 * block - 1,
        ),
    )
    center = _center_patch(g, r0, c0, block)
    paste(motif, center, block, block)
    _restore_cross_arms(motif, center, block)
    return motif


def _expanded_output(motif: np.ndarray, bg: int) -> np.ndarray:
    block = motif.shape[0] // 3
    expanded = np.full((block * 5, block * 5), bg, dtype=int)
    paste(expanded, motif, block, block)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            paste(expanded, src, (4 - 2 * br) * block, (4 - 2 * bc) * block)
    return expanded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif whose center is corner-coded and whose damage forms a cross.

    Input structure:
    - One 9x9 foreground motif sits on a teal background.
    - The motif is a 3x3 arrangement of 3x3 blocks, but the whole middle block is empty
      and the center pixel of each adjacent edge block is also removed.
    - The missing center is encoded only by the four corner-block pixels that point
      diagonally toward the hole.

    Rule:
    - First rebuild the missing 3x3 center by reading the four diagonal corner-code
      pixels as a 2x2 symbol and enlarging that symbol to 3x3 by nearest-neighbor scaling.
    - Use the decoded center to repair the four extra missing edge-block centers, taking
      each repaired arm pixel from the adjacent cardinal cell of the decoded center.
    - After the 9x9 motif is fully healed, copy each non-center block to its point-
      symmetric outer slot in the implied 5x5 block grid, without rotating the copied block.

    Key insight:
    - The cross-shaped hole is larger than before, but it is not carrying more independent
      information. Four visible corner code pixels determine the whole center, and that
      reconstructed center in turn determines the four punctured arm cells, so healing must
      happen before any outward block copying.
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
    assert block == 3, "this task uses 3x3 blocks"

    motif = _healed_motif(g, int(r0), int(c0), block)
    expanded = _expanded_output(motif, bg)

    out = g.copy()
    paste(out, expanded, int(r0) - block, int(c0) - block)
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

_CORNER_CODE_POS = {
    (0, 0): (2, 2),
    (0, 2): (2, 0),
    (2, 0): (0, 2),
    (2, 2): (0, 0),
}

_EDGE_ARM_FROM_CENTER = {
    (0, 1): (0, 1),
    (1, 0): (1, 0),
    (1, 2): (1, 2),
    (2, 1): (2, 1),
}

_EDGE_INNER_POS = {
    (0, 1): (2, 1),
    (1, 0): (1, 2),
    (1, 2): (1, 0),
    (2, 1): (0, 1),
}


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _sample_symbol(colors: list[int], majority: int) -> np.ndarray:
    """Choose a nontrivial 2x2 code whose decoded center also gives informative arms."""
    while True:
        vals = [random.choice(colors) for _ in range(4)]
        if len(set(vals)) < 2:
            continue
        if sum(v != majority for v in vals) < 2:
            continue
        arm_vals = [vals[0], vals[0], vals[1], vals[2]]
        if len(set(arm_vals)) < 2:
            continue
        if sum(v != majority for v in arm_vals) < 2:
            continue
        return np.array(vals, dtype=int).reshape(2, 2)


def _make_block(majority: int, colors: list[int], forced: dict[tuple[int, int], int]) -> np.ndarray:
    """Create a mostly-majority 3x3 block while preserving required clue cells."""
    block = np.full((BLOCK, BLOCK), majority, dtype=int)
    for coord, color in forced.items():
        block[coord] = color

    accent_colors = [c for c in colors if c != majority]
    available = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if (r, c) not in forced]
    random.shuffle(available)
    accent_count = min(len(available), random.randint(2, 4))
    for r, c in available[:accent_count]:
        block[r, c] = random.choice(accent_colors)
    return block


def _assemble_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _gen_center_patch(symbol: np.ndarray) -> np.ndarray:
    return _decode_symbol(symbol)


def _build_complete_motif(frame: np.ndarray, center: np.ndarray) -> np.ndarray:
    motif = frame.copy()
    paste(motif, center, BLOCK, BLOCK)
    _restore_cross_arms(motif, center, BLOCK)
    return motif


def _carve_cross_input(motif: np.ndarray) -> np.ndarray:
    inp_motif = motif.copy()
    inp_motif[BLOCK : 2 * BLOCK, BLOCK : 2 * BLOCK] = BG
    mid = BLOCK // 2
    inp_motif[mid, BLOCK + mid] = BG
    inp_motif[BLOCK + mid, mid] = BG
    inp_motif[BLOCK + mid, 2 * BLOCK + mid] = BG
    inp_motif[2 * BLOCK + mid, BLOCK + mid] = BG
    return inp_motif


def _old_style_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = np.array([BLOCK - 1, BLOCK + 1, BLOCK + BLOCK], dtype=int)
    cols = np.array([BLOCK - 1, BLOCK + 1, BLOCK + BLOCK], dtype=int)
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    symbol = _sample_symbol(colors, majority)
    center = _gen_center_patch(symbol)

    blocks: dict[tuple[int, int], np.ndarray] = {}
    symbol_slots = {
        (0, 0): int(symbol[0, 0]),
        (0, 2): int(symbol[0, 1]),
        (2, 0): int(symbol[1, 0]),
        (2, 2): int(symbol[1, 1]),
    }

    for slot, code_color in symbol_slots.items():
        blocks[slot] = _make_block(majority, colors, {_CORNER_CODE_POS[slot]: code_color})

    for slot, center_pos in _EDGE_ARM_FROM_CENTER.items():
        forced = {
            (1, 1): int(center[center_pos]),
            _EDGE_INNER_POS[slot]: majority,
        }
        blocks[slot] = _make_block(majority, colors, forced)

    frame = _assemble_motif(blocks)
    complete_motif = _build_complete_motif(frame, center)
    input_motif = _carve_cross_input(complete_motif)

    old_majority = _major_foreground_color(input_motif, BG)
    if np.array_equal(center, _old_style_center_patch(input_motif, old_majority)):
        return None

    arm_positions = [(1, 4), (4, 1), (4, 7), (7, 4)]
    arm_values = [int(complete_motif[pos]) for pos in arm_positions]
    if len(set(arm_values)) < 2:
        return None
    if sum(v != majority for v in arm_values) < 2:
        return None

    expanded = _expanded_output(complete_motif, BG)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, input_motif, r0, c0)
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
