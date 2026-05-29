# Combo variant of db0c5428 (index 3)
# combo_spec: D1/1+D6/0
# Name: 镜像映射+单轴对称

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a_concretization:
#   - The visible object is still a single 9x9 motif made of eight 3x3 blocks
#     around a blank 3x3 center hole. The eight pixels facing that hole still
#     encode the boundary of the missing center block, and the center pixel is
#     still the majority foreground color.
#   - D6/0 changes the input evidence: the whole 9x9 motif now has exactly one
#     axial symmetry, either horizontal (flipud) or vertical (fliplr), never both.
#   - D1/1 changes the outer copy rule: every visible block is copied to an outer
#     slot by mirroring only across that one axis, not by 180-degree point
#     reflection. For a horizontal-axis motif, block(0,0) goes to outer slot
#     (4,1) instead of the original (4,4); block(1,0) goes to (2,0); block(2,2)
#     goes to (0,3). The vertical-axis case is the transposed analogue.
#   - This combo therefore keeps the full 8-block outer frame from D1/1, but the
#     axis controlling that frame must be inferred from the D6/0 single symmetry.
#     It is not the D6 single-variant's reduced 6-block expansion.
# b_complexity:
#   - There are only two branches, horizontal or vertical, so the rule remains
#     learnable from 2-4 examples.
#   - Each branch has repeated support across many blocks: three outer blocks on
#     one side strip, three on the opposite strip, and two side midpoints. That
#     repeated structure makes the axis-conditioned mapping visible.
#   - The generator deliberately uses internally asymmetric seed blocks where
#     possible; otherwise the mirror-map output could look too similar to the
#     original point-reflection output.
# c_hidden_assumptions_review:
#   - bbox() and the 9x9/3x3 structure still hold, because the input still
#     contains exactly one foreground motif on a plain background.
#   - _center_patch() still holds unchanged; the hole-adjacent encoding pixels
#     remain in the same eight absolute offsets from the hole.
#   - The original destination formula (4 - 2*br, 4 - 2*bc) assumed both block
#     coordinates invert together. That is false here: only one coordinate family
#     mirrors, chosen by the motif's single symmetry axis.
#   - New assumption: the motif must have exactly one axial symmetry so axis
#     inference is unambiguous. The generator enforces horizontal xor vertical.
#   - New anti-degeneracy assumption: the axis-mirror expansion should not collapse
#     to the same 15x15 output as the original point-reflection rule. The generator
#     checks this explicitly before accepting a sample.
# d_placement_legality:
#   - The output still fits inside the same 15x15 frame centered around the motif,
#     so the original one-block padding on all sides remains sufficient.
#   - All 8 destination slots are distinct members of that frame, so no overlap is
#     introduced by the fused mapping.
# e_visual_observability:
#   - A solver can see the control variable directly in the input: the motif is
#     symmetric in exactly one axial direction and clearly not in the other.
#   - The output then reinforces that choice: horizontal symmetry creates full top
#     and bottom outer rows plus left/right midpoints; vertical symmetry creates
#     full left and right outer columns plus top/bottom midpoints.
#   - Because the generator rejects point-reflection-equivalent samples, the combo
#     remains observably different from the base task.
# f_cleanliness:
#   - There is still only one motif, one hole, and one 15x15 expansion frame.
#   - No marker colors or overlays are added. All pasted copies land on empty
#     background slots, so the scene remains visually sparse and parseable.
# g_color_inference:
#   - Colors keep the original role: one majority foreground color and 1-3 minor
#     colors for detail. No color-id ordering is used.
#   - The only color computation is the original center-pixel majority rule, which
#     is inferable from the motif itself.


HORIZONTAL_AXIS = "horizontal"
VERTICAL_AXIS = "vertical"


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


def _infer_axis(motif: np.ndarray) -> str:
    horizontal = np.array_equal(motif, np.flipud(motif))
    vertical = np.array_equal(motif, np.fliplr(motif))
    assert horizontal ^ vertical, "motif must have exactly one axial symmetry"
    return HORIZONTAL_AXIS if horizontal else VERTICAL_AXIS


def _outer_slot(br: int, bc: int, axis: str) -> tuple[int, int]:
    if axis == HORIZONTAL_AXIS:
        if br == 0:
            return 4, 1 + bc
        if br == 2:
            return 0, 1 + bc
        assert br == 1 and bc in (0, 2)
        return 2, 0 if bc == 0 else 4

    if bc == 0:
        return 1 + br, 4
    if bc == 2:
        return 1 + br, 0
    assert bc == 1 and br in (0, 2)
    return (0 if br == 0 else 4), 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3-block motif into a 5x5 frame using an axis-linked mirror map.

    Input structure:
    - A teal background contains one 9x9 motif made from eight visible 3x3 blocks.
    - The center 3x3 block is missing, and the eight pixels facing that hole still
      encode the border of the center block that must be rebuilt.
    - The full 9x9 motif has exactly one axial symmetry, either horizontal or
      vertical, and that lone symmetry is the only global cue for the outer copies.

    Rule:
    - Detect which single symmetry axis the motif obeys.
    - Reconstruct the missing center block from the hole-adjacent pixels, placing
      the motif's majority foreground color at the center pixel.
    - Keep the original motif in place, then copy every non-center 3x3 block into
      the 5x5 outer frame by mirroring only across the detected axis, without
      rotating any copied block.

    Key insight:
    The center hole is still solved locally, but the outer expansion is not the
    original point reflection. The motif's only mirror axis tells you which family
    of outer slots each visible block maps to.
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
    axis = _infer_axis(motif)

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
            dst_br, dst_bc = _outer_slot(br, bc, axis)
            paste(out, src, expanded_r0 + dst_br * block, expanded_c0 + dst_bc * block)

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
MAJORITY_MIN, MAJORITY_MAX = 5, 7

GENERAL_ORBITS = tuple(((r, c),) for r in range(BLOCK) for c in range(BLOCK))
UD_ORBITS = (
    ((0, 0), (2, 0)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 2)),
    ((1, 0),),
    ((1, 1),),
    ((1, 2),),
)
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)
_CENTER_OFFSETS = np.array([-1, 1, 3])


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
    assert choices, "unreachable majority count"
    return random.choice(choices)


def _assign_minor_colors(orbits_left, palette):
    assert palette, "need at least one minor color"
    random.shuffle(orbits_left)
    guaranteed = list(palette[: min(len(palette), len(orbits_left))])
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(palette) for _ in range(len(orbits_left) - len(guaranteed))
    ]
    return assignments


def _make_block(
    orbits,
    majority: int,
    minor_colors: list[int],
    forced_minor,
    *,
    forbid_ud: bool = False,
    forbid_lr: bool = False,
) -> np.ndarray:
    for _ in range(200):
        target_major = random.randint(MAJORITY_MIN, MAJORITY_MAX)
        block = np.full((BLOCK, BLOCK), -1, dtype=int)
        major_idxs = _major_orbits(orbits, forced_minor, target_major)
        leftover = []
        for idx, orbit in enumerate(orbits):
            if idx in major_idxs:
                for r, c in orbit:
                    block[r, c] = majority
            else:
                leftover.append(orbit)

        for orbit, color in zip(leftover, _assign_minor_colors(leftover, minor_colors)):
            for r, c in orbit:
                block[r, c] = color

        if forbid_ud and np.array_equal(block, np.flipud(block)):
            continue
        if forbid_lr and np.array_equal(block, np.fliplr(block)):
            continue
        return block
    raise AssertionError("failed to build asymmetric block")


def _assemble_horizontal_motif(majority: int, minor_colors: list[int]) -> np.ndarray:
    tl = _make_block(GENERAL_ORBITS, majority, minor_colors, [(2, 2)], forbid_ud=True, forbid_lr=True)
    tm = _make_block(GENERAL_ORBITS, majority, minor_colors, [(2, 1)], forbid_ud=True, forbid_lr=True)
    tr = _make_block(GENERAL_ORBITS, majority, minor_colors, [(2, 0)], forbid_ud=True, forbid_lr=True)
    ml = _make_block(UD_ORBITS, majority, minor_colors, [(1, 2)], forbid_lr=True)
    mr = _make_block(UD_ORBITS, majority, minor_colors, [(1, 0)], forbid_lr=True)

    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): tl,
        (0, 1): tm,
        (0, 2): tr,
        (1, 0): ml,
        (1, 2): mr,
        (2, 0): np.flipud(tl),
        (2, 1): np.flipud(tm),
        (2, 2): np.flipud(tr),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _assemble_vertical_motif(majority: int, minor_colors: list[int]) -> np.ndarray:
    tl = _make_block(GENERAL_ORBITS, majority, minor_colors, [(2, 2)], forbid_ud=True, forbid_lr=True)
    ml = _make_block(GENERAL_ORBITS, majority, minor_colors, [(1, 2)], forbid_ud=True, forbid_lr=True)
    bl = _make_block(GENERAL_ORBITS, majority, minor_colors, [(0, 2)], forbid_ud=True, forbid_lr=True)
    tm = _make_block(LR_ORBITS, majority, minor_colors, [(2, 1)], forbid_ud=True)
    bm = _make_block(LR_ORBITS, majority, minor_colors, [(0, 1)], forbid_ud=True)

    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): tl,
        (0, 1): tm,
        (0, 2): np.fliplr(tl),
        (1, 0): ml,
        (1, 2): np.fliplr(ml),
        (2, 0): bl,
        (2, 1): bm,
        (2, 2): np.fliplr(bl),
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


def _expanded_output(motif: np.ndarray, center: np.ndarray, axis: str) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            dst_br, dst_bc = _outer_slot(br, bc, axis)
            paste(expanded, src, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _point_reflection_output(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
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


def _build_motif(axis: str, majority: int, minor_colors: list[int]) -> np.ndarray:
    for _ in range(200):
        motif = (
            _assemble_horizontal_motif(majority, minor_colors)
            if axis == HORIZONTAL_AXIS
            else _assemble_vertical_motif(majority, minor_colors)
        )
        if _infer_axis(motif) != axis:
            continue
        return motif
    raise AssertionError("failed to build motif with a unique axis")


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:] or [random.choice([c for c in FOREGROUND_COLORS if c != majority])]
    axis = random.choice([HORIZONTAL_AXIS, VERTICAL_AXIS])

    motif = _build_motif(axis, majority, minor_colors)
    center = _gen_center_patch(motif, majority)
    expanded = _expanded_output(motif, center, axis)
    if np.array_equal(expanded, _point_reflection_output(motif, center)):
        return None

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
