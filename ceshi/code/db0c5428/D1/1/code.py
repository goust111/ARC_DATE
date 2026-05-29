# Variant of db0c5428
# Dimension: D1 — 块到外部槽位的映射规则 (match_cond)
# Change: 每个非中心3x3块被复制到以中心为对称点的对侧外部槽位，即block(r,c)映射到5x5网格的(4-2r, 4-2c)... → 镜像映射（仅沿一个轴）

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the D1 variant
# a_concretization:
#   - Input still contains one 9x9 motif made from eight visible 3x3 blocks around
#     a teal 3x3 hole. The hole border still encodes the missing center block.
#   - The D1 change is made visible by changing only the block-to-outer-slot map:
#     if the motif is symmetric across the horizontal axis, top/bottom rows copy to
#     the opposite outer strip while the two middle side blocks go to the left/right
#     outer midpoints; if the motif is symmetric across the vertical axis, the left/
#     right columns copy to the opposite outer strip while the two middle vertical
#     blocks go to the top/bottom outer midpoints.
#   - Concrete trace: in the horizontal-axis case, block(0,0) used to go to outer
#     slot (4,4); now it goes to (4,1). The block is still copied without rotation,
#     but only its row side is mirrored. This is the core observable difference.
# b_complexity:
#   - The motif is forced to have exactly one visible axial symmetry, never both.
#     That gives the solver a small discrete choice (horizontal vs vertical) with
#     repeated support across the whole 9x9 object, instead of an arbitrary mapping.
#   - Blocks are internally asymmetric often enough that the new mapping differs
#     visually from point symmetry; otherwise this D1 change would be hard to see.
# c_hidden_assumptions_review:
#   - Original main() assumed both block coordinates were inverted together via
#     (4-2*br, 4-2*bc). That silently relies on point symmetry. In this variant
#     only one axis may be mirrored, so the destination formula must be replaced by
#     an axis-aware lookup.
#   - Axis inference needs a new guarantee: the motif must have exactly one axial
#     symmetry. The generator enforces xor(flipud, fliplr) on the full 9x9 motif.
# d_placement_legality:
#   - The expanded output is still a 15x15 region around the motif, so the original
#     placement margin of one block on every side remains sufficient.
#   - Outer slots are chosen from that same 15x15 frame, so no overlap/out-of-bounds
#     cases are introduced beyond the original task.
# e_visual_observability:
#   - The axis choice is visible directly in the input because the motif is either
#     flipud-symmetric or fliplr-symmetric, but not both.
#   - The output reinforces that cue: one mode produces top/bottom outer strips,
#     the other produces left/right outer strips.
# f_cleanliness:
#   - Only one motif and one expanded frame exist. There are no new markers or
#     overlays, so the scene remains as clean as the original task.
# g_color_inference:
#   - Colors are unchanged structurally: one foreground color remains the majority,
#     minor colors only provide asymmetric detail. No hidden ordering by color id
#     is used anywhere.


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
    Expand a 3x3 block motif into a 5x5 frame using a single-axis mirror rule.

    Input structure:
    - Teal background surrounds one 9x9 foreground motif.
    - The motif is a 3x3 array of 3x3 blocks with a teal center hole.
    - The visible motif has exactly one axial symmetry, and the eight pixels
      around the hole still encode the missing center block.

    Rule:
    - Keep the original 9x9 motif in place.
    - Rebuild the center block from the hole border, using the motif's majority
      foreground color at the center pixel.
    - Copy each non-center 3x3 block to the outer frame by mirroring it across
      the motif's only symmetry axis, without rotating the copied block.

    Key insight:
    The missing center is reconstructed exactly as before, but the expansion is
    no longer point-symmetric: the input's lone mirror axis tells you which
    outer strip receives each copied block.
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
    raise AssertionError("failed to generate a valid db0c5428 input")
