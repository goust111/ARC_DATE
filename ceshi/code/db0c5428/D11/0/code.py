# Variant of db0c5428
# Dimension: D11 — 中心空洞的形状 (src)
# Change: 中心空洞恰好是一个3x3块，与其他块大小相同，位于3x3阵列正中
#   → 中心空洞为十字形（中心块+四个相邻块的中心像素）
#
# Design analysis
# ---
# source_task: none
# variant_spec:
#   a_concretization_and_trace:
#     input_object:
#       - The visible motif still occupies a 9x9 bbox made of eight 3x3 blocks
#         around a nominal center block.
#       - The new hole is a 13-cell cross: the whole 3x3 center block is erased,
#         and the center pixel of each orthogonally adjacent edge block is also
#         erased. The four corner blocks remain intact.
#       - Example with block size 3:
#         original input hole:
#           only rows 3..5, cols 3..5 are bg
#         D11 input hole:
#           rows 3..5, cols 3..5 are bg, plus motif-local cells
#           (1,4), (4,1), (4,7), (7,4) are also bg.
#       - Original solver copied the visible 3x3 edge blocks as-is, which was
#         harmless when those blocks were complete. In D11 that would propagate
#         the missing center pixels to the outer mirrored slots, so the solver
#         must repair the motif before copying.
#       - Concrete trace:
#         original top edge block had no missing pixels, so outer-top copy could
#         be taken directly from input.
#         D11 top edge block is missing its center pixel. The bottom-middle pixel
#         of that same block still faces the hole and remains visible. The
#         generator enforces top-center == bottom-middle (and the analogous rule
#         for all four edge blocks via symmetry), so the solver can restore the
#         missing edge-center from the still-visible inward midpoint, then use the
#         repaired block both in-place and in the outer mirrored slot.
#   b_complexity_calibration:
#     - The variant is visibly different because the input no longer has a clean
#       square cavity; it has four extra punctures on the cross arms.
#     - Reconstruction is still single-rule and low-branching: 8 boundary pixels
#       reconstruct the center block, and 4 edge-center repairs are tied to the
#       4 cardinal pixels of that reconstructed center block.
#     - To avoid ambiguity, the generator keeps the original D4 block symmetries
#       and adds one extra local equality in the edge seed block:
#       edge-center == inward midpoint. This is enough to make the arm repair
#       inferable without introducing unrelated mechanisms.
#   c_implicit_assumption_audit_for_main:
#     - bbox(g != bg): still valid because the motif bbox remains 9x9 even after
#       removing the four arm-center pixels.
#     - h == w and h % 3 == 0: still valid; D11 changes only internal erosion.
#     - Cropping edge blocks directly from input is no longer valid, because
#       those blocks are incomplete. The solver therefore reconstructs a healed
#       9x9 motif first and copies blocks from that healed motif.
#     - The center patch sampling offsets [-1, 1, 3] remain valid because D11
#       does not erase the eight inner-boundary pixels used by the original task.
#     - Majority-color center assignment remains valid because the input still
#       contains enough foreground to recover the majority color robustly.
#   d_placement_legality:
#     - The generator still places a single 9x9 motif with padding >= 3 so the
#       15x15 expanded output fits fully inside the grid.
#     - The input erosion only changes cells inside the 9x9 motif, so there is no
#       new overlap or out-of-bounds risk.
#   e_visual_observability:
#     - Across examples, the input shows a cross-shaped tear rather than a square
#       hole. The output visibly repairs both the center block and the four extra
#       punctures. This can be inferred from 2-4 examples without hidden metadata.
#   f_visual_cleanliness:
#     - No extra markers are added. The grid remains one motif on a clean
#       background, and the cross erosion is sparse and easy to localize.
#   g_color_inference:
#     - No color-id arithmetic or ordering is used.
#     - The repaired arm pixels are copied from structurally corresponding visible
#       pixels, not from a lookup by palette value.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


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


def _restore_cross_arms(motif: np.ndarray, center: np.ndarray, block: int) -> None:
    mid = block // 2
    motif[mid, block + mid] = int(center[0, mid])
    motif[block + mid, mid] = int(center[mid, 0])
    motif[block + mid, 2 * block + mid] = int(center[mid, 2])
    motif[2 * block + mid, block + mid] = int(center[2, mid])


def _healed_motif(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    motif = crop_bbox(
        g,
        (
            int(r0),
            int(c0),
            int(r0) + 3 * block - 1,
            int(c0) + 3 * block - 1,
        ),
    )
    center = _center_patch(g, r0, c0, block, bg)
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
    Expand a 3x3 block motif whose missing region is a cross, not a plain square.

    Input structure:
    - One 9x9 foreground motif sits on a teal background.
    - The motif is arranged as 3x3 blocks of size 3x3, but the center block is
      empty and the center pixel of each adjacent edge block is also removed.
    - The eight inner-facing boundary pixels around the cavity remain visible and
      still encode the missing center block.

    Rule:
    - First repair the cross-shaped damage inside the original 9x9 motif.
    - Rebuild the center 3x3 block from the eight visible boundary pixels, with
      its middle cell set to the motif's majority foreground color.
    - Restore the four erased edge-center pixels, then copy every non-center
      block of the repaired motif to its point-symmetric outer slot in the 5x5
      block layout, without rotating the copied block.

    Key insight:
    - The four extra missing arm pixels are not independent clues; each is tied
      to a still-visible inward midpoint that also determines one cardinal cell of
      the reconstructed center block. So the motif must be healed before any
      outward copying happens.
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

    motif = _healed_motif(g, int(r0), int(c0), block, bg)
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

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 15
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 7

# -- CFG: generation --
MAX_RETRIES = 300

# -- Symmetry orbits for 3x3 blocks --
DIAG_ORBITS = (
    ((0, 0),),
    ((1, 1),),
    ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)

# D11-specific edge seed orbits:
# still LR-symmetric, plus the block center matches the inward midpoint so the
# missing arm-center can be recovered from a visible pixel.
EDGE_ORBITS_D11 = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1), (2, 1)),
)

_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_orbit_options(orbits, forced_minor, target_count):
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
    return choices


def _major_orbits(orbits, forced_minor, target_count):
    choices = _major_orbit_options(orbits, forced_minor, target_count)
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
    b01, edge_minors = _seed_block(EDGE_ORBITS_D11, [(2, 1)], majority, edge_major)
    minor_orbits = corner_minors + edge_minors
    if len(minor_orbits) < len(minor_colors):
        return None
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    return b00, b01


def _assemble_frame_motif(b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
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
    patch[1, 1] = majority
    return patch


def _build_complete_motif(frame: np.ndarray, majority: int) -> np.ndarray:
    motif = frame.copy()
    center = _gen_center_patch(frame, majority)
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


def _expanded_region(motif: np.ndarray) -> np.ndarray:
    return _expanded_output(motif, BG)


def _feasible_pairs(minor_color_count: int) -> list[tuple[int, int]]:
    pairs = []
    for corner_major in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1):
        corner_options = _major_orbit_options(DIAG_ORBITS, [(2, 2)], corner_major)
        if not corner_options:
            continue
        for edge_major in range(EDGE_MAJOR_MIN, EDGE_MAJOR_MAX + 1):
            edge_options = _major_orbit_options(EDGE_ORBITS_D11, [(2, 1)], edge_major)
            if not edge_options:
                continue
            enough_minor_orbits = any(
                (len(DIAG_ORBITS) - len(csel)) + (len(EDGE_ORBITS_D11) - len(esel)) >= minor_color_count
                for csel in corner_options
                for esel in edge_options
            )
            if enough_minor_orbits:
                pairs.append((corner_major, edge_major))
    return pairs


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    pairs = [
        (corner_major, edge_major)
        for corner_major, edge_major in _feasible_pairs(len(minor_colors))
        if MAJORITY_TOTAL_MIN <= corner_major + edge_major <= MAJORITY_TOTAL_MAX
    ]
    if not pairs:
        return None
    corner_major, edge_major = random.choice(pairs)

    blocks = _make_blocks(majority, minor_colors, corner_major, edge_major)
    if blocks is None:
        return None
    b00, b01 = blocks

    frame = _assemble_frame_motif(b00, b01)
    complete_motif = _build_complete_motif(frame, majority)
    input_motif = _carve_cross_input(complete_motif)
    expanded = _expanded_region(complete_motif)

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
    raise AssertionError("failed to generate a valid db0c5428 D11 input")
