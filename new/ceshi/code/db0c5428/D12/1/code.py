# Variant of db0c5428
# Dimension: D12 — 背景色与空洞的区分方式 (meta)
# Change: 空洞区域填充与网格背景相同的颜色8，空洞通过其在图案内部的位置和被非背景像素包围来识别... → 空洞内包含噪声像素
#
# Design analysis
# spec:
#   variant_core:
#     - The 9x9 motif is still a 3x3 array of 3x3 blocks with the middle block
#       being the logical hole, but the input no longer shows that hole as a
#       clean 3x3 teal patch.
#     - Instead, the center 3x3 contains exactly 5 scattered non-background
#       pixels. They are decoys: the output ignores them and reconstructs the
#       true center block exactly as in the original task.
#   behavior_trace:
#     - Original pipeline:
#       1. Treat every non-background pixel inside the 9x9 motif as meaningful.
#       2. Use the global foreground majority as the center pixel color.
#       3. Read the eight inward-facing boundary pixels to fill the other center
#          cells.
#     - Variant pipeline:
#       1. Keep using the inward-facing boundary pixels, because they remain the
#          reliable encoding of the missing block.
#       2. Treat the center 3x3 as a special hole region identified by block
#          position, not by being uniformly teal.
#       3. Compute the center pixel color from only the eight surrounding blocks,
#          excluding the noisy center cells.
#     - Concrete effect:
#       We generate surrounding blocks so that their true majority color beats the
#       runner-up by only 4 visible pixels. Then the 5 center-noise pixels all use
#       the runner-up color. A solver that still counts the noisy hole globally
#       would choose the wrong center color, so D12 is behaviorally active rather
#       than cosmetic.
#   complexity_calibration:
#     - Noise is limited to one 3x3 region and one color per example, so the task
#       stays readable.
#     - The surrounding motif keeps the original D4-style structure, giving strong
#       repeated evidence about which pixels belong to the real pattern.
#     - Exactly 3 foreground colors are used so the runner-up can be close enough
#       to the majority for the noise to matter without making the image chaotic.
#   implicit_assumption_review:
#     - `bbox(g != bg)` still works because the noise stays inside the already
#       occupied 9x9 motif, so the outer bounding box does not change.
#     - The old `_major_foreground_color(g, bg)` is no longer valid, because it
#       would count center noise. The new solver must measure majority only on the
#       eight non-center blocks.
#     - Copying the eight visible blocks is still valid because the center block is
#       never used as a source block.
#   placement_legality:
#     - The motif is placed with the same 3-cell padding as before, so the 15x15
#       expanded output still fits fully inside the canvas.
#     - Noise is injected only inside the center 3x3; it never overlaps the eight
#       real blocks or escapes the motif bbox.
#   visual_observability:
#     - Across examples, the hole visibly contains colored specks instead of a
#       clean teal void.
#     - The output replaces those specks with a structured 3x3 block that matches
#       the surrounding boundary encoding, so a human can infer that the specks are
#       junk rather than part of the intended center.
#   cleanliness:
#     - Noise is restricted to a single compact region and does not stack with any
#       other markers.
#     - The outer pattern remains symmetric and uncluttered, preserving clear
#       figure/ground separation.
#   color_inference:
#     - No hidden ordering by color id is used.
#     - The only color logic is visible counting: the center cell takes the true
#       majority color of the surrounding eight blocks, while the noisy center
#       pixels deliberately do not follow that rule.
#   solver_assumptions_after_change:
#     - `main()` now assumes the motif is still exactly a 3x3 block grid and that
#       the only noisy area is the center 3x3. `generate_input()` guarantees this.
#     - `main()` assumes the eight surrounding blocks contain at least one
#       foreground color and define a unique majority after excluding the center.
#       `generate_input()` enforces visible counts 36/32/4, so that assumption is
#       always satisfied.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


def _ring_majority_color(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> int:
    vals = []
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            blk = g[
                int(r0) + br * block : int(r0) + (br + 1) * block,
                int(c0) + bc * block : int(c0) + (bc + 1) * block,
            ]
            vals.append(blk.reshape(-1))
    ring = np.concatenate(vals)
    fg = ring[ring != int(bg)]
    uniq, counts = np.unique(fg, return_counts=True)
    assert uniq.size, "surrounding blocks must contain foreground colors"
    return int(uniq[int(np.argmax(counts))])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    main = _ring_majority_color(g, r0, c0, block, bg)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 block motif even when the hole is noisy.

    Input structure:
    - A teal canvas contains one 9x9 motif made from eight 3x3 foreground blocks
      around a central 3x3 hole.
    - The hole is not cleanly empty anymore: it can contain a few scattered
      foreground-colored decoy pixels.
    - The eight pixels facing the hole from the surrounding blocks still encode
      the missing center block.

    Rule:
    - Keep the original 9x9 motif in place.
    - Copy each non-central 3x3 block to the opposite outer slot of the implied
      5x5 block layout, without rotating it.
    - Rebuild the center block from the inward-facing boundary pixels, and choose
      its middle cell from the majority color of the eight real surrounding
      blocks rather than from the noisy hole contents.

    Key insight: The center 3x3 is recognized by its fixed position inside the
    3x3 block motif, so any stray pixels inside that block are distractors; the
    trustworthy information is the ring of eight surrounding blocks.
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
            paste(out, src, dst_r, dst_c)

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
COLOR_COUNT = 3

# -- CFG: majority / runner-up balance across the two seed blocks combined --
MAJORITY_TOTAL = 9
CORNER_EDGE_MAJOR_PAIRS = [(4, 5), (5, 4)]

# -- CFG: noise --
NOISE_COUNT = 5
NOISE_PATTERNS = (
    ((0, 0), (0, 1), (1, 2), (2, 0), (2, 2)),
    ((0, 1), (0, 2), (1, 0), (2, 0), (2, 2)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (2, 2)),
    ((0, 0), (0, 2), (1, 2), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 2), (2, 0), (2, 1)),
    ((0, 2), (1, 0), (1, 1), (2, 0), (2, 2)),
)

# -- CFG: generation --
MAX_RETRIES = 300
BLOCK_BUILD_RETRIES = 80

# -- Symmetry orbits for 3x3 blocks --
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

_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_orbit_choices(orbits, forced_minor, target_count):
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


def _seed_block(orbits, forced_minor, majority, target_count):
    choices = _major_orbit_choices(orbits, forced_minor, target_count)
    random.shuffle(choices)
    for major_idxs in choices:
        block = np.full((BLOCK, BLOCK), -1, dtype=int)
        minor_orbits = []
        for idx, orbit in enumerate(orbits):
            if idx in major_idxs:
                for r, c in orbit:
                    block[r, c] = majority
            else:
                minor_orbits.append(orbit)
        if any(len(orbit) == 1 for orbit in minor_orbits):
            return block, minor_orbits
    raise AssertionError("failed to leave a singleton minor orbit")


def _fill_orbit(block: np.ndarray, orbit, color: int) -> None:
    for r, c in orbit:
        block[r, c] = color


def _make_blocks(majority, runner_up, accent):
    for _ in range(BLOCK_BUILD_RETRIES):
        corner_major, edge_major = random.choice(CORNER_EDGE_MAJOR_PAIRS)
        b00, corner_minors = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
        b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
        minor_refs = [(b00, orbit) for orbit in corner_minors] + [(b01, orbit) for orbit in edge_minors]
        singleton_idxs = [idx for idx, (_block, orbit) in enumerate(minor_refs) if len(orbit) == 1]
        if not singleton_idxs:
            continue
        accent_idx = random.choice(singleton_idxs)
        for idx, (block, orbit) in enumerate(minor_refs):
            _fill_orbit(block, orbit, accent if idx == accent_idx else runner_up)
        if np.all(b00 != -1) and np.all(b01 != -1):
            return b00, b01
    raise AssertionError("failed to build seed blocks")


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


def _inject_hole_noise(motif, noise_color):
    noisy = motif.copy()
    pattern = random.choice(NOISE_PATTERNS)
    for dr, dc in pattern:
        noisy[BLOCK + dr, BLOCK + dc] = noise_color
    return noisy


def _global_foreground_majority(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    majority, runner_up, accent = random.sample(FOREGROUND_COLORS, COLOR_COUNT)

    b00, b01 = _make_blocks(majority, runner_up, accent)
    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    expanded = _expanded_output(motif, center)
    noisy_motif = _inject_hole_noise(motif, runner_up)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, noisy_motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    hole = inp[r0 + BLOCK : r0 + 2 * BLOCK, c0 + BLOCK : c0 + 2 * BLOCK]
    if np.count_nonzero(hole != BG) != NOISE_COUNT:
        return None
    if _global_foreground_majority(inp, BG) != runner_up:
        return None
    if _ring_majority_color(inp, r0, c0, BLOCK, BG) != majority:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D12 input")
