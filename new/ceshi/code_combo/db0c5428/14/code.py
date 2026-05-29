# Combo variant of db0c5428 (index 14)
# combo_spec: D5/0+D12/1
# Name: 空洞含噪声+简化中心填充
#
# Design analysis
# spec:
#   variant_core:
#     - The visible object is still a 9x9 motif made from eight 3x3 foreground
#       blocks arranged around a logical center block.
#     - The center block is no longer a clean teal hole: it contains 5 scattered
#       decoy pixels.
#     - The output still expands the motif to the same 5x5 block layout, but the
#       rebuilt center is now a solid 3x3 block of the surrounding ring's
#       majority foreground color.
#   behavior_trace:
#     - Concrete example:
#       1. Suppose the eight real blocks contain 36 pixels of color A, 32 pixels
#          of color B, and 4 pixels of color C.
#       2. The noisy center shows 5 scattered pixels, all color B.
#       3. A naive solver that counts every visible non-background pixel would see
#          A=36 and B=37, choose B, and fill the center with the wrong color.
#       4. The original task logic would also be wrong here, because its center
#          would copy the eight inward-facing boundary pixels into a structured
#          pattern instead of producing a solid block.
#       5. The fused rule ignores the noisy center completely, counts only the
#          eight surrounding blocks, picks color A, and fills all 9 center cells
#          with A.
#     - Difference from the original task:
#       The center block used to mix 8 boundary-derived pixels with 1 majority
#       pixel. Now the boundary-facing pixels remain visible in the input only as
#       distractors or residual structure; they are not used in the output center
#       at all.
#   complexity_calibration:
#     - Noise is confined to one 3x3 region with exactly 5 specks, so the visual
#       change is obvious but still readable.
#     - Exactly 3 foreground colors are used. This keeps color reasoning small
#       while allowing the noise color to overturn the global foreground count.
#     - The outer eight blocks retain the original D4-style repetition, so humans
#       still get repeated evidence for the true majority color.
#   implicit_assumption_review:
#     - `bbox(g != bg)` still works because the noisy cells stay inside the same
#       9x9 motif footprint; they do not enlarge the motif.
#     - The original `_major_foreground_color(g, bg)` assumption is invalid
#       because center noise can flip the global majority. The solver must count
#       only the eight non-center blocks.
#     - The original `_center_patch(...)` assumption that inward-facing pixels are
#       the center payload is also invalid. Those pixels may be non-majority and
#       are intentionally ignored by the fused rule.
#     - Copying the eight visible source blocks remains valid because the center
#       block is still never used as a source for the outer expansion.
#   placement_legality:
#     - The motif is placed with at least 3 cells of padding on each side, so the
#       15x15 expanded output always fits inside the 16-19 square canvas.
#     - Noise is injected only inside the middle 3x3 block, so it never overlaps
#       the real surrounding blocks or escapes the motif bbox.
#   visual_observability:
#     - Across examples, the input visibly differs from the original task because
#       the "hole" is speckled rather than empty.
#     - The output visibly differs from both the original task and the D12-only
#       variant because the center becomes a solid monochrome 3x3 patch instead of
#       an 8-neighbor pattern.
#     - The repeated ring blocks make the true majority color legible by eye, so
#       a human can infer that the center specks are junk and that the fill color
#       should come from the surrounding motif.
#   cleanliness:
#     - Only one kind of extra marking exists: sparse center noise.
#     - The rest of the motif stays symmetric and compact, avoiding clutter.
#   color_inference:
#     - No rule depends on numeric color ids.
#     - The only color logic is visible counting on the surrounding eight blocks:
#       the center is filled with their majority color, while the noisy center
#       pixels deliberately use the runner-up color.
#   solver_assumptions_after_change:
#     - `main()` assumes the foreground object is a square 3x3 block grid whose
#       center block may contain arbitrary distractor pixels. `generate_input()`
#       guarantees that structure.
#     - `main()` assumes the ring of eight blocks has a unique foreground majority
#       color. `generate_input()` enforces counts 36/32/4 on the real ring.
#     - `main()` assumes the output center is a solid color block. The generator
#       explicitly constructs training examples where the original boundary-based
#       center would differ from that solid block.

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


def _solid_center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    main = _ring_majority_color(g, r0, c0, block, bg)
    return np.full((block, block), main, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 block layout and replace a noisy center
    with a solid majority-colored block.

    Input structure:
    - A teal canvas contains one 9x9 motif arranged as a 3x3 grid of 3x3 blocks.
    - The middle 3x3 region is the logical missing block, but it is not cleanly
      empty: a few stray foreground pixels may appear there as noise.
    - The eight surrounding 3x3 blocks remain the real motif content and still
      determine the larger repeated pattern.

    Rule:
    - Keep the original 9x9 motif in place.
    - Copy each of the eight non-central 3x3 blocks to the opposite outer slot
      of the implied 5x5 block arrangement, without rotating it.
    - Rebuild the center as a solid 3x3 patch whose color is the majority
      foreground color across the eight surrounding blocks, ignoring any noisy
      pixels already present in the center region.

    Key insight: The center block is identified by its fixed block position, not
    by being blank. Any colored specks inside that 3x3 region are distractors;
    the dependable evidence is the repeated ring of eight surrounding blocks.
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

    paste(out, _solid_center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
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

# -- CFG: surrounding-ring counts --
# The two seed blocks contain 18 cells total. With D4 expansion, counts multiply
# by 4 across the 8 visible blocks. Setting majority_total=9 yields ring counts
# 36/32/4 after minor-color assignment, which lets 5 center-noise pixels flip the
# naive global majority from `majority` to `runner_up`.
MAJORITY_TOTAL = 9
CORNER_EDGE_MAJOR_PAIRS = [(4, 5), (5, 4)]

# -- CFG: center noise --
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


def _solid_center_from_majority(majority: int) -> np.ndarray:
    return np.full((BLOCK, BLOCK), majority, dtype=int)


def _boundary_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
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
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _ring_color_counts(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> dict[int, int]:
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
    fg = np.concatenate(vals)
    fg = fg[fg != int(bg)]
    uniq, counts = np.unique(fg, return_counts=True)
    return {int(color): int(count) for color, count in zip(uniq, counts)}


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    majority, runner_up, accent = random.sample(FOREGROUND_COLORS, COLOR_COUNT)

    b00, b01 = _make_blocks(majority, runner_up, accent)
    motif = _assemble_motif(b00, b01)
    center = _solid_center_from_majority(majority)
    boundary_center = _boundary_center_patch(motif, majority)
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
    ring_counts = _ring_color_counts(inp, r0, c0, BLOCK, BG)
    if sorted(ring_counts.values(), reverse=True)[:3] != [36, 32, 4]:
        return None
    if np.array_equal(boundary_center, center):
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
