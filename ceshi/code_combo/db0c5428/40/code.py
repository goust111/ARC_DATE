# Combo variant of db0c5428 (index 40)
# combo_spec: D2/0+D7/0+D8/0
# Name: 4x4块阵列+可变块大小+自适应输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concretization:
# - The visible motif is no longer a 3x3 array of fixed 3x3 blocks. It is a
#   4x4 array of square blocks, with block side `k` inferred from the bbox:
#   `k in {2,3,4}` and the foreground bbox is always `(4k) x (4k)`.
# - The middle is now a 2x2 block hole, so the input contains 12 visible blocks
#   and a blank `(2k) x (2k)` cavity.
# - The output is a full 6x6 block arrangement on a resized canvas. The filled
#   4x4 motif is placed at block coordinates `[1:5, 1:5]`, and each visible
#   input block is copied to its point-symmetric outer slot on the surrounding
#   ring. The outer ring uses 12 slots, not all 20 border positions.
#
# behavior trace:
# - Example: let `k=2`, so the input motif bbox is `8x8` and the missing center
#   is `4x4`. If the motif starts at `(1, 0)`, the intended expanded region is
#   `12x12` with top-left `(-1, -2)`.
# - Original logic depended on a `9x9` motif, a single `3x3` hole and a fixed
#   in-place output, so it would either assert or clip immediately.
# - New logic infers `k=2`, rebuilds four center blocks inside the `4x4` hole,
#   renders the full `6x6` block layout, then enlarges the canvas to the union
#   of the old input area and the new expanded area.
#
# complexity calibration:
# - The combo is visibly different from the source task in three independent
#   ways that still reinforce one another: the motif is larger (4x4), the hole
#   is larger (2x2 blocks), and the block size varies across samples.
# - Block size is restricted to `{2,3,4}` so a solver can infer it from the
#   bbox and hole geometry without facing too many regimes.
# - The generator keeps a small family of placements near edges and corners so
#   canvas growth is obvious across examples instead of being a rare side-case.
#
# implicit-assumption audit of the original main():
# - `h % 3 == 0` and `block == 3` are invalid now; the fused variant needs
#   `h % 4 == 0` and must derive `block = h // 4`.
# - A single missing center block no longer exists. Any logic that assumes one
#   `block x block` hole must be replaced by a `2block x 2block` reconstruction.
# - `out = g.copy()` assumed the output shape equals the input shape. D7 breaks
#   that assumption, so the expanded layout must be rendered on a separate
#   canvas and then embedded into the minimal union canvas.
# - The original outer-slot formula only works for the 3x3 -> 5x5 case. The
#   4x4 -> 6x6 case uses a different sparse ring mapping: rows/cols
#   `0,1,4,5` receive the copied blocks via indices `[5,4,1,0]`.
#
# placement legality:
# - The input still contains one connected foreground motif on a uniform
#   background, so bbox detection remains unambiguous.
# - The generator always places the `(4k) x (4k)` motif fully inside the input
#   grid, and the solver computes the resized output from the union bbox, so no
#   paste can go out of bounds.
# - The copied outer blocks never overlap each other or the central 4x4 motif:
#   top/bottom source rows map to output rows `5/0`, while the two side rows
#   map to rows `4/1` but only at columns `0/5`, which keeps the border sparse.
#
# visual observability and neatness:
# - The outer layout is structured and symmetric:
#     X X . . X X
#     X X X X X X
#     . X X X X .
#     . X X X X .
#     X X X X X X
#     X X . . X X
#   where each X denotes one block slot.
# - The four reconstructed center blocks are mostly majority-colored, but their
#   outward-facing rims echo the neighboring blocks. This is visible for all
#   `k` and gives a human a local cue for how the hole is filled.
# - No rule depends on numeric color ids. Colors only serve as visible block
#   content and as the majority color for the center interiors.
#
# post-change assumption audit:
# - `main()` assumes the non-background bbox is a square `4k x 4k` motif and
#   that the center `2x2` block region is the hole. `generate_input()` enforces
#   both properties directly.
# - `main()` assumes the reconstructed center should use the majority foreground
#   color for its interiors. The generator makes the majority color dominant
#   across the motif so this is stable.
# - `main()` assumes the sparse outer-ring mapping `[5,4,1,0]`. The generator's
#   expected-output renderer uses the same geometry and validates every sample
#   against the solver before returning it.


BLOCK_CHOICES = (2, 3, 4)
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
GRID_EXTRA_MIN, GRID_EXTRA_MAX = 1, 4
MAX_RETRIES = 400
PLACEMENT_MODES = ("top", "bottom", "left", "right", "tl", "tr", "bl", "br")
OUTER_SLOT_INDEX = (5, 4, 1, 0)


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _diag_orbits(block: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[int, int]] = set()
    orbits = []
    for r in range(block):
        for c in range(block):
            if (r, c) in seen:
                continue
            orbit = tuple(sorted({(r, c), (c, r)}))
            seen.update(orbit)
            orbits.append(orbit)
    return tuple(orbits)


def _lr_orbits(block: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[int, int]] = set()
    orbits = []
    for r in range(block):
        for c in range(block):
            if (r, c) in seen:
                continue
            orbit = tuple(sorted({(r, c), (r, block - 1 - c)}))
            seen.update(orbit)
            orbits.append(orbit)
    return tuple(orbits)


def _major_orbits(
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    blocked: set[tuple[int, int]],
    min_major: int,
    max_major: int,
) -> set[int]:
    usable = [
        idx for idx, orbit in enumerate(orbits)
        if all(cell not in blocked for cell in orbit)
    ]
    choices: list[set[int]] = []
    for mask in range(1 << len(usable)):
        chosen: set[int] = set()
        total = 0
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                chosen.add(idx)
                total += len(orbits[idx])
        if min_major <= total <= max_major and len(orbits) - len(chosen) >= 1:
            choices.append(chosen)
    assert choices, "failed to find a feasible majority-orbit subset"
    return random.choice(choices)


def _seed_block(
    block: int,
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    blocked_major: set[tuple[int, int]],
    majority: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, tuple[tuple[int, int], ...]]]]:
    total = block * block
    min_major = max(1, total // 2)
    max_major = total - len(blocked_major)
    arr = np.full((block, block), -1, dtype=int)
    major_idxs = _major_orbits(orbits, blocked_major, min_major, max_major)
    minor_orbits = []
    for idx, orbit in enumerate(orbits):
        if idx in major_idxs:
            for r, c in orbit:
                arr[r, c] = majority
        else:
            minor_orbits.append((arr, orbit))
    return arr, minor_orbits


def _assign_minor_colors(
    minor_orbits: list[tuple[np.ndarray, tuple[tuple[int, int], ...]]],
    colors: list[int],
) -> None:
    random.shuffle(minor_orbits)
    guaranteed = list(colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))
    ]
    for (block_arr, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            block_arr[r, c] = int(color)


def _make_seed_blocks(block: int, majority: int, minor_colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    corner_blocked = {(block - 1, block - 1)}
    edge_blocked = {(block - 1, c) for c in range(block)}
    corner, corner_minors = _seed_block(
        block, _diag_orbits(block), corner_blocked, majority
    )
    edge, edge_minors = _seed_block(
        block, _lr_orbits(block), edge_blocked, majority
    )
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(corner != -1) and np.all(edge != -1)
    return corner, edge


def _assemble_motif(corner: np.ndarray, edge: np.ndarray, bg: int) -> np.ndarray:
    block = int(corner.shape[0])
    motif = np.full((block * 4, block * 4), int(bg), dtype=int)
    blocks = {
        (0, 0): corner,
        (0, 1): edge,
        (0, 2): np.fliplr(edge),
        (0, 3): np.fliplr(corner),
        (1, 0): np.rot90(np.fliplr(edge), 1),
        (1, 3): np.rot90(edge, -1),
        (2, 0): np.rot90(edge, 1),
        (2, 3): np.rot90(np.fliplr(edge), -1),
        (3, 0): np.flipud(corner),
        (3, 1): np.flipud(edge),
        (3, 2): np.rot90(edge, 2),
        (3, 3): np.rot90(corner, 2),
    }
    for (br, bc), patch in blocks.items():
        paste(motif, patch, br * block, bc * block)
    return motif


def _center_macro_patch(motif: np.ndarray, block: int, bg: int) -> np.ndarray:
    majority = _major_foreground_color(motif, bg)
    center = np.full((2 * block, 2 * block), majority, dtype=int)

    center[0, :block] = motif[block - 1, block : 2 * block]
    center[:block, 0] = motif[block : 2 * block, block - 1]
    center[0, 0] = motif[block - 1, block - 1]

    center[0, block:] = motif[block - 1, 2 * block : 3 * block]
    center[:block, -1] = motif[block : 2 * block, 3 * block]
    center[0, -1] = motif[block - 1, 3 * block]

    center[-1, :block] = motif[3 * block, block : 2 * block]
    center[block:, 0] = motif[2 * block : 3 * block, block - 1]
    center[-1, 0] = motif[3 * block, block - 1]

    center[-1, block:] = motif[3 * block, 2 * block : 3 * block]
    center[block:, -1] = motif[2 * block : 3 * block, 3 * block]
    center[-1, -1] = motif[3 * block, 3 * block]
    return center


def _filled_motif(motif: np.ndarray, block: int, bg: int) -> np.ndarray:
    filled = motif.copy()
    paste(filled, _center_macro_patch(motif, block, bg), block, block)
    return filled


def _expanded_region_from_motif(motif: np.ndarray, block: int, bg: int) -> np.ndarray:
    expanded = np.full((block * 6, block * 6), int(bg), dtype=int)
    filled = _filled_motif(motif, block, bg)
    paste(expanded, filled, block, block)

    for br in range(4):
        for bc in range(4):
            if 1 <= br <= 2 and 1 <= bc <= 2:
                continue
            src = motif[
                br * block : (br + 1) * block,
                bc * block : (bc + 1) * block,
            ]
            paste(
                expanded,
                src,
                OUTER_SLOT_INDEX[br] * block,
                OUTER_SLOT_INDEX[bc] * block,
            )
    return expanded


def _render_on_resized_canvas(
    original: np.ndarray,
    expanded: np.ndarray,
    expanded_r0: int,
    expanded_c0: int,
    bg: int,
) -> np.ndarray:
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


def _motif_info(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int, int, int]:
    motif_bbox = bbox(grid != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 4 == 0, "motif must be a square 4x4 block grid"
    block = h // 4
    assert block >= 2, "block size must be at least 2"
    motif = crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1)))
    return motif, int(r0), int(c0), int(block)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 4x4 block motif with a 2x2 hole into its full 6x6 frame.

    Input structure:
    - A uniform background contains one square foreground motif whose bbox is a
      4x4 array of equal-sized square blocks.
    - The middle 2x2 blocks are blank background, so the visible motif has
      twelve blocks surrounding one larger central hole.
    - The block side length is not fixed; it must be inferred from the bbox.

    Rule:
    - Infer the block size, rebuild the missing 2x2 center from the
      inward-facing boundary strips of the surrounding blocks, and use the
      motif's majority foreground color for the remaining center interiors.
    - Keep the filled 4x4 motif as the middle of a larger 6x6 block layout.
    - Copy each visible non-hole block to its point-symmetric outer slot on the
      sparse border ring, without rotating the copied block, and enlarge the
      output canvas as needed to contain the whole arrangement.

    Key insight: The observed motif is the inner window of a larger symmetric
    block construction, but both the block size and the output canvas must be
    inferred before the missing center and the outer ring can be rendered.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif, r0, c0, block = _motif_info(g, bg)
    expanded = _expanded_region_from_motif(motif, block, bg)
    return _render_on_resized_canvas(g, expanded, r0 - block, c0 - block, bg)


def _sample_location(n: int, motif_size: int, block: int) -> tuple[int, int]:
    max_r0 = int(n - motif_size)
    max_c0 = int(n - motif_size)
    low_r = list(range(0, min(block, max_r0 + 1)))
    low_c = list(range(0, min(block, max_c0 + 1)))
    high_r = list(range(max(0, max_r0 - block + 1), max_r0 + 1))
    high_c = list(range(max(0, max_c0 - block + 1), max_c0 + 1))
    mid_r = list(range(min(block, max_r0 + 1), max(0, max_r0 - block + 1)))
    mid_c = list(range(min(block, max_c0 + 1), max(0, max_c0 - block + 1)))

    mode = random.choice(PLACEMENT_MODES)
    if mode == "top" and mid_c:
        return random.choice(low_r), random.choice(mid_c)
    if mode == "bottom" and mid_c:
        return random.choice(high_r), random.choice(mid_c)
    if mode == "left" and mid_r:
        return random.choice(mid_r), random.choice(low_c)
    if mode == "right" and mid_r:
        return random.choice(mid_r), random.choice(high_c)
    if mode == "tl":
        return random.choice(low_r), random.choice(low_c)
    if mode == "tr":
        return random.choice(low_r), random.choice(high_c)
    if mode == "bl":
        return random.choice(high_r), random.choice(low_c)
    return random.choice(high_r), random.choice(high_c)


def _expected_output(inp: np.ndarray, motif: np.ndarray, r0: int, c0: int, block: int) -> np.ndarray:
    expanded = _expanded_region_from_motif(motif, block, BG)
    return _render_on_resized_canvas(inp, expanded, r0 - block, c0 - block, BG)


def _gen_attempt() -> np.ndarray | None:
    try:
        block = random.choice(BLOCK_CHOICES)
        motif_size = 4 * block
        n_min = max(motif_size + GRID_EXTRA_MIN, 5 * block)
        n_max = max(n_min, 5 * block + (GRID_EXTRA_MAX - GRID_EXTRA_MIN))
        n = random.randint(n_min, n_max)

        color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
        colors = random.sample(FOREGROUND_COLORS, color_count)
        majority = colors[0]
        minor_colors = colors[1:]

        corner, edge = _make_seed_blocks(block, majority, minor_colors)
        motif = _assemble_motif(corner, edge, BG)
        filled = _filled_motif(motif, block, BG)

        vals, counts = np.unique(filled[filled != BG], return_counts=True)
        if int(vals[int(np.argmax(counts))]) != majority:
            return None

        r0, c0 = _sample_location(n, motif_size, block)
        inp = np.full((n, n), BG, dtype=int)
        paste(inp, motif, r0, c0)
        out = _expected_output(inp, motif, r0, c0, block)
    except (AssertionError, ValueError):
        return None

    if out.shape == inp.shape:
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
