# Combo variant of db0c5428 (index 19)
# combo_spec: D5/0+D8/2
# Name: 偶数块大小+简化中心填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, get_bg, paste


# Design analysis for the fused variant:
# - Concrete behavior change: the original task used a 3x3 missing block whose
#   eight outer cells were copied from the eight facing pixels around the hole,
#   with only the center cell chosen by majority color. In this variant the block
#   size is even (2x2 or 4x4), so there is no single center cell. The middle
#   block is therefore rebuilt as one solid majority-color square, while the
#   point-symmetric outer copying rule is unchanged.
# - Small example: with BLOCK=2 the visible motif is 6x6, the output expansion
#   is 10x10, and the missing 2x2 center becomes [[M, M], [M, M]] even if the
#   cells facing the hole contain other colors. This makes the D5/0 change
#   visible instead of quietly mimicking the original boundary-sampling rule.
# - Complexity calibration: only two even block sizes are used, preserving the
#   same 3x3-to-5x5 block-level geometry and D4-derived layout. This keeps the
#   task inferable from a few examples while making the even-block change obvious.
# - Hidden assumptions updated from the original solver:
#   1. `block == 3` is no longer valid; the solver now accepts any even square
#      block size inferred from `bbox // 3`.
#   2. Sampling center pixels by offsets around the hole is invalid for even
#      blocks, so `_solid_center_patch` depends only on the foreground majority.
#   3. The solver still assumes a single square motif on uniform background and
#      that its side length is divisible by 3; the generator enforces both.
# - Placement legality: the generator picks `r0, c0` in `[block, n - 4*block]`,
#   so the 5x5 expanded region fits exactly inside the canvas with no overlap or
#   out-of-bounds writes.
# - Visual observability and non-degeneracy: the generator forces non-majority
#   colors on cells facing the hole, so the center block's solid fill visibly
#   disagrees with nearby boundary cues. This prevents the task from collapsing
#   into the original "copy boundary pixels into the center" behavior.


BLOCK_OPTIONS = (2, 4)
GRID_SIZE_BY_BLOCK = {
    2: (14, 18),
    4: (22, 26),
}
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
MAJORITY_SHARE_MIN, MAJORITY_SHARE_MAX = 0.55, 0.80
MAX_RETRIES = 500
BLOCK_BUILD_RETRIES = 400


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _solid_center_patch(g: np.ndarray, bg: int, block: int) -> np.ndarray:
    major = _major_foreground_color(g, bg)
    return np.full((block, block), major, dtype=int)


def _diag_orbits(block: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
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
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(block):
        for c in range(block):
            if (r, c) in seen:
                continue
            orbit = tuple(sorted({(r, c), (r, block - 1 - c)}))
            seen.update(orbit)
            orbits.append(orbit)
    return tuple(orbits)


def _candidate_major_sets(
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    forced_minor: set[tuple[int, int]],
) -> list[set[int]]:
    usable = [
        (idx, len(orbit))
        for idx, orbit in enumerate(orbits)
        if all(coord not in forced_minor for coord in orbit)
    ]
    choices: list[set[int]] = []
    for mask in range(1 << len(usable)):
        selected: set[int] = set()
        total = 0
        for bit, (idx, orbit_size) in enumerate(usable):
            if mask & (1 << bit):
                selected.add(idx)
                total += orbit_size
        if total == 0 or len(selected) == len(orbits):
            continue
        choices.append(selected)
    assert choices, "expected at least one majority-orbit choice"
    return choices


def _seed_block(
    block: int,
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    major_idxs: set[int],
    majority: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, tuple[tuple[int, int], ...]]]]:
    arr = np.full((block, block), -1, dtype=int)
    minor_orbits: list[tuple[np.ndarray, tuple[tuple[int, int], ...]]] = []
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
    assignments = list(colors)
    if len(assignments) < len(minor_orbits):
        assignments.extend(random.choice(colors) for _ in range(len(minor_orbits) - len(assignments)))
    random.shuffle(assignments)
    for (arr, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            arr[r, c] = color


def _assemble_motif(block: int, b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
    motif = np.full((block * 3, block * 3), BG, dtype=int)
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
        paste(motif, blk, br * block, bc * block)
    return motif


def _hole_facing_values(motif: np.ndarray, block: int) -> np.ndarray:
    rows = [
        motif[block - 1, block : 2 * block],
        motif[2 * block, block : 2 * block],
    ]
    cols = [
        motif[block : 2 * block, block - 1],
        motif[block : 2 * block, 2 * block],
    ]
    corners = np.array(
        [
            motif[block - 1, block - 1],
            motif[block - 1, 2 * block],
            motif[2 * block, block - 1],
            motif[2 * block, 2 * block],
        ],
        dtype=int,
    )
    return np.concatenate([rows[0], rows[1], cols[0], cols[1], corners])


def _foreground_count_map(g: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(g[g != BG], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _is_unique_majority(motif: np.ndarray, majority: int) -> bool:
    counts = _foreground_count_map(motif)
    assert majority in counts
    major_count = counts[majority]
    others = [count for color, count in counts.items() if color != majority]
    return all(major_count > count for count in others)


def _make_blocks(block: int, majority: int, minor_colors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    corner_orbits = _diag_orbits(block)
    edge_orbits = _lr_orbits(block)
    corner_forced = {(block - 1, block - 1)}
    edge_forced = {(block - 1, block // 2 - 1), (block - 1, block // 2)}
    corner_choices = _candidate_major_sets(corner_orbits, corner_forced)
    edge_choices = _candidate_major_sets(edge_orbits, edge_forced)

    for _ in range(BLOCK_BUILD_RETRIES):
        b00_major = random.choice(corner_choices)
        b01_major = random.choice(edge_choices)
        b00, corner_minors = _seed_block(block, corner_orbits, b00_major, majority)
        b01, edge_minors = _seed_block(block, edge_orbits, b01_major, majority)
        minor_orbits = corner_minors + edge_minors
        if len(minor_orbits) < len(minor_colors):
            continue
        _assign_minor_colors(minor_orbits, minor_colors)
        if np.array_equal(b00, b01):
            continue
        motif = _assemble_motif(block, b00, b01)
        if not _is_unique_majority(motif, majority):
            continue
        counts = _foreground_count_map(motif)
        share = counts[majority] / sum(counts.values())
        if not (MAJORITY_SHARE_MIN <= share <= MAJORITY_SHARE_MAX):
            continue
        facing = _hole_facing_values(motif, block)
        if np.all(facing == majority):
            continue
        return b00, b01
    raise AssertionError("failed to build symmetric blocks for fused variant")


def _expanded_output(motif: np.ndarray, block: int, center: np.ndarray) -> np.ndarray:
    expanded = np.full((block * 5, block * 5), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, block, block)
    paste(expanded, output_motif, block, block)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
            paste(expanded, src, (4 - 2 * br) * block, (4 - 2 * bc) * block)
    return expanded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into its 5x5 point-symmetric envelope, with an
    even-sized missing center rebuilt as one solid majority-color block.

    Input structure:
    - A single square foreground motif sits on a uniform teal background.
    - The motif is a 3x3 arrangement of equal-sized even blocks; generated
      cases use 2x2 or 4x4 blocks.
    - The middle block is entirely background, while the other eight blocks
      already show the corner/edge pattern that must be extended outward.

    Rule:
    - Keep the visible 3x3 motif in place.
    - Copy each non-center block to the point-symmetric outer slot of the
      implied 5x5 block layout, without rotating or recoloring it.
    - Fill the missing center block uniformly with the most frequent
      non-background color found in the motif.

    Key insight:
    - The outer expansion is still determined purely at block level, but the
      old "recover a central pixel from surrounding facing pixels" idea no
      longer applies once blocks have even side length and no unique center.
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
    assert block % 2 == 0 and block >= 2, "fused variant expects even block size"

    out = g.copy()
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = g[
                int(r0) + br * block : int(r0) + (br + 1) * block,
                int(c0) + bc * block : int(c0) + (bc + 1) * block,
            ]
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, src, dst_r, dst_c)

    paste(out, _solid_center_patch(g, bg, block), int(r0) + block, int(c0) + block)
    return out


def _gen_attempt() -> np.ndarray | None:
    block = random.choice(BLOCK_OPTIONS)
    n_min, n_max = GRID_SIZE_BY_BLOCK[block]
    n = random.randint(n_min, n_max)

    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    try:
        b00, b01 = _make_blocks(block, majority, minor_colors)
    except AssertionError:
        return None
    motif = _assemble_motif(block, b00, b01)
    center = np.full((block, block), majority, dtype=int)
    expanded = _expanded_output(motif, block, center)

    r0 = random.randint(block, n - 4 * block)
    c0 = random.randint(block, n - 4 * block)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - block, c0 - block)

    facing = _hole_facing_values(motif, block)
    if np.all(facing == majority):
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid fused db0c5428 input")
