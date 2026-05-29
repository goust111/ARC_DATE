# Combo variant of db0c5428 (index 1)
# combo_spec: D1/0+D4/0
# Name: 旋转映射+旋转变换

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


# Design analysis
# ----------------
# a) Concrete behavior:
#    - The input stays structurally identical to the original task: one 9x9 motif
#      on teal background, arranged as a 3x3 grid of 3x3 blocks with the center
#      block missing.
#    - The center reconstruction rule is unchanged: the 8 cells facing the hole
#      become the border of the missing 3x3 patch, and the center pixel becomes
#      the motif's majority foreground color.
#    - The fused change happens on the outer copies.  For a source block at motif
#      coordinates (br, bc), the original task pasted it unchanged to the point-
#      symmetric outer slot (4 - 2*br, 4 - 2*bc).  This combo instead:
#        1. rotates the block-level position 90 degrees clockwise, giving the
#           destination slot (2*bc, 4 - 2*br);
#        2. rotates the copied 3x3 block itself by 180 degrees before pasting.
#    - Small trace:
#        original: top-left source b00 -> bottom-right outer corner, unchanged
#        combo:    top-left source b00 -> top-right outer corner, rot180(b00)
#      So both "where the block lands" and "how the block is oriented there"
#      differ in one coupled action.
#
# b) Complexity calibration:
#    - A pure mapping change would be weak if the blocks were too self-similar;
#      a pure rot180 change would be weak if the blocks themselves were rot180-
#      symmetric.  The generator therefore enforces visible rot180 asymmetry in
#      both seed block families and then rejects full motifs whose combo output
#      is too close to the original task or to either single-variant baseline.
#    - This keeps the fusion genuinely two-step: examples must show evidence for
#      both the clockwise slot permutation and the rot180 copy transform.
#
# c) Implicit assumption audit:
#    - `bbox(g != bg)` still assumes a single foreground motif.  Generator keeps
#      exactly one 9x9 motif and no stray foreground pixels.
#    - `h == w and h % 3 == 0` still assumes a square 3x3 block grid.  Generator
#      keeps the fixed 9x9 motif size.
#    - `_center_patch(...)` depends only on the hole boundary and majority color,
#      not on the outer-copy rule, so it remains valid unchanged.
#    - The original destination formula encoded point symmetry and is no longer
#      valid.  It must be replaced by the clockwise block-slot mapping.
#    - `paste(out, src, ...)` used to rely on distinct point-symmetric outer
#      slots.  The clockwise slots are also distinct, so no new overlap appears.
#    - Because `out` is written while source blocks are read, solver correctness
#      still relies on reading from the untouched input `g` rather than from the
#      partially written output.  The combo solver preserves that discipline.
#
# d) Placement legality:
#    - The expanded region is still 15x15, so the original 3-cell padding around
#      the 9x9 motif is sufficient to keep every outer slot in bounds.
#    - The eight destination slots under clockwise mapping are exactly the same
#      outer-ring coordinates as before, only permuted, so they remain disjoint.
#
# e) Visual observability:
#    - The generator forces a corner seed with `(0,0)` majority and `(2,2)` minor,
#      and an edge seed with `(0,1)` majority and `(2,1)` minor.  This guarantees
#      that rot180 visibly changes the copied blocks.
#    - It also scores the combo output against three baselines:
#        original mapping + no rotation,
#        clockwise mapping + no rotation,
#        point-symmetric mapping + rot180.
#      Samples are rejected unless the combo differs enough from all three, so a
#      human can actually see both dimensions in 2-4 examples.
#
# f) Visual cleanliness:
#    - No extra markers, frames, or overlays are introduced.  The scene remains
#      one clean motif on a uniform background, with the expanded output using
#      the same 5x5 block frame as the original task.
#    - All structure is communicated by block placement and orientation only,
#      which matches the original task family and avoids clutter.
#
# g) Change inferability:
#    - No hidden use of color ids is introduced.  Colors only serve as visible
#      pixel values inside the 3x3 blocks.
#    - The number of foreground colors stays low (2-4), and the global motif
#      symmetry is preserved, so the new burden on the solver is specifically
#      the combined spatial reasoning: infer the quarter-turn slot mapping and
#      the rot180 block transform together.
#
# Post-change solver assumptions reviewed against the generator:
#    - exactly one 9x9 motif exists: guaranteed
#    - the center 3x3 block is empty in the input: guaranteed
#    - clockwise outer-slot mapping yields 8 distinct in-bounds placements:
#      guaranteed by fixed 5x5 frame geometry and motif padding
#    - every copied block differs enough under rot180 to expose the transform:
#      encouraged by forced-major/forced-minor cells and enforced by visibility
#      scoring against the clockwise-no-rotation baseline
#    - the combined output is not accidentally identical to the original or to a
#      single-axis variant: guaranteed by rejection thresholds


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "solvers" / "arc_utils.py"
        if cand.exists():
            arc_dir = str(cand.parent)
            if arc_dir not in sys.path:
                sys.path.append(arc_dir)
            return
    raise ImportError("could not locate solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import bbox, crop_bbox, get_bg, paste


def _rot180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 2)


def _outer_slot_clockwise(br: int, bc: int) -> tuple[int, int]:
    return 2 * int(bc), 4 - 2 * int(br)


def _outer_slot_point_symmetric(br: int, bc: int) -> tuple[int, int]:
    return 4 - 2 * int(br), 4 - 2 * int(bc)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 frame with quarter-turned placement and rotated copies.

    Input structure:
    - A teal background contains one 9x9 motif arranged as a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is missing and stays teal in the input.
    - The eight colored cells that face this hole encode the border of the missing center block.

    Rule:
    - Keep the visible 9x9 motif in place.
    - Reconstruct the missing center block from the inward-facing boundary cells, using the motif's
      majority foreground color for the center pixel.
    - Copy each non-center 3x3 block to the outer ring of the implied 5x5 block grid by rotating
      its block position 90 degrees clockwise around the motif center, and rotate the copied block
      itself by 180 degrees before placing it there.

    Key insight:
    The input is still the middle ring of a larger block pattern, but the outer echoes are governed
    by a fused spatial rule: a quarter-turn decides the destination slot while a half-turn decides
    the copied block's orientation.
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
            dst_br, dst_bc = _outer_slot_clockwise(br, bc)
            dst_r = expanded_r0 + dst_br * block
            dst_c = expanded_c0 + dst_bc * block
            paste(out, _rot180(src), dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


# --------------- generator (combo variant) ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_PAD = 3

BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

MAX_RETRIES = 600
MIN_DIFF_VS_ORIGINAL = 24
MIN_DIFF_VS_SINGLE = 16

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


def _major_orbits(orbits, forced_minor, forced_major, target_count):
    forced_minor = set(forced_minor)
    forced_major = set(forced_major)

    usable = []
    forced_major_idxs = set()
    for idx, orbit in enumerate(orbits):
        orbit_set = set(orbit)
        if orbit_set & forced_minor and orbit_set & forced_major:
            raise AssertionError("orbit cannot be both forced major and forced minor")
        if orbit_set & forced_minor:
            continue
        usable.append(idx)
        if orbit_set & forced_major:
            forced_major_idxs.add(idx)

    base_total = sum(len(orbits[idx]) for idx in forced_major_idxs)
    if base_total > target_count:
        raise AssertionError("forced major cells exceed target count")

    optional = [idx for idx in usable if idx not in forced_major_idxs]
    choices = []
    for mask in range(1 << len(optional)):
        selected = set(forced_major_idxs)
        total = base_total
        for bit, idx in enumerate(optional):
            if mask & (1 << bit):
                selected.add(idx)
                total += len(orbits[idx])
        if total == target_count:
            choices.append(selected)
    assert choices, "unreachable symmetric majority count"
    return random.choice(choices)


def _seed_block(orbits, forced_minor, forced_major, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_minor, forced_major, target_count)
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
    b00, corner_minors = _seed_block(
        DIAG_ORBITS,
        forced_minor=[(2, 2)],
        forced_major=[(0, 0)],
        majority=majority,
        target_count=corner_major,
    )
    b01, edge_minors = _seed_block(
        LR_ORBITS,
        forced_minor=[(2, 1)],
        forced_major=[(0, 1)],
        majority=majority,
        target_count=edge_major,
    )
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    assert not np.array_equal(b00, _rot180(b00))
    assert not np.array_equal(b01, _rot180(b01))
    return b00, b01


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
        (2, 2): _rot180(b00),
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


def _expanded_output(motif, center, slot_fn, *, rotate_outer: bool) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            patch = _rot180(src) if rotate_outer else src
            dst_br, dst_bc = slot_fn(br, bc)
            paste(expanded, patch, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _combo_visibility_scores(motif, center) -> dict[str, int]:
    combo = _expanded_output(motif, center, _outer_slot_clockwise, rotate_outer=True)
    original = _expanded_output(motif, center, _outer_slot_point_symmetric, rotate_outer=False)
    d1_only = _expanded_output(motif, center, _outer_slot_clockwise, rotate_outer=False)
    d4_only = _expanded_output(motif, center, _outer_slot_point_symmetric, rotate_outer=True)
    return {
        "vs_original": int(np.count_nonzero(combo != original)),
        "vs_d1_only": int(np.count_nonzero(combo != d1_only)),
        "vs_d4_only": int(np.count_nonzero(combo != d4_only)),
    }


def _visible_enough(scores: dict[str, int]) -> bool:
    return (
        scores["vs_original"] >= MIN_DIFF_VS_ORIGINAL
        and scores["vs_d1_only"] >= MIN_DIFF_VS_SINGLE
        and scores["vs_d4_only"] >= MIN_DIFF_VS_SINGLE
    )


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
    scores = _combo_visibility_scores(motif, center)
    if not _visible_enough(scores):
        return None
    expanded = _expanded_output(motif, center, _outer_slot_clockwise, rotate_outer=True)

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
