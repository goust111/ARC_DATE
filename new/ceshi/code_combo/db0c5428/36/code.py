# Combo variant of db0c5428 (index 36)
# combo_spec: D1/0+D4/0+D14/3
# Name: 旋转映射+按距离逐步外扩+180度旋转

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "solvers" / "arc_utils.py"
        if cand.exists():
            sys.path.append(str(cand.parent))
            return
    raise ImportError("could not locate solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for combo 36
# ----------------------------
# concrete_behavior:
#   - Input keeps the original physical scene: one 9x9 motif on teal background,
#     built from a 3x3 array of 3x3 blocks with the center block missing.
#   - Center reconstruction is unchanged: the 8 cells facing the hole become the
#     border of the center block and the center pixel becomes the motif's majority
#     foreground color.
#   - Outer copying is no longer a single independent placement per source block.
#     First, the 4 edge-adjacent source blocks are sent to the clockwise-rotated
#     outer edge slots and each copy is rotated 180 degrees.
#   - Then the 4 corner source blocks are processed. Each corner block also uses a
#     rot180 copy, but it expands two steps along the rotated outer ring: first it
#     writes to the adjacent outer edge slot, then to the final outer corner slot.
#     Because corners are processed after edges, their first step overwrites the
#     earlier edge-slot content.
# behavior_trace:
#   - Source top edge (0,1): original task pasted it unchanged to outer south.
#     Here it is pasted as rot180 into outer east.
#   - Source top-left corner (0,0): original task pasted it once, unchanged, to
#     outer south-east. Here it is pasted as rot180 twice: first to outer north,
#     then to outer north-east. The north write overwrites the earlier copy placed
#     there by the left-edge source.
#   - Net visible difference: the outer ring reflects clockwise block mapping,
#     copied blocks visibly flip by 180 degrees, and each corner pattern appears
#     on a two-slot clockwise segment because of the stepwise overwrite.
# complexity_calibration:
#   - Full-block overwrites can hide the edge-step entirely if corner and edge
#     blocks look too similar. The generator therefore forces asymmetric seeds and
#     rejects samples where overwritten slots change too little.
#   - The rule still uses only one deterministic mapping and one deterministic
#     overwrite path per corner, so the hypothesis space stays small enough for
#     ARC-style induction from 2-4 examples.
# implicit_assumption_audit:
#   - `bbox(g != bg)` still assumes the only foreground object is the motif.
#     Generator guarantee: no extra foreground pixels are placed.
#   - `h == w and h % 3 == 0` still assumes a square block grid.
#     Generator guarantee: motif size remains fixed at 9x9.
#   - `_center_patch(...)` depends only on the hole boundary and majority color.
#     The fusion does not change those coordinates, so the helper remains valid.
#   - The original solver assumed one destination per block and no outer-slot
#     overlap. That assumption is invalid here; solver logic is changed to use an
#     explicit two-pass schedule where corner writes intentionally overwrite edge
#     writes in four known slots.
#   - Solver must read every source block from the untouched input rather than from
#     partially written output; otherwise overwrites would corrupt later reads.
# placement_legality:
#   - The expanded footprint is still 15x15, so the original 3-cell padding is
#     sufficient. Intermediate corner writes stay inside the same 5x5 block frame.
# visual_observability:
#   - The generator forces both seed families to differ from their own rot180 so
#     the D4 transform is visible.
#   - It also measures three signals: combo-vs-no-D14 difference, combo-vs-point-
#     symmetric difference, and the amount of overwritten edge-slot change. This
#     prevents degeneration into samples where the fused rule looks like a simpler
#     single-variant task.
# cleanliness_and_inferability:
#   - All action stays on a clean 5x5 block scaffold; there are no pixel-shifted
#     placements, new markers, or color-id-dependent rules.
#   - The repeated corner patterns on short clockwise outer-ring runs are visually
#     coherent rather than noisy, making the overwrite mechanic legible.
# new_solver_assumptions_checked_against_generator:
#   - Every edge-slot overwrite pair is fixed and deterministic:
#       top-left corner -> north slot,
#       top-right corner -> east slot,
#       bottom-right corner -> south slot,
#       bottom-left corner -> west slot.
#     Generator guarantee: those slots exist within the 15x15 frame and the source
#     blocks are sufficiently distinct for the overwrite to matter.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8
MAX_RETRIES = 600
MIN_OVERWRITE_DELTA = 12
MIN_COMBO_VS_NO_D14 = 12
MIN_COMBO_VS_POINT = 18

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
_EDGE_SOURCES = ((0, 1), (1, 2), (2, 1), (1, 0))
_CORNER_SOURCES = ((0, 0), (0, 2), (2, 2), (2, 0))
_CORNER_STEP1 = {
    (0, 0): (0, 2),
    (0, 2): (2, 4),
    (2, 2): (4, 2),
    (2, 0): (2, 0),
}
_EDGE_SOURCE_BY_SLOT = {
    (0, 2): (1, 0),
    (2, 4): (0, 1),
    (4, 2): (1, 2),
    (2, 0): (2, 1),
}


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


def _source_block(g: np.ndarray, r0: int, c0: int, block: int, br: int, bc: int) -> np.ndarray:
    return crop_bbox(
        g,
        (
            int(r0) + br * block,
            int(c0) + bc * block,
            int(r0) + (br + 1) * block - 1,
            int(c0) + (bc + 1) * block - 1,
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 frame using clockwise routing and staged overwrites.

    Input structure:
    - One 9x9 motif sits on teal background and is itself a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is missing in the input, leaving a teal hole.
    - The 8 cells immediately facing that hole encode the border of the missing center block.

    Rule:
    - Keep the original 9x9 motif in place and reconstruct the center block from the
      hole boundary, using the motif's majority foreground color at the center pixel.
    - Copy every visible block with a 180-degree rotation.
    - Process outer expansion in two passes under the clockwise block mapping: first
      place the four edge blocks into the outer edge slots, then let each corner block
      advance through the adjacent edge slot and onward to its outer corner slot, so
      the later corner writes overwrite earlier edge-slot copies.

    Key insight:
    The output is not a simple one-copy-per-block reflection. It is a scheduled outer-ring
    construction: clockwise remapping chooses the ring direction, rot180 changes each copy's
    orientation, and the second corner pass determines the final content of the edge slots.
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

    for br, bc in _EDGE_SOURCES:
        src = _rot180(_source_block(g, int(r0), int(c0), int(block), br, bc))
        dst_br, dst_bc = _outer_slot_clockwise(br, bc)
        paste(out, src, expanded_r0 + dst_br * block, expanded_c0 + dst_bc * block)

    for br, bc in _CORNER_SOURCES:
        src = _rot180(_source_block(g, int(r0), int(c0), int(block), br, bc))
        step1_br, step1_bc = _CORNER_STEP1[(int(br), int(bc))]
        final_br, final_bc = _outer_slot_clockwise(br, bc)
        paste(out, src, expanded_r0 + step1_br * block, expanded_c0 + step1_bc * block)
        paste(out, src, expanded_r0 + final_br * block, expanded_c0 + final_bc * block)

    paste(out, _center_patch(g, int(r0), int(c0), int(block), int(bg)), int(r0) + block, int(c0) + block)
    return out


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


def _motif_block(motif: np.ndarray, br: int, bc: int) -> np.ndarray:
    return motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]


def _gen_center_patch(motif, majority):
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _expanded_output_cw_rot(motif, center) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = _rot180(_motif_block(motif, br, bc))
            dst_br, dst_bc = _outer_slot_clockwise(br, bc)
            paste(expanded, src, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _expanded_output_point_rot(motif, center) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = _rot180(_motif_block(motif, br, bc))
            dst_br, dst_bc = _outer_slot_point_symmetric(br, bc)
            paste(expanded, src, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _expanded_output_combo(motif, center) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)

    for br, bc in _EDGE_SOURCES:
        src = _rot180(_motif_block(motif, br, bc))
        dst_br, dst_bc = _outer_slot_clockwise(br, bc)
        paste(expanded, src, dst_br * BLOCK, dst_bc * BLOCK)

    for br, bc in _CORNER_SOURCES:
        src = _rot180(_motif_block(motif, br, bc))
        step1_br, step1_bc = _CORNER_STEP1[(br, bc)]
        final_br, final_bc = _outer_slot_clockwise(br, bc)
        paste(expanded, src, step1_br * BLOCK, step1_bc * BLOCK)
        paste(expanded, src, final_br * BLOCK, final_bc * BLOCK)

    return expanded


def _overwrite_delta(motif: np.ndarray) -> int:
    total = 0
    for br, bc in _CORNER_SOURCES:
        step1 = _CORNER_STEP1[(br, bc)]
        edge_src = _EDGE_SOURCE_BY_SLOT[step1]
        corner_patch = _rot180(_motif_block(motif, br, bc))
        edge_patch = _rot180(_motif_block(motif, edge_src[0], edge_src[1]))
        total += int(np.count_nonzero(corner_patch != edge_patch))
    return total


def _combo_metrics(motif: np.ndarray, center: np.ndarray) -> dict[str, int]:
    combo = _expanded_output_combo(motif, center)
    no_d14 = _expanded_output_cw_rot(motif, center)
    point = _expanded_output_point_rot(motif, center)
    return {
        "overwrite_delta": _overwrite_delta(motif),
        "combo_vs_no_d14": int(np.count_nonzero(combo != no_d14)),
        "combo_vs_point": int(np.count_nonzero(combo != point)),
    }


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    possible_totals = [
        total
        for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
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
    metrics = _combo_metrics(motif, center)
    if metrics["overwrite_delta"] < MIN_OVERWRITE_DELTA:
        return None
    if metrics["combo_vs_no_d14"] < MIN_COMBO_VS_NO_D14:
        return None
    if metrics["combo_vs_point"] < MIN_COMBO_VS_POINT:
        return None

    expanded = _expanded_output_combo(motif, center)
    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    if np.array_equal(main(inp), out):
        return inp, metrics
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise AssertionError("failed to generate a valid db0c5428 combo input")


def sample_metrics(trials: int = 50) -> list[dict[str, int]]:
    stats: list[dict[str, int]] = []
    for _ in range(trials):
        attempt = None
        while attempt is None:
            attempt = _gen_attempt()
        _grid, metrics = attempt
        stats.append(metrics)
    return stats
