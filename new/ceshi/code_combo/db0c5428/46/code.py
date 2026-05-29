# Combo variant of db0c5428 (index 46)
# combo_spec: D6/1+D15/0
# Name: C4旋转对称+块内对称

from __future__ import annotations

import random
import sys

import numpy as np

_SOLVERS_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVERS_DIR not in sys.path:
    sys.path.append(_SOLVERS_DIR)

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# spec_alignment:
# - The canvas still uses a single 9x9 motif placed on a 16..19 square teal grid.
# - The motif is a 3x3 arrangement of 3x3 blocks with the center block missing.
# - The original "fill the center from the 8 boundary-facing pixels" rule stays;
#   the fused change is in the block-to-block symmetry and the within-block pattern.
#
# concretization_and_behavior_trace:
# - Let the top-left seed corner block be
#     A = [[a, b, a],
#          [c, d, c],
#          [e, f, e]]
#   and the top seed edge block be
#     B = [[g, h, g],
#          [i, j, i],
#          [k, l, k]]
#   Both are LR-symmetric but not UD-symmetric.
# - Under D6/1, the 8 visible blocks form one C4 ring:
#     top-left=A, top=B, top-right=rot90(A), right=rot90(B),
#     bottom-right=rot180(A), bottom=rot180(B),
#     bottom-left=rot270(A), left=rot270(B).
# - This differs from the original task in two places:
#   1. block relations become quarter-rotations instead of flips;
#   2. outer expansion copies each block to the same-direction outer slot rather
#      than to the point-opposite slot.
# - The D15/0 part is nested inside that ring: every visible 3x3 block is itself
#   axis-symmetric. Because A and B are LR-symmetric, their quarter-turns become
#   UD-symmetric on the right/left side. So the visible motif alternates
#   LR-symmetric and UD-symmetric blocks while keeping the whole 9x9 arrangement C4.
# - The center reconstruction also inherits the nesting. The boundary-facing cells
#   produce
#     center = [[e, l, e],
#               [l, M, l],
#               [e, l, e]]
#   where M is the majority color. So the missing block becomes symmetric as well.
#
# complexity_calibration:
# - If A or B were solid, 180-symmetric, or mirror-compatible with their rotations,
#   the C4 change would be hard to see. The generator therefore rejects any seed
#   block that is UD-symmetric, 180-invariant, or equal to one of its quarter-turns.
# - Only two seed blocks are still used, so the task remains compact and readable.
# - Colors stay in the original 2..4 foreground range; the symmetry, not a large
#   color vocabulary, carries the rule.
#
# implicit_assumption_audit:
# - `bbox(g != bg)` still assumes exactly one non-background motif. The generator
#   preserves that by placing only one 9x9 object.
# - `block == 3` remains valid because the fused variant does not change geometry.
# - `_center_patch()` still only relies on the 8 cells adjacent to the hole and on
#   a well-defined majority foreground color; both are guaranteed by generation.
# - The old outer-slot formula was the only solver line that encoded point symmetry.
#   Under D6/1 it becomes wrong, so it is replaced by same-direction expansion.
# - The solver does not explicitly verify C4 or block symmetry. Those are new input
#   assumptions, and the generator enforces them by construction plus rejection.
#
# placement_legality:
# - The motif keeps a 3-cell margin from the canvas border, so the 15x15 expanded
#   region always fits after shifting by one block in every direction.
# - There is no second object and no overlap between the outer ring and unrelated
#   content, so the bbox-based extraction remains unambiguous.
#
# visual_observability:
# - A human can see the C4 rule because asymmetric LR-symmetric seed blocks visibly
#   quarter-turn around the hole.
# - A human can also see the D15/0 rule because every visible block has a clear
#   vertical or horizontal mirror axis, and the reconstructed center becomes a
#   symmetric cross-style block.
# - The generator rejects accidental global LR or UD symmetry so the change does not
#   collapse back to the original bilateral motif.
#
# cleanliness_and_inferability:
# - The nested symmetry is expressed purely through shape, orientation, and repeated
#   local pixel equalities. No hidden ordering uses color ids or background choice.
# - The same few colors recur across both seed blocks, which gives enough repeated
#   evidence for the viewer without cluttering the grid.
#
# behavior_metric_for_non_degeneracy:
# - Metric 1: every generated sample must have visible block symmetry labels
#   `['lr', 'lr', 'ud', 'ud', 'ud', 'ud', 'lr', 'lr']` up to the fixed block order.
#   This proves the D15/0 constraint is active in every sample.
# - Metric 2: compare the combo output with the original point-symmetric solver on
#   the same input. The number of differing cells must be positive and typically
#   large, proving the D6/1 expansion change is active too.
#
# new_main_assumptions_and_generator_guarantees:
# - main() assumes the motif is exactly one 9x9 square and that the majority color
#   is still the dominant non-background color; the generator preserves the original
#   majority ranges to keep that true.
# - main() assumes the surrounding 3x3 outer slots are blank background; the fixed
#   placement margin guarantees it.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
MAX_RETRIES = 300
BLOCK_RETRIES = 200

BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 15
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 7
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

_CENTER_OFFSETS = np.array([-1, 1, 3])
_VISIBLE_BLOCKS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)


def _rot90cw(block: np.ndarray, k: int = 1) -> np.ndarray:
    return np.rot90(block, -int(k))


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _block_slice(r0: int, c0: int, block: int, br: int, bc: int) -> tuple[int, int, int, int]:
    return (
        int(r0) + br * block,
        int(c0) + bc * block,
        int(r0) + (br + 1) * block - 1,
        int(c0) + (bc + 1) * block - 1,
    )


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == BLOCK, "this task uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)

    hole_r = int(r0) + block
    hole_c = int(c0) + block
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[hole_r + _CENTER_OFFSETS[r], hole_c + _CENTER_OFFSETS[c]])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a quarter-turn-symmetric 3x3 block motif into a larger 5x5 block frame.

    Input structure:
    - One 9x9 motif sits on a teal background as eight visible 3x3 blocks around a missing center block.
    - Moving around the hole, the surrounding blocks are related by 90-degree rotations rather than mirror flips.
    - Each visible 3x3 block is itself mirror-symmetric across either a vertical or a horizontal axis.

    Rule:
    - Leave the original 9x9 motif in place.
    - Reconstruct the missing center block from the eight pixels that face the hole, with the center cell set to the motif's majority foreground color.
    - Copy every visible block outward into the outer slot in the same direction from the motif center, without applying another rotation.

    Key insight: The 9x9 input is the middle ring of a larger C4 arrangement, and the local mirror symmetry lives inside each block while the global quarter-turn symmetry dictates where each block repeats.
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
    assert block == BLOCK, "this variant uses 3x3 blocks"

    out = g.copy()
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = crop_bbox(g, _block_slice(int(r0), int(c0), block, br, bc))
            dst_r = expanded_r0 + (2 * br) * block
            dst_c = expanded_c0 + (2 * bc) * block
            paste(out, src, dst_r, dst_c)

    paste(out, _center_patch(g, int(r0), int(c0), block, bg), int(r0) + block, int(c0) + block)
    return out


def _major_orbit_choices(
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    forced_minor: list[tuple[int, int]],
    target_count: int,
) -> list[set[int]]:
    forced_minor_set = set(forced_minor)
    usable = [
        idx
        for idx, orbit in enumerate(orbits)
        if all(coord not in forced_minor_set for coord in orbit)
    ]
    choices: list[set[int]] = []
    for mask in range(1 << len(usable)):
        selected: set[int] = set()
        total = 0
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                selected.add(idx)
                total += len(orbits[idx])
        if total == target_count:
            choices.append(selected)
    return choices


def _max_minor_orbits(
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    forced_minor: list[tuple[int, int]],
    target_count: int,
) -> int:
    choices = _major_orbit_choices(orbits, forced_minor, target_count)
    assert choices, "unreachable symmetric majority count"
    return len(orbits) - min(len(choice) for choice in choices)


def _seed_block(
    orbits: tuple[tuple[tuple[int, int], ...], ...],
    forced_minor: list[tuple[int, int]],
    majority: int,
    target_count: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, tuple[tuple[int, int], ...]]]]:
    choices = _major_orbit_choices(orbits, forced_minor, target_count)
    assert choices, "unreachable symmetric majority count"
    major_idxs = random.choice(choices)
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    minor_orbits: list[tuple[np.ndarray, tuple[tuple[int, int], ...]]] = []
    for idx, orbit in enumerate(orbits):
        if idx in major_idxs:
            for r, c in orbit:
                block[r, c] = majority
        else:
            minor_orbits.append((block, orbit))
    return block, minor_orbits


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
    for (block, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            block[r, c] = color


def _is_rotation_distinct(block: np.ndarray) -> bool:
    return all(not np.array_equal(block, _rot90cw(block, k)) for k in (1, 2, 3))


def _same_rotation_family(a: np.ndarray, b: np.ndarray) -> bool:
    return any(np.array_equal(_rot90cw(a, k), b) for k in range(4))


def _has_lr_symmetry(block: np.ndarray) -> bool:
    return np.array_equal(block, np.fliplr(block))


def _has_ud_symmetry(block: np.ndarray) -> bool:
    return np.array_equal(block, np.flipud(block))


def _symmetry_label(block: np.ndarray) -> str:
    lr = _has_lr_symmetry(block)
    ud = _has_ud_symmetry(block)
    if lr and not ud:
        return "lr"
    if ud and not lr:
        return "ud"
    if lr and ud:
        return "both"
    return "none"


def _make_blocks(
    majority: int,
    minor_colors: list[int],
    corner_major: int,
    edge_major: int,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(BLOCK_RETRIES):
        b00, corner_minors = _seed_block(LR_ORBITS, [(2, 2)], majority, corner_major)
        b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
        minor_orbits = corner_minors + edge_minors
        if len(minor_orbits) < len(minor_colors):
            continue
        _assign_minor_colors(minor_orbits, minor_colors)
        if not _has_lr_symmetry(b00) or not _has_lr_symmetry(b01):
            continue
        if _has_ud_symmetry(b00) or _has_ud_symmetry(b01):
            continue
        if not _is_rotation_distinct(b00) or not _is_rotation_distinct(b01):
            continue
        if _same_rotation_family(b00, b01):
            continue
        motif = _assemble_motif(b00, b01)
        if np.array_equal(motif, np.fliplr(motif)):
            continue
        if np.array_equal(motif, np.flipud(motif)):
            continue
        center = _gen_center_patch(motif, majority)
        if _symmetry_label(center) != "both":
            continue
        return b00, b01
    raise AssertionError("failed to build non-degenerate symmetric C4 seed blocks")


def _assemble_motif(b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): _rot90cw(b00, 1),
        (1, 2): _rot90cw(b01, 1),
        (2, 2): _rot90cw(b00, 2),
        (2, 1): _rot90cw(b01, 2),
        (2, 0): _rot90cw(b00, 3),
        (1, 0): _rot90cw(b01, 3),
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


def _expanded_output(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (2 * br) * BLOCK, (2 * bc) * BLOCK)
    return expanded


def _opposite_expanded_output(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
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


def fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None
    r0, c0, r1, c1 = motif_bbox
    assert int(r1 - r0 + 1) == MOTIF_SIZE and int(c1 - c0 + 1) == MOTIF_SIZE
    motif = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    labels = []
    for br, bc in _VISIBLE_BLOCKS:
        block = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
        labels.append(_symmetry_label(block))
    majority = _major_foreground_color(g, bg)
    center = _gen_center_patch(motif, majority)
    same_dir = _expanded_output(motif, center)
    opposite = _opposite_expanded_output(motif, center)
    return {
        "visible_labels": labels,
        "center_label": _symmetry_label(center),
        "same_vs_opposite_diff": int(np.count_nonzero(same_dir != opposite)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    pairs = []
    for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1):
        for corner in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1):
            edge = total - corner
            if not (EDGE_MAJOR_MIN <= edge <= EDGE_MAJOR_MAX):
                continue
            minor_capacity = _max_minor_orbits(LR_ORBITS, [(2, 2)], corner)
            minor_capacity += _max_minor_orbits(LR_ORBITS, [(2, 1)], edge)
            if minor_capacity >= len(minor_colors):
                pairs.append((corner, edge))
    assert pairs, "expected at least one feasible majority split"
    corner_major, edge_major = random.choice(pairs)

    try:
        b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    except AssertionError:
        return None

    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    expanded = _expanded_output(motif, center)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    if not np.array_equal(main(inp), out):
        return None
    metrics = fusion_metrics(inp)
    if metrics["center_label"] != "both":
        return None
    if metrics["same_vs_opposite_diff"] <= 0:
        return None
    if set(metrics["visible_labels"]) != {"lr", "ud"}:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
