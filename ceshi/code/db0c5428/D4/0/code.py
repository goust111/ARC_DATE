# Variant of db0c5428
# Dimension: D4 — 块复制时的几何变换 (transform)
# Change: 复制到对侧外部槽位时不旋转不翻转，保持原始方向... → 复制时做180度旋转

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

# Design analysis
# ----------------
# reference_task_read:
#   source task 2c181942 changes orientation only at placement time while keeping
#   object detection and the rest of the scene logic intact. This variant follows
#   the same discipline: the bbox detection and center reconstruction stay the
#   same, and only the copied outer blocks change orientation.
# concrete_behavior:
#   input is still one 9x9 motif made of eight visible 3x3 blocks around a missing
#   3x3 center hole on bg=8. output is still a 15x15 block-layout expansion placed
#   around the motif. the only rule change is that every copied outer block is the
#   180-degree rotation of its source block. the original 9x9 motif itself keeps
#   its in-place orientation, and the center block is still rebuilt from the 8
#   inner-boundary pixels around the hole.
# behavior_trace:
#   example for the top-left source block b00:
#   - original solver: source b00 is pasted unchanged into the bottom-right outer slot.
#   - variant solver: source b00 is rotated 180 degrees, then pasted into that slot.
#   - difference: if b00[0,0] != b00[2,2], the outer copy visibly swaps those two
#     locations; this is exactly the D4-axis change we want the examples to expose.
# complexity_calibration:
#   changing only the outer placement transform is too weak if the 3x3 blocks are
#   themselves 180-symmetric. therefore the generator explicitly breaks 180 symmetry
#   inside both seed block families while keeping the global motif structure close
#   to the original task. this preserves learnability and avoids adding unrelated
#   mechanics such as color remapping or extra objects.
# implicit_assumption_audit:
#   - bbox(g != bg) assumes the only foreground object is the 9x9 motif.
#     generator guarantee: no stray foreground pixels are placed elsewhere.
#   - h == w and h % 3 == 0 assumes a square 3x3 block grid.
#     generator guarantee: motif size is fixed at 9x9.
#   - expanded_r0/c0 subtract one block and every paste stays in-bounds.
#     generator guarantee: motif top-left is padded by at least BLOCK=3, so the
#     full 15x15 expanded region fits.
#   - center reconstruction samples 8 specific pixels around the hole.
#     generator guarantee: center remains empty bg in input and adjacent blocks
#     occupy the same coordinates as in the original task.
#   - outer copy transform now depends on source orientation.
#     generator guarantee: both seed block types differ from their own rot180, and
#     a visibility score rejects samples where rotated-vs-unrotated outer rings are
#     too similar.
# placement_legality:
#   the motif is placed with 3-cell padding on all sides, so the expanded 15x15
#   output never clips. there is only one motif, no overlaps, and solver reads
#   source blocks from the untouched input grid rather than from partially written
#   output.
# visual_observability:
#   to make the rotation legible in 2-4 examples, the generator forces one
#   rot180-mapped cell pair in each seed family to take different colors:
#   corner seed uses (0,0)=majority and (2,2)=minor; edge seed uses
#   (0,1)=majority and (2,1)=minor. this makes every outer copied block visibly
#   change under 180-degree rotation without introducing extra symbolic cues.
# tidiness_and_colors:
#   no new layers or markers are added; the scene remains a single motif on clean
#   background. colors are only used as visible cell values inside 3x3 blocks; no
#   sorting or hidden logic depends on color ids.


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


def _rot180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 2)


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
    Expand a 3x3 block motif into a 5x5 block layout with rotated outer echoes.

    Input structure:
    - A teal background contains one 9x9 motif made from a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is absent and remains teal in the input.
    - The eight colored cells immediately facing that hole encode the missing
      center block's perimeter.

    Rule:
    - Keep the original 9x9 motif where it is.
    - Rebuild the missing center block from the eight inward-facing cells; its
      center pixel is the motif's majority foreground color.
    - Copy each non-center 3x3 block to the opposite outer slot of the implied
      5x5 block grid, but rotate that copied block by 180 degrees first.

    Key insight:
    The input already shows the middle ring of a larger block pattern. Solving
    the task means reading the block grid, recovering the center from the hole's
    boundary, and applying a fixed rot180 transform to every outward duplication.
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
            paste(out, _rot180(src), dst_r, dst_c)

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

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

# -- CFG: generation --
MAX_RETRIES = 500
MIN_ROTATION_VISIBILITY = 16

# Diagonal-symmetric orbits for the corner seed.
DIAG_ORBITS = (
    ((0, 0),),
    ((1, 1),),
    ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)

# LR-symmetric orbits for the top-edge seed.
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
    """Pick orbit ids summing to target_count while honoring required major/minor orbits."""
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
    """Build one symmetric seed block and return its unresolved minor orbits."""
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
    """Fill all non-majority orbits while ensuring each minor color appears."""
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
    """Build two asymmetric seeds that stay symmetric enough to define the motif."""
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
    """Assemble the 9x9 motif from two seeds; center block remains bg."""
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
    """Recover the missing center from the inner boundary of the motif."""
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _expanded_output(motif, center, *, rotate_outer: bool) -> np.ndarray:
    """Build the 15x15 expanded region for either the old or the new outer-copy rule."""
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
            paste(expanded, patch, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _rotation_visibility_score(motif, center) -> int:
    """Count how many pixels differ between rotated-copy and legacy-copy outer rings."""
    legacy = _expanded_output(motif, center, rotate_outer=False)
    variant = _expanded_output(motif, center, rotate_outer=True)
    return int(np.count_nonzero(legacy != variant))


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build motif, verify via oracle, return input or None."""
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    possible_totals = [
        total for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1) if 18 - total >= len(minor_colors)
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
    if _rotation_visibility_score(motif, center) < MIN_ROTATION_VISIBILITY:
        return None
    expanded = _expanded_output(motif, center, rotate_outer=True)

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
    raise AssertionError("failed to generate a valid db0c5428 D4-rotation input")
