# Combo variant of db0c5428 (index 48)
# combo_spec: D3/0+D5/1+D14/0
# Name: 完全填充5x5+先重建再外扩+对称性重建

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve()
_SOLVER_DIR = None
for _parent in _ROOT.parents:
    _candidate = _parent / "solvers"
    if (_candidate / "arc_utils.py").exists():
        _SOLVER_DIR = _candidate
        break
if _SOLVER_DIR is None:
    raise RuntimeError("could not locate solvers/arc_utils.py")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox, get_bg, paste


# Design analysis for combo D3/0 + D5/1 + D14/0
# a) Concretization + trace:
#    The input is still one 9x9 motif made from eight visible 3x3 blocks around an
#    empty 3x3 hole. Step 1 now repairs that hole by global LR/UD symmetry:
#    center(0,0) reads from the opposite bottom-right corner block, center(0,1)
#    reads from the opposite bottom edge block, center(1,0) reads from the opposite
#    right edge block, and so on; the center pixel stays the majority foreground
#    color. Step 2 treats this completed 3x3 block array as the true source object
#    and expands it to a full 5x5 block array by reflecting the block rows/cols one
#    step outward: row/col source order becomes [1,0,1,2,1]. Example: if the repaired
#    3x3 block matrix is A, the top output row of blocks becomes
#    [A[1,1], A[1,0], A[1,1], A[1,2], A[1,1]]. Because row 0 and row 4 both map to
#    source row 1, and likewise for columns, the repaired center A[1,1] is copied
#    into eight outer-ring positions as well as the true center. This differs from
#    the original task in two places: the center no longer comes from inward-facing
#    boundary pixels, and the 12 previously blank outer slots are now populated by
#    reflected block copies.
# b) Complexity calibration:
#    The fused rule is stronger than either single change alone but still has one
#    clean interpretation: "repair the missing block, then reflect the finished 3x3
#    block matrix outward by one block." There are only three block row classes and
#    three block column classes, so examples repeatedly support the same mapping
#    instead of requiring many ad-hoc cases.
# c) Hidden-assumption audit of main():
#    - bbox/get_bg still assume one isolated non-background motif on a plain canvas.
#    - The solver still assumes the motif bbox is square and divisible into a 3x3
#      block grid; generate_input() preserves that.
#    - Symmetry-based center repair assumes block==3 because each missing cell maps
#      to one of three visible source bands across the hole.
#    - The old sparse outer-copy loop is invalid here because it never fills the 12
#      formerly blank slots and never lets the repaired center influence expansion.
#      The new solver must therefore build the full repaired 3x3 motif first and only
#      then expand from that completed object.
# d) Placement legality:
#    The generator keeps the original 3-cell padding, so the full 15x15 reflected
#    output always fits inside the 16..19 square canvas without overlap or clipping.
# e) Visual observability:
#    The D5 signal is forced to be visible by making the symmetry-repaired center
#    differ from the legacy inward-boundary center in all 8 non-central cells. The
#    D3+D14 signal is visible because the 12 slots that were teal in the original
#    task are now filled, and several of them are copies of the repaired center.
# f) Visual cleanliness:
#    The scene still contains one compact motif on a uniform teal background. The
#    output is one contiguous 15x15 block array with no extra markers or overlaps, so
#    a human can parse it as a single reflected expansion rather than clutter.
# g) Color inference:
#    No hidden color ordering is used. Colors are chosen directly from the ARC palette
#    and the only frequency-based choice is the majority color for the center fixed
#    point, which symmetry alone does not determine.
# 9) New main() assumptions guaranteed by generate_input():
#    - The visible ring keeps the original D4 block relations, so opposite-side reads
#      for the repaired center are consistent.
#    - The repaired center differs from every visible source block by at least a few
#      pixels, making center-copy participation in the outer ring observable.
#    - The reflected 5x5 layout always uses the block-index pattern [1,0,1,2,1] in
#      both axes, so the solver's full-fill expansion rule matches every sample.


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
_REFLECT_PAD_MAP = (1, 0, 1, 2, 1)


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _symmetry_center_patch_from_grid(
    g: np.ndarray, r0: int, c0: int, block: int, bg: int
) -> np.ndarray:
    assert block == BLOCK, "this combo variant uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)

    src_rows = [int(r0) + 3 * block - 1, int(r0) + block + 1, int(r0)]
    src_cols = [int(c0) + 3 * block - 1, int(c0) + block + 1, int(c0)]
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[src_rows[r], src_cols[c]])
    return patch


def _symmetry_center_patch_from_motif(motif: np.ndarray, majority: int) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), majority, dtype=int)
    src_rows = [MOTIF_SIZE - 1, BLOCK + 1, 0]
    src_cols = [MOTIF_SIZE - 1, BLOCK + 1, 0]
    for r in range(BLOCK):
        for c in range(BLOCK):
            if r == BLOCK // 2 and c == BLOCK // 2:
                continue
            patch[r, c] = int(motif[src_rows[r], src_cols[c]])
    return patch


def _legacy_center_patch_from_motif(motif: np.ndarray, majority: int) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), majority, dtype=int)
    rows = [BLOCK - 1, BLOCK + 1, 2 * BLOCK]
    cols = [BLOCK - 1, BLOCK + 1, 2 * BLOCK]
    for r in range(BLOCK):
        for c in range(BLOCK):
            if r == BLOCK // 2 and c == BLOCK // 2:
                continue
            patch[r, c] = int(motif[rows[r], cols[c]])
    return patch


def _completed_motif_from_input(
    g: np.ndarray, r0: int, c0: int, block: int, bg: int
) -> np.ndarray:
    motif = g[int(r0) : int(r0) + 3 * block, int(c0) : int(c0) + 3 * block].copy()
    paste(
        motif,
        _symmetry_center_patch_from_grid(g, int(r0), int(c0), block, bg),
        block,
        block,
    )
    return motif


def _expand_completed_motif(full_motif: np.ndarray, block: int) -> np.ndarray:
    expanded = np.full((5 * block, 5 * block), BG, dtype=int)
    for out_br, src_br in enumerate(_REFLECT_PAD_MAP):
        for out_bc, src_bc in enumerate(_REFLECT_PAD_MAP):
            src = full_motif[
                src_br * block : (src_br + 1) * block,
                src_bc * block : (src_bc + 1) * block,
            ]
            paste(expanded, src, out_br * block, out_bc * block)
    return expanded


def _block_hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _fusion_metrics_from_motif(motif: np.ndarray, majority: int) -> dict[str, int]:
    center = _symmetry_center_patch_from_motif(motif, majority)
    legacy_center = _legacy_center_patch_from_motif(motif, majority)
    visible_blocks = []
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            visible_blocks.append(
                motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            )

    center_outer_copies = 0
    for out_br, src_br in enumerate(_REFLECT_PAD_MAP):
        for out_bc, src_bc in enumerate(_REFLECT_PAD_MAP):
            if (out_br, out_bc) == (2, 2):
                continue
            if src_br == 1 and src_bc == 1:
                center_outer_copies += 1

    return {
        "variant_diff": _block_hamming(center, legacy_center)
        - int(center[1, 1] != legacy_center[1, 1]),
        "newly_filled_slots": 12,
        "center_outer_copies": center_outer_copies,
        "min_center_vs_visible_diff": min(_block_hamming(center, blk) for blk in visible_blocks),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a missing center block, then expand the finished 3x3 motif into a fully
    populated 5x5 block array.

    Input structure:
    - One 9x9 non-background motif sits on a teal canvas.
    - The motif is a 3x3 arrangement of 3x3 blocks with the middle block removed.
    - The eight visible blocks form a globally symmetric ring around the hole.

    Rule:
    - First reconstruct the missing 3x3 center block by reflecting each non-central
      cell across the hole to its opposite visible counterpart; the center cell is
      the motif's majority foreground color.
    - Treat that repaired 3x3 block array as complete data.
    - Expand it to a full 5x5 block layout by reflecting the block rows and block
      columns one step outward, so every one of the 25 block positions is filled.

    Key insight: This is no longer a sparse "copy eight blocks outward" task. The
    solver must repair the center before expanding, because the completed center is
    itself reflected into the outer ring of the final 5x5 pattern.
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

    full_motif = _completed_motif_from_input(g, int(r0), int(c0), block, bg)
    expanded = _expand_completed_motif(full_motif, block)

    out = g.copy()
    paste(out, expanded, int(r0) - block, int(c0) - block)
    return out


# --------------- generator (combo D3/0 + D5/1 + D14/0) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
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
MAX_RETRIES = 300
MIN_VARIANT_DIFF = 8
MIN_CENTER_BLOCK_DIFF = 3

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


def _orbit_indices_containing(orbits, coords):
    coord_set = set(coords)
    found = set()
    for idx, orbit in enumerate(orbits):
        if any(coord in coord_set for coord in orbit):
            found.add(idx)
    return found


def _major_orbits(orbits, forced_major, forced_minor, target_count):
    forced_major_idxs = _orbit_indices_containing(orbits, forced_major)
    forced_minor_idxs = _orbit_indices_containing(orbits, forced_minor)
    assert not (forced_major_idxs & forced_minor_idxs), "orbit cannot be both major and minor"

    base_count = sum(len(orbits[idx]) for idx in forced_major_idxs)
    usable = [
        idx
        for idx in range(len(orbits))
        if idx not in forced_major_idxs and idx not in forced_minor_idxs
    ]
    choices = []
    for mask in range(1 << len(usable)):
        selected = set(forced_major_idxs)
        total = base_count
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                selected.add(idx)
                total += len(orbits[idx])
        if total == target_count:
            choices.append(selected)
    assert choices, "unreachable symmetric majority count"
    return random.choice(choices)


def _seed_block(orbits, forced_major, forced_minor, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_major, forced_minor, target_count)
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
        forced_major=[(0, 0)],
        forced_minor=[(2, 2)],
        majority=majority,
        target_count=corner_major,
    )
    b01, edge_minors = _seed_block(
        LR_ORBITS,
        forced_major=[(0, 1)],
        forced_minor=[(2, 1)],
        majority=majority,
        target_count=edge_major,
    )
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    assert int(b00[0, 0]) == int(majority)
    assert int(b00[2, 2]) != int(majority)
    assert int(b01[0, 1]) == int(majority)
    assert int(b01[2, 1]) != int(majority)
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
        (2, 2): np.rot90(b00, 2),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _completed_motif_from_parts(motif: np.ndarray, majority: int) -> tuple[np.ndarray, np.ndarray]:
    center = _symmetry_center_patch_from_motif(motif, majority)
    full_motif = motif.copy()
    paste(full_motif, center, BLOCK, BLOCK)
    return full_motif, center


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
    metrics = _fusion_metrics_from_motif(motif, majority)
    if metrics["variant_diff"] < MIN_VARIANT_DIFF:
        return None
    if metrics["min_center_vs_visible_diff"] < MIN_CENTER_BLOCK_DIFF:
        return None

    full_motif, _center = _completed_motif_from_parts(motif, majority)
    expanded = _expand_completed_motif(full_motif, BLOCK)

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
    raise AssertionError("failed to generate a valid fused db0c5428 input")
