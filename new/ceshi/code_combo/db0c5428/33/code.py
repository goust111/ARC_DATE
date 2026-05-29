# Combo variant of db0c5428 (index 33)
# combo_spec: D5/1+D6/0
# Name: 单轴对称+对称性重建中心

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve()
_SOLVER_DIR = _ROOT.parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for combo D5/1 + D6/0
# a) Concretization + behavior trace
#    - The input is still one 9x9 motif made of eight visible 3x3 blocks around a
#      missing 3x3 center, placed on a teal canvas.
#    - D6/0 changes the global condition: the ring has exactly one mirror axis,
#      either LR or UD, but never both. Therefore the output cannot use the old
#      point-opposite 5x5 expansion; it only grows along that detected axis.
#    - D5/1 says the center should be recovered by symmetry instead of purely from
#      the inward-facing pixels. Under full D4 symmetry that would determine all
#      eight non-center cells via opposite-side sampling. Here that assumption is
#      broken by D6/0: with only one global axis, symmetry tells us only one
#      coordinate dimension reliably.
#    - Fusion rule used here:
#      * LR motif: recover the center by taking rows from the old inner-boundary
#        bands [top-facing, middle-side, bottom-facing], but columns from the LR
#        symmetric far side [right edge, center axis, left edge].
#      * UD motif: recover the center by taking columns from the old inner-boundary
#        bands, but rows from the UD symmetric far side.
#      This means six of the eight non-center cells are controlled by symmetry, while
#      the two cells lying on the symmetry axis still come from the legacy boundary
#      mechanism. The central fixed point remains the majority foreground color.
#    - Small LR trace:
#      original center(0,0) read motif[2,2] from the hole-facing corner of b00.
#      fused center(0,0) reads motif[2,8], i.e. same top-facing row band but the
#      opposite mirrored column. center(0,1) still reads motif[2,4] because that
#      cell lies on the vertical symmetry axis and cannot be disambiguated further.
# b) Complexity calibration
#    - Only two axis branches exist, so the task remains learnable from few samples.
#    - The center rule is visibly different from both source variants:
#      * versus original / D6-only: six center cells can switch to far-side samples;
#      * versus D5-only: reconstruction is incomplete and still depends on the local
#        boundary in the orthogonal dimension.
#    - The generator rejects motifs where the mixed center would look too similar to
#      the legacy boundary center, so the D5 component stays observable.
# c) Hidden-assumption audit of main()
#    - bbox/get_bg still assume one isolated non-bg motif on a bg canvas.
#    - The task still uses a square 3x3 block grid with block==3.
#    - The old "paste every visible block to the point-opposite outer slot" loop is
#      invalid after D6/0; main() must branch on the unique mirror axis and only
#      paste one outer column pair or one outer row pair.
#    - The old center repair assumed both row and column coordinates could be taken
#      from the local inward-facing ring. The fused rule assumes exactly one global
#      mirror axis exists, and uses that axis to replace only one coordinate family.
# d) Placement legality
#    - The motif still has a 3-cell pad from the grid border, so the 15x15 expansion
#      frame always fits. LR-only expansion writes only the outer left/right block
#      columns; UD-only expansion writes only the outer top/bottom block rows.
# e) Visual observability
#    - A solver can infer the axis from the input because the motif visibly has one
#      mirror direction and one broken direction.
#    - Across examples, LR instances produce a 3-by-5 block band in the output and
#      UD instances produce a 5-by-3 block band. The repaired center also aligns
#      with that same axis, so both changes reinforce each other.
# f) Visual cleanliness
#    - No extra markers are added. The grid still contains one motif on plain teal.
#    - New blocks only land in empty outer slots, so there is no overlap ambiguity.
# g) Change inferability
#    - Colors are not remapped by numeric id. The only color-frequency logic is the
#      majority-color center fixed point, which is directly inferable from the motif.
#    - The generator uses only 2-4 foreground colors and enforces center-rule
#      differences structurally, not by hidden color ordering.
# 9) New main() assumptions guaranteed by generate_input()
#    - The motif has exactly one global mirror axis (xor between LR and UD).
#    - The majority color is genuinely dominant among foreground pixels.
#    - The mixed center differs from the legacy center on multiple non-center cells,
#      so the symmetry-controlled half of the rule is active in generated samples.


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _detect_axis(motif: np.ndarray) -> str:
    lr = np.array_equal(motif, np.fliplr(motif))
    ud = np.array_equal(motif, np.flipud(motif))
    assert lr ^ ud, "motif must have exactly one mirror axis"
    return "lr" if lr else "ud"


def _legacy_center_patch_from_motif(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[BLOCK // 2, BLOCK // 2] = majority
    return patch


def _full_symmetry_center_patch_from_motif(motif: np.ndarray, majority: int) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), majority, dtype=int)
    src_rows = [MOTIF_SIZE - 1, BLOCK + 1, 0]
    src_cols = [MOTIF_SIZE - 1, BLOCK + 1, 0]
    for r in range(BLOCK):
        for c in range(BLOCK):
            if r == BLOCK // 2 and c == BLOCK // 2:
                continue
            patch[r, c] = int(motif[src_rows[r], src_cols[c]])
    return patch


def _mixed_center_patch_from_motif(motif: np.ndarray, majority: int, axis: str) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), majority, dtype=int)
    boundary = [int(x) for x in (BLOCK + _CENTER_OFFSETS)]
    symmetric = [MOTIF_SIZE - 1, BLOCK + 1, 0]
    if axis == "lr":
        src_rows, src_cols = boundary, symmetric
    else:
        src_rows, src_cols = symmetric, boundary

    for r in range(BLOCK):
        for c in range(BLOCK):
            if r == BLOCK // 2 and c == BLOCK // 2:
                continue
            patch[r, c] = int(motif[src_rows[r], src_cols[c]])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a single-axis 3x3-block motif and repair its missing center with mixed symmetry.

    Input structure:
    - One 9x9 foreground motif sits on a teal background.
    - The motif is a 3x3 arrangement of 3x3 blocks with the middle block removed.
    - The visible ring has exactly one whole-motif mirror axis: either LR or UD.

    Rule:
    - Detect whether the motif is LR-symmetric or UD-symmetric.
    - Rebuild the missing center block using that axis for one coordinate dimension
      and the old inward-facing bands for the other; keep the center pixel as the
      motif's majority foreground color.
    - Keep the original motif in place, then extend it only along the detected axis
      by copying the boundary blocks to the opposite outer slots without rotation.

    Key insight: Single-axis symmetry is strong enough to redirect only half of the
    center readout. The same axis both chooses the output growth direction and tells
    you which coordinate of each missing center pixel should come from the far side.
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
    assert block == BLOCK, "this combo variant uses 3x3 blocks"

    motif = crop_bbox(g, motif_bbox)
    axis = _detect_axis(motif)
    majority = _major_foreground_color(g, bg)
    center = _mixed_center_patch_from_motif(motif, majority, axis)

    out = g.copy()
    paste(out, center, int(r0) + block, int(c0) + block)

    if axis == "lr":
        for br in range(3):
            src_left = crop_bbox(
                g,
                (
                    int(r0) + br * block,
                    int(c0),
                    int(r0) + (br + 1) * block - 1,
                    int(c0) + block - 1,
                ),
            )
            src_right = crop_bbox(
                g,
                (
                    int(r0) + br * block,
                    int(c0) + 2 * block,
                    int(r0) + (br + 1) * block - 1,
                    int(c0) + 3 * block - 1,
                ),
            )
            paste(out, src_right, int(r0) + br * block, int(c0) - block)
            paste(out, src_left, int(r0) + br * block, int(c0) + 3 * block)
    else:
        for bc in range(3):
            src_top = crop_bbox(
                g,
                (
                    int(r0),
                    int(c0) + bc * block,
                    int(r0) + block - 1,
                    int(c0) + (bc + 1) * block - 1,
                ),
            )
            src_bottom = crop_bbox(
                g,
                (
                    int(r0) + 2 * block,
                    int(c0) + bc * block,
                    int(r0) + 3 * block - 1,
                    int(c0) + (bc + 1) * block - 1,
                ),
            )
            paste(out, src_bottom, int(r0) - block, int(c0) + bc * block)
            paste(out, src_top, int(r0) + 3 * block, int(c0) + bc * block)

    return out


# --------------- generator (combo D5/1 + D6/0) ---------------

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

# -- CFG: majority density --
ASYM_MAJOR_MIN, ASYM_MAJOR_MAX = 5, 7
SYM_MAJOR_MIN, SYM_MAJOR_MAX = 5, 7

# -- CFG: generation --
MAX_RETRIES = 500
MOTIF_RETRIES = 150
MIN_CENTER_DIFF = 3
MIN_DOWNGRADE_GAP = 1

SINGLETON_ORBITS = tuple(((r, c),) for r in range(BLOCK) for c in range(BLOCK))
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)
UD_ORBITS = (
    ((0, 0), (2, 0)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 2)),
    ((1, 0),),
    ((1, 1),),
    ((1, 2),),
)
_CENTER_OFFSETS = np.array([-1, 1, 3])


def _reachable_major_counts(orbits, forced_minor):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits) if all(coord not in forced_minor for coord in orbit)
    ]
    totals = set()
    for mask in range(1 << len(usable)):
        total = 0
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                total += len(orbits[idx])
        totals.add(total)
    return sorted(totals)


def _major_orbits(orbits, forced_minor, target_count):
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
    assert choices, "unreachable symmetric majority count"
    return random.choice(choices)


def _seed_block(orbits, forced_minor, majority, major_min, major_max):
    valid_counts = [
        count for count in _reachable_major_counts(orbits, forced_minor) if major_min <= count <= major_max
    ]
    assert valid_counts, "no valid majority count for block family"
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_minor, random.choice(valid_counts))
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


def _assemble_lr_motif(b00, b01, b10, b20, b21):
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): np.fliplr(b00),
        (1, 0): b10,
        (1, 2): np.fliplr(b10),
        (2, 0): b20,
        (2, 1): b21,
        (2, 2): np.fliplr(b20),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _assemble_ud_motif(b00, b01, b02, b10, b12):
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): b02,
        (1, 0): b10,
        (1, 2): b12,
        (2, 0): np.flipud(b00),
        (2, 1): np.flipud(b01),
        (2, 2): np.flipud(b02),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _build_lr_motif(majority, minor_colors):
    for _ in range(MOTIF_RETRIES):
        b00, minors00 = _seed_block(SINGLETON_ORBITS, [(2, 2)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b01, minors01 = _seed_block(LR_ORBITS, [(2, 1)], majority, SYM_MAJOR_MIN, SYM_MAJOR_MAX)
        b10, minors10 = _seed_block(SINGLETON_ORBITS, [(1, 2)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b20, minors20 = _seed_block(SINGLETON_ORBITS, [(0, 2)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b21, minors21 = _seed_block(LR_ORBITS, [(0, 1)], majority, SYM_MAJOR_MIN, SYM_MAJOR_MAX)

        minor_orbits = minors00 + minors01 + minors10 + minors20 + minors21
        if len(minor_orbits) < len(minor_colors):
            continue
        _assign_minor_colors(minor_orbits, minor_colors)
        motif = _assemble_lr_motif(b00, b01, b10, b20, b21)
        if np.array_equal(motif, np.fliplr(motif)) and not np.array_equal(motif, np.flipud(motif)):
            return motif
    raise AssertionError("failed to build a strict LR-only motif")


def _build_ud_motif(majority, minor_colors):
    for _ in range(MOTIF_RETRIES):
        b00, minors00 = _seed_block(SINGLETON_ORBITS, [(2, 2)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b01, minors01 = _seed_block(SINGLETON_ORBITS, [(2, 1)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b02, minors02 = _seed_block(SINGLETON_ORBITS, [(2, 0)], majority, ASYM_MAJOR_MIN, ASYM_MAJOR_MAX)
        b10, minors10 = _seed_block(UD_ORBITS, [(1, 2)], majority, SYM_MAJOR_MIN, SYM_MAJOR_MAX)
        b12, minors12 = _seed_block(UD_ORBITS, [(1, 0)], majority, SYM_MAJOR_MIN, SYM_MAJOR_MAX)

        minor_orbits = minors00 + minors01 + minors02 + minors10 + minors12
        if len(minor_orbits) < len(minor_colors):
            continue
        _assign_minor_colors(minor_orbits, minor_colors)
        motif = _assemble_ud_motif(b00, b01, b02, b10, b12)
        if np.array_equal(motif, np.flipud(motif)) and not np.array_equal(motif, np.fliplr(motif)):
            return motif
    raise AssertionError("failed to build a strict UD-only motif")


def _expanded_output(motif, center, axis):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)

    if axis == "lr":
        for br in range(3):
            left = motif[br * BLOCK : (br + 1) * BLOCK, 0:BLOCK]
            right = motif[br * BLOCK : (br + 1) * BLOCK, 2 * BLOCK : 3 * BLOCK]
            paste(expanded, right, BLOCK + br * BLOCK, 0)
            paste(expanded, left, BLOCK + br * BLOCK, 4 * BLOCK)
    else:
        for bc in range(3):
            top = motif[0:BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            bottom = motif[2 * BLOCK : 3 * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, bottom, 0, BLOCK + bc * BLOCK)
            paste(expanded, top, 4 * BLOCK, BLOCK + bc * BLOCK)

    return expanded


def _center_rule_diff(motif, majority, axis) -> int:
    mixed = _mixed_center_patch_from_motif(motif, majority, axis)
    legacy = _legacy_center_patch_from_motif(motif, majority)
    mask = np.ones((BLOCK, BLOCK), dtype=bool)
    mask[BLOCK // 2, BLOCK // 2] = False
    return int(np.count_nonzero(mixed[mask] != legacy[mask]))


def _downgrade_gap(motif, majority, axis) -> int:
    mixed = _mixed_center_patch_from_motif(motif, majority, axis)
    full = _full_symmetry_center_patch_from_motif(motif, majority)
    mask = np.ones((BLOCK, BLOCK), dtype=bool)
    mask[BLOCK // 2, BLOCK // 2] = False
    return int(np.count_nonzero(mixed[mask] != full[mask]))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    axis = random.choice(("lr", "ud"))
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    motif = _build_lr_motif(majority, minor_colors) if axis == "lr" else _build_ud_motif(majority, minor_colors)
    if _center_rule_diff(motif, majority, axis) < MIN_CENTER_DIFF:
        return None
    if _downgrade_gap(motif, majority, axis) < MIN_DOWNGRADE_GAP:
        return None

    center = _mixed_center_patch_from_motif(motif, majority, axis)
    expanded = _expanded_output(motif, center, axis)

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
