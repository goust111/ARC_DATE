# Combo variant of db0c5428 (index 13)
# combo_spec: D9/2+D11/3
# Name: 多图案输入+多个空洞

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox, get_bg, paste

# Design analysis
# - Concrete form and behavior trace:
#   The source task had one 9x9 motif with exactly one missing 3x3 center block.
#   This fusion places two copies of the same hidden motif on one larger canvas,
#   and each visible copy is missing the center plus 1 or 2 extra whole blocks.
#   Example trace: old solver -> `bbox(g != bg)` finds one motif -> read 8 center
#   clue pixels -> fill center -> mirror outer blocks. New trace -> a single bbox
#   would wrap both motifs, and connected components are unreliable because extra
#   holes can split one motif into several pieces. The solver must first localize
#   each 9x9 motif by its block-grid signature, then restore every non-center hole
#   from the 180-rotated opposite donor, then reconstruct the center, then expand.
# - Complexity calibration:
#   The canvas always contains exactly two motifs, so the multiplicity is obvious.
#   Each motif has 2 or 3 holes total, enough to visibly differ from the source
#   task but still constrained: every extra hole comes from a distinct opposite
#   pair, so the donor block is always unique and visible. The two motif instances
#   share the same hidden full motif, but their hole sets are sampled independently
#   to show the repair rule acts locally per instance rather than globally once.
# - Hidden assumptions broken in the old `main()`:
#   1. `bbox(g != bg)` assumed one connected foreground object; false with two motifs.
#   2. A connected-component pass over foreground would still be insufficient,
#      because removing edge/corner blocks can disconnect a single motif.
#   3. `_center_patch(...)` assumed all eight center-facing clue pixels already
#      exist; false once adjacent non-center blocks may also be holes.
#   4. The expansion loop assumed every non-center source block is already present
#      in the input; false until those extra holes are repaired.
#   The replacement strategy is window scanning: because the block size stays 3,
#   a true motif is exactly a 9x9 patch whose 3x3 sub-blocks are each either all
#   background or all non-background, whose center sub-block is background, and
#   whose hole pattern is one of the legal "center + 1/2 extras" cases.
# - Placement legality:
#   The generator samples two disjoint 15x15 expanded territories, then embeds one
#   9x9 motif at offset (+3,+3) inside each territory. This guarantees that every
#   repaired-and-expanded output stays in bounds and the two 15x15 outputs never
#   overlap. Independent hole choices only blank whole 3x3 blocks inside those
#   motif footprints, so no partial overlaps or off-grid writes can occur.
# - Visual observability and cleanliness:
#   Humans see two separated hollow motifs on a plain teal background. Each motif
#   contains 2-3 empty 3x3 slots, and the output shows both the local hole repair
#   and the larger 5x5 expansion around each instance. No color-id arithmetic or
#   invisible ordering is used; every reconstruction cue is visible either as an
#   opposite donor block or as center-adjacent boundary pixels. Because the two
#   expanded territories are disjoint, the output remains easy to parse.
# - New solver assumptions audited against the generator:
#   1. Every real motif occupies one 9x9 window aligned to 3x3 blocks.
#   2. Each 3x3 sub-block is either fully background (a hole) or contains no
#      background at all.
#   3. Each motif has exactly 2 or 3 holes total, always including the center.
#   4. No non-center hole removes both members of one opposite pair.
#   5. The two motifs' 9x9 windows do not overlap.
#   The generator constructs samples to satisfy each condition directly.

BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_OFFSET = BLOCK
MOTIF_COUNT = 2

GRID_SIZE_MIN, GRID_SIZE_MAX = 31, 36
PLACEMENT_GAP_MIN, PLACEMENT_GAP_MAX = 1, 3

BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8
MAX_RETRIES = 400
LAYOUT_MODES = ("horizontal", "vertical", "diagonal")

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
NONCENTER_OPPOSITE_PAIRS = (
    ((0, 0), (2, 2)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 0)),
    ((1, 0), (1, 2)),
)


def _all_legal_hole_sets() -> tuple[frozenset[tuple[int, int]], ...]:
    legal: list[frozenset[tuple[int, int]]] = []
    for extra_count in (1, 2):
        for pairs in itertools.combinations(NONCENTER_OPPOSITE_PAIRS, extra_count):
            for picks in itertools.product((0, 1), repeat=extra_count):
                holes = {(1, 1)}
                for pair, pick in zip(pairs, picks):
                    holes.add(pair[pick])
                legal.append(frozenset(holes))
    return tuple(legal)


LEGAL_HOLE_SETS = _all_legal_hole_sets()


def _block_slice(br: int, bc: int) -> tuple[slice, slice]:
    return slice(br * BLOCK, (br + 1) * BLOCK), slice(bc * BLOCK, (bc + 1) * BLOCK)


def _extract_block(motif: np.ndarray, br: int, bc: int) -> np.ndarray:
    rs, cs = _block_slice(br, bc)
    return motif[rs, cs].copy()


def _is_bg_block(block_arr: np.ndarray, bg: int) -> bool:
    return bool(np.all(block_arr == int(bg)))


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _holes_are_legal(holes: frozenset[tuple[int, int]]) -> bool:
    if holes not in LEGAL_HOLE_SETS:
        return False
    for br, bc in holes:
        if (br, bc) == (1, 1):
            continue
        if (2 - br, 2 - bc) in holes:
            return False
    return True


def _block_occupancy(window: np.ndarray, bg: int) -> np.ndarray | None:
    occ = np.zeros((3, 3), dtype=bool)
    for br in range(3):
        for bc in range(3):
            block_arr = _extract_block(window, br, bc)
            is_bg = np.all(block_arr == int(bg))
            is_fg = np.all(block_arr != int(bg))
            if not (is_bg or is_fg):
                return None
            occ[br, bc] = bool(is_fg)
    return occ


def _find_motifs(g: np.ndarray, bg: int) -> list[tuple[int, int, np.ndarray, frozenset[tuple[int, int]]]]:
    h, w = g.shape
    candidates: list[tuple[int, int, np.ndarray, frozenset[tuple[int, int]]]] = []
    for r0 in range(h - MOTIF_SIZE + 1):
        for c0 in range(w - MOTIF_SIZE + 1):
            window = g[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
            occ = _block_occupancy(window, bg)
            if occ is None:
                continue
            holes = frozenset((br, bc) for br in range(3) for bc in range(3) if not occ[br, bc])
            if not _holes_are_legal(holes):
                continue
            window_bbox = bbox(window != bg)
            if window_bbox != (0, 0, MOTIF_SIZE - 1, MOTIF_SIZE - 1):
                continue
            candidates.append((int(r0), int(c0), window.copy(), holes))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[tuple[int, int, np.ndarray, frozenset[tuple[int, int]]]] = []
    for candidate in candidates:
        r0, c0, _window, _holes = candidate
        overlaps = any(
            not (
                r0 + MOTIF_SIZE <= sr
                or sr + MOTIF_SIZE <= r0
                or c0 + MOTIF_SIZE <= sc
                or sc + MOTIF_SIZE <= c0
            )
            for sr, sc, _sw, _sh in selected
        )
        if not overlaps:
            selected.append(candidate)
    return selected


def _restore_noncenter_holes(motif: np.ndarray, bg: int) -> frozenset[tuple[int, int]]:
    holes = frozenset(
        (br, bc)
        for br in range(3)
        for bc in range(3)
        if _is_bg_block(_extract_block(motif, br, bc), bg)
    )
    assert _holes_are_legal(holes), "motif hole pattern is not supported"
    for br, bc in holes:
        if (br, bc) == (1, 1):
            continue
        donor = _extract_block(motif, 2 - br, 2 - bc)
        assert not _is_bg_block(donor, bg), "non-center hole lacks a visible opposite donor"
        paste(motif, np.rot90(donor, 2), br * BLOCK, bc * BLOCK)
    return holes


def _center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    main = _major_foreground_color(motif, bg)
    patch = np.full((BLOCK, BLOCK), main, dtype=int)
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    for r in range(BLOCK):
        for c in range(BLOCK):
            if (r, c) == (1, 1):
                continue
            patch[r, c] = int(motif[rows[r], cols[c]])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand two damaged 3x3-block motifs after repairing several missing blocks.

    Input structure:
    - Teal background surrounds two separate 9x9 motifs built from 3x3 blocks.
    - Each motif is a 3x3 block array with the center block missing, plus 1 or 2
      additional missing corner/edge blocks.
    - Every extra hole keeps its point-opposite block visible, so each damaged
      motif still reveals both whole-block symmetry clues and center-border clues.

    Rule:
    - Locate each motif independently, even if extra holes split it into pieces.
    - For each motif, rebuild every non-center hole by rotating the point-opposite
      visible block by 180 degrees, then reconstruct the center block from the
      eight pixels facing the center hole with the center pixel set to the motif's
      majority foreground color.
    - Put the repaired 9x9 motif back in place and copy each non-center block to
      the opposite outer slot of that motif's implied 5x5 block expansion.

    Key insight: the fusion breaks both old shortcuts at once: one global bbox no
    longer isolates a motif, and center reconstruction cannot start before the
    extra holes are repaired. The stable cue is the 9x9 block-grid fingerprint of
    each motif, after which the repair-and-expand logic applies locally.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motifs = _find_motifs(g, bg)
    assert len(motifs) == MOTIF_COUNT, "expected exactly two motifs in this variant"

    out = g.copy()
    for r0, c0, motif, _holes in motifs:
        repaired = motif.copy()
        _restore_noncenter_holes(repaired, bg)
        paste(repaired, _center_patch(repaired, bg), BLOCK, BLOCK)
        paste(out, repaired, r0, c0)

        expanded_r0 = r0 - BLOCK
        expanded_c0 = c0 - BLOCK
        for br in range(3):
            for bc in range(3):
                if (br, bc) == (1, 1):
                    continue
                src = _extract_block(repaired, br, bc)
                dst_r = expanded_r0 + (4 - 2 * br) * BLOCK
                dst_c = expanded_c0 + (4 - 2 * bc) * BLOCK
                paste(out, src, dst_r, dst_c)
    return out


def _major_orbits(orbits, forced_minor, target_count):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits)
        if all(coord not in forced_minor for coord in orbit)
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


def _seed_block(orbits, forced_minor, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_minor, target_count)
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
    b00, corner_minors = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
    b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
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


def _gen_center_patch(motif, majority):
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _expanded_output(full_motif):
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    paste(expanded, full_motif, BLOCK, BLOCK)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = _extract_block(full_motif, br, bc)
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _choose_holes() -> frozenset[tuple[int, int]]:
    extra_count = random.choice((1, 2))
    pairs = random.sample(list(NONCENTER_OPPOSITE_PAIRS), extra_count)
    holes = {(1, 1)}
    for a, b in pairs:
        holes.add(random.choice((a, b)))
    return frozenset(holes)


def _apply_holes(full_motif: np.ndarray, holes: frozenset[tuple[int, int]]) -> np.ndarray:
    out = full_motif.copy()
    for br, bc in holes:
        rs, cs = _block_slice(br, bc)
        out[rs, cs] = BG
    return out


def _rand_gap(n: int) -> int:
    max_gap = min(PLACEMENT_GAP_MAX, n - 2 * EXPANDED_SIZE)
    assert max_gap >= PLACEMENT_GAP_MIN, "grid too small for two expanded motifs"
    return random.randint(PLACEMENT_GAP_MIN, max_gap)


def _choose_expanded_boxes(n: int) -> list[tuple[int, int]]:
    mode = random.choice(LAYOUT_MODES)
    if mode == "horizontal":
        gap = _rand_gap(n)
        total_w = 2 * EXPANDED_SIZE + gap
        r0 = random.randint(0, n - EXPANDED_SIZE)
        c0 = random.randint(0, n - total_w)
        return [(r0, c0), (r0, c0 + EXPANDED_SIZE + gap)]

    if mode == "vertical":
        gap = _rand_gap(n)
        total_h = 2 * EXPANDED_SIZE + gap
        r0 = random.randint(0, n - total_h)
        c0 = random.randint(0, n - EXPANDED_SIZE)
        return [(r0, c0), (r0 + EXPANDED_SIZE + gap, c0)]

    gap_r = _rand_gap(n)
    gap_c = _rand_gap(n)
    total_h = 2 * EXPANDED_SIZE + gap_r
    total_w = 2 * EXPANDED_SIZE + gap_c
    r0 = random.randint(0, n - total_h)
    c0 = random.randint(0, n - total_w)
    if random.choice((False, True)):
        return [(r0, c0), (r0 + EXPANDED_SIZE + gap_r, c0 + EXPANDED_SIZE + gap_c)]
    return [(r0, c0 + EXPANDED_SIZE + gap_c), (r0 + EXPANDED_SIZE + gap_r, c0)]


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
    full_motif = motif.copy()
    paste(full_motif, center, BLOCK, BLOCK)
    expanded = _expanded_output(full_motif)

    holes_a = _choose_holes()
    holes_b = _choose_holes()
    while holes_b == holes_a:
        holes_b = _choose_holes()

    input_motifs = [
        _apply_holes(full_motif, holes_a),
        _apply_holes(full_motif, holes_b),
    ]

    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    for (box_r, box_c), input_motif in zip(_choose_expanded_boxes(n), input_motifs):
        paste(inp, input_motif, box_r + MOTIF_OFFSET, box_c + MOTIF_OFFSET)
        paste(out, expanded, box_r, box_c)

    motifs = _find_motifs(inp, BG)
    if len(motifs) != MOTIF_COUNT:
        return None
    if sorted(len(holes) for _r, _c, _m, holes in motifs) != sorted((len(holes_a), len(holes_b))):
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo variant input")
