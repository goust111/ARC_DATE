# Variant of db0c5428
# Dimension: D9 — 图案在网格中的位置 (meta)
# Change: 图案嵌入在较大网格中，位置不固定（各示例偏移不同），周围填充背景色8... → 网格中包含多个相同图案

# Design analysis
# - Concrete form: each input now contains two disconnected copies of the same
#   9x9 motif. Each motif is still a 3x3 arrangement of 3x3 blocks with a teal
#   3x3 hole in the middle. The output keeps both original motifs in place and
#   expands each one into its own 15x15 region, so the canvas must be larger
#   than the original task to fit two non-overlapping expansions.
# - Behavior trace: the original solver called bbox(g != bg), so a single box
#   around all foreground was enough. With two motifs that same line would wrap
#   both copies into one large rectangle, making h != w or block != 3. The new
#   solver therefore finds connected components first, crops each 9x9 motif
#   independently, reconstructs its center, and pastes its outer mirrored blocks
#   into the local 15x15 neighborhood of that motif only.
# - Complexity calibration: the variant changes only multiplicity and placement.
#   Each sample always has exactly two motifs, and they are exact duplicates, so
#   the visual novelty is obvious without introducing extra matching rules. The
#   relative layout is limited to three repeated modes (horizontal, vertical,
#   diagonal), which gives visible variation while keeping the inference space
#   small enough for ARC.
# - Hidden assumptions audited from the old main():
#   1. bbox(g != bg) assumed one connected foreground object; invalid now.
#   2. _major_foreground_color(g, bg) used the whole grid; harmless for one motif
#      but incorrect if different motifs ever had different color frequencies.
#      The new code computes the majority per motif crop.
#   3. Reading source blocks from g while writing to out assumed one target
#      region. The new code still reads only from the untouched motif crop, so
#      later pastes cannot corrupt subsequent source reads.
# - Placement legality: the generator samples two 15x15 expanded boxes that are
#   guaranteed disjoint with at least one background row/column between them.
#   Input motifs are pasted at (+3, +3) inside those boxes, so every expansion
#   stays in bounds and the two outputs never overlap.
# - Visual observability: humans can see two identical hollow motifs in the
#   input, then see each one independently grow into the same larger 5x5 block
#   lattice in the output. No invisible ordering, color-id arithmetic, or
#   per-instance exceptions are used.
# - Visual cleanliness: only two objects exist, both disconnected on input and
#   with disjoint 15x15 output territories. This avoids ambiguous overlaps.
# - New solver assumptions: every non-background connected component is one 9x9
#   motif with a 3x3 teal center hole, and the input contains exactly two such
#   components. The generator enforces both conditions directly.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    block = motif.shape[0] // 3
    assert block == 3, "this task uses 3x3 blocks"
    main = _major_foreground_color(motif, bg)
    patch = np.full((block, block), main, dtype=int)

    hole_r = block
    hole_c = block
    offsets = (-1, 1, 3)
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(motif[hole_r + offsets[r], hole_c + offsets[c]])
    return patch


def _find_motifs(g: np.ndarray, bg: int) -> list[tuple[int, int, np.ndarray]]:
    labels, count = label_mask(g != bg, connectivity=4)
    motifs: list[tuple[int, int, np.ndarray]] = []
    for idx in range(1, count + 1):
        comp_bbox = bbox(labels == idx)
        assert comp_bbox is not None, "foreground component vanished"
        r0, c0, r1, c1 = comp_bbox
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h == w and h % 3 == 0, "each component must be a square 3x3 block grid"
        block = h // 3
        assert block == 3, "this variant keeps 3x3 blocks"
        motif = crop_bbox(g, comp_bbox)
        assert np.all(motif[block : 2 * block, block : 2 * block] == bg), "motif center must be empty"
        motifs.append((int(r0), int(c0), motif))
    motifs.sort(key=lambda item: (item[0], item[1]))
    return motifs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each disconnected hollow motif into its own 5x5 block halo.

    Input structure:
    - Teal background surrounds two separate copies of the same 9x9 motif.
    - Each copy is a 3x3 array of 3x3 blocks whose center block is empty teal.
    - The colored pixels facing the hole inside each copy encode that copy's
      missing center block.

    Rule:
    - Leave both 9x9 motifs where they are.
    - For each motif independently, copy its eight visible 3x3 blocks to the
      opposite outer slots of the implied 5x5 block grid, without rotation.
    - Rebuild each missing center block from the eight pixels around its hole,
      with the center pixel set to that motif's majority foreground color.

    Key insight: the task is no longer "find the only motif" but "segment the
    grid into motif components first"; once each 9x9 component is isolated, the
    original expansion rule applies locally and independently to every copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motifs = _find_motifs(g, bg)
    assert len(motifs) == 2, "expected exactly two motifs in this variant"

    out = g.copy()
    for r0, c0, motif in motifs:
        block = motif.shape[0] // 3
        expanded_r0 = r0 - block
        expanded_c0 = c0 - block

        for br in range(3):
            for bc in range(3):
                if (br, bc) == (1, 1):
                    continue
                src = motif[br * block : (br + 1) * block, bc * block : (bc + 1) * block]
                dst_r = expanded_r0 + (4 - 2 * br) * block
                dst_c = expanded_c0 + (4 - 2 * bc) * block
                paste(out, src, dst_r, dst_c)

        paste(out, _center_patch(motif, bg), r0 + block, c0 + block)
    return out


# --------------- generator (o5x) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 31, 36       # larger canvas needed for two 15x15 outputs
BLOCK = 3
MOTIF_SIZE = BLOCK * 3                       # 9x9 motif
EXPANDED_SIZE = BLOCK * 5                    # 15x15 local output region
MOTIF_OFFSET = BLOCK                          # motif sits at (+3, +3) inside its expanded box
MOTIF_COUNT = 2
PLACEMENT_GAP_MIN, PLACEMENT_GAP_MAX = 1, 3

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
LAYOUT_MODES = ("horizontal", "vertical", "diagonal")

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


def _rand_gap(n: int) -> int:
    max_gap = min(PLACEMENT_GAP_MAX, n - 2 * EXPANDED_SIZE)
    assert max_gap >= PLACEMENT_GAP_MIN, "grid too small for two expanded motifs"
    return random.randint(PLACEMENT_GAP_MIN, max_gap)


def _choose_expanded_boxes(n: int) -> tuple[str, list[tuple[int, int]]]:
    mode = random.choice(LAYOUT_MODES)
    if mode == "horizontal":
        gap = _rand_gap(n)
        total_w = 2 * EXPANDED_SIZE + gap
        r0 = random.randint(0, n - EXPANDED_SIZE)
        c0 = random.randint(0, n - total_w)
        return mode, [(r0, c0), (r0, c0 + EXPANDED_SIZE + gap)]

    if mode == "vertical":
        gap = _rand_gap(n)
        total_h = 2 * EXPANDED_SIZE + gap
        r0 = random.randint(0, n - total_h)
        c0 = random.randint(0, n - EXPANDED_SIZE)
        return mode, [(r0, c0), (r0 + EXPANDED_SIZE + gap, c0)]

    gap_r = _rand_gap(n)
    gap_c = _rand_gap(n)
    total_h = 2 * EXPANDED_SIZE + gap_r
    total_w = 2 * EXPANDED_SIZE + gap_c
    r0 = random.randint(0, n - total_h)
    c0 = random.randint(0, n - total_w)
    if random.choice((False, True)):
        return mode, [(r0, c0), (r0 + EXPANDED_SIZE + gap_r, c0 + EXPANDED_SIZE + gap_c)]
    return mode, [(r0, c0 + EXPANDED_SIZE + gap_c), (r0 + EXPANDED_SIZE + gap_r, c0)]


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
    expanded = _expanded_output(motif, center)

    _mode, boxes = _choose_expanded_boxes(n)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    for box_r, box_c in boxes:
        paste(inp, motif, box_r + MOTIF_OFFSET, box_c + MOTIF_OFFSET)
        paste(out, expanded, box_r, box_c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D9 input")
