# Combo variant of db0c5428 (index 44)
# combo_spec: D13/1+D14/1
# Name: 迭代外扩+插值填充空白

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The original task reconstructs the missing center block and copies the 8 visible
#    directional blocks outward once, yielding a sparse 5x5 block layout. This combo
#    first makes the center explicit, then treats that completed 3x3 as a dense seed.
#    Round 1 expands the seed to a sparse 5x5 by shifting each perimeter block one
#    step farther outward along the sign of its offset from the center. The 8 gaps left
#    between copied blocks are filled by interpolation: horizontal gaps alternate the
#    touching edge columns of their left/right neighbors, and vertical gaps alternate
#    the touching edge rows of their up/down neighbors. Round 2 repeats the same logic
#    from the now-dense 5x5 to a sparse 7x7, then fills the new 8 outer gaps with the
#    same interpolation rule. Because half of the 5x5 perimeter blocks were created by
#    Round-1 interpolation, the second expansion copies interpolated blocks outward,
#    creating the required cascade.
# b) Complexity calibration:
#    The result is visibly different from both the original sparse 5x5 and the single
#    D14 ring-only 7x7: the final answer is a fully dense 7x7 block grid whose outer
#    ring contains striped blocks derived from edge interpolation, not only direct
#    copies of the original 8 source blocks. The rule still uses one fixed block size,
#    one fixed interpolation pattern, and exactly two expansion rounds, so the task is
#    richer but still learnable from a few examples.
# c) Review of old assumptions against the new rule:
#    - `bbox(g != bg)` still assumes one isolated foreground motif. The generator keeps
#      that invariant.
#    - The original `block == 3` assumption still matters because center recovery and
#      interpolation both read fixed rows/cols from 3x3 source blocks.
#    - The one-pass "copy to opposite slot" logic no longer suffices. Expansion must be
#      modeled as a dense-grid process: embed the old grid, push its perimeter outward,
#      then fill the axis-aligned holes before the next round.
#    - The final region is now a dense 7x7 block grid, so the generator must reserve a
#      full 21x21 output box, not only the sparse footprint used by the original task.
#    - The dimension note for D13 mentions 12 gaps in the 5x5 stage, but the shipped
#      solver/spec reconstructs the center block before expansion, which leaves 8 block
#      gaps. The combo therefore follows the executable base task and fills those 8
#      stage-1 gaps plus 8 stage-2 gaps.
# d) Placement legality:
#    The generator samples a square canvas large enough for the final 21x21 region with
#    at least one background cell of padding. The 9x9 input motif is placed so that the
#    expanded 7x7 output starts exactly two block widths above/left of the motif bbox.
# e) Visual observability:
#    A human can see three simultaneous cues: the missing center is reconstructed, the
#    pattern grows twice, and gap blocks show deterministic stripes built from adjacent
#    touching edges. Across samples, those stripes repeat in both rounds, making the
#    interpolation rule visible rather than hidden in color ids.
# f) Visual cleanliness:
#    The output is a clean 7x7 block tessellation with no overlaps. Every block-sized
#    slot contains exactly one block, and interpolation is confined to the previously
#    empty slots, so provenance remains easy to parse.
# g) Color inferability:
#    No arbitrary color remapping is introduced. Interpolated blocks reuse only colors
#    already present on adjacent source edges. The generator rejects degenerate cases
#    where too many interpolated blocks collapse into direct copies, keeping the combo
#    behavior visually recoverable.


BLOCK = 3
MOTIF_BLOCKS = 3
STAGE1_BLOCKS = 5
OUTPUT_BLOCKS = 7
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
EXPANDED_SIZE = BLOCK * OUTPUT_BLOCKS
_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE), "motif must be 9x9"
    main = _major_foreground_color(motif, bg)
    patch = motif[np.ix_(BLOCK + _CENTER_OFFSETS, BLOCK + _CENTER_OFFSETS)].copy()
    patch[1, 1] = main
    return patch


def _empty_block_grid(size: int) -> list[list[np.ndarray | None]]:
    return [[None for _ in range(size)] for _ in range(size)]


def _copy_block(block: np.ndarray) -> np.ndarray:
    return np.array(block, dtype=int, copy=True)


def _motif_block_grid(motif: np.ndarray, center: np.ndarray) -> list[list[np.ndarray]]:
    blocks = _empty_block_grid(MOTIF_BLOCKS)
    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                blocks[br][bc] = _copy_block(center)
            else:
                blocks[br][bc] = _copy_block(
                    motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
                )
    return blocks  # type: ignore[return-value]


def _is_perimeter(r: int, c: int, size: int) -> bool:
    return r in (0, size - 1) or c in (0, size - 1)


def _step_toward_outside(delta: int) -> int:
    if delta < 0:
        return -1
    if delta > 0:
        return 1
    return 0


def _expand_dense_grid(
    blocks: list[list[np.ndarray]],
) -> tuple[list[list[np.ndarray | None]], dict[tuple[int, int], tuple[int, int]]]:
    size = len(blocks)
    center = size // 2
    out = _empty_block_grid(size + 2)

    for r in range(size):
        for c in range(size):
            out[r + 1][c + 1] = _copy_block(blocks[r][c])

    copied_from: dict[tuple[int, int], tuple[int, int]] = {}
    new_center = center + 1
    for r in range(size):
        for c in range(size):
            if not _is_perimeter(r, c, size):
                continue
            dr = r - center
            dc = c - center
            dst_r = new_center + dr + _step_toward_outside(dr)
            dst_c = new_center + dc + _step_toward_outside(dc)
            out[dst_r][dst_c] = _copy_block(blocks[r][c])
            copied_from[(dst_r, dst_c)] = (r, c)
    return out, copied_from


def _interpolate_horizontal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    block = np.empty((BLOCK, BLOCK), dtype=int)
    left_edge = left[:, -1]
    right_edge = right[:, 0]
    for c in range(BLOCK):
        block[:, c] = left_edge if c % 2 == 0 else right_edge
    return block


def _interpolate_vertical(up: np.ndarray, down: np.ndarray) -> np.ndarray:
    block = np.empty((BLOCK, BLOCK), dtype=int)
    up_edge = up[-1, :]
    down_edge = down[0, :]
    for r in range(BLOCK):
        block[r, :] = up_edge if r % 2 == 0 else down_edge
    return block


def _fill_interpolation_gaps(
    blocks: list[list[np.ndarray | None]],
) -> tuple[list[list[np.ndarray]], list[tuple[int, int]]]:
    size = len(blocks)
    out = [
        [None if blocks[r][c] is None else _copy_block(blocks[r][c]) for c in range(size)]
        for r in range(size)
    ]
    interpolated: list[tuple[int, int]] = []

    while True:
        empties = [(r, c) for r in range(size) for c in range(size) if out[r][c] is None]
        if not empties:
            return out, interpolated  # type: ignore[return-value]

        changed = False
        for r, c in empties:
            has_lr = 0 < c < size - 1 and out[r][c - 1] is not None and out[r][c + 1] is not None
            has_ud = 0 < r < size - 1 and out[r - 1][c] is not None and out[r + 1][c] is not None
            if has_lr == has_ud:
                continue
            if has_lr:
                out[r][c] = _interpolate_horizontal(out[r][c - 1], out[r][c + 1])  # type: ignore[arg-type]
            else:
                out[r][c] = _interpolate_vertical(out[r - 1][c], out[r + 1][c])  # type: ignore[arg-type]
            interpolated.append((r, c))
            changed = True

        if not changed:
            raise AssertionError("unfillable interpolation gap")


def _render_block_grid(blocks: list[list[np.ndarray]], bg: int) -> np.ndarray:
    size = len(blocks)
    canvas = np.full((size * BLOCK, size * BLOCK), bg, dtype=int)
    for r in range(size):
        for c in range(size):
            paste(canvas, blocks[r][c], r * BLOCK, c * BLOCK)
    return canvas


def _build_trace(motif: np.ndarray, bg: int) -> dict[str, object]:
    center = _center_patch(motif, bg)
    stage0 = _motif_block_grid(motif, center)

    stage1_sparse, stage1_copied_from = _expand_dense_grid(stage0)
    stage1_dense, stage1_interp = _fill_interpolation_gaps(stage1_sparse)

    stage2_sparse, stage2_copied_from = _expand_dense_grid(stage1_dense)
    stage2_dense, stage2_interp = _fill_interpolation_gaps(stage2_sparse)

    stage1_interp_set = set(stage1_interp)
    cascade_copies = [
        dst for dst, src in stage2_copied_from.items() if src in stage1_interp_set
    ]

    return {
        "center": center,
        "stage0": stage0,
        "stage1_sparse": stage1_sparse,
        "stage1_dense": stage1_dense,
        "stage1_interp": stage1_interp,
        "stage1_copied_from": stage1_copied_from,
        "stage2_sparse": stage2_sparse,
        "stage2_dense": stage2_dense,
        "stage2_interp": stage2_interp,
        "stage2_copied_from": stage2_copied_from,
        "cascade_copies": cascade_copies,
    }


def _expanded_output(motif: np.ndarray, bg: int) -> np.ndarray:
    trace = _build_trace(motif, bg)
    return _render_block_grid(trace["stage2_dense"], bg)  # type: ignore[arg-type]


def _novel_interp_count(
    blocks: list[list[np.ndarray]], positions: list[tuple[int, int]]
) -> int:
    size = len(blocks)
    novel = 0
    for r, c in positions:
        block = blocks[r][c]
        if 0 < c < size - 1:
            if not np.array_equal(block, blocks[r][c - 1]) and not np.array_equal(block, blocks[r][c + 1]):
                novel += 1
        elif 0 < r < size - 1:
            if not np.array_equal(block, blocks[r - 1][c]) and not np.array_equal(block, blocks[r + 1][c]):
                novel += 1
    return novel


def _fusion_metrics(motif: np.ndarray, bg: int) -> dict[str, int]:
    trace = _build_trace(motif, bg)
    stage1_dense = trace["stage1_dense"]  # type: ignore[assignment]
    stage2_dense = trace["stage2_dense"]  # type: ignore[assignment]
    stage1_interp = trace["stage1_interp"]  # type: ignore[assignment]
    stage2_interp = trace["stage2_interp"]  # type: ignore[assignment]
    cascade_copies = trace["cascade_copies"]  # type: ignore[assignment]
    return {
        "stage1_interp_count": len(stage1_interp),
        "stage2_interp_count": len(stage2_interp),
        "cascade_copy_count": len(cascade_copies),
        "novel_stage1": _novel_interp_count(stage1_dense, stage1_interp),
        "novel_stage2": _novel_interp_count(stage2_dense, stage2_interp),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a missing-center motif, expand it twice, and interpolate the new gaps.

    Input structure:
    - One 9x9 foreground motif sits on a uniform background.
    - The motif is a 3x3 array of 3x3 blocks with its center block missing.
    - The eight pixels bordering the hole still encode the missing center block.

    Rule:
    - Reconstruct the center block from those inner-boundary pixels, keeping the
      motif's majority foreground color at the center of that 3x3 patch.
    - Treat the completed 3x3 block grid as a dense seed. Expand it to 5x5 by
      pushing every perimeter block one step farther outward, then fill each new
      empty slot by alternating the touching edge row/column of the two opposite
      neighbors.
    - Repeat the same expand-and-fill process from 5x5 to 7x7, again without
      rotating copied blocks.

    Key insight:
    - The first interpolation round is part of the source for the second round.
      Some outermost 7x7 blocks come from already-interpolated 5x5 perimeter
      blocks, so the final answer is a cascade of outward copying plus edge-based
      stripe interpolation rather than a single sparse reflection shell.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h == MOTIF_SIZE, "motif must be a 9x9 square"

    motif = crop_bbox(g, motif_bbox)
    expanded = _expanded_output(motif, bg)

    out = g.copy()
    paste(out, expanded, int(r0) - 2 * BLOCK, int(c0) - 2 * BLOCK)
    return out


# --------------- generator (combo: iterative_cascade_interpolation) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 24, 27
OUTPUT_PAD = 1

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

# -- CFG: generation --
MAX_RETRIES = 600
MIN_NOVEL_STAGE1 = 4
MIN_NOVEL_STAGE2 = 4

# Diagonal-symmetric orbits (b00 == b00.T)
DIAG_ORBITS = (
    ((0, 0),),
    ((1, 1),),
    ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)

# LR-symmetric orbits (b01 == fliplr(b01))
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)


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


def _is_non_degenerate_combo_sample(motif: np.ndarray) -> bool:
    metrics = _fusion_metrics(motif, BG)
    return (
        metrics["stage1_interp_count"] == 8
        and metrics["stage2_interp_count"] == 8
        and metrics["cascade_copy_count"] == 8
        and metrics["novel_stage1"] >= MIN_NOVEL_STAGE1
        and metrics["novel_stage2"] >= MIN_NOVEL_STAGE2
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
    if not _is_non_degenerate_combo_sample(motif):
        return None
    expanded = _expanded_output(motif, BG)

    min_pos = 2 * BLOCK + OUTPUT_PAD
    max_pos = n - EXPANDED_SIZE - OUTPUT_PAD + 2 * BLOCK
    if min_pos > max_pos:
        return None
    r0 = random.randint(min_pos, max_pos)
    c0 = random.randint(min_pos, max_pos)

    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - 2 * BLOCK, c0 - 2 * BLOCK)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
