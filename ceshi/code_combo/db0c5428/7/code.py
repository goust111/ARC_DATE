# Combo variant of db0c5428 (index 7)
# combo_spec: D3/1+D7/2
# Name: 单方向扩展+仅输出外扩部分

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The input still contains a single 9x9 motif made from eight visible 3x3 blocks
#   around a missing 3x3 center. The old task rebuilt the center and mirrored every
#   visible block to all opposite outer slots of a 5x5 block lattice.
# - This combo keeps the center reconstruction rule but reduces the expansion to one
#   cardinal side only. Example: if the right side is selected, only the left column
#   of source blocks is copied to a new far-right column, with one block-column of
#   background gap preserved between old motif and new copies.
# - D7/2 is then applied on top of that directional layout: the output keeps only the
#   newly created pieces. So the rightward example becomes a 3x5 block frame whose
#   non-background blocks are just the rebuilt center at block slot (1,1) and the
#   three copied blocks at slots (0,4),(1,4),(2,4). The original eight input blocks
#   are removed from the answer.
#
# b) Complexity calibration
# - Only four extension directions exist: up/down/left/right. This keeps the control
#   variable small enough for ARC while still being a genuine branch.
# - A human-visible cue is added inside exactly one side-middle block: an outward bar
#   plus the block center share a unique accent color. That cue is strong enough to
#   reveal the chosen side without relying on hidden color ordering.
# - The output remains sparse but not trivial: solvers must infer both the direction
#   and the old opposite-side copy rule, then also rebuild the center patch.
#
# c) Hidden assumptions audit for main()
# - The original `out = g.copy()` assumption is invalid because the output is now a
#   standalone cropped frame, not the input canvas.
# - The original double loop over all eight source blocks is invalid because only one
#   opposite-side strip is expanded; the generator guarantees a single indicator side.
# - The center reconstruction helper still assumes a 3x3 hole encoded by eight inner
#   boundary pixels. The generator preserves that exact geometry.
# - Direction detection assumes exactly one side-middle block carries the accent-bar
#   signature. The generator enforces uniqueness and keeps the accent color out of all
#   other blocks.
#
# d) Placement legality
# - The input generator still places a single 9x9 motif inside a 16..19 square canvas
#   with at least three background cells of margin, so bbox-based motif recovery is
#   unambiguous.
# - The output is either 9x15 or 15x9 pixels. Because it is built on a fresh canvas,
#   copied blocks never overlap and never clip.
#
# e) Visual observability
# - Across examples, humans can see one side-middle block marked by an outward accent
#   bar. The output shape aligns with that side and retains only the corresponding new
#   outer strip plus the rebuilt center.
# - The copied blocks are visually traceable because they are literal copies of the
#   opposite input row/column, not recolored abstractions.
#
# f) Visual cleanliness
# - The output contains only four separated 3x3 blocks on a rectangular frame, so the
#   center patch and the directional strip do not overlap.
#
# g) Color inference
# - Foreground colors are limited to 3-4 so one accent color can be reserved for the
#   direction cue while the remaining colors still provide enough variation for block
#   identities and center reconstruction.


BLOCK = 3
MOTIF_BLOCKS = 3
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3

BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 3, 4
MAX_RETRIES = 400

SIDE_BLOCK = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}

SIGNATURE_POS = {
    "up": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "down": ((2, 0), (2, 1), (2, 2), (1, 1)),
    "left": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "right": ((0, 2), (1, 2), (2, 2), (1, 1)),
}

SOURCE_SLOTS = {
    "up": ((2, 0), (2, 1), (2, 2)),
    "down": ((0, 0), (0, 1), (0, 2)),
    "left": ((0, 2), (1, 2), (2, 2)),
    "right": ((0, 0), (1, 0), (2, 0)),
}

OUTER_DST_SLOTS = {
    "up": ((0, 0), (0, 1), (0, 2)),
    "down": ((4, 0), (4, 1), (4, 2)),
    "left": ((0, 0), (1, 0), (2, 0)),
    "right": ((0, 4), (1, 4), (2, 4)),
}

CENTER_DST_SLOT = {
    "up": (3, 1),
    "down": (1, 1),
    "left": (1, 3),
    "right": (1, 1),
}

OUTPUT_BLOCK_SHAPE = {
    "up": (5, 3),
    "down": (5, 3),
    "left": (3, 5),
    "right": (3, 5),
}

BOUNDARY_BINDINGS = (
    ((0, 0), (0, 0, 2, 2)),
    ((0, 1), (0, 1, 2, 1)),
    ((0, 2), (0, 2, 2, 0)),
    ((1, 0), (1, 0, 1, 2)),
    ((1, 2), (1, 2, 1, 0)),
    ((2, 0), (2, 0, 0, 2)),
    ((2, 1), (2, 1, 0, 1)),
    ((2, 2), (2, 2, 0, 0)),
)

_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == BLOCK, "this task uses 3x3 blocks"
    major = _major_foreground_color(g, bg)
    patch = np.full((block, block), major, dtype=int)

    hole_r = int(r0) + block
    hole_c = int(c0) + block
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[hole_r + _CENTER_OFFSETS[r], hole_c + _CENTER_OFFSETS[c]])
    return patch


def _block_slice(br: int, bc: int, block: int) -> tuple[slice, slice]:
    return slice(br * block, (br + 1) * block), slice(bc * block, (bc + 1) * block)


def _signature_values(block_arr: np.ndarray, direction: str) -> list[int]:
    return [int(block_arr[r, c]) for r, c in SIGNATURE_POS[direction]]


def _detect_direction(motif: np.ndarray, bg: int, block: int) -> str:
    assert block == BLOCK, "direction cue is defined for 3x3 blocks"
    hits: list[str] = []
    for direction, (br, bc) in SIDE_BLOCK.items():
        rs, cs = _block_slice(br, bc, block)
        vals = _signature_values(motif[rs, cs], direction)
        if vals[0] != int(bg) and len(set(vals)) == 1:
            hits.append(direction)
    assert len(hits) == 1, "expected exactly one directional indicator"
    return hits[0]


def _directional_outer_only_output(
    motif: np.ndarray,
    center: np.ndarray,
    direction: str,
    bg: int,
    block: int,
) -> np.ndarray:
    block_rows, block_cols = OUTPUT_BLOCK_SHAPE[direction]
    out = np.full((block_rows * block, block_cols * block), int(bg), dtype=int)
    for (src_br, src_bc), (dst_br, dst_bc) in zip(SOURCE_SLOTS[direction], OUTER_DST_SLOTS[direction]):
        rs, cs = _block_slice(src_br, src_bc, block)
        paste(out, motif[rs, cs], dst_br * block, dst_bc * block)
    center_br, center_bc = CENTER_DST_SLOT[direction]
    paste(out, center, center_br * block, center_bc * block)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center and return only the one-sided extension it implies.

    Input structure:
    - A single 9x9 motif sits on a teal background as a 3x3 arrangement of 3x3 blocks.
    - The middle 3x3 block is missing, and the eight pixels touching that hole encode
      the border of the absent center block.
    - Exactly one side-middle block carries a directional cue: an outward bar and the
      block center share the same accent color.

    Rule:
    - Use the hole boundary to reconstruct the missing center block; its middle cell is
      the motif's majority foreground color.
    - Read the accent-bar cue to determine whether the motif extends up, down, left, or
      right.
    - Copy only the opposite input row or column to the corresponding far outer side and
      return just those three new blocks plus the rebuilt center. Omit the original eight
      input blocks from the output.

    Key insight: The marker does not tell you which blocks to copy directly; it tells you
    which side of the larger layout is missing. The copied strip therefore comes from the
    opposite side of the input motif, while the center still comes from the hole boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % MOTIF_BLOCKS == 0, "motif must be a square 3x3 block grid"
    block = h // MOTIF_BLOCKS
    assert block == BLOCK, "this combo expects 3x3 blocks"

    motif = crop_bbox(g, motif_bbox)
    direction = _detect_direction(motif, bg, block)
    center = _center_patch(g, int(r0), int(c0), block, bg)
    return _directional_outer_only_output(motif, center, direction, bg, block)


def _random_block(majority: int, fillers: list[int]) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), int(majority), dtype=int)
    coords = [(r, c) for r in range(BLOCK) for c in range(BLOCK)]
    random.shuffle(coords)
    change_count = random.randint(3, 5)
    for r, c in coords[:change_count]:
        block[r, c] = random.choice(fillers)
    return block


def _random_center_patch(majority: int, fillers: list[int]) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), int(majority), dtype=int)
    patch[1, 1] = int(majority)
    boundary = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if (r, c) != (1, 1)]
    non_major_positions = set(random.sample(boundary, random.randint(3, 6)))
    for r, c in boundary:
        if (r, c) in non_major_positions:
            patch[r, c] = random.choice(fillers)
        elif random.random() < 0.25:
            patch[r, c] = random.choice(fillers)
    return patch


def _apply_indicator(block_arr: np.ndarray, direction: str, accent: int) -> None:
    for r, c in SIGNATURE_POS[direction]:
        block_arr[r, c] = int(accent)


def _break_signature(block_arr: np.ndarray, direction: str, palette: list[int]) -> None:
    vals = _signature_values(block_arr, direction)
    if len(set(vals)) != 1:
        return
    r, c = SIGNATURE_POS[direction][0]
    choices = [color for color in palette if color != vals[0]]
    block_arr[r, c] = random.choice(choices)


def _inject_center_boundary(blocks: dict[tuple[int, int], np.ndarray], center: np.ndarray) -> None:
    for (pr, pc), (br, bc, lr, lc) in BOUNDARY_BINDINGS:
        blocks[(br, bc)][lr, lc] = int(center[pr, pc])


def _assemble_motif(blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                continue
            paste(motif, blocks[(br, bc)], br * BLOCK, bc * BLOCK)
    return motif


def _distinct_source_strip(blocks: dict[tuple[int, int], np.ndarray], direction: str) -> bool:
    seen = {
        tuple(blocks[pos].reshape(-1).tolist())
        for pos in SOURCE_SLOTS[direction]
    }
    return len(seen) == 3


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    direction = random.choice(tuple(SIDE_BLOCK))
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    accent = colors[1]
    fillers = colors[2:]
    assert fillers, "generator reserves one accent color and needs at least one filler"

    blocks: dict[tuple[int, int], np.ndarray] = {}
    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                continue
            blocks[(br, bc)] = _random_block(majority, fillers)

    center = _random_center_patch(majority, fillers)
    _inject_center_boundary(blocks, center)

    cue_pos = SIDE_BLOCK[direction]
    _apply_indicator(blocks[cue_pos], direction, accent)

    palette_wo_accent = [majority] + fillers
    for side_name, pos in SIDE_BLOCK.items():
        if side_name == direction:
            continue
        _break_signature(blocks[pos], side_name, palette_wo_accent)

    for pos in SOURCE_SLOTS[direction]:
        if np.all(blocks[pos] == majority):
            blocks[pos][0, 0] = fillers[0]

    if not _distinct_source_strip(blocks, direction):
        return None

    motif = _assemble_motif(blocks)
    expected = _directional_outer_only_output(motif, center, direction, BG, BLOCK)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
