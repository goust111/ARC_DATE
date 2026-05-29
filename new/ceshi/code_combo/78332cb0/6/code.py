from __future__ import annotations

# Combo variant of 78332cb0 (index 6)
# combo_spec: D5/0+D14/1
# Name: 颜色重映射与位置编码

"""
Design analysis
===============

spec:
  preserved_structure:
    background: 7
    separator: 6
    blocks:
      count: 3-4
      size: 5x5
      shape: single connected monochrome sprite on bg
      chain_geometry: exactly one of horizontal / vertical is a valid strip
    layouts:
      input: scrambled strip or 2x2 grid when n=4
      output: ordered 1xN strip with separator lines

  fusion_rule:
    D14_1_color_position_code:
      - each block still has exactly one foreground color
      - chain colors are strictly increasing from head to tail
      - the generator restricts this cue to the tiny fixed ladder {2,5,8,9}
        so examples repeatedly teach a human-visible order instead of relying on
        arbitrary one-off colors
    D5_0_recolor_after_recovery:
      - after recovering the chain, block k is recolored to POS_COLORS[k]
      - the output palette is fixed to [1,2,3,4]
    synergy:
      - color rank recovers the chain order
      - edge contacts are still needed to decide whether that recovered order
        should be rendered as a horizontal or vertical strip
      - output colors therefore depend on the recovered order, not on the
        original block colors

  behavior_trace:
    original_flow:
      - split the scrambled layout into 5x5 blocks
      - search permutations until one edge-matching chain is found
      - render the blocks in that order with original colors unchanged
    fused_flow:
      - split the scrambled layout into 5x5 blocks
      - read each block's single foreground color and sort blocks by
        color rank, smallest to largest
      - test that color-sorted list against horizontal and vertical edge
        contacts; exactly one orientation must fit
      - recolor the 1st/2nd/3rd/4th chain blocks to 1/2/3/4
      - render the recolored strip in the validated orientation
    concrete_example:
      - if the scrambled input colors are [8,2,9,5], the hidden chain order is
        the blocks colored [2,5,8,9]
      - suppose only their left/right contacts align, not their top/bottom
      - then the output is a horizontal strip of the same shapes recolored to
        [1,2,3,4]
      - compared with the source task, color no longer survives to output;
        compared with plain recolor, the input colors now actively encode order

  complexity_calibration:
    - the variant is not reduced to pure color sorting: the solver still needs
      geometry to distinguish horizontal from vertical strip output
    - the input color cue has only four repeatable states {2,5,8,9}; this keeps
      the mapping learnable from 2-4 examples
    - the generator rejects any sample where the recolored output would equal
      the original ordered strip, preventing collapse back to the base task

  implicit_assumption_audit:
    - separator detection still assumes full separator rows/cols of a single
      non-background color; generator preserves that
    - block extraction still assumes either strip layout or a 2x2 quadrant split
      around one separator row and one separator col; generator preserves that
    - edge-contact logic still assumes each sprite has no accidental foreground
      on the two edges perpendicular to the true chain direction; generator
      preserves that when building masks
    - new assumption: each block must have exactly one non-background color and
      those colors must be pairwise distinct; generator enforces the fixed
      strictly increasing color ladder
    - new assumption: the color-sorted order must validate in exactly one
      orientation; generator checks this with the same solver

  placement_legality:
    - every sprite stays inside its own 5x5 slot
    - strip layouts use one full separator row/col between neighbors
    - 2x2 layouts place the four blocks into non-overlapping quadrants around a
      single separator cross
    - generation is output-first and rejected unless main(input) reproduces the
      intended recolored output

  visual_observability:
    - the visible difference from the source task is strong: the output always
      uses the same positional ramp [1,2,3,4], while the input uses the repeated
      cue ladder {2,5,8,9}
    - because the input ladder is fixed and reused across many samples, a human
      can infer "red/grey/teal/maroon indicate earlier to later chain slots"
      from examples instead of needing access to hidden numeric ids
    - the strip orientation still varies, so examples show that color order alone
      does not decide whether the result is horizontal or vertical

  visual_neatness:
    - no extra markers or overlays are introduced
    - recoloring affects whole sprites only, so output remains a clean strip
    - restricting to monochrome blocks prevents mixed-color clutter inside slots

  color_change_inference:
    - output recolor uses only four fixed colors, one per chain position
    - input order cue also uses only a tiny repeated color family, making the
      "input color rank -> output position color" relationship easy to compare
      across samples

  generator_validation:
    - build the ordered chain first with monotone input colors
    - verify that exactly one structural orientation admits a valid chain
    - scramble the blocks for the input layout
    - reject unless the fused solver returns the recolored target and that target
      differs from the plain original-color strip

  post_edit_main_assumptions:
    - `_block_color` assumes one monochrome foreground per block
    - `_ordered_blocks_by_color` assumes those block colors are distinct
    - `_is_chain_in_orientation` assumes the generator has zero perpendicular
      edge contacts for the true orientation
    - `generate_input` explicitly constructs and checks all three assumptions
"""

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


BG = 7
SEP = 6
INPUT_COLOR_POOL = [2, 5, 8, 9]
POS_COLORS = [1, 2, 3, 4]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
FOUR_BLOCK_PROB = 0.7
MAX_RETRIES = 300

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_PERP_EDGES = {"h": ("top", "bot"), "v": ("left", "right")}
_OUTER_EDGES = {"h": ("left", "right"), "v": ("top", "bot")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _find_sep(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for r in range(int(grid.shape[0])):
        if bool((grid[int(r), :] == int(grid[int(r), 0])).all()) and int(grid[int(r), 0]) != int(bg):
            candidates.append(int(grid[int(r), 0]))
    for c in range(int(grid.shape[1])):
        if bool((grid[:, int(c)] == int(grid[0, int(c)])).all()) and int(grid[0, int(c)]) != int(bg):
            candidates.append(int(grid[0, int(c)]))
    assert candidates, "separator lines not found"
    return int(Counter(candidates).most_common(1)[0][0])


def _full_sep_rows(grid: np.ndarray, sep: int) -> list[int]:
    return [int(r) for r in range(int(grid.shape[0])) if bool((grid[int(r), :] == int(sep)).all())]


def _full_sep_cols(grid: np.ndarray, sep: int) -> list[int]:
    return [int(c) for c in range(int(grid.shape[1])) if bool((grid[:, int(c)] == int(sep)).all())]


def _split_by_rows(grid: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(grid, sep)
    segments: list[np.ndarray] = []
    start = 0
    for r in rows + [int(grid.shape[0])]:
        if int(r) > int(start):
            segments.append(grid[int(start):int(r), :].copy())
        start = int(r + 1)
    return segments


def _split_by_cols(grid: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(grid, sep)
    segments: list[np.ndarray] = []
    start = 0
    for c in cols + [int(grid.shape[1])]:
        if int(c) > int(start):
            segments.append(grid[:, int(start):int(c)].copy())
        start = int(c + 1)
    return segments


def _split_blocks(grid: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(grid, sep)
    cols = _full_sep_cols(grid, sep)
    if rows and cols:
        r = int(rows[0])
        c = int(cols[0])
        return [
            grid[:r, :c].copy(),
            grid[:r, int(c + 1):].copy(),
            grid[int(r + 1):, :c].copy(),
            grid[int(r + 1):, int(c + 1):].copy(),
        ]
    if rows:
        return _split_by_rows(grid, sep)
    if cols:
        return _split_by_cols(grid, sep)
    return []


def _block_color(block: np.ndarray, bg: int) -> int:
    colors = sorted(int(v) for v in np.unique(block) if int(v) != int(bg))
    assert len(colors) == 1, "each block must be monochrome"
    return int(colors[0])


def _edge_sets(block: np.ndarray, bg: int) -> dict[str, set[int]]:
    mask = block != int(bg)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _ordered_blocks_by_color(blocks: list[np.ndarray], bg: int) -> list[np.ndarray]:
    colors = [_block_color(block, bg) for block in blocks]
    assert len(set(colors)) == len(colors), "block colors must encode a strict order"
    return [blocks[i] for i in sorted(range(len(blocks)), key=lambda i: int(colors[i]))]


def _is_chain_in_orientation(blocks: list[np.ndarray], bg: int, orient: str) -> bool:
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]
    outer_head, outer_tail = _OUTER_EDGES[orient]
    perp_a, perp_b = _PERP_EDGES[orient]

    if contacts[0][outer_head] or contacts[-1][outer_tail]:
        return False
    for edge in contacts:
        if edge[perp_a] or edge[perp_b]:
            return False
    return all(bool(contacts[i][lead] & contacts[i + 1][trail]) for i in range(len(blocks) - 1))


def _recolor_block(block: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = block.copy()
    out[out != int(bg)] = int(color)
    return out


def _render_strip(blocks: list[np.ndarray], orient: str, bg: int, sep: int) -> np.ndarray:
    n = int(len(blocks))
    stride = BLOCK_SIZE + 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, int(n * BLOCK_SIZE + (n - 1))), int(bg), dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, int(i * stride))
            if int(i) < int(n - 1):
                out[:, int(i * stride + BLOCK_SIZE)] = int(sep)
        return out

    out = np.full((int(n * BLOCK_SIZE + (n - 1)), BLOCK_SIZE), int(bg), dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, int(i * stride), 0)
        if int(i) < int(n - 1):
            out[int(i * stride + BLOCK_SIZE), :] = int(sep)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the chain by reading block colors as position codes, then recolor the
    recovered strip by chain rank.

    Input structure:
    - The grid contains 3-4 separate 5x5 monochrome sprites on background 7.
    - Full separator rows or columns of color 6 partition the sprites into a
      scrambled strip or a 2x2 layout.
    - Each sprite color is distinct, and smaller colors represent earlier chain
      positions than larger colors.

    Rule:
    - Split the input into blocks and sort them by foreground color from small to
      large to recover the chain order.
    - Use edge-contact overlap to determine whether that recovered order forms a
      horizontal chain or a vertical chain.
    - Recolor the 1st/2nd/3rd/4th chain blocks to 1/2/3/4 and render them as a
      single ordered strip with separator lines.

    Key insight:
    - Color now encodes who comes first, middle, and last, while edge matching
      only resolves which strip orientation the color-sorted chain should take.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)

    blocks = _split_blocks(g, sep)
    assert blocks, "must extract at least one block"
    assert len(blocks) <= len(POS_COLORS), "position palette too short"

    ordered = _ordered_blocks_by_color(blocks, bg)
    chain_h = _is_chain_in_orientation(ordered, bg, "h")
    chain_v = _is_chain_in_orientation(ordered, bg, "v")

    assert chain_h != chain_v, "color-sorted chain must fit exactly one orientation"
    orient = "h" if chain_h else "v"
    recolored = [_recolor_block(block, bg, POS_COLORS[i]) for i, block in enumerate(ordered)]
    return _render_strip(recolored, orient, bg, sep)


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_edge_sets(block, BG) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _interior_frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    interior = mask[1:4, 1:4]
    has_neighbor = np.zeros_like(interior)
    has_neighbor[1:, :] |= interior[:-1, :]
    has_neighbor[:-1, :] |= interior[1:, :]
    has_neighbor[:, 1:] |= interior[:, :-1]
    has_neighbor[:, :-1] |= interior[:, 1:]
    has_neighbor[0, :] |= mask[0, 1:4]
    has_neighbor[2, :] |= mask[4, 1:4]
    has_neighbor[:, 0] |= mask[1:4, 0]
    has_neighbor[:, 2] |= mask[1:4, 4]
    candidates = has_neighbor & ~interior
    rs, cs = np.where(candidates)
    return [(int(r) + 1, int(c) + 1) for r, c in zip(rs, cs)]


def _base_mask(left_offset: int | None, right_offset: int | None, target_area: int) -> np.ndarray | None:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2):max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:BLOCK_SIZE] = True
        mask[min(right_offset, 2):max(right_offset, 2) + 1, 2] = True

    if mask.sum() < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    while mask.sum() < target_area:
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    if left_offset is None:
        if mask[:, 0].any():
            return None
    elif np.where(mask[:, 0])[0].tolist() != [left_offset]:
        return None

    if right_offset is None:
        if mask[:, BLOCK_SIZE - 1].any():
            return None
    elif np.where(mask[:, BLOCK_SIZE - 1])[0].tolist() != [right_offset]:
        return None

    if mask[0].any() or mask[BLOCK_SIZE - 1].any():
        return None
    if any(mask[r].all() for r in range(BLOCK_SIZE)):
        return None
    if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_block(
    color: int,
    *,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
) -> np.ndarray | None:
    if top is not None or bottom is not None:
        mask = _base_mask(bottom, top, random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None
        mask = np.rot90(mask, 1)
    else:
        mask = _base_mask(left, right, random.randint(FG_AREA_MIN, FG_AREA_MAX))
        if mask is None:
            return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        slots = [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]
        for block, (r0, c0) in zip(scrambled, slots):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient], BG, SEP)


def _expected_output(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    recolored = [_recolor_block(block, BG, POS_COLORS[i]) for i, block in enumerate(blocks)]
    return _render_strip(recolored, orient, BG, SEP)


def _gen_attempt() -> np.ndarray | None:
    n = 4 if random.random() < FOUR_BLOCK_PROB else 3
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)
    chain_colors = sorted(random.sample(INPUT_COLOR_POOL, n))

    lead_kw, trail_kw = _ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i, color in enumerate(chain_colors):
        kw = {}
        if i > 0:
            kw[lead_kw] = shared_offsets[i - 1]
        if i < n - 1:
            kw[trail_kw] = shared_offsets[i]
        block = _make_block(color, **kw)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, orient) != 1:
        return None
    if _count_orders(blocks, _OTHER[orient]) != 0:
        return None
    if not _is_chain_in_orientation(blocks, BG, orient):
        return None
    if [_block_color(block, BG) for block in blocks] != chain_colors:
        return None

    plain_out = _render_strip(blocks, orient, BG, SEP)
    expected = _expected_output(blocks, orient)
    if np.array_equal(plain_out, expected):
        return None

    inp = _render_input(blocks, orient)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
