from __future__ import annotations

# Variant of 78332cb0
# Dimension: D5 — 变换操作 (transform)
# Change: 仅平移（将块从打散位置移到链序位置），不旋转不变色... → 匹配后重新着色

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
    input_layout:
      forms: 2x2 grid for some 4-block cases, otherwise strip
      separator_lines: full rows/cols of color 6
      order: scrambled, never already in chain order
    chain_rule:
      criterion: consecutive blocks share at least one foreground offset on the
        facing edges
      orientation: exactly one of horizontal or vertical is valid

  D5_variant:
    core_change:
      original: after recovering the chain, output the ordered blocks unchanged
      new: after recovering the chain, recolor block k by its chain position
    position_palette:
      first: 1
      second: 2
      third: 3
      fourth: 4
    invariants:
      no rotation
      no flip
      no shape editing
      only relocation + recolor

  behavior_trace:
    original_flow:
      - split the scrambled grid into 5x5 blocks
      - compare edge-contact offsets to find the unique chain orientation/order
      - render the ordered strip using original block colors
    variant_flow:
      - split the scrambled grid into 5x5 blocks
      - compare edge-contact offsets to find the unique chain orientation/order
      - recolor every foreground pixel in the 1st/2nd/3rd/4th chain block to
        1/2/3/4
      - render the ordered strip with those recolored blocks
    concrete_difference:
      if three input blocks are colored 8, 9, 5 and the recovered chain order is
      [block_B, block_C, block_A], then the original output colors would stay
      [9, 5, 8], while the variant output becomes [1, 2, 3].

  complexity_calibration:
    - Edge matching stays identical to the source task, so the main reasoning
      load is unchanged.
    - The new color rule is low-arity and global: only 3-4 possible output
      colors, determined solely by chain rank.
    - Generator uses input colors from {5,8,9}, disjoint from output colors
      {1,2,3,4}, so recolor is always visible and never accidentally collapses
      back to the old task.

  implicit_assumption_audit:
    - `_find_sep` assumes separator lines are full non-background rows/cols.
      Generator preserves this and never uses color 6 inside sprites.
    - `_split_blocks` assumes the layout is either a strip or a 2x2 grid formed
      by those separator lines. Generator preserves exactly those layouts.
    - `_contacts` assumes foreground is detected as `block != bg`; this requires
      a single global background and no stray detached cells outside block slots.
      Generator preserves both.
    - `_find_chain` assumes a unique valid orientation/permutation exists.
      Generator explicitly rejects ambiguous chains and wrong-orientation chains.
    - The new recolor step assumes block count <= len(POS_COLORS). Generator
      keeps N in {3,4}, so the fixed palette is sufficient.

  placement_legality:
    - Blocks are still rendered only into disjoint 5x5 slots.
    - Strip layouts have one full separator row/col between neighboring blocks.
    - 2x2 layouts place the four 5x5 blocks in non-overlapping quadrants around
      one separator row and one separator col.
    - Because the generator is output-first and validates `main(input) == output`,
      any placement bug is rejected.

  visual_observability:
    - The human-visible difference from the source task is strong: output colors
      now form a positional color ramp even when the input block colors differ.
    - Across samples with horizontal/vertical chains and 3/4 blocks, the only
      stable explanation is "recover the chain, then recolor by position."

  visual_neatness:
    - No extra overlays or marker pixels are added.
    - Recoloring happens on whole block foregrounds, so the output remains as
      clean as the original strip layout.

  color_change_inference:
    - The color mapping is fixed and small, not data-dependent.
    - Position is recoverable from structure alone, so examples provide the
      necessary cue for humans to infer the color assignment.

  generator_validation:
    - Output-first generation builds a valid ordered chain in original colors.
    - Expected output is the same chain after positional recolor.
    - Input is a scrambled rendering of the original-colored blocks.
    - Generation is rejected unless the new solver reproduces the recolored
      target and the unrecolored strip is different from that target.
"""

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


BG = 7
SEP = 6
INPUT_PALETTE = [5, 8, 9]
POS_COLORS = [1, 2, 3, 4]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 300

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a connected strip by matching edge foreground pixels,
    then recolor each block by its recovered chain position.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _find_sep(a: np.ndarray) -> int:
        cand: list[int] = []
        for r in range(int(a.shape[0])):
            if bool((a[int(r), :] == int(a[int(r), 0])).all()) and int(a[int(r), 0]) != int(bg):
                cand.append(int(a[int(r), 0]))
        for c in range(int(a.shape[1])):
            if bool((a[:, int(c)] == int(a[0, int(c)])).all()) and int(a[0, int(c)]) != int(bg):
                cand.append(int(a[0, int(c)]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(int(a.shape[0])) if bool((a[int(r), :] == int(sep)).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(int(a.shape[1])) if bool((a[:, int(c)] == int(sep)).all())]

    def _split_by_rows(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        segs: list[np.ndarray] = []
        start = 0
        for r in rows + [int(a.shape[0])]:
            if int(r) > int(start):
                segs.append(a[int(start):int(r), :].copy())
            start = int(r + 1)
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if int(c) > int(start):
                segs.append(a[:, int(start):int(c)].copy())
            start = int(c + 1)
        return segs

    def _split_blocks() -> list[np.ndarray]:
        rows = _full_sep_rows(g)
        cols = _full_sep_cols(g)
        if rows and cols:
            r = int(rows[0])
            c = int(cols[0])
            return [
                g[:r, :c].copy(),
                g[:r, int(c + 1):].copy(),
                g[int(r + 1):, :c].copy(),
                g[int(r + 1):, int(c + 1):].copy(),
            ]
        if rows:
            return _split_by_rows(g)
        if cols:
            return _split_by_cols(g)
        return []

    def _contacts(block: np.ndarray) -> dict[str, set[int]]:
        mask = block != int(bg)
        return {
            "top": set(np.where(mask[0, :])[0].tolist()),
            "bot": set(np.where(mask[-1, :])[0].tolist()),
            "left": set(np.where(mask[:, 0])[0].tolist()),
            "right": set(np.where(mask[:, -1])[0].tolist()),
        }

    def _find_chain(blocks: list[np.ndarray], orient: str):
        cont = [_contacts(b) for b in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return bool(cont[int(a)]["right"] & cont[int(b)]["left"])
            return bool(cont[int(a)]["bot"] & cont[int(b)]["top"])

        idx = list(range(int(len(blocks))))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    def _recolor_block(block: np.ndarray, color: int) -> np.ndarray:
        out = block.copy()
        out[out != int(bg)] = int(color)
        return out

    def _render_strip(blocks: list[np.ndarray], axis: int) -> np.ndarray:
        n = int(len(blocks))
        if int(axis) == 0:
            out = np.full((int(n * 5 + (n - 1)), 5), int(bg), dtype=int)
            rr = 0
            for i, b in enumerate(blocks):
                out[int(rr):int(rr + 5), :] = b
                rr += 5
                if int(i) < int(n - 1):
                    out[int(rr), :] = int(sep)
                    rr += 1
            return out
        out = np.full((5, int(n * 5 + (n - 1))), int(bg), dtype=int)
        cc = 0
        for i, b in enumerate(blocks):
            out[:, int(cc):int(cc + 5)] = b
            cc += 5
            if int(i) < int(n - 1):
                out[:, int(cc)] = int(sep)
                cc += 1
        return out

    blocks = _split_blocks()
    assert blocks, "must extract at least one block"
    assert len(blocks) <= len(POS_COLORS), "position palette too short for block count"

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        ordered = [_recolor_block(blocks[i], POS_COLORS[k]) for k, i in enumerate(chain_h)]
        return _render_strip(ordered, axis=1)
    if chain_v is not None and chain_h is None:
        ordered = [_recolor_block(blocks[i], POS_COLORS[k]) for k, i in enumerate(chain_v)]
        return _render_strip(ordered, axis=0)
    assert chain_h is None or chain_v is None, "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    mask = block != BG
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_edge_sets(block) for block in blocks]
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


def _recolor_block(block: np.ndarray, color: int) -> np.ndarray:
    out = block.copy()
    out[out != BG] = int(color)
    return out


def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    n = len(blocks)
    stride = BLOCK_SIZE + 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), BG, dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + BLOCK_SIZE] = SEP
        return out
    out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), BG, dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * stride, 0)
        if i < n - 1:
            out[i * stride + BLOCK_SIZE, :] = SEP
    return out


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

    return _render_strip(scrambled, _OTHER[output_orient])


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    lead_kw, trail_kw = _ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(n):
        kw = {}
        if i > 0:
            kw[lead_kw] = shared_offsets[i - 1]
        if i < n - 1:
            kw[trail_kw] = shared_offsets[i]
        block = _make_block(random.choice(INPUT_PALETTE), **kw)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, _OTHER[orient]) != 0:
        return None

    recolored = [_recolor_block(block, POS_COLORS[i]) for i, block in enumerate(blocks)]
    out = _render_strip(recolored, orient)
    plain_out = _render_strip(blocks, orient)
    inp = _render_input(blocks, orient)

    if np.array_equal(plain_out, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
