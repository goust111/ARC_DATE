# Variant of 78332cb0
# Dimension: D5 — 变换操作 (transform)
# Change: 仅平移（将块从打散位置移到链序位置），不旋转不变色 → 颜色互换

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior:
#    - Input still contains 3-4 monochrome 5x5 sprite blocks separated by full
#      sep lines. The block masks and edge-contact logic are unchanged.
#    - Solver still reconstructs the unique horizontal/vertical chain from edge
#      overlaps only. The new action happens after the chain order is known:
#      blocks at chain positions (0,1), (2,3), ... swap their foreground colors
#      while keeping each block's own 5x5 mask.
#    - Trace example with a horizontal 4-block chain:
#        input chain colors by mask identity = [red, green, blue, yellow]
#        original output would be             [red, green, blue, yellow]
#        new output becomes                   [green, red, yellow, blue]
#      The shapes stay in chain order; only the colors move locally between
#      adjacent partners.
# b) Complexity calibration:
#    - A full arbitrary color permutation would be too unconstrained. A local
#      pair-swap keeps the transform simple and visible.
#    - To avoid invisible swaps, generator uses four blocks with distinct colors,
#      so every output block in fact changes color.
# c) Implicit-assumption audit of the original main():
#    - `_find_sep` assumes sep is the only non-bg color forming full rows/cols.
#      Generator still forbids full fg rows/cols inside blocks, so this holds.
#    - `_contacts` assumes block background is the dominant color inside a 5x5
#      tile. This still holds because each tile has only 4-10 fg cells.
#    - `_find_chain` only depends on masks touching the appropriate edges; color
#      changes do not affect connectivity, so it remains valid.
#    - The original final `_render_strip(ordered, axis=...)` relied on the old
#      invariant "output block colors equal input block colors". This is exactly
#      what breaks under D5, so the final render step must recolor the ordered
#      blocks before rendering.
# d) Placement legality:
#    - Layouts stay identical to the source task: 2x2 grid or strip input, strip
#      output, with single-cell sep rows/cols and no overlaps.
# e) Visual observability:
#    - Because each block keeps its unique silhouette but takes its neighbor's
#      color, humans can compare input/output and see that reordering alone is
#      insufficient.
# f) Visual cleanliness:
#    - No extra markers are added. Each 5x5 area still contains one connected
#      monochrome sprite on a clean background, so the grids stay readable.
# g) Color-rule inferability:
#    - The rule uses only adjacent positions after chain reconstruction. It is
#      not keyed off hidden numeric color ids.
#    - Generator forces four distinct chain colors, which makes both adjacent
#      swaps visible in each sample.
# h) Generator non-degeneration checks:
#    - It verifies unique chain orientation/permutation exactly as before.
#    - It also rejects samples where swapped output would equal the old output.
# i) New solver assumptions introduced here:
#    - Each block still has exactly one fg color. Generator guarantees this.
#    - Odd-length chains are handled by leaving the last block unchanged, though
#      the generator sticks to four blocks so every block participates in a swap.


def _fg_color(block: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(block) if int(x) != int(bg)})
    assert len(colors) == 1, "each block must be single-color"
    return int(colors[0])


def _recolor_block(block: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = np.full(block.shape, int(bg), dtype=int)
    out[block != int(bg)] = int(color)
    return out


def _swap_adjacent_colors(blocks: list[np.ndarray], bg: int) -> list[np.ndarray]:
    colors = [_fg_color(block, bg) for block in blocks]
    swapped: list[np.ndarray] = []
    i = 0
    while i + 1 < len(blocks):
        swapped.append(_recolor_block(blocks[i], bg, colors[i + 1]))
        swapped.append(_recolor_block(blocks[i + 1], bg, colors[i]))
        i += 2
    if i < len(blocks):
        swapped.append(blocks[i].copy())
    return swapped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a connected strip, then swap colors within adjacent pairs.

    Input structure:
    - bg = background; sep = divider color (full row/column lines).
    - Several 5x5 sprite blocks in a grid, separated by sep lines.

    Rule:
    - Two blocks connect when their facing edges share foreground pixels at the
      same offset.
    - Chain all blocks in the orientation (horizontal or vertical) that admits a
      complete sequence.
    - After ordering, swap the foreground colors of adjacent chain pairs:
      (0,1), (2,3), ...
    - Render the recolored chain as a 1xN strip with sep dividers.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _most_common(a: np.ndarray) -> int:
        return int(Counter(map(int, a.flatten())).most_common(1)[0][0])

    def _find_sep(a: np.ndarray) -> int:
        cand: list[int] = []
        for r in range(int(a.shape[0])):
            if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
                cand.append(int(a[r, 0]))
        for c in range(int(a.shape[1])):
            if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
                cand.append(int(a[0, c]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(int(a.shape[0])) if bool((a[r, :] == int(sep)).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(int(a.shape[1])) if bool((a[:, c] == int(sep)).all())]

    def _split_by_rows(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        segs: list[np.ndarray] = []
        start = 0
        for r in rows + [int(a.shape[0])]:
            if int(r) > int(start):
                segs.append(a[int(start) : int(r), :].copy())
            start = int(r + 1)
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if int(c) > int(start):
                segs.append(a[:, int(start) : int(c)].copy())
            start = int(c + 1)
        return segs

    def _render_strip(blocks: list[np.ndarray], axis: int) -> np.ndarray:
        n = int(len(blocks))
        if int(axis) == 0:
            out = np.full((int(n * 5 + (n - 1)), 5), int(bg), dtype=int)
            rr = 0
            for i, block in enumerate(blocks):
                out[int(rr) : int(rr + 5), :] = block
                rr += 5
                if int(i) < int(n - 1):
                    out[int(rr), :] = int(sep)
                    rr += 1
            return out
        out = np.full((5, int(n * 5 + (n - 1))), int(bg), dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, int(cc) : int(cc + 5)] = block
            cc += 5
            if int(i) < int(n - 1):
                out[:, int(cc)] = int(sep)
                cc += 1
        return out

    def _split_blocks() -> list[np.ndarray]:
        rows = _full_sep_rows(g)
        cols = _full_sep_cols(g)
        if rows and cols:
            r = int(rows[0])
            c = int(cols[0])
            return [
                g[:r, :c].copy(),
                g[:r, int(c + 1) :].copy(),
                g[int(r + 1) :, :c].copy(),
                g[int(r + 1) :, int(c + 1) :].copy(),
            ]
        if rows:
            return _split_by_rows(g)
        if cols:
            return _split_by_cols(g)
        return []

    def _contacts(block: np.ndarray) -> dict[str, set[int]]:
        bbg = int(_most_common(block))
        m = block != int(bbg)
        return {
            "top": set(np.where(m[0, :])[0].tolist()),
            "bot": set(np.where(m[-1, :])[0].tolist()),
            "left": set(np.where(m[:, 0])[0].tolist()),
            "right": set(np.where(m[:, -1])[0].tolist()),
        }

    def _find_chain(blocks: list[np.ndarray], orient: str):
        contacts = [_contacts(block) for block in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return bool(contacts[a]["right"] & contacts[b]["left"])
            return bool(contacts[a]["bot"] & contacts[b]["top"])

        idx = list(range(int(len(blocks))))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    blocks = _split_blocks()
    assert blocks, "must find blocks"

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        ordered = [blocks[i] for i in chain_h]
        return _render_strip(_swap_adjacent_colors(ordered, bg), axis=1)
    if chain_v is not None and chain_h is None:
        ordered = [blocks[i] for i in chain_v]
        return _render_strip(_swap_adjacent_colors(ordered, bg), axis=0)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


# --------------- generator ---------------

BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
BLOCK_COUNTS = [4]
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 300

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    mask = block != int(BG)
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
        mask[min(left_offset, 2) : max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:BLOCK_SIZE] = True
        mask[min(right_offset, 2) : max(right_offset, 2) + 1, 2] = True

    if int(mask.sum()) < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    while int(mask.sum()) < int(target_area):
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

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), int(BG), dtype=int)
    block[mask] = int(color)
    return block


def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    n = len(blocks)
    stride = BLOCK_SIZE + 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), int(BG), dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + BLOCK_SIZE] = int(SEP)
    else:
        out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), int(BG), dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, i * stride, 0)
            if i < n - 1:
                out[i * stride + BLOCK_SIZE, :] = int(SEP)
    return out


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), int(BG), dtype=int)
        grid[BLOCK_SIZE, :] = int(SEP)
        grid[:, BLOCK_SIZE] = int(SEP)
        for block, (r0, c0) in zip(
            scrambled,
            [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)],
        ):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient])


def _count_recolored_positions(before: list[np.ndarray], after: list[np.ndarray]) -> int:
    total = 0
    for block_before, block_after in zip(before, after):
        if _fg_color(block_before, BG) != _fg_color(block_after, BG):
            total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(BLOCK_COUNTS)
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)
    colors = random.sample(PALETTE, n)

    lead_kw, trail_kw = _ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i, color in enumerate(colors):
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

    swapped_blocks = _swap_adjacent_colors(blocks, BG)
    if _count_recolored_positions(blocks, swapped_blocks) != n:
        return None

    out = _render_strip(swapped_blocks, orient)
    old_out = _render_strip(blocks, orient)
    if np.array_equal(out, old_out):
        return None

    inp = _render_input(blocks, orient)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
