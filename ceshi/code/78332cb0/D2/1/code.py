# Variant of 78332cb0
# Dimension: D2 — 链方向唯一性约束 (match_cond)
# Change: 恰好一个方向（水平或垂直）存在有效链，另一个方向不存在... → 方向由特殊标记指示

from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

"""
Design analysis for the D2 variant
==================================

spec_variant:
  inherited_structure:
    background: 7
    separator: 6
    block_count: 3-4
    block_size: 5x5
    block_color_palette: {1,3,4,5,8,9}
    output: blocks reordered into a 1xN strip with sep dividers
  core_change:
    original: exactly one of horizontal/vertical chain directions exists
    variant: both directions are intentionally constructible and unique;
      a single red marker pixel chooses which direction to use
  marker_encoding:
    marker_color: 2
    canvas_padding: always add a 1-cell top margin and a 1-cell left margin
    vertical_choice: one red pixel on the top margin
    horizontal_choice: one red pixel on the left margin
    output: marker disappears; only the reordered strip remains
  concrete_effect:
    - The same scrambled 5x5 blocks can support a horizontal chain and a vertical chain.
    - The solver must first decode the marker, then choose the corresponding chain.
    - This replaces the old "direction inferred by exclusivity" step with an explicit
      symbol-decoding step.

behavior_trace:
  old_logic:
    - split the 5x5 blocks
    - search for a valid H permutation and a valid V permutation
    - whichever orientation exists uniquely becomes the output orientation
  new_logic:
    - read the red marker on the outer padded border
    - crop away the marker margins
    - split the same kind of 5x5 blocks
    - find the chain for the marker-selected orientation
    - render that ordered chain as the output strip
  difference:
    - ambiguity is no longer an error condition by itself
    - the direction branch is controlled by a visible symbol, not by absence of the
      alternative orientation

complexity_calibration:
  - A single pixel marker is visually simple but very noticeable because it is placed
    alone in the extra border, outside all sprites and separators.
  - Blocks are generated with both left/right and top/bottom contacts, so the variant
    is visibly different from the original perpendicular-empty sprites.
  - Only two marker positions are used (top vs left), so the new decoding rule is
    learnable from a few examples.

implicit_assumption_audit:
  - Original _contacts() inferred block background using the most common color inside
    each 5x5 block. That was harmless when sprites were sparse. In this variant the
    two-axis connector masks can be denser, so the solver now compares against the
    global background color instead.
  - Original main() assumed "exactly one direction exists" and asserted when both did.
    That assumption is removed; the chosen direction is read from the marker instead.
  - Original split logic assumed the whole grid was just the block layout. The variant
    therefore adds fixed 1-cell top/left margins and crops them away before splitting.

placement_legality:
  - The block layout inside the cropped core grid is unchanged: either a strip or a
    2x2 arrangement with full sep rows/cols.
  - The marker lives only in the outer padding row/col, so it never overlaps any block
    or separator and cannot create fake 5x5 segments.
  - Each generated block preserves exact edge-contact constraints, and generator
    verification rejects any sample whose selected chain is not reproduced by main().

visual_observability:
  - Human-solvers can see that the same puzzle family sometimes outputs a horizontal
    strip and sometimes a vertical strip even though both kinds of edge matches are
    present; the red pixel explains which branch to follow.
  - The marker is spatial, not numeric: top means choose the vertical chain, left means
    choose the horizontal chain.

tidiness_and_color:
  - Marker color 2 is removed from the sprite palette so the lone red pixel remains
    unambiguous.
  - The marker sits in empty margin space, keeping the interior arrangement clean.
"""


BG = 7
SEP = 6
MARKER = 2
PALETTE = [1, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
EDGE_CONTACT_OFFSETS = [1, 2, 3]
GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    h, w = a.shape
    for r in range(h):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
            cand.append(int(a[r, 0]))
    for c in range(w):
        if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
            cand.append(int(a[0, c]))
    return int(Counter(cand).most_common(1)[0][0])


def _full_sep_rows(a: np.ndarray, sep: int) -> list[int]:
    return [int(r) for r in range(int(a.shape[0])) if bool((a[int(r), :] == int(sep)).all())]


def _full_sep_cols(a: np.ndarray, sep: int) -> list[int]:
    return [int(c) for c in range(int(a.shape[1])) if bool((a[:, int(c)] == int(sep)).all())]


def _split_by_rows(a: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for r in rows + [int(a.shape[0])]:
        if int(r) > int(start):
            segs.append(a[int(start) : int(r), :].copy())
        start = int(r + 1)
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if int(c) > int(start):
            segs.append(a[:, int(start) : int(c)].copy())
        start = int(c + 1)
    return segs


def _split_blocks(g: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(g, sep)
    cols = _full_sep_cols(g, sep)
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
        return _split_by_rows(g, sep)
    if cols:
        return _split_by_cols(g, sep)
    return []


def _render_strip(blocks: list[np.ndarray], orient: str, bg: int, sep: int) -> np.ndarray:
    n = len(blocks)
    s = BLOCK_SIZE
    if orient == "v":
        out = np.full((n * s + (n - 1), s), bg, dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            out[rr : rr + s, :] = block
            rr += s
            if i < n - 1:
                out[rr, :] = sep
                rr += 1
        return out
    out = np.full((s, n * s + (n - 1)), bg, dtype=int)
    cc = 0
    for i, block in enumerate(blocks):
        out[:, cc : cc + s] = block
        cc += s
        if i < n - 1:
            out[:, cc] = sep
            cc += 1
    return out


def _edge_sets(block: np.ndarray, bg: int) -> dict[str, set[int]]:
    mask = block != int(bg)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _find_unique_chain(blocks: list[np.ndarray], orient: str, bg: int) -> tuple[int, ...] | None:
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]
    found: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(len(blocks))):
        if all(bool(contacts[perm[i]][lead] & contacts[perm[i + 1]][trail]) for i in range(len(perm) - 1)):
            if found is not None:
                return None
            found = tuple(int(x) for x in perm)
    return found


def _count_orders(blocks: list[np.ndarray], orient: str, bg: int) -> int:
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]
    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(bool(contacts[perm[i]][lead] & contacts[perm[i + 1]][trail]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _decode_marker(g: np.ndarray) -> str:
    top = [int(c) for c in np.where(g[0, :] == MARKER)[0].tolist()]
    left = [int(r) for r in np.where(g[:, 0] == MARKER)[0].tolist()]
    assert bool(top) ^ bool(left), "expected marker on exactly one padded side"
    return "v" if top else "h"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a connected strip by matching edge foreground pixels.

    Variant rule:
    - Both horizontal and vertical chain directions may be valid.
    - A single red marker pixel on the padded border chooses which direction to use.
      marker on top -> vertical chain, marker on left -> horizontal chain.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    orient = _decode_marker(g)
    core = g[1:, 1:]
    sep = _find_sep(core, bg)
    blocks = _split_blocks(core, sep)
    assert blocks, "no blocks found"

    chain = _find_unique_chain(blocks, orient, bg)
    assert chain is not None, f"no unique {orient} chain found"
    ordered = [blocks[i] for i in chain]
    return _render_strip(ordered, orient, bg, sep)


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
    rs, cs = np.where(has_neighbor & ~interior)
    return [(int(r) + 1, int(c) + 1) for r, c in zip(rs, cs)]


def _base_mask(
    *,
    left: int | None,
    right: int | None,
    top: int | None,
    bottom: int | None,
) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left is not None:
        mask[left, 0:3] = True
        mask[min(left, 2) : max(left, 2) + 1, 2] = True
    if right is not None:
        mask[right, 2:BLOCK_SIZE] = True
        mask[min(right, 2) : max(right, 2) + 1, 2] = True
    if top is not None:
        mask[0:3, top] = True
        mask[2, min(top, 2) : max(top, 2) + 1] = True
    if bottom is not None:
        mask[2:BLOCK_SIZE, bottom] = True
        mask[2, min(bottom, 2) : max(bottom, 2) + 1] = True

    if not mask.any():
        mask[2, 2] = True
    return mask


def _grow_mask(mask: np.ndarray) -> np.ndarray | None:
    out = mask.copy()
    target = random.randint(int(out.sum()), min(14, int(out.sum()) + 2))
    while int(out.sum()) < int(target):
        frontier = _interior_frontier(out)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        out[r, c] = True
    if not is_contiguous(out.astype(int), background=0, connectivity=4):
        return None
    return out


def _make_block(
    color: int,
    *,
    left: int | None,
    right: int | None,
    top: int | None,
    bottom: int | None,
) -> np.ndarray | None:
    mask = _base_mask(left=left, right=right, top=top, bottom=bottom)
    mask = _grow_mask(mask)
    if mask is None:
        return None

    if left is None:
        if mask[:, 0].any():
            return None
    elif np.where(mask[:, 0])[0].tolist() != [left]:
        return None

    if right is None:
        if mask[:, -1].any():
            return None
    elif np.where(mask[:, -1])[0].tolist() != [right]:
        return None

    if top is None:
        if mask[0, :].any():
            return None
    elif np.where(mask[0, :])[0].tolist() != [top]:
        return None

    if bottom is None:
        if mask[-1, :].any():
            return None
    elif np.where(mask[-1, :])[0].tolist() != [bottom]:
        return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _render_input_core(blocks: list[np.ndarray]) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        s = BLOCK_SIZE
        grid = np.full((2 * s + 1, 2 * s + 1), BG, dtype=int)
        grid[s, :] = SEP
        grid[:, s] = SEP
        for block, (r0, c0) in zip(scrambled, [(0, 0), (0, s + 1), (s + 1, 0), (s + 1, s + 1)]):
            paste(grid, block, r0, c0)
        return grid

    orient = random.choice(["h", "v"])
    return _render_strip(scrambled, orient, BG, SEP)


def _add_marker(core: np.ndarray, orient: str) -> np.ndarray:
    h, w = core.shape
    out = np.full((h + 1, w + 1), BG, dtype=int)
    out[1:, 1:] = core
    if orient == "v":
        out[0, 1 + w // 2] = MARKER
    else:
        out[1 + h // 2, 0] = MARKER
    return out


def _assign_side_maps(
    n: int, perm_h: list[int], perm_v: list[int], h_offsets: list[int], v_offsets: list[int]
) -> tuple[dict[int, int | None], dict[int, int | None], dict[int, int | None], dict[int, int | None]]:
    lefts = {i: None for i in range(n)}
    rights = {i: None for i in range(n)}
    tops = {i: None for i in range(n)}
    bottoms = {i: None for i in range(n)}

    for pos, idx in enumerate(perm_h):
        if pos > 0:
            lefts[idx] = int(h_offsets[pos - 1])
        if pos < n - 1:
            rights[idx] = int(h_offsets[pos])

    for pos, idx in enumerate(perm_v):
        if pos > 0:
            tops[idx] = int(v_offsets[pos - 1])
        if pos < n - 1:
            bottoms[idx] = int(v_offsets[pos])

    return lefts, rights, tops, bottoms


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    n = random.choice([3, 4])
    perm_h = random.sample(range(n), n)
    perm_v = random.sample(range(n), n)
    h_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)
    v_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    lefts, rights, tops, bottoms = _assign_side_maps(n, perm_h, perm_v, h_offsets, v_offsets)

    blocks: list[np.ndarray] = []
    colors = random.choices(PALETTE, k=n)
    for i in range(n):
        block = _make_block(
            colors[i],
            left=lefts[i],
            right=rights[i],
            top=tops[i],
            bottom=bottoms[i],
        )
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, "h", BG) != 1:
        return None
    if _count_orders(blocks, "v", BG) != 1:
        return None

    marker_orient = random.choice(["h", "v"])
    chosen_perm = perm_h if marker_orient == "h" else perm_v
    out = _render_strip([blocks[i] for i in chosen_perm], marker_orient, BG, SEP)

    core = _render_input_core(blocks)
    inp = _add_marker(core, marker_orient)

    if not np.array_equal(main(inp), out):
        return None

    meta = {
        "n": n,
        "marker_orient": marker_orient,
        "perm_h": tuple(perm_h),
        "perm_v": tuple(perm_v),
        "same_order": tuple(perm_h) == tuple(perm_v),
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
