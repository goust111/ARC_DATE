# Combo variant of 78332cb0 (index 4)
# combo_spec: D4/1+D5/1
# Name: 无缝拼接边缘消融

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

# Design analysis
# a) Concrete behavior and trace:
#    - Input remains the source-task layout: 3-4 scrambled 5x5 single-color blocks separated by
#      full separator rows/cols. The solver still has to recover the only chain direction and order
#      from matching edge offsets.
#    - Source output placed blocks as [block][sep][block]. D4/1 changes that to a compact strip
#      where the next block starts 4 cells later, so neighboring 5x5 blocks share one seam column
#      (or seam row vertically).
#    - D5/1 then acts on that compact seam. Example for a horizontal pair with a shared contact at
#      row 2:
#          source strip:  A[2,4], sep, B[2,0]
#          D4-only strip: A[2,4] overlaps B[2,0] into one seam cell
#          combo strip:   that seam cell is erased to background
#      so the fusion is not "compact OR erase"; it is "compact first, then remove the internal
#      matched seam pixel(s)". The output width becomes 4N+1 instead of 6N-1, and each adjacency
#      contributes a seam hole.
# b) Complexity calibration:
#    - We keep N in {3,4}, keep the original unique-orientation and unique-order constraints, and
#      keep exactly one intended contact offset per adjacency from the small set {1,2,3}.
#    - To make the fused rule visible, generator instances use one foreground color across the whole
#      chain. That avoids color-overwrite ambiguity and makes the missing seam cells the main signal.
#    - Freedom is still limited: no rotation, no recoloring rule, no multiple seam types.
# c) Hidden-assumption audit against the source solver:
#    - `_find_sep`, separator scans, and block splitting still assume the input contains intact full
#      separator lines. The combo does not alter the input, so this remains valid.
#    - `_find_chain` still assumes contact pixels are present in the input. Therefore seam erasure
#      must happen only after ordering; generator keeps the input un-erased.
#    - The original renderer assumed an extra separator row/col and no overlap. That assumption is
#      broken twice here: separators disappear, and neighbors occupy a shared seam row/col.
#    - Compact rendering introduces a new safety assumption: after overlap, the only cells ever
#      written by two blocks are matched seam cells. Generator guarantees this by keeping the
#      original "one fg on each chain edge, none on perpendicular edges" construction.
# d) Placement legality:
#    - Input placement is unchanged from the source task, so blocks never overlap there.
#    - Output compact placement uses stride 4 for 5x5 blocks, yielding one-cell seam overlap only.
#      Because shared edge pixels are erased before pasting, the final compact output has no color
#      conflicts on overlapped cells and stays in bounds.
# e) Visual observability:
#    - Across samples, humans can see two coupled differences from the source task: magenta dividers
#      vanish, and each internal adjacency leaves a background notch exactly at the matched offset.
#    - The rule is inferable from 2-4 examples because every sample has 2 or 3 adjacencies, so the
#      compact geometry and seam erasure repeat multiple times.
# f) Visual cleanliness:
#    - Input remains easy to parse because separators are untouched there.
#    - Output remains a single thin strip, not a pile of overlaps: only one seam row/col is shared,
#      and erasing the matched seam removes any double-painted foreground.
# g) Change inferability:
#    - No hidden color mapping is introduced. Background 7 is the only erase target, triggered only
#      by positional edge matching, which is directly visible in the input sprites.
# h) Variant metrics used after testing:
#    - Compactness gain over the source layout is `2 * (N - 1)` cells on the long axis.
#    - Extra combo effect over D4-only is `N - 1` erased seam pixels, one per adjacency.
#      Both metrics must stay positive; otherwise the generator would have degenerated.
# i) New solver assumptions introduced by the combo:
#    - All extracted tiles are still 5x5 blocks.
#    - Any valid chain orientation has neighbors that share at least one edge offset in the input.
#    - Compact output always uses the same stride `BLOCK_SIZE - 1`; generator verifies this by
#      checking `main(input) == expected_combo_output` on every accepted sample.

BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

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


def _edge_sets(block: np.ndarray, bg: int) -> dict[str, set[int]]:
    """Return fg offsets on each border of a 5x5 block."""
    mask = block != int(bg)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _erase_shared_edges(blocks: list[np.ndarray], orient: str, bg: int) -> list[np.ndarray]:
    """Erase matched seam pixels on both sides before compact rendering."""
    erased = [block.copy() for block in blocks]
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]
    for i in range(len(erased) - 1):
        shared = contacts[i][lead] & contacts[i + 1][trail]
        if orient == "h":
            for r in shared:
                erased[i][int(r), -1] = int(bg)
                erased[i + 1][int(r), 0] = int(bg)
        else:
            for c in shared:
                erased[i][-1, int(c)] = int(bg)
                erased[i + 1][0, int(c)] = int(bg)
    return erased


def _render_compact_strip(
    blocks: list[np.ndarray],
    orient: str,
    bg: int,
    *,
    erase_shared: bool,
) -> np.ndarray:
    """Render a 1xN strip with no separator lines and one-cell seam overlap."""
    tiles = _erase_shared_edges(blocks, orient, bg) if erase_shared else [block.copy() for block in blocks]
    n = len(tiles)
    stride = BLOCK_SIZE - 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, BLOCK_SIZE + (n - 1) * stride), int(bg), dtype=int)
        for i, block in enumerate(tiles):
            paste(out, block, 0, i * stride, transparent=int(bg))
        return out
    out = np.full((BLOCK_SIZE + (n - 1) * stride, BLOCK_SIZE), int(bg), dtype=int)
    for i, block in enumerate(tiles):
        paste(out, block, i * stride, 0, transparent=int(bg))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a compact fused strip by matching edge foreground pixels.

    Input structure:
    - The grid contains 3-4 scrambled 5x5 sprite blocks on background, still separated in the
      input by full-width or full-height separator lines.
    - A valid chain exists in exactly one orientation, and neighboring blocks advertise that chain
      by sharing at least one foreground offset on their facing edges.

    Rule:
    - Recover the unique chain order exactly as in the source task.
    - Render the ordered blocks with no separator lines and with a one-cell seam overlap between
      neighbors. Before placing them, erase every matched seam pixel on both touching edges so that
      only externally exposed foreground survives in the compact strip.

    Key insight:
    - The same edge matches determine both the ordering and which compact seam cells disappear, so
      the solver must predict the final fused shape instead of merely re-spacing the tiles.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

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
            if r > start:
                segs.append(a[start:r, :].copy())
            start = r + 1
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if c > start:
                segs.append(a[:, start:c].copy())
            start = c + 1
        return segs

    def _split_blocks() -> list[np.ndarray]:
        rows = _full_sep_rows(g)
        cols = _full_sep_cols(g)
        if rows and cols:
            r = int(rows[0])
            c = int(cols[0])
            return [
                g[:r, :c].copy(),
                g[:r, c + 1 :].copy(),
                g[r + 1 :, :c].copy(),
                g[r + 1 :, c + 1 :].copy(),
            ]
        if rows:
            return _split_by_rows(g)
        if cols:
            return _split_by_cols(g)
        return []

    def _find_chain(blocks: list[np.ndarray], orient: str):
        contacts = [_edge_sets(block, bg) for block in blocks]
        lead, trail = _EDGE_PAIRS[orient]

        def ok(a: int, b: int) -> bool:
            return bool(contacts[a][lead] & contacts[b][trail])

        idx = list(range(len(blocks)))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return perm
        return None

    blocks = _split_blocks()
    assert blocks, "must detect at least one block"

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        return _render_compact_strip([blocks[i] for i in chain_h], "h", bg, erase_shared=True)
    if chain_v is not None and chain_h is None:
        return _render_compact_strip([blocks[i] for i in chain_v], "v", bg, erase_shared=True)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


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
    """Build a connected 5x5 mask with optional left/right edge contacts."""
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
    """Generate a single-color 5x5 block with the requested chain-edge contacts."""
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


def _render_input_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    """Render a separator-based strip for the scrambled input."""
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
        out = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        out[BLOCK_SIZE, :] = SEP
        out[:, BLOCK_SIZE] = SEP
        slots = [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]
        for block, (r0, c0) in zip(scrambled, slots):
            paste(out, block, r0, c0)
        return out

    return _render_input_strip(scrambled, _OTHER[output_orient])


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)
    instance_color = random.choice(PALETTE)

    lead_kw, trail_kw = _ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = shared_offsets[i - 1]
        if i < n - 1:
            kw[trail_kw] = shared_offsets[i]
        block = _make_block(instance_color, **kw)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, orient) != 1:
        return None
    if _count_orders(blocks, _OTHER[orient]) != 0:
        return None

    expected = _render_compact_strip(blocks, orient, BG, erase_shared=True)
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
