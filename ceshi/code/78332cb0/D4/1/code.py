# Variant of 78332cb0
# Dimension: D4 — 目标排列方式 (dst)
# Change: 1xN条带（水平或垂直），块按链序排列，用品红分隔线分开... → 紧密拼接无分隔线

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

# Design analysis
# a) Concrete behavior:
#    - Input stays the same as the source task: 3-4 scrambled 5x5 blocks are separated by full
#      magenta rows/cols, either in a strip or a 2x2 grid.
#    - The chain rule also stays the same: consecutive blocks connect when the predecessor's
#      trailing edge and the successor's leading edge have a foreground pixel at the same offset.
#    - The D4 change happens only in output rendering. Instead of:
#        [block][sep][block][sep][block]
#      we render:
#        [block]
#            [block]
#                [block]
#      where each next block starts 4 cells after the previous one, so neighboring 5x5 blocks
#      overlap by exactly one column (horizontal chain) or one row (vertical chain). The matched
#      edge pixels are therefore merged into a single seam cell and the long strip becomes
#      5 x (4N+1) or (4N+1) x 5.
#    - Small horizontal example:
#        original: width = 5 + 1 + 5 = 11, seam is a full magenta column
#        variant : width = 5 + 4 = 9, col 4 is both A.right and B.left
# b) Complexity calibration:
#    - We keep N in {3,4}, keep the unique-chain requirement, and keep one-pixel edge contacts.
#    - To make the seam visually read as "merged" instead of "one block overwrote the other", the
#      generator uses one shared foreground color per instance. Shapes still differ, and the chain
#      order is still determined only by edge-contact offsets.
# c) Hidden-assumption audit against the source main():
#    - `_find_sep`, `_full_sep_rows/_cols`, `_split_blocks`, `_contacts`, `_find_chain` depend only
#      on the input layout. Since input separators remain unchanged, those assumptions still hold.
#    - The original `_render_strip` assumes an extra separator row/col between blocks and writes the
#      separator unconditionally. That assumption is invalid in this variant and would destroy the
#      new compact output, so rendering must change to overlap-aware placement.
#    - The source code also hardcodes the output long-axis size as `n*5 + (n-1)`. In the variant it
#      must become `5 + (n-1)*4`.
# d) Placement legality:
#    - Input placement is unchanged, so no blocks overlap there.
#    - Output overlap is deliberate and bounded: only one shared row/col per seam. Generator-side
#      edge constraints guarantee exactly one fg pixel on each connecting edge and zero fg pixels on
#      the perpendicular edges, so overlap occurs only at the intended seam.
# e) Visual observability:
#    - Across examples, the strip is visibly shorter than the source task by exactly two cells per
#      connection (one removed separator cell plus one merged edge cell), and the seam no longer
#      shows magenta. Human solvers can infer that blocks are now fused, not merely reordered.
# f) Visual cleanliness:
#    - Because input still uses magenta dividers, block extraction remains clear.
#    - Because output uses one shared fg color per instance, the fused seam looks continuous rather
#      than noisy or ambiguous.
# g) Color inferability:
#    - There is no color remapping rule. Colors are structural only; the generator restricts each
#      instance to one fg color purely to make seam merging unambiguous and human-readable.
# h) Variant metric used after testing:
#    - Compactness reduction = old strip long-axis size - new strip long-axis size = 2*(N-1).
#      Positive reduction on every sample confirms that D4 is actually active.
# i) New solver assumptions introduced by the variant:
#    - All blocks remain 5x5 and are separated only in the input.
#    - Compact output overlap is always exactly one row/col; generator enforces this by keeping the
#      original one-pixel edge-contact construction.


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


def _render_compact_strip(blocks: list[np.ndarray], orient: str, bg: int) -> np.ndarray:
    """Render chain-ordered blocks with 1-cell seam overlap and no separator lines."""
    n = len(blocks)
    if n == 0:
        raise ValueError("need at least one block")
    h, w = map(int, blocks[0].shape)
    if orient == "h":
        out = np.full((h, w + (n - 1) * (w - 1)), int(bg), dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * (w - 1), transparent=int(bg))
        return out
    out = np.full((h + (n - 1) * (h - 1), w), int(bg), dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * (h - 1), 0, transparent=int(bg))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a connected compact strip by matching edge foreground pixels.

    The input layout is unchanged from the source task, but the output now removes separator
    lines and overlaps neighboring blocks by one column/row so the matched seam is merged.
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
        return _render_compact_strip([blocks[i] for i in chain_h], "h", bg)
    if chain_v is not None and chain_h is None:
        return _render_compact_strip([blocks[i] for i in chain_v], "v", bg)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
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
    """Build a connected 5x5 mask with optional left/right edge contacts."""
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2) : max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:BLOCK_SIZE] = True
        mask[min(right_offset, 2) : max(right_offset, 2) + 1, 2] = True

    if mask.sum() < FG_AREA_MIN:
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
    """Generate a single-color 5x5 block with the required edge contacts."""
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
    """Render the scrambled input strip with magenta separators, unchanged from source."""
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
        for block, (r0, c0) in zip(scrambled, [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]):
            paste(grid, block, r0, c0)
        return grid

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

    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, _OTHER[orient]) != 0:
        return None

    out = _render_compact_strip(blocks, orient, BG)
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
