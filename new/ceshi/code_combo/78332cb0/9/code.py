# Combo variant of 78332cb0 (index 9)
# combo_spec: D3/1+D5/0+D14/0
# Name: 多色独色精灵链序重染

from __future__ import annotations

# Design analysis
# a) Concrete behavior trace:
#    - Original task: every 5x5 block is monochrome, so the solver can treat
#      all non-background pixels as one undifferentiated foreground mask, chain
#      the blocks by shared edge offsets, and output the same colors unchanged.
#    - This combo changes three stages at once. Each block now has exactly two
#      foreground colors, and all 2N colors in a sample are globally distinct.
#      The larger color inside a block is the main color; the smaller 4-pixel
#      accent is a decoy. Only the main color participates in edge matching.
#    - Example for a 3-block horizontal chain:
#      block A main touches right at row 1, block B main touches left/right at
#      rows 1/3, block C main touches left at row 3. Each block also has a
#      vertical 4-pixel accent stripe touching top and bottom. A legacy solver
#      that merges both colors would see horizontal links from the main color
#      and vertical links from the accent stripe, so role identification now
#      matters before chain recovery.
#    - After the chain order is recovered, recoloring is no longer "whole block
#      -> one color". The kth block's main color becomes palette[k], while its
#      accent becomes palette[k+N]. The output therefore depends on both role
#      separation and chain rank.
# b) Complexity calibration:
#    - The role cue is low-arity and repeated: every block has one larger color
#      and one smaller fixed-size accent. This keeps the primary/secondary split
#      visible from a few examples instead of requiring arbitrary color lookup.
#    - Global color uniqueness per sample satisfies the D14 constraint and makes
#      every input block easy to track visually, but the output palette stays a
#      fixed ordered list so the recolor rule remains small and learnable.
#    - The accent still creates a perpendicular decoy orientation when one
#      ignores color roles, so D3 and D5 interact rather than behaving as two
#      separate cosmetic edits.
# c) Implicit assumption audit against the original main():
#    - Original `_contacts` assumed all non-background pixels were equivalent.
#      That breaks here; the new solver must isolate the block-local main color.
#    - A "shared color across all blocks" assumption from the D3 single-variant
#      is also invalid here because D14 forces global color uniqueness.
#    - The new solver instead assumes each block has exactly two fg colors with
#      unequal counts, and that the more frequent color is the match-carrying
#      main color. The generator enforces a final 5-vs-4 or 6-vs-4 split.
#    - `_find_sep` still assumes separator lines are the only full constant
#      non-background rows/cols. The generator avoids constant full rows/cols
#      inside blocks; the only full row/col created by the accent is mixed-color.
#    - `_find_chain` still assumes a unique valid orientation/permutation after
#      the correct role extraction. The generator rejects ambiguous samples.
# d) Placement legality:
#    - Blocks remain disjoint 5x5 objects separated by full magenta lines.
#    - Input layouts are still a strip or a 2x2 arrangement for some 4-block
#      cases, so no overlap or boundary violation is introduced.
#    - Recoloring edits only existing foreground pixels inside the ordered strip.
# e) Visual observability:
#    - A human can see that each block has a larger body color and a smaller
#      accent color, that the larger one carries the left/right or top/bottom
#      continuation pattern, and that the output recolors those two roles into
#      two ordered ramps tied to chain position.
#    - Because the accent touches the perpendicular edges, examples visibly
#      differ from the source task and also demonstrate why all-foreground
#      matching would be wrong.
# f) Visual cleanliness:
#    - Each block still has one connected foreground component and at most two
#      colors, so the scenes stay sparse and readable.
#    - The accent shape is regular and centered, which keeps the decoy obvious
#      without turning the 5x5 sprite into noise.
# g) Change inferability:
#    - The output palette is fixed and ordered. For N blocks, main colors use
#      the first N colors of that palette and accents use the next N colors.
#      This matches the "k / k+N" description while keeping the mapping small.
#    - The rule is inferable from 2-4 examples because position is recovered
#      structurally and the two recolor ramps are repeated across every sample.
#
# New solver assumptions introduced by this combo:
# - Every extracted block is 5x5 and has exactly two non-background colors.
# - The more frequent foreground color is the main role color.
# - The less frequent foreground color is the accent role color.
# - Exactly one orientation/permutation becomes valid when matching only main
#   colors, and the block count is 3 or 4 so 2N output colors fit the palette.

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


BG = 7
SEP = 6
BLOCK_SIZE = 5

ALL_FG_COLORS = [0, 1, 2, 3, 4, 5, 8, 9]
OUTPUT_COLOR_ORDER = [1, 2, 3, 4, 5, 8, 9, 0]

EDGE_CONTACT_OFFSETS = [1, 2, 3]
PRIMARY_TARGET_BASE = 5
GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 600

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    for r in range(int(a.shape[0])):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
            cand.append(int(a[r, 0]))
    for c in range(int(a.shape[1])):
        if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
            cand.append(int(a[0, c]))
    return int(Counter(cand).most_common(1)[0][0])


def _full_sep_rows(a: np.ndarray, sep: int) -> list[int]:
    return [int(r) for r in range(int(a.shape[0])) if bool((a[r, :] == int(sep)).all())]


def _full_sep_cols(a: np.ndarray, sep: int) -> list[int]:
    return [int(c) for c in range(int(a.shape[1])) if bool((a[:, c] == int(sep)).all())]


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


def _render_strip(blocks: list[np.ndarray], axis: int, bg: int, sep: int) -> np.ndarray:
    n = int(len(blocks))
    if int(axis) == 0:
        out = np.full((int(n * BLOCK_SIZE + (n - 1)), BLOCK_SIZE), int(bg), dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            out[int(rr) : int(rr + BLOCK_SIZE), :] = block
            rr += BLOCK_SIZE
            if int(i) < int(n - 1):
                out[int(rr), :] = int(sep)
                rr += 1
        return out

    out = np.full((BLOCK_SIZE, int(n * BLOCK_SIZE + (n - 1))), int(bg), dtype=int)
    cc = 0
    for i, block in enumerate(blocks):
        out[:, int(cc) : int(cc + BLOCK_SIZE)] = block
        cc += BLOCK_SIZE
        if int(i) < int(n - 1):
            out[:, int(cc)] = int(sep)
            cc += 1
    return out


def _non_bg_colors(block: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(block) if int(v) != int(bg))


def _main_and_accent(block: np.ndarray, bg: int) -> tuple[int, int]:
    colors = _non_bg_colors(block, bg)
    assert len(colors) == 2, "each block must have exactly two foreground colors"
    counts = sorted(((int(np.count_nonzero(block == color)), color) for color in colors), reverse=True)
    assert counts[0][0] > counts[1][0], "main color must be strictly more frequent than accent"
    return int(counts[0][1]), int(counts[1][1])


def _contacts_for_color(block: np.ndarray, color: int | None, bg: int = BG) -> dict[str, set[int]]:
    mask = block != int(bg) if color is None else block == int(color)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _find_chain(blocks: list[np.ndarray], main_colors: list[int], orient: str) -> tuple[int, ...] | None:
    contacts = [_contacts_for_color(block, color) for block, color in zip(blocks, main_colors)]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    matches: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(int(perm[i]), int(perm[i + 1])) for i in range(len(perm) - 1)):
            matches.append(tuple(int(v) for v in perm))
            if len(matches) > 1:
                return None
    return matches[0] if matches else None


def _recolor_block(block: np.ndarray, main_color: int, accent_color: int, rank: int, n: int, bg: int) -> np.ndarray:
    out = np.full(block.shape, int(bg), dtype=int)
    out[block == int(main_color)] = int(OUTPUT_COLOR_ORDER[int(rank)])
    out[block == int(accent_color)] = int(OUTPUT_COLOR_ORDER[int(rank + n)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scrambled two-color 5x5 sprites into a chain, then recolor both roles by chain rank.

    Input structure:
    - The dominant background fills empty space, and a separator color forms full divider rows or columns.
    - Each 5x5 block contains exactly two foreground colors with unequal counts: a larger main color and a smaller accent color.
    - Blocks are scrambled in a strip or a 2x2 layout, with separators between block slots.

    Rule:
    - In every block, treat the more frequent foreground color as the main shape and ignore the smaller accent color when matching neighbors.
    - Find the unique horizontal or vertical block order whose facing main-color edge offsets overlap.
    - If the chain length is N, recolor the kth ordered block so its main pixels become the kth fixed output color and its accent pixels become the (k+N)th fixed output color.
    - Render the ordered recolored blocks as a 1xN strip separated by single separator lines.

    Key insight: The larger color in each block carries the true linkage pattern; after recovering that chain, both block-local roles are remapped by chain position.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)
    blocks = _split_blocks(g, sep)

    assert blocks, "must extract at least one block"
    assert all(block.shape == (BLOCK_SIZE, BLOCK_SIZE) for block in blocks), "all blocks must be 5x5"
    assert len(blocks) <= 4, "position palette supports at most four blocks"

    roles = [_main_and_accent(block, bg) for block in blocks]
    main_colors = [main_color for main_color, _accent in roles]

    chain_h = _find_chain(blocks, main_colors, "h")
    chain_v = _find_chain(blocks, main_colors, "v")

    if chain_h is not None and chain_v is None:
        ordered = [
            _recolor_block(blocks[idx], roles[idx][0], roles[idx][1], rank, len(blocks), bg)
            for rank, idx in enumerate(chain_h)
        ]
        return _render_strip(ordered, axis=1, bg=bg, sep=sep)
    if chain_v is not None and chain_h is None:
        ordered = [
            _recolor_block(blocks[idx], roles[idx][0], roles[idx][1], rank, len(blocks), bg)
            for rank, idx in enumerate(chain_v)
        ]
        return _render_strip(ordered, axis=0, bg=bg, sep=sep)

    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


def _interior_frontier(mask: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
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
    out: list[tuple[int, int]] = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        cell = (int(r) + 1, int(c) + 1)
        if cell not in forbidden:
            out.append(cell)
    return out


def _base_mask(left_offset: int | None, right_offset: int | None, target_area: int, forbidden: set[tuple[int, int]]) -> np.ndarray | None:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left_offset is not None:
        mask[int(left_offset), 0:3] = True
        mask[min(int(left_offset), 2) : max(int(left_offset), 2) + 1, 2] = True
    if right_offset is not None:
        mask[int(right_offset), 2:BLOCK_SIZE] = True
        mask[min(int(right_offset), 2) : max(int(right_offset), 2) + 1, 2] = True

    while int(mask.sum()) < int(target_area):
        frontier = _interior_frontier(mask, forbidden)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[int(r), int(c)] = True

    if left_offset is None:
        if mask[:, 0].any():
            return None
    elif np.where(mask[:, 0])[0].tolist() != [int(left_offset)]:
        return None

    if right_offset is None:
        if mask[:, -1].any():
            return None
    elif np.where(mask[:, -1])[0].tolist() != [int(right_offset)]:
        return None

    if mask[0, :].any() or mask[-1, :].any():
        return None
    if any(mask[r, :].all() for r in range(BLOCK_SIZE)):
        return None
    if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _accent_spec(orient: str) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    if orient == "h":
        coords = [(0, 2), (1, 2), (3, 2), (4, 2)]
        return coords, {(1, 2), (3, 2)}
    coords = [(2, 0), (2, 1), (2, 3), (2, 4)]
    return coords, {(2, 1), (2, 3)}


def _expected_overlap(orient: str, *, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> int:
    if orient == "h":
        offsets = {int(v) for v in (left, right) if v is not None}
    else:
        offsets = {int(v) for v in (top, bottom) if v is not None}
    return len(offsets & {1, 3})


def _make_main_mask(
    orient: str,
    *,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
) -> np.ndarray | None:
    _coords, forbidden = _accent_spec(orient)
    overlap = _expected_overlap(orient, left=left, right=right, top=top, bottom=bottom)
    target_area = PRIMARY_TARGET_BASE + overlap + random.choice([0, 1])

    if orient == "h":
        return _base_mask(left, right, target_area, forbidden)

    rotated_forbidden = {(int(c), int(BLOCK_SIZE - 1 - r)) for r, c in forbidden}
    mask = _base_mask(bottom, top, target_area, rotated_forbidden)
    if mask is None:
        return None
    return np.rot90(mask, 1)


def _make_block(
    main_color: int,
    accent_color: int,
    orient: str,
    *,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
) -> np.ndarray | None:
    main_mask = _make_main_mask(orient, left=left, right=right, top=top, bottom=bottom)
    if main_mask is None:
        return None

    accent_coords, _forbidden = _accent_spec(orient)
    full_mask = main_mask.copy()
    for r, c in accent_coords:
        full_mask[int(r), int(c)] = True

    total_fg = int(full_mask.sum())
    main_count = int(total_fg - len(accent_coords))
    if total_fg not in (9, 10):
        return None
    if main_count <= len(accent_coords):
        return None
    if not is_contiguous(full_mask.astype(int), background=0, connectivity=4):
        return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[full_mask] = int(main_color)
    for r, c in accent_coords:
        block[int(r), int(c)] = int(accent_color)

    if any((block[r, :] == block[r, 0]).all() and int(block[r, 0]) != BG for r in range(BLOCK_SIZE)):
        return None
    if any((block[:, c] == block[0, c]).all() and int(block[0, c]) != BG for c in range(BLOCK_SIZE)):
        return None

    colors = _non_bg_colors(block, BG)
    if colors != sorted([int(main_color), int(accent_color)]):
        return None
    return block


def _count_orders(blocks: list[np.ndarray], main_colors: list[int], orient: str) -> int:
    contacts = [_contacts_for_color(block, color) for block, color in zip(blocks, main_colors)]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(int(perm[i]), int(perm[i + 1])) for i in range(len(perm) - 1)):
            total += 1
    return int(total)


def _count_orders_all_fg(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_contacts_for_color(block, color=None) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(int(perm[i]), int(perm[i + 1])) for i in range(len(perm) - 1)):
            total += 1
    return int(total)


def _expected_output(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    roles = [_main_and_accent(block, BG) for block in blocks]
    recolored = [
        _recolor_block(block, roles[i][0], roles[i][1], i, len(blocks), BG)
        for i, block in enumerate(blocks)
    ]
    axis = 1 if orient == "h" else 0
    return _render_strip(recolored, axis=axis, bg=BG, sep=SEP)


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    perm = list(range(len(blocks)))
    while perm == list(range(len(blocks))):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if len(blocks) == 4 and random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        slots = [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]
        for block, (r0, c0) in zip(scrambled, slots):
            paste(grid, block, int(r0), int(c0))
        return grid

    axis = 1 if OTHER[output_orient] == "h" else 0
    return _render_strip(scrambled, axis=axis, bg=BG, sep=SEP)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    colors = random.sample(ALL_FG_COLORS, 2 * n)
    main_colors = colors[:n]
    accent_colors = colors[n:]
    lead_kw, trail_kw = ORIENT_KW[orient]

    blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = int(shared_offsets[i - 1])
        if i < n - 1:
            kw[trail_kw] = int(shared_offsets[i])
        block = _make_block(int(main_colors[i]), int(accent_colors[i]), orient, **kw)
        if block is None:
            return None
        blocks.append(block)

    if any(_main_and_accent(block, BG)[0] != int(main_colors[i]) for i, block in enumerate(blocks)):
        return None
    if _count_orders(blocks, main_colors, orient) != 1:
        return None
    if _count_orders(blocks, main_colors, OTHER[orient]) != 0:
        return None
    if _count_orders_all_fg(blocks, OTHER[orient]) == 0:
        return None

    plain_out = _render_strip(blocks, axis=(1 if orient == "h" else 0), bg=BG, sep=SEP)
    target_out = _expected_output(blocks, orient)
    if np.array_equal(plain_out, target_out):
        return None
    fg_mask = (plain_out != BG) & (plain_out != SEP)
    changed_frac = float(np.count_nonzero((plain_out != target_out) & fg_mask)) / float(np.count_nonzero(fg_mask))
    if changed_frac < 0.5:
        return None

    inp = _render_input(blocks, orient)
    if not np.array_equal(main(inp), target_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
