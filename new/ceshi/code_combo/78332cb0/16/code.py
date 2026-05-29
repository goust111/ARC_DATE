# Combo variant of 78332cb0 (index 16)
# combo_spec: D1/4+D15/2
# Name: 互补齿合含洞精灵

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

# Design analysis
# a) Concrete behavior:
#    - Source task: scramble 3-4 monochrome 5x5 sprites, then recover the unique strip order by
#      looking for shared foreground pixels on facing edges.
#    - Fusion change 1 (D1/4): a valid seam is no longer "same offset is occupied on both sides".
#      Instead, each of the three usable interior edge slots must flip: if the predecessor has a
#      tooth at that slot, the successor must have a gap there, and vice versa.
#    - Fusion change 2 (D15/2): sprites are no longer sparse tree-like blobs. Each block is built
#      around a visible hollow center, so the edge pattern sits on a ring/U-like body rather than
#      on a solid chunk. This makes the edge occupancy distribution visibly different from the
#      source puzzle and also breaks the old "infer local bg from the block majority" shortcut.
#    - Concrete horizontal example with interior edge slots {1,2,3}:
#      A.right = {1,3}, B.left = {2}; these are exact complements on the three interior slots, so
#      A can precede B. Under the source rule, the same pair does not match because their overlap
#      is empty. The center hole remains background in both blocks and is carried unchanged to the
#      output strip.
# b) Complexity calibration:
#    - We keep block size at 5x5 and use only the three interior edge slots. Allowing the corner
#      slots too would either destroy the clean hollow silhouette or force noisy corner contacts.
#    - Seam patterns are chosen from the six non-trivial 3-bit masks (neither empty nor full) and
#      sampled without replacement. This keeps the chain unique while still giving enough variety
#      for n=4 instances.
#    - Every block contains exactly one enclosed hole. That is strong enough to make the variant
#      visually obvious without introducing extra matching keys humans would have to infer.
# c) Hidden-assumption audit against the source solver:
#    - `_split_blocks` and separator detection still hold because the input layout family is
#      unchanged.
#    - The source `_contacts` used the most common color inside each 5x5 block as local background.
#      That assumption becomes fragile once hollow ring-like sprites may use 8-14 fg cells, so the
#      fused solver reads foreground relative to the global background color instead.
#    - The source adjacency test only required a non-empty intersection on one offset. That is
#      invalid now; the solver must compare the full 3-slot edge bitmaps and require exact
#      complementarity.
#    - New solver assumptions introduced here:
#      1. all informative seam contacts live on interior slots {1,2,3};
#      2. exactly one orientation and one permutation satisfy the complement rule.
#      The generator enforces both properties and validates them by brute force.
# d) Placement legality:
#    - Input placement reuses the original strip / 2x2 layouts, so blocks never overlap and always
#      stay inside the canvas.
#    - Output placement is still a separated 1xN strip, which keeps the hollow centers legible.
# e) Visual observability:
#    - Across a few examples, humans can see two stable cues at once:
#      1. neighboring edge teeth alternate as complements instead of lining up on the same rows;
#      2. every sprite has a clear center void, so the family does not look like the source task.
#    - Because the source overlap rule would reject these seams, the fusion is not a cosmetic
#      rewrite of the original behavior.
# f) Visual cleanliness:
#    - All blocks share the same simple ring skeleton, so the only changing visual signal is which
#      edge slots are occupied. That keeps the puzzle readable despite the added hole structure.
# g) Color inferability:
#    - No recoloring rule is introduced. Colors remain per-block decoration only, which avoids
#      burdening the examples with an unrelated color mapping problem.
# h) Fusion metrics used after testing:
#    - `hole_blocks`: every generated block must have exactly one enclosed hole.
#    - `legacy_chain_orders`: under the source overlap rule, the intended chain orientation should
#      admit zero valid permutations, confirming the complement control variable is active.
# i) Generator guarantees for the new solver assumptions:
#    - each block is a single 4-connected component with one enclosed hole;
#    - every true seam is an exact complement on the three interior slots;
#    - seam patterns are pairwise distinct;
#    - the opposite orientation admits no valid complement chain.

BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
INTERIOR_OFFSETS = (1, 2, 3)
EMPTY_PATTERN = (0, 0, 0)
SEAM_PATTERNS = [
    tuple(int(i in rows) for i in INTERIOR_OFFSETS)
    for rows in (
        (1,),
        (2,),
        (3,),
        (1, 2),
        (1, 3),
        (2, 3),
    )
]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400

OTHER = {"h": "v", "v": "h"}


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    for r in range(int(a.shape[0])):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
            cand.append(int(a[r, 0]))
    for c in range(int(a.shape[1])):
        if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
            cand.append(int(a[0, c]))

    def valid_sep(sep: int) -> bool:
        rows = _full_sep_rows(a, sep)
        cols = _full_sep_cols(a, sep)
        if rows and cols:
            if len(rows) != 1 or len(cols) != 1:
                return False
            r = int(rows[0])
            c = int(cols[0])
            quads = [
                a[:r, :c],
                a[:r, c + 1 :],
                a[r + 1 :, :c],
                a[r + 1 :, c + 1 :],
            ]
            return all(block.shape == (BLOCK_SIZE, BLOCK_SIZE) for block in quads)
        if rows:
            segs = _split_by_rows(a, sep)
            return 3 <= len(segs) <= 4 and all(seg.shape == (BLOCK_SIZE, BLOCK_SIZE) for seg in segs)
        if cols:
            segs = _split_by_cols(a, sep)
            return 3 <= len(segs) <= 4 and all(seg.shape == (BLOCK_SIZE, BLOCK_SIZE) for seg in segs)
        return False

    valid = [color for color in cand if valid_sep(int(color))]
    assert valid, "separator color not found"
    return int(Counter(valid).most_common(1)[0][0])


def _full_sep_rows(a: np.ndarray, sep: int) -> list[int]:
    return [int(r) for r in range(int(a.shape[0])) if bool((a[r, :] == int(sep)).all())]


def _full_sep_cols(a: np.ndarray, sep: int) -> list[int]:
    return [int(c) for c in range(int(a.shape[1])) if bool((a[:, c] == int(sep)).all())]


def _split_by_rows(a: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for r in rows + [int(a.shape[0])]:
        if r > start:
            segs.append(a[start:r, :].copy())
        start = r + 1
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if c > start:
            segs.append(a[:, start:c].copy())
        start = c + 1
    return segs


def _split_blocks(g: np.ndarray, sep: int) -> list[np.ndarray]:
    rows = _full_sep_rows(g, sep)
    cols = _full_sep_cols(g, sep)
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
        return _split_by_rows(g, sep)
    if cols:
        return _split_by_cols(g, sep)
    return []


def _complement(pattern: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(1 - int(v) for v in pattern)


def _edge_pattern(block: np.ndarray, side: str, bg: int) -> tuple[int, int, int]:
    mask = block != int(bg)
    if side == "left":
        return tuple(int(mask[r, 0]) for r in INTERIOR_OFFSETS)
    if side == "right":
        return tuple(int(mask[r, BLOCK_SIZE - 1]) for r in INTERIOR_OFFSETS)
    if side == "top":
        return tuple(int(mask[0, c]) for c in INTERIOR_OFFSETS)
    if side == "bot":
        return tuple(int(mask[BLOCK_SIZE - 1, c]) for c in INTERIOR_OFFSETS)
    raise ValueError(f"unknown side: {side}")


def _all_edge_patterns(block: np.ndarray, bg: int) -> dict[str, tuple[int, int, int]]:
    return {
        "left": _edge_pattern(block, "left", bg),
        "right": _edge_pattern(block, "right", bg),
        "top": _edge_pattern(block, "top", bg),
        "bot": _edge_pattern(block, "bot", bg),
    }


def _find_chain_perms(blocks: list[np.ndarray], orient: str, bg: int) -> list[tuple[int, ...]]:
    edges = [_all_edge_patterns(block, bg) for block in blocks]

    def ok(a: int, b: int) -> bool:
        if orient == "h":
            return edges[a]["right"] == _complement(edges[b]["left"])
        return edges[a]["bot"] == _complement(edges[b]["top"])

    out: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            out.append(tuple(int(x) for x in perm))
    return out


def _render_strip(blocks: list[np.ndarray], axis: int, bg: int, sep: int) -> np.ndarray:
    n = len(blocks)
    if axis == 0:
        out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), int(bg), dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            out[rr : rr + BLOCK_SIZE, :] = block
            rr += BLOCK_SIZE
            if i < n - 1:
                out[rr, :] = int(sep)
                rr += 1
        return out

    out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), int(bg), dtype=int)
    cc = 0
    for i, block in enumerate(blocks):
        out[:, cc : cc + BLOCK_SIZE] = block
        cc += BLOCK_SIZE
        if i < n - 1:
            out[:, cc] = int(sep)
            cc += 1
    return out


def _hole_count(block: np.ndarray, bg: int) -> int:
    bg_mask = block == int(bg)
    seen: set[tuple[int, int]] = set()
    stack = [(r, c) for r in range(BLOCK_SIZE) for c in range(BLOCK_SIZE) if bg_mask[r, c] and (r in (0, BLOCK_SIZE - 1) or c in (0, BLOCK_SIZE - 1))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BLOCK_SIZE and 0 <= nc < BLOCK_SIZE and bg_mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))

    holes = 0
    seen_inner: set[tuple[int, int]] = set()
    for r in range(BLOCK_SIZE):
        for c in range(BLOCK_SIZE):
            if not bg_mask[r, c] or (r, c) in seen or (r, c) in seen_inner:
                continue
            holes += 1
            comp = [(r, c)]
            while comp:
                rr, cc = comp.pop()
                if (rr, cc) in seen_inner:
                    continue
                seen_inner.add((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < BLOCK_SIZE and 0 <= nc < BLOCK_SIZE and bg_mask[nr, nc] and (nr, nc) not in seen and (nr, nc) not in seen_inner:
                        comp.append((nr, nc))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder hollow 5x5 sprites into a strip by complementary edge teeth.

    Input structure:
    - Background cells fill the canvas; separator rows/columns use one solid non-background color.
    - The actual objects are 3-4 monochrome 5x5 sprites, laid out in a strip or a 2x2 grid.
    - Each sprite contains a visible interior void, so the block should be read against the global
      background color rather than by assuming the block's majority color is background.

    Rule:
    - Split the input into the 5x5 blocks.
    - Test both horizontal and vertical orderings. Two neighboring blocks connect only when their
      facing edges are exact complements on the three interior edge slots: at each usable slot,
      exactly one side has foreground.
    - Exactly one orientation admits one complete ordering. Output that ordered chain as a 1xN
      strip, keeping the original 5x5 blocks unchanged and separating them with single sep lines.

    Key insight: The ordering key is the full interior edge bit pattern, not overlap. The hollow
    sprite bodies matter because they make multi-slot edge teeth readable and invalidate the old
    local-majority background heuristic.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)
    blocks = _split_blocks(g, sep)
    assert blocks, "no blocks found"

    chain_h = _find_chain_perms(blocks, "h", bg)
    chain_v = _find_chain_perms(blocks, "v", bg)

    if len(chain_h) == 1 and not chain_v:
        return _render_strip([blocks[i] for i in chain_h[0]], axis=1, bg=bg, sep=sep)
    if len(chain_v) == 1 and not chain_h:
        return _render_strip([blocks[i] for i in chain_v[0]], axis=0, bg=bg, sep=sep)
    assert not chain_h and not chain_v, "ambiguous chain"
    raise AssertionError("must find exactly one valid chain")


def _ring_mask_horizontal(left: tuple[int, int, int], right: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    mask[1:4, 1:4] = True
    mask[2, 2] = False
    for idx, bit in enumerate(left, start=1):
        if bit:
            mask[idx, 0] = True
    for idx, bit in enumerate(right, start=1):
        if bit:
            mask[idx, BLOCK_SIZE - 1] = True
    return mask


def _ring_mask_vertical(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    mask[1:4, 1:4] = True
    mask[2, 2] = False
    for idx, bit in enumerate(top, start=1):
        if bit:
            mask[0, idx] = True
    for idx, bit in enumerate(bottom, start=1):
        if bit:
            mask[BLOCK_SIZE - 1, idx] = True
    return mask


def _make_block(color: int, orient: str, lead: tuple[int, int, int], trail: tuple[int, int, int]) -> np.ndarray:
    if orient == "h":
        mask = _ring_mask_horizontal(lead, trail)
    else:
        mask = _ring_mask_vertical(lead, trail)

    assert is_contiguous(mask.astype(int), background=0, connectivity=4)
    assert _hole_count(np.where(mask, color, BG), BG) == 1

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _count_chain_orders(blocks: list[np.ndarray], orient: str) -> int:
    return len(_find_chain_perms(blocks, orient, BG))


def _count_legacy_overlap_orders(blocks: list[np.ndarray], orient: str) -> int:
    edges = [_all_edge_patterns(block, BG) for block in blocks]

    def ok(a: int, b: int) -> bool:
        if orient == "h":
            return any(edges[a]["right"][i] and edges[b]["left"][i] for i in range(3))
        return any(edges[a]["bot"][i] and edges[b]["top"][i] for i in range(3))

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


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
        for block, (r0, c0) in zip(scrambled, ((0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1))):
            paste(grid, block, r0, c0)
        return grid

    axis = 1 if OTHER[output_orient] == "h" else 0
    return _render_strip(scrambled, axis=axis, bg=BG, sep=SEP)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    seam_patterns = random.sample(SEAM_PATTERNS, n - 1)

    blocks: list[np.ndarray] = []
    for i in range(n):
        lead = EMPTY_PATTERN if i == 0 else _complement(seam_patterns[i - 1])
        trail = EMPTY_PATTERN if i == n - 1 else seam_patterns[i]
        color = random.choice(PALETTE)
        blocks.append(_make_block(color, orient, lead, trail))

    if any(_hole_count(block, BG) != 1 for block in blocks):
        return None
    if _count_chain_orders(blocks, orient) != 1:
        return None
    if _count_chain_orders(blocks, OTHER[orient]) != 0:
        return None
    if _count_legacy_overlap_orders(blocks, orient) != 0:
        return None

    expected = _render_strip(blocks, axis=1 if orient == "h" else 0, bg=BG, sep=SEP)
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
