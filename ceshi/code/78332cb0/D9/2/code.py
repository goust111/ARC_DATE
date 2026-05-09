# Variant of 78332cb0
# Dimension: D9 — 链的连接语义 (match_cond)
# Change: 每对相邻块的连接边缘恰好有1个前景像素在相同偏移处重合... → 拼成连续矩形的边缘对齐

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, paste

"""
Design analysis for the D9 variant
==================================

variant_spec:
  unchanged_core:
    - bg is still the most common color
    - sep is still a full-row/full-column divider color
    - input still contains 3-4 scrambled 5x5 monochrome blocks
    - output is still a 1xN strip in the recovered chain order
  changed_dimension:
    old_match:
      Two neighbors were valid when the predecessor's trailing edge and the
      successor's leading edge shared at least one foreground offset.
    new_match:
      Two neighbors are valid only when, after removing the separator and
      abutting them directly, the union of their full foreground masks fills
      one solid rectangle whose bbox crosses the seam.
  concrete_shape_choice:
    - Horizontal chains use 2-pixel-high horizontal bands on interior rows.
    - Vertical chains use the rotated analogue on interior columns.
    - First/last blocks are rectangular "caps" that are open on the outer side.
    - Interior blocks are full-width/full-height bands.
    - For n=4, the two middle blocks are made visually identical so swapping
      them does not create a second visible answer.

behavior_trace:
  example_horizontal:
    span_rows: [1, 2]
    start_block: rows 1..2, cols 2..4
    middle_block: rows 1..2, cols 0..4
    end_block: rows 1..2, cols 0..2
    original_logic:
      It would only ask whether right-edge rows and left-edge rows intersect.
      Many thin edge contacts could qualify even if the two full shapes formed
      an L-shape or left a cavity.
    new_logic:
      Place the two blocks side by side with no separator, crop to the
      foreground bbox, and require every cell of that bbox to be foreground.
      The pair above passes because it becomes one 2x8 filled rectangle.
      If start/end are swapped, the seam creates a gap and the rectangle test
      fails immediately.

complexity_calibration:
  - Using whole rectangular bands makes the D9 change visible at a glance:
    the contact is now a 2-pixel bar, not a singleton pixel.
  - Restricting spans to two interior bands, [1,2] or [2,3], keeps the rule
    legible from 2-4 examples and avoids accidental vertical chaining.
  - Keeping n in {3,4} preserves the original scale; the only adaptation is
    that four-block cases duplicate the two middle blocks to avoid a hidden
    permutation ambiguity that pure rectangle semantics would otherwise create.

implicit_assumption_audit_for_main:
  - _split_blocks still assumes separator rows/cols are complete lines.
    The generator preserves that exactly.
  - _pair_forms_rectangle assumes each block keeps its own 5x5 extent and that
    matching is evaluated by direct abutment without overlap. The generator
    constructs only cap/full-band rectangles, so this assumption is satisfied.
  - _find_chain still brute-forces permutations. This remains cheap because
    there are at most 4 blocks.
  - The solver now assumes exactly one orientation yields at least one visible
    strip answer. The generator verifies horizontal/vertical visible-output
    sets and rejects ambiguous samples.

placement_legality:
  - Blocks are still rendered into either a strip with 1-cell separators or a
    2x2 grid (only when n=4), so there is no overlap and no out-of-bounds paste.
  - Because every block is exactly 5x5 and the separator thickness stays 1, the
    existing block-splitting logic remains valid unchanged.

visual_observability:
  - Humans can now see that pieces are stitched by aligning full rectangular
    bars into larger rectangles, not by matching a single touching pixel.
  - Horizontal chains never touch top/bottom borders; vertical chains never
    touch left/right borders. This makes the orientation cue explicit.

main_generator_contract:
  - main() expects the rectangle pair test to be the only way a chain forms.
  - generate_input() enforces this by constructing cap/full-band pieces and by
    rejecting any sample whose other orientation yields a valid visible strip.
"""


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]
BLOCK_SIZE = 5
GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 300
SPAN_CHOICES = [(1, 2), (2, 3)]
CAP_INSET_CHOICES = [1, 2]
OTHER = {"h": "v", "v": "h"}


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    for r in range(int(a.shape[0])):
        if bool((a[int(r), :] == int(a[int(r), 0])).all()) and int(a[int(r), 0]) != int(bg):
            cand.append(int(a[int(r), 0]))
    for c in range(int(a.shape[1])):
        if bool((a[:, int(c)] == int(a[0, int(c)])).all()) and int(a[0, int(c)]) != int(bg):
            cand.append(int(a[0, int(c)]))
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
            segs.append(a[int(start):int(r), :].copy())
        start = int(r + 1)
    return segs


def _split_by_cols(a: np.ndarray, sep: int) -> list[np.ndarray]:
    cols = _full_sep_cols(a, sep)
    segs: list[np.ndarray] = []
    start = 0
    for c in cols + [int(a.shape[1])]:
        if int(c) > int(start):
            segs.append(a[:, int(start):int(c)].copy())
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
            g[:r, int(c + 1):].copy(),
            g[int(r + 1):, :c].copy(),
            g[int(r + 1):, int(c + 1):].copy(),
        ]
    if rows:
        return _split_by_rows(g, sep)
    if cols:
        return _split_by_cols(g, sep)
    return []


def _split_blocks_and_sep(g: np.ndarray, bg: int) -> tuple[list[np.ndarray], int]:
    """
    Prefer geometry-based splitting because this variant intentionally contains
    full monochrome rows/cols inside blocks, which makes color-only separator
    inference unreliable.
    """
    h, w = map(int, g.shape)

    if h == int(2 * BLOCK_SIZE + 1) and w == int(2 * BLOCK_SIZE + 1):
        sep = int(Counter(map(int, np.concatenate([g[BLOCK_SIZE, :], g[:, BLOCK_SIZE]]))).most_common(1)[0][0])
        blocks = [
            g[:BLOCK_SIZE, :BLOCK_SIZE].copy(),
            g[:BLOCK_SIZE, BLOCK_SIZE + 1:].copy(),
            g[BLOCK_SIZE + 1:, :BLOCK_SIZE].copy(),
            g[BLOCK_SIZE + 1:, BLOCK_SIZE + 1:].copy(),
        ]
        return blocks, sep

    if h == BLOCK_SIZE and (w + 1) % (BLOCK_SIZE + 1) == 0:
        n = (w + 1) // (BLOCK_SIZE + 1)
        if n in (3, 4):
            sep_cols = [BLOCK_SIZE + i * (BLOCK_SIZE + 1) for i in range(n - 1)]
            sep = int(Counter(map(int, g[:, sep_cols].ravel())).most_common(1)[0][0])
            blocks = [g[:, i * (BLOCK_SIZE + 1):i * (BLOCK_SIZE + 1) + BLOCK_SIZE].copy() for i in range(n)]
            return blocks, sep

    if w == BLOCK_SIZE and (h + 1) % (BLOCK_SIZE + 1) == 0:
        n = (h + 1) // (BLOCK_SIZE + 1)
        if n in (3, 4):
            sep_rows = [BLOCK_SIZE + i * (BLOCK_SIZE + 1) for i in range(n - 1)]
            sep = int(Counter(map(int, g[sep_rows, :].ravel())).most_common(1)[0][0])
            blocks = [g[i * (BLOCK_SIZE + 1):i * (BLOCK_SIZE + 1) + BLOCK_SIZE, :].copy() for i in range(n)]
            return blocks, sep

    sep = _find_sep(g, bg)
    return _split_blocks(g, sep), sep


def _render_strip(blocks: list[np.ndarray], axis: int, bg: int, sep: int) -> np.ndarray:
    n = int(len(blocks))
    if int(axis) == 0:
        out = np.full((int(n * BLOCK_SIZE + (n - 1)), BLOCK_SIZE), int(bg), dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            out[int(rr):int(rr + BLOCK_SIZE), :] = block
            rr += BLOCK_SIZE
            if int(i) < int(n - 1):
                out[int(rr), :] = int(sep)
                rr += 1
        return out
    out = np.full((BLOCK_SIZE, int(n * BLOCK_SIZE + (n - 1))), int(bg), dtype=int)
    cc = 0
    for i, block in enumerate(blocks):
        out[:, int(cc):int(cc + BLOCK_SIZE)] = block
        cc += BLOCK_SIZE
        if int(i) < int(n - 1):
            out[:, int(cc)] = int(sep)
            cc += 1
    return out


def _pair_forms_rectangle(a: np.ndarray, b: np.ndarray, orient: str, bg: int) -> bool:
    ma = a != int(bg)
    mb = b != int(bg)
    if not ma.any() or not mb.any():
        return False

    if orient == "h":
        joined = np.zeros((BLOCK_SIZE, BLOCK_SIZE * 2), dtype=bool)
        joined[:, :BLOCK_SIZE] = ma
        joined[:, BLOCK_SIZE:] = mb
        pts = np.argwhere(joined)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if not (int(c0) < BLOCK_SIZE <= int(c1)):
            return False
    else:
        joined = np.zeros((BLOCK_SIZE * 2, BLOCK_SIZE), dtype=bool)
        joined[:BLOCK_SIZE, :] = ma
        joined[BLOCK_SIZE:, :] = mb
        pts = np.argwhere(joined)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if not (int(r0) < BLOCK_SIZE <= int(r1)):
            return False

    return bool(joined[int(r0):int(r1 + 1), int(c0):int(c1 + 1)].all())


def _find_chain(blocks: list[np.ndarray], orient: str, bg: int) -> tuple[int, ...] | None:
    idx = list(range(int(len(blocks))))
    for perm in itertools.permutations(idx):
        if all(_pair_forms_rectangle(blocks[perm[i]], blocks[perm[i + 1]], orient, bg) for i in range(len(idx) - 1)):
            return perm
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a strip whose adjacent pairs complete solid rectangles.

    Relative to the original task, the only semantic change is the chain test:
    matching no longer means "share at least one aligned edge pixel". It now
    means "when the separator is removed, the two full block masks tile one
    larger rectangle across their seam".
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    blocks, sep = _split_blocks_and_sep(g, bg)
    assert blocks, "input must contain at least one separated 5x5 block"

    chain_h = _find_chain(blocks, "h", bg)
    chain_v = _find_chain(blocks, "v", bg)

    if chain_h is not None and chain_v is None:
        return _render_strip([blocks[i] for i in chain_h], axis=1, bg=bg, sep=sep)
    if chain_v is not None and chain_h is None:
        return _render_strip([blocks[i] for i in chain_v], axis=0, bg=bg, sep=sep)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"


def _make_band_block(
    color: int,
    orient: str,
    span: tuple[int, int],
    role: str,
    inset: int,
) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    lo, hi = span

    if orient == "h":
        r0, r1 = int(lo), int(hi + 1)
        if role == "start":
            c0, c1 = int(inset), BLOCK_SIZE
        elif role == "middle":
            c0, c1 = 0, BLOCK_SIZE
        else:
            c0, c1 = 0, int(BLOCK_SIZE - inset)
        block[r0:r1, c0:c1] = int(color)
        return block

    c0, c1 = int(lo), int(hi + 1)
    if role == "start":
        r0, r1 = int(inset), BLOCK_SIZE
    elif role == "middle":
        r0, r1 = 0, BLOCK_SIZE
    else:
        r0, r1 = 0, int(BLOCK_SIZE - inset)
    block[r0:r1, c0:c1] = int(color)
    return block


def _valid_output_signatures(blocks: list[np.ndarray], orient: str) -> set[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    idx = list(range(int(len(blocks))))
    for perm in itertools.permutations(idx):
        if all(_pair_forms_rectangle(blocks[perm[i]], blocks[perm[i + 1]], orient, BG) for i in range(len(idx) - 1)):
            out = _render_strip([blocks[i] for i in perm], axis=1 if orient == "h" else 0, bg=BG, sep=SEP)
            seen.add(tuple(map(int, out.ravel())))
    return seen


def _build_chain_blocks(n: int, orient: str) -> list[np.ndarray]:
    span = random.choice(SPAN_CHOICES)
    start_inset = random.choice(CAP_INSET_CHOICES)
    end_inset = random.choice(CAP_INSET_CHOICES)

    if n == 4:
        start_color, middle_color, end_color = random.sample(PALETTE, 3)
        middle_block = _make_band_block(middle_color, orient, span, "middle", 0)
        return [
            _make_band_block(start_color, orient, span, "start", start_inset),
            middle_block.copy(),
            middle_block.copy(),
            _make_band_block(end_color, orient, span, "end", end_inset),
        ]

    colors = random.sample(PALETTE, 3)
    return [
        _make_band_block(colors[0], orient, span, "start", start_inset),
        _make_band_block(colors[1], orient, span, "middle", 0),
        _make_band_block(colors[2], orient, span, "end", end_inset),
    ]


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

    return _render_strip(scrambled, axis=0 if output_orient == "h" else 1, bg=BG, sep=SEP)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    blocks = _build_chain_blocks(n, orient)

    valid_here = _valid_output_signatures(blocks, orient)
    valid_other = _valid_output_signatures(blocks, OTHER[orient])
    if len(valid_here) != 1 or valid_other:
        return None

    out = _render_strip(blocks, axis=1 if orient == "h" else 0, bg=BG, sep=SEP)
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
