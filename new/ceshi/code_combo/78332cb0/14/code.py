# Combo variant of 78332cb0 (index 14)
# combo_spec: D5/3+D9/0
# Name: 双点连接颜色互换

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The input still contains four scrambled 5x5 monochrome sprite blocks,
#      separated by full sep lines and arranged either as a 2x2 grid or as a
#      strip in the non-output orientation.
#    - The original task linked neighbors when their facing edges shared at
#      least one aligned fg pixel. Here each valid adjacency is stricter: the
#      two facing edges must share exactly two aligned fg pixels.
#    - After recovering the unique chain, the output no longer preserves each
#      block's own color. Instead chain positions (0,1) swap colors and
#      positions (2,3) swap colors, while every block keeps its own 5x5 mask.
#    - Trace example:
#        chain edge signatures = [{1,2}, {1,3}, {2,3}]
#        chain colors before swap = [1, 2, 3, 4]
#        exact-two matching yields order [A, B, C, D]
#        final colors become        [2, 1, 4, 3]
#      So the solver must first solve the geometric chain and then apply the
#      color transform on adjacent pairs.
# b) Complexity calibration:
#    - The generator always uses four blocks with four distinct colors so both
#      pairwise swaps are visible in every sample; no odd tail block remains.
#    - There are only three possible 2-of-3 edge signatures on offsets 1..3:
#      {1,2}, {1,3}, {2,3}. Each sample uses all three exactly once along the
#      chain, which keeps the state space small but makes the chain direction
#      and ordering identifiable.
# c) Implicit-assumption audit of the original solver:
#    - `_find_sep` still needs sep to be the only full row/column color. The
#      generator therefore rejects any sprite mask that would create a full 5-cell
#      row or column inside a block.
#    - `_contacts` / edge extraction remains valid because each block is still a
#      single connected component with a dominant background color.
#    - The old chain test `intersection > 0` is no longer sufficient. It must
#      become `intersection == 2`, otherwise the fused task collapses back to
#      the base rule.
#    - The old final render assumed geometry-only reordering. That invariant is
#      broken here, so the ordered blocks must be recolored before rendering.
# d) Placement legality:
#    - Blocks are still isolated by one-cell sep lines; output is still a clean
#      1x4 strip. Pre-enumerated 5x5 templates guarantee in-bounds placement,
#      zero overlap, zero perpendicular-edge fg, and 4-connectedness.
# e) Visual observability:
#    - On the geometry side, every valid connection visibly uses two aligned
#      edge pixels instead of one. On the color side, each output pair clearly
#      exchanges colors, so simple reordering is visibly insufficient.
# f) Visual cleanliness:
#    - Each block remains monochrome and connected with no extra marks. The
#      puzzle is still read as four simple sprites plus separator lines.
# g) Color-rule inferability:
#    - The recoloring rule is local and repeated twice per sample: swap colors
#      within each adjacent output pair. Because all four chain colors differ,
#      the color motion is unambiguous to a human observer.
# h) Generator non-degeneration checks:
#    - It verifies unique solvability in exactly one orientation under the
#      exact-two matcher.
#    - It rejects any sample where the fused output equals the old geometry-only
#      output, so the color-swap dimension is always active.
# i) New solver assumptions introduced here:
#    - Valid chain edges have exactly two aligned edge contacts, not one.
#    - Each block is single-color and every sample uses four blocks; the
#      generator enforces both assumptions directly.


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
BLOCK_COUNT = 4
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = (1, 2, 3)
EDGE_SIGNATURES = [(1, 2), (1, 3), (2, 3)]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 300

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_EMPTY_SIG: tuple[int, ...] = ()


def _edge_sets(block: np.ndarray, bg: int) -> dict[str, set[int]]:
    mask = block != int(bg)
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


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
    out: list[np.ndarray] = []
    for i in range(0, len(blocks), 2):
        out.append(_recolor_block(blocks[i], bg, colors[i + 1]))
        out.append(_recolor_block(blocks[i + 1], bg, colors[i]))
    return out


def _enumerate_horizontal_templates() -> dict[tuple[tuple[int, ...], tuple[int, ...]], list[np.ndarray]]:
    cells = [(r, c) for r in range(1, 4) for c in range(BLOCK_SIZE)]
    templates: dict[tuple[tuple[int, ...], tuple[int, ...]], list[np.ndarray]] = {}

    for bits in range(1 << len(cells)):
        mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
        for i, (r, c) in enumerate(cells):
            if (bits >> i) & 1:
                mask[r, c] = True

        area = int(mask.sum())
        if not (FG_AREA_MIN <= area <= FG_AREA_MAX):
            continue
        if mask[0].any() or mask[-1].any():
            continue
        if any(mask[r].all() for r in range(BLOCK_SIZE)):
            continue
        if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue

        left = tuple(int(x) for x in np.where(mask[:, 0])[0].tolist())
        right = tuple(int(x) for x in np.where(mask[:, -1])[0].tolist())
        if len(left) not in (0, 2) or len(right) not in (0, 2):
            continue
        if any(x not in EDGE_CONTACT_OFFSETS for x in left + right):
            continue

        key = (left, right)
        templates.setdefault(key, []).append(mask.copy())

    needed: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for sig in EDGE_SIGNATURES:
        needed.append((_EMPTY_SIG, sig))
        needed.append((sig, _EMPTY_SIG))
    for left in EDGE_SIGNATURES:
        for right in EDGE_SIGNATURES:
            if left != right:
                needed.append((left, right))
    for key in needed:
        assert key in templates, f"missing template for {key}"
    return templates


_HORIZONTAL_TEMPLATES = _enumerate_horizontal_templates()


def _make_block(
    color: int,
    orient: str,
    lead_sig: tuple[int, ...],
    trail_sig: tuple[int, ...],
) -> np.ndarray:
    if orient == "h":
        mask = random.choice(_HORIZONTAL_TEMPLATES[(lead_sig, trail_sig)]).copy()
    else:
        mask = np.rot90(random.choice(_HORIZONTAL_TEMPLATES[(trail_sig, lead_sig)]), 1).copy()
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), int(BG), dtype=int)
    block[mask] = int(color)
    return block


def _render_strip(blocks: list[np.ndarray], orient: str, bg: int, sep: int) -> np.ndarray:
    stride = BLOCK_SIZE + 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, BLOCK_COUNT * BLOCK_SIZE + (BLOCK_COUNT - 1)), int(bg), dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < len(blocks) - 1:
                out[:, i * stride + BLOCK_SIZE] = int(sep)
        return out

    out = np.full((BLOCK_COUNT * BLOCK_SIZE + (BLOCK_COUNT - 1), BLOCK_SIZE), int(bg), dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * stride, 0)
        if i < len(blocks) - 1:
            out[i * stride + BLOCK_SIZE, :] = int(sep)
    return out


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    perm = list(range(len(blocks)))
    while perm == list(range(len(blocks))):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), int(BG), dtype=int)
        grid[BLOCK_SIZE, :] = int(SEP)
        grid[:, BLOCK_SIZE] = int(SEP)
        for block, (r0, c0) in zip(
            scrambled,
            [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)],
        ):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient], BG, SEP)


def _count_orders(blocks: list[np.ndarray], orient: str, bg: int) -> int:
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return len(contacts[a][lead] & contacts[b][trail]) == 2

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _count_recolored_positions(before: list[np.ndarray], after: list[np.ndarray]) -> int:
    total = 0
    for block_before, block_after in zip(before, after):
        if _fg_color(block_before, BG) != _fg_color(block_after, BG):
            total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scrambled 5x5 blocks into the unique strip whose neighbors share exactly two aligned edge pixels, then swap colors within each adjacent pair.

    Input structure:
    - bg is the dominant background color and sep is the full-line divider color.
    - The grid contains four separate 5x5 monochrome sprite blocks, either laid out as a 2x2 panel or as a strip, with sep rows/columns between blocks.

    Rule:
    - Test both horizontal and vertical chain directions.
    - Two consecutive blocks are adjacent only when their facing edges have exactly two foreground pixels at the same offsets.
    - The one direction that yields a complete unique chain gives the block order.
    - After ordering, swap the foreground colors of chain positions (0,1) and (2,3), then render the strip with sep dividers.

    Key insight:
    - The geometry clue is stricter than the base task: valid links are identified by a two-point edge signature, and once that chain is recovered the colors exchange ownership inside each neighboring pair.
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

    def _find_unique_chain(blocks: list[np.ndarray], orient: str) -> tuple[int, ...] | None:
        contacts = [_edge_sets(block, bg) for block in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return len(contacts[a]["right"] & contacts[b]["left"]) == 2
            return len(contacts[a]["bot"] & contacts[b]["top"]) == 2

        matches: list[tuple[int, ...]] = []
        for perm in itertools.permutations(range(len(blocks))):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
                matches.append(tuple(int(x) for x in perm))
                if len(matches) > 1:
                    raise AssertionError(f"ambiguous {orient}-chain")
        return matches[0] if matches else None

    blocks = _split_blocks()
    assert len(blocks) == BLOCK_COUNT, "must find exactly four blocks"

    chain_h = _find_unique_chain(blocks, "h")
    chain_v = _find_unique_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        ordered = [blocks[i] for i in chain_h]
        return _render_strip(_swap_adjacent_colors(ordered, bg), "h", bg, sep)
    if chain_v is not None and chain_h is None:
        ordered = [blocks[i] for i in chain_v]
        return _render_strip(_swap_adjacent_colors(ordered, bg), "v", bg, sep)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(["h", "v"])
    signatures = random.sample(EDGE_SIGNATURES, len(EDGE_SIGNATURES))
    colors = random.sample(PALETTE, BLOCK_COUNT)

    blocks: list[np.ndarray] = []
    for i, color in enumerate(colors):
        lead_sig = signatures[i - 1] if i > 0 else _EMPTY_SIG
        trail_sig = signatures[i] if i < BLOCK_COUNT - 1 else _EMPTY_SIG
        blocks.append(_make_block(color, orient, lead_sig, trail_sig))

    if _count_orders(blocks, orient, BG) != 1:
        return None
    if _count_orders(blocks, _OTHER[orient], BG) != 0:
        return None

    swapped_blocks = _swap_adjacent_colors(blocks, BG)
    if _count_recolored_positions(blocks, swapped_blocks) != BLOCK_COUNT:
        return None

    out = _render_strip(swapped_blocks, orient, BG, SEP)
    old_out = _render_strip(blocks, orient, BG, SEP)
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
