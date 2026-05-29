from __future__ import annotations

# Combo variant of 78332cb0 (index 5)
# combo_spec: D6/0+D7/0
# Name: 大网格散布五块拼链
#
# Design analysis for the fused variant
# -------------------------------------
# spec_like_analysis:
#   concrete_trace:
#     - original input used full-row/full-column separator lines, so the solver could slice
#       fixed 5x5 blocks directly from a strip or a 2x2 partition and then search at most
#       24 permutations
#     - fused input instead shows five isolated framed cards scattered on a larger board;
#       for example, five 7x7 cards might sit at unrelated coordinates such as
#       (2,3), (4,19), (12,8), (18,24), (23,2)
#     - each card contains one untouched 5x5 sprite; after detecting the five frames, the
#       solver crops those interiors and searches 120 permutations to find the only chain
#   complexity_calibration:
#     - fixing the count at five is the real D7 change, but single-pixel offsets on only
#       {1,2,3} are not enough to label four links robustly
#     - therefore each connecting edge uses a small visible subset of the interior offsets
#       {1,2,3}; only subset sequences that yield exactly one Hamiltonian path are accepted
#     - this keeps the rule human-visible while avoiding uncontrolled freedom or ambiguity
#   implicit_assumption_audit:
#     - old separator inference assumed board-spanning divider lines; invalid after D6, so the
#       solver now treats the most common non-bg color as the local card-frame color
#     - old block extraction assumed one global strip/grid partition; invalid after D6, so the
#       solver now finds 4-connected sep components and validates each as a 7x7 frame
#     - old generator sampled distinct offsets with random.sample(..., n-1); invalid for five
#       blocks because there are four links but only three singleton interior offsets
#     - old solver returned the first matching permutation; too weak for the 5-block setting,
#       so the new solver enumerates all valid orders and demands uniqueness explicitly
#   placement_legality:
#     - every input card is 7x7, stays inside the board, and is separated from every other
#       card by at least one background cell, so sep components never merge
#     - the generator rejects strip-like placements and requires at least three distinct card
#       rows and three distinct card columns, making the "scattered large board" behavior clear
#   visual_observability:
#     - the five cards are visible objects with identical sep frames, so a human can first
#       notice "find five items on the large board" and then compare edge-contact patterns
#     - no hidden ordering by color is used; colors vary independently of the chain
#   cleanliness:
#     - the board contains only background, five framed cards, and their inner monochrome
#       sprites; there is no overlapping markup and no mixed encoding in the same region
#   new_main_assumptions:
#     - every sep component in the input is one valid 7x7 frame
#     - there are exactly five such frames
#     - once the 5x5 interiors are recovered, exactly one orientation has exactly one valid
#       chain order; the generator enforces all three assumptions

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
CARD_SIZE = BLOCK_SIZE + 2
BLOCK_COUNT = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
INTERIOR_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 2, 3]

BOARD_MIN = 26
BOARD_MAX = 36
OUTER_MARGIN = 1
CARD_GAP = 1
MAX_LAYOUT_TRIES = 400
MAX_POSITION_TRIES = 500
MAX_RETRIES = 500

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}
CONTACT_LABELS = [
    frozenset(label)
    for size in range(1, len(INTERIOR_OFFSETS) + 1)
    for label in itertools.combinations(INTERIOR_OFFSETS, size)
]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _infer_sep(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.flatten() if int(v) != int(bg))
    return int(counts.most_common(1)[0][0])


def _is_card(tile: np.ndarray, sep: int) -> bool:
    if tile.shape != (CARD_SIZE, CARD_SIZE):
        return False
    if not bool((tile[0, :] == sep).all()):
        return False
    if not bool((tile[-1, :] == sep).all()):
        return False
    if not bool((tile[:, 0] == sep).all()):
        return False
    if not bool((tile[:, -1] == sep).all()):
        return False
    return not bool((tile[1:-1, 1:-1] == sep).any())


def _find_card_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate all sep-framed 7x7 cards on the scattered input board."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _infer_sep(g, bg)
    comps = _connected_components(g == sep)
    boxes: list[tuple[int, int, int, int]] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        tile = g[r0 : r1 + 1, c0 : c1 + 1]
        assert _is_card(tile, sep), "separator component is not a valid 7x7 card"
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    boxes.sort()
    return boxes


def _edge_sets(block: np.ndarray, bg: int = BG) -> dict[str, set[int]]:
    mask = block != int(bg)
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _valid_orders(blocks: list[np.ndarray], orient: str, bg: int = BG) -> list[tuple[int, ...]]:
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    good: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            good.append(tuple(int(x) for x in perm))
    return good


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover five scattered sprite cards and reorder their inner 5x5 blocks into the
    only edge-compatible strip.

    Input structure:
    - The board is a larger background field with exactly five isolated 7x7 cards.
    - Each card has a 1-cell sep-colored frame and contains one 5x5 monochrome sprite.
    - There are no board-spanning divider lines; card positions are unrelated and scattered.

    Rule:
    - Detect the five framed cards, crop their 5x5 interiors, and compare foreground offsets
      on the facing edges of every pair of blocks.
    - A horizontal step uses right-edge overlap with left-edge offsets; a vertical step uses
      bottom-edge overlap with top-edge offsets.
    - Exactly one orientation has exactly one 5-block chain. Output that ordered chain as a
      sep-delimited strip in the winning orientation.

    Key insight:
    - The large board only hides where the five blocks are; once the framed 5x5 interiors are
      extracted, the solution is the unique chain encoded by shared edge-offset signatures.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _infer_sep(g, bg)

    blocks: list[np.ndarray] = []
    for r0, c0, r1, c1 in _find_card_bboxes(g):
        tile = g[r0 : r1 + 1, c0 : c1 + 1]
        assert _is_card(tile, sep), "malformed input card"
        blocks.append(tile[1:-1, 1:-1].copy())

    assert len(blocks) == BLOCK_COUNT, f"expected {BLOCK_COUNT} scattered cards"

    orders_h = _valid_orders(blocks, "h", bg)
    orders_v = _valid_orders(blocks, "v", bg)

    if len(orders_h) == 1 and len(orders_v) == 0:
        return _render_strip([blocks[i] for i in orders_h[0]], "h")
    if len(orders_v) == 1 and len(orders_h) == 0:
        return _render_strip([blocks[i] for i in orders_v[0]], "v")
    assert len(orders_h) == 0 or len(orders_h) == 1, "horizontal order is ambiguous"
    assert len(orders_v) == 0 or len(orders_v) == 1, "vertical order is ambiguous"
    assert not (orders_h and orders_v), "both orientations are valid"
    assert orders_h or orders_v, "must find one valid chain"
    raise AssertionError("unreachable")


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


def _normalize_offsets(offsets: frozenset[int] | set[int] | list[int] | tuple[int, ...] | None) -> set[int]:
    if offsets is None:
        return set()
    return {int(v) for v in offsets}


def _base_mask(
    left_offsets: frozenset[int] | set[int] | list[int] | tuple[int, ...] | None,
    right_offsets: frozenset[int] | set[int] | list[int] | tuple[int, ...] | None,
    target_area: int,
) -> np.ndarray | None:
    left = _normalize_offsets(left_offsets)
    right = _normalize_offsets(right_offsets)
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    anchors = sorted(left | right)
    if not anchors:
        return None

    for row in left:
        mask[int(row), 0:3] = True
    for row in right:
        mask[int(row), 2:BLOCK_SIZE] = True
    mask[int(min(anchors)) : int(max(anchors)) + 1, 2] = True

    if int(mask.sum()) < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    while int(mask.sum()) < int(target_area):
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    if set(np.where(mask[:, 0])[0].tolist()) != left:
        return None
    if set(np.where(mask[:, -1])[0].tolist()) != right:
        return None
    if bool(mask[0].any()) or bool(mask[-1].any()):
        return None
    if any(bool(mask[r].all()) for r in range(BLOCK_SIZE)):
        return None
    if any(bool(mask[:, c].all()) for c in range(BLOCK_SIZE)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_block(
    color: int,
    *,
    left: frozenset[int] | set[int] | list[int] | tuple[int, ...] | None = None,
    right: frozenset[int] | set[int] | list[int] | tuple[int, ...] | None = None,
    top: frozenset[int] | set[int] | list[int] | tuple[int, ...] | None = None,
    bottom: frozenset[int] | set[int] | list[int] | tuple[int, ...] | None = None,
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


def _make_card(block: np.ndarray) -> np.ndarray:
    tile = np.full((CARD_SIZE, CARD_SIZE), BG, dtype=int)
    tile[0, :] = SEP
    tile[-1, :] = SEP
    tile[:, 0] = SEP
    tile[:, -1] = SEP
    paste(tile, block, 1, 1)
    return tile


def _positions_overlap(a: tuple[int, int], b: tuple[int, int], pad: int = CARD_GAP) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + CARD_SIZE - 1, ac0 + CARD_SIZE - 1
    br1, bc1 = br0 + CARD_SIZE - 1, bc0 + CARD_SIZE - 1
    return not (
        ar1 + pad < br0 or
        br1 + pad < ar0 or
        ac1 + pad < bc0 or
        bc1 + pad < ac0
    )


def _is_scattered_layout(positions: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    return len(set(rows)) >= 3 and len(set(cols)) >= 3


def _sample_positions(H: int, W: int) -> list[tuple[int, int]] | None:
    low_r = OUTER_MARGIN
    high_r = H - CARD_SIZE - OUTER_MARGIN
    low_c = OUTER_MARGIN
    high_c = W - CARD_SIZE - OUTER_MARGIN
    if low_r > high_r or low_c > high_c:
        return None

    for _ in range(MAX_POSITION_TRIES):
        positions: list[tuple[int, int]] = []
        for _card in range(BLOCK_COUNT):
            placed = False
            for _ in range(MAX_POSITION_TRIES):
                r0 = random.randint(low_r, high_r)
                c0 = random.randint(low_c, high_c)
                pos = (int(r0), int(c0))
                if any(_positions_overlap(pos, prev) for prev in positions):
                    continue
                positions.append(pos)
                placed = True
                break
            if not placed:
                break
        if len(positions) == BLOCK_COUNT and _is_scattered_layout(positions):
            return positions
    return None


def _render_input(blocks: list[np.ndarray]) -> np.ndarray | None:
    perm = list(range(len(blocks)))
    while perm == list(range(len(blocks))):
        random.shuffle(perm)
    cards = [_make_card(blocks[i]) for i in perm]

    for _ in range(MAX_LAYOUT_TRIES):
        H = random.randint(BOARD_MIN, BOARD_MAX)
        W = random.randint(BOARD_MIN, BOARD_MAX)
        positions = _sample_positions(H, W)
        if positions is None:
            continue
        board = np.full((H, W), BG, dtype=int)
        for card, (r0, c0) in zip(cards, positions):
            paste(board, card, r0, c0)
        return board
    return None


def _valid_orders_from_labels(seq: tuple[frozenset[int], ...]) -> list[tuple[int, ...]]:
    left_labels = [set()] + [set(label) for label in seq]
    right_labels = [set(label) for label in seq] + [set()]

    def ok(a: int, b: int) -> bool:
        return bool(right_labels[a] & left_labels[b])

    good: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(BLOCK_COUNT)):
        if all(ok(perm[i], perm[i + 1]) for i in range(BLOCK_COUNT - 1)):
            good.append(tuple(int(x) for x in perm))
    return good


def _good_link_sequences() -> list[tuple[frozenset[int], ...]]:
    good: list[tuple[frozenset[int], ...]] = []
    for seq in itertools.product(CONTACT_LABELS, repeat=BLOCK_COUNT - 1):
        orders = _valid_orders_from_labels(tuple(seq))
        if len(orders) == 1 and orders[0] == tuple(range(BLOCK_COUNT)):
            good.append(tuple(seq))
    assert good, "no valid 5-block link sequences found"
    return good


GOOD_LINK_SEQUENCES = _good_link_sequences()


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(["h", "v"])
    link_seq = random.choice(GOOD_LINK_SEQUENCES)

    lead_kw, trail_kw = ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(BLOCK_COUNT):
        kwargs: dict[str, frozenset[int] | set[int]] = {}
        if i > 0:
            kwargs[lead_kw] = link_seq[i - 1]
        if i < BLOCK_COUNT - 1:
            kwargs[trail_kw] = link_seq[i]
        block = _make_block(random.choice(PALETTE), **kwargs)
        if block is None:
            return None
        blocks.append(block)

    if len(_valid_orders(blocks, orient)) != 1:
        return None
    if len(_valid_orders(blocks, OTHER[orient])) != 0:
        return None

    out = _render_strip(blocks, orient)
    inp = _render_input(blocks)
    if inp is None:
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
