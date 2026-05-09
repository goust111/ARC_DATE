from __future__ import annotations

# Variant of 78332cb0
# Dimension: D6 — 输入布局 (grid_shape)
# Change: 4块用2x2网格或strip布局，3块用strip布局；输入块顺序与链序不同；strip方向与输出方向不同... → 随机散布在大网格中
#
# Design analysis for the D6 variant
# ----------------------------------
# spec_like_analysis:
#   original_objects:
#     - 3 or 4 monochrome 5x5 sprites on bg=7
#     - valid chain is determined only by shared foreground offsets on facing edges
#     - original input exposes 5x5 block boundaries through full-row/full-column sep lines
#   d6_change:
#     - input is no longer a strip or a 2x2 partitioned board
#     - sprites are scattered across a larger canvas and must be located before matching
#   necessary_supporting_adjustment:
#     - if the 5x5 sprites were pasted directly onto the global bg, many blank cells of each
#       sprite would become invisible and the 5x5 extent would not be recoverable from a few
#       human examples
#     - therefore each sprite is packaged as a 7x7 "card": a 1-cell sep-colored frame around
#       the original 5x5 block; solver extracts the inner 5x5 region and then runs the same
#       edge-matching logic as the source task
#   pipeline_trace:
#     - original: detect full separator rows/cols -> slice fixed 5x5 blocks -> try H/V chains
#     - variant: detect sep-colored rectangular card frames anywhere in the large board ->
#       crop each card interior to recover 5x5 blocks -> try H/V chains
#     - unchanged stage: once 5x5 blocks are recovered, chain finding and strip rendering are
#       identical in spirit to the original task
#   complexity_calibration:
#     - only 3-4 cards are present, keeping object search manageable
#     - cards are separated by at least one bg cell, so detection is robust and visually clean
#     - generator rejects layouts that collapse back to the old strip or 2x2 arrangements
#   implicit_assumption_audit:
#     - old _find_sep assumed full grid-wide sep lines; invalid after D6, replaced by
#       "most common non-bg color" because every input card contributes a large sep frame
#     - old _split_blocks assumed partition lines span the whole grid; invalid after D6,
#       replaced by connected sep-frame extraction
#     - chain logic still assumes recovered blocks are exactly the original 5x5 sprites; the
#       generator guarantees this by placing untouched 5x5 blocks inside the card interiors
#   placement_legality:
#     - cards are 7x7, never overlap, never touch, and stay within a larger bg canvas
#     - output remains the original 1xN strip with sep dividers
#   visual_observability:
#     - humans can see several isolated framed cards on the input and a reordered strip on the
#       output; the "find the scattered cards first" behavior is directly visible
#   cleanliness:
#     - no overlapping symbols, no hidden color coding, and cards are separated by bg moats

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
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]
FILLER_ROWS = [1, 3]

BOARD_MIN = 20
BOARD_MAX = 30
OUTER_MARGIN = 1
CARD_GAP = 1
MAX_LAYOUT_TRIES = 300
MAX_POSITION_TRIES = 400
MAX_RETRIES = 400

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
_ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _most_common(a: np.ndarray) -> int:
    return int(Counter(map(int, a.flatten())).most_common(1)[0][0])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of a boolean mask."""
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
    """Locate 7x7 framed cards in the scattered input."""
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
        assert _is_card(tile, sep), "separator component is not a valid card frame"
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    boxes.sort()
    return boxes


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    mask = block != _most_common(block)
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _find_chain(blocks: list[np.ndarray], orient: str):
    contacts = [_edge_sets(block) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    idx = list(range(len(blocks)))
    for perm in itertools.permutations(idx):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
            return perm
    return None


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
    Recover scattered framed cards, extract their 5x5 sprites, and reorder them into
    the unique valid strip determined by edge-contact overlap.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _infer_sep(g, bg)

    blocks: list[np.ndarray] = []
    for r0, c0, r1, c1 in _find_card_bboxes(g):
        tile = g[r0 : r1 + 1, c0 : c1 + 1]
        assert _is_card(tile, sep), "malformed input card"
        blocks.append(tile[1:-1, 1:-1].copy())

    assert len(blocks) in (3, 4), "expected 3 or 4 scattered cards"

    chain_h = _find_chain(blocks, "h")
    chain_v = _find_chain(blocks, "v")

    if chain_h is not None and chain_v is None:
        return _render_strip([blocks[i] for i in chain_h], "h")
    if chain_v is not None and chain_h is None:
        return _render_strip([blocks[i] for i in chain_v], "v")
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


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
    rs, cs = np.where(has_neighbor & ~interior)
    return [(int(r) + 1, int(c) + 1) for r, c in zip(rs, cs)]


def _base_mask(left_offset: int | None, right_offset: int | None, target_area: int) -> np.ndarray | None:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2):max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:BLOCK_SIZE] = True
        mask[min(right_offset, 2):max(right_offset, 2) + 1, 2] = True

    if int(mask.sum()) < FG_AREA_MIN:
        mask[random.choice(FILLER_ROWS), 2] = True

    while int(mask.sum()) < int(target_area):
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    if left_offset is None:
        if bool(mask[:, 0].any()):
            return None
    elif np.where(mask[:, 0])[0].tolist() != [left_offset]:
        return None

    if right_offset is None:
        if bool(mask[:, -1].any()):
            return None
    elif np.where(mask[:, -1])[0].tolist() != [right_offset]:
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


def _looks_like_old_layout(positions: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    if len(set(rows)) == 1 or len(set(cols)) == 1:
        return True
    if len(positions) == 4 and len(set(rows)) == 2 and len(set(cols)) == 2:
        if sorted(Counter(rows).values()) == [2, 2] and sorted(Counter(cols).values()) == [2, 2]:
            return True
    return False


def _sample_positions(n: int, H: int, W: int) -> list[tuple[int, int]] | None:
    low_r = OUTER_MARGIN
    high_r = H - CARD_SIZE - OUTER_MARGIN
    low_c = OUTER_MARGIN
    high_c = W - CARD_SIZE - OUTER_MARGIN
    if low_r > high_r or low_c > high_c:
        return None

    for _ in range(MAX_POSITION_TRIES):
        positions: list[tuple[int, int]] = []
        for _card in range(n):
            placed = False
            for _ in range(MAX_POSITION_TRIES):
                r0 = random.randint(low_r, high_r)
                c0 = random.randint(low_c, high_c)
                if any(_positions_overlap((r0, c0), prev) for prev in positions):
                    continue
                positions.append((int(r0), int(c0)))
                placed = True
                break
            if not placed:
                break
        if len(positions) == n and not _looks_like_old_layout(positions):
            return positions
    return None


def _render_input(blocks: list[np.ndarray]) -> np.ndarray | None:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]
    cards = [_make_card(block) for block in scrambled]

    for _ in range(MAX_LAYOUT_TRIES):
        H = random.randint(BOARD_MIN, BOARD_MAX)
        W = random.randint(BOARD_MIN, BOARD_MAX)
        positions = _sample_positions(n, H, W)
        if positions is None:
            continue
        board = np.full((H, W), BG, dtype=int)
        for card, (r0, c0) in zip(cards, positions):
            paste(board, card, r0, c0)
        return board
    return None


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
        block = _make_block(random.choice(PALETTE), **kw)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, orient) != 1 or _count_orders(blocks, _OTHER[orient]) != 0:
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
