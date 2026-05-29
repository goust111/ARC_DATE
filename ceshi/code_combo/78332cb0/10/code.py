from __future__ import annotations

# Combo variant of 78332cb0 (index 10)
# combo_spec: D6/0+D8/2
# Name: 无标记散布块提取
#
# Design analysis for the combo variant
# -------------------------------------
# spec_like_analysis:
#   fused_rule:
#     - the input still contains 3-4 monochrome 5x5 sprites whose order is scrambled
#     - unlike the source task, the sprites are scattered on a larger bg canvas
#     - unlike both the source task and the framed D6 reference, there are no explicit
#       separator marks around the sprites; only background gaps separate clusters
#   concrete_pixel_trace:
#     - original pipeline:
#       1. detect full-grid sep rows/cols
#       2. slice out exact 5x5 blocks directly from those partitions
#       3. compare edge-contact offsets to recover the unique chain
#     - fused pipeline:
#       1. find 4-connected foreground clusters on the large canvas
#       2. crop each cluster to its tight bbox
#       3. under an H hypothesis, embed each crop back into a 5x5 canvas as either
#          3x5 (interior block) or 3x4 shifted left/right (endpoint); under a V
#          hypothesis, do the analogous 5x3 / 4x3 embedding
#       4. keep the unique orientation+embedding assignment whose edge contacts form a
#          complete chain, then render the original sep-separated output strip
#     - example:
#       * a horizontal interior sprite occupies rows 1..3 and all five columns, so its
#         visible cluster is exactly 3x5
#       * a horizontal endpoint with only a right contact occupies rows 1..3 and cols
#         1..4, so its visible cluster is 3x4 and must be re-embedded with one blank
#         column on the left
#       * the solver does not know that left/right padding in advance, so it enumerates
#         the two endpoint embeddings and accepts only the assignment that yields a
#         unique chain
#   complexity_calibration:
#     - the no-marker variant would be underconstrained if arbitrary source sprites were
#       scattered directly onto bg, because many blank border cells would become
#       invisible and the original 5x5 coordinates could not be recovered reliably
#     - to keep the task visually solvable, the generator restricts the sprite family so
#       each block's tight cluster already reveals most of its hidden 5x5 frame:
#         H blocks crop to 3x4 or 3x5
#         V blocks crop to 4x3 or 5x3
#     - only endpoint blocks need a 2-way left/right or top/bottom embedding choice, so
#       the added ambiguity is small and repeated across samples
#   implicit_assumption_audit:
#     - old _find_sep assumed a non-bg color forms full rows/cols across the board;
#       invalid because the input now has no separator color at all
#     - old _split_blocks assumed partition lines expose exact 5x5 extents; invalid
#       because only foreground clusters are visible on the large canvas
#     - old edge matching still works after recovery, but only if the recovered 5x5
#       canvases preserve the original edge offsets; the new embedding stage restores
#       those offsets before chaining
#     - new solver assumption: each cluster tight bbox must be one of the supported
#       recoverable shapes for exactly one global orientation; the generator enforces
#       this by construction
#   placement_legality:
#     - generator places full hidden 5x5 footprints, not just tight crops
#     - footprints never overlap, never leave the board, and keep at least one cell of
#       bg moat, so clusters remain disconnected and extraction is stable
#   visual_observability:
#     - humans can literally see several isolated monochrome clusters on the input and a
#       sep-separated ordered strip on the output
#     - the fusion is observable because the input contains no separator marks at all,
#       yet the output restores whole 5x5 sprites whose ordering depends on edge-contact
#       continuation
#   cleanliness:
#     - there is no overdraw and only two visual roles exist in the input: bg and sprite
#       pixels
#     - the scattered layout is rejected if it degenerates into a simple strip or clean
#       2x2 arrangement, preserving the intended "search first, then chain" behavior

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 6
FG_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]

BOARD_MIN = 18
BOARD_MAX = 28
OUTER_MARGIN = 1
BLOCK_GAP = 1
MAX_LAYOUT_TRIES = 300
MAX_POSITION_TRIES = 400
MAX_RETRIES = 400

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _foreground_crops(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    """Extract tight crops of each isolated foreground cluster."""
    comps = _connected_components(grid != bg)
    crops: list[np.ndarray] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        crops.append(grid[r0 : r1 + 1, c0 : c1 + 1].copy())
    return crops


def _candidate_blocks(crop: np.ndarray, orient: str, bg: int) -> list[np.ndarray]:
    """Recover possible 5x5 blocks from a tight foreground crop."""
    h, w = crop.shape
    out: list[np.ndarray] = []
    if orient == "h":
        if h != 3 or w not in (4, 5):
            return out
        if w == 5:
            block = np.full((BLOCK_SIZE, BLOCK_SIZE), bg, dtype=int)
            block[1:4, :] = crop
            out.append(block)
            return out
        for c0 in (0, 1):
            block = np.full((BLOCK_SIZE, BLOCK_SIZE), bg, dtype=int)
            block[1:4, c0 : c0 + 4] = crop
            out.append(block)
        return out
    if w != 3 or h not in (4, 5):
        return out
    if h == 5:
        block = np.full((BLOCK_SIZE, BLOCK_SIZE), bg, dtype=int)
        block[:, 1:4] = crop
        out.append(block)
        return out
    for r0 in (0, 1):
        block = np.full((BLOCK_SIZE, BLOCK_SIZE), bg, dtype=int)
        block[r0 : r0 + 4, 1:4] = crop
        out.append(block)
    return out


def _edge_sets(block: np.ndarray, bg: int) -> dict[str, set[int]]:
    mask = block != bg
    return {
        "top": set(np.where(mask[0])[0].tolist()),
        "bot": set(np.where(mask[-1])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _count_orders(blocks: list[np.ndarray], orient: str, bg: int) -> int:
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _find_chain(blocks: list[np.ndarray], orient: str, bg: int) -> tuple[int, ...] | None:
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    idx = list(range(len(blocks)))
    for perm in itertools.permutations(idx):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
            return tuple(int(v) for v in perm)
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


def _solve_from_crops(crops: list[np.ndarray], bg: int) -> np.ndarray:
    outputs: dict[bytes, np.ndarray] = {}
    for orient in ("h", "v"):
        candidate_lists = [_candidate_blocks(crop, orient, bg) for crop in crops]
        if any(len(cands) == 0 for cands in candidate_lists):
            continue
        for choice in itertools.product(*candidate_lists):
            blocks = [np.array(block, dtype=int) for block in choice]
            if _count_orders(blocks, orient, bg) != 1:
                continue
            if _count_orders(blocks, OTHER[orient], bg) != 0:
                continue
            chain = _find_chain(blocks, orient, bg)
            assert chain is not None
            out = _render_strip([blocks[i] for i in chain], orient)
            outputs[out.tobytes()] = out
    assert len(outputs) == 1, "input does not admit a unique recovered chain"
    return next(iter(outputs.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover scattered unmarked sprite clusters, reconstruct their hidden 5x5 canvases,
    and order them into the only strip whose edge contacts link all blocks.

    Input structure:
    - bg is the only filler color; there are no separator lines or frames in the input.
    - Each sprite is one isolated monochrome foreground cluster scattered on a larger board.
    - The visible cluster is a tight crop of an underlying 5x5 block, so some blank border
      cells of the original block are implicit rather than drawn.

    Rule:
    - First identify the foreground clusters by spatial separation.
    - Re-embed each cluster into the possible 5x5 blocks consistent with a horizontal or
      vertical chain hypothesis.
    - Keep the unique orientation and ordering where consecutive blocks share at least one
      foreground offset on their facing edges, then render that chain as a sep-separated strip.

    Key insight:
    - The missing separators force reconstruction from cluster geometry first; once the hidden
      5x5 canvases are restored, the original edge-overlap chaining rule resolves the order.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crops = _foreground_crops(g, bg)
    assert len(crops) in (3, 4), "expected 3 or 4 isolated foreground clusters"
    return _solve_from_crops(crops, bg)


def _interior_frontier(mask: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    frontier: list[tuple[int, int]] = []
    for r in range(BLOCK_SIZE):
        for c in range(BLOCK_SIZE):
            if not allowed[r, c] or mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c]) or
                (r + 1 < BLOCK_SIZE and mask[r + 1, c]) or
                (c > 0 and mask[r, c - 1]) or
                (c + 1 < BLOCK_SIZE and mask[r, c + 1])
            ):
                frontier.append((int(r), int(c)))
    return frontier


def _seed_h_mask(left_offset: int | None, right_offset: int | None) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    allowed = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    c0, c1 = 0, 4
    if left_offset is None:
        c0 = 1
    if right_offset is None:
        c1 = 3
    allowed[1:4, c0 : c1 + 1] = True

    # Force a 3-row visible crop so the solver can infer the hidden blank rows.
    mask[1:4, 2] = True

    if left_offset is not None:
        mask[left_offset, 0:3] = True
    else:
        mask[2, 1] = True

    if right_offset is not None:
        mask[right_offset, 2:5] = True
    else:
        mask[2, 3] = True

    mask &= allowed
    return mask, allowed


def _make_h_block(color: int, left_offset: int | None, right_offset: int | None) -> np.ndarray | None:
    mask, allowed = _seed_h_mask(left_offset, right_offset)
    seed_area = int(mask.sum())
    if seed_area > FG_AREA_MAX:
        return None

    target_area = random.randint(max(FG_AREA_MIN, seed_area), FG_AREA_MAX)
    while int(mask.sum()) < target_area:
        frontier = _interior_frontier(mask, allowed)
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
    if sorted(np.where(mask.any(axis=1))[0].tolist()) != [1, 2, 3]:
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _make_block(
    color: int,
    *,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
) -> np.ndarray | None:
    if top is not None or bottom is not None:
        block = _make_h_block(int(color), bottom, top)
        if block is None:
            return None
        return np.rot90(block, 1)
    return _make_h_block(int(color), left, right)


def _positions_overlap(a: tuple[int, int], b: tuple[int, int], pad: int = BLOCK_GAP) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + BLOCK_SIZE - 1, ac0 + BLOCK_SIZE - 1
    br1, bc1 = br0 + BLOCK_SIZE - 1, bc0 + BLOCK_SIZE - 1
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


def _sample_positions(n: int, h: int, w: int) -> list[tuple[int, int]] | None:
    low_r = OUTER_MARGIN
    high_r = h - BLOCK_SIZE - OUTER_MARGIN
    low_c = OUTER_MARGIN
    high_c = w - BLOCK_SIZE - OUTER_MARGIN
    if low_r > high_r or low_c > high_c:
        return None

    for _ in range(MAX_POSITION_TRIES):
        positions: list[tuple[int, int]] = []
        for _block in range(n):
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

    for _ in range(MAX_LAYOUT_TRIES):
        h = random.randint(BOARD_MIN, BOARD_MAX)
        w = random.randint(BOARD_MIN, BOARD_MAX)
        positions = _sample_positions(n, h, w)
        if positions is None:
            continue
        board = np.full((h, w), BG, dtype=int)
        for block, (r0, c0) in zip(scrambled, positions):
            paste(board, block, r0, c0)
        return board
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    lead_kw, trail_kw = ORIENT_KW[orient]
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

    if _count_orders(blocks, orient, BG) != 1:
        return None
    if _count_orders(blocks, OTHER[orient], BG) != 0:
        return None

    out = _render_strip(blocks, orient)
    inp = _render_input(blocks)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
