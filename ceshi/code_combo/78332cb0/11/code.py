# Combo variant of 78332cb0 (index 11)
# combo_spec: D7/0+D10/0+D11/1
# Name: 多线索五块排序

from __future__ import annotations

from collections import Counter
import itertools
import random
import numpy as np

from arc_utils import get_bg, is_contiguous, paste


# Design analysis for the fused variant:
# - Original task logic: edge-overlap alone determined both orientation and order, and
#   endpoints were identifiable because one outer chain edge was empty. Here that
#   assumption is intentionally broken: all 5 blocks have a foreground pixel on both
#   chain-direction sides, including the two endpoints.
# - New visible cues:
#   1. Fixed block count is 5, increasing the search space.
#   2. The endpoint blocks carry internal marker pixels: color 2 for the start block,
#      color 3 for the end block. These markers are visible but do not touch block edges.
#   3. The two edges perpendicular to the chain carry exactly k foreground pixels on the
#      kth block. For horizontal chains the count is on top+bottom; for vertical chains
#      the count is on left+right. The templates use disjoint offset sets so the wrong
#      orientation does not accidentally create edge matches.
# - Small concrete trace:
#   Suppose the true output is a horizontal chain. The input is a vertical strip of the
#   same 5 blocks in scrambled order. Every block still has left/right edge contacts, so
#   the old "empty leading/trailing edge" heuristic is gone. Block #4 is recognizable
#   because its top+bottom edges contain exactly four foreground pixels; block #1 also
#   contains the red marker (2), and block #5 contains the green marker (3). The solver
#   first detects that only the horizontal candidate has perpendicular-edge counts
#   {1,2,3,4,5}, then sorts by those counts, checks that the markers sit on the first
#   and last block, and finally verifies that consecutive blocks really connect by
#   left/right overlap. This keeps the three clues coupled instead of independent.
# - Complexity calibration: the count encoding is deliberately low-arity (1..5 only) and
#   repeated across every sample, so humans can see it. The contact offsets are drawn from
#   a small repeated alphabet, which prevents edge matching from degenerating into a trivial
#   unique permutation while still leaving a single correct orientation.
# - Placement legality / cleanliness: the generator uses output-first construction, splits
#   blocks with full separator rows/cols, keeps every block inside a 5x5 cell, forbids full
#   monochrome rows/cols inside blocks (so separators stay visually unique), and validates
#   that the final input is solvable by the new main().


BG = 7
SEP = 6
START_MARKER = 2
END_MARKER = 3
BASE_PALETTE = [1, 4, 5, 8, 9]

BLOCK_SIZE = 5
NUM_BLOCKS = 5
FG_AREA_MIN = 8
FG_AREA_MAX = 12
MAX_RETRIES = 400

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}

# Rank k is encoded by total pixels on the edges perpendicular to the chain.
# The chosen slots stay away from corners so every required edge pixel can connect to the
# 3x3 interior through ordinary 4-neighbor paths.
_COUNT_TOP = {
    1: {1},
    2: {1, 2},
    3: {1, 2, 3},
    4: {1, 2, 3},
    5: {1, 2, 3},
}
_COUNT_BOT = {
    1: set(),
    2: set(),
    3: set(),
    4: {1},
    5: {1, 2},
}

_CONTACT_PATTERN = [1, 3, 1, 3]
_END_DUMMY = 2
_MASK_CACHE: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]], list[np.ndarray]] = {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder five 5x5 blocks into the marked chain using edge counts, endpoint markers,
    and edge-contact consistency.

    Input structure:
    - Background is the most common color; separator lines are full rows or columns of a
      second color.
    - The grid contains exactly five 5x5 sprite blocks in a scrambled strip, separated by
      single separator lines.
    - Each block may contain one internal marker pixel: color 2 marks the chain start,
      color 3 marks the chain end.

    Rule:
    - Find whether the chain is horizontal or vertical by checking which candidate treats
      the perpendicular-edge foreground counts as the sequence 1,2,3,4,5.
    - Order the blocks by that count, require the start marker on the first block and the
      end marker on the last block, and verify that consecutive blocks really connect by
      sharing a foreground offset on their facing chain edges.
    - Output the ordered blocks as a single strip in the detected chain orientation, with
      separator lines preserved between neighbors.

    Key insight:
    - Edge matching still validates local adjacency, but global order is encoded redundantly:
      the perpendicular-edge counts reveal positions 1..5, and the two marker colors tell
      which ranked block is the head and which is the tail.
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
        if rows and not cols:
            return _split_by_rows(g)
        if cols and not rows:
            return _split_by_cols(g)
        if rows and cols:
            out: list[np.ndarray] = []
            row_stops = [-1] + rows + [int(g.shape[0])]
            col_stops = [-1] + cols + [int(g.shape[1])]
            for i in range(len(row_stops) - 1):
                r0 = row_stops[i] + 1
                r1 = row_stops[i + 1]
                if r1 <= r0:
                    continue
                for j in range(len(col_stops) - 1):
                    c0 = col_stops[j] + 1
                    c1 = col_stops[j + 1]
                    if c1 <= c0:
                        continue
                    cell = g[r0:r1, c0:c1]
                    if cell.size and np.any(cell != bg):
                        out.append(cell.copy())
            return out
        return []

    def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
        mask = block != int(bg)
        return {
            "top": set(np.where(mask[0])[0].tolist()),
            "bot": set(np.where(mask[-1])[0].tolist()),
            "left": set(np.where(mask[:, 0])[0].tolist()),
            "right": set(np.where(mask[:, -1])[0].tolist()),
        }

    def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
        n = len(blocks)
        if orient == "h":
            out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), int(bg), dtype=int)
            cc = 0
            for i, block in enumerate(blocks):
                out[:, cc : cc + BLOCK_SIZE] = block
                cc += BLOCK_SIZE
                if i < n - 1:
                    out[:, cc] = int(sep)
                    cc += 1
            return out
        out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), int(bg), dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            out[rr : rr + BLOCK_SIZE, :] = block
            rr += BLOCK_SIZE
            if i < n - 1:
                out[rr, :] = int(sep)
                rr += 1
        return out

    blocks = _split_blocks()
    assert len(blocks) == NUM_BLOCKS, f"expected {NUM_BLOCKS} blocks, got {len(blocks)}"
    assert all(block.shape == (BLOCK_SIZE, BLOCK_SIZE) for block in blocks)

    edges = [_edge_sets(block) for block in blocks]

    def _rank_count(i: int, orient: str) -> int:
        if orient == "h":
            return len(edges[i]["top"]) + len(edges[i]["bot"])
        return len(edges[i]["left"]) + len(edges[i]["right"])

    def _has_marker(block: np.ndarray, color: int) -> bool:
        return bool(np.any(block == int(color)))

    def _valid_candidate(orient: str):
        counts = [_rank_count(i, orient) for i in range(len(blocks))]
        if sorted(counts) != list(range(1, NUM_BLOCKS + 1)):
            return None
        order = sorted(range(len(blocks)), key=lambda i: counts[i])
        if [counts[i] for i in order] != list(range(1, NUM_BLOCKS + 1)):
            return None
        if sum(_has_marker(block, START_MARKER) for block in blocks) != 1:
            return None
        if sum(_has_marker(block, END_MARKER) for block in blocks) != 1:
            return None
        if not _has_marker(blocks[order[0]], START_MARKER):
            return None
        if not _has_marker(blocks[order[-1]], END_MARKER):
            return None
        lead, trail = _EDGE_PAIRS[orient]
        for a, b in zip(order, order[1:]):
            if not (edges[a][lead] & edges[b][trail]):
                return None
        return orient, order

    candidates = [cand for orient in ("h", "v") if (cand := _valid_candidate(orient)) is not None]
    assert len(candidates) == 1, f"expected one valid orientation, got {len(candidates)}"
    orient, order = candidates[0]
    return _render_strip([blocks[i] for i in order], orient)


def _boundary_requirements(orient: str, lead_offset: int, trail_offset: int, rank: int) -> dict[str, set[int]]:
    top = set(_COUNT_TOP[rank])
    bot = set(_COUNT_BOT[rank])
    if orient == "h":
        return {
            "top": top,
            "bot": bot,
            "left": {int(lead_offset)},
            "right": {int(trail_offset)},
        }
    return {
        "top": {int(lead_offset)},
        "bot": {int(trail_offset)},
        "left": top,
        "right": bot,
    }


def _boundary_cells(req: dict[str, set[int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    cells |= {(0, int(c)) for c in req["top"]}
    cells |= {(BLOCK_SIZE - 1, int(c)) for c in req["bot"]}
    cells |= {(int(r), 0) for r in req["left"]}
    cells |= {(int(r), BLOCK_SIZE - 1) for r in req["right"]}
    return cells


def _validate_mask(mask: np.ndarray, req: dict[str, set[int]]) -> bool:
    top = set(np.where(mask[0])[0].tolist())
    bot = set(np.where(mask[-1])[0].tolist())
    left = set(np.where(mask[:, 0])[0].tolist())
    right = set(np.where(mask[:, -1])[0].tolist())
    if top != req["top"] or bot != req["bot"] or left != req["left"] or right != req["right"]:
        return False
    if any(mask[r].all() for r in range(BLOCK_SIZE)):
        return False
    if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
        return False
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return False
    return True


def _req_key(req: dict[str, set[int]]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return tuple(sorted(req["top"])), tuple(sorted(req["bot"])), tuple(sorted(req["left"])), tuple(sorted(req["right"]))


def _mask_candidates(req: dict[str, set[int]]) -> list[np.ndarray]:
    key = _req_key(req)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    required = _boundary_cells(req)
    cands: list[np.ndarray] = []
    interior = [(r, c) for r in range(1, BLOCK_SIZE - 1) for c in range(1, BLOCK_SIZE - 1)]

    for bits in range(1 << len(interior)):
        mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
        for r, c in required:
            mask[r, c] = True
        for i, (r, c) in enumerate(interior):
            if bits & (1 << i):
                mask[r, c] = True
        area = int(mask.sum())
        if not (FG_AREA_MIN <= area <= FG_AREA_MAX):
            continue
        if not np.any(mask[1:4, 1:4]):
            continue
        if _validate_mask(mask, req):
            cands.append(mask)

    _MASK_CACHE[key] = cands
    return cands


def _make_block(color: int, orient: str, lead_offset: int, trail_offset: int, rank: int, marker: int | None) -> np.ndarray | None:
    req = _boundary_requirements(orient, lead_offset, trail_offset, rank)
    cands = _mask_candidates(req)
    if not cands:
        return None
    mask = random.choice(cands)
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    if marker is not None:
        inner = [(r, c) for r in range(1, 4) for c in range(1, 4) if mask[r, c]]
        if not inner:
            return None
        mr, mc = random.choice(inner)
        block[mr, mc] = int(marker)
    return block


def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    n = len(blocks)
    if orient == "h":
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), BG, dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * (BLOCK_SIZE + 1))
            if i < n - 1:
                out[:, i * (BLOCK_SIZE + 1) + BLOCK_SIZE] = SEP
        return out
    out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), BG, dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * (BLOCK_SIZE + 1), 0)
        if i < n - 1:
            out[i * (BLOCK_SIZE + 1) + BLOCK_SIZE, :] = SEP
    return out


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    perm = list(range(len(blocks)))
    while perm == list(range(len(blocks))):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]
    return _render_strip(scrambled, _OTHER[output_orient])


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    edges = []
    for block in blocks:
        mask = block != BG
        edges.append(
            {
                "top": set(np.where(mask[0])[0].tolist()),
                "bot": set(np.where(mask[-1])[0].tolist()),
                "left": set(np.where(mask[:, 0])[0].tolist()),
                "right": set(np.where(mask[:, -1])[0].tolist()),
            }
        )
    lead, trail = _EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(edges[a][lead] & edges[b][trail])

    total = 0
    idx = list(range(len(blocks)))
    for perm in itertools.permutations(idx):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _rank_vector(blocks: list[np.ndarray], orient: str) -> list[int]:
    out = []
    for block in blocks:
        mask = block != BG
        if orient == "h":
            out.append(int(mask[0].sum() + mask[-1].sum()))
        else:
            out.append(int(mask[:, 0].sum() + mask[:, -1].sum()))
    return out


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(["h", "v"])
    shared = list(_CONTACT_PATTERN)
    start_dummy = _END_DUMMY
    end_dummy = _END_DUMMY

    blocks: list[np.ndarray] = []
    for rank in range(1, NUM_BLOCKS + 1):
        lead = start_dummy if rank == 1 else shared[rank - 2]
        trail = end_dummy if rank == NUM_BLOCKS else shared[rank - 1]
        marker = START_MARKER if rank == 1 else END_MARKER if rank == NUM_BLOCKS else None
        block = _make_block(
            color=random.choice(BASE_PALETTE),
            orient=orient,
            lead_offset=lead,
            trail_offset=trail,
            rank=rank,
            marker=marker,
        )
        if block is None:
            return None
        blocks.append(block)

    if _rank_vector(blocks, orient) != [1, 2, 3, 4, 5]:
        return None
    # Edge matching remains present, but the small repeated contact alphabet should leave
    # multiple admissible permutations if counts/markers are ignored.
    if _count_orders(blocks, orient) <= 1:
        return None

    out = _render_strip(blocks, orient)
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
