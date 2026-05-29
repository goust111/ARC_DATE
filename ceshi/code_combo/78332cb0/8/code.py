# Combo variant of 78332cb0 (index 8)
# combo_spec: D1/3+D18/2
# Name: 计数匹配与渐进偏移

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    The original task linked blocks when facing edge pixels overlapped at the same row/col.
#    Here each chain-direction edge is summarized by two visible attributes:
#    - edge fg count: 1 or 2
#    - edge placement class: top/middle/bottom, encoded by a small catalog of offsets
#      count 1 -> {1}, {2}, {3}
#      count 2 -> {1,2}, {1,3}, {2,3}
#    Example horizontal chain:
#      B1: R=(1, top), B2: L=(1, bottom), R=(1, middle),
#      B3: L=(1, top),    R=(1, bottom), B4: L=(1, middle)
#    Count-only adjacency allows both B1-B2-B3-B4 and B1-B3-B2-B4.
#    The fused rule keeps only the order whose outgoing placements increase 1<2<3.
#    This differs from the original overlap rule, which would prefer the swapped middle pair
#    because equal-position single pixels line up there.
# b) Complexity calibration:
#    We keep exactly four 5x5 blocks so the three placement levels {1,2,3} all appear in
#    every sample, and there is always a genuine middle-block ambiguity before applying the
#    increasing-placement rule. The solver understands counts {1,2}, but the generator fixes
#    the actual chain count to 1: on a 5x5 edge with usable offsets restricted to {1,2,3},
#    every two-pixel pattern intersects every other two-pixel pattern, which would let the
#    sample collapse back toward the original overlap rule. Fixing count=1 preserves the
#    count-vs-position distinction in every generated instance.
# c) Assumption audit:
#    The old solver assumed "intersection of facing offsets > 0". That assumption is removed.
#    The new solver assumes only:
#    - chain-direction edge patterns come from the small visible catalog above
#    - perpendicular edges are empty
#    - there is exactly one start block (no leading edge) and one end block (no trailing edge)
#    The generator constructs inputs that satisfy those assumptions explicitly.
# d) Placement legality:
#    Block masks are not hand-drawn ad hoc. For each requested pair of edge signatures we
#    exhaustively search the legal 5x5 interior cells and cache a smallest connected mask
#    with 4..10 fg pixels, so shapes stay in bounds, non-overlapping, and valid by construction.
# e/f/g) Observability and cleanliness:
#    The decisive information is visible on the boundary itself: humans can count edge pixels
#    and see whether the active edge cluster sits high/mid/low. No color ordering, hidden
#    numeric ids, or cluttered overlays are used. Scrambling still happens only at block level,
#    so the scene stays tidy.


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
NUM_BLOCKS = 4
GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 300

OTHER = {"h": "v", "v": "h"}
MATCH_EDGES = {"h": ("right", "left"), "v": ("bot", "top")}
BLOCK_ENDS = {"h": ("left", "right"), "v": ("top", "bot")}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}

# Visible boundary-pattern catalog.
EDGE_ROWS = {
    1: {1: (1,), 2: (2,), 3: (3,)},
    2: {1: (1, 2), 2: (1, 3), 3: (2, 3)},
}
ROWS_TO_SIG = {
    tuple(rows): (count, place)
    for count, by_place in EDGE_ROWS.items()
    for place, rows in by_place.items()
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 sprite blocks into the only strip whose edge counts match and whose edge placements rise step by step.

    Input structure:
    - Background cells fill the canvas and one separator color forms full divider rows/columns.
    - Four 5x5 single-color sprite blocks are placed in a strip or a 2x2 arrangement, scrambled out of chain order.

    Rule:
    - Two neighboring blocks are compatible when the predecessor's trailing edge and the successor's leading edge contain the same number of foreground pixels, regardless of exact overlap.
    - Among count-compatible orders, keep the one whose successive trailing-edge placement classes increase monotonically (top -> middle -> bottom).
    - Output the recovered chain as a 1x4 strip in its true orientation, separated by one divider line each time.

    Key insight:
    - Count equality tells which edges may pair, while the progressive shift of the active edge cluster breaks the remaining ambiguity and fixes the unique order.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _find_sep(a: np.ndarray) -> int:
        cand: list[int] = []
        for r in range(a.shape[0]):
            if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != bg:
                cand.append(int(a[r, 0]))
        for c in range(a.shape[1]):
            if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != bg:
                cand.append(int(a[0, c]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(a.shape[0]) if bool((a[r, :] == sep).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(a.shape[1]) if bool((a[:, c] == sep).all())]

    def _split_by_rows(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        segs: list[np.ndarray] = []
        start = 0
        for r in rows + [int(a.shape[0])]:
            if r > start:
                segs.append(a[start:r, :].copy())
            start = int(r + 1)
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if c > start:
                segs.append(a[:, start:c].copy())
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
                g[:r, c + 1 :].copy(),
                g[r + 1 :, :c].copy(),
                g[r + 1 :, c + 1 :].copy(),
            ]
        if rows:
            return _split_by_rows(g)
        if cols:
            return _split_by_cols(g)
        return []

    def _render_strip(blocks: list[np.ndarray], axis: int) -> np.ndarray:
        n = len(blocks)
        if axis == 0:
            out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), bg, dtype=int)
            rr = 0
            for i, block in enumerate(blocks):
                out[rr : rr + BLOCK_SIZE, :] = block
                rr += BLOCK_SIZE
                if i < n - 1:
                    out[rr, :] = sep
                    rr += 1
            return out
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), bg, dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, cc : cc + BLOCK_SIZE] = block
            cc += BLOCK_SIZE
            if i < n - 1:
                out[:, cc] = sep
                cc += 1
        return out

    def _edge_sig(block: np.ndarray, side: str) -> tuple[int, int | None]:
        mask = block != bg
        if side == "top":
            rows = tuple(np.where(mask[0, :])[0].tolist())
        elif side == "bot":
            rows = tuple(np.where(mask[-1, :])[0].tolist())
        elif side == "left":
            rows = tuple(np.where(mask[:, 0])[0].tolist())
        else:
            rows = tuple(np.where(mask[:, -1])[0].tolist())
        if not rows:
            return 0, None
        assert rows in ROWS_TO_SIG, f"unexpected edge pattern {rows}"
        return ROWS_TO_SIG[rows]

    def _block_sigs(block: np.ndarray) -> dict[str, tuple[int, int | None]]:
        return {
            "top": _edge_sig(block, "top"),
            "bot": _edge_sig(block, "bot"),
            "left": _edge_sig(block, "left"),
            "right": _edge_sig(block, "right"),
        }

    def _valid_perms(blocks: list[np.ndarray], orient: str) -> list[tuple[int, ...]]:
        sigs = [_block_sigs(block) for block in blocks]
        leading_edge, trailing_edge = BLOCK_ENDS[orient]
        prev_edge, next_edge = MATCH_EDGES[orient]
        orth = ("top", "bot") if orient == "h" else ("left", "right")
        idx = list(range(len(blocks)))
        valid: list[tuple[int, ...]] = []

        for perm in itertools.permutations(idx):
            first = sigs[perm[0]][leading_edge]
            last = sigs[perm[-1]][trailing_edge]
            if first[0] != 0 or last[0] != 0:
                continue
            if any(sigs[p][side][0] != 0 for p in perm for side in orth):
                continue

            ok = True
            placements: list[int] = []
            for i in range(len(perm) - 1):
                prev_sig = sigs[perm[i]][prev_edge]
                next_sig = sigs[perm[i + 1]][next_edge]
                if prev_sig[0] == 0 or next_sig[0] == 0 or prev_sig[0] != next_sig[0]:
                    ok = False
                    break
                placements.append(int(prev_sig[1]))
            if ok and all(placements[i] < placements[i + 1] for i in range(len(placements) - 1)):
                valid.append(tuple(int(x) for x in perm))
        return valid

    blocks = _split_blocks()
    assert len(blocks) == NUM_BLOCKS, "expected exactly four 5x5 blocks"

    chain_h = _valid_perms(blocks, "h")
    chain_v = _valid_perms(blocks, "v")

    if len(chain_h) == 1 and not chain_v:
        return _render_strip([blocks[i] for i in chain_h[0]], axis=1)
    if len(chain_v) == 1 and not chain_h:
        return _render_strip([blocks[i] for i in chain_v[0]], axis=0)
    assert not chain_h and not chain_v, "ambiguous orientation/order"
    raise AssertionError("must find exactly one valid chain")


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    mask = block != BG
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _edge_sig_from_rows(rows: tuple[int, ...]) -> tuple[int, int | None]:
    if not rows:
        return 0, None
    return ROWS_TO_SIG[rows]


def _edge_sigs_for_block(block: np.ndarray) -> dict[str, tuple[int, int | None]]:
    edges = _edge_sets(block)
    return {side: _edge_sig_from_rows(tuple(sorted(vals))) for side, vals in edges.items()}


def _count_valid_orders(blocks: list[np.ndarray], orient: str, require_increasing: bool) -> int:
    sigs = [_edge_sigs_for_block(block) for block in blocks]
    leading_edge, trailing_edge = BLOCK_ENDS[orient]
    prev_edge, next_edge = MATCH_EDGES[orient]
    orth = ("top", "bot") if orient == "h" else ("left", "right")
    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if sigs[perm[0]][leading_edge][0] != 0 or sigs[perm[-1]][trailing_edge][0] != 0:
            continue
        if any(sigs[p][side][0] != 0 for p in perm for side in orth):
            continue
        placements: list[int] = []
        ok = True
        for i in range(len(perm) - 1):
            a = sigs[perm[i]][prev_edge]
            b = sigs[perm[i + 1]][next_edge]
            if a[0] == 0 or a[0] != b[0]:
                ok = False
                break
            placements.append(int(a[1]))
        if ok and (not require_increasing or all(placements[i] < placements[i + 1] for i in range(len(placements) - 1))):
            total += 1
    return total


def _count_overlap_orders(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_edge_sets(block) for block in blocks]
    leading_edge, trailing_edge = BLOCK_ENDS[orient]
    prev_edge, next_edge = MATCH_EDGES[orient]
    orth = ("top", "bot") if orient == "h" else ("left", "right")
    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        first_lead = contacts[perm[0]][leading_edge]
        last_trail = contacts[perm[-1]][trailing_edge]
        if first_lead or last_trail:
            continue
        if any(contacts[p][side] for p in perm for side in orth):
            continue
        if all(contacts[perm[i]][prev_edge] & contacts[perm[i + 1]][next_edge] for i in range(len(perm) - 1)):
            total += 1
    return total


def _connected_mask(mask: np.ndarray) -> bool:
    return bool(is_contiguous(mask.astype(int), background=0, connectivity=4))


@lru_cache(maxsize=None)
def _mask_for_lr(left_spec: tuple[int, int] | None, right_spec: tuple[int, int] | None) -> np.ndarray:
    cells = [(r, c) for r in (1, 2, 3) for c in range(BLOCK_SIZE)]
    required: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    for side, spec in (("left", left_spec), ("right", right_spec)):
        col = 0 if side == "left" else BLOCK_SIZE - 1
        rows = () if spec is None else EDGE_ROWS[spec[0]][spec[1]]
        for r in (1, 2, 3):
            cell = (r, col)
            if r in rows:
                required.add(cell)
            else:
                forbidden.add(cell)

    allowed = [cell for cell in cells if cell not in forbidden]
    optional = [cell for cell in allowed if cell not in required]

    for area in range(max(4, len(required)), 11):
        need = area - len(required)
        for extra in itertools.combinations(optional, need):
            mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
            for r, c in itertools.chain(required, extra):
                mask[r, c] = True
            if not _connected_mask(mask):
                continue
            return mask
    raise RuntimeError(f"no legal mask for {left_spec=} {right_spec=}")


def _make_block(
    color: int,
    *,
    left: tuple[int, int] | None = None,
    right: tuple[int, int] | None = None,
    top: tuple[int, int] | None = None,
    bottom: tuple[int, int] | None = None,
) -> np.ndarray:
    if top is not None or bottom is not None:
        mask = np.rot90(_mask_for_lr(bottom, top), 1)
    else:
        mask = _mask_for_lr(left, right)
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


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


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    perm = list(range(len(blocks)))
    while perm == list(range(len(blocks))):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        placements = [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]
        for block, (r0, c0) in zip(scrambled, placements):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, OTHER[output_orient])


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(["h", "v"])
    edge_count = 1
    colors = random.sample(PALETTE, NUM_BLOCKS)

    # Outgoing placements must increase 1<2<3.
    out_places = [1, 2, 3, None]
    # Incoming placements are chosen to keep count-only ambiguity while breaking exact-overlap logic.
    in_places = [None, 3, 1, 2]

    lead_kw, trail_kw = ORIENT_KW[orient]
    blocks: list[np.ndarray] = []
    for i in range(NUM_BLOCKS):
        kwargs: dict[str, tuple[int, int] | None] = {}
        if in_places[i] is not None:
            kwargs[lead_kw] = (edge_count, int(in_places[i]))
        if out_places[i] is not None:
            kwargs[trail_kw] = (edge_count, int(out_places[i]))
        blocks.append(_make_block(colors[i], **kwargs))

    if _count_valid_orders(blocks, orient, require_increasing=False) <= 1:
        return None
    if _count_valid_orders(blocks, orient, require_increasing=True) != 1:
        return None
    if _count_valid_orders(blocks, OTHER[orient], require_increasing=True) != 0:
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
