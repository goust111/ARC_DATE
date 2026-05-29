from __future__ import annotations

# Variant of 78332cb0
# Dimension: D1 — 边缘匹配条件 (match_cond)
# Change: 相邻块的facing边缘在相同偏移处共享前景像素（交集大小=1），形成唯一链... → 形状嵌合匹配
#
# Design analysis
# ===============
# a) Concrete behavior and flow impact
#    - We keep the original object inventory: 3-4 monochrome 5x5 blocks on bg=7, separated by full sep lines in color 6.
#    - The output is still a 1xN strip in the recovered chain order.
#    - The D1 change is localized to "how two neighboring blocks are recognized as consecutive".
#    - Old task: horizontal neighbors matched when A.right_edge_fg_offsets intersected B.left_edge_fg_offsets.
#    - New task: each active connecting edge carries a 3-row "jigsaw profile" on rows 1..3:
#         bit 1 = the shape reaches the outer boundary on that row (a tab / convex part)
#         bit 0 = the boundary cell is empty while the shape stays one cell inset (a notch / concavity)
#      Two facing edges match iff the successor profile is the bitwise complement of the predecessor profile.
#    - Small traced example for a horizontal pair:
#         A.right = (1,0,1)  -> boundary pixels on rows {1,3}
#         B.left  = (0,1,0)  -> boundary pixel only on row {2}
#      Old logic sees no shared boundary offset and would reject the pair.
#      New logic accepts it because (0,1,0) is the exact complement of (1,0,1).
#
# b) Complexity calibration
#    - Profiles are 3-bit patterns restricted to the six mixed cases (exclude 000 and 111).
#    - This is visibly richer than the original single-contact offset, but still small enough that humans can infer
#      the "complementary outline" rule across 2-4 examples.
#    - Endpoints still have no connector on the outer side, which keeps the chain directional and avoids overloading
#      the visual scene with too many possible matches.
#
# c) Hidden-assumption audit against the original main()
#    - Original _contacts() assumed "matching info = fg positions on the outermost edge only". This is invalid now,
#      because the informative structure includes both the boundary cell and the one-cell inset support shape.
#    - Original ok(a,b) tested set intersection > 0. This is invalid now; complement is the intended relation and
#      overlap is usually exactly 0.
#    - Original _contacts() inferred per-block bg by local majority. That happened to be safe under 4-10 fg pixels.
#      The variant can produce blocks with up to 11 fg pixels, so the solver now uses the global bg directly.
#    - The splitting/rendering logic does not depend on the old D1 assumption and remains unchanged.
#
# d) Placement legality
#    - Every block is still exactly 5x5 and placed in the same strip/grid layouts as the original task.
#    - Generator templates only occupy rows/cols 1..3 plus the designated chain-facing boundary, so blocks never
#      spill outside their 5x5 cell and never overlap separators or one another.
#    - Horizontal blocks leave top/bottom boundaries empty; vertical blocks leave left/right boundaries empty.
#      This guarantees the non-chain orientation has no accidental connectors.
#
# e) Visual observability
#    - The important visible cue is not color, but contour: on each connector side, some rows hit the boundary while
#      the complementary partner has matching recesses on exactly those rows.
#    - Because all true links have zero old-style edge overlap, the variant is not a hidden implementation detail:
#      the new rule is necessary to solve the samples.
#
# f) Visual cleanliness
#    - Blocks stay isolated by the original separator lines; there is no layering or overlap.
#    - Connector templates are sparse, single-color, and aligned to the 5x5 frame, so the notch/tab silhouettes are
#      easy to read without cluttering the grid.
#
# g) Color inferability
#    - Colors remain irrelevant to the ordering rule, exactly as in the original puzzle family.
#      The generator may reuse colors, but no matching or sorting depends on them.
#
# h) Quantitative non-degeneration target
#    - For every true adjacent pair in generated samples, the old rule's boundary-overlap count is 0 while the new
#      complement relation is 1. This is the metric used after testing to confirm that samples truly require D1.
#
# i) New solver assumptions introduced by the variant
#    - A connecting side, if present, follows a fixed 3-row template with an inset support column/row plus optional
#      boundary cells. The solver extracts profiles from that template.
#    - The generator enforces those templates, enforces exactly one valid orientation/permutation, and leaves the
#      perpendicular edges blank so the solver's profile logic remains well-posed.

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]
BLOCK_SIZE = 5
GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400

OTHER = {"h": "v", "v": "h"}
PROFILES = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
]


def _complement(profile: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(1 - int(bit) for bit in profile)


def _find_sep(a: np.ndarray, bg: int) -> int:
    H, W = a.shape
    cand: list[int] = []
    for r in range(H):
        if bool((a[int(r), :] == int(a[int(r), 0])).all()) and int(a[int(r), 0]) != int(bg):
            cand.append(int(a[int(r), 0]))
    for c in range(W):
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


def _render_strip(blocks: list[np.ndarray], orient: str, bg: int, sep: int) -> np.ndarray:
    n = len(blocks)
    S = BLOCK_SIZE
    stride = S + 1
    if orient == "h":
        out = np.full((S, n * S + (n - 1)), int(bg), dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + S] = int(sep)
        return out
    out = np.full((n * S + (n - 1), S), int(bg), dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * stride, 0)
        if i < n - 1:
            out[i * stride + S, :] = int(sep)
    return out


def _edge_profile(block: np.ndarray, bg: int, side: str) -> tuple[int, int, int] | None:
    mask = block != int(bg)
    if side == "left":
        if not mask[1:4, 0:2].any():
            return None
        if not bool(mask[1:4, 1].all()):
            return None
        return tuple(int(mask[r, 0]) for r in range(1, 4))
    if side == "right":
        if not mask[1:4, 3:5].any():
            return None
        if not bool(mask[1:4, 3].all()):
            return None
        return tuple(int(mask[r, 4]) for r in range(1, 4))
    if side == "top":
        if not mask[0:2, 1:4].any():
            return None
        if not bool(mask[1, 1:4].all()):
            return None
        return tuple(int(mask[0, c]) for c in range(1, 4))
    if side == "bot":
        if not mask[3:5, 1:4].any():
            return None
        if not bool(mask[3, 1:4].all()):
            return None
        return tuple(int(mask[4, c]) for c in range(1, 4))
    raise ValueError(side)


def _profiles(block: np.ndarray, bg: int) -> dict[str, tuple[int, int, int] | None]:
    return {
        "left": _edge_profile(block, bg, "left"),
        "right": _edge_profile(block, bg, "right"),
        "top": _edge_profile(block, bg, "top"),
        "bot": _edge_profile(block, bg, "bot"),
    }


def _compatible(a: dict[str, tuple[int, int, int] | None], b: dict[str, tuple[int, int, int] | None], orient: str) -> bool:
    if orient == "h":
        trail = a["right"]
        lead = b["left"]
    else:
        trail = a["bot"]
        lead = b["top"]
    return trail is not None and lead is not None and tuple(lead) == _complement(tuple(trail))


def _all_chains(blocks: list[np.ndarray], orient: str, bg: int) -> list[tuple[int, ...]]:
    info = [_profiles(block, bg) for block in blocks]
    chains: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(_compatible(info[perm[i]], info[perm[i + 1]], orient) for i in range(len(perm) - 1)):
            chains.append(tuple(int(x) for x in perm))
    return chains


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)
    blocks = _split_blocks(g, sep)
    assert blocks, "no blocks found"

    chains_h = _all_chains(blocks, "h", bg)
    chains_v = _all_chains(blocks, "v", bg)

    if len(chains_h) == 1 and not chains_v:
        return _render_strip([blocks[i] for i in chains_h[0]], "h", bg, sep)
    if len(chains_v) == 1 and not chains_h:
        return _render_strip([blocks[i] for i in chains_v[0]], "v", bg, sep)
    if chains_h and chains_v:
        raise AssertionError("ambiguous: both H and V chains found")
    raise AssertionError("must find exactly one valid chain")


def _make_h_block(color: int, *, left: tuple[int, int, int] | None, right: tuple[int, int, int] | None) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    mask[2, 2] = True
    if left is not None:
        mask[1:4, 1] = True
        for r, bit in enumerate(left, start=1):
            if int(bit):
                mask[int(r), 0] = True
    if right is not None:
        mask[1:4, 3] = True
        for r, bit in enumerate(right, start=1):
            if int(bit):
                mask[int(r), 4] = True
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _make_v_block(color: int, *, top: tuple[int, int, int] | None, bottom: tuple[int, int, int] | None) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    mask[2, 2] = True
    if top is not None:
        mask[1, 1:4] = True
        for c, bit in enumerate(top, start=1):
            if int(bit):
                mask[0, int(c)] = True
    if bottom is not None:
        mask[3, 1:4] = True
        for c, bit in enumerate(bottom, start=1):
            if int(bit):
                mask[4, int(c)] = True
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _make_block(
    color: int,
    orient: str,
    *,
    lead: tuple[int, int, int] | None,
    trail: tuple[int, int, int] | None,
) -> np.ndarray:
    if orient == "h":
        return _make_h_block(color, left=lead, right=trail)
    return _make_v_block(color, top=lead, bottom=trail)


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    return len(_all_chains(blocks, orient, BG))


def _render_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        S = BLOCK_SIZE
        grid = np.full((2 * S + 1, 2 * S + 1), BG, dtype=int)
        grid[S, :] = SEP
        grid[:, S] = SEP
        slots = [(0, 0), (0, S + 1), (S + 1, 0), (S + 1, S + 1)]
        for block, (r0, c0) in zip(scrambled, slots):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, OTHER[output_orient], BG, SEP)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    link_profiles = random.sample(PROFILES, n - 1)

    blocks: list[np.ndarray] = []
    for i in range(n):
        lead = _complement(link_profiles[i - 1]) if i > 0 else None
        trail = link_profiles[i] if i < n - 1 else None
        color = random.choice(PALETTE)
        blocks.append(_make_block(color, orient, lead=lead, trail=trail))

    if _count_orders(blocks, orient) != 1:
        return None
    if _count_orders(blocks, OTHER[orient]) != 0:
        return None

    out = _render_strip(blocks, orient, BG, SEP)
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


def _edge_overlap_count(a: np.ndarray, b: np.ndarray, orient: str) -> int:
    ma = a != BG
    mb = b != BG
    if orient == "h":
        return int(np.logical_and(ma[:, -1], mb[:, 0]).sum())
    return int(np.logical_and(ma[-1, :], mb[0, :]).sum())

