# Combo variant of 78332cb0 (index 1)
# combo_spec: D1/0+D3/1
# Name: 多色精灵形状嵌合匹配
#
# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Base task: 3-4 scrambled 5x5 blocks are reordered into a strip. The original solver treated
#      all non-background pixels as one foreground mask and linked neighbors by edge-overlap.
#    - Fused variant: each block now has exactly two foreground colors. One color is shared by
#      every block and is the only color whose edge contour counts for chaining. That contour is
#      not an overlap offset any more; it is a 3-bit jigsaw profile on rows/cols 1..3, and two
#      facing sides match only when those 3 bits are exact complements.
#    - Concrete horizontal pair:
#         A shared-color right profile  = (1,0,1)
#         B shared-color left profile   = (0,1,0)
#      So under the new rule A→B is valid. Under the old overlap rule the shared-color edge
#      intersection is empty, so the original solver would reject the pair.
#    - The second color is not passive recoloring. On every active connector side it fills exactly
#      the boundary positions where the shared-color profile has 0s. Visually A's right edge
#      becomes [shared, accent, shared] and B's left edge becomes [accent, shared, accent].
#      If one ignores color and reads the whole boundary silhouette, both sides look fully filled
#      (111), so the shape-complement cue disappears. The solver must therefore first isolate the
#      repeated color and only then compare complementary contours.
#
# b) Complexity calibration
#    - Jigsaw profiles are restricted to the six mixed 3-bit patterns (exclude 000 and 111). This
#      keeps the profile vocabulary small enough for humans to infer from few examples, while still
#      being visibly different from single-pixel edge contact.
#    - Only one role color is shared across all blocks; the accent colors vary by block. This keeps
#      the color-choice hypothesis narrow and repeated across the full training set.
#    - The accent color is guaranteed to appear on the same connector boundary as the shared color,
#      so the fused behavior is visually explicit instead of hidden in metadata or degenerate noise.
#
# c) Hidden-assumption audit against the original main()
#    - Original `_contacts(block)` assumed all non-bg pixels can be merged. That is false now:
#      merged colors erase the intended notch/tab pattern because the accent fills the shared-color
#      zeros on active sides.
#    - Original matching predicate used edge-set intersection > 0. That is false now because true
#      neighbors usually have zero shared-color overlap; complement, not intersection, defines a link.
#    - Original local-majority background inference inside each block is no longer necessary and is
#      risky once blocks become denser. The fused solver therefore uses the global background and
#      explicitly reasons over colors present in each block.
#    - New solver assumption: every block has exactly two foreground colors and exactly one of them
#      appears in every block. The generator must guarantee this, otherwise the color-selection step
#      is underdetermined.
#
# d) Placement legality
#    - All objects remain 5x5 and are still separated by full separator rows/cols, so the existing
#      split/render logic remains legal.
#    - Connector templates only use the 5x5 cell interior plus the intended boundary side; they do
#      not spill into separators or neighboring blocks.
#    - Boundary accent cells are placed only on already active connector sides, adjacent to the
#      shared-color support column/row, so each block stays a single connected foreground component.
#
# e) Visual observability
#    - The repeated color across all blocks is the only viable candidate for the matching color.
#      On that color alone, neighboring blocks show complementary notch/tab contours.
#    - The accent color is visible on the same boundaries but never repeats across all blocks, so a
#      human can infer that "common color = signal, local color = distractor/filler".
#    - Samples therefore force both inferences: identify the common color, then read complementarity.
#
# f) Visual cleanliness
#    - Blocks stay sparse enough to read because each side only exposes three possible connector
#      rows/cols, and the accent is confined to those same small templates.
#    - There is no overlap between blocks, no extra legends, and no hidden indexing; the scene is
#      still just separated sprites that need reordering.
#
# g) Color-change inferability
#    - The rule is structural, not value-specific: the shared color matters because it is the only
#      foreground color present in every block, not because a fixed color id always has a special role.
#    - Accent colors are distinct per block, making the shared color easy to spot from 2-4 examples.
#
# h) Quantitative non-degeneration targets used after implementation
#    - Correct criterion: shared-color + shape-complement must yield exactly one chain in exactly one orientation.
#    - D1 check: shared-color + old overlap matching must yield zero valid chains.
#    - D3 check: all-foreground + shape-complement must yield zero valid chains because the accent
#      fills the shared-color notches on active connector sides.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]
BLOCK_SIZE = 5
GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 500

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
    stride = BLOCK_SIZE + 1
    if orient == "h":
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), int(bg), dtype=int)
        for i, block in enumerate(blocks):
            paste(out, block, 0, i * stride)
            if i < n - 1:
                out[:, i * stride + BLOCK_SIZE] = int(sep)
        return out
    out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), int(bg), dtype=int)
    for i, block in enumerate(blocks):
        paste(out, block, i * stride, 0)
        if i < n - 1:
            out[i * stride + BLOCK_SIZE, :] = int(sep)
    return out


def _shared_match_color(blocks: list[np.ndarray], bg: int) -> int | None:
    color_sets = [{int(v) for v in np.unique(block) if int(v) != int(bg)} for block in blocks]
    if not color_sets or any(len(colors) != 2 for colors in color_sets):
        return None
    common = set.intersection(*color_sets)
    if len(common) != 1:
        return None
    return int(next(iter(common)))


def _edge_profile(block: np.ndarray, side: str, *, color: int | None, bg: int) -> tuple[int, int, int] | None:
    mask = (block != int(bg)) if color is None else (block == int(color))
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


def _profiles(block: np.ndarray, *, color: int | None, bg: int) -> dict[str, tuple[int, int, int] | None]:
    return {
        "left": _edge_profile(block, "left", color=color, bg=bg),
        "right": _edge_profile(block, "right", color=color, bg=bg),
        "top": _edge_profile(block, "top", color=color, bg=bg),
        "bot": _edge_profile(block, "bot", color=color, bg=bg),
    }


def _compatible_shape(
    a: dict[str, tuple[int, int, int] | None],
    b: dict[str, tuple[int, int, int] | None],
    orient: str,
) -> bool:
    if orient == "h":
        trail = a["right"]
        lead = b["left"]
    else:
        trail = a["bot"]
        lead = b["top"]
    return trail is not None and lead is not None and tuple(lead) == _complement(tuple(trail))


def _edge_sets(block: np.ndarray, *, color: int | None, bg: int) -> dict[str, set[int]]:
    mask = (block != int(bg)) if color is None else (block == int(color))
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _all_shape_chains(blocks: list[np.ndarray], orient: str, *, color: int | None, bg: int) -> list[tuple[int, ...]]:
    info = [_profiles(block, color=color, bg=bg) for block in blocks]
    chains: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(_compatible_shape(info[perm[i]], info[perm[i + 1]], orient) for i in range(len(perm) - 1)):
            chains.append(tuple(int(x) for x in perm))
    return chains


def _all_overlap_chains(blocks: list[np.ndarray], orient: str, *, color: int | None, bg: int) -> list[tuple[int, ...]]:
    contacts = [_edge_sets(block, color=color, bg=bg) for block in blocks]

    def ok(a: int, b: int) -> bool:
        if orient == "h":
            return bool(contacts[a]["right"] & contacts[b]["left"])
        return bool(contacts[a]["bot"] & contacts[b]["top"])

    chains: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            chains.append(tuple(int(x) for x in perm))
    return chains


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scrambled 5x5 blocks into the unique strip whose shared-color contours interlock.

    Input structure:
    - The scene contains 3-4 separate 5x5 blocks on a global background, divided by full separator lines.
    - Every block uses exactly two foreground colors.
    - Exactly one of those foreground colors appears in every block; the other foreground color varies locally.

    Rule:
    - First identify the unique foreground color common to all blocks.
    - On that color only, read each active edge as a 3-bit jigsaw profile over rows/cols 1..3.
    - Two consecutive blocks connect when their facing profiles are exact complements.
    - Order the blocks in the only orientation and permutation that forms a full chain, then render them as a strip with separator lines.

    Key insight:
    - The repeated color carries the real connector silhouette, while the other color deliberately fills some of the visible edge gaps.
      Treating all foreground pixels alike destroys the notch/tab information; isolating the shared color restores the unique chain.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _find_sep(g, bg)
    blocks = _split_blocks(g, sep)

    assert blocks, "no blocks found"
    assert all(block.shape == (BLOCK_SIZE, BLOCK_SIZE) for block in blocks), "all blocks must remain 5x5"

    match_color = _shared_match_color(blocks, bg)
    assert match_color is not None, "must have exactly one shared match color"

    chains_h = _all_shape_chains(blocks, "h", color=match_color, bg=bg)
    chains_v = _all_shape_chains(blocks, "v", color=match_color, bg=bg)

    if len(chains_h) == 1 and not chains_v:
        return _render_strip([blocks[i] for i in chains_h[0]], "h", bg, sep)
    if len(chains_v) == 1 and not chains_h:
        return _render_strip([blocks[i] for i in chains_v[0]], "v", bg, sep)
    if chains_h and chains_v:
        raise AssertionError("ambiguous: both H and V chains found")
    raise AssertionError("must find exactly one valid chain")


def _paint_side(
    block: np.ndarray,
    side: str,
    profile: tuple[int, int, int],
    *,
    match_color: int,
    accent_color: int,
) -> None:
    if side == "left":
        block[1:4, 1] = int(match_color)
        for i, bit in enumerate(profile, start=1):
            block[i, 0] = int(match_color) if int(bit) else int(accent_color)
        return
    if side == "right":
        block[1:4, 3] = int(match_color)
        for i, bit in enumerate(profile, start=1):
            block[i, 4] = int(match_color) if int(bit) else int(accent_color)
        return
    if side == "top":
        block[1, 1:4] = int(match_color)
        for i, bit in enumerate(profile, start=1):
            block[0, i] = int(match_color) if int(bit) else int(accent_color)
        return
    if side == "bot":
        block[3, 1:4] = int(match_color)
        for i, bit in enumerate(profile, start=1):
            block[4, i] = int(match_color) if int(bit) else int(accent_color)
        return
    raise ValueError(side)


def _make_block(
    match_color: int,
    accent_color: int,
    orient: str,
    *,
    lead: tuple[int, int, int] | None,
    trail: tuple[int, int, int] | None,
) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[2, 2] = int(match_color)
    if orient == "h":
        if lead is not None:
            _paint_side(block, "left", lead, match_color=match_color, accent_color=accent_color)
        if trail is not None:
            _paint_side(block, "right", trail, match_color=match_color, accent_color=accent_color)
    else:
        if lead is not None:
            _paint_side(block, "top", lead, match_color=match_color, accent_color=accent_color)
        if trail is not None:
            _paint_side(block, "bot", trail, match_color=match_color, accent_color=accent_color)

    fg_mask = block != BG
    assert is_contiguous(fg_mask.astype(int), background=0, connectivity=4)
    return block


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
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, OTHER[output_orient], BG, SEP)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    link_profiles = random.sample(PROFILES, n - 1)
    match_color = random.choice(PALETTE)
    accent_colors = random.sample([c for c in PALETTE if c != match_color], n)

    blocks: list[np.ndarray] = []
    for i in range(n):
        lead = _complement(link_profiles[i - 1]) if i > 0 else None
        trail = link_profiles[i] if i < n - 1 else None
        block = _make_block(match_color, accent_colors[i], orient, lead=lead, trail=trail)
        blocks.append(block)

    if _shared_match_color(blocks, BG) != match_color:
        return None
    if len(_all_shape_chains(blocks, orient, color=match_color, bg=BG)) != 1:
        return None
    if _all_shape_chains(blocks, OTHER[orient], color=match_color, bg=BG):
        return None
    if _all_shape_chains(blocks, orient, color=None, bg=BG):
        return None
    if _all_shape_chains(blocks, OTHER[orient], color=None, bg=BG):
        return None
    if _all_overlap_chains(blocks, orient, color=match_color, bg=BG):
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
