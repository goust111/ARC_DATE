# Variant of 78332cb0
# Dimension: D3 — 源对象结构 (src)
# Change: 5x5的单色精灵块，每块一个连通分量，4-10个前景像素，由品红分隔线分开... → 多色精灵块
#
# Design analysis
# a) Concrete behavior trace:
#    - Original blocks are single-color. Any non-background pixel on a block edge counts as a
#      possible chain contact. The solver can treat "foreground" as one undifferentiated mask.
#    - Variant blocks still live in 5x5 cells and are still separated by full magenta lines, but
#      each block now uses exactly two foreground colors:
#        * one shared color that appears in every block: this is the match color
#        * one block-local accent color: this is visually salient but must be ignored for chaining
#    - The shared color keeps the original left/right or top/bottom contact logic. The accent
#      color draws a stripe on the perpendicular axis, touching the opposite pair of edges and
#      therefore acting as a decoy if one naively matches all foreground.
#    - Example trace for a horizontal chain:
#        original: block A right edge {row 1}, block B left edge {row 1} -> A→B
#        variant: both A and B also contain an accent vertical stripe touching top/bottom at
#        column 2. Correct logic still uses only the shared color and keeps A→B horizontal.
#        Old logic that merges both colors would also see a vertical chain, so color choice now
#        changes the answer space rather than merely recoloring shapes.
# b) Complexity calibration:
#    - The match color is the unique color common to all blocks, giving a small but repeated clue
#      across every training example.
#    - Accent colors are different per block, so the common color is easy to isolate and the
#      variant is visually legible from few examples.
#    - Only one special role color exists per sample; this avoids exploding the number of color
#      cases while still introducing a real inference layer.
# c) Implicit assumption audit against the original main():
#    - `_contacts` originally used `block != most_common(block)`, which silently assumes "all
#      non-bg colors are equivalent foreground". That is invalid now because one foreground color
#      is signal and the other is decoy.
#    - `_find_chain` can stay unchanged once `_contacts` is redefined to look only at the shared
#      match color.
#    - `_find_sep` assumes true separator rows/cols are the only full constant non-bg lines in the
#      whole grid. The generator therefore must avoid making a block contain a full constant row or
#      column when the whole grid width/height is 5. The accent stripe keeps the center cell in the
#      match color to prevent a full constant line.
# d) Placement legality:
#    - Input layouts remain the original strip-or-2x2 arrangements, so blocks never overlap and
#      stay inside the grid.
#    - The accent pixels are added inside each 5x5 block only, preserving separator spacing.
#    - Output rendering is still a 1xN strip in the discovered chain order.
# e) Visual observability:
#    - In every example, one color repeats across every block and the other changes by block.
#      Humans can inspect aligned edge pixels of the repeated color and see that the local accent
#      stripe is irrelevant to ordering.
#    - Because the accent stripe reaches the perpendicular edges, samples visibly differ from the
#      original task where those edges were empty.
# f) Visual cleanliness:
#    - Each block remains only 5x5 with two colors and one connected foreground component, so the
#      scene stays sparse and readable.
#    - The decoy stripe is centered and consistent, avoiding noisy scattered markings.
# g) Color-rule inferability:
#    - The rule is not "color id X always matters". Instead, the structurally repeated color shared
#      by all blocks matters. This is visible from the examples because the repeated color supports
#      the chain while the varying accent color does not.
#
# New solver assumptions introduced by this variant:
# - Every block contains exactly two non-background colors.
# - Exactly one non-background color is shared by all blocks; that color is the match color.
# - Generator guarantee: under shared-color contacts there is exactly one valid chain orientation
#   and one permutation, while under legacy all-foreground contacts the perpendicular orientation
#   is also available via the accent stripe.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks into a connected strip by matching only the shared color.

    Variant rule:
    - Each 5x5 block contains exactly two foreground colors.
    - Exactly one foreground color is shared by all blocks.
    - Only that shared color contributes edge-contact offsets for chaining.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _find_sep(a: np.ndarray) -> int:
        cand: list[int] = []
        for r in range(int(a.shape[0])):
            if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != bg:
                cand.append(int(a[r, 0]))
        for c in range(int(a.shape[1])):
            if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != bg:
                cand.append(int(a[0, c]))
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(int(a.shape[0])) if bool((a[r, :] == sep).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(int(a.shape[1])) if bool((a[:, c] == sep).all())]

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
            out = np.full((n * 5 + (n - 1), 5), bg, dtype=int)
            rr = 0
            for i, block in enumerate(blocks):
                out[rr : rr + 5, :] = block
                rr += 5
                if i < n - 1:
                    out[rr, :] = sep
                    rr += 1
            return out
        out = np.full((5, n * 5 + (n - 1)), bg, dtype=int)
        cc = 0
        for i, block in enumerate(blocks):
            out[:, cc : cc + 5] = block
            cc += 5
            if i < n - 1:
                out[:, cc] = sep
                cc += 1
        return out

    blocks = _split_blocks()
    assert blocks, "must find at least one block"
    assert all(block.shape == (5, 5) for block in blocks), "all blocks must remain 5x5"

    color_sets = [{int(v) for v in np.unique(block) if int(v) != bg} for block in blocks]
    assert all(len(colors) == 2 for colors in color_sets), "each block must have exactly two fg colors"
    common = set.intersection(*color_sets)
    assert len(common) == 1, "must have a unique shared match color"
    match_color = int(next(iter(common)))

    def _contacts(block: np.ndarray) -> dict[str, set[int]]:
        mask = block == match_color
        return {
            "top": set(np.where(mask[0, :])[0].tolist()),
            "bot": set(np.where(mask[-1, :])[0].tolist()),
            "left": set(np.where(mask[:, 0])[0].tolist()),
            "right": set(np.where(mask[:, -1])[0].tolist()),
        }

    def _find_chain(orient: str) -> tuple[int, ...] | None:
        contacts = [_contacts(block) for block in blocks]

        def ok(a: int, b: int) -> bool:
            if orient == "h":
                return bool(contacts[a]["right"] & contacts[b]["left"])
            return bool(contacts[a]["bot"] & contacts[b]["top"])

        idx = list(range(len(blocks)))
        for perm in itertools.permutations(idx):
            if all(ok(perm[i], perm[i + 1]) for i in range(len(idx) - 1)):
                return tuple(int(i) for i in perm)
        return None

    chain_h = _find_chain("h")
    chain_v = _find_chain("v")

    if chain_h is not None and chain_v is None:
        return _render_strip([blocks[i] for i in chain_h], axis=1)
    if chain_v is not None and chain_h is None:
        return _render_strip([blocks[i] for i in chain_v], axis=0)
    assert not (chain_h is not None and chain_v is not None), "ambiguous: both H and V chains found"
    assert chain_h is not None or chain_v is not None, "must find at least one valid chain"
    raise AssertionError("unreachable")


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
BASE_AREA_MIN = 4
BASE_AREA_MAX = 6
FINAL_AREA_MIN = 6
FINAL_AREA_MAX = 10
EDGE_CONTACT_OFFSETS = [1, 2, 3]

GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}
ORIENT_KW = {"h": ("left", "right"), "v": ("top", "bottom")}


def _shared_match_color(blocks: list[np.ndarray]) -> int | None:
    color_sets = [{int(v) for v in np.unique(block) if int(v) != BG} for block in blocks]
    common = set.intersection(*color_sets)
    if len(common) != 1:
        return None
    return int(next(iter(common)))


def _edge_sets(block: np.ndarray, *, color: int | None) -> dict[str, set[int]]:
    mask = block != BG if color is None else block == color
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _count_orders(blocks: list[np.ndarray], orient: str, *, color: int | None) -> int:
    contacts = [_edge_sets(block, color=color) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]

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
    candidates = has_neighbor & ~interior
    rs, cs = np.where(candidates)
    return [(int(r) + 1, int(c) + 1) for r, c in zip(rs, cs)]


def _base_mask(left_offset: int | None, right_offset: int | None, target_area: int) -> np.ndarray | None:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left_offset is not None:
        mask[left_offset, 0:3] = True
        mask[min(left_offset, 2) : max(left_offset, 2) + 1, 2] = True
    if right_offset is not None:
        mask[right_offset, 2:BLOCK_SIZE] = True
        mask[min(right_offset, 2) : max(right_offset, 2) + 1, 2] = True

    if mask.sum() < BASE_AREA_MIN:
        mask[random.choice([1, 3]), 2] = True

    while int(mask.sum()) < target_area:
        frontier = _interior_frontier(mask)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        mask[r, c] = True

    if left_offset is None:
        if mask[:, 0].any():
            return None
    elif np.where(mask[:, 0])[0].tolist() != [left_offset]:
        return None

    if right_offset is None:
        if mask[:, -1].any():
            return None
    elif np.where(mask[:, -1])[0].tolist() != [right_offset]:
        return None

    if mask[0, :].any() or mask[-1, :].any():
        return None
    if any(mask[r, :].all() for r in range(BLOCK_SIZE)):
        return None
    if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _make_match_mask(
    *, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None
) -> np.ndarray | None:
    target_area = random.randint(BASE_AREA_MIN, BASE_AREA_MAX)
    if top is not None or bottom is not None:
        mask = _base_mask(bottom, top, target_area)
        if mask is None:
            return None
        return np.rot90(mask, 1)
    return _base_mask(left, right, target_area)


def _make_block(
    match_color: int,
    accent_color: int,
    orient: str,
    *,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
) -> np.ndarray | None:
    match_mask = _make_match_mask(left=left, right=right, top=top, bottom=bottom)
    if match_mask is None:
        return None

    if orient == "h":
        accent_coords = [(0, 2), (1, 2), (3, 2), (4, 2)]
    else:
        accent_coords = [(2, 0), (2, 1), (2, 3), (2, 4)]

    full_mask = match_mask.copy()
    for r, c in accent_coords:
        full_mask[r, c] = True

    area = int(full_mask.sum())
    if not (FINAL_AREA_MIN <= area <= FINAL_AREA_MAX):
        return None
    if not is_contiguous(full_mask.astype(int), background=0, connectivity=4):
        return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[full_mask] = match_color
    for r, c in accent_coords:
        block[r, c] = accent_color

    fg_colors = {int(v) for v in np.unique(block) if int(v) != BG}
    if fg_colors != {match_color, accent_color}:
        return None
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

    if len(blocks) == 4 and random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        starts = [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]
        for block, (r0, c0) in zip(scrambled, starts):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, OTHER[output_orient])


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    shared_offsets = random.sample(EDGE_CONTACT_OFFSETS, n - 1)

    match_color = random.choice(PALETTE)
    accent_colors = random.sample([c for c in PALETTE if c != match_color], n)
    lead_kw, trail_kw = ORIENT_KW[orient]

    blocks: list[np.ndarray] = []
    for i in range(n):
        kw: dict[str, int] = {}
        if i > 0:
            kw[lead_kw] = shared_offsets[i - 1]
        if i < n - 1:
            kw[trail_kw] = shared_offsets[i]
        block = _make_block(match_color, accent_colors[i], orient, **kw)
        if block is None:
            return None
        blocks.append(block)

    if _shared_match_color(blocks) != match_color:
        return None
    if _count_orders(blocks, orient, color=match_color) != 1:
        return None
    if _count_orders(blocks, OTHER[orient], color=match_color) != 0:
        return None
    if _count_orders(blocks, OTHER[orient], color=None) == 0:
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
