# Combo variant of 78332cb0 (index 7)
# combo_spec: D2/1+D10/0
# Name: 双标记符号解码

from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
# - The inherited scene is still a scrambled set of 3-4 disjoint 5x5 blocks on
#   bg=7 with full separator lines sep=6, and the output is still a single
#   ordered strip.
# - D2/1 adds an outer 1-cell top+left padding band and places exactly one red
#   marker pixel there: top band means use the vertical chain; left band means
#   use the horizontal chain.
# - D10/0 replaces "empty outer edge reveals the endpoint" with explicit block
#   identity. One block contains a green marker pixel (start block) and one
#   block contains a yellow marker pixel (end block). These pixels stay inside
#   the blocks and therefore also stay visible in the output.
# - Example with three blocks:
#   old task: detect the only valid orientation, then infer first/last because
#   endpoint outer edges are empty.
#   combo task: decode the red border marker to choose H or V, locate the green
#   and yellow marked blocks inside the scrambled set, then search only chains
#   of the chosen orientation whose first block is green-marked and whose last
#   block is yellow-marked.
# - The important behavioral difference is that the chosen orientation is no
#   longer inferred from exclusivity, and the chosen direction's ordering is not
#   left to edge geometry alone either: endpoint colors now break ties.
#
# b) Complexity calibration
# - Only three marker colors are introduced and each has a single clear role:
#   red=orientation choice, green=start, yellow=end.
# - The border marker uses only two positions (top or left), so the direction
#   code is easy to learn.
# - Endpoint markers live at a fixed interior cell of the marked block, so they
#   are visually salient without disturbing edge contacts.
# - Generator intentionally keeps both H and V chain directions viable, and for
#   the chosen direction it requires more than one valid chain before endpoint
#   restriction. This prevents the endpoint colors from becoming decorative.
#
# c) Hidden-assumption audit against the original solver
# - Original main() assumed the entire canvas was just the block layout. The
#   combo variant adds a padded border, so the solver must decode the direction
#   marker and crop away row 0 / col 0 before separator detection.
# - Original main() assumed exactly one orientation had a valid chain. That is
#   false here; the solver must not reject "both orientations work".
# - Original generator advertised endpoints with empty outer chain edges. That is
#   false here; the solver must instead find explicit start/end marker colors.
# - Edge matching still uses foreground-vs-background only. This remains safe
#   because start/end markers are guaranteed to sit on interior cells, never on
#   chain-contact edges.
#
# d) Placement legality
# - The border marker occupies only the added top row or left column, so it
#   cannot overlap separators or 5x5 slots.
# - Start/end markers recolor the center cell of existing foreground inside two
#   distinct blocks, so they do not change block size, overlap other objects, or
#   alter edge contacts.
# - Scrambled input layouts remain legal inherited layouts: 2x2 grid for some
#   four-block cases, otherwise a strip with one-cell sep lines.
#
# e) Visual observability
# - Across examples, the red border pixel consistently predicts whether the
#   output is horizontal or vertical.
# - The green-marked block always becomes the first output block in that chosen
#   strip, and the yellow-marked block always becomes the last.
# - Middle blocks are still determined by visible edge-offset continuation, so
#   the rule is inferable from 2-4 examples.
#
# f) Visual cleanliness
# - Marker colors 2/3/4 are removed from the normal sprite palette, keeping all
#   three symbolic pixels unambiguous.
# - All markers are single pixels in visually separated regions: border margin
#   for orientation, block centers for endpoints.
#
# g) Behavior metric used after implementation
# - orientation_branching: the non-chosen orientation still has at least one
#   valid full chain.
# - endpoint_disambiguation: in the chosen orientation, unrestricted valid-chain
#   count is >1, but the count becomes exactly 1 after fixing the green/yellow
#   endpoints. This is the key non-degeneracy check for D10/0.
#
# h) New solver assumptions and generator guarantees
# - Solver assumes exactly one red border marker on either the top band or the
#   left band. Generator always creates exactly one such marker.
# - Solver assumes exactly one green-marked block and one yellow-marked block.
#   Generator marks two distinct blocks and excludes those colors from sprites.
# - Solver assumes the endpoint markers do not touch matching edges. Generator
#   places them at the center cell, which is foreground in every generated block.
# ---------------------------------------------------------------------------


BG = 7
SEP = 6
ORIENT_MARKER = 2
START_MARKER = 3
END_MARKER = 4
PALETTE = [1, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
EDGE_OFFSETS = [1, 2, 3]
GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 1200

OTHER = {"h": "v", "v": "h"}
EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}


def _find_sep(a: np.ndarray, bg: int) -> int:
    cand: list[int] = []
    h, w = a.shape
    for r in range(h):
        if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
            cand.append(int(a[r, 0]))
    for c in range(w):
        if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
            cand.append(int(a[0, c]))
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
    s = BLOCK_SIZE
    if orient == "v":
        out = np.full((n * s + (n - 1), s), int(bg), dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            out[rr : rr + s, :] = block
            rr += s
            if i < n - 1:
                out[rr, :] = int(sep)
                rr += 1
        return out
    out = np.full((s, n * s + (n - 1)), int(bg), dtype=int)
    cc = 0
    for i, block in enumerate(blocks):
        out[:, cc : cc + s] = block
        cc += s
        if i < n - 1:
            out[:, cc] = int(sep)
            cc += 1
    return out


def _edge_sets(block: np.ndarray, bg: int) -> dict[str, set[int]]:
    mask = block != int(bg)
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _valid_orders(
    blocks: list[np.ndarray],
    orient: str,
    bg: int,
    *,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> list[tuple[int, ...]]:
    contacts = [_edge_sets(block, bg) for block in blocks]
    lead, trail = EDGE_PAIRS[orient]
    valid: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(blocks))):
        if start_idx is not None and int(perm[0]) != int(start_idx):
            continue
        if end_idx is not None and int(perm[-1]) != int(end_idx):
            continue
        if all(bool(contacts[perm[i]][lead] & contacts[perm[i + 1]][trail]) for i in range(len(perm) - 1)):
            valid.append(tuple(int(x) for x in perm))
    return valid


def _decode_orient_marker(g: np.ndarray) -> str:
    top = [int(c) for c in np.where(g[0, :] == ORIENT_MARKER)[0].tolist()]
    left = [int(r) for r in np.where(g[:, 0] == ORIENT_MARKER)[0].tolist()]
    assert bool(top) ^ bool(left), "expected one orientation marker on exactly one padded side"
    return "v" if top else "h"


def _find_marked_block(blocks: list[np.ndarray], color: int) -> int:
    hits = [i for i, block in enumerate(blocks) if bool((block == int(color)).any())]
    assert len(hits) == 1, f"expected exactly one block containing marker color {color}"
    return int(hits[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder scrambled 5x5 blocks into a strip by decoding two marker systems.

    Input structure:
    - The puzzle still contains several separated 5x5 sprite blocks on a background
      with full separator rows/cols, but the whole layout is wrapped by a 1-cell
      top+left border used only for a direction marker.
    - A red pixel on the top border means "use vertical chaining"; a red pixel on
      the left border means "use horizontal chaining".
    - Inside the blocks themselves, one block has a green marker pixel (start) and
      one block has a yellow marker pixel (end).

    Rule:
    - First decode the red border marker to choose whether edge matching should be
      evaluated horizontally or vertically.
    - Then identify the green-marked block as the first block and the yellow-marked
      block as the last block of the chosen chain.
    - Among all permutations consistent with those endpoints, keep the unique one
      whose consecutive blocks share foreground pixels on the chosen facing edges,
      and render that ordered strip in the chosen orientation.

    Key insight:
    - The border marker selects the chain axis, the endpoint colors pin the two ends,
      and edge-contact continuation only has to determine the middle ordering.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    orient = _decode_orient_marker(g)
    core = g[1:, 1:]
    sep = _find_sep(core, bg)
    blocks = _split_blocks(core, sep)
    assert blocks, "no 5x5 blocks found"

    start_idx = _find_marked_block(blocks, START_MARKER)
    end_idx = _find_marked_block(blocks, END_MARKER)
    assert start_idx != end_idx, "start and end markers must be on different blocks"

    chains = _valid_orders(blocks, orient, bg, start_idx=start_idx, end_idx=end_idx)
    assert len(chains) == 1, f"expected exactly one {orient} chain after endpoint decoding"
    ordered = [blocks[i] for i in chains[0]]
    return _render_strip(ordered, orient, bg, sep)


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


def _base_mask(
    *,
    left: int | None,
    right: int | None,
    top: int | None,
    bottom: int | None,
) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)

    if left is not None:
        mask[left, 0:3] = True
        mask[min(left, 2) : max(left, 2) + 1, 2] = True
    if right is not None:
        mask[right, 2:BLOCK_SIZE] = True
        mask[min(right, 2) : max(right, 2) + 1, 2] = True
    if top is not None:
        mask[0:3, top] = True
        mask[2, min(top, 2) : max(top, 2) + 1] = True
    if bottom is not None:
        mask[2:BLOCK_SIZE, bottom] = True
        mask[2, min(bottom, 2) : max(bottom, 2) + 1] = True

    if not mask.any():
        mask[2, 2] = True
    mask[2, 2] = True
    return mask


def _grow_mask(mask: np.ndarray) -> np.ndarray | None:
    out = mask.copy()
    base = int(out.sum())
    if base > FG_AREA_MAX:
        return None
    target = random.randint(max(FG_AREA_MIN, base), min(FG_AREA_MAX, base + 2))
    while int(out.sum()) < int(target):
        frontier = _interior_frontier(out)
        if not frontier:
            return None
        r, c = random.choice(frontier)
        out[r, c] = True
    if not is_contiguous(out.astype(int), background=0, connectivity=4):
        return None
    return out


def _make_block(
    color: int,
    *,
    left: int | None,
    right: int | None,
    top: int | None,
    bottom: int | None,
) -> np.ndarray | None:
    mask = _base_mask(left=left, right=right, top=top, bottom=bottom)
    mask = _grow_mask(mask)
    if mask is None:
        return None

    if left is None:
        if mask[:, 0].any():
            return None
    elif np.where(mask[:, 0])[0].tolist() != [left]:
        return None

    if right is None:
        if mask[:, -1].any():
            return None
    elif np.where(mask[:, -1])[0].tolist() != [right]:
        return None

    if top is None:
        if mask[0, :].any():
            return None
    elif np.where(mask[0, :])[0].tolist() != [top]:
        return None

    if bottom is None:
        if mask[-1, :].any():
            return None
    elif np.where(mask[-1, :])[0].tolist() != [bottom]:
        return None

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _mark_block(block: np.ndarray, marker_color: int) -> np.ndarray:
    out = block.copy()
    assert int(out[2, 2]) != BG, "center cell must be foreground before marker recolor"
    out[2, 2] = int(marker_color)
    return out


def _render_input_core(blocks: list[np.ndarray], output_orient: str, chain_order: tuple[int, ...]) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    forbidden = list(chain_order)
    while perm == list(range(n)) or perm == forbidden:
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        s = BLOCK_SIZE
        grid = np.full((2 * s + 1, 2 * s + 1), BG, dtype=int)
        grid[s, :] = SEP
        grid[:, s] = SEP
        starts = [(0, 0), (0, s + 1), (s + 1, 0), (s + 1, s + 1)]
        for block, (r0, c0) in zip(scrambled, starts):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, random.choice([OTHER[output_orient], output_orient]), BG, SEP)


def _add_orient_marker(core: np.ndarray, orient: str) -> np.ndarray:
    h, w = core.shape
    out = np.full((h + 1, w + 1), BG, dtype=int)
    out[1:, 1:] = core
    if orient == "v":
        out[0, 1 + w // 2] = ORIENT_MARKER
    else:
        out[1 + h // 2, 0] = ORIENT_MARKER
    return out


def _valid_orders_from_specs(specs: list[dict[str, int]], orient: str) -> list[tuple[int, ...]]:
    lead, trail = EDGE_PAIRS[orient]
    valid: list[tuple[int, ...]] = []
    for perm in itertools.permutations(range(len(specs))):
        if all(int(specs[perm[i]][lead]) == int(specs[perm[i + 1]][trail]) for i in range(len(perm) - 1)):
            valid.append(tuple(int(x) for x in perm))
    return valid


def _choose_endpoint_restriction(orders: list[tuple[int, ...]]) -> list[tuple[int, int, tuple[int, ...]]]:
    by_pair: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for perm in orders:
        by_pair.setdefault((int(perm[0]), int(perm[-1])), []).append(perm)
    out: list[tuple[int, int, tuple[int, ...]]] = []
    for (start_idx, end_idx), perms in by_pair.items():
        if len(perms) == 1:
            out.append((int(start_idx), int(end_idx), perms[0]))
    return out


def _search_specs() -> tuple[list[dict[str, int]], str, tuple[int, ...], int, int] | None:
    n = random.choice([3, 4])
    for _ in range(400):
        specs = []
        for _block_idx in range(n):
            specs.append(
                {
                    "left": int(random.choice(EDGE_OFFSETS)),
                    "right": int(random.choice(EDGE_OFFSETS)),
                    "top": int(random.choice(EDGE_OFFSETS)),
                    "bot": int(random.choice(EDGE_OFFSETS)),
                }
            )

        orders_h = _valid_orders_from_specs(specs, "h")
        orders_v = _valid_orders_from_specs(specs, "v")
        if not orders_h or not orders_v:
            continue

        candidates: list[tuple[str, tuple[int, ...], int, int]] = []
        for orient, orders in (("h", orders_h), ("v", orders_v)):
            if len(orders) <= 1:
                continue
            for start_idx, end_idx, perm in _choose_endpoint_restriction(orders):
                candidates.append((orient, perm, start_idx, end_idx))

        if not candidates:
            continue

        orient, perm, start_idx, end_idx = random.choice(candidates)
        return specs, orient, perm, start_idx, end_idx
    return None


def _chosen_endpoint_outer_count(blocks: list[np.ndarray], orient: str, perm: tuple[int, ...]) -> int:
    contacts = [_edge_sets(block, BG) for block in blocks]
    first = int(perm[0])
    last = int(perm[-1])
    if orient == "h":
        return int(bool(contacts[first]["left"])) + int(bool(contacts[last]["right"]))
    return int(bool(contacts[first]["top"])) + int(bool(contacts[last]["bot"]))


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    spec_choice = _search_specs()
    if spec_choice is None:
        return None
    specs, orient, chosen_perm, start_idx, end_idx = spec_choice
    n = len(specs)

    blocks: list[np.ndarray] = []
    colors = random.choices(PALETTE, k=n)
    for i in range(n):
        block = _make_block(
            colors[i],
            left=specs[i]["left"],
            right=specs[i]["right"],
            top=specs[i]["top"],
            bottom=specs[i]["bot"],
        )
        if block is None:
            return None
        blocks.append(block)

    blocks[start_idx] = _mark_block(blocks[start_idx], START_MARKER)
    blocks[end_idx] = _mark_block(blocks[end_idx], END_MARKER)

    orders_chosen = _valid_orders(blocks, orient, BG)
    orders_other = _valid_orders(blocks, OTHER[orient], BG)
    orders_restricted = _valid_orders(blocks, orient, BG, start_idx=start_idx, end_idx=end_idx)

    if len(orders_chosen) <= 1:
        return None
    if len(orders_restricted) != 1 or tuple(orders_restricted[0]) != tuple(chosen_perm):
        return None
    if not orders_other:
        return None
    if _chosen_endpoint_outer_count(blocks, orient, chosen_perm) != 2:
        return None

    out = _render_strip([blocks[i] for i in chosen_perm], orient, BG, SEP)
    core = _render_input_core(blocks, orient, chosen_perm)
    inp = _add_orient_marker(core, orient)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None

    meta = {
        "n": int(n),
        "chosen_orders": int(len(orders_chosen)),
        "other_orders": int(len(orders_other)),
        "restricted_orders": int(len(orders_restricted)),
        "endpoint_outer_count": int(_chosen_endpoint_outer_count(blocks, orient, chosen_perm)),
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError(f"failed to generate a valid combo sample after {MAX_RETRIES} attempts")
