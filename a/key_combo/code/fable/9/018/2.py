from __future__ import annotations

from collections import Counter, deque
from itertools import combinations, product
import random

import numpy as np


SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

BG = 0
PORT = 1
FRAME = 5
FILL_POOL = (2, 3, 4, 6, 7, 8, 9)

SIZE_CHOICES = (7, 8)
BLOCK_COUNT_CHOICES = (4, 5, 6)
MAX_GENERATION_ATTEMPTS = 500


def _mode(values) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _edge_pattern(patch: np.ndarray, side: str, port_color: int) -> tuple[int, ...]:
    size = patch.shape[0]
    if side == "top":
        line = patch[0, 1 : size - 1]
    elif side == "bottom":
        line = patch[size - 1, 1 : size - 1]
    elif side == "left":
        line = patch[1 : size - 1, 0]
    elif side == "right":
        line = patch[1 : size - 1, size - 1]
    else:
        raise ValueError(f"bad side {side}")
    return tuple(i + 1 for i, val in enumerate(line) if int(val) == int(port_color))


def _paint_edge_pattern(patch: np.ndarray, side: str, pattern: tuple[int, ...], port_color: int) -> None:
    size = patch.shape[0]
    for off in pattern:
        if side == "top":
            patch[0, off] = port_color
        elif side == "bottom":
            patch[size - 1, off] = port_color
        elif side == "left":
            patch[off, 0] = port_color
        elif side == "right":
            patch[off, size - 1] = port_color
        else:
            raise ValueError(f"bad side {side}")


def _parse_blocks(grid: np.ndarray) -> tuple[int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode(g.reshape(-1))
    comps = _components(g != bg)
    assert len(comps) >= 3, "expected several framed blocks"

    prepared = []
    border_colors = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h >= 5, "every block must be a square frame"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h)
        assert np.all(patch[border] != bg), "frame border must be fully present"
        prepared.append({"bbox": (r0, c0, r1, c1), "patch": patch.copy()})
        border_colors.extend(map(int, patch[border].reshape(-1)))

    frame_color = _mode(border_colors)
    port_candidates = sorted(set(border_colors) - {frame_color})
    assert len(port_candidates) == 1, "expected one shared interface color"
    port_color = int(port_candidates[0])

    sizes = {item["patch"].shape[0] for item in prepared}
    assert len(sizes) == 1, "all blocks must have the same size"
    size = next(iter(sizes))

    blocks = []
    hole_count = 0
    for item in prepared:
        patch = item["patch"]
        border = _border_mask(size)
        assert set(map(int, np.unique(patch[border]))) <= {frame_color, port_color}
        interior = patch[1 : size - 1, 1 : size - 1]
        interior_colors = set(map(int, np.unique(interior))) - {bg}
        assert len(interior_colors) <= 1, "each frame interior must be uniform or empty"
        fill = bg if not interior_colors else int(next(iter(interior_colors)))
        if fill == bg:
            assert np.all(interior == bg), "the hole block must be fully empty"
            hole_count += 1
        else:
            assert np.all(interior == fill), "filled interiors must be solid"
        blocks.append(
            {
                "bbox": item["bbox"],
                "fill": fill,
                "patterns": {side: _edge_pattern(patch, side, port_color) for side in SIDES},
            }
        )

    assert hole_count == 1, "expected exactly one empty block"
    return bg, frame_color, port_color, blocks


def _build_chain(blocks: list[dict]) -> tuple[list[tuple[int, int]], list[list[int]], int]:
    n = len(blocks)
    matches: list[tuple[int, int]] = []
    side_use: set[tuple[int, str]] = set()
    pair_hits: set[tuple[int, int]] = set()
    adj = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            pair_count = 0
            for side in SIDES:
                pat = blocks[i]["patterns"][side]
                if pat and pat == blocks[j]["patterns"][OPPOSITE[side]]:
                    pair_count += 1
                    assert (i, side) not in side_use and (j, OPPOSITE[side]) not in side_use
                    side_use.add((i, side))
                    side_use.add((j, OPPOSITE[side]))
                    adj[i].append(j)
                    adj[j].append(i)
                    matches.append((i, j))
            assert pair_count <= 1, "a pair of blocks may match along only one edge pair"
            if pair_count:
                pair_hits.add((i, j))

    assert len(matches) == n - 1, "interface matches must form one chain"
    assert len(pair_hits) == n - 1, "chain edges must be unique"

    degrees = [len(nei) for nei in adj]
    endpoints = [i for i, deg in enumerate(degrees) if deg == 1]
    assert len(endpoints) == 2, "a chain must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees), "every block must have chain degree 1 or 2"
    return matches, adj, endpoints[0]


def _chain_order_from_hole(blocks: list[dict], bg: int) -> list[int]:
    _matches, adj, _some_endpoint = _build_chain(blocks)
    hole = [i for i, block in enumerate(blocks) if int(block["fill"]) == int(bg)]
    assert len(hole) == 1
    hole_idx = int(hole[0])
    assert len(adj[hole_idx]) == 1, "the empty block must lie at a chain endpoint"

    order = [hole_idx]
    prev = -1
    cur = hole_idx
    while True:
        nxt = [nb for nb in adj[cur] if nb != prev]
        if not nxt:
            break
        assert len(nxt) == 1
        prev, cur = cur, int(nxt[0])
        order.append(cur)
    assert len(order) == len(blocks), "chain traversal must visit every block exactly once"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift the solid interior colors along the hidden port-matching chain toward the empty framed block.

    Input structure:
    - The background contains several disjoint square framed blocks of equal size.
    - Each frame border is mostly one frame color, but each edge also carries some interface-color pixels.
    - Every block interior is a solid color except for exactly one empty block whose interior is background.

    Rule:
    - Two blocks are chain neighbors when one edge's interface-pixel offset pattern exactly matches the opposite edge pattern of the other block.
    - These matches form one linear chain; the empty block is one endpoint of that chain.
    - Keep every frame, interface pixel, and block position fixed, and shift the interior colors one block toward the empty endpoint so the hole reappears at the opposite endpoint.

    Key insight:
    - Physical proximity is a decoy: only exact edge-offset matches reveal the chain order that controls the color shift.
    """
    g = np.asarray(grid, dtype=int)
    bg, _frame_color, _port_color, blocks = _parse_blocks(g)
    order = _chain_order_from_hole(blocks, bg)

    out = g.copy()
    shifted_fills = [blocks[idx]["fill"] for idx in order[1:]] + [bg]
    for idx, fill in zip(order, shifted_fills):
        r0, c0, r1, c1 = blocks[idx]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(fill)
    return out


def _all_patterns(size: int) -> list[tuple[int, ...]]:
    offsets = list(range(1, size - 1))
    out = []
    for count in range(1, len(offsets) + 1):
        for combo in combinations(offsets, count):
            out.append(tuple(int(x) for x in combo))
    return out


def _make_chain_sides(n_blocks: int) -> list[str]:
    sides = [random.choice(SIDES)]
    incoming = OPPOSITE[sides[0]]
    for _ in range(1, n_blocks - 1):
        choices = [side for side in SIDES if side != incoming]
        nxt = random.choice(choices)
        sides.append(nxt)
        incoming = OPPOSITE[nxt]
    return sides


def _make_edge_patterns(n_blocks: int, size: int) -> list[dict[str, tuple[int, ...]]]:
    all_patterns = _all_patterns(size)
    chain_patterns = random.sample(all_patterns, n_blocks - 1)
    used = set(chain_patterns)

    edge_patterns = [{side: () for side in SIDES} for _ in range(n_blocks)]
    chain_sides = _make_chain_sides(n_blocks)
    for i, side in enumerate(chain_sides):
        pattern = chain_patterns[i]
        edge_patterns[i][side] = pattern
        edge_patterns[i + 1][OPPOSITE[side]] = pattern

    remaining_sides = [(i, side) for i in range(n_blocks) for side in SIDES if not edge_patterns[i][side]]
    decoys = random.sample([pat for pat in all_patterns if pat not in used], len(remaining_sides))
    for (i, side), pattern in zip(remaining_sides, decoys):
        edge_patterns[i][side] = pattern
    return edge_patterns


def _make_patch(size: int, fill: int, edge_patterns: dict[str, tuple[int, ...]]) -> np.ndarray:
    patch = np.full((size, size), fill, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    if fill == BG:
        patch[1 : size - 1, 1 : size - 1] = BG
    for side, pattern in edge_patterns.items():
        _paint_edge_pattern(patch, side, pattern, PORT)
    return patch


def _sample_axis_positions(count: int, size: int, limit: int) -> list[int] | None:
    min_gap = 2
    starts = list(range(1, limit - size))
    for _ in range(200):
        picks = sorted(random.sample(starts, count))
        if all(picks[i + 1] - picks[i] >= size + min_gap for i in range(count - 1)):
            return picks
    return None


def _placement_boxes(top_lefts: list[tuple[int, int]], size: int) -> list[tuple[int, int, int, int]]:
    return [(r, c, r + size - 1, c + size - 1) for r, c in top_lefts]


def _boxes_with_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _nearest_neighbor_decoy(
    top_lefts: list[tuple[int, int]],
    chain_edges: set[tuple[int, int]],
    size: int,
    chain_order: list[int],
) -> bool:
    centers = [(r + (size - 1) / 2.0, c + (size - 1) / 2.0) for r, c in top_lefts]
    ok = False
    for i, (r0, c0) in enumerate(centers):
        dists = []
        for j, (r1, c1) in enumerate(centers):
            if i == j:
                continue
            dist = abs(r0 - r1) + abs(c0 - c1)
            dists.append((dist, j))
        dists.sort()
        nearest_dist = dists[0][0]
        nearest_group = [int(j) for dist, j in dists if dist == nearest_dist]
        if any(tuple(sorted((i, j))) not in chain_edges for j in nearest_group):
            ok = True
            break
    if not ok:
        return False

    xs = sorted(range(len(top_lefts)), key=lambda i: (top_lefts[i][1], top_lefts[i][0]))
    ys = sorted(range(len(top_lefts)), key=lambda i: (top_lefts[i][0], top_lefts[i][1]))
    if xs == chain_order or xs == chain_order[::-1]:
        return False
    if ys == chain_order or ys == chain_order[::-1]:
        return False
    return True


def _place_blocks(n_blocks: int, size: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    layout_choices = [(2, 3), (3, 2), (2, 2)]
    random.shuffle(layout_choices)
    chain_edges = {tuple(sorted((i, i + 1))) for i in range(n_blocks - 1)}

    for rows, cols in layout_choices:
        if rows * cols < n_blocks:
            continue
        min_h = rows * size + (rows - 1) * 2 + 2
        min_w = cols * size + (cols - 1) * 2 + 2
        if min_h > 30 or min_w > 30:
            continue
        for _ in range(120):
            h = random.randint(min_h, 30)
            w = random.randint(min_w, 30)
            row_starts = _sample_axis_positions(rows, size, h)
            col_starts = _sample_axis_positions(cols, size, w)
            if row_starts is None or col_starts is None:
                continue
            slots = list(product(row_starts, col_starts))
            random.shuffle(slots)
            top_lefts = list(slots[:n_blocks])
            boxes = _placement_boxes(top_lefts, size)
            if any(
                not _boxes_with_gap(boxes[i], boxes[j], gap=1)
                for i in range(n_blocks)
                for j in range(i + 1, n_blocks)
            ):
                continue
            if _nearest_neighbor_decoy(top_lefts, chain_edges, size, list(range(n_blocks))):
                return h, w, top_lefts
    return None


def _expected_output(input_grid: np.ndarray, order: list[int], top_lefts: list[tuple[int, int]], size: int, fills: list[int]) -> np.ndarray:
    out = input_grid.copy()
    shifted = fills[1:] + [BG]
    for idx, fill in zip(order, shifted):
        r, c = top_lefts[idx]
        out[r + 1 : r + size - 1, c + 1 : c + size - 1] = int(fill)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_blocks = random.choice(BLOCK_COUNT_CHOICES)
    size = random.choice(SIZE_CHOICES)
    placed = _place_blocks(n_blocks, size)
    if placed is None:
        return None
    height, width, top_lefts = placed

    fills = [BG] + random.sample(FILL_POOL, n_blocks - 1)
    edge_patterns = _make_edge_patterns(n_blocks, size)
    grid = np.full((height, width), BG, dtype=int)
    for idx in range(n_blocks):
        r, c = top_lefts[idx]
        grid[r : r + size, c : c + size] = _make_patch(size, fills[idx], edge_patterns[idx])

    expected = _expected_output(grid, list(range(n_blocks)), top_lefts, size, fills)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    bg, _frame_color, _port_color, blocks = _parse_blocks(grid)
    if bg != BG:
        return None
    order = _chain_order_from_hole(blocks, bg)
    bbox_to_abstract = {(r, c, r + size - 1, c + size - 1): idx for idx, (r, c) in enumerate(top_lefts)}
    recovered_abstract = [bbox_to_abstract[blocks[idx]["bbox"]] for idx in order]
    if recovered_abstract != list(range(n_blocks)):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-chain hole-shift instance")
