from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
FG = 5
BLOCK_SIZE = 5
LEGEND_HEIGHT = 3
CHAIN_GAP = 1
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 600
MAX_MASK_ATTEMPTS = 120


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_rows(block: np.ndarray, side: str) -> set[int]:
    if side == "left":
        return set(np.where(block[:, 0] != BG)[0].tolist())
    if side == "right":
        return set(np.where(block[:, -1] != BG)[0].tolist())
    raise ValueError(f"unknown side: {side}")


def _valid_chain_orders(blocks: list[np.ndarray]) -> list[tuple[int, ...]]:
    left_edges = [_edge_rows(block, "left") for block in blocks]
    right_edges = [_edge_rows(block, "right") for block in blocks]

    def links(i: int, j: int) -> bool:
        return i != j and bool(right_edges[i] & left_edges[j])

    orders = []
    for perm in itertools.permutations(range(len(blocks))):
        if all(links(perm[k], perm[k + 1]) for k in range(len(perm) - 1)):
            orders.append(tuple(int(x) for x in perm))
    return orders


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    color_mask = (grid != BG) & (grid != FG)
    box = _bbox(color_mask)
    assert box is not None, "legend not found"
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == LEGEND_HEIGHT, "legend must be 3 rows tall"
    assert 4 <= w <= 5, "legend width must match 4-5 colors"
    assert int(color_mask.sum()) == h * w, "legend must be a solid rectangle"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = []
    for c in range(w):
        col = patch[:, c]
        assert np.all(col == col[0]), "legend columns must be uniform"
        color = int(col[0])
        assert color not in {BG, FG}, "legend colors must differ from bg and gray"
        colors.append(color)
    assert len(set(colors)) == len(colors), "legend colors must be distinct"
    return box, colors


def _find_block_strip(
    grid: np.ndarray, legend_bottom: int, block_count: int
) -> tuple[int, int, list[np.ndarray]]:
    lower_nonzero = np.argwhere(grid[legend_bottom + 2 :, :] != BG)
    assert lower_nonzero.size > 0, "block strip missing"
    lower_nonzero[:, 0] += legend_bottom + 2

    candidates: list[tuple[int, int, list[np.ndarray]]] = []
    H, W = grid.shape
    for top in range(legend_bottom + 2, H - BLOCK_SIZE + 1):
        band = grid[top : top + BLOCK_SIZE, :]
        sep_cols = [c for c in range(W) if np.all(band[:, c] == FG)]
        if len(sep_cols) < block_count - 1:
            continue

        for idx in range(len(sep_cols) - block_count + 2):
            window = sep_cols[idx : idx + block_count - 1]
            if any(window[k + 1] - window[k] != BLOCK_SIZE + 1 for k in range(len(window) - 1)):
                continue

            start = window[0] - BLOCK_SIZE
            width = block_count * BLOCK_SIZE + (block_count - 1)
            end = start + width
            if start < 0 or end > W:
                continue
            if any(window[k] != start + BLOCK_SIZE + k * (BLOCK_SIZE + 1) for k in range(len(window))):
                continue

            rows = lower_nonzero[:, 0]
            cols = lower_nonzero[:, 1]
            if np.any(rows < top) or np.any(rows >= top + BLOCK_SIZE):
                continue
            if np.any(cols < start) or np.any(cols >= end):
                continue

            blocks = []
            ok = True
            for i in range(block_count):
                c0 = start + i * (BLOCK_SIZE + 1)
                block = band[:, c0 : c0 + BLOCK_SIZE].copy()
                if block.shape != (BLOCK_SIZE, BLOCK_SIZE):
                    ok = False
                    break
                if np.any((block != BG) & (block != FG)):
                    ok = False
                    break
                if not np.any(block == FG):
                    ok = False
                    break
                blocks.append(block)
            if ok:
                candidates.append((top, start, blocks))

    assert len(candidates) == 1, "block strip must be uniquely locatable"
    return candidates[0]


def _render_output(blocks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    width = len(blocks) * BLOCK_SIZE + (len(blocks) - 1)
    out = np.full((BLOCK_SIZE, width), BG, dtype=int)
    for i, (block, color) in enumerate(zip(blocks, colors)):
        c0 = i * (BLOCK_SIZE + 1)
        recolored = np.where(block == FG, int(color), BG)
        out[:, c0 : c0 + BLOCK_SIZE] = recolored
        if i < len(blocks) - 1:
            out[:, c0 + BLOCK_SIZE] = FG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor and reorder gray sprite blocks by matching edge contacts to the legend order.

    Input structure:
    - The background is black.
    - Near the top sits a 3-row horizontal legend made of adjacent single-column color bars,
      each a distinct non-black, non-gray color.
    - One black spacer row separates the legend from a horizontal strip of gray 5x5 sprite
      blocks, with gray single-column separators between neighboring blocks.

    Rule:
    - Read the legend colors from left to right.
    - For every ordered pair of blocks A, B, treat A -> B as valid when A has a gray pixel on
      its right edge and B has a gray pixel on its left edge at some shared row offset.
    - Find the unique full chain whose consecutive pairs satisfy that edge-contact rule.
    - Remove the legend, reorder the blocks into chain order, and recolor each block's gray
      foreground to the correspondingly ranked legend color while leaving black background and
      gray separators unchanged.

    Key insight:
    - The gray edge overlaps determine the only valid block order, and that recovered order is
      exactly what the legend colors index.
    """
    g = np.asarray(grid, dtype=int)
    legend_box, colors = _find_legend(g)
    _top, _left, legend_bottom, _right = legend_box
    _strip_top, _strip_left, blocks = _find_block_strip(g, legend_bottom, len(colors))

    orders = _valid_chain_orders(blocks)
    assert len(orders) == 1, "edge matching must yield one chain"
    ordered_blocks = [blocks[i] for i in orders[0]]
    return _render_output(ordered_blocks, colors)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BLOCK_SIZE and 0 <= nc < BLOCK_SIZE and mask[nr, nc]:
                node = (nr, nc)
                if node not in seen:
                    seen.add(node)
                    stack.append(node)
    return len(seen) == int(mask.sum())


def _interior_frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return [(r, c) for r in range(BLOCK_SIZE) for c in range(1, BLOCK_SIZE - 1)]

    candidates = []
    for r in range(BLOCK_SIZE):
        for c in range(1, BLOCK_SIZE - 1):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r + 1 < BLOCK_SIZE and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c + 1 < BLOCK_SIZE and mask[r, c + 1])
            ):
                candidates.append((r, c))
    return candidates


def _make_block_mask(left_rows: set[int], right_rows: set[int]) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
        for r in left_rows:
            mask[int(r), 0] = True
        for r in right_rows:
            mask[int(r), BLOCK_SIZE - 1] = True

        target_area = random.randint(max(5, len(left_rows) + len(right_rows) + 2), 11)
        for _grow in range(40):
            if mask.sum() >= target_area and _is_connected(mask):
                break
            frontier = _interior_frontier(mask)
            if not frontier:
                break
            r, c = random.choice(frontier)
            mask[r, c] = True

        if not _is_connected(mask):
            continue
        if _edge_rows(np.where(mask, FG, BG), "left") != left_rows:
            continue
        if _edge_rows(np.where(mask, FG, BG), "right") != right_rows:
            continue
        if any(np.all(mask[:, c]) for c in range(BLOCK_SIZE)):
            continue
        if mask.sum() < 5:
            continue
        return np.where(mask, FG, BG).astype(int)
    return None


def _sample_edge_sets(n: int) -> list[tuple[set[int], set[int]]] | None:
    link_rows = [random.randrange(BLOCK_SIZE) for _ in range(n - 1)]
    if len(set(link_rows)) < 2:
        return None

    edge_sets: list[tuple[set[int], set[int]]] = []
    used_multi = False
    for i in range(n):
        left_rows = set()
        right_rows = set()
        if i > 0:
            left_rows.add(link_rows[i - 1])
        if i + 1 < n:
            right_rows.add(link_rows[i])

        for side_rows in (left_rows, right_rows):
            available = [r for r in range(BLOCK_SIZE) if r not in side_rows]
            extra_cap = min(3 - len(side_rows), len(available))
            extra_count = random.choices(
                population=list(range(extra_cap + 1)),
                weights=[5, 3, 2, 1][: extra_cap + 1],
                k=1,
            )[0]
            if extra_count:
                side_rows.update(random.sample(available, extra_count))
                used_multi = True

        if len(left_rows) > 3 or len(right_rows) > 3:
            return None
        edge_sets.append((left_rows, right_rows))

    if not used_multi:
        return None
    return edge_sets


def _build_chain_blocks(n: int) -> list[np.ndarray] | None:
    sampled = _sample_edge_sets(n)
    if sampled is None:
        return None

    blocks = []
    seen = set()
    for left_rows, right_rows in sampled:
        block = _make_block_mask(left_rows, right_rows)
        if block is None:
            return None
        key = block.tobytes()
        if key in seen:
            return None
        seen.add(key)
        blocks.append(block)

    if _valid_chain_orders(blocks) != [tuple(range(n))]:
        return None
    return blocks


def _scramble_blocks(blocks: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[int, ...]] | None:
    n = len(blocks)
    indices = list(range(n))
    for _ in range(80):
        perm = indices[:]
        random.shuffle(perm)
        scrambled = [blocks[i] for i in perm]
        orders = _valid_chain_orders(scrambled)
        if len(orders) == 1 and orders[0] != tuple(range(n)):
            return scrambled, orders[0]
    return None


def _build_input_grid(scrambled_blocks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    n = len(scrambled_blocks)
    strip_width = n * BLOCK_SIZE + (n - 1)
    max_extra_w = 30 - strip_width
    extra_w = random.randint(0, max_extra_w)
    grid_w = strip_width + extra_w

    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 3)
    grid_h = top_margin + LEGEND_HEIGHT + CHAIN_GAP + BLOCK_SIZE + bottom_margin

    legend_left = random.randint(0, grid_w - n)
    strip_left = random.randint(0, grid_w - strip_width)
    legend_top = top_margin
    strip_top = legend_top + LEGEND_HEIGHT + CHAIN_GAP

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    legend = np.tile(np.array(colors, dtype=int), (LEGEND_HEIGHT, 1))
    grid[legend_top : legend_top + LEGEND_HEIGHT, legend_left : legend_left + n] = legend

    for i, block in enumerate(scrambled_blocks):
        c0 = strip_left + i * (BLOCK_SIZE + 1)
        grid[strip_top : strip_top + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = block
        if i < n - 1:
            grid[strip_top : strip_top + BLOCK_SIZE, c0 + BLOCK_SIZE] = FG
    return grid


def _validate_case(
    inp: np.ndarray,
    scrambled_blocks: list[np.ndarray],
    chain_order: tuple[int, ...],
    colors: list[int],
) -> bool:
    if chain_order == tuple(range(len(scrambled_blocks))):
        return False

    expected = _render_output([scrambled_blocks[i] for i in chain_order], colors)
    if np.array_equal(inp, expected):
        return False

    legend_box, found_colors = _find_legend(inp)
    if found_colors != colors:
        return False

    _top, _left, legend_bottom, _right = legend_box
    _strip_top, _strip_left, found_blocks = _find_block_strip(inp, legend_bottom, len(colors))
    if len(found_blocks) != len(scrambled_blocks):
        return False
    if any(not np.array_equal(a, b) for a, b in zip(found_blocks, scrambled_blocks)):
        return False

    if _valid_chain_orders(found_blocks) != [chain_order]:
        return False

    if not np.array_equal(main(inp), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = random.choice([4, 5])
        blocks = _build_chain_blocks(n)
        if blocks is None:
            continue

        scrambled = _scramble_blocks(blocks)
        if scrambled is None:
            continue
        scrambled_blocks, chain_order = scrambled

        colors = random.sample(PALETTE, n)
        inp = _build_input_grid(scrambled_blocks, colors)
        if _validate_case(inp, scrambled_blocks, chain_order, colors):
            return inp

    raise RuntimeError("failed to generate a valid edge-chain legend case")
