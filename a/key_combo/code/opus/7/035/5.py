from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
ORANGE = 7
PALETTE_COLORS = (1, 2, 3, 4, 6)
GRID_SIZE = 15
PALETTE_SIZE = 3
MAIN_OFFSET = 3
BLOCK_SIZE = 4
BLOCK_ROWS = 3
BLOCK_COLS = 3


def _block_neighbors(block: tuple[int, int]) -> list[tuple[int, int]]:
    br, bc = block
    out: list[tuple[int, int]] = []
    if br > 0:
        out.append((br - 1, bc))
    if br + 1 < BLOCK_ROWS:
        out.append((br + 1, bc))
    if bc > 0:
        out.append((br, bc - 1))
    if bc + 1 < BLOCK_COLS:
        out.append((br, bc + 1))
    return out


def _border_sides(block: tuple[int, int]) -> list[str]:
    br, bc = block
    sides: list[str] = []
    if br == 0:
        sides.append("N")
    if br == BLOCK_ROWS - 1:
        sides.append("S")
    if bc == 0:
        sides.append("W")
    if bc == BLOCK_COLS - 1:
        sides.append("E")
    return sides


def _grow_connected_set(
    rng: random.Random,
    start: tuple[int, int],
    size: int,
    allowed: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[frozenset[tuple[int, int]]]] | None:
    chosen = {start}
    edges: set[frozenset[tuple[int, int]]] = set()
    while len(chosen) < size:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for block in chosen:
            for nbr in _block_neighbors(block):
                if nbr in allowed and nbr not in chosen:
                    frontier.append((block, nbr))
        if not frontier:
            return None
        parent, nxt = rng.choice(frontier)
        chosen.add(nxt)
        edges.add(frozenset((parent, nxt)))
    return chosen, edges


def _add_extra_edges(
    rng: random.Random,
    blocks: set[tuple[int, int]],
    edges: set[frozenset[tuple[int, int]]],
    prob: float,
) -> None:
    for block in blocks:
        for nbr in _block_neighbors(block):
            if nbr not in blocks or nbr < block:
                continue
            edge = frozenset((block, nbr))
            if edge not in edges and rng.random() < prob:
                edges.add(edge)


def _trace_reachable_orange(main_area: np.ndarray) -> np.ndarray:
    orange = main_area == ORANGE
    seen = np.zeros_like(orange, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    h, w = orange.shape
    for r in range(h):
        for c in range(w):
            if not orange[r, c]:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and orange[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _reachable_blocks(main_area: np.ndarray) -> np.ndarray:
    reachable_orange = _trace_reachable_orange(main_area)
    out = np.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=bool)
    for br in range(BLOCK_ROWS):
        for bc in range(BLOCK_COLS):
            r0 = br * BLOCK_SIZE
            c0 = bc * BLOCK_SIZE
            out[br, bc] = bool(
                reachable_orange[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE].any()
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the 3x3 reference palette only into those 4x4 maze blocks whose orange channel is connected to the outer edge.

    Input structure:
    - The input is a 15x15 grid with a 3x3 non-black reference palette in the top-left corner.
    - The lower-right 12x12 main region is partitioned into a 3x3 array of 4x4 blocks.
    - Inside the main region, black background and orange channel pixels define which blocks are externally reachable.

    Rule:
    - Restrict attention to the 12x12 main region and find all orange pixels that are 4-connected to any orange pixel on that region's outer border.
    - For each 4x4 block (br, bc), inspect the corresponding reference color at palette position (br, bc).
    - If the block contains any border-reachable orange pixel, fill the entire 4x4 block with that reference color; otherwise fill the block with black.

    Key insight:
    - The palette supplies colors, but the orange maze topology decides which palette positions survive into the output, so adjacency between blocks is not enough without tracing true channel connectivity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected the fixed 15x15 puzzle layout"

    palette = g[:PALETTE_SIZE, :PALETTE_SIZE]
    main_area = g[MAIN_OFFSET:, MAIN_OFFSET:]
    assert palette.shape == (PALETTE_SIZE, PALETTE_SIZE)
    assert main_area.shape == (BLOCK_ROWS * BLOCK_SIZE, BLOCK_COLS * BLOCK_SIZE)

    reachable = _reachable_blocks(main_area)
    out = np.zeros_like(main_area)
    for br in range(BLOCK_ROWS):
        for bc in range(BLOCK_COLS):
            r0 = br * BLOCK_SIZE
            c0 = bc * BLOCK_SIZE
            color = int(palette[br, bc]) if reachable[br, bc] else BLACK
            out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = color
    return out


def _draw_local_path(
    main_area: np.ndarray,
    block: tuple[int, int],
    sides: set[str],
    row_anchor: list[int],
    col_anchor: list[int],
) -> None:
    br, bc = block
    r0 = br * BLOCK_SIZE
    c0 = bc * BLOCK_SIZE
    hr = row_anchor[br]
    hc = col_anchor[bc]
    main_area[r0 + hr, c0 + hc] = ORANGE
    for side in sides:
        if side == "N":
            for rr in range(0, hr + 1):
                main_area[r0 + rr, c0 + hc] = ORANGE
        elif side == "S":
            for rr in range(hr, BLOCK_SIZE):
                main_area[r0 + rr, c0 + hc] = ORANGE
        elif side == "W":
            for cc in range(0, hc + 1):
                main_area[r0 + hr, c0 + cc] = ORANGE
        elif side == "E":
            for cc in range(hc, BLOCK_SIZE):
                main_area[r0 + hr, c0 + cc] = ORANGE


def _draw_isolated_stub(
    main_area: np.ndarray,
    block: tuple[int, int],
    row_anchor: list[int],
    col_anchor: list[int],
) -> None:
    br, bc = block
    r0 = br * BLOCK_SIZE
    c0 = bc * BLOCK_SIZE
    hr = row_anchor[br]
    hc = col_anchor[bc]
    main_area[r0 + hr, c0 + hc] = ORANGE
    rr = 2 if hr == 1 else 1
    main_area[r0 + rr, c0 + hc] = ORANGE


def _render_main_area(
    reachable_blocks: set[tuple[int, int]],
    unreachable_blocks: set[tuple[int, int]],
    edges: set[frozenset[tuple[int, int]]],
    border_contacts: dict[tuple[int, int], set[str]],
    row_anchor: list[int],
    col_anchor: list[int],
) -> np.ndarray:
    main_area = np.zeros((BLOCK_ROWS * BLOCK_SIZE, BLOCK_COLS * BLOCK_SIZE), dtype=int)
    sides_by_block: dict[tuple[int, int], set[str]] = {}
    for edge in edges:
        a, b = tuple(edge)
        (ar, ac), (br, bc) = a, b
        sa = sides_by_block.setdefault(a, set())
        sb = sides_by_block.setdefault(b, set())
        if ar == br:
            if ac < bc:
                sa.add("E")
                sb.add("W")
            else:
                sa.add("W")
                sb.add("E")
        else:
            if ar < br:
                sa.add("S")
                sb.add("N")
            else:
                sa.add("N")
                sb.add("S")
    for block, sides in border_contacts.items():
        sides_by_block.setdefault(block, set()).update(sides)

    for block in reachable_blocks | unreachable_blocks:
        sides = sides_by_block.get(block, set())
        if sides:
            _draw_local_path(main_area, block, sides, row_anchor, col_anchor)
        else:
            _draw_isolated_stub(main_area, block, row_anchor, col_anchor)
    return main_area


def _make_palette(rng: random.Random) -> np.ndarray:
    n_colors = rng.randint(3, 5)
    chosen = rng.sample(PALETTE_COLORS, n_colors)
    values = chosen[:]
    while len(values) < PALETTE_SIZE * PALETTE_SIZE:
        values.append(rng.choice(chosen))
    rng.shuffle(values)
    return np.array(values, dtype=int).reshape(PALETTE_SIZE, PALETTE_SIZE)


def _adjacent_difference(mask: np.ndarray) -> bool:
    for br in range(BLOCK_ROWS):
        for bc in range(BLOCK_COLS):
            if br + 1 < BLOCK_ROWS and mask[br, bc] != mask[br + 1, bc]:
                return True
            if bc + 1 < BLOCK_COLS and mask[br, bc] != mask[br, bc + 1]:
                return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    all_blocks = {(br, bc) for br in range(BLOCK_ROWS) for bc in range(BLOCK_COLS)}
    border_blocks = [block for block in all_blocks if _border_sides(block)]
    start_r = rng.choice(border_blocks)
    reachable_size = rng.randint(3, 7)
    reachable_part = _grow_connected_set(rng, start_r, reachable_size, all_blocks)
    if reachable_part is None:
        return None
    reachable_blocks, reachable_edges = reachable_part

    remaining = all_blocks - reachable_blocks
    if len(remaining) < 2:
        return None
    adjacent_unreachable = sorted(
        block for block in remaining if any(nbr in reachable_blocks for nbr in _block_neighbors(block))
    )
    if not adjacent_unreachable:
        return None

    start_u = rng.choice(adjacent_unreachable)
    unreachable_size = rng.randint(1, min(3, len(remaining)))
    unreachable_part = _grow_connected_set(rng, start_u, unreachable_size, remaining)
    if unreachable_part is None:
        return None
    unreachable_blocks, unreachable_edges = unreachable_part

    _add_extra_edges(rng, reachable_blocks, reachable_edges, prob=0.35)
    _add_extra_edges(rng, unreachable_blocks, unreachable_edges, prob=0.2)

    border_options = [
        (block, side)
        for block in sorted(reachable_blocks)
        for side in _border_sides(block)
    ]
    if not border_options:
        return None
    rng.shuffle(border_options)
    n_contacts = rng.randint(1, min(3, len(border_options)))
    border_contacts: dict[tuple[int, int], set[str]] = {}
    for block, side in border_options[:n_contacts]:
        border_contacts.setdefault(block, set()).add(side)

    row_anchor = [rng.choice((1, 2)) for _ in range(BLOCK_ROWS)]
    col_anchor = [rng.choice((1, 2)) for _ in range(BLOCK_COLS)]

    main_area = _render_main_area(
        reachable_blocks,
        unreachable_blocks,
        reachable_edges | unreachable_edges,
        border_contacts,
        row_anchor,
        col_anchor,
    )
    reachable_mask = _reachable_blocks(main_area)
    if int(reachable_mask.sum()) < 3:
        return None
    if int((~reachable_mask).sum()) < 2:
        return None
    if not _adjacent_difference(reachable_mask):
        return None
    if not any(block in unreachable_blocks and not reachable_mask[block] for block in unreachable_blocks):
        return None

    palette = _make_palette(rng)
    reachable_colors = {
        int(palette[br, bc])
        for br in range(BLOCK_ROWS)
        for bc in range(BLOCK_COLS)
        if reachable_mask[br, bc]
    }
    if len(reachable_colors) < 2:
        return None

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[:PALETTE_SIZE, :PALETTE_SIZE] = palette
    grid[MAIN_OFFSET:, MAIN_OFFSET:] = main_area

    expected = np.zeros((BLOCK_ROWS * BLOCK_SIZE, BLOCK_COLS * BLOCK_SIZE), dtype=int)
    for br in range(BLOCK_ROWS):
        for bc in range(BLOCK_COLS):
            if reachable_mask[br, bc]:
                expected[
                    br * BLOCK_SIZE : (br + 1) * BLOCK_SIZE,
                    bc * BLOCK_SIZE : (bc + 1) * BLOCK_SIZE,
                ] = int(palette[br, bc])
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(expected, np.zeros_like(expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
