from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BLOCK_SIZE = 5
SEP_SIZE = 1
MAX_GENERATION_ATTEMPTS = 400
CHAIN_SIGS = [
    tuple(i for i in range(1, BLOCK_SIZE - 1) if (mask >> (i - 1)) & 1)
    for mask in range(1, 1 << (BLOCK_SIZE - 2))
]
INTERIOR_CELLS = [(r, c) for r in range(1, BLOCK_SIZE - 1) for c in range(1, BLOCK_SIZE - 1)]
LAYOUTS = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
}


def _non_separator_segments(length: int, separator_indices: list[int]) -> list[list[int]]:
    separators = set(separator_indices)
    segments: list[list[int]] = []
    current: list[int] = []
    for idx in range(length):
        if idx in separators:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(idx)
    if current:
        segments.append(current)
    return segments


def _parse_panels(grid: np.ndarray) -> tuple[int, list[list[int]], list[list[int]], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(x) for x in np.unique(g)]
    candidates = []
    for sep in colors:
        full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == sep)]
        full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == sep)]
        row_segments = _non_separator_segments(g.shape[0], full_rows)
        col_segments = _non_separator_segments(g.shape[1], full_cols)
        if len(row_segments) < 2 or len(col_segments) < 2:
            continue
        if not all(len(seg) == BLOCK_SIZE for seg in row_segments):
            continue
        if not all(len(seg) == BLOCK_SIZE for seg in col_segments):
            continue
        candidates.append((sep, row_segments, col_segments))
    assert len(candidates) == 1, "expected a unique separator partition"
    sep, row_segments, col_segments = candidates[0]
    blocks = [g[np.ix_(rows, cols)].copy() for rows in row_segments for cols in col_segments]
    return int(sep), row_segments, col_segments, blocks


def _background_color(blocks: list[np.ndarray]) -> int:
    counts = Counter(int(v) for block in blocks for v in block.flat)
    return int(counts.most_common(1)[0][0])


def _edge_signature(mask: np.ndarray, side: str) -> tuple[int, ...]:
    if side == "left":
        return tuple(int(r) for r in range(BLOCK_SIZE) if mask[r, 0])
    if side == "right":
        return tuple(int(r) for r in range(BLOCK_SIZE) if mask[r, BLOCK_SIZE - 1])
    if side == "top":
        return tuple(int(c) for c in range(BLOCK_SIZE) if mask[0, c])
    if side == "bottom":
        return tuple(int(c) for c in range(BLOCK_SIZE) if mask[BLOCK_SIZE - 1, c])
    raise ValueError(side)


def _solve_chain(blocks: list[np.ndarray], bg: int, orientation: str) -> list[int] | None:
    masks = [block != bg for block in blocks]
    if orientation == "h":
        incoming = [_edge_signature(mask, "left") for mask in masks]
        outgoing = [_edge_signature(mask, "right") for mask in masks]
    else:
        incoming = [_edge_signature(mask, "top") for mask in masks]
        outgoing = [_edge_signature(mask, "bottom") for mask in masks]

    inc_map: dict[tuple[int, ...], int] = {}
    for idx, sig in enumerate(incoming):
        if not sig:
            continue
        if sig in inc_map:
            return None
        inc_map[sig] = idx

    out_map: dict[tuple[int, ...], int] = {}
    for idx, sig in enumerate(outgoing):
        if not sig:
            continue
        if sig in out_map:
            return None
        out_map[sig] = idx

    successor = [None] * len(blocks)
    predecessor = [None] * len(blocks)
    for i, sig in enumerate(outgoing):
        if not sig:
            continue
        j = inc_map.get(sig)
        if j is None:
            continue
        if i == j:
            return None
        if predecessor[j] is not None:
            return None
        successor[i] = j
        predecessor[j] = i

    starts = [i for i, p in enumerate(predecessor) if p is None]
    ends = [i for i, s in enumerate(successor) if s is None]
    if len(starts) != 1 or len(ends) != 1:
        return None

    path = []
    seen = set()
    cur = starts[0]
    while cur is not None and cur not in seen:
        path.append(cur)
        seen.add(cur)
        cur = successor[cur]
    if len(path) != len(blocks) or cur is not None:
        return None
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange scattered 5x5 sprite blocks into one connected strip using shared edge matches.

    Input structure:
    - Uniform full separator rows/columns divide the grid into 5x5 panels.
    - Each panel contains a sprite on a common background.

    Rule:
    - Two blocks connect horizontally when the foreground offsets on one block's right edge
      exactly match the offsets on the other's left edge.
    - Two blocks connect vertically when the foreground offsets on one block's bottom edge
      exactly match the offsets on the other's top edge.
    - Exactly one of those directions yields a single chain through every block.
    - Output that chain as a 1xN strip, preserving each 5x5 block and using the separator color
      between consecutive blocks.
    """
    g = np.asarray(grid, dtype=int)
    sep, _row_segments, _col_segments, blocks = _parse_panels(g)
    bg = _background_color(blocks)

    horizontal = _solve_chain(blocks, bg, "h")
    vertical = _solve_chain(blocks, bg, "v")
    assert (horizontal is None) != (vertical is None), "expected exactly one valid orientation"

    path = horizontal if horizontal is not None else vertical
    assert path is not None
    horizontal_strip = horizontal is not None

    n = len(blocks)
    if horizontal_strip:
        out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1) * SEP_SIZE), sep, dtype=int)
        for i, idx in enumerate(path):
            c0 = i * (BLOCK_SIZE + SEP_SIZE)
            out[:, c0 : c0 + BLOCK_SIZE] = blocks[idx]
    else:
        out = np.full((n * BLOCK_SIZE + (n - 1) * SEP_SIZE, BLOCK_SIZE), sep, dtype=int)
        for i, idx in enumerate(path):
            r0 = i * (BLOCK_SIZE + SEP_SIZE)
            out[r0 : r0 + BLOCK_SIZE, :] = blocks[idx]
    return out


def _path_to_anchor(cell: tuple[int, int], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    ar, ac = anchor
    path = [(r, c)]
    if r == 0:
        r = 1
        path.append((r, c))
    elif r == BLOCK_SIZE - 1:
        r = BLOCK_SIZE - 2
        path.append((r, c))
    if c == 0:
        c = 1
        if path[-1] != (r, c):
            path.append((r, c))
    elif c == BLOCK_SIZE - 1:
        c = BLOCK_SIZE - 2
        if path[-1] != (r, c):
            path.append((r, c))
    while r != ar:
        r += 1 if ar > r else -1
        path.append((r, c))
    while c != ac:
        c += 1 if ac > c else -1
        path.append((r, c))
    return path


def _boundary_required(left: tuple[int, ...], right: tuple[int, ...], top: tuple[int, ...], bottom: tuple[int, ...]) -> set[tuple[int, int]]:
    cells = {(r, 0) for r in left}
    cells |= {(r, BLOCK_SIZE - 1) for r in right}
    cells |= {(0, c) for c in top}
    cells |= {(BLOCK_SIZE - 1, c) for c in bottom}
    return cells


def _make_block(
    left: tuple[int, ...],
    right: tuple[int, ...],
    top: tuple[int, ...],
    bottom: tuple[int, ...],
    bg: int,
    fg: int,
    rng: random.Random,
) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    required = _boundary_required(left, right, top, bottom)
    anchor = rng.choice(INTERIOR_CELLS)
    if not required:
        required = {anchor}
    for cell in required:
        for rr, cc in _path_to_anchor(cell, anchor):
            mask[rr, cc] = True

    frontier = [cell for cell in INTERIOR_CELLS if mask[cell]]
    rng.shuffle(frontier)
    target_extra = rng.randint(0, 4)
    added = 0
    while frontier and added < target_extra:
        r, c = frontier.pop()
        for dr, dc in rng.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 4):
            rr, cc = r + dr, c + dc
            if not (1 <= rr <= BLOCK_SIZE - 2 and 1 <= cc <= BLOCK_SIZE - 2):
                continue
            if mask[rr, cc]:
                continue
            mask[rr, cc] = True
            frontier.append((rr, cc))
            added += 1
            break

    assert _edge_signature(mask, "left") == left
    assert _edge_signature(mask, "right") == right
    assert _edge_signature(mask, "top") == top
    assert _edge_signature(mask, "bottom") == bottom

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), bg, dtype=int)
    block[mask] = fg
    return block


def _random_signature_pool(count: int, rng: random.Random) -> list[tuple[int, ...]]:
    sigs = rng.sample(CHAIN_SIGS, count)
    return [tuple(sig) for sig in sigs]


def _generate_chain_blocks(n: int, orientation: str, bg: int, fg: int, rng: random.Random) -> list[np.ndarray]:
    link_sigs = _random_signature_pool(n + 1, rng)
    blocks = []

    for i in range(n):
        if orientation == "h":
            left = link_sigs[i]
            right = link_sigs[i + 1]
            top = ()
            bottom = ()
        else:
            top = link_sigs[i]
            bottom = link_sigs[i + 1]
            left = ()
            right = ()
        block = _make_block(left, right, top, bottom, bg, fg, rng)
        blocks.append(block)
    return blocks


def _permute_into_grid(blocks: list[np.ndarray], layout: tuple[int, int], bg: int, sep: int, rng: random.Random) -> np.ndarray:
    rows, cols = layout
    order = list(range(len(blocks)))
    rng.shuffle(order)
    shuffled = [blocks[i] for i in order]

    h = rows * BLOCK_SIZE + (rows - 1) * SEP_SIZE
    w = cols * BLOCK_SIZE + (cols - 1) * SEP_SIZE
    grid = np.full((h, w), bg, dtype=int)
    for r in range(1, rows):
        grid[r * BLOCK_SIZE + (r - 1) * SEP_SIZE, :] = sep
    for c in range(1, cols):
        grid[:, c * BLOCK_SIZE + (c - 1) * SEP_SIZE] = sep

    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (BLOCK_SIZE + SEP_SIZE)
            c0 = cc * (BLOCK_SIZE + SEP_SIZE)
            grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = shuffled[idx]
            idx += 1
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(sorted(LAYOUTS))
    layout = rng.choice(LAYOUTS[n])
    orientation = rng.choice(["h", "v"])

    colors = rng.sample(list(range(10)), 3)
    bg, sep, fg = colors

    try:
        chain_blocks = _generate_chain_blocks(n, orientation, bg, fg, rng)
    except AssertionError:
        return None

    grid = _permute_into_grid(chain_blocks, layout, bg, sep, rng)

    try:
        parsed_sep, row_segments, col_segments, parsed_blocks = _parse_panels(grid)
        if parsed_sep != sep:
            return None
        if len(parsed_blocks) != n or len(row_segments) * len(col_segments) != n:
            return None
        solved_h = _solve_chain(parsed_blocks, _background_color(parsed_blocks), "h")
        solved_v = _solve_chain(parsed_blocks, _background_color(parsed_blocks), "v")
        if orientation == "h" and not (solved_h is not None and solved_v is None):
            return None
        if orientation == "v" and not (solved_v is not None and solved_h is None):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
