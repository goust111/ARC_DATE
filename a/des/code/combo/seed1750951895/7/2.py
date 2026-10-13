import random
from itertools import permutations

import numpy as np


BLACK = 0
GREY = 5
BLOCK = 5
SEP = 1
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    prev = -1
    for idx in sorted(separators) + [size]:
        if idx - prev > 1:
            bands.append((prev + 1, idx - 1))
        prev = idx
    return bands


def _find_separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    sep_colors = (row_colors & col_colors) - {BLACK}
    assert len(sep_colors) == 1, "expected one non-black separator color"
    return next(iter(sep_colors))


def _extract_blocks(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    sep_color = _find_separator_color(g)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == sep_color)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == sep_color)]
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    assert row_bands and col_bands, "expected at least one block row and column"
    assert all(r1 - r0 + 1 == BLOCK for r0, r1 in row_bands)
    assert all(c1 - c0 + 1 == BLOCK for c0, c1 in col_bands)

    blocks: list[np.ndarray] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            fg = set(np.unique(block).tolist()) - {BLACK, sep_color}
            assert len(fg) == 1, "each block must use exactly one foreground color"
            blocks.append(block.copy())
    assert len(blocks) >= 2
    return blocks, sep_color


def _edge_signature(block: np.ndarray, side: str, sep_color: int) -> tuple[bool, ...]:
    if side == "top":
        edge = block[0, :]
    elif side == "bottom":
        edge = block[-1, :]
    elif side == "left":
        edge = block[:, 0]
    elif side == "right":
        edge = block[:, -1]
    else:
        raise ValueError(side)
    return tuple(bool(v != BLACK and v != sep_color) for v in edge.tolist())


def _chain_orders(blocks: list[np.ndarray], sep_color: int, orientation: str) -> list[tuple[int, ...]]:
    if orientation == "horizontal":
        src_side, dst_side = "right", "left"
    elif orientation == "vertical":
        src_side, dst_side = "bottom", "top"
    else:
        raise ValueError(orientation)

    src = [_edge_signature(block, src_side, sep_color) for block in blocks]
    dst = [_edge_signature(block, dst_side, sep_color) for block in blocks]

    valid = []
    for perm in permutations(range(len(blocks))):
        ok = True
        for a, b in zip(perm, perm[1:]):
            if src[a] != dst[b] or not any(src[a]):
                ok = False
                break
        if ok:
            valid.append(perm)
            if len(valid) > 1:
                break
    return valid


def _assemble(blocks: list[np.ndarray], sep_color: int, orientation: str) -> np.ndarray:
    n = len(blocks)
    if orientation == "horizontal":
        out = np.full((BLOCK, n * BLOCK + (n - 1) * SEP), BLACK, dtype=int)
        c = 0
        for i, block in enumerate(blocks):
            out[:, c : c + BLOCK] = block
            c += BLOCK
            if i + 1 < n:
                out[:, c : c + SEP] = sep_color
                c += SEP
        return out

    out = np.full((n * BLOCK + (n - 1) * SEP, BLOCK), BLACK, dtype=int)
    r = 0
    for i, block in enumerate(blocks):
        out[r : r + BLOCK, :] = block
        r += BLOCK
        if i + 1 < n:
            out[r : r + SEP, :] = sep_color
            r += SEP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 sprite blocks into a single chain by matching occupied edge positions.

    Input structure:
    - Black background with one separator color forming full rows and columns.
    - Non-separator regions are 5x5 blocks, each using exactly one foreground color.

    Rule:
    - Two blocks connect horizontally when the right-edge occupancy pattern of the first
      equals the left-edge occupancy pattern of the second, with at least one occupied edge cell.
    - Two blocks connect vertically when the bottom-edge occupancy pattern of the first
      equals the top-edge occupancy pattern of the second, with at least one occupied edge cell.
    - Exactly one orientation yields a complete chain through all blocks.
    - Output that chain in the matching orientation, separating adjacent blocks by the
      same separator color.
    """
    blocks, sep_color = _extract_blocks(grid)

    horizontal = _chain_orders(blocks, sep_color, "horizontal")
    vertical = _chain_orders(blocks, sep_color, "vertical")

    if len(horizontal) == 1 and not vertical:
        order = list(horizontal[0])
        orientation = "horizontal"
    elif len(vertical) == 1 and not horizontal:
        order = list(vertical[0])
        orientation = "vertical"
    else:
        raise AssertionError("expected exactly one valid chain orientation")

    ordered_blocks = [blocks[i] for i in order]
    return _assemble(ordered_blocks, sep_color, orientation)


def _sig_pool() -> list[tuple[int, ...]]:
    pool = []
    for mask in range(1, 1 << BLOCK):
        sig = tuple(1 if (mask >> bit) & 1 else 0 for bit in range(BLOCK))
        pool.append(sig)
    random.shuffle(pool)
    return pool


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    step = 1 if tr >= r else -1
    for rr in range(r, tr + step, step):
        mask[rr, c] = True
    step = 1 if tc >= c else -1
    for cc in range(c, tc + step, step):
        mask[tr, cc] = True


def _make_block(
    top: tuple[int, ...],
    right: tuple[int, ...],
    bottom: tuple[int, ...],
    left: tuple[int, ...],
    color: int,
) -> np.ndarray:
    mask = np.zeros((BLOCK, BLOCK), dtype=bool)
    center = (2, 2)

    for c, bit in enumerate(top):
        if bit:
            _draw_path(mask, (0, c), center)
    for r, bit in enumerate(right):
        if bit:
            _draw_path(mask, (r, BLOCK - 1), center)
    for c, bit in enumerate(bottom):
        if bit:
            _draw_path(mask, (BLOCK - 1, c), center)
    for r, bit in enumerate(left):
        if bit:
            _draw_path(mask, (r, 0), center)

    if mask.sum() < 7:
        extras = [(1, 1), (1, 3), (3, 1), (3, 3), (2, 1), (2, 3)]
        random.shuffle(extras)
        for r, c in extras[: random.randint(1, 3)]:
            mask[r, c] = True

    block = np.zeros((BLOCK, BLOCK), dtype=int)
    block[mask] = color
    return block


def _choose_layout(n: int) -> tuple[int, int]:
    options = [(r, c) for r in range(2, n + 1) for c in range(2, n + 1) if r * c == n]
    assert options
    return random.choice(options)


def _make_chain_blocks(n: int, orientation: str) -> list[np.ndarray]:
    pool = _sig_pool()
    colors = random.sample(PALETTE, n)

    if orientation == "horizontal":
        internal = [pool.pop() for _ in range(n - 1)]
        left_exterior = [pool.pop() for _ in range(n)]
        right_exterior = [pool.pop() for _ in range(n)]
        top_sigs = [pool.pop() for _ in range(n)]
        bottom_sigs = [pool.pop() for _ in range(n)]
        blocks = []
        for i in range(n):
            left = left_exterior[i] if i == 0 else internal[i - 1]
            right = right_exterior[i] if i == n - 1 else internal[i]
            blocks.append(_make_block(top_sigs[i], right, bottom_sigs[i], left, colors[i]))
        return blocks

    internal = [pool.pop() for _ in range(n - 1)]
    top_exterior = [pool.pop() for _ in range(n)]
    bottom_exterior = [pool.pop() for _ in range(n)]
    left_sigs = [pool.pop() for _ in range(n)]
    right_sigs = [pool.pop() for _ in range(n)]
    blocks = []
    for i in range(n):
        top = top_exterior[i] if i == 0 else internal[i - 1]
        bottom = bottom_exterior[i] if i == n - 1 else internal[i]
        blocks.append(_make_block(top, right_sigs[i], bottom, left_sigs[i], colors[i]))
    return blocks


def _grid_from_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    n = len(blocks)
    rows, cols = _choose_layout(n)
    perm = list(range(n))
    while True:
        random.shuffle(perm)
        if perm != list(range(n)):
            break

    h = rows * BLOCK + (rows + 1) * SEP
    w = cols * BLOCK + (cols + 1) * SEP
    grid = np.full((h, w), GREY, dtype=int)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            block = blocks[perm[idx]]
            r0 = SEP + r * (BLOCK + SEP)
            c0 = SEP + c * (BLOCK + SEP)
            grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = block
            idx += 1
    return grid


def _expected_output(blocks: list[np.ndarray], orientation: str) -> np.ndarray:
    return _assemble(blocks, GREY, orientation)


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["horizontal", "vertical"])
    n = random.choice([4, 6])
    blocks = _make_chain_blocks(n, orientation)
    grid = _grid_from_blocks(blocks)

    blocks2, sep_color = _extract_blocks(grid)
    horizontal = _chain_orders(blocks2, sep_color, "horizontal")
    vertical = _chain_orders(blocks2, sep_color, "vertical")
    if orientation == "horizontal":
        if not (len(horizontal) == 1 and not vertical):
            return None
        ordered = [blocks2[i] for i in horizontal[0]]
    else:
        if not (len(vertical) == 1 and not horizontal):
            return None
        ordered = [blocks2[i] for i in vertical[0]]

    expected = _expected_output(blocks, orientation)
    actual = _assemble(ordered, sep_color, orientation)
    if not np.array_equal(expected, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
