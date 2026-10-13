import math
import random

import numpy as np


def _infer_block_size(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    assert h == w, "expected a square grid"
    n = math.isqrt(h)
    assert n * n == h, "expected a square-of-square side length"
    assert n >= 2, "block size must be at least 2"
    return n


def _get_blocks(grid: np.ndarray, n: int) -> list[list[np.ndarray]]:
    return [
        [grid[br * n : (br + 1) * n, bc * n : (bc + 1) * n] for bc in range(n)]
        for br in range(n)
    ]


def _pick_empty_block(blocks: list[list[np.ndarray]]) -> np.ndarray:
    for row in blocks:
        for block in row:
            if np.all(block == block[0, 0]):
                return block
    raise AssertionError("expected at least one empty block")


def _pick_pattern_block(blocks: list[list[np.ndarray]], bg: int) -> np.ndarray:
    for row in blocks:
        for block in row:
            if np.any(block != bg):
                return block
    raise AssertionError("expected at least one occupied block")


def _paint_mask(mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    return np.where(mask, int(fg), int(bg)).astype(int)


def _sample_mask(n: int) -> np.ndarray:
    modes = ("noise", "rects", "bands", "path")
    for _ in range(200):
        mode = random.choice(modes)
        mask = np.zeros((n, n), dtype=bool)

        if mode == "noise":
            density = random.uniform(0.3, 0.7)
            mask = np.random.random((n, n)) < density
        elif mode == "rects":
            for _ in range(random.randint(1, 3)):
                r0 = random.randint(0, n - 1)
                c0 = random.randint(0, n - 1)
                r1 = random.randint(r0, n - 1)
                c1 = random.randint(c0, n - 1)
                mask[r0 : r1 + 1, c0 : c1 + 1] = True
        elif mode == "bands":
            if random.choice((False, True)):
                rows = random.sample(range(n), random.randint(1, n - 1))
                mask[rows, :] = True
            else:
                cols = random.sample(range(n), random.randint(1, n - 1))
                mask[:, cols] = True
            if random.choice((False, True)):
                rr = random.sample(range(n), random.randint(1, max(1, n // 2)))
                cc = random.sample(range(n), random.randint(1, max(1, n // 2)))
                mask[np.ix_(rr, cc)] = ~mask[np.ix_(rr, cc)]
        else:
            r = random.randrange(n)
            c = random.randrange(n)
            mask[r, c] = True
            for _ in range(random.randint(n, 2 * n + 2)):
                dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
                r = min(max(r + dr, 0), n - 1)
                c = min(max(c + dc, 0), n - 1)
                mask[r, c] = True
            if random.choice((False, True)):
                mask |= np.flipud(mask)
            if random.choice((False, True)):
                mask |= np.fliplr(mask)

        ones = int(mask.sum())
        if n <= ones <= n * n - n:
            return mask
    raise RuntimeError("failed to sample a nondegenerate self-similar mask")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the corrupted self-referential block array so the macro occupancy matches the repeated micro mask.

    Input structure:
    - The grid is a square of size N^2 by N^2, implicitly split into N by N blocks of size N by N.
    - One block type is fully background and represents an empty macro cell.
    - Every non-empty block contains the same two-color N by N mask in the same position.

    Rule:
    - Read any non-empty block as the micro mask M, using the empty block color as background.
    - Read the N by N macro occupancy O by marking which blocks are non-empty.
    - O differs from M at exactly two cells: one extra occupied macro cell and one missing occupied macro cell.
    - Erase the extra block to background and redraw the missing block with the same mask M.

    Key insight:
    - The block contents and the block layout are supposed to encode the same binary pattern at two scales, so
      comparing those two views exposes the swapped occupied/empty pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    n = _infer_block_size(g)
    blocks = _get_blocks(g, n)

    empty_block = _pick_empty_block(blocks)
    bg = int(empty_block[0, 0])
    pattern_block = _pick_pattern_block(blocks, bg)

    mask = pattern_block != bg
    assert mask.any() and (~mask).any(), "micro mask must contain both occupied and empty cells"

    fg_values = np.unique(pattern_block[mask])
    assert len(fg_values) == 1, "occupied block must use a single foreground color"
    fg = int(fg_values[0])
    template = _paint_mask(mask, bg, fg)

    occupancy = np.zeros((n, n), dtype=bool)
    for br in range(n):
        for bc in range(n):
            block = blocks[br][bc]
            if np.all(block == bg):
                continue
            assert np.array_equal(block, template), "all occupied blocks must share the same mask"
            occupancy[br, bc] = True

    diff = occupancy ^ mask
    assert int(diff.sum()) == 2, "expected exactly one extra block and one missing block"

    extra = np.argwhere(occupancy & ~mask)
    missing = np.argwhere(~occupancy & mask)
    assert len(extra) == 1 and len(missing) == 1, "expected one extra and one missing macro cell"

    out = g.copy()
    er, ec = map(int, extra[0])
    mr, mc = map(int, missing[0])
    out[er * n : (er + 1) * n, ec * n : (ec + 1) * n] = bg
    out[mr * n : (mr + 1) * n, mc * n : (mc + 1) * n] = template
    return out


def _build_input(mask: np.ndarray, bg: int, fg: int, missing: tuple[int, int], extra: tuple[int, int]) -> np.ndarray:
    n = int(mask.shape[0])
    block = _paint_mask(mask, bg, fg)
    occupancy = mask.copy()
    occupancy[missing] = False
    occupancy[extra] = True

    grid = np.full((n * n, n * n), int(bg), dtype=int)
    for br in range(n):
        for bc in range(n):
            if occupancy[br, bc]:
                grid[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = block
    return grid


def _build_output(mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    n = int(mask.shape[0])
    block = _paint_mask(mask, bg, fg)
    grid = np.full((n * n, n * n), int(bg), dtype=int)
    for br in range(n):
        for bc in range(n):
            if mask[br, bc]:
                grid[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = block
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.choice((4, 5))
    mask = _sample_mask(n)

    ones = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    zeros = [tuple(map(int, pos)) for pos in np.argwhere(~mask)]
    missing = random.choice(ones)
    extra = random.choice(zeros)

    bg, fg = random.sample(range(10), 2)
    inp = _build_input(mask, bg, fg, missing, extra)
    expected = _build_output(mask, bg, fg)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
