from __future__ import annotations

import random

import numpy as np


BLOCK = 3
BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
LOW_H_RANGE = (3, 5)
LOW_W_RANGE = (3, 5)
MAX_ATTEMPTS = 400


def _compress_blocks(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % BLOCK == 0 and w % BLOCK == 0, "grid must decompose into 3x3 blocks"

    out = np.zeros((h // BLOCK, w // BLOCK), dtype=int)
    seen_colors: set[int] = set()

    for br in range(h // BLOCK):
        for bc in range(w // BLOCK):
            block = g[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            real = sorted(set(int(v) for v in block.flat) - {BG, GREY})
            assert len(real) <= 1, "each block must contain at most one real color"
            if not real:
                assert np.all((block == BG) | (block == GREY)), "background blocks may only contain black/grey"
                continue

            color = real[0]
            assert np.count_nonzero(block == color) >= 8, "foreground blocks must be almost full"
            assert np.all((block == color) | (block == GREY)), "foreground damage must be grey"
            out[br, bc] = color
            seen_colors.add(color)

    assert len(seen_colors) <= 1, "compressed picture should use one foreground color"
    return out


def _stamp_complement(pattern: np.ndarray) -> np.ndarray:
    p = np.asarray(pattern, dtype=int)
    fg_colors = sorted(set(int(v) for v in np.unique(p)) - {BG})
    assert len(fg_colors) == 1, "compressed picture must contain exactly one foreground color"

    color = fg_colors[0]
    src_mask = p == color
    tile_mask = p == BG
    assert np.any(src_mask) and np.any(tile_mask), "compressed picture must mix foreground and background"
    return (np.kron(src_mask.astype(int), tile_mask.astype(int)) * color).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a noisy block picture, then stamp its complementary mask at every foreground location.

    Input structure:
    - The input grid is partitioned into 3x3 blocks on a black background.
    - Real foreground blocks are almost solid in one non-black, non-grey color; grey cells are noise.
    - Empty blocks contain only black and optional grey noise.

    Rule:
    - Collapse each 3x3 block to one pixel: black if the block has only black/grey, otherwise the block's unique real color.
    - Treat that compressed picture as both a placement mask and a stamp template.
    - For every foreground pixel in the compressed picture, place one tile shaped like the compressed picture's black cells; background pixels place nothing.

    Key insight:
    - The denoised compressed picture is self-referential: its foreground pixels say where to stamp, while its background pixels define the stamp's shape.
    """
    pattern = _compress_blocks(grid)
    return _stamp_complement(pattern)


def _mask_is_nondegenerate(mask: np.ndarray) -> bool:
    h, w = mask.shape
    fg = int(mask.sum())
    bg = h * w - fg
    if fg < 3 or bg < 3:
        return False
    if np.all(mask == mask[0:1, :]):
        return False
    if np.all(mask == mask[:, 0:1]):
        return False
    if np.count_nonzero(mask.any(axis=1)) < 2 or np.count_nonzero(mask.any(axis=0)) < 2:
        return False
    bg_mask = ~mask
    if np.count_nonzero(bg_mask.any(axis=1)) < 2 or np.count_nonzero(bg_mask.any(axis=0)) < 2:
        return False
    return True


def _sample_walk_mask(h: int, w: int, fg_count: int) -> np.ndarray:
    cells = [(r, c) for r in range(h) for c in range(w)]
    cur = random.choice(cells)
    chosen = {cur}
    while len(chosen) < fg_count:
        r, c = cur
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                neighbors.append((rr, cc))
        cur = random.choice(neighbors)
        chosen.add(cur)
        if random.random() < 0.25:
            cur = random.choice(tuple(chosen))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _sample_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*LOW_H_RANGE)
        w = random.randint(*LOW_W_RANGE)
        total = h * w
        fg_count = random.randint(3, total - 3)
        mode = random.choice(["sample", "walk", "carve"])

        if mode == "sample":
            chosen = random.sample(range(total), fg_count)
            mask = np.zeros((h, w), dtype=bool)
            for idx in chosen:
                mask[idx // w, idx % w] = True
        elif mode == "walk":
            mask = _sample_walk_mask(h, w, fg_count)
        else:
            mask = np.ones((h, w), dtype=bool)
            holes = _sample_walk_mask(h, w, total - fg_count)
            mask[holes] = False

        if _mask_is_nondegenerate(mask):
            return mask

    raise RuntimeError("failed to sample a nondegenerate compressed mask")


def _paint_block(grid: np.ndarray, br: int, bc: int, color: int, add_noise: bool) -> None:
    r0 = br * BLOCK
    c0 = bc * BLOCK
    if color != BG:
        grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = color
    if add_noise:
        rr = r0 + random.randrange(BLOCK)
        cc = c0 + random.randrange(BLOCK)
        grid[rr, cc] = GREY


def _render_input(mask: np.ndarray, color: int) -> np.ndarray:
    h, w = mask.shape
    grid = np.zeros((h * BLOCK, w * BLOCK), dtype=int)
    fg_positions = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
    bg_positions = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]

    noise_blocks: set[tuple[int, int]] = set()
    if random.random() < 0.85:
        noise_blocks.add(random.choice(fg_positions))
        noise_blocks.add(random.choice(bg_positions))
        extra_pool = [(r, c) for r in range(h) for c in range(w) if (r, c) not in noise_blocks]
        extra_max = min(len(extra_pool), max(0, (h * w) // 2))
        extra_count = random.randint(0, extra_max)
        if extra_count:
            noise_blocks.update(random.sample(extra_pool, extra_count))

    for r in range(h):
        for c in range(w):
            _paint_block(grid, r, c, color if mask[r, c] else BG, (r, c) in noise_blocks)
    return grid


def _oracle_compress(grid: np.ndarray) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] % BLOCK or g.shape[1] % BLOCK:
        return None

    out = np.zeros((g.shape[0] // BLOCK, g.shape[1] // BLOCK), dtype=int)
    seen: set[int] = set()

    for br in range(out.shape[0]):
        for bc in range(out.shape[1]):
            block = g[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            values = set(int(v) for v in block.flat)
            real = values - {BG, GREY}
            if len(real) > 1:
                return None
            if not real:
                if not values <= {BG, GREY}:
                    return None
                continue

            color = next(iter(real))
            if np.count_nonzero(block == color) < 8:
                return None
            if not np.all((block == color) | (block == GREY)):
                return None
            out[br, bc] = color
            seen.add(color)

    if len(seen) != 1:
        return None
    if not _mask_is_nondegenerate(out != BG):
        return None
    return out


def _oracle_stamp(pattern: np.ndarray) -> np.ndarray | None:
    p = np.asarray(pattern, dtype=int)
    colors = sorted(set(int(v) for v in np.unique(p)) - {BG})
    if len(colors) != 1:
        return None

    color = colors[0]
    h, w = p.shape
    out = np.zeros((h * h, w * w), dtype=int)
    bg_positions = [tuple(map(int, rc)) for rc in np.argwhere(p == BG)]
    if not bg_positions or not np.any(p == color):
        return None

    for r in range(h):
        for c in range(w):
            if p[r, c] != color:
                continue
            r0 = r * h
            c0 = c * w
            for rr, cc in bg_positions:
                out[r0 + rr, c0 + cc] = color
    return out


def _validate_generated_case(grid: np.ndarray) -> bool:
    pattern = _oracle_compress(grid)
    if pattern is None:
        return False

    expected = _oracle_stamp(pattern)
    if expected is None:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if GREY in set(int(v) for v in np.unique(actual)):
        return False
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mask = _sample_mask()
    color = random.choice(PALETTE)
    grid = _render_input(mask, color)
    return grid if _validate_generated_case(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for new puzzle 2")
