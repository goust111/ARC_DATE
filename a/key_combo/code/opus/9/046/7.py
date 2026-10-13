from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
SPRITE_SIZES = (3, 4)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _swap_colors(sprite: np.ndarray) -> np.ndarray:
    swapped = np.array(sprite, copy=True)
    swapped[sprite == BLUE] = RED
    swapped[sprite == RED] = BLUE
    return swapped


def _transform_sprite(sprite: np.ndarray) -> np.ndarray:
    n = int(sprite.shape[0])
    assert sprite.shape == (n, n)

    out = np.zeros((n * n, n * n), dtype=int)
    swapped = _swap_colors(sprite)
    for r in range(n):
        for c in range(n):
            if int(sprite[r, c]) == BG:
                continue
            block = sprite if (r + c) % 2 == 0 else swapped
            out[r * n : (r + 1) * n, c * n : (c + 1) * n] = block
    return out


def _alternative_no_swap(sprite: np.ndarray) -> np.ndarray:
    n = int(sprite.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for r in range(n):
        for c in range(n):
            if int(sprite[r, c]) != BG:
                out[r * n : (r + 1) * n, c * n : (c + 1) * n] = sprite
    return out


def _alternative_swap_all(sprite: np.ndarray) -> np.ndarray:
    n = int(sprite.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    swapped = _swap_colors(sprite)
    for r in range(n):
        for c in range(n):
            if int(sprite[r, c]) != BG:
                out[r * n : (r + 1) * n, c * n : (c + 1) * n] = swapped
    return out


def _alternative_ignore_mask(sprite: np.ndarray) -> np.ndarray:
    n = int(sprite.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    swapped = _swap_colors(sprite)
    for r in range(n):
        for c in range(n):
            block = sprite if (r + c) % 2 == 0 else swapped
            out[r * n : (r + 1) * n, c * n : (c + 1) * n] = block
    return out


def _alternative_inverse_parity(sprite: np.ndarray) -> np.ndarray:
    n = int(sprite.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    swapped = _swap_colors(sprite)
    for r in range(n):
        for c in range(n):
            if int(sprite[r, c]) == BG:
                continue
            block = swapped if (r + c) % 2 == 0 else sprite
            out[r * n : (r + 1) * n, c * n : (c + 1) * n] = block
    return out


def _alternative_color_driven(sprite: np.ndarray, original_for: int) -> np.ndarray:
    n = int(sprite.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    swapped = _swap_colors(sprite)
    for r in range(n):
        for c in range(n):
            cell = int(sprite[r, c])
            if cell == BG:
                continue
            block = sprite if cell == original_for else swapped
            out[r * n : (r + 1) * n, c * n : (c + 1) * n] = block
    return out


def _alternative_uniform_fill(sprite: np.ndarray) -> np.ndarray:
    n = int(sprite.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for r in range(n):
        for c in range(n):
            cell = int(sprite[r, c])
            if cell != BG:
                out[r * n : (r + 1) * n, c * n : (c + 1) * n] = cell
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the top-left two-color sprite by a self-Kronecker stamp, filtered by active cells and odd-even color flips.

    Input structure:
    - The grid is a small black-background canvas with height and width between 4 and 5.
    - Its only non-black content is one top-left-aligned sprite whose bounding box is exactly NxN with N in {3, 4}.
    - Inside that bbox, cells may be black, blue, or red, with both non-black colors present.

    Rule:
    - Extract the sprite bbox M.
    - Create an all-black output of shape (N*N, N*N), viewed as an NxN grid of NxN macro-blocks.
    - For each sprite cell M[i, j]: if it is black, leave macro-block (i, j) black; otherwise stamp the whole sprite there.
    - Use the original sprite when (i + j) is even, and use the red/blue-swapped sprite when (i + j) is odd.

    Key insight:
    - The active macro-block locations come from the sprite mask itself, while the macro-block contents are always the whole sprite, with parity deciding whether the two non-black colors stay or swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 4 <= h <= 5 and 4 <= w <= 5, "expected a 4x4 to 5x5 input grid"

    colors = set(map(int, np.unique(g)))
    assert colors <= {BG, BLUE, RED}, "unexpected colors in input"

    box = _bbox(g != BG)
    assert box is not None, "expected a non-black sprite"
    r0, c0, r1, c1 = box
    assert r0 == 0 and c0 == 0, "expected the sprite bbox to be top-left aligned"

    n = int(r1 - r0 + 1)
    assert n == int(c1 - c0 + 1), "expected a square sprite bbox"
    assert n in SPRITE_SIZES, "expected sprite size 3 or 4"

    sprite = g[:n, :n]
    assert np.all(g[n:, :] == BG) and np.all(g[:, n:] == BG), "expected only black padding outside the sprite bbox"
    assert (sprite == RED).sum() >= 3, "expected at least three red cells"
    assert (sprite == BLUE).sum() >= 2, "expected at least two blue cells"
    assert np.any(sprite == BG), "expected at least one black cell inside the sprite bbox"
    assert np.any(sprite != BG), "expected active sprite cells"

    return _transform_sprite(sprite)


def _sample_sprite(n: int) -> np.ndarray | None:
    probs = [0.22, 0.34, 0.44] if n == 3 else [0.28, 0.32, 0.40]
    sprite = np.random.choice([BG, BLUE, RED], size=(n, n), p=probs).astype(int)

    if int((sprite == RED).sum()) < 3 or int((sprite == BLUE).sum()) < 2:
        return None
    if not np.any(sprite == BG):
        return None

    if not np.any(sprite[0, :] != BG):
        return None
    if not np.any(sprite[-1, :] != BG):
        return None
    if not np.any(sprite[:, 0] != BG):
        return None
    if not np.any(sprite[:, -1] != BG):
        return None

    active_even = int(np.sum((sprite != BG) & ((np.indices((n, n)).sum(axis=0) % 2) == 0)))
    active_odd = int(np.sum((sprite != BG) & ((np.indices((n, n)).sum(axis=0) % 2) == 1)))
    if active_even == 0 or active_odd == 0:
        return None

    out = _transform_sprite(sprite)
    shortcuts = (
        _alternative_no_swap(sprite),
        _alternative_swap_all(sprite),
        _alternative_ignore_mask(sprite),
        _alternative_inverse_parity(sprite),
        _alternative_color_driven(sprite, RED),
        _alternative_color_driven(sprite, BLUE),
        _alternative_uniform_fill(sprite),
    )
    if any(np.array_equal(out, alt) for alt in shortcuts):
        return None

    active_positions = np.argwhere(sprite != BG)
    for r, c in active_positions:
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        expected = sprite if (int(r) + int(c)) % 2 == 0 else _swap_colors(sprite)
        if not np.array_equal(block, expected):
            return None

    inactive_positions = np.argwhere(sprite == BG)
    if inactive_positions.size == 0:
        return None
    for r, c in inactive_positions:
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        if np.any(block != BG):
            return None

    return sprite


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = random.choice(SPRITE_SIZES)
        sprite = _sample_sprite(n)
        if sprite is None:
            continue

        grid = np.zeros((n + 1, n + 1), dtype=int)
        grid[:n, :n] = sprite

        try:
            out = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(out, _transform_sprite(sprite)):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
