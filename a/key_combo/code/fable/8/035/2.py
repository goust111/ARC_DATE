from __future__ import annotations

import math
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()
USED_OUTPUT_SIGNATURES: set[tuple[int, bytes, int]] = set()


def _infer_n(side: int) -> int:
    n = math.isqrt(int(side))
    if n * n != int(side):
        raise AssertionError(f"grid side must be a perfect square, got {side}")
    return n


def _infer_fg(grid: np.ndarray) -> int:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != BG)
    if len(colors) != 1:
        raise AssertionError(f"expected exactly one foreground color, got {colors}")
    return colors[0]


def _extract_scales(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise AssertionError("expected a square 2D grid")

    side = int(g.shape[0])
    n = _infer_n(side)
    fg = _infer_fg(g)
    if not np.all((g == BG) | (g == fg)):
        raise AssertionError("input may only contain background and one foreground color")

    macro = np.zeros((n, n), dtype=bool)
    micro = np.zeros((n, n), dtype=bool)
    for br in range(n):
        for bc in range(n):
            block = g[br * n : (br + 1) * n, bc * n : (bc + 1) * n]
            nonempty = bool(np.any(block == fg))
            if nonempty:
                macro[br, bc] = True
                micro |= block == fg

    sprite = macro | micro
    if not np.any(sprite):
        raise AssertionError("reconstructed sprite must be nonempty")
    return n, fg, macro, micro, sprite


def _expand_sprite(sprite: np.ndarray, fg: int) -> np.ndarray:
    mask = np.asarray(sprite, dtype=bool)
    patch = np.where(mask, int(fg), BG).astype(int)
    return np.kron(mask.astype(int), patch).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct the hidden generator sprite by merging macro block occupancy with micro block contents, then re-stamp it without loss.

    Input structure:
    - The grid is an N^2 by N^2 black-background canvas split into N by N equal blocks.
    - Originally, some blocks were empty and every non-empty block contained the same monochrome N by N sprite.
    - Damage only erased foreground pixels to black, possibly clearing whole non-empty blocks.

    Rule:
    - Infer N from the grid side length and detect the single foreground color.
    - Read one N by N macro mask from which blocks are still non-empty, and one N by N micro mask from the cellwise union of all visible non-empty block contents.
    - Take the union of those two masks to recover the sprite, then stamp that recovered sprite into every macro-on block position of the recovered sprite.

    Key insight:
    - The same sprite is encoded twice at different scales: missing whole blocks are recoverable from surviving block interiors, while cell positions erased from every interior copy are recoverable from the surviving block-occupancy layout.
    """
    _n, fg, _macro, _micro, sprite = _extract_scales(grid)
    return _expand_sprite(sprite, fg)


def _mask_to_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(mask)]


def _random_sprite(n: int) -> np.ndarray | None:
    area_min = 2
    area_max = n * n - 1
    if area_min > area_max:
        return None

    for _ in range(80):
        density = float(RNG.uniform(0.3, 0.72))
        sprite = RNG.random((n, n)) < density
        area = int(np.count_nonzero(sprite))
        if not (area_min <= area <= area_max):
            continue
        return sprite
    return None


def _random_keep_subset(coords: list[tuple[int, int]], minimum_keep: int) -> set[tuple[int, int]]:
    keep = {coord for coord in coords if RNG.random() < 0.55}
    while len(keep) < minimum_keep:
        keep.add(coords[int(RNG.integers(len(coords)))])
    return keep


def _damage_expansion(sprite: np.ndarray, fg: int) -> np.ndarray | None:
    coords = _mask_to_coords(sprite)
    area = len(coords)
    if area < 2:
        return None

    max_full = min(area - 1, max(1, area // 2))
    full_count = int(RNG.integers(1, max_full + 1))
    erased_blocks = {
        coords[int(i)]
        for i in RNG.choice(len(coords), size=full_count, replace=False)
    }
    remaining_blocks = [coord for coord in coords if coord not in erased_blocks]
    if not remaining_blocks:
        return None

    macro_only_cell = remaining_blocks[int(RNG.integers(len(remaining_blocks)))]
    witness_block = remaining_blocks[int(RNG.integers(len(remaining_blocks)))]

    blocks: dict[tuple[int, int], np.ndarray] = {}
    baseline = sprite.copy()
    baseline[macro_only_cell] = False
    if not np.any(baseline):
        return None

    for block_coord in coords:
        if block_coord in erased_blocks:
            blocks[block_coord] = np.zeros_like(sprite, dtype=bool)
            continue

        patch = baseline.copy()
        if block_coord != witness_block:
            candidate_coords = _mask_to_coords(patch)
            keep = _random_keep_subset(candidate_coords, minimum_keep=1)
            new_patch = np.zeros_like(patch, dtype=bool)
            for rr, cc in keep:
                new_patch[rr, cc] = True
            patch = new_patch
        blocks[block_coord] = patch

    n = int(sprite.shape[0])
    side = n * n
    grid = np.zeros((side, side), dtype=int)
    for br, bc in coords:
        patch = blocks[(br, bc)]
        if not np.any(patch):
            continue
        grid[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = np.where(patch, fg, BG)
    return grid


def _semantic_check(inp: np.ndarray, sprite: np.ndarray, fg: int) -> bool:
    expected = _expand_sprite(sprite, fg)
    try:
        n, seen_fg, macro, micro, rebuilt = _extract_scales(inp)
        actual = main(inp)
    except AssertionError:
        return False

    if seen_fg != fg or n != int(sprite.shape[0]):
        return False
    if not np.array_equal(rebuilt, sprite):
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(macro, sprite):
        return False
    if np.array_equal(micro, sprite):
        return False
    if not np.any(sprite & ~macro):
        return False
    if not np.any(sprite & ~micro):
        return False
    if not np.array_equal((macro | micro), sprite):
        return False
    if int(np.max(inp.shape)) > 30 or int(np.max(actual.shape)) > 30:
        return False
    return True


def _output_signature(sprite: np.ndarray, fg: int) -> tuple[int, bytes, int]:
    return int(sprite.shape[0]), np.packbits(sprite.astype(np.uint8)).tobytes(), int(fg)


def generate_input() -> np.ndarray:
    for _ in range(600):
        n = int(RNG.integers(2, 6))
        sprite = _random_sprite(n)
        if sprite is None:
            continue

        fg = int(PALETTE[int(RNG.integers(len(PALETTE)))])
        signature = _output_signature(sprite, fg)
        if signature in USED_OUTPUT_SIGNATURES:
            continue

        inp = _damage_expansion(sprite, fg)
        if inp is None:
            continue
        if not _semantic_check(inp, sprite, fg):
            continue

        USED_OUTPUT_SIGNATURES.add(signature)
        return inp
    raise RuntimeError("failed to generate a valid cross-scale self-stamp sample")
